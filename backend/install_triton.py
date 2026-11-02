#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交互式安装 / 卸载 triton-windows。

为什么单独一个脚本：triton 的版本必须和 torch 精确配对，装错会让
torch.compile 在编译期报 undefined symbol；而 torch.compile 的 CUDA 后端
在 Windows 上又依赖它。这个脚本负责：检测 → 版本推导 → 选源 → 安装 →
冒烟测试 → 同步 config.json。

用法（在项目根目录）：
    4-install_triton.bat                 交互式
    4-install_triton.bat --yes           全部用默认值（自动选源）
    4-install_triton.bat --uninstall     直接卸载
    4-install_triton.bat --channel 2     指定源：1 清华 / 2 阿里 / 3 PyPI

仅影响项目自己的 .venv，不碰系统 Python。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import components_paths  # noqa: E402,F401  首次运行时生成 config.json

CONFIG_PATH = ROOT / "config.json"

# torch 次版本 → 需要的 triton-windows 版本前缀（上游 README 的对应表）
TORCH_TRITON_MAP = {
    "2.4": "3.1.*",
    "2.5": "3.1.*",
    "2.6": "3.2.*",
    "2.7": "3.3.*",
    "2.8": "3.4.*",
    "2.9": "3.5.*",
    "2.10": "3.6.*",
}

CHANNELS = {
    1: ("清华镜像", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    2: ("阿里镜像", "https://mirrors.aliyun.com/pypi/simple"),
    3: ("PyPI 官方", "https://pypi.org/simple"),
}


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def venv_python() -> Path | None:
    for rel in ("Scripts/python.exe", "bin/python"):
        candidate = ROOT / ".venv" / rel
        if candidate.exists():
            return candidate
    return None


def run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return value or default


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    ans = ask(f"{prompt} ({hint})").lower()
    if not ans:
        return default
    return ans in ("y", "yes", "是")


def pip_cmd(command: str) -> list[str]:
    """优先用 uv（venv 里通常没 pip），否则退回 python -m pip。

    uv 的写法是 `uv pip install --python <exe> ...`，--python 必须在子命令之后。
    """
    if shutil.which("uv"):
        return ["uv", "pip", command, "--python", str(venv_python())]
    return [str(venv_python()), "-m", "pip", command]


# --------------------------------------------------------------------------- #
# 环境探测
# --------------------------------------------------------------------------- #
def probe_env() -> dict:
    """一次性拿到 torch 版本、CUDA 能力、已装 triton 版本。"""
    py = venv_python()
    script = r"""
import json
info = {"torch": None, "cc": None, "cuda": False, "triton": None}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda"] = bool(torch.cuda.is_available())
    if info["cuda"]:
        info["cc"] = list(torch.cuda.get_device_capability(0))
except Exception as exc:
    info["torch_error"] = f"{type(exc).__name__}: {exc}"
try:
    import importlib.metadata as md
    info["triton"] = md.version("triton-windows")
except Exception:
    info["triton"] = None
print(json.dumps(info))
"""
    result = run([str(py), "-c", script], timeout=180)
    if result.returncode != 0:
        return {"error": (result.stderr or result.stdout or "").strip()[:300]}
    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": result.stdout.strip()[:300]}

    # 仅测试用：允许伪造探测结果，方便在没有对应硬件/版本时走通分支
    fake_cc = os.environ.get("STG_TRITON_TEST_CC")
    if fake_cc:
        data["cc"] = [int(x) for x in fake_cc.replace(".", ",").split(",") if x.strip()]
        data["cuda"] = True
    fake_torch = os.environ.get("STG_TRITON_TEST_TORCH")
    if fake_torch:
        data["torch"] = fake_torch
    return data


def torch_minor(version: str | None) -> str | None:
    if not version:
        return None
    match = re.match(r"(\d+)\.(\d+)", version)
    return f"{match.group(1)}.{match.group(2)}" if match else None


def installed_version() -> str | None:
    return probe_env().get("triton")


def gpu_ok(info: dict) -> bool:
    cc = info.get("cc")
    return bool(cc) and int(cc[0]) >= 7


# --------------------------------------------------------------------------- #
# 源测速
# --------------------------------------------------------------------------- #
def simple_url(index: str) -> str:
    return index.rstrip("/") + "/triton-windows/"


def wheel_url(index: str, spec_prefix: str) -> str | None:
    """从 simple 索引里找第一个匹配版本前缀的 Windows 轮子。"""
    page = urllib.request.urlopen(simple_url(index), timeout=15).read().decode("utf-8", "replace")
    hrefs = re.findall(r'href="([^"]+)"', page)
    for href in hrefs:
        name = href.rsplit("/", 1)[-1]
        if not name.startswith("triton_windows-"):
            continue
        if not spec_prefix or not name.startswith(f"triton_windows-{spec_prefix}"):
            continue
        if "win_amd64" not in name:
            continue
        if href.startswith("../"):
            return index.rstrip("/").split("/simple")[0] + "/" + href.lstrip("./")
        if href.startswith("http"):
            return href
        base = index.rstrip("/")
        return base.rsplit("/", 1)[0] + "/" + href.lstrip("./")
    return None


def measure(index: str, spec_prefix: str) -> dict:
    """测首页延迟 + 1 MiB 实际下载速度。"""
    result = {"index": index, "latency_ms": None, "speed_mbps": 0.0, "ok": False, "error": ""}
    try:
        t0 = time.time()
        urllib.request.urlopen(simple_url(index), timeout=15).read(2048)
        result["latency_ms"] = round((time.time() - t0) * 1000)
        url = wheel_url(index, spec_prefix)
        if not url:
            result["error"] = "索引里没有匹配的轮子"
            return result
        req = urllib.request.Request(url, headers={"Range": "bytes=0-1048575"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=20) as resp:
            chunk = resp.read(1024 * 1024)
        dt = max(time.time() - t0, 1e-6)
        result["speed_mbps"] = round(len(chunk) / dt / 1024 / 1024, 2)
        result["ok"] = len(chunk) > 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def probe_channels(spec_prefix: str) -> list[dict]:
    with ThreadPoolExecutor(max_workers=3) as pool:
        return list(pool.map(lambda idx: measure(idx, spec_prefix), [c[1] for c in CHANNELS.values()]))


def choose_channel(spec_prefix: str, auto_only: bool = False) -> tuple[str, str]:
    print("\n== 选择下载源 ==")
    results = probe_channels(spec_prefix)
    for num, (label, index) in CHANNELS.items():
        item = next((r for r in results if r["index"] == index), {})
        if item.get("ok"):
            print(f"  {num}) {label:<8} {item['latency_ms']:>5} ms · {item['speed_mbps']:>5.1f} MiB/s")
        else:
            print(f"  {num}) {label:<8} 不可用（{item.get('error', '未知')[:40]}）")

    best = max((r for r in results if r["ok"]), key=lambda r: r["speed_mbps"], default=None)
    if auto_only:
        if not best:
            raise SystemExit("[错误] 三个源都不可用，请检查网络或代理后重试。")
        label = next(k for k, v in CHANNELS.items() if v[1] == best["index"])
        print(f"  自动选择：{CHANNELS[label][0]} （{best['speed_mbps']} MiB/s）")
        return CHANNELS[label]

    default = "1" if best is None else str(next(k for k, v in CHANNELS.items() if v[1] == best["index"]))
    choice = ask(f"  输入 1/2/3，直接回车 = 自动（推荐 {default}）", "")
    if not choice:
        if not best:
            raise SystemExit("[错误] 三个源都不可用。")
        label = next(k for k, v in CHANNELS.items() if v[1] == best["index"])
        print(f"  自动选择：{CHANNELS[label][0]}")
        return CHANNELS[label]
    if choice not in ("1", "2", "3"):
        raise SystemExit("[错误] 无效的选择，已退出。")
    return CHANNELS[int(choice)]


# --------------------------------------------------------------------------- #
# 安装 / 卸载
# --------------------------------------------------------------------------- #
def do_install(spec: str, index: str) -> bool:
    print(f"\n== 安装 {spec} ← {index} ==")
    cmd = pip_cmd("install") + [spec, "--index-url", index]
    result = run(cmd)
    if result.returncode != 0:
        print("[失败] 安装命令返回非零：")
        print((result.stderr or result.stdout or "").strip()[-1500:])
        return False
    text = (result.stdout or "").strip() or (result.stderr or "").strip()
    if text:
        print("  " + text.splitlines()[-1])
    return True


def do_uninstall() -> bool:
    print("\n== 卸载 triton-windows ==")
    cmd = pip_cmd("uninstall") + ["triton-windows"]
    if shutil.which("uv") is None:
        cmd.append("-y")
    result = run(cmd)
    if result.returncode != 0:
        print("[失败] " + (result.stderr or result.stdout or "").strip()[-800:])
        return False
    print("  已卸载")
    write_config(False)
    print("  已把 config.json 的 compile_dequant 改回 false")
    return True


# --------------------------------------------------------------------------- #
# 冒烟测试与配置
# --------------------------------------------------------------------------- #
SMOKE = r"""
import json, os, sys
root = os.getcwd()          # run() 固定以项目根为 cwd（python -c 没有 __file__）
stm = os.path.join(root, "stm")
for entry in (os.path.join(stm, "common"), stm):
    if entry not in sys.path:
        sys.path.insert(0, entry)

def emit(ok, reason=""):
    print(json.dumps({"ok": ok, "reason": reason}))
    sys.exit(0)

try:
    import torch
except Exception as exc:
    emit(False, f"import torch 失败: {exc}")
if not torch.cuda.is_available():
    emit(False, "CUDA 不可用")

# 直接编译 STM 真正会用到的那两个反量化 kernel —— 它含非连续访存，
# 是 sm<70 上最先失败的地方；用 sin/cos 之类会漏判。
try:
    import gguf
    from utils import ggml_runtime as G
except Exception as exc:
    emit(False, f"无法加载 STM 反量化模块: {type(exc).__name__}: {exc}")

qtype = gguf.GGMLQuantizationType.Q4_K
block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
blocks = torch.randint(0, 256, (128, type_size), dtype=torch.uint8, device="cuda")
blocks[:, 0:2] = torch.tensor([0x00, 0x38], dtype=torch.uint8, device="cuda")
blocks[:, 2:4] = torch.tensor([0x00, 0x34], dtype=torch.uint8, device="cuda")

try:
    eager = G.dequantize_blocks_Q4_K(blocks, block_size, type_size, torch.float16)
    fn = torch.compile(G.dequantize_blocks_Q4_K, dynamic=True)
    out = fn(blocks, block_size, type_size, torch.float16)
    torch.cuda.synchronize()
except Exception as exc:
    emit(False, f"{type(exc).__name__}: {str(exc)[:240]}")

if not torch.allclose(out, eager, atol=1e-3, rtol=1e-3):
    emit(False, "编译结果与 eager 不一致")

emit(True)
"""


def smoke_test() -> tuple[bool, str]:
    print("\n== 冒烟测试（真实编译一次 kernel）==")
    result = run([str(venv_python()), "-c", SMOKE], timeout=900)
    out = (result.stdout or "").strip().splitlines()
    if not out:
        reason = (result.stderr or "无输出").strip()[:300]
        print("  失败：" + reason.splitlines()[-1])
        return False, reason
    try:
        payload = json.loads(out[-1])
    except Exception:
        return False, out[-1][:300]
    if payload.get("ok"):
        print("  通过")
        return True, ""
    print("  失败：" + payload.get("reason", ""))
    return False, payload.get("reason", "")


def write_config(enabled: bool) -> None:
    if not CONFIG_PATH.exists():
        return
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["compile_dequant"] = bool(enabled)
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="安装 / 卸载 triton-windows（仅作用于项目 .venv）")
    parser.add_argument("--yes", action="store_true", help="全部使用默认值（自动选源）")
    parser.add_argument("--uninstall", action="store_true", help="直接卸载")
    parser.add_argument("--reinstall", action="store_true", help="卸载后重装")
    parser.add_argument("--channel", type=int, choices=(1, 2, 3), help="指定下载源")
    args = parser.parse_args()

    print("=" * 60)
    print("  triton-windows 安装向导")
    print("=" * 60)

    if venv_python() is None:
        print("[拒绝] 项目根目录下没有 .venv。")
        print("       请先运行 1-setup.bat 建好虚拟环境，本脚本不会碰系统 Python。")
        return 1

    info = probe_env()
    if info.get("error"):
        print(f"[错误] 无法探测环境：{info['error']}")
        return 1

    have = info.get("triton")
    torch_ver = info.get("torch")
    minor = torch_minor(torch_ver)

    print(f"  Python : {venv_python()}")
    print(f"  torch  : {torch_ver or '未安装'}")
    if info.get("cc"):
        print(f"  显卡   : compute capability {info['cc'][0]}.{info['cc'][1]}")
    print(f"  triton : {have or '未安装'}")

    # ---- 已安装：保留 / 卸载 / 重装 ----
    if have and not args.reinstall:
        expected = TORCH_TRITON_MAP.get(minor or "", "")
        match = (not expected) or have.startswith(expected.replace(".*", ""))
        print(f"\n  检测到已安装 triton-windows {have}" + ("" if match else f"（与 torch {torch_ver} 期望的 {expected} 不匹配）"))
        if args.uninstall:
            return 0 if do_uninstall() else 1
        print("\n  1) 保留并退出    2) 卸载    3) 重装")
        choice = "2" if args.yes and args.uninstall else ask("  选择", "1")
        if choice == "1":
            print("  已保留，未做任何修改。")
            return 0
        if choice == "2":
            return 0 if do_uninstall() else 1
        if choice != "3":
            print("  无效选择，已退出。")
            return 1

    # 走到这里仍带着 have，说明用户已经选了「重装」，显卡门槛不必再问一遍
    forced = bool(have)

    # ---- 版本推导 ----
    spec = TORCH_TRITON_MAP.get(minor or "")
    if not minor:
        print("\n[警告] 没有检测到 torch（或无法读取版本），无法确定兼容的 Triton 版本。")
        if not ask_yes_no("  仍然安装最新版 triton-windows？（可能不兼容）", False):
            print("  已取消。")
            return 0
        spec = "triton-windows"
    elif not spec:
        print(f"\n[警告] torch {torch_ver} 没有验证过的 Triton 组合。")
        print("       已知对应：2.8→3.4、2.9→3.5、2.10→3.6")
        if not ask_yes_no("  仍然安装最新版？（可能不兼容）", False):
            print("  已取消。")
            return 0
        spec = "triton-windows"
    else:
        spec = f"triton-windows=={spec}"
        print(f"\n  torch {torch_ver} → 安装 {spec}")

    # ---- 显卡门槛 ----
    cc = info.get("cc")
    if not info.get("cuda") or not cc:
        print("\n[警告] 没有可用的 NVIDIA GPU。triton-windows 装了也用不上。")
        if not ask_yes_no("  仍然继续安装？", False):
            print("  已取消。")
            return 0
    elif not gpu_ok(info) and not forced:
        print(f"\n[提示] 当前显卡是 sm_{cc[0]}{cc[1]}，低于 Triton 要求的 7.0。")
        print("       装上也能 import，但 torch.compile 无法为它生成 GPU kernel，")
        print("       加速不会生效（任务仍可正常运行）。")
        if not ask_yes_no("  仍然继续安装？", False):
            print("  已取消。")
            return 0

    # ---- 选源 ----
    prefix = ""
    if spec != "triton-windows":
        prefix = spec.split("==")[1].replace(".*", ".")
    if args.channel:
        label, index = CHANNELS[args.channel]
        print(f"\n== 使用指定下载源：{label} ==")
    else:
        label, index = choose_channel(prefix, auto_only=args.yes)

    # ---- 安装 ----
    if have:                     # 走到这里说明用户选了「重装」或传了 --reinstall
        do_uninstall()
    if not do_install(spec, index):
        return 1

    # ---- 验证 ----
    after = installed_version()
    if not after:
        print("\n[失败] 安装后仍然检测不到 triton-windows。")
        return 1
    print(f"\n  当前版本：{after}")

    # ---- 冒烟测试 + 同步配置 ----
    ok, reason = smoke_test()
    write_config(ok)
    if ok:
        print("\n[完成] 已启用 compile_dequant（config.json → true）。")
        print("       重启服务后生效；常驻进程也会在下次任务时重建。")
        return 0

    print("\n[已安装，但未启用] config.json 的 compile_dequant 保持 false。")
    if "No module named" in reason:
        print("       原因：当前虚拟环境缺少 STM 依赖，请先运行 1-setup.bat。")
    elif "6.1" in reason or "GPU is too old" in reason or "sm_" in reason or "GPUTooOldForTriton" in reason:
        print("       原因：当前显卡架构低于 Triton 要求（CC≥7.0），属于预期结果。")
    print("       卸载：再运行一次本脚本，选 2。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
