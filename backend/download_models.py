#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按网络情况自动选择下载源，准备运行所需的模型与组件。

默认只下载两个 GGUF 主模型；加 --with-components 会把首次推理要用的辅助组件
（Marigold 文本编码器、动漫实例分割、LaMa、可选 RTMPose）也一并缓存好，
这样首次推理不需要再联网。

用法：
  python backend\\download_models.py                    # 只下两个 GGUF
  python backend\\download_models.py --with-components  # GGUF + 辅助组件（1-setup.bat 默认）
  python backend\\download_models.py --components-only  # 只补辅助组件
  python backend\\download_models.py --check-only       # 只探测，不下载
  python backend\\download_models.py --endpoint hf-mirror
  python backend\\download_models.py --no-ragdoll       # 不预下 RTMPose（省 293 MB）

规则：先探测 huggingface.co 与 hf-mirror.com，国内通常落到镜像、海外落到官方；
也可用 --endpoint 强制指定。下载位置遵循 HF_HOME / TORCH_HOME 环境变量。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import components_paths  # noqa: E402
import netcheck  # noqa: E402
import tls  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

MODELS = (
    {
        "key": "layerdiff_gguf",
        "repo": "icefog72/seethroughv0.0.2_layerdiff3d_gguf",
        "default_file": "layerdiff3d-Q5_1.gguf",
        "label": "LayerDiff3D",
    },
    {
        "key": "depth_gguf",
        "repo": "icefog72/seethroughv0.0.1_marigold_gguf",
        "default_file": "marigold3d-Q8_0.gguf",
        "label": "Marigold3D",
    },
)

# 首次推理时 STM 会拉取的辅助组件。文件清单对应代码里的实际调用：
#   marigold_depth_pipeline.py: CLIPTextModel / CLIPTokenizer from prs-eth/marigold-depth-v1-1
#   animeinsseg/skytnt_segmentation.py: skytnt/anime-seg 的 isnetis.ckpt（无 alpha 抠图；「透明图也精修」也用它）
#   animeinsseg/instance_segmentation.py: rtmdetl_e60.ckpt
#   lama_inpainter/__init__.py: torch.hub -> <torch home>/hub/checkpoints/lama_large_512px.ckpt
#   common/utils/ragdoll.py: Tau-J/RTMPose 的两个 onnx zip（仅骨骼切割）
COMPONENTS = (
    {
        "key": "marigold",
        "label": "Marigold 文本编码器",
        "repo": "prs-eth/marigold-depth-v1-1",
        "approx_mb": 1300,
        "files": (
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "tokenizer/tokenizer_config.json",
            "tokenizer/special_tokens_map.json",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
        ),
    },
    {
        "key": "animeseg",
        "label": "动漫前景分割",
        "repo": "skytnt/anime-seg",
        "approx_mb": 195,
        "files": ("isnetis.ckpt",),
    },
    {
        "key": "animeinsseg",
        "label": "动漫实例分割",
        "repo": "dreMaz/AnimeInstanceSegmentation",
        "approx_mb": 219,
        "files": ("rtmdetl_e60.ckpt",),
    },
    {
        "key": "lama",
        "label": "LaMa 修补权重",
        "repo": "dreMaz/AnimeMangaInpainting",
        "approx_mb": 195,
        "files": ("lama_large_512px.ckpt",),
        "torch_hub": True,
    },
    {
        "key": "rtmpose",
        "label": "RTMPose（骨骼切割）",
        "repo": "Tau-J/RTMPose",
        "approx_mb": 293,
        "files": (
            "rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
            "rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
        ),
        "ragdoll_only": True,
    },
)

MIN_VALID_BYTES = 50 * 1024 * 1024  # 小于这个大小基本可判定为下载中断/占位文件


def load_targets() -> list[dict]:
    """从 config.json 读取目标路径，保证下载地址与后端实际读取的路径一致。"""
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[警告] config.json 解析失败，使用默认路径: {exc}")

    targets = []
    for item in MODELS:
        configured = cfg.get(item["key"], f"models/{item['default_file']}")
        path = Path(configured)
        if not path.is_absolute():
            path = ROOT / path
        targets.append({**item, "path": path})
    return targets


def curl_binary() -> str | None:
    found = shutil.which("curl")
    if found:
        return found
    if os.name == "nt":
        fallback = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "curl.exe"
        if fallback.exists():
            return str(fallback)
    return None


def curl_download(url: str, dest: Path) -> bool:
    """Python 侧 TLS 失败时的兜底：curl 走系统证书库，支持断点续传与重试。"""
    curl = curl_binary()
    if not curl:
        return False
    part = dest.with_suffix(dest.suffix + ".part")
    cmd = [curl, "-L", "--fail", "--retry", "3", "--retry-delay", "2", "-C", "-", "-o", str(part), url]
    print(f"       回退到 curl 下载（支持续传）：{dest.name}")
    code = subprocess.call(cmd)
    if code != 0 or not part.exists():
        return False
    part.replace(dest)
    return True


def remote_size(endpoint: str, repo: str, filename: str) -> int | None:
    import requests

    url = f"{endpoint}/{repo}/resolve/main/{filename}"
    try:
        resp = requests.head(url, allow_redirects=True, timeout=20)
        if resp.status_code == 200:
            return int(resp.headers.get("Content-Length") or 0) or None
    except Exception:
        return None
    return None


def human(num: int | None) -> str:
    if not num:
        return "未知"
    return f"{num / 1024 ** 3:.2f} GB" if num >= 1024 ** 3 else f"{num / 1024 ** 2:.0f} MB"


def torch_checkpoints_dir() -> Path:
    """与 torch.hub.get_dir()/checkpoints 保持一致（不 import torch，避免拖慢脚本）。"""
    base = os.environ.get("TORCH_HOME")
    root = Path(base) if base else (Path.home() / ".cache" / "torch")
    return root / "hub" / "checkpoints"


def cache_paths() -> tuple[str, str]:
    try:
        from huggingface_hub import constants

        hf_cache = constants.HF_HUB_CACHE
    except Exception:
        hf_cache = str(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    return hf_cache, str(torch_checkpoints_dir())


def is_missing_file(exc: Exception) -> bool:
    """区分「仓库里没有这个文件」与真正的网络/TLS 失败。"""
    text = str(exc)
    if "404" in text or "EntryNotFound" in text or "not found" in text.lower():
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 404


def download_models(endpoint: str, targets: list[dict], force: bool) -> list[tuple[dict, Exception]]:
    from huggingface_hub import hf_hub_download

    failures = []
    for t in targets:
        path: Path = t["path"]
        if path.exists() and not force:
            if path.stat().st_size >= MIN_VALID_BYTES:
                print(f"\n[跳过] {t['label']} 已存在（{human(path.stat().st_size)}）：{path}")
                continue
            print(f"\n[重下] {t['label']} 本地文件过小（{path.stat().st_size} 字节），判定为不完整")

        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[下载] {t['label']} ← {endpoint}/{t['repo']}")
        try:
            hf_hub_download(repo_id=t["repo"], filename=path.name, local_dir=str(path.parent), force_download=force)
        except Exception as exc:
            print(f"[提示] huggingface_hub 下载失败（{type(exc).__name__}），尝试 curl 回退")
            url = f"{endpoint}/{t['repo']}/resolve/main/{path.name}"
            if not curl_download(url, path):
                failures.append((t, exc))
                print(f"[失败] {t['label']}: {type(exc).__name__}: {exc}")
                continue

        if not path.exists() or path.stat().st_size < MIN_VALID_BYTES:
            failures.append((t, RuntimeError("下载后文件不完整")))
            print(f"[失败] {t['label']} 下载后文件不完整：{path}")
        else:
            print(f"[完成] {t['label']} → {path}（{human(path.stat().st_size)}）")
    return failures


def download_component(comp: dict, force: bool) -> tuple[list[str], list[tuple[str, Exception]]]:
    """下载单个组件的全部文件；返回 (缺失文件, 失败列表)。"""
    from huggingface_hub import hf_hub_download

    missing: list[str] = []
    failures: list[tuple[str, Exception]] = []
    hub_dir = torch_checkpoints_dir() if comp.get("torch_hub") else None
    if hub_dir:
        hub_dir.mkdir(parents=True, exist_ok=True)

    for filename in comp["files"]:
        short = filename.split("/")[-1]
        try:
            cached = Path(hf_hub_download(repo_id=comp["repo"], filename=filename, force_download=force))
            if hub_dir:
                dest = hub_dir / short
                if force or not dest.exists() or dest.stat().st_size != cached.stat().st_size:
                    shutil.copy2(cached, dest)
                    print(f"        {short} → {dest}")
            else:
                print(f"        {short}（{human(cached.stat().st_size)}）")
        except Exception as exc:
            if is_missing_file(exc):
                missing.append(filename)
                print(f"        {short}: 远端不存在，跳过")
            else:
                failures.append((filename, exc))
                print(f"        {short}: 失败（{type(exc).__name__}）")
    return missing, failures


def main() -> int:
    # 组件缓存默认落在项目内 components/（HF_HOME / TORCH_HOME），与 WebUI 保持一致
    try:
        components_paths.ensure_dirs()
        components_paths.apply_env(os.environ)
    except OSError:
        pass

    parser = argparse.ArgumentParser(description="下载 SeeThrough GGUF 主模型与运行期组件（自动选择下载源）")
    parser.add_argument("--endpoint", choices=("auto", "huggingface", "hf-mirror"), default="auto")
    parser.add_argument("--check-only", action="store_true", help="只探测网络与远端文件，不下载")
    parser.add_argument("--force", action="store_true", help="本地已存在也重新下载")
    parser.add_argument("--with-components", action="store_true", help="同时预取首次推理所需的辅助组件")
    parser.add_argument("--components-only", action="store_true", help="只预取辅助组件，不下载 GGUF")
    parser.add_argument("--no-ragdoll", action="store_true", help="不预取 RTMPose（省约 293 MB）")
    parser.add_argument("--component", action="append", choices=[c["key"] for c in COMPONENTS],
                        help="只预取指定组件，可重复")
    args = parser.parse_args()

    if str(os.environ.get("STG_SKIP_MODELS", "")).strip() == "1":
        print("[跳过] STG_SKIP_MODELS=1，未执行任何下载")
        return 0

    # Windows 默认不支持符号链接，huggingface_hub 每个仓库都会刷一遍警告，关掉
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    bundle = tls.configure_env()
    if bundle:
        print(f"[证书] 已合并系统根证书：{Path(bundle).name}")

    print("== 网络探测 ==")
    info = netcheck.detect(prefer=None if args.endpoint == "auto" else args.endpoint)
    for name, item in info["results"].items():
        if item["ok"]:
            print(f"  [可达]   {name:<12} {item['latency_ms']:>5} ms")
        else:
            print(f"  [不可达] {name:<12} {item.get('error') or item.get('status')}")
    print(f"== 选择下载源：{info['chosen'] or '无'} ==")

    if not info["chosen"]:
        print("[错误] huggingface.co 与 hf-mirror.com 都不可达。")
        print("       请检查网络/代理后重试，或手动下载后放入 models/：")
        for t in load_targets():
            print(f"       - {t['repo']} / {t['path'].name}")
        return 2

    endpoint = info["endpoint"]
    os.environ["HF_ENDPOINT"] = endpoint
    # 走镜像时关掉 HF 的 xet 传输：它会把请求直连 cas-bridge.xethub.hf.co，
    # 在国内网络下经常读超时（实测卡在 31 MB 数分钟），退回普通 HTTP 反而稳定。
    if "hf-mirror.com" in endpoint and "HF_HUB_DISABLE_XET" not in os.environ:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    want_components = args.with_components or args.components_only
    selected = [c for c in COMPONENTS if not args.component or c["key"] in args.component]
    if args.no_ragdoll:
        selected = [c for c in selected if not c.get("ragdoll_only")]

    targets = [] if args.components_only else load_targets()

    if targets:
        print("\n== 主模型 ==")
        for t in targets:
            size = remote_size(endpoint, t["repo"], t["path"].name)
            exists = t["path"].exists()
            local = human(t["path"].stat().st_size) if exists else "缺失"
            rel = t["path"].relative_to(ROOT) if t["path"].is_relative_to(ROOT) else t["path"]
            print(f"  {t['label']:<12} 远端 {human(size):>8} · 本地 {local:>8} · {rel}")

    if want_components:
        print("\n== 辅助组件（首次推理用） ==")
        total_mb = sum(c["approx_mb"] for c in selected)
        for c in selected:
            print(f"  {c['label']:<18} 约 {c['approx_mb']:>5} MB · {c['repo']}")
        print(f"  合计约 {total_mb / 1024:.1f} GB")
    else:
        print("\n[提示] 未指定 --with-components，只处理两个 GGUF；首次推理时 STM 仍会自行下载组件。")

    if args.check_only:
        print("\n[完成] 仅探测，未下载。")
        return 0

    failures = download_models(endpoint, targets, args.force) if targets else []

    comp_failures: list[tuple[str, str, Exception]] = []
    if want_components:
        for comp in selected:
            print(f"\n[组件] {comp['label']} ← {endpoint}/{comp['repo']}")
            missing, bad = download_component(comp, args.force)
            for filename, exc in bad:
                comp_failures.append((comp["label"], filename, exc))
            if missing:
                print(f"       （{len(missing)} 个可选文件远端不存在，已跳过）")
            if not bad:
                print(f"       {comp['label']} 就绪")

    hf_cache, torch_cache = cache_paths()
    print("\n== 缓存位置 ==")
    print(f"  HuggingFace: {hf_cache}")
    print(f"  Torch hub  : {torch_cache}")

    if failures:
        print("\n== 主模型下载未全部成功 ==")
        for t, exc in failures:
            print(f"  - {t['label']}: {exc}")
            print(f"    手动地址（官方）: https://huggingface.co/{t['repo']}/resolve/main/{t['path'].name}")
            print(f"    手动地址（镜像）: https://hf-mirror.com/{t['repo']}/resolve/main/{t['path'].name}")
        return 1

    if comp_failures:
        print("\n[警告] 部分辅助组件未就绪（不影响启动，首次推理时会再尝试）：")
        for label, filename, exc in comp_failures:
            print(f"  - {label} / {filename}: {type(exc).__name__}: {exc}")
    elif args.components_only:
        print("\n[OK] 辅助组件已就位，首次推理无需再联网。")
        print("     主模型未处理：请把 GGUF 放进 models/，或运行本脚本（不带 --components-only）下载。")
    elif want_components:
        print("\n[OK] 主模型与辅助组件已就位，首次推理无需再联网。")
    else:
        print("\n[OK] 主模型已就位。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
