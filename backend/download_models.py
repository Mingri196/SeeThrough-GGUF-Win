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
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
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


# ── 辅助组件并发下载 ────────────────────────────────────────────────────
# 五个组件各开一个线程同时下，每个组件自己维护状态与字节数，界面按行固定刷新。
# 某个组件完成/失败/重试只改它自己那一行的数据，不影响其它行。

S_WAITING = "等待"
S_PROBE = "探测大小"
S_DOWNLOAD = "下载中"
S_RETRY = "重试中"
S_EXISTS = "已存在"
S_DONE = "完成"
S_FAILED = "失败"

_STATUS_COLOR = {
    S_WAITING: "90",
    S_PROBE: "36",
    S_DOWNLOAD: "36",
    S_RETRY: "33",
    S_EXISTS: "90",
    S_DONE: "32",
    S_FAILED: "31",
}


def _char_width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _text_width(text: str) -> int:
    return sum(_char_width(ch) for ch in text)


def _fit(text: str, width: int) -> str:
    """按显示宽度截断并补空格。中文占两格，直接用 len() 对齐会歪。"""
    out, used = [], 0
    for ch in text:
        cw = _char_width(ch)
        if used + cw > width:
            if used < width:
                out.append("…")
                used += 1
            break
        out.append(ch)
        used += cw
    return "".join(out) + " " * max(0, width - used)


def _bytes_text(num: int | None) -> str:
    if num is None:
        return "—"
    if num == 0:
        return "0 B"
    if num < 1024:
        return f"{int(num)} B"
    if num < 1024 ** 2:
        return f"{num / 1024:.0f} KB"
    if num < 1024 ** 3:
        return f"{num / 1024 ** 2:.1f} MB"
    return f"{num / 1024 ** 3:.2f} GB"


def _enable_ansi() -> bool:
    """Windows 10+ 打开 VT 序列支持，才能把光标移回去重画那几行。"""
    try:
        if not sys.stdout.isatty():
            return False
    except Exception:
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if mode.value & 0x0004:
            return True
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


class _Comp:
    """单个组件的进度状态。每个线程只写自己的实例。"""

    def __init__(self, spec: dict):
        self.spec = spec
        self.key = spec["key"]
        self.label = spec["label"]
        self.repo = spec["repo"]
        self.files = list(spec["files"])
        self.lock = threading.Lock()
        self.status = S_WAITING
        self.done = 0
        self.total = 0
        self.total_known = False
        self.error = ""

    def set_status(self, status: str, error: str = "") -> None:
        with self.lock:
            self.status = status
            if error:
                self.error = error

    def set_total(self, total: int) -> None:
        with self.lock:
            if total:
                self.total = total
                self.total_known = True

    def add(self, n: int) -> None:
        with self.lock:
            self.done += max(0, int(n or 0))

    def set_done(self, value: int) -> None:
        with self.lock:
            self.done = max(0, int(value))

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "done": self.done,
                "total": self.total,
                "total_known": self.total_known,
                "error": self.error,
            }


class _Dashboard:
    """固定 N 行，按列重画。每行只读自己那组数据。"""

    def __init__(self, comps: list[_Comp]):
        self.comps = comps
        self.stream = sys.stdout
        self.ansi = _enable_ansi()
        self.rows_drawn = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last: dict[str, tuple[int, float, float]] = {}
        self._last_status: dict[str, str] = {}
        cols = shutil.get_terminal_size((100, 24)).columns
        self.width = max(70, min(cols - 1, 108))
        # 一行 = 序号3 + 名称(label_w+2) + 状态10 + 进度(bar+5) + 大小22 + 速度10
        self.bar = 18 if self.width >= 96 else (12 if self.width >= 84 else 8)
        self.label_w = max(10, min(24, self.width - self.bar - 52))

    # -- 渲染 ----------------------------------------------------------
    def _color(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.ansi else text

    def _row(self, idx: int, comp: _Comp) -> str:
        s = comp.snapshot()
        known = s["total_known"] and s["total"] > 0
        pct = int(min(100, s["done"] * 100 / s["total"])) if known else 0
        if s["status"] in (S_DONE, S_EXISTS):
            pct = 100
        filled = int(round(pct * self.bar / 100)) if known else 0
        bar = "█" * filled + "░" * (self.bar - filled)
        pct_text = f"{pct:>3}%" if known else "  --"

        # 速度：两次采样之间的增量，采样间隔不到 0.5 s 就沿用上次的值，免得闪烁
        now = time.monotonic()
        prev = self._last.get(comp.key)
        speed = 0.0
        if prev is None:
            self._last[comp.key] = (s["done"], now, 0.0)
        else:
            speed = prev[2]
            dt = now - prev[1]
            if dt >= 0.5:
                speed = max(0.0, (s["done"] - prev[0]) / dt)
                self._last[comp.key] = (s["done"], now, speed)
        if s["status"] != S_DOWNLOAD:
            speed = 0.0

        size = f"{_bytes_text(s['done'])} / {_bytes_text(s['total'])}" if known else _bytes_text(s["done"])
        speed_text = f"{_bytes_text(speed)}/s" if speed > 0 else ""

        status = _fit(s["status"], 9)
        colored = self._color(status, _STATUS_COLOR.get(s["status"], "0"))
        return (
            f"{idx:>2} "
            f"{_fit(comp.label, self.label_w)}  "
            f"{colored} "
            f"{bar} {pct_text} "
            f"{_fit(size, 21)} "
            f"{_fit(speed_text, 9)}"
        )

    def _header(self) -> None:
        self.stream.write("\n")
        self.stream.write(self._color("辅助组件下载（各组件独立进度，同时进行）", "1") + "\n")
        title = (
            f"{'#':>2} "
            f"{_fit('组件', self.label_w)}  "
            f"{_fit('状态', 9)} "
            f"{_fit('进度', self.bar + 5)} "
            f"{_fit('已下载 / 总大小', 21)} "
            f"{_fit('速度', 9)}"
        )
        self.stream.write(self._color(title, "90") + "\n")
        self.stream.write(self._color("-" * (self.label_w + self.bar + 52), "90") + "\n")
        self.stream.flush()

    def _draw(self) -> None:
        rows = [self._row(i, c) for i, c in enumerate(self.comps, 1)]
        if self.ansi:
            if self.rows_drawn:
                self.stream.write(f"\033[{self.rows_drawn}A")
            for row in rows:
                self.stream.write("\033[2K" + row + "\n")
            self.stream.flush()
            self.rows_drawn = len(rows)
            return
        # 没有 VT 支持：只在状态变化时追加一行，探测/等待这类过程态不刷屏
        for comp, row in zip(self.comps, rows):
            status = comp.snapshot()["status"]
            if status in (S_WAITING, S_PROBE):
                continue
            if self._last_status.get(comp.key) == status:
                continue
            self._last_status[comp.key] = status
            s = comp.snapshot()
            extra = f" {_bytes_text(s['done'])}"
            if s["total_known"] and s["total"]:
                extra += f" / {_bytes_text(s['total'])}"
            self.stream.write(f"[{status}] {comp.label}{extra}\n")
        self.stream.flush()

    # -- 生命周期 ------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.wait(0.25):
            try:
                self._draw()
            except Exception:
                pass

    def start(self) -> None:
        self._header()
        self._draw()
        self._thread = threading.Thread(target=self._loop, name="dl-dashboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            self._draw()
        except Exception:
            pass
        if not self.ansi:
            for comp in self.comps:
                s = comp.snapshot()
                if not s["error"]:
                    continue
                self.stream.write(f"[失败] {comp.label}: {s['error']}\n")
        self.stream.write("\n")


def _make_tracker(comp: _Comp):
    """给 hf_hub_download 用的进度条替身：不自己画，只把字节数喂给组件状态。"""
    from huggingface_hub.utils import tqdm as hf_tqdm

    class _Tracker(hf_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.pop("disable", None)
            initial = kwargs.get("initial") or 0
            total = kwargs.get("total") or 0
            super().__init__(*args, **kwargs)
            self.disable = False
            if total:
                comp.set_total(total)
            if initial:
                comp.add(initial)

        def display(self, *args, **kwargs):  # 不输出，界面由 _Dashboard 负责
            pass

        def close(self):  # tqdm 收尾会写换行，会打乱光标位置
            pass

        def update(self, n=1):
            if n:
                comp.add(n)

    return _Tracker


def _component_worker(comp: _Comp, endpoint: str, force: bool) -> None:
    from huggingface_hub import hf_hub_download

    hub_dir = torch_checkpoints_dir() if comp.spec.get("torch_hub") else None
    if hub_dir:
        hub_dir.mkdir(parents=True, exist_ok=True)

    comp.set_status(S_PROBE)
    # 先看缓存里有什么：缓存命中就以本地文件大小为准（HEAD 走 hf-mirror 时
    # Content-Length 经常不准，小文件会返回几十字节，拿它比大小会误判成未缓存）
    cached_paths: dict[str, Path | None] = {}
    sizes: dict[str, int] = {}
    for filename in comp.files:
        hit = None
        if not force:
            try:
                probe = Path(hf_hub_download(repo_id=comp.repo, filename=filename, local_files_only=True))
                if probe.exists():
                    hit = probe
            except Exception:
                hit = None
        cached_paths[filename] = hit
        if hit is not None:
            sizes[filename] = hit.stat().st_size
        else:
            sizes[filename] = remote_size(endpoint, comp.repo, filename) or 0
    comp.set_total(sum(sizes.values()))

    counted = 0
    downloaded_any = False
    for filename in comp.files:
        short = filename.split("/")[-1]

        cached_path = cached_paths.get(filename)
        if cached_path is not None:
            counted += cached_path.stat().st_size
            comp.set_done(counted)
            comp.set_status(S_EXISTS)
            if hub_dir:
                dest = hub_dir / short
                if not dest.exists() or dest.stat().st_size != cached_path.stat().st_size:
                    shutil.copy2(cached_path, dest)
            continue

        comp.set_status(S_DOWNLOAD)
        last_exc: Exception | None = None
        for attempt in (1, 2):
            try:
                path = Path(
                    hf_hub_download(
                        repo_id=comp.repo,
                        filename=filename,
                        force_download=force,
                        tqdm_class=_make_tracker(comp),
                    )
                )
                counted += path.stat().st_size
                comp.set_done(counted)
                downloaded_any = True
                if hub_dir:
                    dest = hub_dir / short
                    if force or not dest.exists() or dest.stat().st_size != path.stat().st_size:
                        shutil.copy2(path, dest)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if is_missing_file(exc):
                    last_exc = None  # 仓库里没有这个文件，跳过即可
                    break
                if attempt == 1:
                    comp.set_status(S_RETRY, f"{type(exc).__name__}: {exc}")
                    time.sleep(1.0)
                    comp.set_status(S_DOWNLOAD)
        if last_exc is not None:
            comp.set_status(S_FAILED, f"{type(last_exc).__name__}: {last_exc}")
            raise last_exc

    s = comp.snapshot()
    comp.set_done(max(s["done"], s["total"] if s["total_known"] else 0))
    comp.set_status(S_DONE if downloaded_any else S_EXISTS)


def download_components(endpoint: str, selected: list[dict], force: bool,
                        jobs: int = 0, dashboard: bool = True) -> list[tuple[str, str]]:
    """并发下载辅助组件，返回 [(组件名, 错误)]。"""
    comps = [_Comp(spec) for spec in selected]
    if not comps:
        return []

    dash = _Dashboard(comps) if dashboard else None
    if dash:
        dash.start()
    else:
        for comp in comps:
            print(f"[组件] {comp.label} ← {endpoint}/{comp.repo}")

    failures: list[tuple[str, str]] = []

    def run(comp: _Comp) -> None:
        try:
            _component_worker(comp, endpoint, force)
        except Exception as exc:
            comp.set_status(S_FAILED, f"{type(exc).__name__}: {exc}")
            failures.append((comp.label, f"{type(exc).__name__}: {exc}"))

    workers = jobs if jobs > 0 else len(comps)
    with ThreadPoolExecutor(max_workers=min(workers, len(comps))) as pool:
        list(pool.map(run, comps))

    if dash:
        dash.stop()

    done = sum(1 for c in comps if c.snapshot()["status"] in (S_DONE, S_EXISTS))
    if not failures:
        print(f"[OK] {len(comps)} 个辅助组件全部就位（{done} 个已就绪）")
    else:
        print(f"[警告] {len(comps)} 个辅助组件中 {len(failures)} 个未就绪：")
        for label, err in failures:
            print(f"       - {label}: {err}")
    return failures


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
    parser.add_argument("--jobs", type=int, default=0,
                        help="同时下载的组件数，0 = 全部并发")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="关闭多行进度面板，改为逐行输出（日志/CI 用）")
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

    comp_failures: list[tuple[str, str]] = []
    if want_components:
        comp_failures = download_components(
            endpoint,
            selected,
            args.force,
            jobs=args.jobs,
            dashboard=not args.no_dashboard,
        )

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
        print("[警告] 部分辅助组件未就绪（不影响启动，首次推理时会再尝试）：")
        for label, err in comp_failures:
            print(f"  - {label}: {err}")
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
