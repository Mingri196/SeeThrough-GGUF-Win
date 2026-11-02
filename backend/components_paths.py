#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""组件缓存目录（HF_HOME / TORCH_HOME）的解析。

首次推理需要的辅助组件（Marigold 文本编码器、动漫分割、LaMa、RTMPose、anime-seg）
默认缓存到 **项目内的 `components/`**，这样「一个文件夹 = 一套完整运行体」：删掉项目
就等于卸载干净，换个盘符也只是拖一个目录。

`config.json` 里的 `components_dir` 可以改成 `"user"`，退回系统默认位置
（Windows 是 `%USERPROFILE%\\.cache\\huggingface`），适合同时装了很多 AI 工具、
想共享同一份缓存的用户。

命令行用法（`1-setup.bat` 会调）：
    python backend/components_paths.py --ensure
    输出两行，可直接被 bat 的 for /f 读成 set 语句：
        HF_HOME=D:\\...\\components\\hf
        TORCH_HOME=D:\\...\\components\\torch
    配置成 `user` 时不输出任何内容（保持系统默认）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("STG_CONFIG") or (ROOT / "config.json"))
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = ROOT / CONFIG_PATH

DEFAULT_DIR = "components"
_USER_ALIASES = {"user", "system", "default", "off", "c"}

EXAMPLE_PATH = ROOT / "config.example.json"


def ensure_config_file() -> None:
    """首次运行时把 config.example.json 复制成 config.json。

    仓库里不再跟踪 config.json（它存每人的端口和路径），但所有入口脚本都假定
    这个文件存在，所以在模块导入时补一次。已存在就什么都不做。
    """
    if CONFIG_PATH.exists() or not EXAMPLE_PATH.is_file():
        return
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_bytes(EXAMPLE_PATH.read_bytes())
        print(f"[配置] 已从 {EXAMPLE_PATH.name} 生成 {CONFIG_PATH.name}")
    except OSError as exc:
        print(f"[WARN] 无法生成 {CONFIG_PATH.name}: {exc}")


ensure_config_file()


def _configured() -> str:
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        value = str(data.get("components_dir") or "").strip()
    except (OSError, ValueError):
        value = ""
    return value or DEFAULT_DIR


def is_project_local() -> bool:
    """组件是否放项目内（False = 用系统默认缓存）。"""
    return _configured().lower() not in _USER_ALIASES


def base_dir() -> Path | None:
    if not is_project_local():
        return None
    raw = Path(_configured())
    return raw if raw.is_absolute() else (ROOT / raw)


def hf_home() -> Path | None:
    base = base_dir()
    return None if base is None else base / "hf"


def torch_home() -> Path | None:
    base = base_dir()
    return None if base is None else base / "torch"


def user_cache_dir() -> Path:
    """系统默认缓存根（仅在配置成 user 时才有意义，供界面展示）。"""
    return Path(os.path.expanduser("~")) / ".cache" / "huggingface"


def apply_env(env: dict) -> dict:
    """把 HF_HOME / TORCH_HOME 写进子进程环境；配置成 user 时原样返回。"""
    hf, th = hf_home(), torch_home()
    if hf is not None:
        env["HF_HOME"] = str(hf)
        env["HF_HUB_CACHE"] = str(hf / "hub")
    if th is not None:
        env["TORCH_HOME"] = str(th)
    return env


def ensure_dirs() -> list[Path]:
    """创建 hf/ 与 torch/ 子目录（已存在则不动），返回创建/确认过的路径。"""
    made: list[Path] = []
    for path in (hf_home(), torch_home()):
        if path is None:
            continue
        path.mkdir(parents=True, exist_ok=True)
        if path.name == "hf":
            (path / "hub").mkdir(parents=True, exist_ok=True)
        made.append(path)
    return made


def main() -> int:
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else "--print"

    if arg == "--ensure":
        ensure_dirs()
        if is_project_local():
            print(f"HF_HOME={hf_home()}")
            print(f"TORCH_HOME={torch_home()}")
        return 0

    if arg == "--hf":
        print(hf_home() or "")
        return 0
    if arg == "--torch":
        print(torch_home() or "")
        return 0
    if arg == "--base":
        print(base_dir() or "")
        return 0

    print(f"components_dir = {_configured()} ({'project' if is_project_local() else 'user cache'})")
    print(f"HF_HOME        = {hf_home() or user_cache_dir()}")
    print(f"TORCH_HOME     = {torch_home() or Path.home() / '.cache' / 'torch'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
