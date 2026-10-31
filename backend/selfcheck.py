#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""环境自检：确认虚拟环境里的关键依赖、CUDA、模型文件与补丁状态。

用法：.venv\\Scripts\\python.exe backend\\selfcheck.py
退出码 0 = 核心依赖可用（模型缺失只算警告），1 = 核心依赖缺失。
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ["flask", "PIL", "torch", "torchvision", "diffusers", "transformers", "gguf", "psd_tools", "onnxruntime"]
OPTIONAL = ["accelerate", "safetensors", "sklearn", "cv2", "rtmlib"]


def main() -> int:
    print(f"Python  : {sys.version.split()[0]}  ({sys.executable})")
    print(f"项目根  : {ROOT}")
    if sys.version_info[:2] != (3, 12):
        print(f"[警告] 期望 Python 3.12，当前 {sys.version_info.major}.{sys.version_info.minor}")

    failed = []
    for name in CORE:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "")
            print(f"[OK]   {name:<14} {version}")
        except Exception as exc:
            failed.append(name)
            print(f"[缺失] {name:<14} {exc}")
    for name in OPTIONAL:
        try:
            importlib.import_module(name)
            print(f"[OK]   {name:<14} (可选)")
        except Exception as exc:
            print(f"[跳过] {name:<14} {exc}")

    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
            archs = ", ".join(torch.cuda.get_arch_list())
            print(f"[OK]   CUDA {torch.version.cuda} · {name} · {total:.1f} GB")
            print(f"       编译架构 {archs}")
            cc = f"sm_{torch.cuda.get_device_capability(idx)[0]}{torch.cuda.get_device_capability(idx)[1]}"
            if cc not in torch.cuda.get_arch_list():
                print(f"[警告] 当前显卡是 {cc}，但该 PyTorch 轮子不包含它，GPU 算子会报 no kernel image")
        else:
            print("[警告] torch 已安装，但 CUDA 不可用（检查驱动或安装的是否为 cu128 轮子）")
    except Exception as exc:
        print(f"[跳过] CUDA 检查失败: {exc}")

    try:
        cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[错误] config.json 读取失败: {exc}")
        return 1

    base = ROOT

    def path_of(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else base / p

    stm_script = path_of(cfg.get("stm_path", "stm")) / "inference" / "scripts" / "inference_psd.py"
    print(f"[{'OK' if stm_script.exists() else '缺失'}] STM 推理脚本 {stm_script}")
    utils = path_of(cfg.get("stm_path", "stm")) / "common" / "utils" / "inference_utils.py"
    applied = utils.exists() and "__STG_PATCHED_INFERENCE_UTILS__" in utils.read_text(encoding="utf-8", errors="ignore")
    print(f"[{'OK' if applied else '警告'}] use_lama 补丁 {'已应用' if applied else '未应用（LaMa 将无法关闭）'}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
