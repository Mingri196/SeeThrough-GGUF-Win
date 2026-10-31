#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""常驻推理进程：把 STM 的 pipeline 留在进程里，下一轮任务免加载。

协议（stdin 逐行 JSON，stdout 原样透出 STM 日志）：

    输入  {"argv": [... inference_psd.py 的命令行参数 ...], "reset": false}
    输出  ... STM 原始输出 ...
          @@STG_DONE@@ {"rc": 0}

为什么要单独一个进程：STM 的 `inference_utils` 把 `layerdiff_pipeline` /
`marigold_pipeline` 存在模块级全局里，只要进程不退出、模型路径不变，第二次
调用就直接复用，省掉 3.4 GB GGUF 的读取与显存分配（本机约 100 秒）。

实现上不复制 STM 的流程，而是把 `inference_psd.py` 在同一个模块命名空间里
重新执行一遍——这样行为与一次性进程完全一致，只是全局缓存被保留下来。
"""
from __future__ import annotations

import gc
import json
import os
import sys
import traceback
import types
from pathlib import Path

READY = "@@STG_WORKER_READY@@"
DONE = "@@STG_DONE@@"

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = Path(os.environ.get("STG_INFERENCE_SCRIPT") or (ROOT / "stm" / "inference" / "scripts" / "inference_psd.py"))

# 与一次性进程保持同一套 import 路径（server 会通过 PYTHONPATH 传进来）
for entry in reversed([p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def reset_models() -> None:
    """模型路径/分辨率变化时卸载旧 pipeline，避免用错模型。"""
    try:
        from utils import inference_utils as iu
    except Exception:
        return
    if getattr(iu, "layerdiff_pipeline", None) is None and getattr(iu, "marigold_pipeline", None) is None:
        return
    iu.layerdiff_pipeline = None
    iu.marigold_pipeline = None
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    print("[常驻] 已卸载上一轮模型，按新设置重新加载", flush=True)


def main() -> int:
    if not SCRIPT.exists():
        print(f"[常驻] 找不到推理脚本: {SCRIPT}", flush=True)
        return 2

    code = compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec")
    module = types.ModuleType("stg_inference_main")
    module.__file__ = str(SCRIPT)
    sys.modules["stg_inference_main"] = module

    print(READY, flush=True)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            spec = json.loads(raw)
        except Exception:
            continue
        if spec.get("cmd") == "exit":
            break
        if spec.get("reset"):
            reset_models()

        module.__dict__["__name__"] = "__main__"
        # 每轮按任务设置：图层修补补丁（丢层 / 眼白叠序）在调用时才读这个变量
        os.environ["STG_LAYER_FIX"] = "1" if spec.get("layer_fix", True) else "0"
        sys.argv = [str(SCRIPT)] + [str(a) for a in (spec.get("argv") or [])]
        rc = 0
        try:
            exec(code, module.__dict__)
        except SystemExit as exc:                     # argparse 出错
            rc = exc.code if isinstance(exc.code, int) else 1
        except Exception:
            traceback.print_exc()
            rc = 1
        print(f"{DONE} {json.dumps({'rc': rc})}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
