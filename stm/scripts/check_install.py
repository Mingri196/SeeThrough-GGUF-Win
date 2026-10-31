#!/usr/bin/env python3
import argparse
import importlib

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-no-sage", action="store_true")
    args = parser.parse_args()

    for name in ("accelerate", "cv2", "diffusers", "gguf", "psd_tools", "rtmlib"):
        importlib.import_module(name)
    ort = importlib.import_module("onnxruntime")
    try:
        sage = importlib.import_module("sageattention")
    except ImportError:
        if not args.allow_no_sage:
            raise
        sage = None

    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("onnxruntime-gpu is installed incorrectly: CUDAExecutionProvider is unavailable")
    print(f"onnxruntime={ort.__version__} providers={','.join(ort.get_available_providers())}")
    print(f"sageattention={getattr(sage, '__version__', 'installed (version unknown)') if sage else 'not installed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
