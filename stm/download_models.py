#!/usr/bin/env python3
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


LAYER_REPO = "icefog72/seethroughv0.0.2_layerdiff3d_gguf"
DEPTH_REPO = "icefog72/seethroughv0.0.1_marigold_gguf"


def download(repo: str, filename: str, root: Path) -> Path:
    return Path(hf_hub_download(repo_id=repo, filename=filename, local_dir=root / repo))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download STM GGUF models from Hugging Face")
    parser.add_argument("--layerdiff-file", default="layerdiff3d-Q5_1.gguf")
    parser.add_argument("--depth-file", default="marigold3d-Q8_0.gguf")
    parser.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parent / "models")
    args = parser.parse_args()

    print(download(LAYER_REPO, args.layerdiff_file, args.models_dir))
    print(download(DEPTH_REPO, args.depth_file, args.models_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
