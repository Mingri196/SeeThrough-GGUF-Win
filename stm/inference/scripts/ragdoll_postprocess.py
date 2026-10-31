"""
ragdoll_postprocess.py – re-run further_extr on existing STM output directories.

Skips LayerDiff and Marigold entirely. Reads the per-character output directory
written by inference_psd.py and re-generates the PSD, cut images, and ragdoll bone
layers from the already-saved PNG layers.

Usage:
    ./ragdoll.sh workspace/output/fairy [workspace/output/character ...] [options]

Or via Python directly:
    .venv/bin/python inference/scripts/ragdoll_postprocess.py \
        workspace/output/fairy --ragdoll-confidence 0.15
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

from utils.inference_utils import further_extr


def main():
    parser = argparse.ArgumentParser(
        description="Re-run PSD assembly / ragdoll split on existing STM output directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "src_dirs", nargs="+", metavar="OUTPUT_DIR",
        help="Per-character output directories produced by the main STM pipeline "
             "(e.g. workspace/output/fairy). Accepts multiple directories.",
    )
    parser.add_argument("--save_to_psd", action="store_true",
                        help="Write a layered .psd alongside the output directory.")
    parser.add_argument("--save_cut_images", action="store_true",
                        help="Keep all extracted RGBA cut PNGs in <OUTPUT_DIR>/cuts/.")
    parser.add_argument("--tblr_split", action="store_true",
                        help="Split parts (handwear, eyes, etc.) into left/right components.")
    parser.add_argument("--ragdoll_split", action="store_true",
                        help="Split completed limbs into RTMPose-guided bone layers.")
    parser.add_argument("--ragdoll-overlap", type=float, default=0.2,
                        help="Joint overlap as a fraction of the shorter adjacent bone.")
    parser.add_argument("--ragdoll-confidence", type=float, default=0.3,
                        help="Minimum RTMPose joint confidence required for ragdoll split. "
                             "Lower this for characters with occluded limbs.")
    args = parser.parse_args()

    for src_dir in args.src_dirs:
        src_dir = Path(src_dir).resolve()
        if not src_dir.is_dir():
            print(f"[ERROR] not a directory: {src_dir}", flush=True)
            sys.exit(1)
        # Verify this looks like an STM output dir
        if not (src_dir / "src_img.png").exists():
            print(
                f"[ERROR] {src_dir} does not look like an STM output directory "
                "(src_img.png not found). Run the main pipeline first.",
                flush=True,
            )
            sys.exit(1)

        print(f"[postprocess] {src_dir}", flush=True)
        further_extr(
            str(src_dir),
            rotate=False,
            save_to_psd=args.save_to_psd,
            tblr_split=args.tblr_split,
            ragdoll_split=args.ragdoll_split,
            save_cut_images=args.save_cut_images,
            ragdoll_overlap=args.ragdoll_overlap,
            ragdoll_confidence=args.ragdoll_confidence,
        )
        if args.save_cut_images:
            print(f"cut images saved to {src_dir / 'cuts'}", flush=True)
        if args.save_to_psd:
            psd = src_dir.parent / f"{src_dir.name}.psd"
            print(f"psd saved to {psd}", flush=True)
        print(f"[postprocess] done: {src_dir.name}", flush=True)


if __name__ == "__main__":
    main()
