# See-through-GGUF

Python GGUF runtime for turning a single character image into layered PSD/PNG assets, depth, and RTMPose-guided ragdoll cuts. It keeps LayerDiff3D weights quantized and can stream them from CPU RAM so the 1280-pixel pipeline runs on GPUs with 6 GB VRAM.

This repository contains code only. Model weights are downloaded from Hugging Face.

## Requirements

- Linux x86-64 and an NVIDIA GPU
- CPython 3.12
- CUDA Toolkit 12.8 only when building SageAttention2 (`--skip-sage` does not need `nvcc`)
- 16 GB system RAM recommended
- About 20 GB free disk space for the environment and default models

## Install

```bash
git clone https://github.com/IceFog72/see-through-gguf
cd STM
./install.sh
.venv/bin/python download_models.py
```

The installer uses PyTorch 2.11.0 with CUDA 12.8. It builds standard upstream SageAttention 2.2.0 except on sm_75. On sm_75 it uses `build_sageattention2_sm_75.sh` only when that local script exists; otherwise it falls back to native attention. To disable SageAttention explicitly:

```bash
./install.sh --skip-sage
```

`run.sh` automatically selects native PyTorch attention when SageAttention is not installed. An explicit `--attention-backend` always overrides the automatic selection. `STM_SKIP_SAGE=1 ./install.sh` remains available for scripted installs.

## Run

```bash
./run.sh \
  --srcp path/to/character.png \
  --save_dir workspace/output \
  --gguf_layerdiff models/icefog72/seethroughv0.0.2_layerdiff3d_gguf/layerdiff3d-Q5_1.gguf \
  --gguf_depth models/icefog72/seethroughv0.0.1_marigold_gguf/marigold3d-Q8_0.gguf \
  --resolution 1280 \
  --resolution_depth auto \
  --inference_steps 30 \
  --inference_steps_depth 4 \
  --quantized \
  --save_to_psd \
  --save_cut_images \
  --tblr_split \
  --ragdoll_split \
  --ragdoll-confidence 0.3
```

For a source without transparency, the foreground segmenter is downloaded automatically. `--mask-resolution auto` (the default) uses the image's native long edge up to a VRAM-safe cap; pass a positive integer to override it. RTMPose models are downloaded when `--ragdoll_split` is first used.

Depth resolution is also automatic. It uses the LayerDiff canvas up to `768`, `1024`, `1280`, or `1536` according to available VRAM. `--resolution_depth -1` still forces the full LayerDiff resolution.

Use `--parts face handwear` to write only selected layers. Quote names containing spaces, such as `--parts "front hair"`. `hair` expands to front/back hair and `eyes` expands to the four eye layers. Head-layer requests generate the `head` locator automatically. LayerDiff evaluates the complete body group, plus the complete head group when any head layer is requested, because their frame embeddings were trained together. Body and head passes use independent seeded noise, so changing resolution or body output does not change the head sample.

Outputs are written below `--save_dir`: extracted RGBA layers, depth assets, a PSD, pose diagnostics, cut images, and a cut manifest containing pivots, bones, parents, and joint-overlap metadata. Adjust duplicated model-pixel coverage around joints with `--ragdoll-overlap` (default `0.2`).

`--ragdoll_split` uses RTMPose to locate joints and partition limb layers into overlapping bone segments. When RTMPose returns low confidence on one or more joints (common on characters with wings, props, or heavy clothing occlusion), STM prints a `[WARNING]` and skips the ragdoll split rather than crashing; the PSD and all other outputs are still saved. Lower the confidence gate with `--ragdoll-confidence` (default `0.3`) to attempt splitting despite occlusion, or omit `--ragdoll_split` and run `ragdoll.sh` later (see below).

STM estimates VRAM demand from the selected resolution and LayerDiff GGUF size, then compares it with detected GPU memory. It selects the least restrictive profile expected to fit: `resident`, `balanced`, `streamed`, or `low-vram`. For runs of 8 or more denoising steps, streamed profiles measure the first step, reserve at least 768 MiB, and keep weights in the remaining VRAM. Promoted UNet weights return to CPU before VAE decode; an OOM also returns them to CPU and retries with streaming. Short runs skip promotion because transfer overhead would not improve their total time. In quantized runs, LayerDiff VAE and TransparentVAE use float32 and move to GPU only for encode/decode.

With Q4_K_M, a 6 GB GPU selects `resident` at 720, `streamed` at 1024, and `low-vram` at 1280. At 1280, the first step measures VRAM near the conservative baseline; later steps use the measured headroom. Larger quantizations such as Q8 move to streaming sooner. An RTX 2060 6 GB reference run at 1280 with 30 body-group steps, 30 head-group steps, and 4 depth steps took about 24 minutes (`layerdiff3d-Q4_K_M.gguf`).

## Post-process

`ragdoll.sh` re-runs PSD assembly, tblr split, and ragdoll split on an already-completed inference output directory without re-running LayerDiff or Marigold. Use it to retry a failed or skipped ragdoll pass, adjust `--ragdoll-confidence`, or rebuild the PSD with different split options.

```bash
./ragdoll.sh workspace/output/character \
  --ragdoll_split \
  --ragdoll-confidence 0.15 \
  --ragdoll-overlap 0.2 \
  --tblr_split \
  --save_to_psd \
  --save_cut_images
```

Multiple directories can be passed at once:

```bash
./ragdoll.sh workspace/output/character workspace/output/fairy \
  --ragdoll_split --save_to_psd
```

Check `<OUTPUT_DIR>/pose_keypoints.json` for per-joint confidence scores to decide how far to lower `--ragdoll-confidence`.

## Models

- [LayerDiff3D GGUF](https://huggingface.co/icefog72/seethroughv0.0.2_layerdiff3d_gguf)
- [Marigold3D GGUF](https://huggingface.co/icefog72/seethroughv0.0.1_marigold_gguf)

Alternative uploaded quantizations can be selected with `download_models.py --layerdiff-file ... --depth-file ...`.

## Attribution

STM builds on the See-through [project (shitagaki-lab/see-through)](https://github.com/shitagaki-lab/see-through) and paper, [See-through: Layer Decomposition for Live2D Character Animation](https://arxiv.org/abs/2602.03749). Preserve upstream notices when redistributing modified components.

Licensed under Apache-2.0; see [LICENSE](LICENSE).
