# Design: frame-chunked attention to run LayerDiff3D at 1280 on a 6 GB card

## Problem

On the RTX 2060 (6 GB, ~5.59 GiB usable after the display), the quantized GGUF
path fits comfortably at 896 and marginally at 1024, but **1280 OOMs**. The
cause is *not* weights (Q4 keeps the UNet at 2148 MiB) and *not* attention
scores (xformers memory-efficient attention already bounds those). It is the
**dense activation tensors of the 13-frame body batch at the top-resolution
transformer block**.

At 1280:

- latent spatial = `1280 / 8 = 160`, so `160 × 160 = 25600` tokens per frame.
- body group = 13 frames -> the hidden-state tensor entering the top-res
  `Transformer3DModel` is `[13, 25600, 1280]` fp16 ≈ **850 MiB**, and several
  such tensors (norm input, attn output, FF intermediate, residual) are live at
  once. The failing allocation observed was a single **814 MiB** block with only
  ~750 MiB free.

Lowering resolution or quantizing further cannot recover this; the batch of 13
frames is processed jointly because of cross-frame (temporal) attention.

## Key observation

Not every block needs all 13 frames resident simultaneously:

- **Spatial blocks** (`transformer_blocks`: self-attn + cross-attn to the text
  embeds) operate **per-frame**. Frames are only stacked into the batch
  dimension; there is no interaction *between* frames inside these blocks.
- **Temporal / cross-frame blocks** (`temporal_transformer_blocks`, run every
  `temporal_block_stride` layers in `Transformer3DModel.forward`) are the only
  place frames interact — they attend across the frame axis.

So the 850 MiB peak is created by the **spatial** blocks holding all 13 frames
at once, even though they process each frame independently. If the spatial
blocks looped over frame-chunks (e.g. 1–3 frames at a time) and only the
temporal blocks saw the full stack, the peak spatial activation would drop by
`chunk / 13` — a 3-frame chunk cuts the 850 MiB spatial peak to ~200 MiB.

## Where this lives

`common/modules/layerdiffuse/transformer3d.py`:

- `Transformer3DModel.forward` — the loop at ~L410-L453 alternates spatial
  `block(...)` calls and, every `temporal_block_stride`, a `temporal_block(...)`
  call that mixes frames (`hidden_states = hidden_states + hidden_states_mix`).
- `hidden_states` enters shaped `[batch*frames, tokens, dim]` (frames folded
  into batch). `num_frames` is threaded through for the temporal blocks to
  unfold `[b, f, t, d]`.

The UNet wrapper is `common/modules/layerdiffuse/layerdiff3d.py`
(`UNetFrameConditionModel`, `SeeThroughTransformerBlock`), which calls into the
`Transformer3DModel`s inside its down/mid/up blocks.

## Proposed change (spatial frame-chunking)

Add an inference-only `spatial_frame_chunk` (int, default `0` = disabled) to
`Transformer3DModel`. In `forward`, when set and `not torch.is_grad_enabled()`:

```python
def _run_spatial(block, hs, **kw):
    if not self.spatial_frame_chunk or self.spatial_frame_chunk >= bf:
        return block(hs, **kw)
    outs = []
    for s in range(0, bf, self.spatial_frame_chunk):     # bf = batch*frames
        sl = slice(s, s + self.spatial_frame_chunk)
        kw_c = _slice_batch_kwargs(kw, sl)               # slice attn_mask,
                                                         # encoder_hidden_states,
                                                         # timestep, class_labels
        outs.append(block(hs[sl], **kw_c))
    return torch.cat(outs, dim=0)
```

- Replace the direct `block(hidden_states, ...)` spatial call (the `else` branch
  at ~L436) with `_run_spatial(block, hidden_states, ...)`.
- Leave the temporal block call untouched — it must see all frames.
- `_slice_batch_kwargs` must slice every per-sample tensor along dim 0 by the
  same frame slice: `attention_mask`, `encoder_hidden_states`,
  `encoder_attention_mask`, `timestep`, `class_labels`, and any
  `added_cond_kwargs` (`text_embeds`, `time_ids`) if they are passed through at
  this level. Anything broadcast/shared (e.g. a single timestep scalar) is
  passed as-is.

Wire the knob down from the pipeline: set
`unet.<...>.spatial_frame_chunk = N` for every `Transformer3DModel` after load
(a small recursive setter, mirroring how `enable_xformers_memory_efficient_attention`
walks modules), gated by a pipeline flag like `_st_spatial_frame_chunk`.

Expose it in `apply_layerdiff` / `inference_psd.py` as `--spatial-frame-chunk N`
(0 = off). Default off so nothing changes for cards that already fit.

## Expected effect

- Spatial peak activation: `× chunk/frames`. Chunk=3 -> ~0.23× -> 850 MiB
  becomes ~200 MiB; combined with the existing VAE-offload-during-diffusion this
  should bring 1280 under the ~5.59 GiB budget.
- Cost: more, smaller kernel launches in spatial blocks -> modest per-step
  slowdown (rough estimate +10–30% over the un-chunked spatial cost; the
  temporal blocks and matmuls are unchanged). Still far cheaper than the
  RAM-streaming offload path (which was 179 s/step at 1280).
- Numerically identical: chunking the batch dimension of per-frame blocks
  changes nothing about the math (no cross-frame interaction inside spatial
  blocks). Verify with an allclose check on a 512, 2-frame case: chunk=1 output
  vs chunk=0 output.

## Risks / things to check

1. **Hidden per-frame state.** Confirm the spatial `BasicTransformerBlock`
   variants used here carry no state across the batch dim (norms are per-token;
   AdaLayerNorm uses `timestep`/`class_labels` which must be sliced with the
   chunk). If any block uses a batch-wide reduction, chunking would change it —
   audit `SeeThroughTransformerBlock.forward` in `layerdiff3d.py`.
2. **added_cond_kwargs plumbing.** The SDXL time/text embeds enter at the UNet
   top; ensure the per-block `encoder_hidden_states` seen by
   `Transformer3DModel` is already per-frame-indexable (shape `[bf, ...]`), not
   a shared `[1, ...]` that must be expanded before slicing.
3. **`num_frames` divisibility.** `spatial_frame_chunk` need not divide 13;
   the last chunk is smaller. Temporal blocks still get the full `num_frames`.
4. **CFG batch doubling.** If `do_classifier_free_guidance` is ever enabled
   (currently guidance_scale=1.0, so no), `bf` doubles; chunking still works but
   the slice must respect the cond/uncond layout.

## Alternative levers (smaller wins, complementary)

- **Attention-slicing on the temporal blocks** (diffusers
  `set_attention_slice`) if a temporal score tensor ever spikes — unlikely to be
  the bottleneck here but cheap to add.
- **`torch.utils.checkpoint` on spatial blocks at inference** does *not* help:
  at inference activations are already freed as computation proceeds, so there
  is nothing to trade.
- **Sequential VAE decode** is already done; **VAE-offload-during-diffusion**
  (`_st_vae_diffusion_offload`) is already implemented and freed the last
  ~470 MiB.

## Validation plan

1. Unit: 512 res, 2 frames, `spatial_frame_chunk=1` vs `0` -> `torch.allclose`
   on UNet output (atol ~1e-3 fp16).
2. Memory: 1280 quantized, chunk=3, `--inference_steps 2` -> completes without
   OOM; record peak VRAM.
3. Quality: 1280 quantized, chunk=3, 30 steps -> compare layers against the
   896/1024 output; confirm no seams or per-frame artifacts (there should be
   none, since decode and temporal mixing are unchanged).

## Status

Not implemented. The quantized path (this doc's prerequisite) is done:
`common/utils/ggml_runtime.py`, `gguf_loader.py --quantized`,
`inference_psd.py --quantized`, and `_st_vae_diffusion_offload` in
`diffusers_kdiffusion_sdxl.py`. Frame-chunking is the remaining step to reach
the model's native 1280 on 6 GB.
