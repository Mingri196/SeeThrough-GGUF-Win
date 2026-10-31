# See-through GGUF converter v2

This converter targets the exact three model revisions validated by the probe and the user's dry-run manifests:

- `layerdiff3d` -> `seethrough_layerdiff3d`
- `marigold` -> `seethrough_marigold3d`
- `semantic-sam` -> `seethrough_semantic_sam_h_hq`

## What changed from v1

The original dry run proved that all source tensor schemas match exactly, but it also showed that the generated GGUF would unnecessarily contain the CLIP text encoders.

The See-through inference code uses a fixed conditioning set:

- LayerDiff: 13 body tags + 11 head tags. `cache_tag_embeds()` computes their SDXL CLIP embeddings once and then deletes both text encoders.
- Marigold: `cache_tag_embeds()` computes the single empty-prompt embedding once and then deletes its text encoder.

Therefore v2 defaults to `--conditioning cached`:

- validates the complete original checkpoint first, including all CLIP tensors;
- precomputes the exact fixed conditioning during actual conversion;
- stores the conditioning tensors in GGUF;
- omits CLIP weights from the GGUF;
- leaves `--conditioning full` available if a general-purpose model container is wanted instead.

The cached tensors are stored as BF16 by default. On CUDA, conditioning computation defaults to BF16. On CPU, it defaults to FP32 computation followed by BF16 storage for compatibility. Override with `--conditioning-compute-dtype` when desired.

## Dry-run results already validated

The supplied v1 dry-run manifests validated the canonical source checkpoints:

| Model | Source tensors | v1 Q5_1 output | v2 cached-conditioning Q5_1 |
|---|---:|---:|---:|
| LayerDiff3D | 3969 | 4.28 GiB | 3.60 GiB / 3260 tensors |
| Marigold3D | 1610 | 1.96 GiB | 1.66 GiB / 1239 tensors |
| SemanticSAM | 817 | 474.25 MiB | 474.25 MiB / 817 tensors |

The full source schemas remain pinned and validated before the v2 output tensor list is formed.

## Dependencies on Arch Linux

```bash
git clone https://github.com/ggml-org/llama.cpp.git
python -m pip install -e ./llama.cpp/gguf-py
python -m pip install numpy torch safetensors huggingface_hub tqdm transformers
```

Using the See-through Python environment is preferable because its Hugging Face cache and Transformers installation are already available.

## Recommended verification

```bash
python convert_seethrough_to_gguf_v2.py layerdiff3d \
  --type q5_1 --conditioning cached --dry-run \
  --manifest layerdiff3d-q5_1-v2.map.json

python convert_seethrough_to_gguf_v2.py marigold \
  --type q5_1 --conditioning cached --dry-run \
  --manifest marigold-q5_1-v2.map.json

python convert_seethrough_to_gguf_v2.py semantic-sam \
  --type q5_1 --dry-run \
  --manifest semantic-sam-q5_1-v2.map.json
```

Expected output sizes before GGUF alignment/metadata:

- LayerDiff3D: about 3.60 GiB
- Marigold3D: about 1.66 GiB
- SemanticSAM: about 474.25 MiB

## Actual conversion

```bash
python convert_seethrough_to_gguf_v2.py layerdiff3d \
  --type q5_1 \
  --conditioning cached \
  --no-embed-tokenizer \
  --gguf-py ./llama.cpp/gguf-py \
  -o layerdiff3d-Q5_1.gguf

python convert_seethrough_to_gguf_v2.py layerdiff3d \
  --type q4_k_m --conditioning cached --no-embed-tokenizer \
  --ggml-quant-lib /path/to/stable-diffusion.cpp/build/bin/libseethrough-ggml-quant.so \
  -o layerdiff3d-Q4_K_M.gguf

# Use --type q5_k_m and layerdiff3d-Q5_K_M.gguf for the larger mixed variant.

python convert_seethrough_to_gguf_v2.py marigold \
  --type q5_1 \
  --conditioning cached \
  --no-embed-tokenizer \
  --gguf-py ./llama.cpp/gguf-py \
  -o marigold3d-Q5_1.gguf

python convert_seethrough_to_gguf_v2.py semantic-sam \
  --type q5_1 \
  --gguf-py ./llama.cpp/gguf-py \
  -o semantic-sam-Q5_1.gguf
```

If conversion is run in a CUDA-enabled See-through environment, cached LayerDiff/Marigold conditioning is computed in BF16 by default. To force it explicitly:

```bash
--conditioning-device cuda --conditioning-compute-dtype bf16
```

If the active PyTorch build is CPU-only, the default is CPU FP32 conditioning computation with BF16 GGUF storage. This is only a one-time conversion step.

## Quantization policy

The converter intentionally quantizes only large, block-aligned rank-2 learned matrices. Convolutions, normalization parameters, embeddings, biases, positional tensors, and other non-matrix/small tensors remain in the fallback precision (F16 by default).

This is a conservative first implementation intended to establish numerical parity before tuning model-specific quantization.

`q4_k_m` and `q5_k_m` use Q6_K for attention value and feed-forward output matrices.
Other eligible matrices use Q4_K or Q5_K; widths not divisible by 256 remain in fallback precision.

## Runtime boundary

These GGUFs use custom architecture identifiers. They are valid GGUF model containers but stock llama.cpp/stable-diffusion.cpp does not yet execute the See-through custom graphs. Native runtime support still needs implementations for:

- `UNetFrameConditionModel`
- `CrossFrameTransformerBlock`
- LayerDiff group embeddings and two group/tag sets
- cached LayerDiff conditioning lookup
- TransparentVAE decoder
- Marigold 3D conditioning + depth pipeline
- SemanticSAM ViT-H/HQ + 19-class semantic decoder

The cached-conditioning v2 format intentionally removes the requirement to implement CLIP/tokenization in that runtime for the See-through inference path.
