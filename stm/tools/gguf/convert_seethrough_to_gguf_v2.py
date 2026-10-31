#!/usr/bin/env python3
"""
Convert the three See-through model families to custom GGUF containers.

Architectures emitted:
  - seethrough_layerdiff3d
  - seethrough_marigold3d
  - seethrough_semantic_sam_h_hq

This script is intentionally a *model serializer/converter*. The emitted GGUFs use
custom architecture identifiers and exact state-dict-derived tensor names. They
will not execute in stock llama.cpp/stable-diffusion.cpp until corresponding
runtime/model-loader graph support is implemented.

The implementation is designed for large models:
  * safetensors headers are scanned without loading all tensors;
  * tensors are loaded/quantized one-at-a-time;
  * GGUF tensor-info is written in a first pass and tensor bytes in a second pass;
  * no complete model state_dict is materialized for the diffusion checkpoints.

Typical Arch/Linux setup:

  git clone https://github.com/ggml-org/llama.cpp.git
  python -m pip install -e ./llama.cpp/gguf-py
  python -m pip install safetensors huggingface_hub torch numpy tqdm

Examples:

  # Inspect/estimate first; downloads only files missing from your HF cache.
  python convert_seethrough_to_gguf.py layerdiff3d --type q5_1 --dry-run

  # Convert pinned canonical revisions found by the See-through probe.
  python convert_seethrough_to_gguf.py layerdiff3d --type q5_1 -o layerdiff3d-q5_1.gguf
  python convert_seethrough_to_gguf.py marigold    --type q5_1 -o marigold-q5_1.gguf
  python convert_seethrough_to_gguf.py semantic-sam --type q5_1 -o semantic-sam-q5_1.gguf

  # Use a local HF snapshot instead of downloading/resolving anything.
  python convert_seethrough_to_gguf.py layerdiff3d \
      --source ~/.cache/huggingface/hub/models--layerdifforg--seethroughv0.0.2_layerdiff3d/snapshots/966721... \
      --type f16 -o layerdiff3d-f16.gguf

Quantization policy:
  The diffusion/SAM graphs contain many conv, norm, embedding, position and small
  tensors for which generic row-wise GGML quants are either invalid or poor fits.
  Therefore q8_0/q5_*/q4_* are applied only to rank-2 learned matrix weights whose
  final PyTorch dimension is block-aligned. Everything else uses --fallback
  (f16 by default). This is deliberate and matches the conservative first-pass
  strategy needed before image/depth/mask quality validation.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence


FORMAT_VERSION = 2


@dataclasses.dataclass(frozen=True)
class ComponentSpec:
    name: str
    weight_file: str
    expected_tensors: int
    config_files: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    key: str
    architecture: str
    display_name: str
    repo_id: str
    revision: str
    expected_tensors: int
    components: tuple[ComponentSpec, ...]
    allow_patterns: tuple[str, ...]
    source_weight_sizes: Mapping[str, int]
    source_weight_sha256: Mapping[str, str]
    metadata: Mapping[str, Any]


MODEL_SPECS: dict[str, ModelSpec] = {
    "layerdiff3d": ModelSpec(
        key="layerdiff3d",
        architecture="seethrough_layerdiff3d",
        display_name="See-through LayerDiff 3D",
        repo_id="layerdifforg/seethroughv0.0.2_layerdiff3d",
        revision="966721bb4ef2ddc3af3696862fa10b3f78d9785d",
        expected_tensors=3969,
        components=(
            ComponentSpec("text_encoder", "text_encoder/model.safetensors", 196,
                          ("text_encoder/config.json", "tokenizer/tokenizer_config.json",
                           "tokenizer/special_tokens_map.json", "tokenizer/vocab.json", "tokenizer/merges.txt")),
            ComponentSpec("text_encoder_2", "text_encoder_2/model.safetensors", 517,
                          ("text_encoder_2/config.json", "tokenizer_2/tokenizer_config.json",
                           "tokenizer_2/special_tokens_map.json", "tokenizer_2/vocab.json", "tokenizer_2/merges.txt")),
            ComponentSpec("trans_vae", "trans_vae/diffusion_pytorch_model.safetensors", 556,
                          ("trans_vae/config.json",)),
            ComponentSpec("unet", "unet/diffusion_pytorch_model.safetensors", 2452,
                          ("unet/config.json",)),
            ComponentSpec("vae", "vae/diffusion_pytorch_model.safetensors", 248,
                          ("vae/config.json",)),
        ),
        allow_patterns=(
            "model_index.json", "scheduler/scheduler_config.json",
            "text_encoder/*", "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*",
            "trans_vae/*", "unet/*", "vae/*",
        ),
        source_weight_sizes={
            "text_encoder/model.safetensors": 246144352,
            "text_encoder_2/model.safetensors": 1389382688,
            "trans_vae/diffusion_pytorch_model.safetensors": 210639792,
            "unet/diffusion_pytorch_model.safetensors": 8142919672,
            "vae/diffusion_pytorch_model.safetensors": 167335590,
        },
        source_weight_sha256={
            "text_encoder/model.safetensors": "e4c36502d1f3fca20e6a8db3af2ab8213bb59178a5ff77aab086f0992cd0237d",
            "text_encoder_2/model.safetensors": "b361da6fb93514411949c779d3abd7684e19900a6e566eaf64ea5a3697b23a21",
            "trans_vae/diffusion_pytorch_model.safetensors": "925ffed4df7b2f57dbb549ac1ef374a26a1ead8d247eaf054dab641f82a8e0b7",
            "unet/diffusion_pytorch_model.safetensors": "05b5cd5ee2cbd7cd9c23008b7f76918cfc8e75aa0c45e7ce4cd1c741629b8f1d",
            "vae/diffusion_pytorch_model.safetensors": "ce4b137acad050c9072769cfb02fdc6ecb8dc0957586db5f6fab71d18103ccc1",
        },
        metadata={
            "family": "layerdiff3d",
            "unet_class": "UNetFrameConditionModel",
            "unet.sample_size": 128,
            "unet.in_channels": 8,
            "unet.out_channels": 4,
            "unet.cross_attention_dim": 2048,
            "unet.block_out_channels": [320, 640, 1280],
            "unet.down_block_types": ["DownBlock2D", "CrossFrameAttnDownBlock", "CrossFrameAttnDownBlock"],
            "unet.up_block_types": ["CrossFrameAttnUpBlock", "CrossFrameAttnUpBlock", "UpBlock2D"],
            "unet.transformer_layers_per_block": [1, 2, 10],
            "unet.attention_head_dim": [5, 10, 20],
            "unet.group_embedding": True,
            "unet.group_embedding_num": [13, 11],
            "unet.tag_version": "v3",
            "unet.addition_embed_type": "text_time",
            "unet.addition_time_embed_dim": 256,
            "unet.projection_class_embeddings_input_dim": 2816,
            "vae.scaling_factor": 0.13025,
            "vae.force_upcast": False,
            "conditioning.body_tags": [
                "front hair", "back hair", "head", "neck", "neckwear", "topwear", "handwear",
                "bottomwear", "legwear", "footwear", "tail", "wings", "objects",
            ],
            "conditioning.head_tags": [
                "headwear", "face", "irides", "eyebrow", "eyewhite", "eyelash", "eyewear",
                "ears", "earwear", "nose", "mouth",
            ],
        },
    ),
    "marigold": ModelSpec(
        key="marigold",
        architecture="seethrough_marigold3d",
        display_name="See-through Marigold 3D Depth",
        repo_id="24yearsold/seethroughv0.0.1_marigold",
        revision="aa7a892f83ff68d7b09186a405ba08d5d33f770f",
        expected_tensors=1610,
        components=(
            ComponentSpec("text_encoder", "text_encoder/model.safetensors", 372,
                          ("text_encoder/config.json", "tokenizer/tokenizer_config.json",
                           "tokenizer/special_tokens_map.json", "tokenizer/vocab.json", "tokenizer/merges.txt")),
            ComponentSpec("unet", "unet/diffusion_pytorch_model.safetensors", 990,
                          ("unet/config.json",)),
            ComponentSpec("vae", "vae/diffusion_pytorch_model.safetensors", 248,
                          ("vae/config.json",)),
        ),
        allow_patterns=(
            "model_index.json", "scheduler/scheduler_config.json",
            "text_encoder/*", "tokenizer/*", "unet/*", "vae/*",
        ),
        source_weight_sizes={
            "text_encoder/model.safetensors": 680820760,
            "unet/diffusion_pytorch_model.safetensors": 2426482432,
            "vae/diffusion_pytorch_model.safetensors": 167335590,
        },
        source_weight_sha256={
            "text_encoder/model.safetensors": "305b018799a2e1dba83e9db17654309814ff4212e40b8096b237ce2e3794a132",
            "unet/diffusion_pytorch_model.safetensors": "baf19abcc3b59af0e9a32b19781bf062aa99ced5d6f7d661902b59b0b4e1fb29",
            "vae/diffusion_pytorch_model.safetensors": "8d0c34f57abe50f323040f2366c8e22b941068dcdf53c8eb1d6fafb838afecb7",
        },
        metadata={
            "family": "marigold3d",
            "unet_class": "UNetFrameConditionModel",
            "unet.sample_size": 96,
            "unet.in_channels": 12,
            "unet.out_channels": 4,
            "unet.cross_attention_dim": 1024,
            "unet.block_out_channels": [320, 640, 1280, 1280],
            "unet.down_block_types": [
                "CrossFrameAttnDownBlock", "CrossFrameAttnDownBlock", "CrossFrameAttnDownBlock", "DownBlock2D"
            ],
            "unet.up_block_types": [
                "UpBlock2D", "CrossFrameAttnUpBlock", "CrossFrameAttnUpBlock", "CrossFrameAttnUpBlock"
            ],
            "unet.transformer_layers_per_block": 1,
            "unet.attention_head_dim": [5, 10, 20, 20],
            "vae.scaling_factor": 0.18215,
            "vae.force_upcast": True,
            "pipeline.default_denoising_steps": 4,
            "pipeline.default_processing_resolution": 768,
            "pipeline.scale_invariant": True,
            "pipeline.shift_invariant": True,
            "conditioning.empty_prompt": True,
        },
    ),
    "semantic-sam": ModelSpec(
        key="semantic-sam",
        architecture="seethrough_semantic_sam_h_hq",
        display_name="See-through SemanticSAM H-HQ 19-class",
        repo_id="24yearsold/l2d_sam_iter2",
        revision="91c07b2150c0971b439dd050d34b6641f08679dd",
        expected_tensors=817,
        components=(
            ComponentSpec("semantic_sam", "checkpoint-18000.pt", 817, ()),
        ),
        allow_patterns=("checkpoint-18000.pt",),
        source_weight_sizes={"checkpoint-18000.pt": 1294766917},
        source_weight_sha256={
            "checkpoint-18000.pt": "dda82b916f6c340bc5a6c3a077ff1214f9906906abe3707b1076d55b508f253f"
        },
        metadata={
            "family": "semantic_sam",
            "sam.model_type": "h_hq",
            "sam.hq": True,
            "sam.mask_decoder": "mlp_variant",
            "sam.class_count": 19,
            "sam.image_size": 1024,
            "sam.patch_size": 16,
            "sam.encoder_embed_dim": 1280,
            "sam.encoder_depth": 32,
            "sam.encoder_num_heads": 16,
            "sam.encoder_global_attention_indexes": [7, 15, 23, 31],
            "sam.window_size": 14,
            "sam.prompt_embed_dim": 256,
            "sam.image_embedding_size": 64,
            "sam.mask_in_chans": 16,
            "sam.two_way_transformer_depth": 2,
            "sam.two_way_transformer_mlp_dim": 2048,
            "sam.two_way_transformer_num_heads": 8,
            "sam.pixel_mean": [123.675, 116.28, 103.53],
            "sam.pixel_std": [58.395, 57.12, 57.375],
        },
    ),
}


# Exact counts from the probe; used to ensure we do not silently lose custom tensors.
# These are counts inside each component state_dict, not GGUF total aliases.
EXPECTED_COMPONENT_COUNTS: dict[str, dict[str, int]] = {
    "layerdiff3d": {
        "text_encoder": 196,
        "text_encoder_2": 517,
        "trans_vae": 556,
        "unet": 2452,
        "vae": 248,
    },
    "marigold": {
        "text_encoder": 372,
        "unet": 990,
        "vae": 248,
    },
    "semantic-sam": {"semantic_sam": 817},
}

# SHA256 of the sorted exact state-dict schema: name\tshape\tdtype\n.
# This validates names + shapes + source dtypes without embedding thousands of names here.
EXPECTED_SCHEMA_SHA256: dict[str, dict[str, str]] = {
    "layerdiff3d": {
        "text_encoder": "f53b294135ff6e69c46880ceec64825c08099b7a5af1f0f7962911ea488781aa",
        "text_encoder_2": "174c49bb5a07ab4e7ffa4e7ef2115c05da372df3dd217a7d8dc195eff5da64e8",
        "trans_vae": "390a3f5480d92f39cbc5e8a485557f09639a7b7a54be73c0b54ee8e42d993d69",
        "unet": "03a4d0ffaf23df8ab1f074dc1349e6037d4ea08ba3aa0c7b137c4fee48288b4c",
        "vae": "92223f1c140d05d69e92be9fed2dbd0f2bbd8f8c9dbd585889713042c4edd406",
    },
    "marigold": {
        "text_encoder": "7187804615867e2c3cbc6088ab96daec512a5200d87bf277512ff3de148ff3d0",
        "unet": "cbd98c137a4aae3c38c071c966d58edfafdc8d2e506a9789aaf2544b10874c91",
        "vae": "92223f1c140d05d69e92be9fed2dbd0f2bbd8f8c9dbd585889713042c4edd406",
    },
    "semantic-sam": {
        "semantic_sam": "fb5667f38244e6b9a8e8d34b571019419dc4213285f69f2969e2ee40e7428d1a",
    },
}


# Bytes per GGML block for the conservative quant types exposed here.
# gguf.quants is still used for the actual bit-exact quantization.
QUANT_BLOCK_LAYOUT: dict[str, tuple[int, int]] = {
    "q8_0": (32, 34),
    "q5_0": (32, 22),
    "q5_1": (32, 24),
    "q4_0": (32, 18),
    "q4_1": (32, 20),
    "q4_k": (256, 144),
    "q5_k": (256, 176),
    "q6_k": (256, 210),
}

MIXED_QUANT_BASE = {"q4_k_m": "q4_k", "q5_k_m": "q5_k"}
_GGML_QUANTIZE: Any | None = None


@dataclasses.dataclass
class TensorRecord:
    component: str
    source_file: Path | None
    state_name: str
    gguf_name: str
    shape: tuple[int, ...]
    source_dtype: str
    output_kind: str
    output_nbytes: int
    numel: int


class ConversionError(RuntimeError):
    pass


def load_ggml_quantizer(path: str | None) -> None:
    global _GGML_QUANTIZE
    candidates = [path, os.environ.get("SEETHROUGH_GGML_QUANT_LIB")]
    candidates += [str(Path.cwd() / "build" / "bin" / "libseethrough-ggml-quant.so")]
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        func = ctypes.CDLL(str(Path(candidate).resolve())).seethrough_ggml_quantize
        func.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        func.restype = ctypes.c_size_t
        _GGML_QUANTIZE = func
        return
    raise ConversionError(
        "K-quant conversion needs libseethrough-ggml-quant.so; build target "
        "seethrough-ggml-quant and pass --ggml-quant-lib /path/to/the/library"
    )


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if abs(x) < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{n} B"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ConversionError(f"Not a safetensors file (short header): {path}")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 0 or header_len > 512 * 1024 * 1024:
            raise ConversionError(f"Invalid safetensors header length {header_len} in {path}")
        blob = f.read(header_len)
        if len(blob) != header_len:
            raise ConversionError(f"Truncated safetensors header in {path}")
    try:
        obj = json.loads(blob)
    except Exception as exc:
        raise ConversionError(f"Invalid safetensors JSON header in {path}: {exc}") from exc
    return {k: v for k, v in obj.items() if k != "__metadata__"}


def dtype_nbytes(dtype: str) -> int:
    m = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "U16": 2,
        "I16": 2,
        "F16": 2,
        "BF16": 2,
        "U32": 4,
        "I32": 4,
        "F32": 4,
        "U64": 8,
        "I64": 8,
        "F64": 8,
    }
    key = dtype.upper()
    if key not in m:
        raise ConversionError(f"Unsupported source dtype {dtype!r}")
    return m[key]


def is_embedding_like(name: str) -> bool:
    n = name.lower()
    bad = (
        "embed", "embedding", "position", "pos_embed", "rel_pos", "mask_tokens", "iou_token",
        "hf_token", "point_embeddings", "no_mask_embed", "not_a_point_embed", "params",
    )
    return any(x in n for x in bad)


def is_norm_like(name: str) -> bool:
    n = name.lower()
    return any(x in n for x in ("norm", "layer_norm", "ln_"))


def mixed_quant_kind(name: str, requested: str) -> str:
    base = MIXED_QUANT_BASE.get(requested, requested)
    if requested in MIXED_QUANT_BASE and name.endswith(("to_v.weight", "ff.net.2.weight")):
        return "q6_k"
    return base


def choose_output_kind(name: str, shape: Sequence[int], requested: str, fallback: str) -> str:
    requested = requested.lower()
    fallback = fallback.lower()
    if requested in ("f16", "bf16", "f32"):
        return requested
    if requested not in QUANT_BLOCK_LAYOUT and requested not in MIXED_QUANT_BASE:
        raise ConversionError(f"Unknown output type: {requested}")

    # Conservative first-pass quantization policy:
    # learned rank-2 matrices only, excluding embeddings/positions/norm-like tensors.
    quant_kind = mixed_quant_kind(name, requested)
    block_size, _ = QUANT_BLOCK_LAYOUT[quant_kind]
    if (
        len(shape) == 2
        and int(shape[-1]) % block_size == 0
        and name.endswith(".weight")
        and not is_embedding_like(name)
        and not is_norm_like(name)
    ):
        return quant_kind
    return fallback


def output_nbytes(shape: Sequence[int], kind: str) -> int:
    numel = math.prod(int(x) for x in shape)
    k = kind.lower()
    if k in ("f16", "bf16"):
        return numel * 2
    if k == "f32":
        return numel * 4
    if k in QUANT_BLOCK_LAYOUT:
        block, bytes_per_block = QUANT_BLOCK_LAYOUT[k]
        if not shape or int(shape[-1]) % block != 0:
            raise ConversionError(f"Shape {tuple(shape)} cannot be row-quantized as {kind}")
        if numel % block != 0:
            raise ConversionError(f"numel={numel} is not divisible by block={block} for {kind}")
        return (numel // block) * bytes_per_block
    raise ConversionError(f"Unsupported output kind {kind}")


def gguf_tensor_name(component: str, state_name: str) -> str:
    # Keep the original exact state_dict name visible and reversible.
    return f"{component}.{state_name}"


def normalize_schema_dtype(dtype: str) -> str:
    s = str(dtype).upper().replace("TORCH.", "")
    return s.replace("BFLOAT16", "BF16").replace("FLOAT16", "F16").replace("FLOAT32", "F32")


def component_schema_sha256(records: Sequence[TensorRecord]) -> str:
    h = hashlib.sha256()
    for r in sorted(records, key=lambda x: x.state_name):
        shape = ",".join(str(int(x)) for x in r.shape)
        dtype = normalize_schema_dtype(r.source_dtype)
        h.update(f"{r.state_name}\t{shape}\t{dtype}\n".encode("utf-8"))
    return h.hexdigest()


def resolve_source(spec: ModelSpec, source: str | None, revision: str | None,
                   local_files_only: bool) -> tuple[Path, str]:
    if source:
        p = Path(source).expanduser()
        if p.exists():
            if p.is_file():
                if spec.key != "semantic-sam":
                    raise ConversionError("--source may be a file only for semantic-sam")
                return p.parent.resolve(), "local-file"
            return p.resolve(), "local-directory"
        repo_id = source
    else:
        repo_id = spec.repo_id

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise ConversionError(
            "huggingface_hub is required when --source is a HF repo id. "
            "Install it with: python -m pip install huggingface_hub"
        ) from exc

    rev = revision or spec.revision
    eprint(f"Resolving {repo_id}@{rev} via Hugging Face cache...")
    path = snapshot_download(
        repo_id=repo_id,
        revision=rev,
        allow_patterns=list(spec.allow_patterns),
        local_files_only=local_files_only,
    )
    return Path(path).resolve(), rev


def locate_component_file(root: Path, spec: ModelSpec, component: ComponentSpec,
                          original_source: str | None) -> Path:
    p = root / component.weight_file
    if p.exists():
        return p
    # For semantic-sam, allow --source /exact/checkpoint.pt.
    if spec.key == "semantic-sam" and original_source:
        direct = Path(original_source).expanduser()
        if direct.is_file() and direct.exists():
            return direct.resolve()
    raise ConversionError(f"Missing weight file: {p}")


def verify_source_files(root: Path, spec: ModelSpec, original_source: str | None,
                        verify_sha256: bool) -> None:
    for component in spec.components:
        p = locate_component_file(root, spec, component, original_source)
        rel = component.weight_file
        expected_size = spec.source_weight_sizes.get(rel)
        if expected_size is not None and p.stat().st_size != expected_size:
            raise ConversionError(
                f"Unexpected source size for {rel}: got {p.stat().st_size}, expected {expected_size}. "
                "Use the pinned canonical revision or explicitly update the converter spec."
            )
        if verify_sha256:
            expected_hash = spec.source_weight_sha256.get(rel)
            if expected_hash:
                eprint(f"sha256 {rel} ...")
                got = sha256_file(p)
                if got != expected_hash:
                    raise ConversionError(f"SHA256 mismatch for {rel}: {got} != {expected_hash}")


def scan_safetensor_component(component: ComponentSpec, path: Path,
                              requested: str, fallback: str) -> list[TensorRecord]:
    header = read_safetensors_header(path)
    records: list[TensorRecord] = []
    for name in sorted(header):
        info = header[name]
        if not isinstance(info, dict) or "shape" not in info or "dtype" not in info:
            continue
        shape = tuple(int(x) for x in info["shape"])
        dtype = str(info["dtype"])
        numel = math.prod(shape)
        kind = choose_output_kind(name, shape, requested, fallback)
        records.append(TensorRecord(
            component=component.name,
            source_file=path,
            state_name=name,
            gguf_name=gguf_tensor_name(component.name, name),
            shape=shape,
            source_dtype=dtype,
            output_kind=kind,
            output_nbytes=output_nbytes(shape, kind),
            numel=numel,
        ))
    return records


def load_semantic_sam_state(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise ConversionError("PyTorch is required for SemanticSAM checkpoint conversion") from exc
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    # mmap + weights_only are supported by the probe environment/current PyTorch.
    kwargs["weights_only"] = True
    kwargs["mmap"] = True
    try:
        state = torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        state = torch.load(path, **kwargs)
    if not isinstance(state, Mapping):
        raise ConversionError(f"SemanticSAM checkpoint root is {type(state).__name__}, expected mapping")
    if not all(hasattr(v, "shape") for v in state.values()):
        raise ConversionError("SemanticSAM checkpoint is not the expected direct 817-tensor state_dict")
    return state


def torch_dtype_name(dtype: Any) -> str:
    s = str(dtype)
    if s.startswith("torch."):
        s = s[6:]
    return s.upper().replace("FLOAT16", "F16").replace("FLOAT32", "F32").replace("BFLOAT16", "BF16")


def scan_semantic_sam(component: ComponentSpec, path: Path, state: Mapping[str, Any],
                      requested: str, fallback: str) -> list[TensorRecord]:
    records: list[TensorRecord] = []
    for name in sorted(state):
        t = state[name]
        shape = tuple(int(x) for x in t.shape)
        numel = math.prod(shape)
        kind = choose_output_kind(name, shape, requested, fallback)
        records.append(TensorRecord(
            component=component.name,
            source_file=path,
            state_name=name,
            gguf_name=gguf_tensor_name(component.name, name),
            shape=shape,
            source_dtype=torch_dtype_name(t.dtype),
            output_kind=kind,
            output_nbytes=output_nbytes(shape, kind),
            numel=numel,
        ))
    return records


def scan_records(root: Path, spec: ModelSpec, original_source: str | None,
                 requested: str, fallback: str) -> tuple[list[TensorRecord], Mapping[str, Any] | None]:
    records: list[TensorRecord] = []
    sam_state: Mapping[str, Any] | None = None
    for component in spec.components:
        path = locate_component_file(root, spec, component, original_source)
        if path.suffix == ".safetensors":
            rs = scan_safetensor_component(component, path, requested, fallback)
        elif path.suffix in (".pt", ".pth", ".ckpt"):
            if spec.key != "semantic-sam":
                raise ConversionError(f"Unsupported .pt component for {spec.key}: {path}")
            sam_state = load_semantic_sam_state(path)
            rs = scan_semantic_sam(component, path, sam_state, requested, fallback)
        else:
            raise ConversionError(f"Unsupported weight format: {path}")
        expected = EXPECTED_COMPONENT_COUNTS[spec.key][component.name]
        if len(rs) != expected:
            raise ConversionError(
                f"{spec.key}/{component.name}: found {len(rs)} tensors, expected {expected}. "
                "Refusing to silently convert an unknown revision."
            )
        got_schema = component_schema_sha256(rs)
        expected_schema = EXPECTED_SCHEMA_SHA256[spec.key][component.name]
        if got_schema != expected_schema:
            raise ConversionError(
                f"{spec.key}/{component.name}: state-dict schema hash mismatch. "
                f"got {got_schema}, expected {expected_schema}. Names/shapes/dtypes differ from the probed model."
            )
        records.extend(rs)

    if len(records) != spec.expected_tensors:
        raise ConversionError(
            f"{spec.key}: found {len(records)} total tensors, expected {spec.expected_tensors}"
        )

    # GGUF names must be globally unique.
    names = [r.gguf_name for r in records]
    if len(names) != len(set(names)):
        raise ConversionError("Duplicate GGUF tensor names generated")
    records.sort(key=lambda r: r.gguf_name)
    return records, sam_state



def make_synthetic_record(name: str, shape: Sequence[int], storage_kind: str) -> TensorRecord:
    shape_t = tuple(int(x) for x in shape)
    return TensorRecord(
        component="conditioning",
        source_file=None,
        state_name=name,
        gguf_name=f"conditioning.{name}",
        shape=shape_t,
        source_dtype=storage_kind.upper(),
        output_kind=storage_kind,
        output_nbytes=output_nbytes(shape_t, storage_kind),
        numel=math.prod(shape_t),
    )


def apply_conditioning_mode(spec: ModelSpec, source_records: Sequence[TensorRecord],
                            mode: str, storage_kind: str) -> list[TensorRecord]:
    """Build the actual GGUF tensor list after validating the complete source checkpoint.

    In cached mode the CLIP text encoders are deliberately omitted from LayerDiff/Marigold
    because See-through only uses a fixed conditioning set, caches those embeddings once,
    then deletes the text encoders. The corresponding embeddings are emitted as tiny
    synthetic GGUF tensors instead.
    """
    mode = mode.lower()
    if spec.key == "semantic-sam" or mode == "full":
        return sorted(list(source_records), key=lambda r: r.gguf_name)
    if mode != "cached":
        raise ConversionError(f"Unknown conditioning mode {mode!r}")
    if storage_kind not in ("bf16", "f16", "f32"):
        raise ConversionError("--conditioning-storage must be bf16, f16, or f32")

    out = [r for r in source_records if r.component not in ("text_encoder", "text_encoder_2")]

    if spec.key == "layerdiff3d":
        # Exact shapes from the two CLIP encoders used by
        # KDiffusionStableDiffusionXLPipeline.encode_cropped_prompt_77tokens().
        # 768 + 1280 hidden channels = SDXL cross_attention_dim 2048.
        out.extend([
            make_synthetic_record("body_prompt_embeds", (13, 77, 2048), storage_kind),
            make_synthetic_record("body_pooled_prompt_embeds", (13, 1280), storage_kind),
            make_synthetic_record("head_prompt_embeds", (11, 77, 2048), storage_kind),
            make_synthetic_record("head_pooled_prompt_embeds", (11, 1280), storage_kind),
        ])
    elif spec.key == "marigold":
        # Empty CLIP prompt with padding="do_not_pad" is BOS+EOS => 2 tokens.
        out.append(make_synthetic_record("empty_prompt_embeds", (1, 2, 1024), storage_kind))
    else:
        raise ConversionError(f"Cached conditioning is not defined for {spec.key}")

    names = [r.gguf_name for r in out]
    if len(names) != len(set(names)):
        raise ConversionError("Duplicate tensor name after conditioning transformation")
    return sorted(out, key=lambda r: r.gguf_name)


def _conditioning_device(torch: Any, requested: str) -> Any:
    requested = requested.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ConversionError(
            f"--conditioning-device={requested} requested but this PyTorch build has no CUDA. "
            "Use --conditioning-device cpu or activate your CUDA PyTorch environment."
        )
    try:
        return torch.device(requested)
    except Exception as exc:
        raise ConversionError(f"Invalid --conditioning-device {requested!r}") from exc


def _conditioning_compute_dtype(torch: Any, requested: str, device: Any) -> Any:
    requested = requested.lower()
    if requested == "auto":
        # Match See-through's BF16 inference path when a CUDA device is available.
        # FP32 is the compatibility default on CPU; outputs are cast to the chosen
        # GGUF conditioning storage type afterwards.
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if requested == "bf16":
        return torch.bfloat16
    if requested == "f16":
        return torch.float16
    if requested == "f32":
        return torch.float32
    raise ConversionError("--conditioning-compute-dtype must be auto, bf16, f16, or f32")


def _storage_torch_dtype(torch: Any, kind: str) -> Any:
    return {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[kind]


def _load_clip_tokenizer(path: Path) -> Any:
    try:
        from transformers import CLIPTokenizer
    except Exception as exc:
        raise ConversionError(
            "transformers is required to precompute cached conditioning. "
            "Install it in the See-through environment: python -m pip install transformers"
        ) from exc
    return CLIPTokenizer.from_pretrained(str(path), local_files_only=True)


def _load_clip_model(path: Path, with_projection: bool, dtype: Any, device: Any) -> Any:
    try:
        from transformers import CLIPTextModel, CLIPTextModelWithProjection
    except Exception as exc:
        raise ConversionError(
            "transformers is required to precompute cached conditioning. "
            "Install it in the See-through environment: python -m pip install transformers"
        ) from exc
    cls = CLIPTextModelWithProjection if with_projection else CLIPTextModel
    # Current and older transformers releases accept torch_dtype. Keeping this explicit
    # prevents an accidental FP32 load of the multi-GB SDXL text encoder.
    try:
        model = cls.from_pretrained(
            str(path),
            local_files_only=True,
            torch_dtype=dtype,
        )
    except TypeError:
        # Newer Transformers releases also expose the argument as `dtype`.
        model = cls.from_pretrained(
            str(path),
            local_files_only=True,
            dtype=dtype,
        )
    model.eval()
    model.to(device)
    return model


def _clip_hidden_penultimate(model: Any, input_ids: Any) -> tuple[Any, Any]:
    """Reproduce the project's tuple indexing exactly.

    Returns (penultimate_hidden_state, tuple_item_0). For CLIPTextModelWithProjection
    tuple_item_0 is the projected pooled text embedding used by See-through SDXL.
    """
    out = model(input_ids, output_hidden_states=True, return_dict=False)
    if len(out) < 2:
        raise ConversionError("Unexpected CLIP output tuple")
    hidden_states = out[-1]
    if not isinstance(hidden_states, (tuple, list)) or len(hidden_states) < 2:
        raise ConversionError("CLIP output does not contain hidden_states in the expected position")
    return hidden_states[-2], out[0]


def build_cached_conditioning(root: Path, spec: ModelSpec, mode: str, storage_kind: str,
                              device_arg: str, compute_dtype_arg: str) -> dict[str, Any]:
    if spec.key == "semantic-sam" or mode == "full":
        return {}
    try:
        import torch
    except Exception as exc:
        raise ConversionError("PyTorch is required to precompute cached conditioning") from exc

    device = _conditioning_device(torch, device_arg)
    compute_dtype = _conditioning_compute_dtype(torch, compute_dtype_arg, device)
    storage_dtype = _storage_torch_dtype(torch, storage_kind)

    eprint(
        f"Precomputing fixed conditioning on {device} "
        f"(compute={str(compute_dtype).replace('torch.', '')}, storage={storage_kind}) ..."
    )

    generated: dict[str, Any] = {}
    if spec.key == "layerdiff3d":
        body_tags = list(spec.metadata["conditioning.body_tags"])
        head_tags = list(spec.metadata["conditioning.head_tags"])
        prompts = body_tags + head_tags

        tok1 = _load_clip_tokenizer(root / "tokenizer")
        enc1 = _load_clip_model(root / "text_encoder", False, compute_dtype, device)
        with torch.inference_mode():
            ids1 = tok1(
                prompts,
                padding="max_length",
                max_length=tok1.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            h1, _ = _clip_hidden_penultimate(enc1, ids1)
            h1 = h1.detach().to(device="cpu", dtype=storage_dtype).contiguous()
        del enc1, ids1, tok1
        if device.type == "cuda":
            torch.cuda.empty_cache()

        tok2 = _load_clip_tokenizer(root / "tokenizer_2")
        enc2 = _load_clip_model(root / "text_encoder_2", True, compute_dtype, device)
        with torch.inference_mode():
            ids2 = tok2(
                prompts,
                padding="max_length",
                max_length=tok2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            h2, pooled = _clip_hidden_penultimate(enc2, ids2)
            h2 = h2.detach().to(device="cpu", dtype=storage_dtype).contiguous()
            pooled = pooled.detach().to(device="cpu", dtype=storage_dtype).contiguous()
        del enc2, ids2, tok2
        if device.type == "cuda":
            torch.cuda.empty_cache()

        prompt = torch.cat([h1, h2], dim=-1).contiguous()
        if tuple(prompt.shape) != (24, 77, 2048):
            raise ConversionError(f"LayerDiff cached prompt shape mismatch: {tuple(prompt.shape)}")
        if tuple(pooled.shape) != (24, 1280):
            raise ConversionError(f"LayerDiff cached pooled shape mismatch: {tuple(pooled.shape)}")

        generated["conditioning.body_prompt_embeds"] = prompt[:13].contiguous()
        generated["conditioning.body_pooled_prompt_embeds"] = pooled[:13].contiguous()
        generated["conditioning.head_prompt_embeds"] = prompt[13:].contiguous()
        generated["conditioning.head_pooled_prompt_embeds"] = pooled[13:].contiguous()

    elif spec.key == "marigold":
        tok = _load_clip_tokenizer(root / "tokenizer")
        enc = _load_clip_model(root / "text_encoder", False, compute_dtype, device)
        with torch.inference_mode():
            ids = tok(
                "",
                padding="do_not_pad",
                max_length=tok.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            out = enc(ids, return_dict=False)
            empty = out[0].detach().to(device="cpu", dtype=storage_dtype).contiguous()
        del enc, ids, tok
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if tuple(empty.shape) != (1, 2, 1024):
            raise ConversionError(
                f"Marigold empty prompt shape mismatch: {tuple(empty.shape)}; expected (1, 2, 1024)"
            )
        generated["conditioning.empty_prompt_embeds"] = empty
    else:
        raise ConversionError(f"Cached conditioning is not defined for {spec.key}")

    return generated


def validate_generated_tensors(records: Sequence[TensorRecord], generated: Mapping[str, Any]) -> None:
    expected = {r.gguf_name: r for r in records if r.source_file is None}
    if set(expected) != set(generated):
        raise ConversionError(
            "Generated conditioning tensor set mismatch: "
            f"missing={sorted(set(expected) - set(generated))}, "
            f"unexpected={sorted(set(generated) - set(expected))}"
        )
    for name, r in expected.items():
        t = generated[name]
        shape = tuple(int(x) for x in t.shape)
        if shape != r.shape:
            raise ConversionError(f"Generated {name} shape {shape} != planned {r.shape}")


def read_text_if_exists(root: Path, rel: str) -> str | None:
    p = root / rel
    if not p.exists() or not p.is_file():
        return None
    return p.read_text(encoding="utf-8")


def read_json_if_exists(root: Path, rel: str) -> Any | None:
    text = read_text_if_exists(root, rel)
    if text is None:
        return None
    return json.loads(text)


def add_json_metadata(writer: Any, key: str, obj: Any) -> None:
    writer.add_string(key, json.dumps(obj, separators=(",", ":"), ensure_ascii=False, sort_keys=True))


def add_custom_value(writer: Any, key: str, value: Any) -> None:
    full = f"seethrough.{key}"
    if isinstance(value, bool):
        writer.add_bool(full, value)
    elif isinstance(value, int):
        writer.add_int64(full, value)
    elif isinstance(value, float):
        writer.add_float64(full, value)
    elif isinstance(value, str):
        writer.add_string(full, value)
    elif isinstance(value, (list, tuple)):
        # GGUF arrays require homogeneous primitive element types; JSON avoids mixed/compat issues.
        if len(value) == 0:
            writer.add_string(full + ".json", "[]")
        elif all(isinstance(x, bool) for x in value):
            writer.add_array(full, list(value))
        elif all(isinstance(x, int) and not isinstance(x, bool) for x in value):
            writer.add_array(full, list(value))
        elif all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
            writer.add_array(full, [float(x) for x in value])
        elif all(isinstance(x, str) for x in value):
            writer.add_array(full, list(value))
        else:
            add_json_metadata(writer, full + ".json", value)
    elif value is None:
        writer.add_string(full + ".json", "null")
    else:
        add_json_metadata(writer, full + ".json", value)


def add_model_metadata(writer: Any, root: Path, spec: ModelSpec, resolved_revision: str,
                       requested: str, fallback: str, records: Sequence[TensorRecord],
                       embed_tokenizer: bool, conditioning_mode: str,
                       conditioning_storage: str) -> None:
    if hasattr(writer, "add_name"):
        writer.add_name(spec.display_name)
    else:
        writer.add_string("general.name", spec.display_name)
    if hasattr(writer, "add_description"):
        writer.add_description(
            "Custom GGUF serialization of the See-through model family. "
            "Requires See-through-specific GGML runtime graph support."
        )
    else:
        writer.add_string("general.description", "See-through custom GGUF model")
    if hasattr(writer, "add_source_repo_url"):
        writer.add_source_repo_url(f"https://huggingface.co/{spec.repo_id}")

    writer.add_uint32("seethrough.format_version", FORMAT_VERSION)
    writer.add_string("seethrough.model_key", spec.key)
    writer.add_string("seethrough.source.repo_id", spec.repo_id)
    writer.add_string("seethrough.source.revision", resolved_revision)
    writer.add_string("seethrough.tensor_name_scheme", "<component>.<exact_state_dict_name>")
    writer.add_uint32("seethrough.tensor_count", len(records))
    writer.add_string("seethrough.quantization.requested", requested)
    writer.add_string("seethrough.quantization.fallback", fallback)
    policy = "rank2_weight_matrix_mixed_k_v1" if requested in MIXED_QUANT_BASE else "rank2_weight_matrix_only_v1"
    writer.add_string("seethrough.quantization.policy", policy)
    writer.add_string("seethrough.conditioning.mode", conditioning_mode if spec.key != "semantic-sam" else "none")
    if spec.key != "semantic-sam":
        writer.add_string("seethrough.conditioning.storage", conditioning_storage)
        writer.add_bool("seethrough.conditioning.text_encoders_in_gguf", conditioning_mode == "full")

    for k, v in spec.metadata.items():
        add_custom_value(writer, k, v)

    # Preserve the original Diffusers/runtime configuration verbatim as JSON/text.
    for rel, key in (
        ("model_index.json", "seethrough.config.model_index_json"),
        ("scheduler/scheduler_config.json", "seethrough.config.scheduler_json"),
        ("unet/config.json", "seethrough.config.unet_json"),
        ("vae/config.json", "seethrough.config.vae_json"),
        ("trans_vae/config.json", "seethrough.config.trans_vae_json"),
        ("text_encoder/config.json", "seethrough.config.text_encoder_json"),
        ("text_encoder_2/config.json", "seethrough.config.text_encoder_2_json"),
    ):
        obj = read_json_if_exists(root, rel)
        if obj is not None:
            add_json_metadata(writer, key, obj)

    if embed_tokenizer:
        tok_roots = ["tokenizer"]
        if spec.key == "layerdiff3d":
            tok_roots.append("tokenizer_2")
        for i, tr in enumerate(tok_roots):
            for relname, suffix in (
                ("tokenizer_config.json", "config_json"),
                ("special_tokens_map.json", "special_tokens_json"),
                ("vocab.json", "vocab_json"),
                ("merges.txt", "merges"),
            ):
                txt = read_text_if_exists(root, f"{tr}/{relname}")
                if txt is not None:
                    writer.add_string(f"seethrough.tokenizer.{i}.{suffix}", txt)

    # Per-component counts make loader validation trivial.
    by_component: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    bytes_by_kind: dict[str, int] = {}
    for r in records:
        by_component[r.component] = by_component.get(r.component, 0) + 1
        by_kind[r.output_kind] = by_kind.get(r.output_kind, 0) + 1
        bytes_by_kind[r.output_kind] = bytes_by_kind.get(r.output_kind, 0) + r.output_nbytes
    add_json_metadata(writer, "seethrough.component_tensor_counts_json", by_component)
    add_json_metadata(writer, "seethrough.output_tensor_type_counts_json", by_kind)
    add_json_metadata(writer, "seethrough.output_tensor_type_bytes_json", bytes_by_kind)


def import_gguf(gguf_py: str | None) -> Any:
    candidates: list[Path] = []
    if gguf_py:
        candidates.append(Path(gguf_py).expanduser())
    env = os.environ.get("GGUF_PY")
    if env:
        candidates.append(Path(env).expanduser())
    llama_dir = os.environ.get("LLAMA_CPP_DIR")
    if llama_dir:
        candidates.append(Path(llama_dir).expanduser() / "gguf-py")
    candidates.extend([
        Path.cwd() / "llama.cpp" / "gguf-py",
        Path(__file__).resolve().parent / "llama.cpp" / "gguf-py",
    ])
    for p in candidates:
        if p.exists():
            sys.path.insert(0, str(p.resolve()))
            break
    try:
        import gguf  # type: ignore
    except Exception as exc:
        raise ConversionError(
            "Could not import official gguf-py. Install current llama.cpp gguf-py or pass --gguf-py.\n"
            "  git clone https://github.com/ggml-org/llama.cpp.git\n"
            "  python -m pip install -e ./llama.cpp/gguf-py\n"
            "or:\n"
            "  python convert_seethrough_to_gguf.py ... --gguf-py ./llama.cpp/gguf-py"
        ) from exc
    required = ["GGUFWriter", "GGMLQuantizationType", "quants"]
    missing = [x for x in required if not hasattr(gguf, x)]
    if missing:
        raise ConversionError(f"gguf-py is too old/incompatible; missing: {', '.join(missing)}")
    writer_methods = ("add_tensor_info", "write_header_to_file", "write_kv_data_to_file", "write_ti_data_to_file", "write_tensor_data")
    missing_methods = [x for x in writer_methods if not hasattr(gguf.GGUFWriter, x)]
    if missing_methods:
        raise ConversionError(
            "gguf-py is too old for streaming conversion; missing GGUFWriter methods: " + ", ".join(missing_methods)
        )
    return gguf


def ggml_qtype(gguf: Any, kind: str) -> Any:
    enum = gguf.GGMLQuantizationType
    mapping = {
        "bf16": "BF16",
        "q8_0": "Q8_0",
        "q5_0": "Q5_0",
        "q5_1": "Q5_1",
        "q4_0": "Q4_0",
        "q4_1": "Q4_1",
        "q4_k": "Q4_K",
        "q5_k": "Q5_K",
        "q6_k": "Q6_K",
    }
    if kind not in mapping:
        return None
    try:
        return getattr(enum, mapping[kind])
    except AttributeError as exc:
        raise ConversionError(f"Your gguf-py does not expose {mapping[kind]}") from exc


def register_tensor_infos(writer: Any, gguf: Any, records: Sequence[TensorRecord]) -> None:
    import numpy as np
    for r in records:
        if r.output_kind == "f16":
            writer.add_tensor_info(r.gguf_name, r.shape, np.dtype(np.float16), r.output_nbytes)
        elif r.output_kind == "f32":
            writer.add_tensor_info(r.gguf_name, r.shape, np.dtype(np.float32), r.output_nbytes)
        else:
            raw = ggml_qtype(gguf, r.output_kind)
            # Pass a non-u8 dtype so GGUFWriter treats r.shape as the original logical shape.
            writer.add_tensor_info(r.gguf_name, r.shape, np.dtype(np.float32), r.output_nbytes, raw_dtype=raw)


def torch_tensor_to_output(t: Any, kind: str, gguf: Any) -> Any:
    import numpy as np
    import torch

    t = t.detach().cpu().contiguous()
    if kind == "f16":
        return t.to(torch.float16).numpy()
    if kind == "f32":
        return t.to(torch.float32).numpy()

    # gguf.quants quantizers consume float32 arrays and produce raw uint8 block data.
    arr = t.to(torch.float32).numpy()
    qtype = ggml_qtype(gguf, kind)
    if kind in ("q4_k", "q5_k", "q6_k"):
        if _GGML_QUANTIZE is None:
            raise ConversionError("K-quant helper was not loaded")
        out = np.empty(output_nbytes(arr.shape, kind), dtype=np.uint8)
        written = _GGML_QUANTIZE(
            int(qtype), arr.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p),
            arr.size // arr.shape[-1], arr.shape[-1],
        )
        if written != out.nbytes:
            raise ConversionError(f"GGML wrote {written} bytes for {kind}, expected {out.nbytes}")
        return out
    try:
        out = gguf.quants.quantize(arr, qtype)
    except Exception as exc:
        raise ConversionError(f"Failed to quantize tensor shape={tuple(t.shape)} as {kind}: {exc}") from exc
    if out.dtype != np.uint8:
        out = out.view(np.uint8)
    return np.ascontiguousarray(out)


def iter_safetensor_outputs(records: Sequence[TensorRecord], gguf: Any) -> Iterator[tuple[TensorRecord, Any]]:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise ConversionError("safetensors is required: python -m pip install safetensors") from exc

    # Records are globally sorted by GGUF name; group by file while retaining that order.
    by_file: dict[Path, list[TensorRecord]] = {}
    for r in records:
        by_file.setdefault(r.source_file, []).append(r)
    for path in sorted(by_file, key=lambda p: str(p)):
        rs = sorted(by_file[path], key=lambda r: r.gguf_name)
        with safe_open(str(path), framework="pt", device="cpu") as f:
            file_keys = set(f.keys())
            for r in rs:
                if r.state_name not in file_keys:
                    raise ConversionError(f"Tensor disappeared between passes: {r.state_name} in {path}")
                t = f.get_tensor(r.state_name)
                if tuple(int(x) for x in t.shape) != r.shape:
                    raise ConversionError(f"Shape changed for {r.gguf_name}")
                out = torch_tensor_to_output(t, r.output_kind, gguf)
                if out.nbytes != r.output_nbytes:
                    raise ConversionError(
                        f"Internal size mismatch for {r.gguf_name}: got {out.nbytes}, expected {r.output_nbytes}"
                    )
                yield r, out


def iter_sam_outputs(records: Sequence[TensorRecord], state: Mapping[str, Any], gguf: Any) -> Iterator[tuple[TensorRecord, Any]]:
    by_name = {r.state_name: r for r in records}
    for r in sorted(records, key=lambda r: r.gguf_name):
        if r.state_name not in state:
            raise ConversionError(f"SAM tensor disappeared between passes: {r.state_name}")
        t = state[r.state_name]
        out = torch_tensor_to_output(t, r.output_kind, gguf)
        if out.nbytes != r.output_nbytes:
            raise ConversionError(
                f"Internal size mismatch for {r.gguf_name}: got {out.nbytes}, expected {r.output_nbytes}"
            )
        yield r, out


def output_iterator(spec: ModelSpec, records: Sequence[TensorRecord], sam_state: Mapping[str, Any] | None,
                    gguf: Any, generated_tensors: Mapping[str, Any] | None = None
                    ) -> Iterator[tuple[TensorRecord, Any]]:
    generated_tensors = generated_tensors or {}

    if spec.key == "semantic-sam":
        if sam_state is None:
            raise ConversionError("SemanticSAM state was not loaded")
        yield from iter_sam_outputs(records, sam_state, gguf)
        return

    # Walk in the exact registration order. Source tensors are opened lazily; synthetic
    # fixed-conditioning tensors are injected from generated_tensors.
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise ConversionError("safetensors is required: python -m pip install safetensors") from exc

    current_path: Path | None = None
    current_ctx: Any = None
    current_file: Any = None
    try:
        for r in records:
            if r.source_file is None:
                if r.gguf_name not in generated_tensors:
                    raise ConversionError(f"Missing generated tensor {r.gguf_name}")
                t = generated_tensors[r.gguf_name]
                if tuple(int(x) for x in t.shape) != r.shape:
                    raise ConversionError(f"Generated tensor shape changed for {r.gguf_name}")
                out = torch_tensor_to_output(t, r.output_kind, gguf)
                if out.nbytes != r.output_nbytes:
                    raise ConversionError(
                        f"Internal size mismatch for {r.gguf_name}: got {out.nbytes}, expected {r.output_nbytes}"
                    )
                yield r, out
                continue

            if current_path != r.source_file:
                if current_ctx is not None:
                    current_ctx.__exit__(None, None, None)
                current_path = r.source_file
                current_ctx = safe_open(str(current_path), framework="pt", device="cpu")
                current_file = current_ctx.__enter__()
            t = current_file.get_tensor(r.state_name)
            if tuple(int(x) for x in t.shape) != r.shape:
                raise ConversionError(f"Shape changed for {r.gguf_name}")
            out = torch_tensor_to_output(t, r.output_kind, gguf)
            if out.nbytes != r.output_nbytes:
                raise ConversionError(
                    f"Internal size mismatch for {r.gguf_name}: got {out.nbytes}, expected {r.output_nbytes}"
                )
            yield r, out
    finally:
        if current_ctx is not None:
            current_ctx.__exit__(None, None, None)


def print_plan(spec: ModelSpec, root: Path, records: Sequence[TensorRecord], requested: str, fallback: str) -> None:
    source_bytes = sum(r.numel * dtype_nbytes(r.source_dtype) for r in records)
    output_bytes = sum(r.output_nbytes for r in records)
    by_component: dict[str, tuple[int, int]] = {}
    by_kind: dict[str, tuple[int, int]] = {}
    for r in records:
        c_n, c_b = by_component.get(r.component, (0, 0))
        by_component[r.component] = (c_n + 1, c_b + r.output_nbytes)
        k_n, k_b = by_kind.get(r.output_kind, (0, 0))
        by_kind[r.output_kind] = (k_n + 1, k_b + r.output_nbytes)

    print(f"Model:        {spec.display_name}")
    print(f"Architecture: {spec.architecture}")
    print(f"Source:       {root}")
    print(f"Tensors:      {len(records)}")
    print(f"Source data:  {human_bytes(source_bytes)}")
    print(f"Output data:  {human_bytes(output_bytes)} (tensor bytes before GGUF alignment/metadata)")
    print(f"Requested:    {requested}; fallback: {fallback}")
    print("Components:")
    for k, (n, b) in sorted(by_component.items()):
        print(f"  {k:16s} {n:5d} tensors  {human_bytes(b):>12s}")
    print("Output tensor types:")
    for k, (n, b) in sorted(by_kind.items()):
        print(f"  {k:8s} {n:5d} tensors  {human_bytes(b):>12s}")


def write_manifest(path: Path, spec: ModelSpec, records: Sequence[TensorRecord],
                   resolved_revision: str, requested: str, fallback: str,
                   conditioning_mode: str, conditioning_storage: str,
                   source_tensor_count: int) -> None:
    obj = {
        "format_version": FORMAT_VERSION,
        "model_key": spec.key,
        "architecture": spec.architecture,
        "repo_id": spec.repo_id,
        "revision": resolved_revision,
        "requested_type": requested,
        "fallback_type": fallback,
        "conditioning_mode": conditioning_mode if spec.key != "semantic-sam" else "none",
        "conditioning_storage": conditioning_storage if spec.key != "semantic-sam" else None,
        "source_tensor_count": source_tensor_count,
        "tensor_count": len(records),
        "output_tensor_bytes": sum(r.output_nbytes for r in records),
        "tensors": [
            {
                "component": r.component,
                "source_file": str(r.source_file) if r.source_file is not None else None,
                "state_name": r.state_name,
                "gguf_name": r.gguf_name,
                "shape": list(r.shape),
                "source_dtype": r.source_dtype,
                "output_kind": r.output_kind,
                "output_nbytes": r.output_nbytes,
            }
            for r in records
        ],
    }
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def convert(args: argparse.Namespace) -> int:
    spec = MODEL_SPECS[args.model]
    if args.fallback not in ("f16", "bf16", "f32"):
        raise ConversionError("--fallback must be f16, bf16, or f32")
    if args.type in ("f16", "bf16", "f32") and args.fallback != "f16" and args.fallback != args.type:
        eprint("Note: --fallback is irrelevant for an unquantized --type")

    root, resolved_revision = resolve_source(spec, args.source, args.revision, args.local_files_only)
    verify_source_files(root, spec, args.source, args.verify_sha256)

    requested = args.type.lower()
    fallback = args.fallback.lower()

    if requested in MIXED_QUANT_BASE and not args.dry_run:
        load_ggml_quantizer(args.ggml_quant_lib)

    # Always validate the complete original checkpoint against the probe first.
    source_records, sam_state = scan_records(root, spec, args.source, requested, fallback)
    records = apply_conditioning_mode(
        spec, source_records, args.conditioning, args.conditioning_storage
    )

    print_plan(spec, root, records, requested, fallback)
    if spec.key != "semantic-sam":
        print(f"Conditioning: {args.conditioning}")
        if args.conditioning == "cached":
            removed = len(source_records) - len([r for r in source_records if r.component not in ("text_encoder", "text_encoder_2")])
            print(f"  CLIP tensors omitted from GGUF: {removed}")
            print(f"  Fixed conditioning tensors:     {sum(1 for r in records if r.component == 'conditioning')}")

    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        write_manifest(
            manifest_path, spec, records, resolved_revision, requested, fallback,
            args.conditioning, args.conditioning_storage, len(source_records)
        )
        print(f"Manifest:     {manifest_path}")

    if args.dry_run:
        return 0

    generated_tensors = build_cached_conditioning(
        root, spec, args.conditioning, args.conditioning_storage,
        args.conditioning_device, args.conditioning_compute_dtype
    )
    validate_generated_tensors(records, generated_tensors)

    gguf = import_gguf(args.gguf_py)
    out = Path(args.output or f"{spec.key}-{requested}.gguf").expanduser().resolve()
    if out.exists() and not args.overwrite:
        raise ConversionError(f"Output already exists: {out} (pass --overwrite to replace)")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        tmp.unlink()

    writer = gguf.GGUFWriter(tmp, spec.architecture)
    add_model_metadata(
        writer, root, spec, resolved_revision, requested, fallback, records,
        args.embed_tokenizer, args.conditioning, args.conditioning_storage
    )
    register_tensor_infos(writer, gguf, records)

    # Current gguf-py writer separates tensor-info and tensor data, letting us stream the huge model.
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    start = time.monotonic()
    done_bytes = 0
    total_bytes = sum(r.output_nbytes for r in records)
    try:
        for i, (record, data) in enumerate(
            output_iterator(spec, records, sam_state, gguf, generated_tensors), start=1
        ):
            writer.write_tensor_data(data)
            done_bytes += record.output_nbytes
            if args.verbose or i == 1 or i == len(records) or i % 25 == 0:
                elapsed = max(time.monotonic() - start, 1e-6)
                rate = done_bytes / elapsed
                pct = 100.0 * done_bytes / max(total_bytes, 1)
                eprint(
                    f"[{i:4d}/{len(records)}] {pct:6.2f}%  {human_bytes(done_bytes)}/{human_bytes(total_bytes)}  "
                    f"{human_bytes(int(rate))}/s  {record.output_kind:5s}  {record.gguf_name}"
                )
        writer.close()
    except BaseException:
        try:
            writer.close()
        finally:
            if tmp.exists():
                tmp.unlink()
        raise

    if out.exists():
        out.unlink()
    tmp.replace(out)
    print(f"GGUF:         {out}")
    print(f"File size:    {human_bytes(out.stat().st_size)}")
    print("Important: this uses a custom See-through GGUF architecture; stock runtimes need loader/graph support.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert See-through LayerDiff3D, Marigold3D, or SemanticSAM to custom GGUF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("model", choices=sorted(MODEL_SPECS))
    p.add_argument("--source", help="Local HF snapshot/checkpoint path or HF repo id. Defaults to canonical repo.")
    p.add_argument("--revision", help="HF revision. Defaults to the exact commit SHA found by the probe.")
    p.add_argument("-o", "--output", help="Output GGUF path")
    p.add_argument("--type", default="q5_1", choices=[
        "f32", "f16", "bf16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "q5_k_m", "q4_k_m",
    ])
    p.add_argument("--fallback", default="f16", choices=["f32", "f16", "bf16"],
                   help="Storage type for tensors that are not safely row-quantizable")
    p.add_argument("--conditioning", default="cached", choices=["cached", "full"],
                   help="For LayerDiff/Marigold: cache the fixed See-through CLIP conditioning and omit CLIP weights, or keep full text encoders")
    p.add_argument("--conditioning-storage", default="bf16", choices=["bf16", "f16", "f32"],
                   help="Storage type for precomputed fixed conditioning tensors")
    p.add_argument("--conditioning-device", default="auto",
                   help="Device used once to precompute fixed conditioning (auto, cpu, cuda, cuda:0, ...)")
    p.add_argument("--conditioning-compute-dtype", default="auto", choices=["auto", "bf16", "f16", "f32"],
                   help="Compute dtype for one-time CLIP conditioning generation; auto=bf16 on CUDA, f32 on CPU")
    p.add_argument("--gguf-py", help="Path to llama.cpp/gguf-py if it is not installed in this Python environment")
    p.add_argument("--ggml-quant-lib", help="Built libseethrough-ggml-quant.so required by Q4_K_M/Q5_K_M")
    p.add_argument("--local-files-only", action="store_true", help="Never access Hugging Face network; use cache only")
    p.add_argument("--verify-sha256", action="store_true", help="Hash the multi-GB source weight files and verify canonical LFS SHA256")
    p.add_argument("--dry-run", action="store_true", help="Validate exact tensors and estimate output, but do not write GGUF")
    p.add_argument("--manifest", help="Write a JSON manifest of exact source->GGUF tensor mapping")
    p.add_argument("--embed-tokenizer", action=argparse.BooleanOptionalAction, default=True,
                   help="Embed CLIP tokenizer metadata too (not needed at runtime with --conditioning cached)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return convert(args)
    except KeyboardInterrupt:
        eprint("Interrupted")
        return 130
    except ConversionError as exc:
        eprint(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
