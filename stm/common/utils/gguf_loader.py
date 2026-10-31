import json
from pathlib import Path

import gguf
import numpy as np
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler

from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from modules.layerdiffuse.vae import TransparentVAE
from modules.marigold import MarigoldDepthPipeline
from modules.semanticsam import SemanticSam
from utils.ggml_runtime import GGMLTensor, attach_ggml_weight, dequantize_functions


def _config(reader, name):
    return json.loads(reader.fields[f"seethrough.config.{name}_json"].contents())


def _tensor(reader, name, dtype):
    tensor = next((tensor for tensor in reader.tensors if tensor.name == name), None)
    if tensor is None:
        raise KeyError(f"GGUF tensor not found: {name}")
    return torch.from_numpy(gguf.dequantize(tensor.data, tensor.tensor_type)).to(dtype=dtype)


def _load_state(reader, prefix, model, dtype):
    expected = set(model.state_dict())
    tensors = [tensor for tensor in reader.tensors if tensor.name.startswith(prefix)]
    actual = {tensor.name[len(prefix):] for tensor in tensors}
    if expected != actual:
        raise ValueError(
            f"{prefix[:-1]} schema mismatch: missing={sorted(expected - actual)[:5]}, "
            f"extra={sorted(actual - expected)[:5]}"
        )

    print(f"Loading {prefix[:-1]} GGUF...")
    for tensor in tensors:
        name = tensor.name[len(prefix):]
        value = torch.from_numpy(gguf.dequantize(tensor.data, tensor.tensor_type)).to(dtype=dtype)
        set_module_tensor_to_device(model, name, "cpu", value=value, dtype=dtype, clear_cache=False)
    print(f"  {prefix[:-1]}: {len(tensors)} tensors loaded")
    return model


def _load_state_quantized(reader, prefix, model, dtype, stream=False, stream_unquantized=False):
    """Load a component keeping big Linear/Conv weights quantized (dequantized on
    the fly per forward). Resident mode keeps the quantized bytes on cuda;
    streamed mode keeps them in (pinned) CPU RAM so only a tiny working set is on
    the GPU during the forward -- this is what lets 1280 fit on 6 GB. Everything
    else (norms/bias) -> dtype on CPU (moved to cuda later with the module)."""
    expected = set(model.state_dict())
    tensors = [t for t in reader.tensors if t.name.startswith(prefix)]
    actual = {t.name[len(prefix):] for t in tensors}
    if expected != actual:
        raise ValueError(
            f"{prefix[:-1]} schema mismatch: missing={sorted(expected - actual)[:5]}, "
            f"extra={sorted(actual - expected)[:5]}"
        )

    quant_bytes = 0
    streamed_f16_bytes = 0
    print(f"Loading {prefix[:-1]} GGUF (quantized)...")
    for t in tensors:
        name = t.name[len(prefix):]
        is_quant = t.tensor_type in dequantize_functions
        parent = None
        if name.endswith(".weight"):
            parent = model.get_submodule(name[: -len(".weight")])
        if is_quant and isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d)):
            logical = tuple(model.get_parameter(name).shape)
            raw = torch.from_numpy(np.ascontiguousarray(t.data).view(np.uint8).copy())
            raw = raw.pin_memory() if stream else raw.to("cuda")
            quant_bytes += attach_ggml_weight(parent, GGMLTensor(raw, t.tensor_type, logical))
        elif (
            stream_unquantized
            and t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32)
            and isinstance(parent, (torch.nn.Linear, torch.nn.Conv2d))
        ):
            logical = tuple(model.get_parameter(name).shape)
            value = torch.from_numpy(gguf.dequantize(t.data, t.tensor_type).copy()).to(dtype=dtype).pin_memory()
            streamed_f16_bytes += attach_ggml_weight(parent, GGMLTensor(value, None, logical))
        else:
            value = torch.from_numpy(gguf.dequantize(t.data, t.tensor_type)).to(dtype=dtype)
            set_module_tensor_to_device(model, name, "cpu", value=value, dtype=dtype, clear_cache=False)
    where = "pinned RAM (streamed)" if stream else "cuda"
    print(f"  {prefix[:-1]}: {quant_bytes / 1024**2:.0f} MiB quantized on {where}")
    if streamed_f16_bytes:
        print(f"  {prefix[:-1]}: {streamed_f16_bytes / 1024**2:.0f} MiB fp16 on pinned RAM (streamed)")
    return model


def _load_component(
    reader, prefix, model_class, config_name, dtype, quantized=False, stream=False,
    stream_unquantized=False,
):
    with init_empty_weights(include_buffers=True):
        model = model_class.from_config(_config(reader, config_name))
    if quantized:
        return _load_state_quantized(
            reader, prefix, model, dtype, stream=stream, stream_unquantized=stream_unquantized
        )
    return _load_state(reader, prefix, model, dtype)


def _inference_dtype():
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def load_layerdiff_gguf(path, dtype=None, quantized=False, stream=False, stream_unquantized=False):
    dtype = dtype or _inference_dtype()
    vae_dtype = torch.float32
    reader = gguf.GGUFReader(Path(path))
    if reader.fields["seethrough.family"].contents() != "layerdiff3d":
        raise ValueError(f"not a LayerDiff3D GGUF: {path}")

    unet = _load_component(
        reader, "unet.", UNetFrameConditionModel, "unet", dtype,
        quantized=quantized, stream=stream, stream_unquantized=stream_unquantized,
    )
    vae = _load_component(reader, "vae.", AutoencoderKL, "vae", vae_dtype, quantized=quantized, stream=stream)
    trans_vae = _load_component(reader, "trans_vae.", TransparentVAE, "trans_vae", vae_dtype)
    scheduler = DPMSolverMultistepScheduler.from_config(_config(reader, "scheduler"))
    pipeline = KDiffusionStableDiffusionXLPipeline(
        vae=vae,
        unet=unet,
        trans_vae=trans_vae,
        scheduler=scheduler,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
    )
    pipeline.vae.enable_tiling()

    body_tags = reader.fields["seethrough.conditioning.body_tags"].contents()
    head_tags = reader.fields["seethrough.conditioning.head_tags"].contents()
    body_prompt = _tensor(reader, "conditioning.body_prompt_embeds", dtype)
    body_pooled = _tensor(reader, "conditioning.body_pooled_prompt_embeds", dtype)
    head_prompt = _tensor(reader, "conditioning.head_prompt_embeds", dtype)
    head_pooled = _tensor(reader, "conditioning.head_pooled_prompt_embeds", dtype)
    pipeline._cached_prompt_embeds = {
        tag: [prompt[None], pooled[None]]
        for tags, prompts, pooled_values in (
            (body_tags, body_prompt, body_pooled),
            (head_tags, head_prompt, head_pooled),
        )
        for tag, prompt, pooled in zip(tags, prompts, pooled_values)
    }
    return pipeline


def load_marigold_gguf(path, dtype=None):
    dtype = dtype or _inference_dtype()
    reader = gguf.GGUFReader(Path(path))
    if reader.fields["seethrough.family"].contents() != "marigold3d":
        raise ValueError(f"not a Marigold3D GGUF: {path}")

    unet = _load_component(reader, "unet.", UNetFrameConditionModel, "unet", dtype)
    vae = _load_component(reader, "vae.", AutoencoderKL, "vae", dtype)
    scheduler = DDIMScheduler.from_config(_config(reader, "scheduler"))
    model_index = _config(reader, "model_index")
    pipeline = MarigoldDepthPipeline(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        text_encoder=None,
        tokenizer=None,
        scale_invariant=model_index["scale_invariant"],
        shift_invariant=model_index["shift_invariant"],
        default_denoising_steps=model_index["default_denoising_steps"],
        default_processing_resolution=model_index["default_processing_resolution"],
    )
    pipeline.vae.enable_tiling()
    pipeline.empty_text_embed = _tensor(reader, "conditioning.empty_prompt_embeds", dtype)
    return pipeline


def load_semantic_sam_gguf(path, dtype=None):
    dtype = dtype or _inference_dtype()
    reader = gguf.GGUFReader(Path(path))
    if reader.fields["seethrough.family"].contents() != "semantic_sam":
        raise ValueError(f"not a SemanticSAM GGUF: {path}")

    with init_empty_weights(include_buffers=True):
        model = SemanticSam(class_num=19, model_type="h_hq", mask_decoder="mlp_variant")
    return _load_state(reader, "semantic_sam.", model, dtype).eval()


def set_feed_forward_chunk(pipeline, chunk_size):
    for module in pipeline.unet.modules():
        if hasattr(module, "set_chunk_feed_forward"):
            module.set_chunk_feed_forward(chunk_size=chunk_size or None, dim=0)


def enable_streamed_gpu_compute(pipeline):
    set_feed_forward_chunk(pipeline, 1)
    block_streaming = torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3
    pipeline.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level" if block_streaming else "leaf_level",
        num_blocks_per_group=1 if block_streaming else None,
        non_blocking=block_streaming,
        use_stream=block_streaming,
        record_stream=False,
    )
