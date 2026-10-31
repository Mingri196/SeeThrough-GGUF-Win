"""On-the-fly GGUF dequantization for the see-through torch pipeline.

Keeps quantized weights (Q4_K/Q5_K/Q6_K/Q8_0/...) resident in VRAM at their
native ~4-8 bit footprint and dequantizes each weight transiently, on-device,
inside the module forward. This is the see-through-standalone port of
city96/ComfyUI-GGUF (Apache-2.0): the block dequant kernels are his, adapted to
run without any ComfyUI dependency.

Compared to gguf_loader's default path (upfront ``gguf.dequantize -> fp16`` for
every tensor), this makes a Q4 GGUF actually occupy Q4 space in VRAM/RAM. Only
large matmul/conv weights are kept quantized; tiny norm/bias tensors stay fp16.
"""
import os
import types

import gguf
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Dequant kernels (on-device torch), ported from city96/ComfyUI-GGUF dequant.py
# --------------------------------------------------------------------------- #
TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)


def is_torch_compatible(tensor):
    return tensor is None or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES


def is_quantized(tensor):
    return not is_torch_compatible(tensor)


def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)
    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)
    if qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        try:
            return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(dtype)
        except torch.OutOfMemoryError:
            raise
        except Exception:
            if dequantize_functions[qtype] is eager_dequantize_functions[qtype]:
                raise
            print(f"Compiled {qtype.name} dequantization failed; using eager PyTorch")
            dequantize_functions[qtype] = eager_dequantize_functions[qtype]
            return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(dtype)
    # numpy fallback for any qtype without a torch kernel here
    new = gguf.quants.dequantize(tensor.data.cpu().numpy(), qtype)
    return torch.from_numpy(new).to(tensor.device, dtype=dtype)


def dequantize(data, qtype, oshape, dtype=None):
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_functions[qtype](blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs = ql | (qh << 4)
    return (d * qs) + m


def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return (d * qs) + m


def dequantize_blocks_Q8_0K_noop(*a, **k):  # placeholder to keep table tidy
    raise NotImplementedError


QK_K = 256
K_SCALE_SIZE = 12


def _to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8).reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8))


def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(list(range(8)), device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)
    return (d * q - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))


eager_dequantize_functions = {
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
}
dequantize_functions = eager_dequantize_functions.copy()
compiled_dequantize_functions = {}

if os.environ.get("SEETHROUGH_COMPILE_DEQUANT", "1") != "0" and hasattr(torch, "compile"):
    for qtype in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K):
        compiled_dequantize_functions[qtype] = torch.compile(
            eager_dequantize_functions[qtype], dynamic=True, fullgraph=True, mode="default"
        )


def set_dequant_compile(enabled):
    for qtype, eager in eager_dequantize_functions.items():
        dequantize_functions[qtype] = compiled_dequantize_functions.get(qtype, eager) if enabled else eager


def adaptive_weight_budget(total_bytes, peak_bytes, free_bytes, reserve_bytes=None):
    """VRAM that can safely hold streamed quantized weights after a measured step."""
    reserve_bytes = reserve_bytes or max(768 * 1024**2, total_bytes // 8)
    return max(0, min(total_bytes - peak_bytes, free_bytes) - reserve_bytes)


def promote_streamed_weights(model, budget_bytes, device="cuda"):
    """Keep the largest streamed weights on GPU without exceeding *budget_bytes*."""
    candidates = [
        module for module in model.modules()
        if hasattr(module, "_ggml_weight") and module._ggml_weight.device.type == "cpu"
    ]
    candidates.sort(
        key=lambda module: module._ggml_weight.numel() * module._ggml_weight.element_size(),
        reverse=True,
    )
    promoted = 0
    for module in candidates:
        weight = module._ggml_weight
        size = weight.numel() * weight.element_size()
        if size > budget_bytes - promoted:
            continue
        try:
            module._ggml_weight = weight.to(device, non_blocking=True)
        except torch.OutOfMemoryError:
            break
        module._st_promoted_ggml = True
        promoted += size
    return promoted


def demote_promoted_weights(model):
    """Return adaptively promoted weights to CPU after a tuned-step OOM."""
    demoted = 0
    for module in model.modules():
        if not getattr(module, "_st_promoted_ggml", False):
            continue
        weight = module._ggml_weight
        size = weight.numel() * weight.element_size()
        cpu_weight = weight.to("cpu")
        try:
            cpu_weight = cpu_weight.pin_memory()
            cpu_weight.tensor_type = weight.tensor_type
            cpu_weight.tensor_shape = weight.tensor_shape
        except RuntimeError:
            pass
        module._ggml_weight = cpu_weight
        module._st_promoted_ggml = False
        demoted += size
    return demoted


# --------------------------------------------------------------------------- #
# GGMLTensor + on-the-fly module forwards
# --------------------------------------------------------------------------- #
class GGMLTensor(torch.Tensor):
    """Holds raw quantized bytes plus the ggml qtype and logical shape."""

    @staticmethod
    def __new__(cls, data, tensor_type, tensor_shape):
        return super().__new__(cls, data)

    def __init__(self, data, tensor_type, tensor_shape):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", None)
        return new


def _stream_weight(module, input):
    """Return the quantized weight on the compute device. When it lives in RAM
    (streamed mode) this moves only the small Q4/Q6 block bytes for this module;
    the transient copy is freed when the forward returns."""
    qw = module._ggml_weight
    if qw.device != input.device:
        qw = qw.to(input.device, non_blocking=True)
    return qw


def _ggml_linear_forward(self, input):
    w = dequantize_tensor(_stream_weight(self, input), input.dtype)
    b = None if self.bias is None else self.bias.to(input.dtype)
    return F.linear(input, w, b)


def _ggml_conv2d_forward(self, input):
    w = dequantize_tensor(_stream_weight(self, input), input.dtype)
    b = None if self.bias is None else self.bias.to(input.dtype)
    return self._conv_forward(input, w, b)


def attach_ggml_weight(module, qtensor):
    """Replace a Linear/Conv2d module's weight with a quantized tensor and swap
    in a dequant-on-the-fly forward. The forward dequantizes on the tensor's
    current device; leave qtensor on cuda for a resident weight, or on (pinned)
    CPU for a streamed weight. Returns the quantized byte size."""
    module._ggml_weight = qtensor
    module.weight = None
    if isinstance(module, torch.nn.Linear):
        module.forward = types.MethodType(_ggml_linear_forward, module)
    elif isinstance(module, torch.nn.Conv2d):
        module.forward = types.MethodType(_ggml_conv2d_forward, module)
    else:
        raise TypeError(f"unsupported ggml module: {type(module).__name__}")
    return qtensor.numel() * qtensor.element_size()
