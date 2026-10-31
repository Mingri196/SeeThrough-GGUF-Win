#!/usr/bin/env python3
import gguf
import numpy as np
import torch

from annotators.animeinsseg.skytnt_segmentation import resolve_segmentation_resolution
from utils.ggml_runtime import (
    GGMLTensor, adaptive_weight_budget, demote_promoted_weights, promote_streamed_weights,
)
from utils.inference_utils import (
    resolve_depth_resolution, resolve_layerdiff_memory_options, resolve_target_parts,
    use_adaptive_weight_residency,
)


gib = 1024**3
expected = {
    (6, 720): "resident",
    (6, 1024): "streamed",
    (6, 1280): "low-vram",
    (8, 1280): "streamed",
    (12, 1280): "balanced",
    (24, 1280): "resident",
}
for (memory, resolution), profile in expected.items():
    actual = resolve_layerdiff_memory_options(resolution, total_memory=memory * gib)[0]
    assert actual == profile, (memory, resolution, actual, profile)

q8_size = 5_030_389_216
q8_expected = {720: "resident", 896: "balanced", 1024: "balanced", 1280: "streamed"}
for resolution, profile in q8_expected.items():
    actual = resolve_layerdiff_memory_options(
        resolution, total_memory=8 * gib, model_size=q8_size,
    )[0]
    assert actual == profile, ("Q8", resolution, actual, profile)

assert adaptive_weight_budget(6 * gib, 5 * gib, gib, reserve_bytes=gib // 2) == gib // 2
assert adaptive_weight_budget(6 * gib, 5.75 * gib, gib, reserve_bytes=gib // 2) == 0
image = np.empty((1536, 1024, 3), dtype=np.uint8)
assert resolve_segmentation_resolution(image, 'auto', 6 * gib, 5 * gib) == 1536
large_image = np.empty((4096, 4096, 3), dtype=np.uint8)
assert resolve_segmentation_resolution(large_image, 'auto', 6 * gib, 5 * gib) == 2560
assert resolve_segmentation_resolution(image, 1280) == 1280
square = (1280, 1280, 4)
assert resolve_depth_resolution('auto', square, 6 * gib, 5 * gib) == [768, 768]
assert resolve_depth_resolution('auto', square, 8 * gib, 7 * gib) == [1024, 1024]
assert resolve_depth_resolution('auto', square, 12 * gib, 9 * gib) == [1280, 1280]
assert resolve_depth_resolution(-1, square) == [1280, 1280]
assert use_adaptive_weight_residency('streamed', True, True, 30)
assert use_adaptive_weight_residency('low-vram', True, True, 8)
assert not use_adaptive_weight_residency('low-vram', True, True, 2)
assert resolve_target_parts(['face', 'hair']) == (
    ['front hair', 'back hair'], ['face'],
)
try:
    resolve_target_parts(['not-a-part'])
except ValueError:
    pass
else:
    raise AssertionError('unknown part accepted')

if torch.cuda.is_available():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    for module, size in zip(model, (1024, 2048)):
        raw = torch.zeros(size, dtype=torch.uint8).pin_memory()
        module._ggml_weight = GGMLTensor(raw, gguf.GGMLQuantizationType.Q4_K, (4, 4))
    assert promote_streamed_weights(model, 2048) == 2048
    assert model[1]._ggml_weight.is_cuda and not model[0]._ggml_weight.is_cuda
    assert demote_promoted_weights(model) == 2048
    assert model[1]._ggml_weight.device.type == 'cpu'
print("memory profile selection OK")
