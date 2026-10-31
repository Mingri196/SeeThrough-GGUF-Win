#!/usr/bin/env python3
"""Check that spatial frame chunking preserves Transformer3D and UNet output."""

from pathlib import Path
import sys

import torch
import gguf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from modules.layerdiffuse.transformer3d import Transformer3DModel
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from utils.inference_utils import resolve_layerdiff_memory_options
from utils.ggml_runtime import dequantize_blocks_Q4_K, dequantize_functions


def main() -> int:
    gib = 1024**3
    assert resolve_layerdiff_memory_options(1280, total_memory=6 * gib) == (
        "low-vram", [True, 1, 256, 1, True]
    )
    assert resolve_layerdiff_memory_options(1280, total_memory=8 * gib) == (
        "streamed", [True, 3, 512, 1, False]
    )
    assert resolve_layerdiff_memory_options(1280, total_memory=12 * gib) == (
        "balanced", [False, 3, 512, 1, False]
    )
    assert resolve_layerdiff_memory_options(1280, total_memory=24 * gib) == (
        "resident", [False, 0, 0, 0, False]
    )
    assert resolve_layerdiff_memory_options(512, total_memory=6 * gib)[0] == "resident"

    if torch.cuda.is_available():
        blocks = torch.randint(0, 256, (4, 144), dtype=torch.uint8, device="cuda")
        blocks[:, :4] = torch.tensor([0, 60, 0, 60], dtype=torch.uint8, device="cuda")
        expected = dequantize_blocks_Q4_K(blocks, 256, 144, torch.float16)
        actual = dequantize_functions[gguf.GGMLQuantizationType.Q4_K](
            blocks, 256, 144, torch.float16
        )
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=5e-5)

    torch.manual_seed(0)
    model = Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=8,
        num_layers=1,
        norm_num_groups=4,
        cross_attention_dim=8,
        use_linear_projection=True,
    ).eval()
    hidden_states = torch.randn(3, 8, 2, 2)
    encoder_hidden_states = torch.randn(3, 2, 8)

    with torch.inference_mode():
        expected = model(hidden_states, encoder_hidden_states=encoder_hidden_states, num_frames=3).sample
        model.set_spatial_frame_chunk(1)
        for module in model.modules():
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=1, dim=0)
            if hasattr(module, "set_temporal_token_chunk"):
                module.set_temporal_token_chunk(2)
        actual = model(hidden_states, encoder_hidden_states=encoder_hidden_states, num_frames=3).sample

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    transformer_max_abs = torch.max(torch.abs(actual - expected)).item()

    unet = UNetFrameConditionModel(
        sample_size=8,
        in_channels=8,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossFrameAttnDownBlock"),
        up_block_types=("CrossFrameAttnUpBlock", "UpBlock2D"),
        mid_block_type="UNetMidBlockCrossFrameAttn",
        block_out_channels=(8, 8),
        layers_per_block=1,
        cross_attention_dim=8,
        norm_num_groups=4,
        attention_head_dim=(4, 4),
        use_linear_projection=True,
    ).eval()
    sample = torch.randn(1, 3, 8, 8, 8)
    encoder_hidden_states = torch.randn(3, 2, 8)

    with torch.inference_mode():
        expected = unet(sample, torch.tensor(1), encoder_hidden_states).sample
        for module in unet.modules():
            if hasattr(module, "set_spatial_frame_chunk"):
                module.set_spatial_frame_chunk(1)
        actual = unet(sample, torch.tensor(1), encoder_hidden_states).sample

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    unet_max_abs = torch.max(torch.abs(actual - expected)).item()
    print(f"PASS transformer_max_abs={transformer_max_abs:.3g} unet_max_abs={unet_max_abs:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
