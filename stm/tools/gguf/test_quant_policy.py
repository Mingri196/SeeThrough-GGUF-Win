#!/usr/bin/env python3
from convert_seethrough_to_gguf_v2 import choose_output_kind, output_nbytes


def test_mixed_k_policy() -> None:
    assert choose_output_kind("attn.to_q.weight", (1280, 1280), "q4_k_m", "f16") == "q4_k"
    assert choose_output_kind("attn.to_v.weight", (1280, 1280), "q4_k_m", "f16") == "q6_k"
    assert choose_output_kind("attn.to_q.weight", (320, 320), "q4_k_m", "f16") == "f16"
    assert output_nbytes((256, 256), "q4_k") == 256 * 144


if __name__ == "__main__":
    test_mixed_k_policy()
    print("PASS: mixed K-quant policy")
