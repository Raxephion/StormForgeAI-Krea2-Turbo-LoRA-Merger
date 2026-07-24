"""
check_fp8_scale.py

Inspects a sample fp8-scaled weight + its companion weight_scale tensor to
determine the correct dequantization convention:
  - Is weight_scale a single scalar, or per-channel (one value per output row)?
  - Does `fp8_value * scale` or `fp8_value / scale` produce a plausible
    dequantized weight (typical neural net weights have small magnitude,
    roughly in the -1..1 to -5..5 range, not huge or tiny)?

Usage:
    python check_fp8_scale.py "C:\\path\\to\\base_model.safetensors"
"""

import sys
from safetensors import safe_open
import torch

if len(sys.argv) != 2:
    print("Usage: python check_fp8_scale.py \"C:\\path\\to\\file.safetensors\"")
    sys.exit(1)

path = sys.argv[1]

# A handful of representative weight+scale pairs to check
SAMPLE_KEYS = [
    "blocks.0.attn.wq.weight",
    "blocks.0.mlp.up.weight",
    "blocks.5.attn.wv.weight",
]

with safe_open(path, framework="pt") as f:
    all_keys = set(f.keys())
    print(f"File: {path}\n")

    for wkey in SAMPLE_KEYS:
        skey = wkey.replace(".weight", ".weight_scale")
        if wkey not in all_keys or skey not in all_keys:
            print(f"[skip] {wkey} or {skey} not found\n")
            continue

        w = f.get_tensor(wkey)
        s = f.get_tensor(skey)

        print(f"=== {wkey} ===")
        print(f"  weight       dtype={w.dtype}  shape={tuple(w.shape)}")
        print(f"  weight_scale dtype={s.dtype}  shape={tuple(s.shape)}  value(s)={s.flatten()[:5].tolist()}"
              f"{' ...' if s.numel() > 5 else ''}")

        w_f32 = w.to(torch.float32)
        s_f32 = s.to(torch.float32)

        raw_absmax = w_f32.abs().max().item()
        print(f"  raw fp8 stored values: absmax={raw_absmax:.4f} "
              f"(fp8_e4m3 max representable magnitude is 448)")

        # Try both conventions
        deq_mult = w_f32 * s_f32
        deq_div = w_f32 / s_f32.clamp(min=1e-12)

        print(f"  dequant via (weight * scale):  absmax={deq_mult.abs().max().item():.6f}  "
              f"std={deq_mult.std().item():.6f}")
        print(f"  dequant via (weight / scale):  absmax={deq_div.abs().max().item():.6f}  "
              f"std={deq_div.std().item():.6f}")
        print("  (a real neural net weight tensor's std is typically ~0.001-0.1; "
              "whichever line above is in that ballpark is the correct convention)")
        print()
