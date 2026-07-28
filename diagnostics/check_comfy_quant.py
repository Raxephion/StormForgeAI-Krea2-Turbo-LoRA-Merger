"""
check_comfy_quant.py

Inspects a sample .comfy_quant tensor to see what it stores. This project's
merge logic only ever modifies .weight and .weight_scale, leaving anything
else (including .comfy_quant) untouched and copied through as-is. This
script exists to confirm that assumption is actually safe for a given
checkpoint, rather than assuming it blindly.

Usage:
    python check_comfy_quant.py "C:\\path\\to\\file.safetensors"
"""

import sys
from safetensors import safe_open

if len(sys.argv) != 2:
    print("Usage: python check_comfy_quant.py \"C:\\path\\to\\file.safetensors\"")
    sys.exit(1)

path = sys.argv[1]
SAMPLE_KEYS = [
    "model.diffusion_model.blocks.0.attn.wq.comfy_quant",
    "model.diffusion_model.blocks.0.mlp.up.comfy_quant",
]

with safe_open(path, framework="pt") as f:
    print(f"File: {path}\n")
    for key in SAMPLE_KEYS:
        if key not in f.keys():
            print(f"[skip] {key} not found\n")
            continue
        t = f.get_tensor(key)
        print(f"=== {key} ===")
        print(f"  dtype: {t.dtype}")
        print(f"  shape: {tuple(t.shape)}")
        print(f"  numel: {t.numel()}")
        try:
            print(f"  values: {t.flatten()[:20].tolist()}")
        except Exception as e:
            print(f"  (could not print values directly: {e})")
            try:
                raw_bytes = t.view(torch.uint8).flatten()[:64].tolist() if hasattr(t, "view") else None
                print(f"  raw bytes sample: {raw_bytes}")
            except Exception:
                pass
        print()
