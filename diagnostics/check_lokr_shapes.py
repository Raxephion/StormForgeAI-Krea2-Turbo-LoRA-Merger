"""
check_lokr_shapes.py

Prints shapes/dtypes for a sample LoKr layer (lokr_w1, lokr_w2, alpha) from
a LyCORIS LoKr-format LoRA, plus the corresponding base model weight shape,
to confirm the Kronecker-product merge math before implementing it.

Usage:
    python check_lokr_shapes.py "C:\\path\\to\\lokr_lora.safetensors" "C:\\path\\to\\base_model.safetensors"
"""

import sys
from safetensors import safe_open

if len(sys.argv) != 3:
    print("Usage: python check_lokr_shapes.py \"C:\\path\\to\\lokr_lora.safetensors\" \"C:\\path\\to\\base_model.safetensors\"")
    sys.exit(1)

lora_path, base_path = sys.argv[1], sys.argv[2]

SAMPLE_BASE_NAME = "diffusion_model.blocks.0.attn.wq"

print(f"LoRA file: {lora_path}")
with safe_open(lora_path, framework="pt") as f:
    for suffix in ["lokr_w1", "lokr_w2", "alpha"]:
        key = f"{SAMPLE_BASE_NAME}.{suffix}"
        if key in f.keys():
            t = f.get_tensor(key)
            print(f"  {key}")
            print(f"    dtype={t.dtype}  shape={tuple(t.shape)}")
            if t.numel() <= 4:
                print(f"    value(s)={t.flatten().tolist()}")
        else:
            print(f"  [not found] {key}")

print(f"\nBase model file: {base_path}")
base_key = "blocks.0.attn.wq.weight"
with safe_open(base_path, framework="pt") as f:
    if base_key in f.keys():
        t = f.get_tensor(base_key)
        print(f"  {base_key}")
        print(f"    dtype={t.dtype}  shape={tuple(t.shape)}")
    else:
        print(f"  [not found] {base_key} (try without any prefix adjustments)")
