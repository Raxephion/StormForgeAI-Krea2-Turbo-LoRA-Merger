"""
check_lokr_scale.py

Computes the raw Kronecker product (w1 (x) w2) for a sample LoKr layer and
reports its magnitude under several candidate scaling conventions, so the
correct one can be picked based on which produces a plausible LoRA-delta
magnitude (typically a small fraction of the base weight's own scale) --
rather than guessing blindly given an unusually large stored alpha value.

Usage:
    python check_lokr_scale.py "C:\\path\\to\\lokr_lora.safetensors"
"""

import sys
import torch
from safetensors import safe_open

if len(sys.argv) != 2:
    print("Usage: python check_lokr_scale.py \"C:\\path\\to\\lokr_lora.safetensors\"")
    sys.exit(1)

path = sys.argv[1]
BASE_NAME = "diffusion_model.blocks.0.attn.wq"

with safe_open(path, framework="pt") as f:
    w1 = f.get_tensor(f"{BASE_NAME}.lokr_w1").to(torch.float32)
    w2 = f.get_tensor(f"{BASE_NAME}.lokr_w2").to(torch.float32)
    alpha_t = f.get_tensor(f"{BASE_NAME}.alpha")
    alpha = float(alpha_t.item())

print(f"w1: shape={tuple(w1.shape)}  std={w1.std().item():.6f}  absmax={w1.abs().max().item():.6f}")
print(f"w2: shape={tuple(w2.shape)}  std={w2.std().item():.6f}  absmax={w2.abs().max().item():.6f}")
print(f"alpha (raw): {alpha}")
print()

kron = torch.kron(w1, w2)
print(f"raw kron(w1, w2): shape={tuple(kron.shape)}  std={kron.std().item():.6f}  absmax={kron.abs().max().item():.6f}")
print()
print("For reference, a typical base weight in this checkpoint has std ~0.02-0.06,")
print("and a sensible LoRA-style delta should generally be a small fraction of that")
print("(not orders of magnitude larger or smaller).\n")

candidates = {
    "scale = 1.0 (no scaling)": 1.0,
    "alpha / w1.shape[0] (=4)": alpha / w1.shape[0],
    "alpha / w1.numel() (=16)": alpha / w1.numel(),
    "alpha / w2.shape[0] (=1536)": alpha / w2.shape[0],
    "alpha / (w1.shape[0]*w2.shape[0]) (=6144)": alpha / (w1.shape[0] * w2.shape[0]),
    "alpha / alpha (=1.0, self-normalizing)": alpha / alpha,
    "1 / w1.shape[0] (=1/4, no alpha at all)": 1.0 / w1.shape[0],
}

for label, scale in candidates.items():
    scaled = kron * scale
    print(f"{label:45s} -> std={scaled.std().item():.8f}  absmax={scaled.abs().max().item():.8f}")
