"""
check_specific.py

Lists the exact (non-deduplicated) key names + shapes for a small set of
modules in a safetensors file. Used to pin down exact index numbers that
get hidden by skeleton-deduplication.

Usage:
    python check_specific.py "C:\\path\\to\\file.safetensors"
"""

import sys
from safetensors import safe_open

if len(sys.argv) != 2:
    print("Usage: python check_specific.py \"C:\\path\\to\\file.safetensors\"")
    sys.exit(1)

path = sys.argv[1]
WATCH = ["tmlp", "txtmlp", "tproj", "time_mod_proj", "time_embed", "txt_in"]

with safe_open(path, framework="pt") as f:
    keys = sorted(f.keys())
    print(f"File: {path}\n")
    for k in keys:
        if any(w in k for w in WATCH):
            t = f.get_tensor(k)
            print(f"  {k}    shape={tuple(t.shape)}  dtype={t.dtype}")
