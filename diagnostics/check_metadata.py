"""
check_metadata.py

Prints the __metadata__ header of a safetensors file. Scaled-fp8 checkpoints
sometimes carry a flag here that tells the inference loader the weights need
their scale applied -- if a merge tool doesn't preserve this, the output
file can silently load as raw (un-rescaled) fp8 and produce garbage.

Usage:
    python check_metadata.py "C:\\path\\to\\file.safetensors"
"""

import sys
from safetensors import safe_open

if len(sys.argv) != 2:
    print("Usage: python check_metadata.py \"C:\\path\\to\\file.safetensors\"")
    sys.exit(1)

path = sys.argv[1]

with safe_open(path, framework="pt") as f:
    meta = f.metadata()

print(f"File: {path}\n")
if not meta:
    print("No __metadata__ header found (metadata() returned empty/None).")
else:
    print(f"__metadata__ header ({len(meta)} entries):")
    for k, v in meta.items():
        print(f"  {k!r}: {v!r}")
