# Krea 2 Turbo LoRA Merger

> ✅ **Confirmed working**, including split checkpoints, all-in-one
> checkpoints (diffusion model + VAE + text encoder bundled), standard LoRA
> adapters, and LyCORIS LoKr adapters (Kronecker-product format). All fixed
> and verified against real files and real renders (Fp8). If you hit issues on a
> different checkpoint/LoRA combination, please open an issue with a
> `check_keys_full.py` dump of both files.


A standalone, offline Gradio app that permanently merges one or more LoRA
files into a base diffusion model checkpoint (built for Krea 2 Turbo, but
works with any safetensors checkpoint using compatible LoRA naming).

No internet access is required to run it — everything happens locally on
your machine.

## Supported checkpoint formats & quantization

The merger automatically detects per-layer precision and quantization metadata in the base checkpoint — including mixed-precision checkpoints where transformer blocks are quantized while boundary/fusion layers remain in `bf16` or `fp16`:

| Checkpoint format | Detection | Merge behavior |
| --- | --- | --- |
| **Standard float (`BF16` / `FP16` / `FP32`)** | Regular floating-point `.weight` tensors | Computes the LoRA/LoKr delta in the selected compute precision and casts back to the original tensor dtype. |
| **Scaled `FP8` (`float8_e4m3fn` / `float8_e5m2`)** | FP8 `.weight` + companion `.weight_scale` | Dequantizes to `fp32`, adds the delta, recalculates `.weight_scale`, and requantizes back to the original FP8 dtype. |
| **ComfyUI native `INT8` (`int8_tensorwise`)** | `int8` `.weight` + `.weight_scale` (`.comfy_quant` or `_quantization_metadata`) | Dequantizes to `fp32`, adds the delta, recalculates per-row (or tensorwise) `.weight_scale`, and requantizes to `int8`. |
| **ComfyUI native `INT8 ConvRot`** | `int8` `.weight` + `.weight_scale` + `convrot: true` (`convrot_groupsize`: power of 4, e.g. `256`) | Dequantizes to `fp32`, rotates the LoRA/LoKr delta into ConvRot space using ComfyUI's normalized 4×4 regular-Hadamard Kronecker transform, merges, and requantizes to `int8` with updated `.weight_scale`. |

### Notes on quantized checkpoints

- **Verify before merging:** Run **Check compatibility (dry run)** first — the report lists the detected layer formats (for example, `int8_convrot (gs=256): 224, bfloat16: 40`) so you can confirm the checkpoint was recognized properly.
- **Output precision:** Leave **Output precision** set to `keep original` for quantized checkpoints. Even if you choose `bf16` or `fp16`, quantized weights (`int8`, `fp8`), `.weight_scale` tensors, and `.comfy_quant` metadata tensors are preserved untouched so the saved file stays valid in ComfyUI.
- **INT8 ConvRot math:** Because the ConvRot regular-Hadamard transform is linear and orthogonal ($R^2 = I$), rotating the LoRA/LoKr delta directly into ConvRot space and adding it to the dequantized rotated weights is mathematically equivalent to un-rotating, merging, and re-rotating, while avoiding extra round-trip error on the base weights.
- **Unsupported formats:** Packed 4-bit/6-bit formats (such as `convrot_w4a4`, `asym_w4a8_int8`, or `w6a8_int8`) and legacy custom-node INT8 formats without `.weight_scale` are not supported for direct merging.


## Requirements

- Windows, with Python 3.10 or 3.11 installed and on PATH
  (https://www.python.org/downloads/ — check "Add python.exe to PATH")
- Your base checkpoint and LoRA file(s) already downloaded locally as
  `.safetensors` files

## Install

Double-click `install.bat` (or run it from a terminal). It will:

1. Create a local virtual environment in `.venv`
2. Install Gradio, PyTorch (CPU version), and safetensors

If you have an NVIDIA GPU and want faster merging, after `install.bat`
finishes, run:

```
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

(swap `cu121` for the CUDA build matching your GPU driver — see
https://pytorch.org/get-started/locally/)

## Run

Double-click `run.bat`. This opens the app in your browser at
`http://127.0.0.1:7860`.

## Using the app

1. **Load files** — upload your base model `.safetensors` file, and one or
   more LoRA `.safetensors` files.
2. **LoRA weights** — a weight slider appears for each LoRA you loaded.
   `1.0` applies the LoRA at its trained strength; lower values apply it
   more subtly, negative values invert its effect.
3. **Settings**:
   - **Compute precision** — precision used for the merge math itself.
     `fp32` is safest/most accurate; `fp16`/`bf16` are faster and use less
     RAM but can lose a little precision.
   - **Output precision** — precision of the saved file. `keep original`
     preserves whatever precision the base checkpoint was in.
   - **Device** — `cpu` works everywhere; `cuda` is faster if you have a
     working GPU build of PyTorch installed.
   - **Output folder / filename** — where the merged `.safetensors` file
     is written.
4. **Check compatibility (dry run)** — before merging, run this to see how
   many LoRA layers actually match your base model's weight names. LoRAs
   trained for a different architecture, or exported in an unexpected
   format, will show up here as "unmatched" rather than silently failing.
5. **Merge & Save** — performs the actual merge and writes the output file.
   The log shows how many layers were merged, skipped, or mismatched.

## How it works

LoRA weight files store, for a subset of layers, a low-rank pair of
matrices (`down`/`up`, sometimes named `lora_A`/`lora_B`) plus an optional
`alpha` scaling value. Merging bakes each LoRA in by computing, for every
matched layer:

```
new_weight = base_weight + (up @ down) * (alpha / rank) * your_weight_slider
```

This is applied directly to the base checkpoint's tensors and the result
is saved as a new, standalone `.safetensors` file — no LoRA loader needed
at inference time.

The app recognizes both common LoRA naming conventions (Kohya/ComfyUI-style
`lora_down`/`lora_up`, and Diffusers/PEFT-style `lora_A`/`lora_B`), and
falls back to fuzzy key matching when prefixes differ between the LoRA and
base file. Any layer it can't confidently match is reported, not guessed.

## Notes / limitations

- This performs a **linear-layer LoRA merge**. It assumes standard
  `(out_features, in_features)` weight/LoRA shapes, which covers Krea 2's
  transformer architecture (and most modern DiT/UNet models). It is not
  designed for convolutional LoRAs.
- Merging loads the full base model into memory (RAM, or VRAM if using
  `cuda`). Make sure you have enough — check your checkpoint's file size
  as a rough guide.
- Always run "Check compatibility" first on a new LoRA/base combination
  before trusting the merged output.

---

## Roadmap

This project is under active development.

Planned additions include more advanced merge algorithms, additional checkpoint
utilities, model inspection tools, and workflow features aimed at making model
engineering accessible without requiring ComfyUI.
