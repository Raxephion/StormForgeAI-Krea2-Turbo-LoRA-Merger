# Krea 2 Turbo LoRA Merger

A standalone desktop application for permanently merging one or more LoRAs into
Krea 2 Turbo checkpoints.

Unlike ComfyUI, this application is built for people who simply want to merge
models without learning node graphs, building workflows, or installing an entire
inference pipeline. Load your model, load your LoRAs, choose the merge settings,
and save.

While it works perfectly alongside ComfyUI, **ComfyUI is not required**.

---

> **Why not just use two merge nodes in ComfyUI?**
>
> Because not everyone wants to use ComfyUI.
>
> This project exists for exactly the same reason as the Krea 2 WebUI project:
> to provide a clean, dedicated application for a specific task.
>
> It already supports merging LoRAs into both split Krea 2 diffusion models and
> all-in-one checkpoints, and over time it will grow far beyond a simple
> one-click LoRA merge utility with additional model engineering features.

---

## Features

- Merge one or multiple LoRAs into a base checkpoint
- Supports both **split Krea 2 models** (`diffusion_models/`)
- Supports **all-in-one checkpoints** (`checkpoints/`)
- Automatic key matching
- Supports Kohya and Diffusers/PEFT LoRA formats
- FP8-aware merging
- Preserves metadata headers
- Adjustable merge strength for every loaded LoRA
- Compatibility checker before merging
- Runs completely offline
- No ComfyUI installation required

---

> ✅ **Confirmed working** on both split (`diffusion_models/`) and
> all-in-one (`checkpoints/`) Krea 2 checkpoints.
>
> FP8 dequantization/requantization, key matching,
> `model.diffusion_model.` prefix handling and metadata preservation have all
> been verified against real-world checkpoints and renders.
>
> If you encounter an unsupported checkpoint or LoRA combination, please open
> an issue together with the output of `check_keys_full.py`.

---

## Requirements

- Windows
- Python 3.10 or 3.11 installed and available on PATH
- Base checkpoint (`.safetensors`)
- One or more LoRA files (`.safetensors`)

---

## Installation

Run:

```
install.bat
```

The installer automatically:

1. Creates a local virtual environment
2. Installs Gradio
3. Installs PyTorch (CPU build)
4. Installs safetensors and all required dependencies

### Optional GPU acceleration

If you have an NVIDIA GPU you can install the CUDA build of PyTorch afterwards:

```bat
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with the version matching your CUDA installation.

---

## Running

Simply run

```
run.bat
```

The application opens in your browser:

```
http://127.0.0.1:7860
```

---

## Usage

### 1. Load files

Select:

- Base checkpoint
- One or more LoRA files

### 2. Configure LoRAs

Each loaded LoRA receives its own weight slider.

- `1.0` = original trained strength
- `< 1.0` = softer effect
- `> 1.0` = stronger effect
- negative values invert the learned changes

### 3. Configure merge

Choose:

- Compute precision
- Output precision
- CPU or CUDA
- Output location
- Output filename

### 4. Check compatibility

The compatibility checker performs a dry run and reports:

- matched layers
- unmatched layers
- skipped layers

before modifying anything.

This makes it easy to spot incompatible LoRAs before committing to a merge.

### 5. Merge

Click **Merge & Save**.

The application permanently bakes every compatible LoRA into the checkpoint and
writes a new standalone `.safetensors` model.

---

## How LoRA merging works

A LoRA stores small low-rank matrices describing changes to selected layers
instead of storing an entire model.

For each matching layer the merger computes:

```
new_weight = base_weight +
             (up @ down) *
             (alpha / rank) *
             user_strength
```

The resulting tensors replace the originals and are written into a completely
new checkpoint.

The output model no longer requires external LoRAs during inference.

---

## Supported formats

The merger currently supports:

- Kohya LoRAs
- ComfyUI LoRAs
- Diffusers / PEFT LoRAs

Automatic key matching handles common naming differences between checkpoints and
LoRAs. Layers that cannot be matched are reported rather than guessed.

---

## Notes

- Designed primarily for transformer-based diffusion models such as Krea 2
- Supports standard linear-layer LoRAs
- Convolutional LoRAs are currently not supported
- The base checkpoint is loaded into RAM (or VRAM when using CUDA), so ensure
  sufficient available memory
- Always run **Check Compatibility** when testing an unfamiliar LoRA

---

## Roadmap

This project is under active development.

Planned additions include more advanced merge algorithms, additional checkpoint
utilities, model inspection tools, and workflow features aimed at making model
engineering accessible without requiring ComfyUI.
