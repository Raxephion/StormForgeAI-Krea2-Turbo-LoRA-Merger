"""
lora_merge.py

Core logic for merging one or more LoRA safetensors files into a base
diffusion model checkpoint (e.g. Krea 2 Turbo), producing a single merged
safetensors file with the LoRA permanently baked into the weights.

Supports:
  - Standard floating-point checkpoints (BF16 / FP16 / FP32)
  - Scaled FP8 checkpoints (float8_e4m3fn / float8_e5m2 + .weight_scale)
  - ComfyUI native INT8 checkpoints (int8_tensorwise, with or without ConvRot
    regular-Hadamard rotation via .comfy_quant or _quantization_metadata)

Supports the two most common LoRA key-naming conventions found in the wild:
  - "Kohya" / ComfyUI style:   lora_unet_..._lora_down.weight / _lora_up.weight (+ .alpha)
  - "Diffusers / PEFT" style:  ....lora_A.weight / ....lora_B.weight
  - LyCORIS LoKr style:        ....lokr_w1 / ....lokr_w2

Because exact key names vary by trainer/tool, key matching is done in two
passes: an exact-match pass, then a normalized fuzzy-match pass. Any LoRA
layers that cannot be matched to a base weight are reported back to the
caller rather than silently dropped, so the user can see what happened
before trusting the merged file.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class LoraLayer:
    """A single matched (down, up) LoRA pair for one base weight."""
    base_key: str
    down: torch.Tensor
    up: torch.Tensor
    alpha: float | None  # None => alpha defaults to rank (scale = 1.0)


@dataclass
class LokrLayer:
    """
    A single matched LyCORIS LoKr pair for one base weight. The weight
    update is delta = kron(w1, w2) -- confirmed empirically (see
    check_lokr_scale.py) that alpha is NOT applied as alpha/rank for this
    format; some trainers store a huge sentinel alpha value that is not
    meant to be divided into the delta. The Kronecker product is already
    correctly scaled as stored.
    """
    base_key: str
    w1: torch.Tensor
    w2: torch.Tensor


@dataclass
class MergeReport:
    lora_file: str
    matched: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    shape_mismatches: list[str] = field(default_factory=list)
    quant_modes: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #

def load_state_dict(path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load a .safetensors file fully into memory."""
    state = {}
    with safe_open(path, framework="pt", device=device) as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return state


def load_metadata(path: str) -> dict[str, str] | None:
    """
    Load the __metadata__ header of a safetensors file. Some checkpoints
    (e.g. scaled-fp8 quantized ones) rely on a header like
    '_quantization_metadata' for the inference loader to know how to treat
    the weights -- this must be carried through to a merged output file or
    the merged model can silently load incorrectly.
    """
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.metadata()


def peek_keys(path: str, limit: int = 20) -> list[str]:
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    return keys[:limit]


# --------------------------------------------------------------------------- #
# ConvRot (regular Hadamard) & quantization metadata helpers
# --------------------------------------------------------------------------- #

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

_QUANT_COMPANION_SUFFIXES = (
    ".weight_scale",
    ".input_scale",
    ".comfy_quant",
    ".weight_scale_2",
    ".weight_s_channel",
    ".weight_s_rel",
    ".weight_codebook",
)


def _build_hadamard(size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Build the normalized regular Hadamard matrix H_size used by ComfyUI /
    comfy-kitchen ConvRot. `size` must be a power of 4 (4, 16, 64, 256, ...).

    Matches comfy_kitchen's exact 4x4 regular Hadamard seed:
        H4 = [[ 1,  1,  1, -1],
              [ 1,  1, -1,  1],
              [ 1, -1,  1,  1],
              [-1,  1,  1,  1]]
    H_size = kron(H4, H4, ..., H4) / sqrt(size).
    H_size is symmetric (H == H.T) and orthogonal (H @ H == I).
    """
    if size < 4 or (size & (size - 1)) != 0 or ((size.bit_length() - 1) % 2) != 0:
        raise ValueError(f"ConvRot Hadamard group size must be a power of 4 (e.g. 16, 64, 256), got {size}.")

    cache_key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    h4 = torch.tensor(
        [
            [1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    h = h / math.sqrt(size)
    _HADAMARD_CACHE[cache_key] = h
    return h


def _rotate_weight(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """
    Apply block-diagonal ConvRot Hadamard rotation along the input-feature
    dimension (dim=1) of a 2D weight matrix: W_rot = W @ (I kron H_g).
    Because H_g is symmetric and orthogonal, this function is its own inverse.
    """
    if weight.ndim != 2:
        raise ValueError(f"ConvRot expects a 2D weight tensor, got shape {tuple(weight.shape)}.")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"ConvRot group_size={group_size} does not divide in_features={in_features} "
            f"for weight shape {tuple(weight.shape)}."
        )
    h = _build_hadamard(group_size, device=weight.device, dtype=weight.dtype)
    return (weight.reshape(out_features, in_features // group_size, group_size) @ h).reshape(
        out_features, in_features
    )


def _decode_comfy_quant_tensor(tensor: torch.Tensor) -> dict | None:
    """Decode a `<module>.comfy_quant` uint8 JSON tensor into a dict."""
    try:
        raw = bytes(tensor.detach().cpu().to(torch.uint8).flatten().tolist())
        text = raw.decode("utf-8").strip("\x00 \t\r\n")
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _get_layer_quant_config(
    base_key: str,
    base_state: dict[str, torch.Tensor],
    base_metadata: dict[str, str] | None = None,
) -> dict | None:
    """
    Look up per-layer quantization config from either:
      1. `<module>.comfy_quant` tensor in `base_state` (ComfyUI native format), or
      2. `_quantization_metadata` JSON in the safetensors header.
    """
    if not base_key.endswith(".weight"):
        return None
    module_prefix = base_key[: -len(".weight")]

    cq_key = module_prefix + ".comfy_quant"
    if cq_key in base_state:
        cfg = _decode_comfy_quant_tensor(base_state[cq_key])
        if cfg is not None:
            return cfg

    if base_metadata and "_quantization_metadata" in base_metadata:
        try:
            qmeta = json.loads(base_metadata["_quantization_metadata"])
            layers = qmeta.get("layers") if isinstance(qmeta, dict) else None
            if isinstance(layers, dict):
                candidates = [module_prefix]
                for pre in ("model.diffusion_model.", "diffusion_model.", "transformer."):
                    if module_prefix.startswith(pre):
                        candidates.append(module_prefix[len(pre):])
                for cand in candidates:
                    if cand in layers and isinstance(layers[cand], dict):
                        return layers[cand]
        except Exception:
            pass

    return None


def _describe_layer_quant(
    base_key: str,
    base_state: dict[str, torch.Tensor],
    base_metadata: dict[str, str] | None = None,
) -> str:
    """Return a concise human-readable label for how a layer will be merged."""
    base_w = base_state[base_key]
    scale_key = base_key[: -len(".weight")] + ".weight_scale" if base_key.endswith(".weight") else None
    has_scale = scale_key is not None and scale_key in base_state

    if base_w.dtype == torch.int8:
        if not has_scale:
            return "int8_unscaled (unsupported)"
        qcfg = _get_layer_quant_config(base_key, base_state, base_metadata) or {}
        params = qcfg.get("params") if isinstance(qcfg.get("params"), dict) else {}
        use_convrot = bool(qcfg.get("convrot", params.get("convrot", False)))
        if use_convrot:
            gs = int(qcfg.get("convrot_groupsize") or params.get("convrot_groupsize") or 256)
            return f"int8_convrot (gs={gs})"
        return "int8_tensorwise"

    if base_w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return "fp8_scaled" if has_scale else "fp8_unscaled"

    return str(base_w.dtype).replace("torch.", "")


# --------------------------------------------------------------------------- #
# Key normalization for fuzzy matching
# --------------------------------------------------------------------------- #

_STRIP_PREFIXES = [
    "lora_unet_", "lora_te_", "lora_transformer_",
    "diffusion_model.", "model.diffusion_model.",
    "base_model.model.", "transformer.",
]
_STRIP_SUFFIXES = [
    ".weight", ".lora_down.weight", ".lora_up.weight",
    ".lora_A.weight", ".lora_B.weight", ".alpha",
    ".lora_A.default.weight", ".lora_B.default.weight",
]


def normalize_key(key: str) -> str:
    """
    Reduce a key to a canonical form so LoRA-side and base-side names can be
    compared even when prefixes/suffixes and dot-vs-underscore conventions
    differ between the two files.
    """
    k = key
    for suf in _STRIP_SUFFIXES:
        if k.endswith(suf):
            k = k[: -len(suf)]
            break
    for pre in _STRIP_PREFIXES:
        if k.startswith(pre):
            k = k[len(pre):]
            break
    # Kohya-style names replace '.' with '_' in the module path -- undo that
    # so both conventions collapse to the same alnum-only representation.
    k = k.lower()
    k = re.sub(r"[._]+", "_", k)
    k = k.strip("_")
    return k


# --------------------------------------------------------------------------- #
# LoRA pair extraction (handles both naming conventions)
# --------------------------------------------------------------------------- #

def extract_lora_pairs(lora_state: dict[str, torch.Tensor]) -> dict[str, dict]:
    """
    Group a raw LoRA state dict into {base_name: {"down":..., "up":..., "alpha":...}}
    base_name is the *raw* (un-normalized) key with the down/up/alpha suffix removed,
    so it still carries whatever prefix convention the file used.
    """
    pairs: dict[str, dict] = {}

    for key, tensor in lora_state.items():
        base_name = None
        role = None

        if key.endswith(".lora_down.weight"):
            base_name, role = key[: -len(".lora_down.weight")], "down"
        elif key.endswith(".lora_up.weight"):
            base_name, role = key[: -len(".lora_up.weight")], "up"
        elif key.endswith(".lora_A.weight"):
            base_name, role = key[: -len(".lora_A.weight")], "down"
        elif key.endswith(".lora_B.weight"):
            base_name, role = key[: -len(".lora_B.weight")], "up"
        elif key.endswith(".lora_A.default.weight"):
            base_name, role = key[: -len(".lora_A.default.weight")], "down"
        elif key.endswith(".lora_B.default.weight"):
            base_name, role = key[: -len(".lora_B.default.weight")], "up"
        elif key.endswith(".alpha"):
            base_name, role = key[: -len(".alpha")], "alpha"
        elif key.endswith(".lokr_w1"):
            base_name, role = key[: -len(".lokr_w1")], "lokr_w1"
        elif key.endswith(".lokr_w2"):
            base_name, role = key[: -len(".lokr_w2")], "lokr_w2"
        else:
            continue  # not a LoRA weight we recognize (skip metadata etc.)

        pairs.setdefault(base_name, {})[role] = tensor

    return pairs


# --------------------------------------------------------------------------- #
# Krea 2 specific: diffusers-style -> native/AI-Toolkit-style key translation
# --------------------------------------------------------------------------- #
#
# Verified directly against a real Krea 2 Turbo fp8-scaled checkpoint and a
# diffusers-format LoRA (both key skeletons dumped and cross-checked -- block
# counts, sub-module names, and component names all line up 1:1). This is an
# explicit rule table, not a guess:
#
#   diffusers (LoRA)                          native (base checkpoint)
#   -----------------------------------------  -------------------------------
#   transformer.transformer_blocks.N.*         blocks.N.*
#   transformer.text_fusion.layerwise_blocks.N  txtfusion.layerwise_blocks.N
#   transformer.text_fusion.refiner_blocks.N    txtfusion.refiner_blocks.N
#   transformer.text_fusion.projector           txtfusion.projector
#   transformer.img_in                          first
#   transformer.final_layer.linear              last.linear
#   transformer.time_embed.linear_1/2           tmlp.0 / tmlp.2   (index 1
#                                                 unused in the base; confirmed
#                                                 via exact key dump)
#   transformer.txt_in.linear_1/2               txtmlp.1 / txtmlp.3  (index 0
#                                                 is a norm scale, not a linear;
#                                                 confirmed via exact key dump)
#   transformer.time_mod_proj                    tproj.1   (confirmed correct)
#
#   attn.to_q / to_k / to_v / to_gate           attn.wq / wk / wv / gate
#   attn.to_out.0                               attn.wo
#   ff.up / ff.down / ff.gate                   mlp.up / mlp.down / mlp.gate

_TOP_LEVEL_RENAMES = [
    (re.compile(r"^transformer\.transformer_blocks\.(\d+)\."), r"blocks.\1."),
    (re.compile(r"^transformer\.text_fusion\.layerwise_blocks\.(\d+)\."), r"txtfusion.layerwise_blocks.\1."),
    (re.compile(r"^transformer\.text_fusion\.refiner_blocks\.(\d+)\."), r"txtfusion.refiner_blocks.\1."),
    (re.compile(r"^transformer\.text_fusion\.projector$"), "txtfusion.projector"),
    (re.compile(r"^transformer\.img_in$"), "first"),
    (re.compile(r"^transformer\.final_layer\.linear$"), "last.linear"),
    (re.compile(r"^transformer\.time_embed\.linear_1$"), "tmlp.0"),
    (re.compile(r"^transformer\.time_embed\.linear_2$"), "tmlp.2"),
    (re.compile(r"^transformer\.txt_in\.linear_1$"), "txtmlp.1"),
    (re.compile(r"^transformer\.txt_in\.linear_2$"), "txtmlp.3"),
    (re.compile(r"^transformer\.time_mod_proj$"), "tproj.1"),
]

_COMPONENT_RENAMES = [
    (re.compile(r"\.attn\.to_out\.0$"), ".attn.wo"),
    (re.compile(r"\.attn\.to_q$"), ".attn.wq"),
    (re.compile(r"\.attn\.to_k$"), ".attn.wk"),
    (re.compile(r"\.attn\.to_v$"), ".attn.wv"),
    (re.compile(r"\.attn\.to_gate$"), ".attn.gate"),
    (re.compile(r"\.ff\.up$"), ".mlp.up"),
    (re.compile(r"\.ff\.down$"), ".mlp.down"),
    (re.compile(r"\.ff\.gate$"), ".mlp.gate"),
]


def diffusers_to_native(base_name: str) -> str | None:
    """
    Translate a diffusers-style Krea 2 LoRA base name into the native/
    AI-Toolkit-style name used by this project's base checkpoints.
    Returns None if base_name doesn't match a known diffusers-style pattern
    (caller should fall back to fuzzy matching in that case).
    """
    if not base_name.startswith("transformer."):
        return None

    translated = base_name
    matched_top = False
    for pattern, repl in _TOP_LEVEL_RENAMES:
        new = pattern.sub(repl, translated)
        if new != translated:
            translated = new
            matched_top = True
            break

    if not matched_top:
        return None

    for pattern, repl in _COMPONENT_RENAMES:
        translated = pattern.sub(repl, translated)

    return translated


def build_base_lookup(base_state: dict[str, torch.Tensor]) -> dict[str, str]:
    """normalized_key -> original base_state key, for weight tensors only."""
    lookup = {}
    for key in base_state.keys():
        if not key.endswith(".weight"):
            continue
        lookup[normalize_key(key)] = key
    return lookup


def _resolve_base_key(base_name: str, base_state: dict[str, torch.Tensor], base_lookup: dict[str, str]) -> str | None:
    """Shared base-key resolution used by both LoRA and LoKr matching."""
    candidate = base_name + ".weight"
    if candidate in base_state:
        return candidate

    translated = diffusers_to_native(base_name)
    if translated is not None:
        # Try both the bare native name (split checkpoints) and the
        # "model.diffusion_model." prefixed form used by all-in-one
        # checkpoints that bundle the diffusion model + VAE + text
        # encoder(s) into a single file.
        for candidate_prefix in ("", "model.diffusion_model."):
            candidate2 = candidate_prefix + translated + ".weight"
            if candidate2 in base_state:
                return candidate2

    norm = normalize_key(base_name)
    return base_lookup.get(norm)


def match_layers(
    lora_pairs: dict[str, dict],
    base_state: dict[str, torch.Tensor],
    base_lookup: dict[str, str],
) -> tuple[list[LoraLayer | LokrLayer], list[str], list[str]]:
    matched_layers: list[LoraLayer | LokrLayer] = []
    unmatched: list[str] = []
    shape_mismatches: list[str] = []

    for base_name, parts in lora_pairs.items():
        is_lokr = "lokr_w1" in parts and "lokr_w2" in parts
        is_lora = "down" in parts and "up" in parts

        if not is_lokr and not is_lora:
            unmatched.append(base_name + " (incomplete pair)")
            continue

        base_key = _resolve_base_key(base_name, base_state, base_lookup)
        if base_key is None:
            unmatched.append(base_name)
            continue

        base_shape = base_state[base_key].shape
        expected_out, expected_in = base_shape[0], base_shape[1] if len(base_shape) > 1 else base_shape[0]

        if is_lokr:
            w1, w2 = parts["lokr_w1"], parts["lokr_w2"]
            kron_out = w1.shape[0] * w2.shape[0]
            kron_in = (w1.shape[1] if w1.dim() > 1 else 1) * (w2.shape[1] if w2.dim() > 1 else 1)
            if kron_out != expected_out or kron_in != expected_in:
                shape_mismatches.append(
                    f"{base_key}: base{tuple(base_shape)} vs lokr kron result ({kron_out}, {kron_in}) "
                    f"[w1{tuple(w1.shape)} x w2{tuple(w2.shape)}]"
                )
                continue
            matched_layers.append(LokrLayer(base_key=base_key, w1=w1, w2=w2))
        else:
            down, up = parts["down"], parts["up"]
            alpha_t = parts.get("alpha")
            alpha = float(alpha_t.item()) if alpha_t is not None else None

            if up.shape[0] != expected_out or down.shape[1] != expected_in:
                shape_mismatches.append(
                    f"{base_key}: base{tuple(base_shape)} vs lora up{tuple(up.shape)}/down{tuple(down.shape)}"
                )
                continue

            matched_layers.append(LoraLayer(base_key=base_key, down=down, up=up, alpha=alpha))

    return matched_layers, unmatched, shape_mismatches


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #

def _compute_delta(
    layer: LoraLayer | LokrLayer,
    user_weight: float,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(layer, LokrLayer):
        # Confirmed empirically (check_lokr_scale.py): no alpha/rank division
        # for this format -- the Kronecker product is already scaled as
        # trained. Only the user-facing weight slider scales it further.
        w1 = layer.w1.to(device=device, dtype=compute_dtype)
        w2 = layer.w2.to(device=device, dtype=compute_dtype)
        return torch.kron(w1, w2) * user_weight
    else:
        rank = layer.down.shape[0]
        scale = (layer.alpha / rank) if layer.alpha is not None else 1.0
        scale *= user_weight
        down = layer.down.to(device=device, dtype=compute_dtype)
        up = layer.up.to(device=device, dtype=compute_dtype)
        return (up @ down) * scale


@torch.no_grad()
def merge_layer_into_base(
    base_state: dict[str, torch.Tensor],
    layer: LoraLayer | LokrLayer,
    user_weight: float,
    compute_dtype: torch.dtype,
    base_metadata: dict[str, str] | None = None,
) -> str:
    base_w = base_state[layer.base_key]
    orig_dtype = base_w.dtype
    device = base_w.device

    scale_key = layer.base_key[: -len(".weight")] + ".weight_scale" if layer.base_key.endswith(".weight") else None
    has_scale = scale_key is not None and scale_key in base_state

    # ------------------------------------------------------------------ #
    # Path 1: INT8 quantized weights (ComfyUI int8_tensorwise +/- ConvRot)
    # ------------------------------------------------------------------ #
    if orig_dtype == torch.int8:
        if not has_scale:
            raise ValueError(
                f"Layer '{layer.base_key}' is int8 without a companion '.weight_scale' tensor; "
                "cannot safely dequantize or merge."
            )

        qcfg = _get_layer_quant_config(layer.base_key, base_state, base_metadata) or {}
        params = qcfg.get("params") if isinstance(qcfg.get("params"), dict) else {}
        use_convrot = bool(qcfg.get("convrot", params.get("convrot", False)))
        gs = int(qcfg.get("convrot_groupsize") or params.get("convrot_groupsize") or 256)

        # Always compute quantized dequant/rotate/requant in fp32 so scale
        # clamps (1e-30) never underflow in fp16 and Hadamard math stays exact.
        delta_f32 = _compute_delta(layer, user_weight, torch.float32, device=device).reshape(base_w.shape)

        orig_scale = base_state[scale_key]
        weight_scale = orig_scale.to(device=device, dtype=torch.float32)

        # Because the ConvRot Hadamard transform R is linear and orthogonal
        # (R^2 = I), R(R(W_rot) + delta) == W_rot + R(delta). Rotating delta
        # directly into ConvRot space is mathematically exact and avoids
        # unnecessary round-trip error on the base weights.
        dequant_rot = base_w.to(torch.float32) * weight_scale
        if use_convrot:
            delta_f32 = _rotate_weight(delta_f32, gs)
        merged_rot = dequant_rot + delta_f32

        if orig_scale.numel() > 1:
            absmax = merged_rot.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
        else:
            absmax = merged_rot.abs().amax().clamp(min=1e-30)
        new_scale = (absmax / 127.0).clamp(min=1e-30)
        requantized = (merged_rot / new_scale).round().clamp(-127, 127).to(torch.int8)

        base_state[layer.base_key] = requantized
        base_state[scale_key] = new_scale.reshape(orig_scale.shape).to(
            device=orig_scale.device, dtype=orig_scale.dtype
        )
        return f"int8_convrot (gs={gs})" if use_convrot else "int8_tensorwise"

    # ------------------------------------------------------------------ #
    # Path 2: Scaled FP8 weights (float8_e4m3fn / float8_e5m2)
    # ------------------------------------------------------------------ #
    is_fp8 = orig_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if is_fp8 and has_scale:
        delta_f32 = _compute_delta(layer, user_weight, torch.float32, device=device).reshape(base_w.shape)
        orig_scale = base_state[scale_key]
        weight_scale = orig_scale.to(device=device, dtype=torch.float32)

        dequantized = base_w.to(torch.float32) * weight_scale
        merged = dequantized + delta_f32

        fp8_max = 448.0 if orig_dtype == torch.float8_e4m3fn else 57344.0
        if orig_scale.numel() > 1:
            max_abs = merged.abs().amax(dim=1, keepdim=True)
        else:
            max_abs = merged.abs().amax()
        new_scale = (max_abs / fp8_max).clamp(min=1e-12)
        requantized = (merged / new_scale).clamp(-fp8_max, fp8_max).to(orig_dtype)

        base_state[layer.base_key] = requantized
        base_state[scale_key] = new_scale.reshape(orig_scale.shape).to(
            device=orig_scale.device, dtype=orig_scale.dtype
        )
        return "fp8_scaled"

    # ------------------------------------------------------------------ #
    # Path 3: Standard floating-point weights (BF16 / FP16 / FP32)
    # ------------------------------------------------------------------ #
    delta = _compute_delta(layer, user_weight, compute_dtype, device=device).reshape(base_w.shape)
    merged = base_w.to(compute_dtype) + delta
    base_state[layer.base_key] = merged.to(orig_dtype)
    return str(orig_dtype).replace("torch.", "")


def merge_lora_file(
    base_state: dict[str, torch.Tensor],
    lora_path: str,
    weight: float,
    compute_dtype: torch.dtype = torch.float32,
    progress_cb: Callable[[str], None] | None = None,
    base_metadata: dict[str, str] | None = None,
) -> MergeReport:
    """Merge a single LoRA file into base_state IN PLACE. Returns a report."""
    lora_state = load_state_dict(lora_path)
    pairs = extract_lora_pairs(lora_state)
    base_lookup = build_base_lookup(base_state)

    matched, unmatched, shape_mismatches = match_layers(pairs, base_state, base_lookup)
    quant_modes: Counter[str] = Counter()

    for i, layer in enumerate(matched):
        mode = merge_layer_into_base(
            base_state, layer, weight, compute_dtype, base_metadata=base_metadata
        )
        quant_modes[mode] += 1
        if progress_cb and i % 25 == 0:
            progress_cb(f"  merged {i}/{len(matched)} layers from {Path(lora_path).name}")

    return MergeReport(
        lora_file=Path(lora_path).name,
        matched=[l.base_key for l in matched],
        unmatched=unmatched,
        shape_mismatches=shape_mismatches,
        quant_modes=dict(quant_modes),
    )


def check_compatibility(base_path: str, lora_path: str) -> MergeReport:
    """Dry-run: report how many LoRA layers would match, without merging."""
    base_state = load_state_dict(base_path)
    base_metadata = load_metadata(base_path)
    lora_state = load_state_dict(lora_path)
    pairs = extract_lora_pairs(lora_state)
    base_lookup = build_base_lookup(base_state)
    matched, unmatched, shape_mismatches = match_layers(pairs, base_state, base_lookup)

    quant_modes: Counter[str] = Counter()
    for layer in matched:
        quant_modes[_describe_layer_quant(layer.base_key, base_state, base_metadata)] += 1

    return MergeReport(
        lora_file=Path(lora_path).name,
        matched=[l.base_key for l in matched],
        unmatched=unmatched,
        shape_mismatches=shape_mismatches,
        quant_modes=dict(quant_modes),
    )


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #

DTYPE_MAP = {
    "keep original": None,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _should_preserve_dtype(key: str, tensor: torch.Tensor) -> bool:
    """Never cast quantized weights, integer metadata, or quantization scales."""
    if not tensor.is_floating_point():
        return True
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    if key.endswith(_QUANT_COMPANION_SUFFIXES):
        return True
    return False


def save_merged(
    base_state: dict[str, torch.Tensor],
    output_path: str,
    out_dtype: torch.dtype | None,
    metadata: dict[str, str] | None = None,
) -> None:
    if out_dtype is not None:
        base_state = {
            k: (v if _should_preserve_dtype(k, v) else v.to(out_dtype))
            for k, v in base_state.items()
        }
    # safetensors requires contiguous CPU tensors
    base_state = {k: v.detach().to("cpu").contiguous() for k, v in base_state.items()}
    save_file(base_state, output_path, metadata=metadata)