"""Robust ViT model loader with HuggingFace → torchvision fallback.

Strategy:
  1. Try HuggingFace from local cache (fast path, no network).
  2. Try HuggingFace download under a wall-clock timeout.
  3. On any failure / timeout, fall back to torchvision `vit_b_16`
     (IMAGENET1K_V1) and adapt it so the compression hook still works.

Both branches return:
    (model, label, transform)
where:
    model      — eval()'d, on the right device, with `.forward(pixel_values=...)`
                 returning an object exposing `.logits` of shape (B, 1000).
    label      — human-readable string for the UI.
    transform  — torchvision transform to apply to a PIL image.
"""
from __future__ import annotations

import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import transforms

from utils import get_device


log = logging.getLogger("turboquant_vit.model_loader")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)


HF_NAME = "google/vit-base-patch16-224"


# ---------------------------------------------------------------------------
# Transforms — must match the model's pretraining normalization
# ---------------------------------------------------------------------------
HF_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

TV_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Torchvision adapter — exposes Q/K/V linears so the hook can find/compress them
# ---------------------------------------------------------------------------
class TVCompatSelfAttention(nn.Module):
    """Replaces torchvision's `nn.MultiheadAttention` inside `EncoderBlock`
    with a module that mirrors the HuggingFace `ViTSelfAttention` interface.

    Splits the combined `in_proj_weight` into separate `query`, `key`, `value`
    Linear modules and keeps `out_proj` so the EncoderBlock's call
    `self_attention(x, x, x, need_weights=False)` still produces post-projected
    output.
    """

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        d = mha.embed_dim
        h = mha.num_heads
        self.num_attention_heads = h
        self.attention_head_size = d // h
        self.all_head_size = d

        self.query = nn.Linear(d, d, bias=mha.in_proj_bias is not None)
        self.key = nn.Linear(d, d, bias=mha.in_proj_bias is not None)
        self.value = nn.Linear(d, d, bias=mha.in_proj_bias is not None)
        with torch.no_grad():
            W = mha.in_proj_weight
            self.query.weight.copy_(W[:d])
            self.key.weight.copy_(W[d:2 * d])
            self.value.weight.copy_(W[2 * d:])
            if mha.in_proj_bias is not None:
                b = mha.in_proj_bias
                self.query.bias.copy_(b[:d])
                self.key.bias.copy_(b[d:2 * d])
                self.value.bias.copy_(b[2 * d:])

        # reuse torchvision's output projection (already trained)
        self.out_proj = mha.out_proj

    def forward(self, query, key=None, value=None, *args, **kwargs):
        """Default (uncompressed) forward. Will be monkey-patched by the
        compression hook at inference time. We keep it correct so the model is
        usable even without the hook.
        """
        x = query
        q = self.query(x); k = self.key(x); v = self.value(x)
        B, N, C = x.shape
        h = self.num_attention_heads
        hd = self.attention_head_size
        q = q.view(B, N, h, hd).transpose(1, 2)
        k = k.view(B, N, h, hd).transpose(1, 2)
        v = v.view(B, N, h, hd).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)
        if "need_weights" in kwargs or len(args) >= 1:
            need_w = kwargs.get("need_weights", False)
            return out, (attn if need_w else None)
        return (out,)


class TVViTWrapper(nn.Module):
    """HuggingFace-compatible facade over a torchvision ViT.

    Exposes `forward(pixel_values=...)` returning an object with `.logits`.
    """

    def __init__(self, tv_model: nn.Module):
        super().__init__()
        self.model = tv_model

    def forward(self, pixel_values=None, **kwargs):
        if pixel_values is None and "x" in kwargs:
            pixel_values = kwargs["x"]
        out = self.model(pixel_values)
        return SimpleNamespace(logits=out)


def _adapt_torchvision_vit(tv_model: nn.Module) -> TVViTWrapper:
    """Replace each block's MultiheadAttention with `TVCompatSelfAttention`."""
    encoder = tv_model.encoder
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise RuntimeError("torchvision ViT structure unexpected: no encoder.layers")
    for blk in layers:
        if hasattr(blk, "self_attention") and isinstance(blk.self_attention, nn.MultiheadAttention):
            blk.self_attention = TVCompatSelfAttention(blk.self_attention)
    return TVViTWrapper(tv_model)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _hf_load_cached():
    from transformers import ViTForImageClassification
    return ViTForImageClassification.from_pretrained(HF_NAME, local_files_only=True)


def _hf_load_download():
    from transformers import ViTForImageClassification
    return ViTForImageClassification.from_pretrained(HF_NAME)


def _tv_load():
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    return _adapt_torchvision_vit(m)


def load_model_with_fallback(hf_timeout_s: int = 120) -> Tuple[nn.Module, str, transforms.Compose]:
    """Returns (model, label, transform). Never raises on the happy path
    where torchvision weights are already downloaded; will raise only if both
    paths are unreachable AND torchvision weights aren't cached."""
    device = get_device()

    # ---- 1. HuggingFace cached ------------------------------------------------
    log.info("Loading HuggingFace ViT...")
    try:
        m = _hf_load_cached()
        m.eval().to(device)
        log.info("Model loaded successfully (HuggingFace, from cache).")
        return m, "HuggingFace ViT (cached)", HF_TRANSFORM
    except Exception as e:
        log.info(f"HF cache miss: {type(e).__name__}: {e}")

    # ---- 2. HuggingFace download with timeout --------------------------------
    log.info(f"Attempting HuggingFace download (timeout={hf_timeout_s}s)...")
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_hf_load_download)
            m = fut.result(timeout=hf_timeout_s)
        m.eval().to(device)
        log.info("Model loaded successfully (HuggingFace, downloaded).")
        return m, "HuggingFace ViT", HF_TRANSFORM
    except FutureTimeoutError:
        log.warning(f"HuggingFace download exceeded {hf_timeout_s}s timeout.")
    except Exception as e:
        log.warning(f"HuggingFace download failed: {type(e).__name__}: {e}")

    # ---- 3. Torchvision fallback ---------------------------------------------
    log.info("Falling back to torchvision model (vit_b_16, IMAGENET1K_V1)...")
    m = _tv_load()
    m.eval().to(device)
    log.info("Model loaded successfully (torchvision fallback).")
    return m, "Torchvision ViT (fallback)", TV_TRANSFORM
