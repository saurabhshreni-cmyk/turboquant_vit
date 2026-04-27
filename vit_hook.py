"""Hook + monkey-patch utilities for intercepting and compressing K/V tensors
in HuggingFace ViTSelfAttention layers.

We do not modify model weights. We replace the *behavior* of a forward pass,
optionally compressing K/V before computing attention scores.

Reference: transformers.models.vit.modeling_vit.ViTSelfAttention
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


_logger = logging.getLogger(__name__)
# Set TURBOQUANT_PROFILE=1 to print per-layer decompression / attention timings.
_PROFILE = os.environ.get("TURBOQUANT_PROFILE", "0") == "1"


# ---------------------------------------------------------------------------
# Helpers to find ViT self-attention modules
# ---------------------------------------------------------------------------
def _find_attention_modules(model) -> List[torch.nn.Module]:
    mods = []
    for name, m in model.named_modules():
        cls = m.__class__.__name__
        # HuggingFace ViTSelfAttention / ViTSdpaSelfAttention etc.
        if cls.endswith("SelfAttention") and hasattr(m, "query") and hasattr(m, "key") and hasattr(m, "value"):
            mods.append(m)
    return mods


def _shape_qkv(t: torch.Tensor, num_heads: int) -> torch.Tensor:
    # (B, N, C) -> (B, num_heads, N, head_dim)
    B, N, C = t.shape
    head_dim = C // num_heads
    return t.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def _unshape(t: torch.Tensor) -> torch.Tensor:
    # (B, h, N, hd) -> (B, N, h*hd)
    B, h, N, hd = t.shape
    return t.permute(0, 2, 1, 3).contiguous().view(B, N, h * hd)


# ---------------------------------------------------------------------------
# Custom forward applied via monkey-patch
# ---------------------------------------------------------------------------
def _make_compressed_forward(orig_module, compressor_factory, layer_idx: int,
                             tracker: "CompressedKVAttention"):
    """
    Build a closure that re-implements ViTSelfAttention.forward but compresses
    K and V before computing attention.
    """
    num_heads = orig_module.num_attention_heads
    head_dim = orig_module.attention_head_size
    all_head_size = orig_module.all_head_size

    # Build per-(layer, vector-kind) compressors lazily on first call
    compressors: Dict[str, object] = {}

    def _get_compressor(kind: str, dim: int, device):
        key = f"{kind}-{dim}"
        if key not in compressors:
            c = compressor_factory(dim)
            try:
                c.to(device)
            except Exception:
                pass
            compressors[key] = c
        return compressors[key]

    def forward(*args, **kwargs):
        # Dual-protocol: HF passes (hidden_states, head_mask=..., output_attentions=...);
        # torchvision passes (query, key, value, ..., need_weights=...).
        tv_style = (
            "need_weights" in kwargs
            or (len(args) >= 3 and all(torch.is_tensor(a) for a in args[:3]))
        )
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        head_mask = kwargs.get("head_mask", None)
        output_attentions = kwargs.get("output_attentions", False)
        need_weights = kwargs.get("need_weights", False)

        # Q, K, V projections (use the module's original linears)
        q = orig_module.query(hidden_states)
        k = orig_module.key(hidden_states)
        v = orig_module.value(hidden_states)

        q = _shape_qkv(q, num_heads)
        k = _shape_qkv(k, num_heads)
        v = _shape_qkv(v, num_heads)

        # Capture originals (small footprint — detach + cpu only on demand)
        if tracker.capture:
            tracker._captured.setdefault(layer_idx, {})
            tracker._captured[layer_idx]["K"] = k.detach()
            tracker._captured[layer_idx]["V"] = v.detach()

        # Compress / decompress K and V along last (head_dim) axis.
        # Done ONCE per layer per forward pass — k_use / v_use are then reused
        # across all heads and across the full attention computation.
        if tracker.compress_enabled:
            t_decomp = time.perf_counter() if _PROFILE else 0.0
            ck = _get_compressor("K", head_dim, k.device)
            cv = _get_compressor("V", head_dim, v.device)
            packed_k = ck.compress(k)
            packed_v = cv.compress(v)
            k_use = ck.decompress(packed_k).to(k.dtype)
            v_use = cv.decompress(packed_v).to(v.dtype)
            if _PROFILE:
                if k.is_cuda:
                    torch.cuda.synchronize()
                _logger.info("layer=%d decompress=%.2fms",
                             layer_idx, (time.perf_counter() - t_decomp) * 1000.0)
            # accounting
            num_vecs = k.shape[0] * k.shape[1] * k.shape[2]
            tracker._mem_bytes[layer_idx] = (
                ck.memory_bytes(num_vecs) + cv.memory_bytes(num_vecs)
            )
            tracker._orig_mem_bytes[layer_idx] = 2 * num_vecs * head_dim * 4
        else:
            k_use, v_use = k, v
            num_vecs = k.shape[0] * k.shape[1] * k.shape[2]
            tracker._mem_bytes[layer_idx] = 2 * num_vecs * head_dim * 4
            tracker._orig_mem_bytes[layer_idx] = 2 * num_vecs * head_dim * 4

        # Standard scaled dot-product attention. k_use / v_use are already
        # reshaped to (B, num_heads, N, head_dim) and matmul is fully vectorized
        # across heads — there is no Python-level loop over heads.
        t_attn = time.perf_counter() if _PROFILE else 0.0
        attn_scores = torch.matmul(q, k_use.transpose(-1, -2)) / math.sqrt(head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask
        context = torch.matmul(attn_probs, v_use)
        if _PROFILE:
            if k.is_cuda:
                torch.cuda.synchronize()
            _logger.info("layer=%d attn=%.2fms",
                         layer_idx, (time.perf_counter() - t_attn) * 1000.0)
        out = _unshape(context)

        # torchvision's EncoderBlock expects out_proj already applied;
        # HuggingFace's ViTAttention wraps a separate dense output layer.
        if hasattr(orig_module, "out_proj"):
            out = orig_module.out_proj(out)

        if tracker.capture_attn:
            tracker._attn[layer_idx] = attn_probs.detach()

        if tv_style:
            return (out, attn_probs if need_weights else None)
        # HuggingFace path: newer transformers always unpacks (out, attn);
        # older versions tolerate any tuple length. Return 2-tuple unconditionally.
        return (out, attn_probs if output_attentions else None)

    return forward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class KVCaptureHook:
    """Capture-only mode (no compression). Stores K, V tensors per layer."""

    def __init__(self, model):
        self.model = model
        self._captured: Dict[int, Dict[str, torch.Tensor]] = {}
        self._handles = []

    def register_hooks(self):
        # Use the same monkey-patch path with compression disabled.
        self._tracker = CompressedKVAttention(self.model, compressor_factory=None,
                                              compress_enabled=False, capture=True)
        self._tracker.install()
        return self

    def remove_hooks(self):
        if hasattr(self, "_tracker"):
            self._tracker.remove()

    def get_captured_kv(self) -> Dict[int, Dict[str, torch.Tensor]]:
        return self._tracker._captured

    def get_memory_usage(self) -> int:
        return sum(self._tracker._orig_mem_bytes.values())


class CompressedKVAttention:
    """Monkey-patches ViT self-attention layers to use compressed K/V.

    Args:
        model: HuggingFace ViTForImageClassification (or any ViT with HF self-attn modules)
        compressor_factory: callable(dim:int) -> compressor with .compress/.decompress
        compress_enabled:   if False, runs original (uncompressed) attention but still tracks memory
        capture:            if True, stores K,V tensors per layer
        capture_attn:       if True, stores attention probs per layer
    """

    def __init__(self, model, compressor_factory=None, compress_enabled: bool = True,
                 capture: bool = False, capture_attn: bool = False):
        self.model = model
        self.compressor_factory = compressor_factory
        self.compress_enabled = bool(compress_enabled and compressor_factory is not None)
        self.capture = bool(capture)
        self.capture_attn = bool(capture_attn)

        self._captured: Dict[int, Dict[str, torch.Tensor]] = {}
        self._attn: Dict[int, torch.Tensor] = {}
        self._mem_bytes: Dict[int, int] = {}
        self._orig_mem_bytes: Dict[int, int] = {}
        self._patched: List = []  # list of (module, original_forward)

    # -- context manager friendliness ---------------------------------------
    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()

    # -----------------------------------------------------------------------
    def install(self):
        attn_modules = _find_attention_modules(self.model)
        if not attn_modules:
            raise RuntimeError("No ViT self-attention modules found on model.")
        for idx, m in enumerate(attn_modules):
            orig_forward = m.forward
            new_forward = _make_compressed_forward(m, self.compressor_factory or (lambda d: None),
                                                   layer_idx=idx, tracker=self)
            m.forward = new_forward  # type: ignore
            self._patched.append((m, orig_forward))
        return self

    def remove(self):
        for m, orig in self._patched:
            m.forward = orig
        self._patched.clear()

    # -- accessors ----------------------------------------------------------
    def total_compressed_bytes(self) -> int:
        return sum(self._mem_bytes.values())

    def total_original_bytes(self) -> int:
        return sum(self._orig_mem_bytes.values())

    def per_layer_attention(self) -> Dict[int, torch.Tensor]:
        return self._attn

    def reset(self):
        self._captured.clear()
        self._attn.clear()
        self._mem_bytes.clear()
        self._orig_mem_bytes.clear()
