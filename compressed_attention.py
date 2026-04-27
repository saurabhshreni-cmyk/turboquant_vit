"""High-level wrapper that ties a HuggingFace ViT model to a compression
strategy. Provides a single context-manager entry point used by the app
and the evaluator.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch

from turbo_compressor import build_compressor
from vit_hook import CompressedKVAttention


@contextmanager
def compressed_vit(model, method: str = "none", bits: int = 3,
                   capture_attn: bool = False, seed: int = 0):
    """Context manager that swaps in compressed attention.

    Yields a CompressedKVAttention tracker so callers can read memory stats
    and per-layer attention maps.
    """
    method_lc = method.lower()
    enabled = method_lc not in ("none", "normal", "original")

    def factory(dim: int):
        return build_compressor(method_lc, dim=dim, bits=bits, seed=seed)

    tracker = CompressedKVAttention(
        model,
        compressor_factory=factory if enabled else None,
        compress_enabled=enabled,
        capture=False,
        capture_attn=capture_attn,
    )
    if not enabled:
        # still install so we can record memory baselines + attention
        tracker.compress_enabled = False
        tracker.compressor_factory = lambda d: None
    tracker.install()
    try:
        yield tracker
    finally:
        tracker.remove()


@torch.no_grad()
def run_inference(model, pixel_values: torch.Tensor, method: str = "none",
                  bits: int = 3, capture_attn: bool = False):
    """Run a forward pass under the chosen compression method.

    Returns: (logits, tracker)
    """
    with compressed_vit(model, method=method, bits=bits,
                        capture_attn=capture_attn) as tracker:
        out = model(pixel_values=pixel_values)
        logits = out.logits if hasattr(out, "logits") else out[0]
    return logits, tracker
