"""Utility helpers for TurboQuant-ViT."""
from __future__ import annotations

import io
import time
from contextlib import contextmanager
from typing import Tuple

import numpy as np
import torch
from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_device() -> torch.device:
    return DEVICE


def tensor_bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.nelement()


def bits_to_bytes(num_bits: int) -> float:
    return num_bits / 8.0


@contextmanager
def timer():
    start = time.perf_counter()
    out = {}
    try:
        yield out
    finally:
        out["elapsed_ms"] = (time.perf_counter() - start) * 1000.0


def pil_to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


def make_banner(width: int = 1280, height: int = 320) -> Image.Image:
    """Generate a purple-gradient banner programmatically (no external assets)."""
    img = Image.new("RGB", (width, height), (15, 23, 42))
    px = img.load()
    for y in range(height):
        for x in range(width):
            t = x / max(width - 1, 1)
            r = int(15 + (139 - 15) * t)
            g = int(23 + (92 - 23) * t)
            b = int(42 + (246 - 42) * t)
            px[x, y] = (r, g, b)
    return img


def safe_load_image(buf: bytes) -> Image.Image:
    return Image.open(io.BytesIO(buf)).convert("RGB")


def frobenius_distortion(a: torch.Tensor, b: torch.Tensor) -> float:
    # Align devices/dtypes so callers can pass a CPU reference and a CUDA
    # candidate (or vice-versa) without triggering a device-mismatch error.
    target_device = a.device if a.is_cuda else b.device
    a = a.detach().to(device=target_device, dtype=torch.float32).flatten()
    b = b.detach().to(device=target_device, dtype=torch.float32).flatten()
    denom = torch.linalg.norm(a) + 1e-12
    return float(torch.linalg.norm(a - b) / denom)


def model_device(model: torch.nn.Module) -> torch.device:
    """Infer the device of a model from its first parameter (falls back to
    the global default when the model has no parameters)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return DEVICE


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, k: int = 1) -> float:
    _, pred = logits.topk(k, dim=-1)
    correct = pred.eq(target.view(-1, 1).expand_as(pred)).any(dim=-1)
    return float(correct.float().mean().item()) * 100.0


def format_mb(num_bytes: float) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def split_shape(t: torch.Tensor) -> Tuple[int, ...]:
    return tuple(t.shape)
