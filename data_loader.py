"""CIFAR-10 loading + ViT preprocessing.

Design goals:
  * Fast UI: `get_sample_images` must return in <1s on a warm cache and never
    block the Streamlit event loop on a cold cache.
  * Correct preprocessing: pluggable transform (HF or torchvision-style),
    settable from the model loader.
  * Lazy downloads: only download the full 170 MB tarball when explicitly
    needed by the benchmark loader.
"""
from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


log = logging.getLogger("turboquant_vit.data_loader")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Pluggable preprocessing (default = HuggingFace-style normalization)
# ---------------------------------------------------------------------------
_TRANSFORM: transforms.Compose = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def set_transform(t: transforms.Compose) -> None:
    """Override the preprocessing transform (call once after model load)."""
    global _TRANSFORM
    _TRANSFORM = t


def _torchvision_transform() -> transforms.Compose:
    return _TRANSFORM


# ---------------------------------------------------------------------------
# CIFAR-10 helpers
# ---------------------------------------------------------------------------
DATA_ROOT = "./data"
CIFAR_DIR = Path(DATA_ROOT) / "cifar-10-batches-py"


def _cifar_already_downloaded() -> bool:
    """torchvision's CIFAR10 extracts to ./data/cifar-10-batches-py/. Treat
    presence of the directory + at least one batch file as 'already downloaded'.
    """
    if not CIFAR_DIR.is_dir():
        return False
    for fname in ("data_batch_1", "test_batch"):
        if (CIFAR_DIR / fname).exists():
            return True
    return False


def get_cifar10(root: str = DATA_ROOT, train: bool = False, download: bool = True):
    """Returns a torchvision CIFAR10 dataset. Forces `download=False` if the
    extracted directory is already on disk to avoid re-checking the network."""
    from torchvision.datasets import CIFAR10
    if _cifar_already_downloaded():
        download = False
    return CIFAR10(root=root, train=train, download=download,
                   transform=_torchvision_transform())


def get_test_loader(batch_size: int = 16, num_samples: int = 1000,
                     num_workers: int = 0):
    log.info(f"Building CIFAR-10 test loader (n={num_samples}, bs={batch_size}) ...")
    ds = get_cifar10(train=False, download=True)
    if num_samples and num_samples < len(ds):
        ds = Subset(ds, list(range(num_samples)))
    log.info("Test loader ready.")
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers)


# ---------------------------------------------------------------------------
# Sample-image loading for the UI (fast path + synthetic fallback)
# ---------------------------------------------------------------------------
def _synthetic_samples(n: int = 10) -> List[Tuple[Image.Image, int, str]]:
    """Generate `n` 32x32 placeholder images, one per CIFAR-10 class, with the
    class name overlaid. Used when CIFAR-10 isn't cached and downloading would
    block the UI."""
    from utils import CIFAR10_CLASSES
    palette = [
        (139, 92, 246), (37, 99, 235), (16, 185, 129), (245, 158, 11),
        (239, 68, 68), (236, 72, 153), (14, 165, 233), (132, 204, 22),
        (168, 85, 247), (249, 115, 22),
    ]
    out = []
    for i in range(min(n, 10)):
        color = palette[i % len(palette)]
        img = Image.new("RGB", (96, 96), color)
        draw = ImageDraw.Draw(img)
        name = CIFAR10_CLASSES[i]
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        draw.text((6, 38), name, fill=(255, 255, 255), font=font)
        out.append((img, i, name))
    return out


def _load_n_from_cifar(n: int) -> List[Tuple[Image.Image, int, str]]:
    from torchvision.datasets import CIFAR10
    from utils import CIFAR10_CLASSES
    raw = CIFAR10(root=DATA_ROOT, train=False, download=False, transform=None)
    out = []
    for i in range(min(n, len(raw))):
        img, label = raw[i]
        out.append((img.convert("RGB"), int(label), CIFAR10_CLASSES[int(label)]))
    return out


def _download_then_load(n: int) -> List[Tuple[Image.Image, int, str]]:
    from torchvision.datasets import CIFAR10
    from utils import CIFAR10_CLASSES
    log.info("CIFAR-10 not on disk — downloading (~170 MB, one-time)...")
    raw = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=None)
    out = []
    for i in range(min(n, len(raw))):
        img, label = raw[i]
        out.append((img.convert("RGB"), int(label), CIFAR10_CLASSES[int(label)]))
    return out


@lru_cache(maxsize=2)
def get_sample_images(n: int = 10, download_timeout: int = 45
                      ) -> List[Tuple[Image.Image, int, str]]:
    """Fast UI sampler.

    Order of attempts:
      1. Load directly from disk if CIFAR-10 was already extracted.
      2. Otherwise, attempt download in a thread with `download_timeout`
         seconds wall-clock cap.
      3. On timeout / failure, return synthetic placeholder images so the UI
         remains usable.

    Cached via `lru_cache` so subsequent calls in the same process are O(1).
    """
    log.info("Loading CIFAR-10 dataset (sample images) ...")

    if _cifar_already_downloaded():
        try:
            samples = _load_n_from_cifar(n)
            log.info(f"Dataset loaded successfully (cached, n={len(samples)}).")
            return samples
        except Exception as e:
            log.warning(f"Cached CIFAR-10 unreadable: {type(e).__name__}: {e}")

    log.info(f"Attempting CIFAR-10 download (timeout={download_timeout}s)...")
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_download_then_load, n)
            samples = fut.result(timeout=download_timeout)
        log.info(f"Dataset loaded successfully (downloaded, n={len(samples)}).")
        return samples
    except FutureTimeoutError:
        log.warning(
            f"CIFAR-10 download exceeded {download_timeout}s — using synthetic "
            "placeholder samples so the UI stays responsive."
        )
    except Exception as e:
        log.warning(f"CIFAR-10 download failed: {type(e).__name__}: {e} — "
                    "using synthetic placeholder samples.")

    samples = _synthetic_samples(n)
    log.info(f"Synthetic samples ready (n={len(samples)}).")
    return samples


def cifar_status() -> str:
    """Human-readable status string for the UI."""
    return "cached" if _cifar_already_downloaded() else "not downloaded"


# ---------------------------------------------------------------------------
# Single-image preprocessing
# ---------------------------------------------------------------------------
def preprocess_pil(img: Image.Image) -> torch.Tensor:
    """Apply the active transform to a PIL image; returns (1, 3, 224, 224)."""
    return _torchvision_transform()(img).unsqueeze(0)
