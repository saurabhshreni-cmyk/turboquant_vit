"""Benchmarking + evaluation against CIFAR-10."""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from compressed_attention import compressed_vit
from utils import frobenius_distortion, get_device, model_device


# Mapping ImageNet ViT predictions back to CIFAR-10 labels.
# CIFAR-10 has 10 classes; ViT-B/16 outputs 1000 ImageNet logits.
# We map each CIFAR-10 class to a representative set of ImageNet class indices,
# then pick the CIFAR class whose summed probability is highest.
_CIFAR10_TO_IMAGENET: Dict[str, List[int]] = {
    # airplane
    "airplane": [404, 895],                # airliner, warplane
    # automobile
    "automobile": [436, 468, 511, 609, 627, 717, 751, 817],
    # bird
    "bird": list(range(7, 25)) + [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                                  91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
    # cat
    "cat": list(range(281, 294)),
    # deer
    "deer": [351, 352],
    # dog
    "dog": list(range(151, 269)),
    # frog
    "frog": [30, 31, 32],
    # horse
    "horse": [339, 340],
    # ship
    "ship": [403, 472, 510, 554, 625, 628, 814, 833, 871, 914],
    # truck
    "truck": [555, 569, 717, 864, 867, 569],
}


CIFAR_ORDER = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def imagenet_logits_to_cifar(logits: torch.Tensor) -> torch.Tensor:
    """Map (B, 1000) ImageNet logits to (B, 10) CIFAR-10 scores by summing
    softmax probability over each CIFAR class' ImageNet group."""
    probs = torch.softmax(logits, dim=-1)
    out = torch.zeros(probs.shape[0], 10, device=probs.device, dtype=probs.dtype)
    for ci, name in enumerate(CIFAR_ORDER):
        idxs = _CIFAR10_TO_IMAGENET[name]
        out[:, ci] = probs[:, idxs].sum(dim=-1)
    return out


# ---------------------------------------------------------------------------
@dataclass
class BenchmarkRow:
    method: str
    bits: int
    top1_acc: float
    top5_acc: float
    kv_memory_mb: float
    compression_ratio: float
    latency_ms: float
    latency_warm_ms: float
    attention_distortion: float
    memory_saved_pct: float = 0.0
    latency_delta_ms: float = 0.0


class ViTEvaluator:
    def __init__(self, model, device: Optional[torch.device] = None,
                 logits_to_cifar: Callable = imagenet_logits_to_cifar):
        self.model = model
        # Prefer the model's existing device when one isn't explicitly supplied,
        # so we never silently move the model between CPU and CUDA.
        self.device = device or model_device(model)
        self.model.eval().to(self.device)
        self.logits_to_cifar = logits_to_cifar

    @torch.no_grad()
    def _eval_one(self, dataloader, method: str, bits: int,
                  ref_attn: Optional[Dict[int, torch.Tensor]] = None,
                  capture_attn: bool = False) -> BenchmarkRow:
        correct1 = correct5 = total = 0
        latencies: List[float] = []
        mem_bytes: List[int] = []
        distortions: List[float] = []

        with compressed_vit(self.model, method=method, bits=bits,
                            capture_attn=capture_attn) as tracker:
            for batch_idx, (imgs, labels) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                t0 = time.perf_counter()
                out = self.model(pixel_values=imgs)
                logits = out.logits if hasattr(out, "logits") else out[0]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000.0 / imgs.shape[0])

                cifar_scores = self.logits_to_cifar(logits)
                top1 = cifar_scores.argmax(dim=-1)
                _, top5 = cifar_scores.topk(min(5, cifar_scores.shape[-1]), dim=-1)

                correct1 += (top1 == labels).sum().item()
                correct5 += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()
                total += labels.shape[0]

                mem_bytes.append(tracker.total_compressed_bytes())

                if capture_attn and ref_attn is not None and batch_idx == 0:
                    cur = tracker.per_layer_attention()
                    for li, cur_attn in cur.items():
                        ref = ref_attn.get(li)
                        if ref is None:
                            continue
                        # frobenius_distortion now handles device alignment,
                        # but be explicit so we can also reuse identical shapes.
                        if ref.shape != cur_attn.shape:
                            continue
                        distortions.append(frobenius_distortion(ref, cur_attn))

                tracker.reset()

        top1_acc = 100.0 * correct1 / max(total, 1)
        top5_acc = 100.0 * correct5 / max(total, 1)
        kv_mb = float(np.mean(mem_bytes)) / (1024 * 1024) if mem_bytes else 0.0
        latency_ms = float(np.mean(latencies)) if latencies else 0.0
        # Drop the first batch to avoid the cold-cache + CUDA warmup spike.
        warm = latencies[1:] if len(latencies) > 1 else latencies
        latency_warm_ms = float(np.mean(warm)) if warm else latency_ms
        # baseline reference for compression ratio
        if method.lower() in ("none", "normal", "original"):
            cratio = 1.0
        else:
            from turbo_compressor import build_compressor
            # head_dim for ViT-B/16 is 64
            c = build_compressor(method, dim=64, bits=bits)
            cratio = c.get_stats().get("compression_ratio", 1.0)
        distortion = float(np.mean(distortions)) if distortions else 0.0

        return BenchmarkRow(
            method=method, bits=bits,
            top1_acc=top1_acc, top5_acc=top5_acc,
            kv_memory_mb=kv_mb, compression_ratio=cratio,
            latency_ms=latency_ms, latency_warm_ms=latency_warm_ms,
            attention_distortion=distortion,
        )

    def run_full_benchmark(self, dataloader, methods=None,
                           progress_cb: Optional[Callable[[float, str], None]] = None) -> pd.DataFrame:
        methods = methods or [
            ("Original", "none", 32),
            ("QJL-1bit", "qjl", 1),
            ("PolarQuant-3bit", "polarquant", 3),
            ("TurboQuant-3bit", "turboquant", 3),
            ("TurboQuant-2bit", "turboquant", 2),
        ]

        # Reference attention from a single batch under "none"
        ref_attn = self._capture_reference_attention(dataloader)

        rows: List[BenchmarkRow] = []
        for i, (label, method, bits) in enumerate(methods):
            if progress_cb:
                progress_cb(i / len(methods), f"Evaluating {label}...")
            row = self._eval_one(dataloader, method=method, bits=bits,
                                 ref_attn=ref_attn, capture_attn=True)
            row.method = label
            rows.append(row)
        if progress_cb:
            progress_cb(1.0, "Done")

        df = pd.DataFrame([asdict(r) for r in rows])
        # Derive comparison metrics against the "Original" baseline so the UI
        # can render memory-saved % and latency delta consistently.
        baseline_mask = df["method"].str.contains("Original", case=False, na=False)
        if baseline_mask.any():
            base_mem = float(df.loc[baseline_mask, "kv_memory_mb"].iloc[0])
            base_lat = float(df.loc[baseline_mask, "latency_warm_ms"].iloc[0])
            df["memory_saved_pct"] = (1.0 - df["kv_memory_mb"] / max(base_mem, 1e-9)) * 100.0
            df["latency_delta_ms"] = df["latency_warm_ms"] - base_lat
        return df

    @torch.no_grad()
    def _capture_reference_attention(self, dataloader) -> Dict[int, torch.Tensor]:
        for imgs, _ in dataloader:
            imgs = imgs.to(self.device)
            with compressed_vit(self.model, method="none", capture_attn=True) as tracker:
                _ = self.model(pixel_values=imgs)
                # Keep reference attention on the model's device so downstream
                # comparisons against compressed-method attention don't trigger
                # CPU/CUDA mismatches inside torch.linalg.norm.
                attn = {k: v.detach().to(self.device)
                        for k, v in tracker.per_layer_attention().items()}
            return attn
        return {}

    @torch.no_grad()
    def compare_attention_maps(self, pixel_values: torch.Tensor,
                               method: str = "turboquant", bits: int = 3,
                               layer_idx: int = -1, head_idx: int = 0):
        pixel_values = pixel_values.to(self.device)
        # original
        with compressed_vit(self.model, method="none", capture_attn=True) as t0:
            self.model(pixel_values=pixel_values)
            attn0 = t0.per_layer_attention()
        with compressed_vit(self.model, method=method, bits=bits, capture_attn=True) as t1:
            self.model(pixel_values=pixel_values)
            attn1 = t1.per_layer_attention()
        # Only compare layers captured by both passes; clamp head index too.
        common = sorted(set(attn0.keys()) & set(attn1.keys()))
        if not common:
            raise RuntimeError("No overlapping attention layers between methods.")
        li = common[layer_idx] if 0 <= layer_idx < len(common) else common[-1]
        head_count = attn0[li].shape[1]
        head_idx = max(0, min(head_idx, head_count - 1))
        a0 = attn0[li][0, head_idx]
        a1 = attn1[li][0, head_idx]
        return a0.cpu(), a1.cpu()
