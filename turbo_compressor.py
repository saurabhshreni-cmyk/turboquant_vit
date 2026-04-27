"""
TurboQuant compressors.

Implements three compression schemes from scratch (numpy + torch):
  - QJLCompressor:        1-bit quantized Johnson-Lindenstrauss sketch.
  - PolarQuantCompressor: random-rotation + Lloyd-Max scalar quantization.
  - TurboQuantCompressor: PolarQuant (MSE-optimal) + 1-bit QJL on residual
                          for unbiased inner-product correction.

Educational implementation — every step is in plain numpy/torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# QJL — Quantized Johnson-Lindenstrauss
# ---------------------------------------------------------------------------
class QJLCompressor:
    """1-bit quantized JL sketch with unbiased inner-product estimator.

    Project x in R^d through a random Gaussian R in R^{k x d} and keep only
    the sign:    q(x) = sign(R x).
    Inner-product estimator:
        <x, y> ≈ (||x|| ||y|| * pi / (2k)) * <q(x), q(y)>     (sign-only form)
    For decompression we use the unbiased reconstruction
        x_hat = (sqrt(pi/2) / sqrt(k)) * R^T q(x)             (analogous form).
    """

    def __init__(self, input_dim: int, compressed_dim: int, seed: int = 0):
        self.input_dim = int(input_dim)
        self.compressed_dim = int(compressed_dim)
        rng = np.random.default_rng(seed)
        # R: (k, d), unit-variance Gaussian
        R = rng.standard_normal((self.compressed_dim, self.input_dim)).astype(np.float32)
        self.R = torch.from_numpy(R)
        self._scale = math.sqrt(math.pi / 2.0) / math.sqrt(self.compressed_dim)

    def to(self, device):
        self.R = self.R.to(device)
        return self

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d) float -> q: (..., k) int8 in {-1, +1}."""
        R = self.R.to(x.device, dtype=x.dtype)
        z = torch.matmul(x, R.t())                  # (..., k)
        q = torch.sign(z)
        q[q == 0] = 1.0
        return q.to(torch.int8)

    def decompress(self, q: torch.Tensor) -> torch.Tensor:
        R = self.R.to(q.device, dtype=torch.float32)
        return self._scale * torch.matmul(q.float(), R)

    def estimate_inner_product(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        ip = (q1.float() * q2.float()).sum(dim=-1)
        return (math.pi / (2.0 * self.compressed_dim)) * ip

    def memory_bytes(self, num_vectors: int) -> int:
        # Each compressed vector stored as k bits.
        return int(num_vectors * self.compressed_dim / 8)


# ---------------------------------------------------------------------------
# PolarQuant — random-rotation + Lloyd-Max scalar quantization
# ---------------------------------------------------------------------------
class PolarQuantCompressor:
    """PolarQuant: rotate x by a random orthogonal matrix Pi, normalize on the
    sphere so coordinates ~ Beta-distributed, then scalar-quantize each
    coordinate with a Lloyd-Max codebook fitted to that distribution.

    Stored payload per vector:
        - norm  (float32)
        - b-bit codeword indices (one per dimension)
    """

    def __init__(self, dim: int, bits: int = 3, seed: int = 0):
        assert bits in (1, 2, 3, 4, 5, 6, 8), "bits must be in {1,2,3,4,5,6,8}"
        self.dim = int(dim)
        self.bits = int(bits)
        self.n_levels = 1 << self.bits
        self.rotation = self._build_rotation_matrix(self.dim, seed)
        self.codebook = self._lloyd_max_codebook(self.n_levels, dim=self.dim)
        # codebook: torch tensor (n_levels,)

    # -- construction helpers ------------------------------------------------
    def _build_rotation_matrix(self, dim: int, seed: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((dim, dim)).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        return torch.from_numpy(Q.astype(np.float32))

    def _lloyd_max_codebook(self, n_levels: int, dim: int, n_samples: int = 200_000,
                             n_iter: int = 60, seed: int = 1) -> torch.Tensor:
        """Fit Lloyd-Max centroids to the marginal distribution of a coordinate
        of a uniformly-random unit vector in R^dim.

        That distribution is symmetric on [-1, 1] with density proportional to
        (1 - x^2)^((dim-3)/2)  (related to a Beta((d-1)/2, (d-1)/2) on [-1,1]).
        We approximate with Monte-Carlo samples.
        """
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((n_samples, dim)).astype(np.float64)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        samples = v[:, 0]   # any coordinate — distribution is symmetric

        # init: equal-mass quantiles
        qs = np.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels)
        centroids = np.quantile(samples, qs)

        for _ in range(n_iter):
            # voronoi boundaries are midpoints of sorted centroids
            sc = np.sort(centroids)
            boundaries = (sc[:-1] + sc[1:]) / 2.0
            assign = np.searchsorted(boundaries, samples)
            new_centroids = np.zeros_like(sc)
            for k in range(n_levels):
                mask = assign == k
                if mask.any():
                    new_centroids[k] = samples[mask].mean()
                else:
                    new_centroids[k] = sc[k]
            if np.allclose(new_centroids, sc, atol=1e-6):
                centroids = new_centroids
                break
            centroids = new_centroids

        return torch.from_numpy(np.sort(centroids).astype(np.float32))

    # -- API -----------------------------------------------------------------
    def to(self, device):
        self.rotation = self.rotation.to(device)
        self.codebook = self.codebook.to(device)
        return self

    def compress(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (..., d) -> dict with 'indices' and 'norm'."""
        rot = self.rotation.to(x.device, dtype=x.dtype)
        y = torch.matmul(x, rot.t())                            # (..., d)
        norm = torch.linalg.norm(y, dim=-1, keepdim=True) + 1e-12
        u = y / norm                                            # on sphere
        cb = self.codebook.to(x.device, dtype=x.dtype)          # (L,)
        # nearest-neighbor scalar quantization
        d = (u.unsqueeze(-1) - cb).abs()                        # (..., d, L)
        idx = d.argmin(dim=-1).to(torch.int32)                  # (..., d)
        return {"indices": idx, "norm": norm.squeeze(-1)}

    def decompress(self, packed: Dict[str, torch.Tensor]) -> torch.Tensor:
        idx = packed["indices"]
        norm = packed["norm"].unsqueeze(-1)
        cb = self.codebook.to(idx.device)
        u = cb[idx.long()]
        y = u * norm
        rot = self.rotation.to(y.device, dtype=y.dtype)
        x_hat = torch.matmul(y, rot)                            # inverse rot = rot.T applied as y @ rot
        return x_hat

    def compression_ratio(self, original_bits: int = 32) -> float:
        # per dim: bits + small per-vector norm overhead amortized
        return original_bits / float(self.bits)

    def memory_bytes(self, num_vectors: int) -> int:
        per_vec_bits = self.dim * self.bits + 32  # +norm
        return int(num_vectors * per_vec_bits / 8)


# ---------------------------------------------------------------------------
# TurboQuant — PolarQuant + 1-bit QJL residual
# ---------------------------------------------------------------------------
@dataclass
class TurboPacked:
    polar: Dict[str, torch.Tensor]
    qjl_q: torch.Tensor          # int8 +/-1
    residual_norm: torch.Tensor  # (...,)
    qjl_dim: int


class TurboQuantCompressor:
    """Two-stage compression:

      1. PolarQuant with `bits` bits/dim    — MSE-optimal main pass.
      2. Compute residual r = x - x_hat.
      3. 1-bit QJL sketch of r              — bias correction for inner products.

    Effective rate: bits + (qjl_ratio) bits per dim, default ~3 + 0.5 ≈ 3.5.
    """

    def __init__(self, dim: int, bits: int = 3, qjl_ratio: float = 0.5, seed: int = 0):
        self.dim = int(dim)
        self.bits = int(bits)
        self.qjl_ratio = float(qjl_ratio)
        self.qjl_dim = max(1, int(round(qjl_ratio * dim)))
        self.polar = PolarQuantCompressor(dim, bits=bits, seed=seed)
        self.qjl = QJLCompressor(dim, self.qjl_dim, seed=seed + 1)

    def to(self, device):
        self.polar.to(device)
        self.qjl.to(device)
        return self

    def compress(self, x: torch.Tensor) -> TurboPacked:
        polar = self.polar.compress(x)
        x_hat = self.polar.decompress(polar)
        r = x - x_hat
        r_norm = torch.linalg.norm(r, dim=-1)
        # normalize residual then sign-sketch (so reconstruction can be rescaled)
        r_unit = r / (r_norm.unsqueeze(-1) + 1e-12)
        q = self.qjl.compress(r_unit)
        return TurboPacked(polar=polar, qjl_q=q, residual_norm=r_norm, qjl_dim=self.qjl_dim)

    def decompress(self, packed: TurboPacked) -> torch.Tensor:
        x_hat = self.polar.decompress(packed.polar)
        r_unit_hat = self.qjl.decompress(packed.qjl_q)
        # rescale to roughly unit norm: QJL reconstruction has expected norm ~1
        r_unit_hat_norm = torch.linalg.norm(r_unit_hat, dim=-1, keepdim=True) + 1e-12
        r_hat = (r_unit_hat / r_unit_hat_norm) * packed.residual_norm.unsqueeze(-1)
        return x_hat + r_hat

    def compression_ratio(self, original_bits: int = 32) -> float:
        effective = self.bits + self.qjl_ratio
        return original_bits / effective

    def memory_bytes(self, num_vectors: int) -> int:
        per_vec_bits = self.dim * self.bits + self.qjl_dim + 32 + 32  # norm + r_norm
        return int(num_vectors * per_vec_bits / 8)

    def get_stats(self) -> dict:
        eff_bits = self.bits + self.qjl_ratio
        return {
            "dim": self.dim,
            "polar_bits": self.bits,
            "qjl_dim": self.qjl_dim,
            "effective_bits_per_dim": eff_bits,
            "compression_ratio": 32.0 / eff_bits,
            "memory_saved_percent": (1.0 - eff_bits / 32.0) * 100.0,
        }


# ---------------------------------------------------------------------------
# Convenience factory used by the rest of the project
# ---------------------------------------------------------------------------
# Process-wide cache. Building a PolarQuant codebook (Lloyd-Max over 200k
# samples × 60 iterations) costs ~1-2s per layer; without this cache every
# forward pass would rebuild 24 compressors (12 layers × {K, V}) and incur
# ~30-40s of latency. Compressors are stateless across calls so sharing is safe.
_COMPRESSOR_CACHE: Dict[Tuple[str, int, int, int], object] = {}


def build_compressor(method: str, dim: int, bits: int = 3, seed: int = 0):
    """Returns an object with .compress(x) -> packed and .decompress(packed) -> x_hat,
    a stats dict via .get_stats() (or default), and .memory_bytes(num_vectors).

    Compressors are cached globally per (method, dim, bits, seed) so that
    expensive setup (Lloyd-Max codebook fitting, Gaussian projection matrices,
    QR rotation) runs only once per process.
    """
    method = method.lower()
    key = (method, int(dim), int(bits), int(seed))
    cached = _COMPRESSOR_CACHE.get(key)
    if cached is not None:
        return cached

    if method in ("none", "normal", "original"):
        c: object = IdentityCompressor(dim)
    elif method == "qjl":
        k = max(1, int(round(bits * dim)))
        c = QJLWrapper(QJLCompressor(dim, k, seed=seed), dim=dim, k=k)
    elif method == "polarquant":
        c = PolarWrapper(PolarQuantCompressor(dim, bits=bits, seed=seed))
    elif method == "turboquant":
        c = TurboWrapper(TurboQuantCompressor(dim, bits=bits, seed=seed))
    else:
        raise ValueError(f"unknown method: {method}")

    _COMPRESSOR_CACHE[key] = c
    return c


class IdentityCompressor:
    def __init__(self, dim: int):
        self.dim = dim

    def to(self, device):
        return self

    def compress(self, x):
        return x

    def decompress(self, packed):
        return packed

    def memory_bytes(self, num_vectors: int) -> int:
        return num_vectors * self.dim * 4  # float32

    def get_stats(self):
        return {"compression_ratio": 1.0, "effective_bits_per_dim": 32.0,
                "memory_saved_percent": 0.0}


class QJLWrapper:
    def __init__(self, qjl: QJLCompressor, dim: int, k: int):
        self.qjl = qjl
        self.dim = dim
        self.k = k

    def to(self, device):
        self.qjl.to(device); return self

    def compress(self, x):
        # store norm so we can rescale on decompression
        norm = torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-12
        u = x / norm
        q = self.qjl.compress(u)
        return {"q": q, "norm": norm.squeeze(-1)}

    def decompress(self, packed):
        u_hat = self.qjl.decompress(packed["q"])
        u_hat = u_hat / (torch.linalg.norm(u_hat, dim=-1, keepdim=True) + 1e-12)
        return u_hat * packed["norm"].unsqueeze(-1)

    def memory_bytes(self, num_vectors: int) -> int:
        return int(num_vectors * (self.k / 8 + 4))

    def get_stats(self):
        bits_per_dim = self.k / max(self.dim, 1)
        return {"compression_ratio": 32.0 / max(bits_per_dim, 1e-9),
                "effective_bits_per_dim": bits_per_dim,
                "memory_saved_percent": (1.0 - bits_per_dim / 32.0) * 100.0}


class PolarWrapper:
    def __init__(self, p: PolarQuantCompressor):
        self.p = p
        self.dim = p.dim

    def to(self, device):
        self.p.to(device); return self

    def compress(self, x):
        return self.p.compress(x)

    def decompress(self, packed):
        return self.p.decompress(packed)

    def memory_bytes(self, num_vectors: int) -> int:
        return self.p.memory_bytes(num_vectors)

    def get_stats(self):
        bits = self.p.bits
        return {"compression_ratio": 32.0 / bits,
                "effective_bits_per_dim": float(bits),
                "memory_saved_percent": (1.0 - bits / 32.0) * 100.0}


class TurboWrapper:
    def __init__(self, t: TurboQuantCompressor):
        self.t = t
        self.dim = t.dim

    def to(self, device):
        self.t.to(device); return self

    def compress(self, x):
        return self.t.compress(x)

    def decompress(self, packed):
        return self.t.decompress(packed)

    def memory_bytes(self, num_vectors: int) -> int:
        return self.t.memory_bytes(num_vectors)

    def get_stats(self):
        return self.t.get_stats()
