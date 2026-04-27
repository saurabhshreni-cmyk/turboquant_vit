"""Render a static dashboard preview (PNG) for the README.

Reads `_bench_results.json` produced by `_bench_run.py` and emits
`assets/dashboard.png` styled to match the live Streamlit UI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Brand palette — kept aligned with visualizer.py.
BG = "#0f172a"
PANEL = "#111827"
PURPLE = "#8b5cf6"
PURPLE_LIGHT = "#a78bfa"
PURPLE_DARK = "#4c1d95"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GREEN = "#34d399"


def _style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color="#c4b5fd", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1f2937")
    ax.grid(True, color="#1f2937", linewidth=0.6, alpha=0.7)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def main() -> None:
    bench_path = ROOT / "_bench_results.json"
    if not bench_path.exists():
        raise SystemExit(f"Missing {bench_path} — run _bench_run.py first.")
    df = pd.DataFrame(json.loads(bench_path.read_text(encoding="utf-8")))

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#1f2937",
    })

    fig = plt.figure(figsize=(15, 9), facecolor=BG)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32,
                          left=0.06, right=0.97, top=0.90, bottom=0.10)

    fig.suptitle("TurboQuant × ViT — Benchmark Dashboard",
                 color="#f5f3ff", fontsize=18, fontweight="bold", y=0.97)
    fig.text(0.5, 0.93,
             "CIFAR-10 · 256 images · ViT-B/16 · RTX 3050",
             color=MUTED, fontsize=11, ha="center")

    base_acc = float(df.loc[df["method"].str.contains("Original"), "top1_acc"].iloc[0])

    # --- 1) Accuracy vs compression ----------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    d = df.sort_values("compression_ratio").reset_index(drop=True)
    ax1.axhspan(base_acc - 1, base_acc + 0.5, color=GREEN, alpha=0.12,
                label="Lossless zone (±1%)")
    ax1.axhline(base_acc, color=GREEN, linestyle=":", linewidth=1, alpha=0.7)
    ax1.plot(d["compression_ratio"], d["top1_acc"], "-o",
             color=PURPLE, markerfacecolor=PURPLE_LIGHT,
             markeredgecolor="#ede9fe", markersize=9, linewidth=2.5)
    for _, row in d.iterrows():
        offset = 1.5 if row["top1_acc"] >= base_acc - 5 else -3.0
        ax1.annotate(row["method"], (row["compression_ratio"], row["top1_acc"]),
                     xytext=(0, offset * 4), textcoords="offset points",
                     ha="center", color="#ede9fe", fontsize=8.5)
    ax1.set_xlabel("Compression ratio (×)")
    ax1.set_ylabel("Top-1 accuracy (%)")
    ax1.legend(facecolor=PANEL, edgecolor=PURPLE_DARK, labelcolor=TEXT, fontsize=9)
    _style_axes(ax1, "Accuracy vs Compression")

    # --- 2) KV memory bar with savings labels ------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(df["method"], df["kv_memory_mb"], color=PURPLE,
                   edgecolor="#c4b5fd", linewidth=1)
    for bar, (_, row) in zip(bars, df.iterrows()):
        h = bar.get_height()
        saved = row["memory_saved_pct"]
        label = f"{h:.1f} MB"
        if abs(saved) > 1e-3:
            label += f"\n−{saved:.0f}% ({row['compression_ratio']:.1f}×)"
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 6, label,
                 ha="center", va="bottom", color=TEXT, fontsize=8.5)
    ax2.set_ylabel("KV memory (MB)")
    ax2.set_ylim(0, max(df["kv_memory_mb"]) * 1.35)
    plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")
    _style_axes(ax2, "KV cache memory per method")

    # --- 3) Latency grouped bars -------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(df))
    w = 0.35
    ax3.bar(x - w / 2, df["latency_ms"], w, color=PURPLE_DARK,
            edgecolor=PURPLE, label="Cold mean")
    ax3.bar(x + w / 2, df["latency_warm_ms"], w, color=PURPLE_LIGHT,
            edgecolor="#ede9fe", label="Warm steady-state")
    for i, v in enumerate(df["latency_warm_ms"]):
        ax3.text(i + w / 2, v + 0.8, f"{v:.1f}", ha="center",
                 color=TEXT, fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(df["method"], rotation=15, ha="right")
    ax3.set_ylabel("Latency / image (ms)")
    ax3.legend(facecolor=PANEL, edgecolor=PURPLE_DARK, labelcolor=TEXT, fontsize=9)
    _style_axes(ax3, "Inference latency")

    # --- 4) Sweet-spot callout ---------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis("off")
    sweet = df[df["method"] == "TurboQuant-3bit"]
    if len(sweet):
        s = sweet.iloc[0]
        text = (f"** Sweet spot **\n\n"
                f"TurboQuant 3-bit\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"Compression  {s['compression_ratio']:.2f}×\n"
                f"Memory saved {s['memory_saved_pct']:.1f}%\n"
                f"Top-1 acc    {s['top1_acc']:.2f}%\n"
                f"Δ vs FP32    {s['top1_acc'] - base_acc:+.2f}%")
        ax4.text(0.5, 0.5, text, ha="center", va="center",
                 color=TEXT, fontsize=11, family="monospace",
                 bbox=dict(boxstyle="round,pad=1.2", facecolor=PANEL,
                           edgecolor=PURPLE, linewidth=2))

    # --- 5) Bits sweep -----------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    turbo = df[df["method"].str.contains("TurboQuant")].sort_values("bits")
    if len(turbo) >= 2:
        ax5.plot(turbo["bits"], turbo["top1_acc"], "-o",
                 color=PURPLE, markerfacecolor=PURPLE_LIGHT,
                 markeredgecolor="#ede9fe", markersize=11, linewidth=2.5)
        if 3 in turbo["bits"].values:
            r3 = turbo[turbo["bits"] == 3].iloc[0]
            ax5.scatter([3], [r3["top1_acc"]], s=320, facecolors="none",
                        edgecolors=GREEN, linewidths=2.5)
            ax5.annotate("sweet spot",
                         (3, r3["top1_acc"]),
                         xytext=(20, 15), textcoords="offset points",
                         color=GREEN, fontsize=10, fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
    ax5.set_xlabel("Bits per dimension")
    ax5.set_ylabel("Top-1 accuracy (%)")
    ax5.set_xticks(sorted(turbo["bits"].unique()) if len(turbo) else [])
    _style_axes(ax5, "Bits vs Accuracy (TurboQuant)")

    # --- 6) Compression-ratio "gauge" --------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    best = df.loc[df["compression_ratio"].idxmax()]
    # exclude QJL from "best meaningful" if its accuracy is broken
    meaningful = df[df["top1_acc"] >= base_acc - 2]
    if len(meaningful):
        best_m = meaningful.loc[meaningful["compression_ratio"].idxmax()]
    else:
        best_m = best
    text = (f"Best lossless compression\n\n"
            f"{best_m['method']}\n\n"
            f"{best_m['compression_ratio']:.2f}×\n\n"
            f"{best_m['memory_saved_pct']:.1f}% KV memory saved\n"
            f"with {best_m['top1_acc']:.2f}% top-1\n"
            f"({best_m['top1_acc'] - base_acc:+.2f}% vs FP32)")
    ax6.text(0.5, 0.5, text, ha="center", va="center", color=TEXT,
             fontsize=12, family="DejaVu Sans",
             bbox=dict(boxstyle="round,pad=1.2", facecolor=PURPLE_DARK,
                       edgecolor=PURPLE, linewidth=2))

    out = ROOT / "assets" / "dashboard.png"
    fig.savefig(out, facecolor=BG, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
