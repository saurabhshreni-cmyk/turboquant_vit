"""Plotly visualizations for the Streamlit app.

All charts share a consistent dark + purple palette and use rich hover
tooltips so the dashboard is self-explanatory without supplementary text.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image


# Purple palette — light → dark — kept in one place so every chart stays on-brand.
PURPLE_50 = "#ede9fe"
PURPLE_300 = "#c4b5fd"
PURPLE_400 = "#a78bfa"
PURPLE_500 = "#8b5cf6"
PURPLE_600 = "#7c3aed"
PURPLE_700 = "#6d28d9"
PURPLE_900 = "#4c1d95"
BG = "#0f172a"
PANEL = "#111827"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GREEN = "#34d399"
RED = "#f87171"


def _style(fig: go.Figure, title: str = "", *, height: int | None = None) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=PURPLE_300)),
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=13),
        margin=dict(l=50, r=30, t=70, b=50),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=PURPLE_500, font_size=12,
                        font_family="Inter, system-ui, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=PURPLE_900,
                    borderwidth=1, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor="#1f2937", zerolinecolor="#1f2937")
    fig.update_yaxes(gridcolor="#1f2937", zerolinecolor="#1f2937")
    if height is not None:
        fig.update_layout(height=height)
    return fig


# ---------------------------------------------------------------------------
# Accuracy vs Compression
# ---------------------------------------------------------------------------
def accuracy_vs_compression_curve(df: pd.DataFrame) -> go.Figure:
    d = df.sort_values("compression_ratio").reset_index(drop=True)
    fig = go.Figure()

    if not d.empty:
        baseline_row = d[d["method"].str.contains("Original", case=False, na=False)]
        if len(baseline_row):
            base_acc = float(baseline_row["top1_acc"].iloc[0])
            # "Lossless zone" — within 1 percentage point of the original.
            fig.add_hrect(
                y0=base_acc - 1.0, y1=base_acc + 0.5,
                fillcolor=GREEN, opacity=0.10, line_width=0,
                annotation_text="Lossless zone (±1%)",
                annotation_position="top right",
                annotation=dict(font=dict(color=GREEN, size=11)),
            )
            fig.add_hline(y=base_acc, line=dict(color=GREEN, width=1, dash="dot"),
                          annotation_text=f"Baseline {base_acc:.1f}%",
                          annotation_position="bottom right",
                          annotation_font_color=GREEN)

    # Stagger label positions so adjacent points don't collide.
    positions = ["top center", "bottom center"] * (len(d) // 2 + 1)
    fig.add_trace(go.Scatter(
        x=d["compression_ratio"], y=d["top1_acc"],
        mode="lines+markers+text",
        text=d["method"],
        textposition=positions[: len(d)],
        textfont=dict(color=PURPLE_50, size=11),
        line=dict(color=PURPLE_500, width=3, shape="spline", smoothing=0.6),
        marker=dict(size=12, color=PURPLE_400, line=dict(color=PURPLE_50, width=1.5)),
        customdata=np.stack([d["top5_acc"], d["kv_memory_mb"]], axis=-1),
        hovertemplate=("<b>%{text}</b><br>"
                       "Compression: %{x:.2f}×<br>"
                       "Top-1: %{y:.2f}%<br>"
                       "Top-5: %{customdata[0]:.2f}%<br>"
                       "KV memory: %{customdata[1]:.2f} MB<extra></extra>"),
        name="Top-1 accuracy",
    ))
    fig.update_xaxes(title="Compression ratio (×)")
    fig.update_yaxes(title="Top-1 accuracy (%)")
    return _style(fig, "Accuracy vs Compression", height=420)


# ---------------------------------------------------------------------------
# KV memory bar (with savings %)
# ---------------------------------------------------------------------------
def memory_savings_bar(df: pd.DataFrame) -> go.Figure:
    saved = df["memory_saved_pct"] if "memory_saved_pct" in df.columns else None
    labels = []
    for i, row in df.iterrows():
        if saved is not None and abs(row["memory_saved_pct"]) > 1e-3:
            labels.append(f"{row['kv_memory_mb']:.2f} MB<br>"
                          f"<span style='color:{GREEN}'>−{row['memory_saved_pct']:.0f}%</span> "
                          f"({row['compression_ratio']:.2f}×)")
        else:
            labels.append(f"{row['kv_memory_mb']:.2f} MB")

    fig = go.Figure(go.Bar(
        x=df["method"], y=df["kv_memory_mb"],
        marker=dict(color=PURPLE_500, line=dict(color=PURPLE_300, width=1)),
        text=labels, textposition="outside", textfont=dict(color=TEXT, size=11),
        customdata=np.stack([df["compression_ratio"],
                             df.get("memory_saved_pct", pd.Series([0]*len(df)))],
                            axis=-1),
        hovertemplate=("<b>%{x}</b><br>"
                       "KV memory: %{y:.2f} MB<br>"
                       "Compression: %{customdata[0]:.2f}×<br>"
                       "Memory saved: %{customdata[1]:.1f}%<extra></extra>"),
    ))
    fig.update_yaxes(title="KV memory (MB)", rangemode="tozero")
    fig.update_xaxes(title="Method", tickangle=-15)
    fig.update_layout(bargap=0.35)
    return _style(fig, "KV cache memory per method", height=420)


# ---------------------------------------------------------------------------
# Bits vs accuracy (smoothed, with sweet-spot annotation)
# ---------------------------------------------------------------------------
def bits_vs_accuracy_tradeoff(bits_list: List[int], acc_list: List[float]) -> go.Figure:
    fig = go.Figure()
    if bits_list and acc_list:
        order = np.argsort(bits_list)
        xs = np.array(bits_list)[order]
        ys = np.array(acc_list)[order]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=PURPLE_500, width=3, shape="spline", smoothing=1.2),
            marker=dict(size=12, color=PURPLE_400,
                        line=dict(color=PURPLE_50, width=1.5)),
            hovertemplate="%{x}-bit → %{y:.2f}% top-1<extra></extra>",
            name="TurboQuant",
        ))
        # Highlight 3-bit if present (the documented sweet spot).
        if 3 in xs:
            i3 = int(np.where(xs == 3)[0][0])
            fig.add_trace(go.Scatter(
                x=[xs[i3]], y=[ys[i3]], mode="markers",
                marker=dict(size=22, color="rgba(0,0,0,0)",
                            line=dict(color=GREEN, width=3)),
                hovertemplate="3-bit sweet spot<extra></extra>",
                showlegend=False,
            ))
            fig.add_annotation(
                x=xs[i3], y=ys[i3],
                text="🎯 sweet spot<br>(~3.5 bits/dim)",
                showarrow=True, arrowhead=2, arrowcolor=GREEN,
                font=dict(color=GREEN, size=11),
                ax=40, ay=-40,
            )
    fig.update_xaxes(title="Bits per dimension", dtick=1)
    fig.update_yaxes(title="Top-1 accuracy (%)")
    return _style(fig, "Bits vs Accuracy (TurboQuant sweep)", height=420)


# ---------------------------------------------------------------------------
# Layer-wise heatmap
# ---------------------------------------------------------------------------
def layer_wise_compression_impact(matrix: np.ndarray, layers: List[int],
                                   bits: List[int]) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix, x=[f"{b}-bit" for b in bits], y=[f"L{l}" for l in layers],
        colorscale="Purples", colorbar=dict(title="Δ acc (%)"),
        hovertemplate="Layer %{y} · %{x}<br>Δ acc: %{z:.3f}%<extra></extra>",
    ))
    fig.update_xaxes(title="Bit-width")
    fig.update_yaxes(title="ViT layer")
    return _style(fig, "Layer-wise accuracy delta")


# ---------------------------------------------------------------------------
# Compression-ratio gauge
# ---------------------------------------------------------------------------
def compression_ratio_gauge(ratio: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(ratio),
        gauge=dict(
            axis=dict(range=[1, 12], tickwidth=1, tickcolor=MUTED),
            bar=dict(color=PURPLE_500, thickness=0.3),
            bgcolor=PANEL,
            borderwidth=2, bordercolor=PURPLE_900,
            steps=[
                dict(range=[1, 2], color="#1e293b"),
                dict(range=[2, 4], color="#312e81"),
                dict(range=[4, 8], color=PURPLE_900),
                dict(range=[8, 12], color=PURPLE_700),
            ],
            threshold=dict(line=dict(color=GREEN, width=4),
                           thickness=0.75, value=float(ratio)),
        ),
        number=dict(suffix="×", font=dict(color=TEXT, size=42)),
    ))
    return _style(fig, "Compression ratio", height=300)


# ---------------------------------------------------------------------------
# Latency bar (warm vs raw)
# ---------------------------------------------------------------------------
def latency_bar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    has_warm = "latency_warm_ms" in df.columns
    if has_warm:
        fig.add_trace(go.Bar(
            x=df["method"], y=df["latency_ms"],
            name="Mean (incl. warmup)",
            marker_color=PURPLE_900,
            text=[f"{v:.1f}" for v in df["latency_ms"]],
            textposition="outside", textfont=dict(color=MUTED, size=10),
            hovertemplate="<b>%{x}</b><br>Mean latency: %{y:.2f} ms<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=df["method"], y=df["latency_warm_ms"],
            name="Warm (steady-state)",
            marker=dict(color=PURPLE_400, line=dict(color=PURPLE_50, width=1)),
            text=[f"{v:.1f} ms" for v in df["latency_warm_ms"]],
            textposition="outside", textfont=dict(color=TEXT, size=11),
            hovertemplate="<b>%{x}</b><br>Warm latency: %{y:.2f} ms<extra></extra>",
        ))
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    else:
        fig.add_trace(go.Bar(
            x=df["method"], y=df["latency_ms"],
            marker=dict(color=PURPLE_400, line=dict(color=PURPLE_50, width=1)),
            text=[f"{v:.1f} ms" for v in df["latency_ms"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Latency: %{y:.2f} ms<extra></extra>",
        ))
    fig.update_yaxes(title="Latency / image (ms)", rangemode="tozero")
    fig.update_xaxes(title="Method", tickangle=-15)
    return _style(fig, "Inference latency (per image)", height=420)


# ---------------------------------------------------------------------------
# Attention visualizations
# ---------------------------------------------------------------------------
def attention_heatmap_comparison(orig_attn: np.ndarray, comp_attn: np.ndarray,
                                  image: Image.Image, grid: int = 14) -> go.Figure:
    """Side-by-side attention heatmap from CLS token to all patches, overlaid."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Original attention", "Compressed attention"))
    for i, attn in enumerate([orig_attn, comp_attn]):
        cls = attn[0, 1:]
        side = int(np.sqrt(cls.shape[0]))
        heat = cls[:side * side].reshape(side, side)
        fig.add_trace(go.Heatmap(z=heat, colorscale="Purples", showscale=(i == 1),
                                 hovertemplate="patch (%{x},%{y})<br>w=%{z:.4f}<extra></extra>"),
                      row=1, col=i + 1)
    return _style(fig, "Attention map: original vs compressed")


def patch_attention_grid(image: Image.Image, attention: np.ndarray,
                         grid: int = 14) -> go.Figure:
    cls = attention[0, 1:]
    side = int(np.sqrt(cls.shape[0]))
    heat = cls[:side * side].reshape(side, side)
    fig = go.Figure(go.Heatmap(
        z=heat, colorscale="Purples", showscale=True,
        colorbar=dict(title="weight", thickness=10),
        hovertemplate="patch (%{x},%{y})<br>weight=%{z:.4f}<extra></extra>",
    ))
    return _style(fig, f"Patch attention grid ({side}×{side})", height=380)
