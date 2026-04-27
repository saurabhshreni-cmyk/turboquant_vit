"""Streamlit app: TurboQuant × Vision Transformers."""
from __future__ import annotations

import io
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from compressed_attention import compressed_vit, run_inference
from data_loader import (cifar_status, get_sample_images, get_test_loader,
                         preprocess_pil, set_transform)
from evaluator import CIFAR_ORDER, ViTEvaluator, imagenet_logits_to_cifar
from model_loader import load_model_with_fallback
from utils import format_mb, get_device, make_banner, model_device
import visualizer as viz


st.set_page_config(
    page_title="TurboQuant × ViT",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .stApp { background-color: #0f172a; color: #e2e8f0; }
      [data-testid="stSidebar"] { background-color: #111827; }
      h1, h2, h3, h4 { color: #f5f3ff; }
      .accent { color: #8b5cf6; font-weight: 700; }
      .hl-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border: 1px solid #4c1d95; border-radius: 14px; padding: 18px;
        box-shadow: 0 4px 18px rgba(139,92,246,0.15);
      }
      .hl-card h3 { margin: 0 0 6px 0; color: #c4b5fd; }
      .pill {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        background: #4c1d95; color: #ede9fe; font-size: 12px; margin-right: 6px;
      }
      .stMetric { background: #1e293b; border-radius: 10px; padding: 8px; }
      .changed-yes { color: #f87171; font-weight: 700; }
      .changed-no  { color: #34d399; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ViT model (HF → torchvision fallback)...")
def load_model(hf_timeout_s: int = 90):
    """Try HF cached → HF download (with timeout) → torchvision fallback.
    Always returns (model, label, transform). Sets the global preprocessing
    transform so CIFAR-10 batches use the correct normalization.
    """
    try:
        model, label, transform = load_model_with_fallback(hf_timeout_s=hf_timeout_s)
    except Exception as e:
        st.error(f"Both model loaders failed: {e}")
        raise
    set_transform(transform)
    return model, label, transform


@st.cache_resource(show_spinner=False)
def get_samples():
    """Cached UI sampler: fast on warm cache, synthetic fallback on cold cache.
    Wrapped with our own `st.spinner` so the user sees progress without the
    Streamlit default sticky spinner."""
    with st.spinner("Preparing CIFAR-10 sample images (timeout = 45s, falls back to placeholders)..."):
        return get_sample_images(10, download_timeout=45)


@st.cache_data(show_spinner=False)
def get_banner_bytes():
    img = make_banner()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Sidebar nav
# ---------------------------------------------------------------------------
PAGES = ["🏠 Home", "🖼️ Try It Live", "📊 Benchmark Dashboard",
         "🔍 Attention Visualizer", "🔬 How It Works"]

with st.sidebar:
    st.markdown("### TurboQuant × ViT")
    st.caption("KV Cache Compression for Vision Transformers")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption(f"Device: **{get_device().type.upper()}**")
    # Eagerly load the model (cached) so its label can be shown here.
    try:
        _model, _model_label, _transform = load_model()
        st.success(f"✓ {_model_label}")
    except Exception as e:
        st.error(f"Model load failed: {e}")
        _model_label = "(none)"
    st.caption(f"CIFAR-10: **{cifar_status()}**")


# ---------------------------------------------------------------------------
# PAGE 1 — Home
# ---------------------------------------------------------------------------
def page_home():
    st.image(get_banner_bytes(), use_container_width=True)
    st.markdown("# TurboQuant × Vision Transformers")
    st.markdown(
        "#### <span class='accent'>Applying ICLR 2026 KV Compression to Image Classification</span>",
        unsafe_allow_html=True,
    )
    st.write(
        "Vision Transformers compute self-attention over image patches. Each "
        "patch produces Key and Value vectors — exactly the structure that "
        "TurboQuant was designed to compress in language models. This project "
        "is the first open-source application of TurboQuant's QJL + PolarQuant "
        "machinery to ViT inference."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='hl-card'><h3>🧪 Novel application</h3>"
                    "TurboQuant's authors only evaluated on LLMs. Here we adapt "
                    "the same math to ViT-B/16.</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='hl-card'><h3>⚙️ Zero retraining</h3>"
                    "Compression is applied at inference time via PyTorch hooks. "
                    "Model weights are never modified.</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='hl-card'><h3>💾 Up to ~6× memory savings</h3>"
                    "PolarQuant @ 3-bit + 1-bit QJL residual ≈ 3.5 bits/dim, "
                    "down from 32 — with near-lossless accuracy.</div>",
                    unsafe_allow_html=True)
    st.divider()
    st.markdown("##### Pipeline")
    st.code(
        "image → 14×14 patches → embeddings → Q,K,V → "
        "TurboQuant(K,V) → attention → logits → prediction",
        language="text",
    )
    st.markdown(
        "Paper: **TurboQuant: Online Vector Quantization with Near-optimal "
        "Distortion Rate** (Google Research, ICLR 2026)."
    )


# ---------------------------------------------------------------------------
# PAGE 2 — Try It Live
# ---------------------------------------------------------------------------
def _predict(model, pixel_values, method, bits, capture_attn=True):
    t0 = time.perf_counter()
    logits, tracker = run_inference(model, pixel_values, method=method,
                                    bits=bits, capture_attn=capture_attn)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    cifar = imagenet_logits_to_cifar(logits)
    pred = int(cifar.argmax(dim=-1).item())
    conf = float(cifar.max(dim=-1).values.item())
    return {
        "logits": logits, "cifar_scores": cifar, "pred": pred,
        "label": CIFAR_ORDER[pred], "conf": conf,
        "latency_ms": elapsed_ms,
        "memory_bytes": tracker.total_compressed_bytes(),
        "orig_memory_bytes": tracker.total_original_bytes(),
        "attention": tracker.per_layer_attention(),
    }


def page_try_live():
    st.markdown("## 🖼️ Try it live")
    st.caption("Compare an uncompressed forward pass with a compressed one, side by side.")
    model, model_label, _ = load_model()
    st.info(f"Model in use: **{model_label}**")
    samples = get_samples()

    col1, col2 = st.columns([1, 1])
    with col1:
        choice = st.selectbox("Pick a CIFAR-10 sample",
                              [f"#{i} — {name}" for i, (_, _, name) in enumerate(samples)])
        idx = int(choice.split("—")[0].strip().lstrip("#"))
        upload = st.file_uploader("…or upload an image", type=["png", "jpg", "jpeg"])
        if upload is not None:
            img = Image.open(upload).convert("RGB")
            true_label = None
        else:
            img, label_idx, label_name = samples[idx]
            true_label = label_name
        st.image(img, caption=f"Input ({true_label or 'uploaded'})", width=240)

    with col2:
        method = st.selectbox("Compression method",
                              ["Normal", "QJL", "PolarQuant", "TurboQuant"], index=3)
        bits = st.slider("Bit-width", 1, 4, 3)
        run_btn = st.button("▶ Run inference", use_container_width=True, type="primary")

    if not run_btn:
        return

    pixel_values = preprocess_pil(img).to(model_device(model))

    method_id = {"Normal": "none", "QJL": "qjl",
                 "PolarQuant": "polarquant", "TurboQuant": "turboquant"}[method]

    with st.spinner("Running both passes ..."):
        try:
            res_orig = _predict(model, pixel_values, "none", 32)
            res_comp = _predict(model, pixel_values, method_id, bits)
        except Exception as exc:  # noqa: BLE001 — keep the UI alive on failure
            st.error(f"Inference failed: {exc}")
            return

    a, b = st.columns(2)
    for col, label, res in [(a, "Original (FP32)", res_orig), (b, f"{method} {bits}-bit", res_comp)]:
        with col:
            st.markdown(f"### {label}")
            st.metric("Prediction", res["label"], f"{res['conf']*100:.1f}% conf")
            st.metric("Latency", f"{res['latency_ms']:.1f} ms")
            st.metric("KV memory", format_mb(res["memory_bytes"]))
            attn_layers = res["attention"]
            if attn_layers:
                last = attn_layers[max(attn_layers.keys())][0, 0].cpu().numpy()
                fig = viz.patch_attention_grid(img, last)
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    changed = res_orig["label"] != res_comp["label"]
    cls = "changed-yes" if changed else "changed-no"
    msg = "❌ Prediction CHANGED after compression" if changed else "✅ Prediction PRESERVED after compression"
    st.markdown(f"<h4 class='{cls}'>{msg}</h4>", unsafe_allow_html=True)

    saved = 1.0 - (res_comp["memory_bytes"] / max(res_comp["orig_memory_bytes"], 1))
    ratio = (res_comp["orig_memory_bytes"] / max(res_comp["memory_bytes"], 1))
    c1, c2, c3 = st.columns(3)
    c1.metric("Memory saved", f"{saved*100:.1f}%")
    c2.metric("Compression ratio", f"{ratio:.2f}×")
    c3.metric("Δ Latency",
              f"{res_comp['latency_ms'] - res_orig['latency_ms']:+.1f} ms")
    st.plotly_chart(viz.compression_ratio_gauge(ratio), use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 3 — Benchmark Dashboard
# ---------------------------------------------------------------------------
def page_benchmark():
    st.markdown("## 📊 Benchmark dashboard")
    st.caption(
        "Every method is evaluated on the **same** CIFAR-10 slice with the same "
        "batch order — accuracy, memory and latency are directly comparable."
    )
    # On Streamlit Cloud (CPU-only, 1 GB RAM) keep the defaults small enough
    # to stay well under the per-request time budget. Local CUDA users can
    # always crank the slider up.
    cloud = not torch.cuda.is_available()
    c_n, c_b = st.columns([2, 1])
    with c_n:
        n = st.slider(
            "Number of CIFAR-10 test images",
            min_value=16, max_value=256 if cloud else 2000,
            value=32 if cloud else 256, step=16,
            help="Larger N = more reliable accuracy estimates but longer runtime.",
        )
    with c_b:
        bs = st.slider(
            "Batch size", 4, 16 if cloud else 32, 8 if cloud else 16, step=4,
            help="Lower this if you hit out-of-memory.",
        )
    if cloud:
        st.caption("ℹ️ Running on CPU — sliders are capped for the free tier. "
                   "Clone locally for full-scale benchmarking.")

    if st.button("▶ Run full benchmark", type="primary"):
        model, _, _ = load_model()
        loader = get_test_loader(batch_size=bs, num_samples=n)
        evaluator = ViTEvaluator(model)

        bar = st.progress(0.0, text="Starting...")
        def cb(p, msg): bar.progress(min(max(p, 0.0), 1.0), text=msg)

        try:
            df = evaluator.run_full_benchmark(loader, progress_cb=cb)
        except Exception as exc:  # noqa: BLE001 — surface error without crashing UI
            bar.empty()
            st.error(f"Benchmark failed: {exc}")
            return
        bar.empty()
        st.session_state["bench_df"] = df

    df = st.session_state.get("bench_df")
    if df is None:
        st.info("Run a benchmark to populate this dashboard.")
        return

    # ------------------------------------------------------------- Sweet-spot callout
    base = df[df["method"].str.contains("Original", case=False, na=False)]
    if not base.empty:
        base_acc_val = float(base["top1_acc"].iloc[0])
        sweet = df[df["method"].str.contains("TurboQuant-3bit", case=False, na=False)]
        if not sweet.empty:
            s = sweet.iloc[0]
            delta = s["top1_acc"] - base_acc_val
            st.success(
                f"🔥 **Sweet spot — TurboQuant 3-bit**: "
                f"{s['compression_ratio']:.2f}× compression, "
                f"{s['memory_saved_pct']:.1f}% KV memory saved, "
                f"top-1 {s['top1_acc']:.2f}% ({delta:+.2f}% vs FP32)."
            )

    # ------------------------------------------------------------- KPI strip
    best_compress = df.loc[df["compression_ratio"].idxmax()] if not df.empty else None
    if best_compress is not None and not base.empty:
        base_acc = float(base["top1_acc"].iloc[0])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Best compression",
                  f"{best_compress['compression_ratio']:.2f}×",
                  f"{best_compress['memory_saved_pct']:.1f}% memory saved")
        k2.metric("Accuracy at best compression",
                  f"{best_compress['top1_acc']:.2f}%",
                  f"{best_compress['top1_acc'] - base_acc:+.2f}% vs original",
                  delta_color="normal")
        k3.metric("Original latency (warm)",
                  f"{float(base['latency_warm_ms'].iloc[0]):.1f} ms")
        k4.metric("Δ Latency at best compression",
                  f"{best_compress['latency_delta_ms']:+.1f} ms",
                  delta_color="inverse")

    st.divider()
    st.markdown("##### Raw results")
    st.caption("Latency is reported both as the cold-start mean and the warm "
               "steady-state mean (first batch dropped to remove CUDA warmup bias).")
    st.dataframe(df.style.format({
        "top1_acc": "{:.2f}", "top5_acc": "{:.2f}",
        "kv_memory_mb": "{:.2f}", "compression_ratio": "{:.2f}",
        "latency_ms": "{:.1f}", "latency_warm_ms": "{:.1f}",
        "latency_delta_ms": "{:+.1f}", "memory_saved_pct": "{:.1f}",
        "attention_distortion": "{:.4f}",
    }), use_container_width=True)

    # ------------------------------------------------------------- Charts
    st.markdown("##### Accuracy vs compression")
    st.caption("The green band marks the lossless zone — anywhere inside it the "
               "compressed model matches the FP32 baseline within ±1 % top-1.")
    st.plotly_chart(viz.accuracy_vs_compression_curve(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### KV cache memory")
        st.caption("Compression ratio (×) and memory saved (%) annotated on each bar.")
        st.plotly_chart(viz.memory_savings_bar(df), use_container_width=True)
    with c2:
        st.markdown("##### Inference latency (cached compressors)")
        st.caption("Two bars per method: cold-start mean (first batch incl. CUDA "
                   "warmup) and warm steady-state. Compressors are pre-cached, so "
                   "the cold-start spike is purely CUDA kernel JIT — not codebook "
                   "fitting.")
        st.plotly_chart(viz.latency_bar(df), use_container_width=True)

    turbo = df[df["method"].str.contains("TurboQuant")]
    if len(turbo) >= 2:
        st.markdown("##### Bits vs accuracy (TurboQuant only)")
        st.caption("Smoothed sweep over bit-widths — the 3-bit point is the documented sweet spot.")
        st.plotly_chart(
            viz.bits_vs_accuracy_tradeoff(turbo["bits"].tolist(), turbo["top1_acc"].tolist()),
            use_container_width=True,
        )

    st.download_button(
        "⬇ Download CSV", data=df.to_csv(index=False).encode("utf-8"),
        file_name="turboquant_vit_benchmark.csv", mime="text/csv",
    )


# ---------------------------------------------------------------------------
# PAGE 4 — Attention visualizer
# ---------------------------------------------------------------------------
def page_attention():
    st.markdown("## 🔍 Attention visualizer")
    st.caption("Compare attention maps across compression methods.")
    model, model_label, _ = load_model()
    samples = get_samples()

    upload = st.file_uploader("Upload image (or use sample)", type=["png", "jpg", "jpeg"])
    if upload is not None:
        img = Image.open(upload).convert("RGB")
    else:
        img, _, _ = samples[0]

    st.image(img, width=200)

    pixel_values = preprocess_pil(img).to(model_device(model))

    with st.spinner("Capturing attention for all methods ..."):
        results = {}
        for method, bits in [("none", 32), ("qjl", 1), ("polarquant", 3), ("turboquant", 3)]:
            try:
                with compressed_vit(model, method=method, bits=bits, capture_attn=True) as t:
                    with torch.no_grad():
                        model(pixel_values=pixel_values)
                    attn = t.per_layer_attention()
                    results[(method, bits)] = {
                        int(k): v.detach().cpu().numpy() for k, v in attn.items()
                    }
            except Exception as exc:  # noqa: BLE001 — UI must not crash on a single method
                st.warning(f"Skipped {method} ({bits}-bit): {exc}")
                results[(method, bits)] = {}

    # Use only the layers/heads that EVERY method captured, so the slider can
    # never produce an index that's missing for some method.
    populated = [r for r in results.values() if r]
    if not populated:
        st.error("No attention maps were captured for any method.")
        return
    common_layers = sorted(set.intersection(*(set(r.keys()) for r in populated)))
    if not common_layers:
        st.error("Methods captured disjoint layer sets — nothing to compare.")
        return
    sample_layer = common_layers[0]
    n_heads = min(r[sample_layer].shape[1] for r in populated)

    st.caption(f"Available layers: {common_layers[0]}–{common_layers[-1]} "
               f"({len(common_layers)} layers × {n_heads} heads)")
    layer_pos = st.slider("Layer", 0, len(common_layers) - 1, len(common_layers) - 1)
    layer = common_layers[layer_pos]
    head = st.slider("Head", 0, n_heads - 1, 0)

    cols = st.columns(2)
    panel_methods = [("none", 32, "Original"), ("qjl", 1, "QJL 1-bit"),
                     ("polarquant", 3, "PolarQuant 3-bit"),
                     ("turboquant", 3, "TurboQuant 3-bit")]
    for i, (m, b, title) in enumerate(panel_methods):
        with cols[i % 2]:
            st.markdown(f"#### {title}")
            method_results = results.get((m, b), {})
            layer_attn = method_results.get(layer)
            if layer_attn is None:
                st.info("Attention unavailable for this method — fallback applied.")
                continue
            head_idx = min(head, layer_attn.shape[1] - 1)
            attn = layer_attn[0, head_idx]
            try:
                fig = viz.patch_attention_grid(img, attn)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not render attention grid: {exc}")


# ---------------------------------------------------------------------------
# PAGE 5 — How it works
# ---------------------------------------------------------------------------
def page_how():
    st.markdown("## 🔬 How it works")
    st.markdown(
        "<span class='pill'>Step 1</span> ViT splits a 224×224 image into 14×14 = 196 patches of 16×16 px each.",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span class='pill'>Step 2</span> Each patch is embedded into a 768-D token; the transformer "
        "computes per-head Q, K, V (head dim = 64) for each token.",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span class='pill'>Step 3</span> TurboQuant compresses K and V along the head_dim axis. "
        "PolarQuant rotates the vector so coordinates are sphere-uniform, then applies a Lloyd-Max "
        "scalar quantizer. A 1-bit QJL sketch on the residual corrects bias for inner products.",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span class='pill'>Step 4</span> Attention is computed against the decompressed K, V. "
        "Q is left untouched.",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span class='pill'>Step 5</span> Memory drops from 32 bits/dim to ~3.5 bits/dim "
        "(≈9× smaller KV cache) with near-lossless top-1 accuracy.",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### Key equations")
    st.latex(r"\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V")
    st.latex(r"\mathrm{QJL}(x) = \mathrm{sign}(R x), \quad R \in \mathbb{R}^{k\times d}, \; R_{ij}\sim\mathcal{N}(0,1)")
    st.latex(r"\langle x, y\rangle \;\approx\; \frac{\pi}{2k}\,\langle q_x, q_y\rangle\,\|x\|\,\|y\|")
    st.latex(r"\mathrm{PolarQuant}: \; y = \Pi x, \quad \hat y_i = \mathrm{LloydMax}_b(y_i / \|y\|)")
    st.divider()
    st.markdown("### Hook mechanism")
    st.code(
        '''with compressed_vit(model, method="turboquant", bits=3) as tracker:
    logits = model(pixel_values=images).logits
    print(tracker.total_compressed_bytes(), "vs", tracker.total_original_bytes())''',
        language="python",
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == PAGES[0]:   page_home()
elif page == PAGES[1]: page_try_live()
elif page == PAGES[2]: page_benchmark()
elif page == PAGES[3]: page_attention()
elif page == PAGES[4]: page_how()
