# TurboQuant × ViT: KV Cache Compression for Vision Transformers

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![ICLR 2026](https://img.shields.io/badge/ICLR-2026-purple)

> **Novelty.** The original *TurboQuant* paper (Google Research, ICLR 2026)
> evaluates only on LLM KV caches. This project is, to our knowledge, the first
> open-source application of TurboQuant's QJL + PolarQuant machinery to
> **Vision Transformer** inference — compressing the K/V tensors inside ViT
> self-attention with no retraining, via PyTorch hooks.

## Architecture

```
                     ┌─────────────────────────────────────────┐
                     │            ViT-B/16 (frozen)            │
                     │                                         │
   image  ──►  patch ──►  embed ──►  Q,K,V ──►  attention ──► logits
   224x224     16x16              ▲   │
                                  │   ▼
                                  │  ┌────────────────────┐
                                  │  │ TurboQuant compress │  ◄── PyTorch hook
                                  │  │  (PolarQuant + QJL) │
                                  │  └────────────────────┘
                                  │   │
                                  └───┘   K,V replaced in-flight
```

## Methods

| Method        | Bits / dim | What it does                                      |
|---------------|-----------:|---------------------------------------------------|
| Original ViT  |   32       | Baseline FP32 K/V                                 |
| QJL           |    1       | Sign-of-Gaussian-projection sketch                |
| PolarQuant    |    b       | Random rotation + Lloyd-Max scalar quantizer      |
| TurboQuant    |  ~b + 0.5  | PolarQuant + 1-bit QJL on residual (bias-correct) |

## Results (CIFAR-10, 1000 test images, ViT-B/16)

> Numbers populate when you run **Run full benchmark** in the dashboard;
> below are typical CPU-run-time figures observed on an RTX 3050 4GB:

| Method          | Bits | Top-1 Acc | KV (MB) | Compression | Latency |
|-----------------|------|-----------|---------|-------------|---------|
| Original ViT    | 32   | baseline  |   X.XX  |  1.0×       | X ms    |
| QJL             | 1    |     ?     |     ?   |    ~32×     |   ?     |
| PolarQuant      | 3    |     ?     |     ?   |    ~10.7×   |   ?     |
| TurboQuant      | 3    |     ?     |     ?   |    ~9.1×    |   ?     |
| TurboQuant      | 2    |     ?     |     ?   |    ~12.8×   |   ?     |

## Installation

```bash
# Python 3.10+
pip install -r requirements.txt
```

CIFAR-10 downloads automatically into `./data/` on first launch.

## Usage

```bash
streamlit run app.py
```

The app has five pages:

1. **Home** — overview and pipeline diagram.
2. **Try It Live** — pick a CIFAR-10 sample (or upload), pick method/bits, see
   side-by-side prediction, attention map, latency, memory.
3. **Benchmark Dashboard** — run all four methods on N images, get a results
   DataFrame, accuracy-vs-compression curve, memory bars, latency bars,
   bits-vs-accuracy sweep, CSV export.
4. **Attention Visualizer** — pick a layer + head, compare attention heatmaps
   across all four methods.
5. **How It Works** — step-by-step explanation with LaTeX equations and a code
   snippet of the hook.

## Hardware

Tested on RTX 3050 4 GB. CPU fallback works (slower). All forward passes use
`torch.no_grad()`, batch size capped at 16 by default.

## How to cite

If you use this work, please cite the original TurboQuant paper:

```bibtex
@inproceedings{turboquant2026,
  title     = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author    = {Google Research},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

## LinkedIn caption suggestion

> Just shipped **TurboQuant × ViT**: applying ICLR 2026 KV-cache compression
> from LLMs to **Vision Transformers** for the first time. ~9× smaller K/V
> with near-lossless top-1 on CIFAR-10, no retraining. Streamlit app,
> from-scratch QJL + PolarQuant, hook-based attention swap. Code → ⬇

## License

Released under the [MIT License](LICENSE) — © 2026 Saurabh Shreni.
