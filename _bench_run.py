"""One-off helper: run the benchmark headless and emit the results
DataFrame as JSON for the README. Not committed to the repo (gitignored)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import get_test_loader, set_transform  # noqa: E402
from evaluator import ViTEvaluator  # noqa: E402
from model_loader import load_model_with_fallback  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    model, label, transform = load_model_with_fallback(hf_timeout_s=10)
    set_transform(transform)
    print(f"Loaded model: {label}", flush=True)

    loader = get_test_loader(batch_size=16, num_samples=256)
    evaluator = ViTEvaluator(model)

    def cb(p: float, msg: str) -> None:
        print(f"[{p*100:5.1f}%] {msg}", flush=True)

    df = evaluator.run_full_benchmark(loader, progress_cb=cb)
    print("\n=== RESULTS ===")
    print(df.to_string(index=False))

    out_path = Path(__file__).parent / "_bench_results.json"
    out_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
