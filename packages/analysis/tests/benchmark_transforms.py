"""Standalone benchmark for the transform engine (not a pytest suite - run directly).

Measures two things:
  1. Pure in-process latency of each operation across synthetic datasets of increasing size,
     to characterize scaling behavior.
  2. Real end-to-end latency through the live API (HTTP + server + transform) against actual
     Kaggle datasets, to confirm the numbers above hold up outside a microbenchmark.

Usage:
    python packages/analysis/tests/benchmark_transforms.py
    python packages/analysis/tests/benchmark_transforms.py --skip-api   # in-process only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.transforms import apply_transform  # noqa: E402

SIZES = [1_000, 10_000, 100_000, 500_000]


def make_df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "id": range(n),
            "amount": rng.normal(500, 150, n),
            "quantity": rng.integers(1, 50, n),
            "category": rng.choice(["A", "B", "C", "D"], n),
            "flag": rng.choice(["Yes", "No", None], n, p=[0.45, 0.45, 0.1]),
            "first": [f"F{i}" for i in range(n)],
            "last": [f"L{i}" for i in range(n)],
        }
    )


OPS = [
    ("filter", {"column": "amount", "comparator": ">", "value": "500"}),
    ("drop_missing", {"column": "flag"}),
    ("fill_missing", {"column": "flag", "strategy": "constant", "constant_value": "Unknown"}),
    ("drop_outliers", {"column": "amount"}),
    ("rename_column", {"column": "quantity", "new_name": "qty"}),
    ("cast_type", {"column": "category", "target_type": "category"}),
    ("combine_columns", {"columns": ["first", "last"], "operation": "concat", "separator": " ", "new_name": "full_name"}),
    ("drop_column", {"column": "id"}),
]


def time_call(fn, *args, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - start)
    return best


def run_in_process_benchmark() -> None:
    print("=" * 78)
    print("IN-PROCESS BENCHMARK (pure Python/pandas, no HTTP)")
    print("=" * 78)
    header = f"{'operation':<18}" + "".join(f"{n:>12,} rows" for n in SIZES)
    print(header)
    print("-" * len(header))
    for op_type, params in OPS:
        row = f"{op_type:<18}"
        for n in SIZES:
            df = make_df(n)
            try:
                elapsed = time_call(apply_transform, df, op_type, params)
                row += f"{elapsed * 1000:>15.1f}ms"
            except Exception as exc:  # pragma: no cover
                row += f"{'ERROR':>17}"
                print(f"  ! {op_type} failed at n={n}: {exc}")
        print(row)
    print()
    print("All times are best-of-3 wall-clock, single operation, on a fresh copy of the data.")
    print()


def run_api_benchmark() -> None:
    import json

    try:
        import requests
    except ImportError:  # pragma: no cover
        print("requests not available; skipping live API benchmark.")
        return

    base = "http://127.0.0.1:8000"
    try:
        requests.get(f"{base}/health", timeout=3)
    except Exception:
        print("API server not reachable at", base, "- skipping live benchmark.")
        print("(start it with: python run.py)")
        return

    print("=" * 78)
    print("LIVE API BENCHMARK (HTTP + server + real Kaggle data)")
    print("=" * 78)

    datasets = ["yasserh/titanic-dataset", "nitirajkulkarni/busan-kr-1838524"]
    for slug in datasets:
        t0 = time.perf_counter()
        resp = requests.post(f"{base}/datasets/analyze", json={"slug": slug}, timeout=60)
        analyze_time = time.perf_counter() - t0
        if resp.status_code != 200:
            print(f"  ! {slug}: analyze failed ({resp.status_code}) - {resp.text[:150]}")
            continue
        data = resp.json()
        rows = data["analysis"]["shape"]["rows"]
        cols = data["analysis"]["columns"]
        numeric_cols = [c["name"] for c in cols if c["type"] == "numeric"]
        print(f"\n{slug}  ({rows:,} rows, analyzed in {analyze_time * 1000:.0f}ms)")

        if numeric_cols:
            col = numeric_cols[0]
            median = data["analysis"]["numeric_summary"].get(col, {}).get("median", 0)
            t0 = time.perf_counter()
            tx_resp = requests.post(
                f"{base}/datasets/transform",
                json={
                    "dataset_id": slug,
                    "dataset_name": data["analysis"]["dataset_name"],
                    "type": "filter",
                    "params": {"column": col, "comparator": ">=", "value": str(median)},
                },
                timeout=30,
            )
            tx_time = time.perf_counter() - t0
            if tx_resp.status_code == 200:
                result = tx_resp.json()
                print(f"  filter '{col}' >= {median}: {tx_time * 1000:.0f}ms -> {result['shape']['rows']:,} rows kept")
            else:
                print(f"  ! filter failed: {tx_resp.status_code} {tx_resp.text[:150]}")

            t0 = time.perf_counter()
            requests.post(f"{base}/datasets/transform/reset", json={"dataset_id": slug, "dataset_name": data["analysis"]["dataset_name"]}, timeout=30)
            reset_time = time.perf_counter() - t0
            print(f"  reset to original: {reset_time * 1000:.0f}ms")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-api", action="store_true", help="Skip the live API benchmark section")
    args = parser.parse_args()

    run_in_process_benchmark()
    if not args.skip_api:
        run_api_benchmark()
