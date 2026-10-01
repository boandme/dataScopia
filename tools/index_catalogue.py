"""Grow the dataset graph from the command line, across any supported dataset source: search,
profile each result's sample, and add its join-key sketch to the catalogue (the same catalogue the
app fills organically as people analyse datasets). Generalizes index_kaggle.py to also cover
Hugging Face, so new sources only need a search/sample/describe triple wired in below, not a
whole new script.

    python tools/index_catalogue.py "world bank" --limit 40
    python tools/index_catalogue.py "covid" "gdp" "population" --provider huggingface --limit 25
    python tools/index_catalogue.py "census" --provider both --limit 20

Kaggle needs credentials (KAGGLE_USERNAME / KAGGLE_KEY in the environment or .env); Hugging Face
works without any for public datasets (HF_TOKEN is optional, for gated data or a higher rate
limit). Only sketches (key value sets, MinHash signatures, column names) are stored - never the
data itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "apps" / "api" / "app", ROOT / "packages" / "analysis" / "src", ROOT / "packages" / "dataset" / "src"):
    sys.path.insert(0, str(path))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import graph  # noqa: E402
from dataset.kaggle_search import get_dataset_description as kaggle_description, search_kaggle_datasets  # noqa: E402
from dataset.streaming import load_dataset_sample as load_kaggle_sample  # noqa: E402
from dataset.huggingface import get_hf_dataset_description, load_hf_dataset_sample, search_hf_datasets  # noqa: E402


def _crawl_kaggle(query: str, limit: int, sample_bytes: int) -> tuple[int, int]:
    done = failed = 0
    for item in search_kaggle_datasets(query, max_results=limit):
        slug = item.get("slug") or item.get("id")
        if item.get("is_demo") or not slug or "/" not in slug:
            continue
        try:
            loaded = load_kaggle_sample(slug, max_sample_bytes=sample_bytes)
            info = kaggle_description(slug)
            graph.index_dataset(slug, info.get("title") or item.get("title") or slug, info.get("license") or item.get("license"), loaded["dataframe"])
            done += 1
            print(f"indexed kaggle:{slug}")
        except Exception as exc:  # noqa: BLE001 - one bad dataset must not stop the crawl
            failed += 1
            print(f"skipped kaggle:{slug}: {exc}")
    return done, failed


def _crawl_huggingface(query: str, limit: int, sample_bytes: int) -> tuple[int, int]:
    done = failed = 0
    max_rows = max(1, sample_bytes // 200)  # same ~200-bytes/row heuristic streaming.py uses for Kaggle parquet
    for item in search_hf_datasets(query, max_results=limit):
        ref = (item.get("id") or "").removeprefix("hf:")
        if item.get("is_demo") or not ref or "/" not in ref:
            continue
        try:
            loaded = load_hf_dataset_sample(ref, max_rows=max_rows)
            info = get_hf_dataset_description(ref)
            graph.index_dataset(f"hf:{ref}", info.get("title") or ref, info.get("license") or item.get("license"), loaded["dataframe"], source="huggingface")
            done += 1
            print(f"indexed hf:{ref}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"skipped hf:{ref}: {exc}")
    return done, failed


PROVIDERS: dict[str, Callable[[str, int, int], tuple[int, int]]] = {
    "kaggle": _crawl_kaggle,
    "huggingface": _crawl_huggingface,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queries", nargs="+", help="search terms")
    parser.add_argument("--provider", choices=[*PROVIDERS, "both"], default="both", help="which source(s) to crawl (default: both)")
    parser.add_argument("--limit", type=int, default=25, help="max datasets per query per provider")
    parser.add_argument("--sample-mb", type=float, default=2.0, help="how much of each file/row-range to sample")
    args = parser.parse_args()

    providers = list(PROVIDERS) if args.provider == "both" else [args.provider]
    sample_bytes = int(args.sample_mb * 1024 * 1024)
    totals = {"done": 0, "failed": 0}
    for query in args.queries:
        for name in providers:
            done, failed = PROVIDERS[name](query, args.limit, sample_bytes)
            totals["done"] += done
            totals["failed"] += failed
    print(f"done: {totals['done']} indexed, {totals['failed']} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
