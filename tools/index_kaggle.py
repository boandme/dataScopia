"""Grow the dataset graph from the command line: search Kaggle, profile each result's sample and add its
join-key sketch to the catalogue (the same catalogue the app fills as people analyse datasets).

    python tools/index_kaggle.py "world bank" --limit 40
    python tools/index_kaggle.py "covid" "gdp" "population" --limit 25

Needs Kaggle credentials (KAGGLE_USERNAME / KAGGLE_KEY in the environment or .env). Only sketches
(key value sets, MinHash signatures, column names) are stored - never the data itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "apps" / "api" / "app", ROOT / "packages" / "analysis" / "src", ROOT / "packages" / "dataset" / "src"):
    sys.path.insert(0, str(path))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import graph  # noqa: E402
from dataset.kaggle_search import get_dataset_description, search_kaggle_datasets  # noqa: E402
from dataset.streaming import load_dataset_sample  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queries", nargs="+", help="Kaggle search terms")
    parser.add_argument("--limit", type=int, default=25, help="max datasets per query")
    parser.add_argument("--sample-mb", type=float, default=2.0, help="how much of each file to stream")
    args = parser.parse_args()
    done = failed = 0
    for query in args.queries:
        for item in search_kaggle_datasets(query, max_results=args.limit):
            slug = item.get("slug") or item.get("id")
            if item.get("is_demo") or not slug or "/" not in slug:
                continue
            try:
                loaded = load_dataset_sample(slug, max_sample_bytes=int(args.sample_mb * 1024 * 1024))
                info = get_dataset_description(slug)
                graph.index_dataset(slug, info.get("title") or item.get("title") or slug, info.get("license") or item.get("license"), loaded["dataframe"])
                done += 1
                print(f"indexed {slug}")
            except Exception as exc:  # noqa: BLE001 - one bad dataset must not stop the crawl
                failed += 1
                print(f"skipped {slug}: {exc}")
    print(f"done: {done} indexed, {failed} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
