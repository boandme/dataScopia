"""Hugging Face Hub as a second dataset source, alongside Kaggle.

Two Hugging Face services are used together:

* The **Hub API** (``huggingface.co/api``) for search and metadata - title, author, license,
  popularity - the same role ``kaggle_search.py`` plays for Kaggle.
* The **Datasets Server** (``datasets-server.huggingface.co``) for the actual data. Unlike Kaggle,
  where a dataset is an arbitrary pile of files we have to sniff and parse ourselves, every public
  Hugging Face dataset is pre-converted to Parquet and exposed as plain JSON rows - so sampling is
  one HTTP call (``/rows``) instead of streaming and parsing raw bytes, and a full read is
  ``pandas.read_parquet`` against a stable URL (``/parquet``) instead of a custom downloader.

Dataset refs are the bare Hub id (``"org/name"``, sometimes just ``"name"`` for canonical
datasets); the caller is responsible for the ``hf:`` namespace prefix used elsewhere in the app
(dataset_id, share links, the dataset-graph catalogue) so a Hugging Face ref can never collide
with a Kaggle slug.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

HF_API_BASE = "https://huggingface.co/api"
HF_DATASETS_SERVER = "https://datasets-server.huggingface.co"
REQUEST_TIMEOUT = 30

DEFAULT_SAMPLE_ROWS = 2000
MAX_FULL_ROWS = 300_000  # safety cap so "run full analysis" can't try to pull in a multi-GB dataset


class HFError(Exception):
    """A Hugging Face request that can't be satisfied; the message is shown to the user."""


def _headers() -> dict[str, str]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_API_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_credential_status() -> dict[str, Any]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_API_TOKEN")
    return {"configured": bool(token), "note": "A token is optional for public datasets (higher rate limits, and required for gated ones)."}


def _license_from(tags: list[str] | None, card_data: dict[str, Any] | None) -> str:
    if card_data and card_data.get("license"):
        value = card_data["license"]
        return value[0] if isinstance(value, list) and value else (value if isinstance(value, str) else "")
    for tag in tags or []:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return ""


def search_hf_datasets(query: str, max_results: int = 60) -> list[dict[str, Any]]:
    """Search public Hugging Face datasets by keyword, ranked by downloads (Kaggle's "hottest"
    equivalent). Never raises: a request failure surfaces as one explanatory result row, the same
    pattern ``search_kaggle_datasets`` uses, so a federated search never breaks on one source."""
    try:
        response = requests.get(
            f"{HF_API_BASE}/datasets",
            params={"search": query, "limit": max(1, min(max_results, 100)), "sort": "downloads", "direction": -1},
            headers=_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        items = response.json()
    except Exception as exc:
        return [
            {
                "id": "hf-demo-fallback",
                "title": "Hugging Face search unavailable",
                "subtitle": f"The Hugging Face Hub API could not be reached: {exc}",
                "url": "",
                "is_demo": True,
            }
        ]

    payload: list[dict[str, Any]] = []
    for item in items:
        ref = item.get("id")
        if not ref:
            continue
        description = " ".join(str(item.get("description") or "").split())
        payload.append(
            {
                "id": f"hf:{ref}",
                "slug": f"hf:{ref}",
                "title": ref,
                "subtitle": description[:200] or f"by {item.get('author') or ref.split('/')[0]}",
                "url": f"https://huggingface.co/datasets/{ref}",
                "size": f"{item.get('downloads', 0):,} downloads · {item.get('likes', 0):,} likes",
                "license": _license_from(item.get("tags"), None),
                "gated": bool(item.get("gated")),
                "is_demo": False,
                "source": "huggingface",
            }
        )
    return payload


def get_hf_dataset_description(ref: str) -> dict[str, Any]:
    """Title/description/license for one dataset - the Hugging Face analogue of
    ``kaggle_search.get_dataset_description``, used both for the license-before-download check
    and to seed column descriptions from the dataset card."""
    try:
        response = requests.get(f"{HF_API_BASE}/datasets/{ref}", params={"full": "true"}, headers=_headers(), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return {"title": ref, "subtitle": "", "description": "", "license": "", "url": f"https://huggingface.co/datasets/{ref}"}
    card = data.get("cardData") or {}
    description = card.get("description") or data.get("description") or ""
    return {
        "title": ref,
        "subtitle": (card.get("pretty_name") or "")[:200],
        "description": str(description),
        "license": _license_from(data.get("tags"), card),
        "url": f"https://huggingface.co/datasets/{ref}",
        "gated": bool(data.get("gated")),
    }


def _pick_config_and_split(ref: str) -> tuple[str, str]:
    response = requests.get(f"{HF_DATASETS_SERVER}/splits", params={"dataset": ref}, headers=_headers(), timeout=REQUEST_TIMEOUT)
    if response.status_code == 401:
        raise HFError(f"'{ref}' is private or gated. Add an HF_TOKEN with access to this dataset before analyzing it.")
    response.raise_for_status()
    splits = response.json().get("splits") or []
    if not splits:
        raise HFError(f"'{ref}' has no readable splits (it may not be a tabular dataset, or is still being processed by Hugging Face).")
    configs = [s["config"] for s in splits]
    config = "default" if "default" in configs else configs[0]
    same_config = [s["split"] for s in splits if s["config"] == config]
    split = "train" if "train" in same_config else same_config[0]
    return config, split


def _flatten_cell(value: Any) -> Any:
    """Most Hugging Face datasets used for tabular analysis have scalar cells, but a feature can
    be a nested list/dict (e.g. a ClassLabel's raw structure, or a list-of-strings column); those
    are stringified rather than dropped, so the profiler always gets a valid dataframe instead of
    the load failing outright on the first non-scalar dataset it meets."""
    if isinstance(value, (dict, list)):
        import json

        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)
    return value


def _rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = [{k: _flatten_cell(v) for k, v in r.get("row", {}).items()} for r in rows]
    return pd.DataFrame.from_records(records)


def load_hf_dataset_sample(ref: str, max_rows: int = DEFAULT_SAMPLE_ROWS) -> dict[str, Any]:
    """Analyze a Hugging Face dataset from a bounded row sample, via the Datasets Server's
    pre-converted, ready-to-read JSON rows - no file format sniffing or byte-range streaming
    needed, since every public dataset is already normalized on Hugging Face's side."""
    config, split = _pick_config_and_split(ref)
    response = requests.get(
        f"{HF_DATASETS_SERVER}/rows",
        params={"dataset": ref, "config": config, "split": split, "offset": 0, "length": max(1, min(max_rows, 100))},
        headers=_headers(),
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 401:
        raise HFError(f"'{ref}' is private or gated. Add an HF_TOKEN with access to this dataset before analyzing it.")
    response.raise_for_status()
    data = response.json()
    total_rows = data.get("num_rows_total")

    rows = list(data.get("rows") or [])
    # the Datasets Server caps a single /rows call at 100 rows; page further offsets ourselves to
    # reach the caller's requested sample size (bounded, so this is at most a handful of calls).
    offset = len(rows)
    while len(rows) < max_rows and (total_rows is None or offset < total_rows):
        page = requests.get(
            f"{HF_DATASETS_SERVER}/rows",
            params={"dataset": ref, "config": config, "split": split, "offset": offset, "length": min(100, max_rows - len(rows))},
            headers=_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        if not page.ok:
            break
        page_rows = list(page.json().get("rows") or [])
        if not page_rows:
            break
        rows.extend(page_rows)
        offset += len(page_rows)

    df = _rows_to_frame(rows)
    if df.empty:
        raise HFError(f"'{ref}' returned no rows to analyze.")
    is_sample = total_rows is not None and len(df) < total_rows

    return {
        "path": None,
        "name": ref.replace("/", "-"),
        "file_type": "hf",
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dataframe": df,
        "sample": {
            "file_name": f"{ref.replace('/', '-')}.parquet",
            "config": config,
            "split": split,
            "is_sample": is_sample,
            "sample_rows": int(df.shape[0]),
            "estimated_total_rows": total_rows if total_rows is not None else int(df.shape[0]),
            "total_bytes": None,
        },
    }


def download_hf_dataset_full(ref: str, max_rows: int = MAX_FULL_ROWS) -> dict[str, Any]:
    """A full (not sample-bounded) read, via the same Parquet files Hugging Face already converted
    the dataset to - ``pandas.read_parquet`` against a stable URL, no custom download step. Still
    capped at ``max_rows`` as a last-resort guard against an accidental multi-gigabyte read."""
    config, split = _pick_config_and_split(ref)
    response = requests.get(f"{HF_DATASETS_SERVER}/parquet", params={"dataset": ref}, headers=_headers(), timeout=REQUEST_TIMEOUT)
    if response.status_code == 401:
        raise HFError(f"'{ref}' is private or gated. Add an HF_TOKEN with access to this dataset before analyzing it.")
    response.raise_for_status()
    files = [f for f in (response.json().get("parquet_files") or []) if f.get("config") == config and f.get("split") == split]
    if not files:
        raise HFError(f"'{ref}' has no Parquet export available for {config}/{split} yet.")

    frames = []
    total = 0
    for file_info in files:
        frame = pd.read_parquet(file_info["url"])
        frames.append(frame)
        total += len(frame)
        if total >= max_rows:
            break
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    truncated = len(df) > max_rows
    if truncated:
        df = df.head(max_rows)
    df = df.map(_flatten_cell) if hasattr(df, "map") else df.applymap(_flatten_cell)

    return {
        "path": None,
        "name": ref.replace("/", "-"),
        "file_type": "hf",
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dataframe": df,
        "sample": {"file_name": f"{ref.replace('/', '-')}.parquet", "config": config, "split": split, "is_sample": truncated,
                   "sample_rows": int(df.shape[0]), "estimated_total_rows": int(df.shape[0]) if not truncated else None, "total_bytes": None},
    }
