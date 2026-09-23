from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from .encoding import decode_bytes, sniff_delimiter
from .kaggle_search import KAGGLE_API_BASE, _kaggle_request_kwargs, validate_kaggle_credentials

TABULAR_EXTENSIONS_PRIORITY = [".csv", ".tsv", ".parquet", ".json", ".xlsx", ".xls"]
DEFAULT_SAMPLE_BYTES = 5 * 1024 * 1024  # 5 MB is enough for robust profiling on most tabular files
MAX_BUFFERED_BYTES = 20 * 1024 * 1024  # hard cap for formats that must be parsed as a whole (json/xlsx)


@dataclass
class RemoteFileInfo:
    name: str
    size: int | None


def list_dataset_files(dataset_slug: str) -> list[RemoteFileInfo]:
    if not validate_kaggle_credentials():
        raise ValueError("Kaggle credentials are not configured.")
    owner, slug = dataset_slug.split("/", 1)
    response = requests.get(
        f"{KAGGLE_API_BASE}/datasets/list/{quote(owner)}/{quote(slug)}",
        **_kaggle_request_kwargs(),
    )
    response.raise_for_status()
    files = response.json().get("datasetFiles", [])
    results = []
    for item in files:
        name = item.get("name")
        if not name:
            continue
        size = item.get("totalBytes") or item.get("total_bytes")
        results.append(RemoteFileInfo(name=name, size=int(size) if size else None))
    if not results:
        raise ValueError(f"Kaggle returned no files for dataset '{dataset_slug}'.")
    return results


def select_primary_file(files: list[RemoteFileInfo]) -> RemoteFileInfo:
    def rank(info: RemoteFileInfo) -> tuple[int, int]:
        suffix = "." + info.name.rsplit(".", 1)[-1].lower() if "." in info.name else ""
        try:
            ext_rank = TABULAR_EXTENSIONS_PRIORITY.index(suffix)
        except ValueError:
            ext_rank = len(TABULAR_EXTENSIONS_PRIORITY)
        return (ext_rank, -(info.size or 0))

    candidates = [
        info for info in files
        if any(info.name.lower().endswith(ext) for ext in TABULAR_EXTENSIONS_PRIORITY)
    ]
    if not candidates:
        raise ValueError("No supported tabular files were found in this dataset.")
    candidates.sort(key=rank)
    return candidates[0]


def _file_download_url(dataset_slug: str, file_name: str) -> str:
    owner, slug = dataset_slug.split("/", 1)
    # file_name is an opaque identifier that may itself contain "/" (e.g. files nested under an
    # output/ folder in notebook-sourced datasets), so it must be fully percent-encoded
    # (safe="") rather than treated as literal path segments, or Kaggle 404s on the request.
    return f"{KAGGLE_API_BASE}/datasets/download/{quote(owner)}/{quote(slug)}/{quote(file_name, safe='')}"


def resolve_direct_url(dataset_slug: str, file_name: str) -> str:
    """Resolve the Kaggle download redirect to a directly fetchable (often range-capable) URL
    without pulling the response body."""
    url = _file_download_url(dataset_slug, file_name)
    response = requests.get(url, stream=True, allow_redirects=True, **_kaggle_request_kwargs())
    try:
        response.raise_for_status()
        return response.url
    finally:
        response.close()


def stream_sample_bytes(
    dataset_slug: str, file_name: str, max_bytes: int = DEFAULT_SAMPLE_BYTES
) -> tuple[bytes, int | None, bool]:
    """Stream at most `max_bytes` of a dataset file and stop, instead of downloading the whole
    file. Returns (sampled_bytes, total_size_if_known, truncated)."""
    url = _file_download_url(dataset_slug, file_name)
    with requests.get(url, stream=True, allow_redirects=True, **_kaggle_request_kwargs()) as response:
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        total_size = int(content_length) if content_length and content_length.isdigit() else None

        buffer = io.BytesIO()
        truncated = False
        for chunk in response.iter_content(chunk_size=256 * 1024):
            if not chunk:
                continue
            buffer.write(chunk)
            if buffer.tell() >= max_bytes:
                truncated = True
                break
        return buffer.getvalue()[:max_bytes], total_size, truncated


def _estimate_total_rows(sample_rows: int, sample_bytes: int, total_bytes: int | None) -> int | None:
    if not total_bytes or not sample_bytes or sample_rows <= 0:
        return None
    return int(round(sample_rows * (total_bytes / sample_bytes)))


def dataframe_from_delimited_sample(data: bytes, truncated: bool, sep: str = ",") -> pd.DataFrame:
    """Parse a streamed byte sample of a CSV/TSV file, detecting its actual text encoding and
    delimiter rather than assuming UTF-8 and the extension's nominal separator - about a fifth
    to a third of real Kaggle CSVs are not UTF-8, and a ".csv" extension doesn't guarantee a
    comma delimiter (many European-locale exports use semicolons)."""
    text = decode_bytes(data)
    if truncated:
        last_newline = text.rfind("\n")
        if last_newline > 0:
            text = text[:last_newline]
    # Only re-sniff when the caller's default is the ambiguous comma case (".csv" doesn't
    # guarantee a comma delimiter); an explicit ".tsv" tab is trusted as-is, since sniffing a
    # tab-delimited file whose fields happen to contain lots of commas can misfire.
    detected_sep = sniff_delimiter(text, fallback=sep) if sep == "," else sep
    try:
        return pd.read_csv(io.StringIO(text), sep=detected_sep)
    except Exception:
        return pd.read_csv(io.StringIO(text), sep=detected_sep, engine="python", on_bad_lines="skip")


def sample_parquet_dataset(
    dataset_slug: str, file_name: str, max_rows: int
) -> tuple[pd.DataFrame, int | None]:
    """Read only as many row groups as needed from a remote parquet file using HTTP range
    requests, relying on parquet's footer metadata for the true row count without a full
    download."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import fsspec

    direct_url = resolve_direct_url(dataset_slug, file_name)
    fs = fsspec.filesystem("http")
    with fs.open(direct_url, "rb") as handle:
        parquet_file = pq.ParquetFile(handle)
        total_rows = parquet_file.metadata.num_rows if parquet_file.metadata else None
        batches = []
        collected = 0
        for batch in parquet_file.iter_batches(batch_size=min(50_000, max(1, max_rows))):
            batches.append(batch)
            collected += batch.num_rows
            if collected >= max_rows:
                break
        table = pa.Table.from_batches(batches) if batches else parquet_file.schema_arrow.empty_table()
        return table.to_pandas(), total_rows


def load_dataset_sample(
    dataset_slug: str, max_sample_bytes: int = DEFAULT_SAMPLE_BYTES
) -> dict[str, Any]:
    """Analyze a Kaggle dataset without downloading it in full. Streams only enough bytes (or,
    for parquet, only enough row groups) to build a representative sample."""
    files = list_dataset_files(dataset_slug)
    primary = select_primary_file(files)
    suffix = "." + primary.name.rsplit(".", 1)[-1].lower() if "." in primary.name else ""

    sample_info: dict[str, Any] = {
        "file_name": primary.name,
        "file_size_bytes": primary.size,
        "all_files": [f.name for f in files],
    }

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        data, total_bytes, truncated = stream_sample_bytes(dataset_slug, primary.name, max_sample_bytes)
        df = dataframe_from_delimited_sample(data, truncated, sep=sep)
        total_bytes = total_bytes or primary.size
        sample_info.update(
            {
                "is_sample": truncated,
                "sample_bytes": len(data),
                "sample_rows": int(df.shape[0]),
                "total_bytes": total_bytes,
                "estimated_total_rows": (
                    _estimate_total_rows(df.shape[0], len(data), total_bytes) if truncated else int(df.shape[0])
                ),
            }
        )
    elif suffix == ".parquet":
        max_rows = max(1, max_sample_bytes // 200)  # ~200 bytes/row heuristic
        df, total_rows = sample_parquet_dataset(dataset_slug, primary.name, max_rows=max_rows)
        sample_info.update(
            {
                "is_sample": total_rows is not None and df.shape[0] < total_rows,
                "sample_rows": int(df.shape[0]),
                "total_bytes": primary.size,
                "estimated_total_rows": total_rows,
            }
        )
    elif suffix == ".json":
        data, total_bytes, truncated = stream_sample_bytes(dataset_slug, primary.name, MAX_BUFFERED_BYTES)
        if truncated:
            raise ValueError(
                f"'{primary.name}' is larger than {MAX_BUFFERED_BYTES // (1024 * 1024)} MB and cannot be "
                "sampled as JSON. Run a full analysis or download the dataset instead."
            )
        df = pd.read_json(io.BytesIO(data))
        sample_info.update(
            {
                "is_sample": False,
                "sample_rows": int(df.shape[0]),
                "total_bytes": total_bytes or primary.size,
                "estimated_total_rows": int(df.shape[0]),
            }
        )
    elif suffix in {".xlsx", ".xls"}:
        data, total_bytes, truncated = stream_sample_bytes(dataset_slug, primary.name, MAX_BUFFERED_BYTES)
        if truncated:
            raise ValueError(
                f"'{primary.name}' is larger than {MAX_BUFFERED_BYTES // (1024 * 1024)} MB and Excel files "
                "cannot be partially sampled. Run a full analysis or download the dataset instead."
            )
        df = pd.read_excel(io.BytesIO(data))
        sample_info.update(
            {
                "is_sample": False,
                "sample_rows": int(df.shape[0]),
                "total_bytes": total_bytes or primary.size,
                "estimated_total_rows": int(df.shape[0]),
            }
        )
    else:
        raise ValueError(f"Unsupported file format for '{primary.name}'.")

    return {
        "path": None,
        "name": dataset_slug.replace("/", "-"),
        "file_type": suffix.lstrip("."),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dataframe": df,
        "sample": sample_info,
    }
