from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .encoding import decode_bytes, sniff_delimiter

DATASET_FILE_EXTENSIONS = {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".json"}


def _read_delimited(data: bytes, default_sep: str) -> pd.DataFrame:
    """Shared CSV/TSV parsing for both local-file and in-memory upload paths: detect the real
    text encoding and (for the ambiguous ".csv" case) the real delimiter, instead of assuming
    UTF-8 + comma - the assumption that silently corrupts a large fraction of real Kaggle CSVs
    into garbled column names and values."""
    text = decode_bytes(data)
    sep = sniff_delimiter(text, fallback=default_sep) if default_sep == "," else default_sep
    try:
        return pd.read_csv(io.StringIO(text), sep=sep)
    except Exception:
        return pd.read_csv(io.StringIO(text), sep=sep, engine="python", on_bad_lines="skip")


def find_dataset_files(directory: str | os.PathLike[str]) -> list[str]:
    root = Path(directory)
    if not root.exists():
        return []
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in DATASET_FILE_EXTENSIONS:
            files.append(str(path))
    return files


def load_dataset(dataset_path: str | os.PathLike[str], file_name: str | None = None) -> dict[str, Any]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    file_candidates = []
    if path.is_file():
        file_candidates = [path]
    else:
        file_candidates = [Path(p) for p in find_dataset_files(path)]
        if file_name:
            file_candidates = [p for p in file_candidates if p.name == file_name] or file_candidates

    if not file_candidates:
        raise ValueError(f"No supported dataset files found in: {dataset_path}")

    main_file = file_candidates[0]
    suffix = main_file.suffix.lower()

    if suffix == ".csv":
        df = _read_delimited(main_file.read_bytes(), default_sep=",")
    elif suffix == ".tsv":
        df = _read_delimited(main_file.read_bytes(), default_sep="\t")
    elif suffix == ".parquet":
        df = pd.read_parquet(main_file)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(main_file)
    elif suffix == ".json":
        df = pd.read_json(main_file)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")

    return {
        "path": str(main_file),
        "name": main_file.stem,
        "file_type": suffix.lstrip("."),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dataframe": df,
    }


def load_dataframe_from_bytes(filename: str, contents: bytes) -> pd.DataFrame:
    """Parse an uploaded file entirely in memory (no disk write), so a user-supplied CSV never
    needs to touch the server's filesystem to be analyzed."""
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(contents)
    if suffix == ".csv":
        return _read_delimited(contents, default_sep=",")
    if suffix == ".tsv":
        return _read_delimited(contents, default_sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(buffer)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(buffer)
    if suffix == ".json":
        return pd.read_json(buffer)
    raise ValueError(f"Unsupported dataset format: {suffix}")
