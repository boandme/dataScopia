from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_FILE_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls", ".json"}


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
        df = pd.read_csv(main_file)
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
