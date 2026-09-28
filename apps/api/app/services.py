"""Helpers shared by the route modules: turning a stored/derived dataframe into the same
`{dataset, analysis}` payload the analyze/upload endpoints return, and reading a session's frames."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from analysis.profiler import analyze_dataframe
from analysis.transforms import TransformError, apply_transform
from dataset.descriptions import extract_column_descriptions, infer_column_description
from dataset.license import describe_license
import workspace_cache


def original_cache_key(dataset_id: str) -> str:
    return f"{dataset_id}::original"


def attach_column_descriptions(dataset: dict[str, Any], df: pd.DataFrame, documented: dict[str, str] | None = None) -> None:
    """Documented descriptions (from the dataset's own text) plus name-based inferred ones."""
    documented = documented or {}
    dataset["column_descriptions"] = documented
    dataset["inferred_column_descriptions"] = {c: infer_column_description(c, df[c]) for c in df.columns if c not in documented}


def analysis_with_preview(df: pd.DataFrame, dataset_name: str, sample_info: dict[str, Any] | None = None) -> dict[str, Any]:
    analysis = analyze_dataframe(df, dataset_name=dataset_name, sample_info=sample_info)
    analysis["preview"] = json.loads(df.head(300).to_json(orient="records", date_format="iso"))
    return analysis


def session_frames(dataset_id: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(current, original) for a cached dataset; original == current when nothing was transformed."""
    current = workspace_cache.get(dataset_id)
    if current is None:
        return None, None
    original = workspace_cache.get(original_cache_key(dataset_id))
    return current, (original if original is not None else current)


def replay(original: pd.DataFrame, transform_log: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    """Re-apply a stored transform log to the original table.

    Returns (result, applied_entries, warnings). A step that no longer applies is skipped with a
    warning rather than failing the whole restore.
    """
    frame, applied, warnings = original, [], []
    for step in transform_log or []:
        if not isinstance(step, dict) or not step.get("type"):
            continue
        try:
            frame, description = apply_transform(frame, str(step["type"]), step.get("params") or {})
            applied.append({"type": step["type"], "params": step.get("params") or {}, "description": description})
        except TransformError as exc:
            warnings.append(f"Skipped '{step.get('description') or step['type']}': {exc}")
    return frame, applied, warnings


def open_in_workspace(original: pd.DataFrame, *, dataset_id: str, name: str, transform_log: list[dict[str, Any]] | None = None,
                      meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a stored table into the workspace cache and return the payload the UI expects.

    The original goes under the shadow key so "Reset to original data" and notebook replay keep working;
    the working copy has the stored transform log replayed on top.
    """
    meta = meta or {}
    current, applied, warnings = replay(original, transform_log or [])
    workspace_cache.put(dataset_id, current)
    workspace_cache.put(original_cache_key(dataset_id), original)
    dataset: dict[str, Any] = {
        "name": name, "dataset_id": dataset_id, "shape": {"rows": int(current.shape[0]), "columns": int(current.shape[1])},
        "columns": list(current.columns), "file_type": meta.get("file_type"),
    }
    kaggle_info = meta.get("kaggle_info") if isinstance(meta.get("kaggle_info"), dict) else {}
    if kaggle_info:
        dataset["kaggle_info"] = kaggle_info
        dataset["license_info"] = describe_license(kaggle_info.get("license"))
    if meta.get("sample"):
        dataset["sample"] = meta["sample"]
    documented = extract_column_descriptions(kaggle_info.get("description", ""), list(current.columns)) if kaggle_info else {}
    attach_column_descriptions(dataset, current, documented)
    return {"dataset": dataset, "analysis": analysis_with_preview(current, name, meta.get("sample")), "transform_log": applied, "warnings": warnings}
