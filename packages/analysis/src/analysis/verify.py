"""Reproducibility: recompute an analysis's headline numbers from its stored data and steps.

A published analysis stores the original table, the ordered cleaning steps, and the headline metrics it
reported. ``verify`` replays the steps and recomputes every metric from scratch, so anyone can confirm
that the numbers really follow from the data - and that the result has not been edited since. The
``repro_hash`` fingerprints the whole chain (data + steps + metrics), so equal hashes mean an identical analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from . import report_deep as D
from . import report_stats as RS
from . import schema as schema_mod
from .profiler import analyze_dataframe
from .transforms import apply_transform

ENGINE = "datascopia-analysis-1"
TOLERANCE = 1e-6


def replay_steps(original: pd.DataFrame, steps: list[dict[str, Any]]) -> pd.DataFrame:
    frame = original
    for step in steps:
        frame, _ = apply_transform(frame, str(step["type"]), step.get("params") or {})
    return frame


def _r(value: float | None, digits: int = 6) -> float | None:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    return round(float(value), digits)


def compute_metrics(df: pd.DataFrame, name: str = "dataset") -> dict[str, Any]:
    """Deterministic headline numbers for a table (no models or sampling, so they reproduce exactly)."""
    analysis = analyze_dataframe(df, dataset_name=name)
    schema = schema_mod.build_schema(df)
    recs = analysis.get("recommendations") or []
    corr = RS.correlations(df, schema, top=5)
    groups = RS.group_tests(df, schema, recs, top=3)
    trends = D.trend_deep(df, schema, recs, limit=1)
    snapshot = analysis["research_snapshot"]
    series = (trends or {"series": []})["series"][:1]
    return {
        "rows": int(len(df)), "columns": int(df.shape[1]), "missing_pct": _r(snapshot["missing_percentage"], 2), "duplicate_rows": int(snapshot["duplicate_rows"]),
        "quality_score": int(analysis["data_quality"]["score"]),
        "correlations": [{"a": p["a"], "b": p["b"], "n": p["n"], "r": _r(p["r"]), "rho": _r(p["rho"])} for p in corr["pairs"]],
        "groups": [{"measure": g["measure"], "group": g["group"], "H": _r(g["H"], 4), "eta2": _r(g["eta2"])} for g in groups["rows"]],
        "trend": [{"measure": s["measure"], "slope": _r(s["slope"], 6), "tau": _r(s["tau"]), "periods": s["periods"]} for s in series],
        "descriptives": [{"column": d["column"], "mean": _r(d["mean"]), "sd": _r(d["sd"]), "median": _r(d["median"])} for d in RS.descriptives(df, schema, limit=6)],
    }


def make_hash(original_fp: str, steps: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    canonical = json.dumps({"engine": ENGINE, "data": original_fp, "steps": [{"type": s["type"], "params": s.get("params") or {}} for s in steps], "metrics": metrics},
                           sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= TOLERANCE * max(1.0, abs(float(a)))
    return a == b


def _flatten(prefix: str, value: Any, out: list[tuple[str, Any]]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            label = ""
            if isinstance(v, dict):
                label = ":".join(str(v[k]) for k in ("a", "b", "measure", "group", "column") if k in v)
            _flatten(f"{prefix}[{label or i}]", v, out)
    else:
        out.append((prefix, value))


LABELS = {"rows": "Rows after cleaning", "columns": "Columns after cleaning", "quality_score": "Data quality score", "missing_pct": "Missing cells (%)",
          "duplicate_rows": "Duplicate rows"}


def _label(path: str) -> str:
    if path in LABELS:
        return LABELS[path]
    for prefix, text in (("correlations", "Correlation"), ("groups", "Group test"), ("descriptives", "Summary"), ("trend", "Trend")):
        if path.startswith(prefix):
            return path.replace(prefix, text, 1)
    return path


def verify(original: pd.DataFrame, steps: list[dict[str, Any]], expected_metrics: dict[str, Any], expected_hash: str, expected_fp: str, name: str = "dataset") -> dict[str, Any]:
    """Recompute everything and compare. Never raises: a failure to replay is itself a failed verification."""
    result: dict[str, Any] = {"ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "engine": ENGINE, "software": RS.software_versions()}
    try:
        fingerprint = RS.data_fingerprint(original)
        final = replay_steps(original, steps)
        metrics = compute_metrics(final, name)
    except Exception as exc:  # noqa: BLE001 - reported to the user as a failed verification
        result.update({"verified": False, "checked": 0, "passed": 0, "checks": [], "error": f"Could not recompute the analysis: {exc}"})
        return result
    checks = [{"name": "Original data fingerprint", "expected": expected_fp, "got": fingerprint, "ok": fingerprint == expected_fp}]
    expected_flat: list[tuple[str, Any]] = []
    got_flat: list[tuple[str, Any]] = []
    _flatten("", expected_metrics, expected_flat)
    _flatten("", metrics, got_flat)
    got = dict(got_flat)
    for path, expected in expected_flat:
        checks.append({"name": _label(path), "expected": expected, "got": got.get(path), "ok": _close(expected, got.get(path))})
    hash_now = make_hash(fingerprint, steps, metrics)
    checks.append({"name": "Reproducibility hash", "expected": expected_hash, "got": hash_now, "ok": hash_now == expected_hash})
    passed = sum(1 for c in checks if c["ok"])
    result.update({"verified": passed == len(checks), "checked": len(checks), "passed": passed, "checks": checks, "repro_hash": hash_now})
    return result
