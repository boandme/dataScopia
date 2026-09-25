from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import quality as quality_mod
from . import recommend as recommend_mod
from . import schema as schema_mod


NUMERIC_DTYPES = {"int64", "int32", "float64", "float32", "int16", "float16"}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _infer_column_type(series: pd.Series) -> str:
    """Coarse type of one column (kept for callers that only have a Series). The real
    classification - roles such as identifier, year, country, ordinal - lives in ``schema``."""
    if series.dropna().empty:
        return "unknown"
    profile = schema_mod.classify_column(str(series.name), series)
    return schema_mod.legacy_type(profile["role"], profile["dtype"], profile.get("notes", {}))


def _measure_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The numeric view of the measure columns, reading text like '$1,299' as numbers."""
    data: dict[str, pd.Series] = {}
    for column in columns:
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            data[column] = series.astype(float)
        elif pd.api.types.is_numeric_dtype(series):
            data[column] = series
        else:
            data[column] = schema_mod.coerce_numeric(series)
    return pd.DataFrame(data, index=df.index)


def _rank_numeric_relationships(df: pd.DataFrame, numeric_cols: list[str], top_n: int = 5) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if len(numeric_cols) < 2:
        return results
    corr = df[numeric_cols].corr(numeric_only=True)
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.isna(value):
                continue
            results.append({"column_a": col_a, "column_b": col_b, "correlation": round(float(value), 4)})
    results.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return results[:top_n]


def _count_skewed_fields(df: pd.DataFrame, numeric_cols: list[str]) -> list[str]:
    skewed: list[str] = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 3:
            continue
        value = s.skew()
        if pd.notna(value) and abs(float(value)) > 1.0:
            skewed.append(col)
    return skewed


def _research_questions(schema: dict[str, Any], recs: list[dict[str, Any]], targets: list[dict[str, Any]], correlations: list[dict[str, Any]]) -> list[str]:
    questions: list[str] = []
    by_type = {}
    for rec in recs:
        by_type.setdefault(rec["type"], rec)
    if "timeseries" in by_type:
        rec = by_type["timeseries"]
        questions.append(f"How has {rec['y']} changed over {rec['x']}, and is the change steady or driven by a few periods?")
    if "map" in by_type and by_type["map"].get("y"):
        rec = by_type["map"]
        questions.append(f"Which places have the highest and lowest {rec['y']}, and what do the top ones have in common?")
    if targets:
        target = targets[0]["column"]
        target_recs = [r for r in recs if r["type"] == "target" and r.get("y") == target]
        if target_recs:
            questions.append(f"What best predicts {target}? ({target_recs[0]['x']} looks like the strongest lead.)")
        else:
            questions.append(f"Which columns are associated with {target}?")
    for rec in recs:
        if rec["type"] == "distribution" and rec.get("group"):
            measure, grouper = rec["x"], rec["group"]
        elif rec["type"] == "box" and rec.get("x") and rec.get("y"):
            measure, grouper = rec["y"], rec["x"]
        else:
            continue
        questions.append(f"Why does {measure} differ across {grouper}, and is that difference practically meaningful?")
        break
    if correlations and abs(correlations[0]["correlation"]) >= 0.3:
        strongest = correlations[0]
        questions.append(f"What explains the relationship between {strongest['column_a']} and {strongest['column_b']}?")
    return questions[:6]


def analyze_dataframe(
    df: pd.DataFrame, dataset_name: str = "dataset", sample_info: dict[str, Any] | None = None
) -> dict[str, Any]:
    columns = list(df.columns)
    rows, cols = df.shape

    missing_by_column = {
        column: round(float(df[column].isna().mean() * 100), 2) if column in df.columns else 0.0
        for column in columns
    }

    schema = schema_mod.build_schema(df)
    profiles = schema["profiles"]
    type_summary = {column: profiles[column]["type"] for column in columns}

    measure_cols = list(schema["measures"])
    continuous_cols = [c for c in measure_cols if profiles[c]["role"] == "numeric"]
    categorical_cols = [c for c in columns if profiles[c]["role"] in {"categorical", "ordinal", "geo_country", "geo_state"}
                        or (profiles[c]["role"] == "binary" and not pd.api.types.is_bool_dtype(df[c]))]
    datetime_cols = [c for c in columns if profiles[c]["role"] == "datetime"]
    numeric_cols = measure_cols

    work = _measure_frame(df, measure_cols)

    duplicate_rows = int(df.duplicated().sum()) if not df.empty else 0
    missing_cells = int(df.isna().sum().sum())
    outlier_count = 0
    for column in continuous_cols:
        s = work[column].dropna()
        if s.empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count += int(((s < lower) | (s > upper)).sum())

    relationship_summary = _rank_numeric_relationships(work, measure_cols)
    skewed_fields = _count_skewed_fields(work, continuous_cols)
    correlations = relationship_summary

    targets = recommend_mod.suggest_targets(df, schema)
    target_candidates = [t["column"] for t in targets]
    recommendations = recommend_mod.recommend_visualizations(df, schema, correlations, targets, skewed_fields, missing_by_column)

    quality_report = quality_mod.build_quality_report(df, schema, targets)
    data_quality_issues = [f"{issue['title']}. {issue['detail']}" for issue in quality_report["issues"] if issue["severity"] in {"high", "medium"}]
    if not measure_cols:
        data_quality_issues.append("No numeric columns were detected; summary statistics may be limited.")

    insights: list[str] = []
    if sample_info and sample_info.get("is_sample"):
        sample_bytes = sample_info.get("sample_bytes")
        if not sample_bytes:
            sample_mb = "a partial download"
        elif sample_bytes < 1024 * 1024:
            sample_mb = f"{sample_bytes / 1024:.0f} KB"
        else:
            sample_mb = f"{sample_bytes / (1024 * 1024):.1f} MB"
        estimated_total = sample_info.get("estimated_total_rows")
        estimate_note = f" out of an estimated {estimated_total:,} total rows" if estimated_total else ""
        insights.append(
            f"This analysis is based on a {sample_mb} sample ({rows:,} rows){estimate_note}. "
            "Run a full analysis or download the dataset for complete results."
        )
    insights.extend(recommend_mod.describe_structure(schema))
    if duplicate_rows:
        insights.append(f"{duplicate_rows} duplicate rows were detected.")
    if missing_cells:
        insights.append(f"{missing_cells} missing values were found across the dataset.")
    if skewed_fields:
        insights.append(f"Highly skewed variables were detected: {', '.join(skewed_fields[:5])}.")
    if targets:
        top = targets[0]
        insights.append(f"Most likely target: '{top['column']}' ({top['kind']}) - {top['reason']}.")
    if correlations:
        strongest = correlations[0]
        insights.append(
            f"The strongest numeric relationship is between {strongest['column_a']} and {strongest['column_b']} (r = {strongest['correlation']})."
        )

    constant_columns = [column for column in columns if profiles[column]["role"] in {"constant", "empty"}]
    high_cardinality_columns = [column for column in columns if profiles[column]["unique_count"] > max(50, rows * 0.8)]
    numeric_summary = {}
    for column in columns:
        if profiles[column]["role"] not in schema_mod.MEASURE_ROLES | {"year", "geo_lat", "geo_lon"}:
            continue
        values = _measure_frame(df, [column])[column].dropna()
        if values.empty:
            continue
        numeric_summary[column] = {
            "mean": round(float(values.mean()), 4),
            "median": round(float(values.median()), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
        }
    research_questions = _research_questions(schema, recommendations, targets, correlations)

    role_counts = schema["role_counts"]
    research_snapshot = {
        "dataset_name": dataset_name,
        "rows": rows,
        "columns": cols,
        "missing_percentage": round(float(df.isna().mean().mean() * 100), 2) if rows else 0.0,
        "duplicate_rows": duplicate_rows,
        "missing_columns": sum(1 for v in missing_by_column.values() if v > 0),
        "highly_skewed_fields": len(skewed_fields),
        "potential_targets": len(target_candidates),
        "notable_relationships": len([r for r in correlations if abs(r["correlation"]) >= 0.3]),
        "data_quality_issues": len(quality_report["issues"]),
        "recommended_visualizations": len(recommendations),
        "numeric_fields": len(measure_cols),
        "categorical_fields": len(categorical_cols),
        "datetime_fields": len(datetime_cols),
        "identifier_fields": role_counts.get("identifier", 0),
        "text_fields": role_counts.get("text", 0),
        "geo_fields": sum(1 for v in schema["geo"].values() if v),
        "has_time_series": bool(schema["time"]),
        "join_key_candidates": len(schema["join_keys"]),
        "constant_columns": len(constant_columns),
        "high_cardinality_columns": len(high_cardinality_columns),
        "quality_score": quality_report["score"],
        "is_sample": bool(sample_info and sample_info.get("is_sample")),
        "estimated_total_rows": (sample_info or {}).get("estimated_total_rows"),
    }

    return {
        "dataset_name": dataset_name,
        "shape": {"rows": int(rows), "columns": int(cols)},
        "columns": [
            {
                "name": column,
                "type": type_summary[column],
                "role": profiles[column]["role"],
                "role_label": profiles[column]["role_label"],
                "notes": profiles[column].get("notes", {}),
                "missing_pct": round(float(missing_by_column[column]), 2),
                "unique_count": int(profiles[column]["unique_count"]),
            }
            for column in columns
        ],
        "type_summary": type_summary,
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicate_rows,
        "missing_cells": missing_cells,
        "outlier_count": int(outlier_count),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "relationship_summary": correlations,
        "skewed_fields": skewed_fields,
        "target_candidates": target_candidates,
        "target_details": targets,
        "research_snapshot": research_snapshot,
        "recommendations": recommendations,
        "insights": insights,
        "data_quality_issues": data_quality_issues,
        "data_quality": quality_report,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "numeric_summary": numeric_summary,
        "research_questions": research_questions,
        "schema": {
            "roles": {c: {"role": p["role"], "role_label": p["role_label"], "notes": p.get("notes", {}), "unique_count": p["unique_count"]} for c, p in profiles.items()},
            "geo": schema["geo"],
            "time": schema["time"],
            "join_keys": schema["join_keys"],
            "role_counts": role_counts,
        },
        "sample": sample_info,
    }
