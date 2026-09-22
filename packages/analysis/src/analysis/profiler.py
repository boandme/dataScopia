from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


NUMERIC_DTYPES = {"int64", "int32", "float64", "float32", "int16", "float16"}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _infer_column_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # Must come before is_numeric_dtype: pandas treats bool as a numeric subtype, so
    # is_numeric_dtype(bool_series) is True. Checking numeric first would silently
    # misclassify boolean columns as "numeric", sending them into IQR/quantile math where
    # `q1 - 1.5 * iqr` on boolean values raises "numpy boolean subtract... not supported".
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if series.nunique() <= 20 and series.astype(str).str.len().mean() < 25:
        return "categorical"
    return "text"


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
            results.append(
                {
                    "column_a": col_a,
                    "column_b": col_b,
                    "correlation": round(float(value), 4),
                }
            )
    results.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return results[:top_n]


def _suggest_target_candidates(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    candidates: list[str] = []
    for col in numeric_cols:
        if df[col].nunique() > 2 and df[col].nunique() < min(df.shape[0], 100):
            candidates.append(col)
    for col in categorical_cols:
        if df[col].nunique() > 2 and df[col].nunique() < min(df.shape[0], 50):
            candidates.append(col)
    return candidates[:10]


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


def _suggest_visualizations(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str], datetime_cols: list[str]) -> list[dict[str, Any]]:
    recommended: list[dict[str, Any]] = []
    if numeric_cols:
        main_numeric = numeric_cols[0]
        recommended.append({
            "type": "histogram",
            "title": f"Distribution of {main_numeric}",
            "x": main_numeric,
            "reason": "Shows central tendency and skewness in a key variable.",
        })
    if len(numeric_cols) >= 2:
        first, second = numeric_cols[:2]
        recommended.append({
            "type": "scatter",
            "title": f"{first} vs {second}",
            "x": first,
            "y": second,
            "reason": "Useful for assessing pairwise relationships and outliers.",
        })
    if numeric_cols and len(numeric_cols) > 1:
        recommended.append({
            "type": "correlation_heatmap",
            "title": "Correlation heatmap",
            "columns": numeric_cols[:10],
            "reason": "Highlights strong numerical relationships in the dataset.",
        })
    if categorical_cols:
        cat_col = categorical_cols[0]
        recommended.append({
            "type": "bar",
            "title": f"Counts by {cat_col}",
            "x": cat_col,
            "y": "count",
            "reason": "Summarizes category frequency and concentration patterns.",
        })
        if numeric_cols:
            recommended.append({
                "type": "box", "title": f"{numeric_cols[0]} by {cat_col}",
                "x": cat_col, "y": numeric_cols[0],
                "reason": "Compares spread and outliers across categories.",
            })
    if len(categorical_cols) >= 2:
        recommended.append({
            "type": "stacked_bar", "title": f"{categorical_cols[0]} by {categorical_cols[1]}",
            "x": categorical_cols[0], "y": categorical_cols[1],
            "reason": "Shows composition across two categorical dimensions.",
        })
    if datetime_cols:
        dt_col = datetime_cols[0]
        if numeric_cols:
            recommended.append({
                "type": "line",
                "title": f"{dt_col} vs {numeric_cols[0]}",
                "x": dt_col,
                "y": numeric_cols[0],
                "reason": "Useful when the dataset has time structure or temporal trends.",
            })
    return recommended[:8]


def analyze_dataframe(
    df: pd.DataFrame, dataset_name: str = "dataset", sample_info: dict[str, Any] | None = None
) -> dict[str, Any]:
    columns = list(df.columns)
    rows, cols = df.shape

    missing_by_column = {
        column: round(float(df[column].isna().mean() * 100), 2) if column in df.columns else 0.0
        for column in columns
    }

    type_summary = {}
    for column in columns:
        type_summary[column] = _infer_column_type(df[column])

    numeric_cols = [column for column, dtype in type_summary.items() if dtype == "numeric"]
    categorical_cols = [column for column, dtype in type_summary.items() if dtype == "categorical"]
    datetime_cols = [column for column, dtype in type_summary.items() if dtype == "datetime"]

    duplicate_rows = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())
    outlier_count = 0
    for column in numeric_cols:
        s = df[column].dropna()
        if s.empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count += int(((s < lower) | (s > upper)).sum())

    relation_summary = _rank_numeric_relationships(df, numeric_cols)
    skewed_fields = _count_skewed_fields(df, numeric_cols)
    correlations = relation_summary
    target_candidates = _suggest_target_candidates(df, numeric_cols, categorical_cols)
    recommendations = _suggest_visualizations(df, numeric_cols, categorical_cols, datetime_cols)

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
    if duplicate_rows:
        insights.append(f"{duplicate_rows} duplicate rows were detected.")
    if missing_cells:
        insights.append(f"{missing_cells} missing values were found across the dataset.")
    if skewed_fields:
        insights.append(f"Highly skewed variables were detected: {', '.join(skewed_fields[:5])}.")
    if target_candidates:
        insights.append(f"Potential target or label-like columns include: {', '.join(target_candidates[:5])}.")
    if correlations:
        strongest = correlations[0]
        insights.append(
            f"The strongest numeric relationship is between {strongest['column_a']} and {strongest['column_b']} (r = {strongest['correlation']})."
        )

    data_quality_issues: list[str] = []
    for column, pct in sorted(missing_by_column.items(), key=lambda item: item[1], reverse=True):
        if pct > 20:
            data_quality_issues.append(f"{column} has {pct}% missing values.")
    if duplicate_rows:
        data_quality_issues.append(f"{duplicate_rows} duplicate rows may affect downstream analysis.")
    if not numeric_cols:
        data_quality_issues.append("No numeric columns were detected; summary statistics may be limited.")

    constant_columns = [column for column in columns if df[column].nunique(dropna=False) <= 1]
    high_cardinality_columns = [column for column in columns if df[column].nunique(dropna=True) > max(50, rows * 0.8)]
    numeric_summary = {
        column: {
            "mean": round(float(df[column].mean()), 4),
            "median": round(float(df[column].median()), 4),
            "min": round(float(df[column].min()), 4),
            "max": round(float(df[column].max()), 4),
        }
        for column in numeric_cols
        if not df[column].dropna().empty
    }
    research_questions = []
    if categorical_cols and numeric_cols:
        research_questions.append(f"How does {numeric_cols[0]} vary across {categorical_cols[0]}?")
    if correlations:
        research_questions.append(f"What explains the relationship between {correlations[0]['column_a']} and {correlations[0]['column_b']}?")
    if datetime_cols and numeric_cols:
        research_questions.append(f"How does {numeric_cols[0]} change over {datetime_cols[0]}?")

    research_snapshot = {
        "dataset_name": dataset_name,
        "rows": rows,
        "columns": cols,
        "missing_percentage": round(float(df.isna().mean().mean() * 100), 2) if rows else 0.0,
        "duplicate_rows": duplicate_rows,
        "missing_columns": sum(1 for v in missing_by_column.values() if v > 0),
        "highly_skewed_fields": len(skewed_fields),
        "potential_targets": len(target_candidates),
        "notable_relationships": len(correlations),
        "data_quality_issues": len(data_quality_issues),
        "recommended_visualizations": len(recommendations),
        "numeric_fields": len(numeric_cols),
        "categorical_fields": len(categorical_cols),
        "datetime_fields": len(datetime_cols),
        "constant_columns": len(constant_columns),
        "high_cardinality_columns": len(high_cardinality_columns),
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
                "missing_pct": round(float(missing_by_column[column]), 2),
                "unique_count": int(df[column].nunique(dropna=True)),
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
        "research_snapshot": research_snapshot,
        "recommendations": recommendations,
        "insights": insights,
        "data_quality_issues": data_quality_issues,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "numeric_summary": numeric_summary,
        "research_questions": research_questions,
        "sample": sample_info,
    }
