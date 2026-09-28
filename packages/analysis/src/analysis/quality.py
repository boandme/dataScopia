"""Data-quality report (spec 5.12): what's wrong with this data that would mislead an analysis.

Unlike a dump of raw metrics, every finding here states *what* is wrong, the *evidence* (counts,
example values), *why it matters*, and - where a safe repair exists - a machine-readable ``fix``
that maps onto a dataset-editor transform, so the UI can offer to apply it in one click and the
notebook export can reproduce it.

Categories: missingness, duplicates, inconsistent formats, suspicious values, outliers,
high-cardinality and constant columns, target-leakage risk, and schema problems.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from . import schema as schema_mod

# Tokens that mean "no value" but are stored as text, which silently defeats every missing-value check.
MISSING_TOKENS = {"n/a", "na", "n.a.", "nan", "null", "none", "nil", "-", "--", "?", "unknown", "missing", "undefined",
                  "not available", "#n/a", "#value!", "tbd", "."}
# Sentinel codes people use in place of a missing number.
SENTINELS = [-1, -9, -99, -999, -9999, 99, 999, 9999, 99999]

_PLAUSIBLE_RANGES = [
    (re.compile(r"(^|[_\s])age($|[_\s])", re.I), 0, 120, "an age"),
    (re.compile(r"percent|pct|_pc$", re.I), 0, 100, "a percentage"),
    (re.compile(r"price|cost|amount|salary|income|revenue|fare|fee|spend|balance_due|quantity|qty|units|population|weight|height|length|distance|duration|area|size|count", re.I), 0, None, "a quantity that can't be negative"),
]

SEVERITY_WEIGHT = {"high": 12, "medium": 5, "low": 2}


def _issue(category: str, severity: str, columns: list[str], title: str, detail: str, why: str, fix: dict[str, Any] | None = None,
           examples: list[Any] | None = None) -> dict[str, Any]:
    return {
        "category": category, "severity": severity, "columns": columns, "title": title, "detail": detail, "why": why,
        "fix": fix, "examples": [str(e) for e in (examples or [])][:4],
    }


def _text_columns(df: pd.DataFrame, schema: dict[str, Any]) -> list[str]:
    return [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]


def _check_missing(df: pd.DataFrame, schema: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    profiles = schema["profiles"]
    pct = df.isna().mean() * 100
    for column, value in pct.sort_values(ascending=False).items():
        if profiles[column]["role"] == "empty":
            issues.append(_issue("missingness", "high", [column], f"'{column}' is completely empty",
                                 "Every value is missing.", "An empty column carries no information and can break code that expects values.",
                                 fix={"type": "drop_column", "params": {"column": column}}))
        elif value >= 20:
            issues.append(_issue("missingness", "high", [column], f"'{column}' is {value:.0f}% missing",
                                 f"{int(df[column].isna().sum()):,} of {len(df):,} rows have no value.",
                                 "Any statistic or model using this column silently loses that share of the rows, and the loss may not be random."))
        elif value >= 5:
            issues.append(_issue("missingness", "medium", [column], f"'{column}' is {value:.1f}% missing",
                                 f"{int(df[column].isna().sum()):,} rows have no value.", "Consider whether missing rows differ from the rest before filling or dropping them."))
    heavy = [c for c, v in pct.items() if v >= 5]
    if 2 <= len(heavy) <= 40:
        miss = df[heavy].isna().astype(float)
        corr = miss.corr().fillna(0)
        seen: set[frozenset[str]] = set()
        for i, a in enumerate(heavy):
            for b in heavy[i + 1:]:
                if corr.loc[a, b] >= 0.9 and frozenset({a, b}) not in seen:
                    seen.add(frozenset({a, b}))
                    issues.append(_issue("missingness", "low", [a, b], f"'{a}' and '{b}' are missing in the same rows",
                                         f"Their missingness overlaps almost perfectly (correlation {corr.loc[a, b]:.2f}).",
                                         "Missing-together columns usually mean a whole block of information was never recorded for those rows."))
                    if len(seen) >= 3:
                        return


def _check_duplicates(df: pd.DataFrame, schema: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    profiles = schema["profiles"]
    try:
        duplicates = int(df.duplicated().sum())
    except TypeError:
        duplicates = 0
    if duplicates:
        share = duplicates / max(1, len(df)) * 100
        issues.append(_issue("duplicates", "high" if share >= 5 else "medium", [], f"{duplicates:,} duplicate rows ({share:.1f}%)",
                             "These rows are exact copies of another row.",
                             "Duplicates over-weight some observations, inflating counts and biasing averages.",
                             fix={"type": "drop_duplicates", "params": {}}))
    for column, profile in profiles.items():
        if profile["role"] != "identifier" or not schema_mod._looks_like_id_name(column):
            continue
        repeated = int(df[column].dropna().duplicated().sum())
        if repeated and profile["unique_ratio"] >= 0.9:
            issues.append(_issue("duplicates", "high", [column], f"'{column}' repeats {repeated:,} times",
                                 "This looks like a unique identifier, but some values appear more than once.",
                                 "Non-unique IDs break joins and usually mean duplicated or mis-keyed records.",
                                 examples=df.loc[df[column].duplicated(keep=False), column].dropna().unique()[:3].tolist()))


def _check_formats(df: pd.DataFrame, schema: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    profiles = schema["profiles"]
    for column in _text_columns(df, schema):
        role = profiles[column]["role"]
        series = df[column].dropna().astype(str)
        if series.empty or role in {"text", "identifier"}:
            continue
        stripped = series.str.strip()
        lowered = stripped.str.lower()

        tokens = lowered[lowered.isin(MISSING_TOKENS) | (lowered == "")]
        if len(tokens) >= 1 and (len(tokens) / len(series) >= 0.005 or len(tokens) >= 3):
            found = sorted({v for v in stripped[lowered.isin(MISSING_TOKENS) | (lowered == "")].unique()})
            issues.append(_issue("formats", "medium" if len(tokens) / len(series) >= 0.02 else "low", [column],
                                 f"'{column}' uses placeholder text for missing values",
                                 f"{len(tokens):,} values are placeholders such as {', '.join(repr(t) for t in found[:4])} rather than real blanks.",
                                 "Placeholders escape every missing-value check and get treated as real categories.",
                                 fix={"type": "replace_missing_tokens", "params": {"column": column, "values": found[:12]}}, examples=found))

        if role in {"categorical", "binary", "ordinal"} and series.nunique() <= 300:
            variants = stripped.groupby(lowered).nunique()
            variants = variants[variants > 1]
            whitespace = int((series != stripped).sum())
            if len(variants):
                sample = stripped[lowered.isin(variants.index[:2])].unique()[:4]
                issues.append(_issue("formats", "medium", [column], f"'{column}' spells the same value several ways",
                                     f"{len(variants)} value(s) appear in more than one case or spacing, e.g. {', '.join(repr(s) for s in sample)}.",
                                     "Each spelling is counted as a separate category, splitting one group into several.",
                                     fix={"type": "standardize_text", "params": {"column": column}}, examples=sample))
            elif whitespace:
                issues.append(_issue("formats", "low", [column], f"'{column}' has stray whitespace",
                                     f"{whitespace:,} values have leading or trailing spaces.", "Invisible spaces make equal values look different.",
                                     fix={"type": "standardize_text", "params": {"column": column}}))


def _check_suspicious(df: pd.DataFrame, schema: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    profiles = schema["profiles"]
    for column, profile in profiles.items():
        if profile["role"] not in {"numeric", "ordinal_numeric", "year"}:
            continue
        values = schema_mod.coerce_numeric(df[column]) if not pd.api.types.is_numeric_dtype(df[column]) else df[column]
        values = values.dropna()
        if len(values) < 10:
            continue
        for pattern, low, high, label in _PLAUSIBLE_RANGES:
            if pattern.search(str(column)):
                bad = values[(values < low) | ((values > high) if high is not None else False)]
                if len(bad):
                    span = f"[{low}, {high}]" if high is not None else f"≥ {low}"
                    issues.append(_issue("suspicious", "high" if len(bad) / len(values) >= 0.01 else "medium", [column],
                                         f"'{column}' has {len(bad):,} implausible values",
                                         f"'{column}' looks like {label} (expected {span}) but ranges from {values.min():,.4g} to {values.max():,.4g}.",
                                         "Impossible values are usually entry errors or codes for 'unknown', and they distort means and correlations.",
                                         fix={"type": "filter", "params": {"column": column, "comparator": ">=", "value": str(low)}} if high is None else None,
                                         examples=bad.unique()[:4].tolist()))
                break
        if profile["role"] == "numeric" and len(values) >= 30:
            for sentinel in SENTINELS:
                hits = int((values == sentinel).sum())
                if hits and hits / len(values) >= 0.005:
                    rest = values[values != sentinel]
                    q1, q3 = rest.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    if iqr > 0 and (sentinel > q3 + 3 * iqr or sentinel < q1 - 3 * iqr):
                        issues.append(_issue("suspicious", "medium", [column], f"'{column}' contains the code {sentinel}",
                                             f"{hits:,} rows ({hits / len(values):.1%}) equal {sentinel}, far outside the other values (which run {rest.min():,.4g} to {rest.max():,.4g}).",
                                             "Codes like -1, 99 or 999 usually stand for 'unknown' and pull averages far from the truth.",
                                             fix={"type": "filter", "params": {"column": column, "comparator": "!=", "value": str(sentinel)}}))
                        break
    now = datetime.now()
    for column, profile in profiles.items():
        if profile["role"] != "datetime":
            continue
        times = schema_mod.time_values(df, column, profile).dropna()
        if times.empty:
            continue
        future = int((times > now + timedelta(days=30)).sum())
        ancient = int((times < datetime(1900, 1, 1)).sum())
        if future:
            issues.append(_issue("suspicious", "low", [column], f"'{column}' has {future:,} dates in the future",
                                 f"The latest is {times.max().date()}.", "Future dates are often typos (wrong year) unless this data is a forecast."))
        if ancient:
            issues.append(_issue("suspicious", "medium", [column], f"'{column}' has {ancient:,} dates before 1900",
                                 f"The earliest is {times.min().date()}.", "Dates like 1900-01-01 or 1970-01-01 are usually default placeholders, not real events."))
        notes = profile.get("notes", {})
        if notes.get("stored_as_text") and notes.get("parsed_fraction", 1) < 0.999:
            bad = int(round((1 - notes["parsed_fraction"]) * df[column].notna().sum()))
            issues.append(_issue("formats", "medium", [column], f"'{column}' has dates in inconsistent formats",
                                 f"About {bad:,} values don't match the dominant date format and can't be read as dates.",
                                 "Unreadable dates become missing when converted, silently thinning the time series."))


def _check_outliers(df: pd.DataFrame, schema: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    profiles = schema["profiles"]
    for column, profile in profiles.items():
        if profile["role"] != "numeric":
            continue
        values = (df[column] if pd.api.types.is_numeric_dtype(df[column]) else schema_mod.coerce_numeric(df[column])).dropna()
        if len(values) < 30:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        extreme = values[(values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)]
        share = len(extreme) / len(values)
        if len(extreme) and share >= 0.005:
            issues.append(_issue("outliers", "medium" if share >= 0.02 else "low", [column],
                                 f"'{column}' has {len(extreme):,} extreme outliers ({share:.1%})",
                                 f"Values beyond 3×IQR of the middle half, out to {extreme.abs().max():,.4g}.",
                                 "A few extreme values can dominate averages, regressions and axis scales; decide whether they are errors or real.",
                                 fix={"type": "drop_outliers", "params": {"column": column}}))


def _check_structure(df: pd.DataFrame, schema: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    profiles = schema["profiles"]
    n = len(df)
    for column, profile in profiles.items():
        if profile["role"] == "constant":
            issues.append(_issue("structure", "low", [column], f"'{column}' has a single value",
                                 f"Every non-missing row equals {df[column].dropna().iloc[0]!r}.", "A constant column can't explain anything and only adds noise to models.",
                                 fix={"type": "drop_column", "params": {"column": column}}))
        elif profile["role"] in {"categorical", "binary"} and n >= 50:
            top = df[column].value_counts(normalize=True)
            if len(top) and float(top.iloc[0]) >= 0.99:
                issues.append(_issue("structure", "low", [column], f"'{column}' is almost constant",
                                     f"{top.iloc[0]:.1%} of rows are {top.index[0]!r}.", "Near-constant columns give models almost nothing to learn from."))
        elif profile["role"] == "categorical" and profile.get("notes", {}).get("cardinality") == "high":
            issues.append(_issue("structure", "low", [column], f"'{column}' has very many categories",
                                 f"{profile['unique_count']:,} distinct values across {n:,} rows.", "Groups this fine are rarely large enough to compare; consider bucketing rare values."))
        notes = profile.get("notes", {})
        if notes.get("stored_as_text") and profile["role"] == "numeric":
            issues.append(_issue("schema", "medium", [column], f"'{column}' is numeric but stored as text",
                                 "Values carry symbols like $, %, or thousands separators, so they load as text.",
                                 "Text can't be averaged, sorted numerically or correlated until it's converted.",
                                 fix={"type": "cast_type", "params": {"column": column, "target_type": "numeric"}}))
        elif notes.get("stored_as_text") and profile["role"] == "datetime":
            issues.append(_issue("schema", "medium", [column], f"'{column}' holds dates stored as text",
                                 "The values are dates but loaded as plain strings.", "Text dates sort alphabetically and can't be grouped by month or year.",
                                 fix={"type": "cast_type", "params": {"column": column, "target_type": "datetime"}}))
        elif profile["role"] in {"categorical", "text"} and profile["dtype"] == "object":
            series = df[column].dropna().astype(str).head(500)
            if len(series) >= 20:
                fraction = float(schema_mod.coerce_numeric(series).notna().mean())
                if 0.1 <= fraction <= 0.9:
                    issues.append(_issue("schema", "medium", [column], f"'{column}' mixes numbers and text",
                                         f"About {fraction:.0%} of values look like numbers and the rest are words.", "Mixed columns can't be treated as either a number or a category reliably."))
    names = [str(c) for c in df.columns]
    unnamed = [c for c in names if c.lower().startswith("unnamed")]
    if unnamed:
        issues.append(_issue("schema", "low", unnamed[:3], f"{len(unnamed)} unnamed column(s)", "Columns like 'Unnamed: 0' usually come from a saved row index.",
                             "They're an artifact of how the file was saved, not data.", fix={"type": "drop_column", "params": {"column": unnamed[0]}} if len(unnamed) == 1 else None))
    padded = [c for c in names if c != c.strip()]
    if padded:
        issues.append(_issue("schema", "low", padded[:3], f"{len(padded)} column name(s) have stray spaces", "Leading or trailing spaces in a name make it easy to mistype.",
                             "Code referring to the name without the space fails with a confusing KeyError.",
                             fix={"type": "rename_column", "params": {"column": padded[0], "new_name": padded[0].strip()}}))
    normalized: dict[str, list[str]] = {}
    for c in names:
        normalized.setdefault(re.sub(r"[\s_\-]+", "", c.lower()), []).append(c)
    clashes = [v for v in normalized.values() if len(v) > 1]
    if clashes:
        issues.append(_issue("schema", "low", clashes[0], "Column names that differ only by case or spacing", f"e.g. {', '.join(repr(c) for c in clashes[0])}.",
                             "These are often the same field loaded twice."))
    if df.shape[1] == 1 and n:
        first = str(df.iloc[0, 0])
        if any(d in first for d in (";", "\t", "|")) or len(str(df.columns[0])) > 40:
            issues.append(_issue("schema", "high", [str(df.columns[0])], "Only one column was detected", "The single column contains delimiter characters.",
                                 "The file probably uses a different separator than expected, so every field is squashed into one column."))


def _check_leakage(df: pd.DataFrame, schema: dict[str, Any], targets: list[dict[str, Any]], issues: list[dict[str, Any]]) -> None:
    """Features that give away the target: they'd make a model look brilliant and then fail on new data."""
    profiles = schema["profiles"]
    sample = df if len(df) <= 20000 else df.sample(20000, random_state=0)
    for target in targets[:2]:
        name = target["column"]
        tgt_numeric = pd.api.types.is_numeric_dtype(sample[name]) or pd.api.types.is_bool_dtype(sample[name])
        for column in df.columns:
            if column == name or profiles[column]["role"] in {"identifier", "text", "empty", "constant", "unknown"}:
                continue
            try:
                if tgt_numeric and pd.api.types.is_numeric_dtype(sample[column]) and not pd.api.types.is_bool_dtype(sample[column]):
                    pair = pd.concat([sample[column].astype(float), sample[name].astype(float)], axis=1).dropna()
                    if len(pair) >= 30 and pair.iloc[:, 0].nunique() > 1:
                        r = float(pair.corr().iloc[0, 1])
                        if abs(r) >= 0.98:
                            issues.append(_issue("leakage", "high", [column, name], f"'{column}' almost duplicates the target '{name}'",
                                                 f"Correlation with '{name}' is {r:.3f}.", "A near-copy of the target lets a model 'predict' it without learning anything real; it won't be available when predicting genuinely new cases."))
                elif profiles[column]["unique_count"] <= 30 and profiles[column]["unique_count"] < 0.5 * len(sample):
                    frame = pd.DataFrame({"f": sample[column].astype(str), "t": sample[name].astype(str)}).dropna()
                    if len(frame) >= 30 and frame["t"].nunique() >= 2:
                        per_value = frame.groupby("f")["t"].nunique()
                        counts = frame["f"].value_counts()
                        determined = float(counts[per_value[per_value == 1].index].sum() / len(frame))
                        if determined >= 0.999 and frame["f"].nunique() >= 2 and profiles[column]["unique_count"] >= profiles[name]["unique_count"]:
                            if frame["f"].nunique() < len(frame) * 0.3:
                                issues.append(_issue("leakage", "high", [column, name], f"'{column}' perfectly determines the target '{name}'",
                                                     f"Every value of '{column}' maps to exactly one value of '{name}'.",
                                                     "That is usually a derived or post-outcome column (leakage), not a genuine predictor."))
            except Exception:
                continue
    numeric = [c for c in schema["measures"] if profiles[c]["role"] == "numeric"][:25]
    if len(numeric) >= 2:
        corr = pd.DataFrame({c: df[c] if pd.api.types.is_numeric_dtype(df[c]) else schema_mod.coerce_numeric(df[c]) for c in numeric}).corr().abs()
        arr = corr.to_numpy(copy=True)
        np.fill_diagonal(arr, 0)
        flagged = 0
        for i, a in enumerate(corr.columns):
            for j in range(i + 1, len(corr.columns)):
                if arr[i, j] >= 0.999 and flagged < 3:
                    flagged += 1
                    b = corr.columns[j]
                    issues.append(_issue("leakage", "medium", [a, b], f"'{a}' and '{b}' carry the same information",
                                         f"Their correlation is {arr[i, j]:.4f}.", "Perfectly redundant columns (unit conversions, copies) double-count one signal and destabilise regressions.",
                                         fix={"type": "drop_column", "params": {"column": b}}))


def build_quality_report(df: pd.DataFrame, schema: dict[str, Any], targets: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for check in (_check_missing, _check_duplicates, _check_formats, _check_suspicious, _check_outliers, _check_structure):
        try:
            check(df, schema, issues)
        except Exception:
            continue
    try:
        _check_leakage(df, schema, targets, issues)
    except Exception:
        pass
    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: order[i["severity"]])
    counts = {level: sum(1 for i in issues if i["severity"] == level) for level in ("high", "medium", "low")}
    deduction = min(100, sum(SEVERITY_WEIGHT[i["severity"]] for i in issues))
    score = max(0, 100 - deduction)
    label = "Good" if score >= 85 else "Fair" if score >= 60 else "Needs attention"
    by_category: dict[str, int] = {}
    for issue in issues:
        by_category[issue["category"]] = by_category.get(issue["category"], 0) + 1
    return {"score": score, "label": label, "counts": counts, "by_category": by_category, "issues": issues,
            "rows": int(len(df)), "columns": int(df.shape[1])}


QUALITY_FORMATS = {"md": ("text/markdown", "md"), "json": ("application/json", "json"), "csv": ("text/csv", "csv")}


def normalize_quality_options(options: dict[str, Any] | None) -> dict[str, Any]:
    o = options if isinstance(options, dict) else {}
    severities = [s for s in (o.get("severities") or ["high", "medium", "low"]) if s in {"high", "medium", "low"}] or ["high", "medium", "low"]
    return {
        "format": o.get("format") if o.get("format") in QUALITY_FORMATS else "md",
        "severities": severities,
        "include_why": o.get("include_why", True) is not False,
        "include_examples": o.get("include_examples", True) is not False,
        "include_fixes": o.get("include_fixes", True) is not False,
        "group_by": o.get("group_by") if o.get("group_by") in {"category", "severity"} else "category",
        "title": str(o.get("title") or "").strip()[:140],
        "author": str(o.get("author") or "").strip()[:120],
        "notes": str(o.get("notes") or "").strip()[:1500],
    }


def quality_export(report: dict[str, Any], dataset_name: str, options: dict[str, Any] | None = None) -> tuple[str, str, str]:
    """(content, mimetype, extension) for the quality report in the chosen format, honouring the
    severity filter and which explanatory parts (why / examples / fixes) to include."""
    import csv
    import io
    import json as _json

    opt = normalize_quality_options(options)
    issues = [i for i in report["issues"] if i["severity"] in opt["severities"]]
    mimetype, ext = QUALITY_FORMATS[opt["format"]]
    if opt["format"] == "json":
        payload = {"dataset": dataset_name, "score": report["score"], "label": report["label"], "rows": report["rows"], "columns": report["columns"], "counts": report["counts"],
                   "issues": [{"category": i["category"], "severity": i["severity"], "columns": i["columns"], "title": i["title"], "detail": i["detail"],
                               **({"why": i["why"]} if opt["include_why"] else {}), **({"examples": i["examples"]} if opt["include_examples"] else {}),
                               **({"fix": i["fix"]} if opt["include_fixes"] else {})} for i in issues]}
        return _json.dumps(payload, indent=2, ensure_ascii=False), mimetype, ext
    if opt["format"] == "csv":
        buf = io.StringIO()
        columns = ["severity", "category", "columns", "title", "detail"] + (["why"] if opt["include_why"] else []) + (["examples"] if opt["include_examples"] else []) + (["fix"] if opt["include_fixes"] else [])
        writer = csv.writer(buf)
        writer.writerow(columns)
        for i in issues:
            row = [i["severity"], i["category"], "; ".join(i["columns"]), i["title"], i["detail"]]
            if opt["include_why"]:
                row.append(i["why"])
            if opt["include_examples"]:
                row.append("; ".join(i["examples"]))
            if opt["include_fixes"]:
                row.append(f"{i['fix']['type']} {i['fix']['params']}" if i["fix"] else "")
            writer.writerow(row)
        return buf.getvalue(), mimetype, ext
    return quality_markdown(report, dataset_name, opt, issues), mimetype, ext


def quality_markdown(report: dict[str, Any], dataset_name: str, options: dict[str, Any] | None = None, issues: list[dict[str, Any]] | None = None) -> str:
    """The report as a standalone Markdown document."""
    opt = normalize_quality_options(options)
    issues = report["issues"] if issues is None else issues
    lines = [f"# {opt['title'] or 'Data quality report: ' + dataset_name}", ""]
    if opt["author"]:
        lines += [f"*Prepared by {opt['author']}.*", ""]
    if opt["notes"]:
        lines += [f"> {opt['notes']}", ""]
    lines += [
             f"**Overall: {report['label']} ({report['score']}/100)** - {report['rows']:,} rows x {report['columns']} columns. "
             f"{report['counts']['high']} high, {report['counts']['medium']} medium, {report['counts']['low']} low severity findings.", ""]
    if not issues:
        lines.append("No data-quality issues were detected." if not report["issues"] else "No issues match the selected severities.")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for issue in issues:
        grouped.setdefault(issue["category"] if opt["group_by"] == "category" else issue["severity"], []).append(issue)
    titles = {"missingness": "Missing data", "duplicates": "Duplicates", "formats": "Inconsistent formats", "suspicious": "Suspicious values",
              "outliers": "Outliers", "structure": "Column structure", "schema": "Schema problems", "leakage": "Leakage risk"}
    for category, items in grouped.items():
        lines += [f"## {titles.get(category, category.title())}", ""]
        for issue in items:
            lines.append(f"### [{issue['severity'].upper()}] {issue['title']}")
            lines.append(issue["detail"])
            if opt["include_why"]:
                lines.append(f"*Why it matters:* {issue['why']}")
            if opt["include_examples"] and issue["examples"]:
                lines.append(f"*Examples:* {', '.join(issue['examples'])}")
            if opt["include_fixes"] and issue["fix"]:
                lines.append(f"*Suggested fix:* `{issue['fix']['type']}` {issue['fix']['params']}")
            lines.append("")
    return "\n".join(lines)
