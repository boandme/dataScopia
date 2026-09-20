from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd


@dataclass
class QAResult:
    answer: str
    computed: bool
    method: str
    code: str = ""
    value: Any = None
    columns_used: list[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _singular(token: str) -> str:
    """Cheap English pluralization reversal so 'countries'/'categories' match a 'country'/
    'category' column without a full stemming library."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ses", "xes", "zes", "ches", "shes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _find_columns(text: str, columns: list[str]) -> list[str]:
    """Return dataset columns mentioned in free text, most-specific match first."""
    norm_text = _normalize(text)
    text_tokens: set[str] = set()
    for tok in norm_text.split():
        text_tokens.add(tok)
        text_tokens.add(_singular(tok))
    scored: list[tuple[str, int]] = []
    for col in columns:
        col_norm = _normalize(col)
        if not col_norm:
            continue
        col_tokens = col_norm.split()
        token_match = all(tok in text_tokens or _singular(tok) in text_tokens for tok in col_tokens)
        if col_norm in norm_text or token_match:
            scored.append((col, len(col_norm)))
    scored.sort(key=lambda item: -item[1])
    seen: set[str] = set()
    ordered: list[str] = []
    for col, _ in scored:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def _fmt_num(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def _corr_strength(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 0.7:
        label = "strong"
    elif magnitude >= 0.4:
        label = "moderate"
    elif magnitude >= 0.2:
        label = "weak"
    else:
        label = "very weak"
    direction = "positive" if value >= 0 else "negative"
    return f"{label} {direction}"


def _parse_number(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Intent handlers. Each takes the raw question + dataframe and returns a
# QAResult if it can confidently answer it, or None to let the next intent
# in the dispatch order try. All are pure/read-only over `df`.
# ---------------------------------------------------------------------------


def _intent_missing(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not any(k in norm for k in ("missing", "null value", "nan values", "nulls")):
        return None
    columns = _find_columns(question, list(df.columns))
    if columns:
        col = columns[0]
        missing = int(df[col].isna().sum())
        pct = (missing / len(df) * 100) if len(df) else 0.0
        return QAResult(
            answer=f"'{col}' has {missing:,} missing values out of {len(df):,} rows ({_fmt_pct(pct)}).",
            computed=True,
            method="missing_values",
            code=f"df['{col}'].isna().sum()",
            value={"column": col, "missing": missing, "missing_pct": round(pct, 2)},
            columns_used=[col],
        )
    missing_by_col = df.isna().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    if missing_by_col.empty:
        return QAResult(
            answer="No missing values were found anywhere in this dataset.",
            computed=True,
            method="missing_values",
            code="df.isna().sum()",
            value={},
        )
    top = missing_by_col.head(5)
    parts = ", ".join(f"{c} ({int(v):,})" for c, v in top.items())
    return QAResult(
        answer=f"The columns with the most missing values are: {parts}.",
        computed=True,
        method="missing_values",
        code="df.isna().sum().sort_values(ascending=False)",
        value={str(c): int(v) for c, v in top.items()},
    )


def _intent_unique(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not any(
        k in norm
        for k in ("unique", "distinct", "how many different", "how many kinds", "how many categories", "how many types")
    ):
        return None
    columns = _find_columns(question, list(df.columns))
    if not columns:
        return None
    col = columns[0]
    n = int(df[col].nunique(dropna=True))
    raw_samples = df[col].dropna().unique()[:5]
    sample_values = [_fmt_num(v) if isinstance(v, (int, float)) else str(v) for v in raw_samples]
    sample_text = ", ".join(sample_values)
    return QAResult(
        answer=f"'{col}' has {n:,} unique values (e.g., {sample_text}).",
        computed=True,
        method="unique_count",
        code=f"df['{col}'].nunique()",
        value=n,
        columns_used=[col],
    )


def _intent_skew(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if "skew" not in norm:
        return None
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None
    matched = _find_columns(question, numeric_cols)
    if matched:
        col = matched[0]
    else:
        skews = {c: df[c].dropna().skew() for c in numeric_cols if df[c].dropna().shape[0] > 2}
        skews = {c: v for c, v in skews.items() if pd.notna(v)}
        if not skews:
            return None
        col = max(skews, key=lambda c: abs(skews[c]))
    series = df[col].dropna()
    if series.empty or len(series) < 3:
        return None
    skew_val = float(series.skew())
    if pd.isna(skew_val):
        return None
    mean_val = float(series.mean())
    median_val = float(series.median())
    direction = "right" if skew_val > 0 else "left"
    tail_note = (
        "a small number of unusually large values pulling the tail right"
        if skew_val > 0
        else "a small number of unusually small values pulling the tail left"
    )
    return QAResult(
        answer=(
            f"'{col}' has a skewness of {skew_val:.2f} ({direction}-skewed). "
            f"The mean ({_fmt_num(mean_val)}) is {'higher' if mean_val > median_val else 'lower'} than the median "
            f"({_fmt_num(median_val)}), which is consistent with {tail_note}."
        ),
        computed=True,
        method="skew_explanation",
        code=f"df['{col}'].skew()",
        value={"column": col, "skew": round(skew_val, 4), "mean": mean_val, "median": median_val},
        columns_used=[col],
    )


def _intent_correlation(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not any(k in norm for k in ("correlat", "relationship", "related to", "relate to")):
        return None
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        return None
    matched = _find_columns(question, numeric_cols)
    if len(matched) >= 2:
        col_a, col_b = matched[0], matched[1]
        value = df[[col_a, col_b]].corr().iloc[0, 1]
        if pd.isna(value):
            return None
        return QAResult(
            answer=f"The correlation between '{col_a}' and '{col_b}' is r = {value:.3f} ({_corr_strength(value)}).",
            computed=True,
            method="correlation",
            code=f"df['{col_a}'].corr(df['{col_b}'])",
            value=round(float(value), 4),
            columns_used=[col_a, col_b],
        )
    corr = df[numeric_cols].corr(numeric_only=True)
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1 :]:
            v = corr.loc[a, b]
            if pd.notna(v):
                pairs.append((a, b, float(v)))
    if not pairs:
        return None
    pairs.sort(key=lambda p: -abs(p[2]))
    top = pairs[:3]
    lines = "; ".join(f"{a} & {b} (r = {v:.3f})" for a, b, v in top)
    return QAResult(
        answer=f"The strongest numeric relationships are: {lines}.",
        computed=True,
        method="correlation",
        code="df.corr(numeric_only=True)",
        value=[{"a": a, "b": b, "r": round(v, 4)} for a, b, v in top],
        columns_used=[c for pair in top for c in (pair[0], pair[1])],
    )


_COMPARATORS: list[tuple[str, str]] = [
    (r"(?:at least|no less than)", ">="),
    (r"(?:at most|no more than)", "<="),
    (r">=", ">="),
    (r"<=", "<="),
    (r"(?:over|above|more than|greater than|exceed(?:ing|s)?)", ">"),
    (r"(?:under|below|less than|fewer than)", "<"),
    (r">", ">"),
    (r"<", "<"),
    (r"(?:equal(?:s|to)?)", "=="),
]

_NUMBER_PATTERN = r"(-?\d[\d,]*\.?\d*)"


def _intent_condition(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = question.lower()
    comparator_symbol = None
    match = None
    for pattern, symbol in _COMPARATORS:
        m = re.search(pattern + r"\s*" + _NUMBER_PATTERN, norm)
        if m:
            comparator_symbol = symbol
            match = m
            break
    if match is None:
        return None
    threshold = _parse_number(match.group(1))
    if threshold is None:
        return None
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    columns = _find_columns(question, numeric_cols)
    if not columns:
        return None
    col = columns[0]
    series = df[col].dropna()
    if series.empty:
        return None
    if comparator_symbol == ">":
        mask = series > threshold
    elif comparator_symbol == ">=":
        mask = series >= threshold
    elif comparator_symbol == "<":
        mask = series < threshold
    elif comparator_symbol == "<=":
        mask = series <= threshold
    else:
        mask = series == threshold
    count = int(mask.sum())
    total = len(series)
    pct = (count / total * 100) if total else 0.0
    wants_pct = any(k in norm for k in ("percent", "%", "proportion", "fraction", "share of"))
    if wants_pct:
        answer = f"{_fmt_pct(pct)} of rows have '{col}' {comparator_symbol} {_fmt_num(threshold)} ({count:,} of {total:,})."
    else:
        answer = f"{count:,} rows have '{col}' {comparator_symbol} {_fmt_num(threshold)} ({_fmt_pct(pct)} of {total:,})."
    return QAResult(
        answer=answer,
        computed=True,
        method="condition_filter",
        code=f"(df['{col}'] {comparator_symbol} {threshold}).mean() * 100",
        value={"column": col, "comparator": comparator_symbol, "threshold": threshold, "count": count, "percentage": round(pct, 2)},
        columns_used=[col],
    )


_SUPERLATIVE_HIGH = ("highest", "largest", "most", "top", "greatest", "maximum", "biggest")
_SUPERLATIVE_LOW = ("lowest", "smallest", "least", "bottom", "minimum")


def _intent_topn(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    ascending: bool | None = None
    if any(k in norm for k in _SUPERLATIVE_LOW):
        ascending = True
    if any(k in norm for k in _SUPERLATIVE_HIGH):
        ascending = False if ascending is None else ascending
    if ascending is None:
        return None
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metric_matches = _find_columns(question, numeric_cols)
    if not metric_matches:
        return None
    metric = metric_matches[0]
    n_match = re.search(r"top\s*(\d+)", norm)
    n = int(n_match.group(1)) if n_match else 1
    n = max(1, min(n, 20))

    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    label_matches = _find_columns(question, categorical_cols)
    label_col = label_matches[0] if label_matches else None
    if label_col is None:
        for c in categorical_cols:
            unique_count = df[c].nunique(dropna=True)
            if 1 < unique_count < len(df):
                label_col = c
                break

    if label_col:
        work = df[[label_col, metric]].dropna()
        if work.empty:
            return None
        grouped = work.groupby(label_col)[metric].mean().sort_values(ascending=ascending)
        top = grouped.head(n)
        lines = "; ".join(f"{idx} ({_fmt_num(val)})" for idx, val in top.items())
        direction = "lowest" if ascending else "highest"
        return QAResult(
            answer=f"The {direction} average '{metric}' by '{label_col}': {lines}.",
            computed=True,
            method="top_n",
            code=f"df.groupby('{label_col}')['{metric}'].mean().sort_values(ascending={ascending}).head({n})",
            value={str(idx): float(val) for idx, val in top.items()},
            columns_used=[metric, label_col],
        )

    series = df[metric].dropna().sort_values(ascending=ascending).head(n)
    lines = ", ".join(_fmt_num(v) for v in series)
    direction = "lowest" if ascending else "highest"
    return QAResult(
        answer=f"The {direction} '{metric}' value(s): {lines}.",
        computed=True,
        method="top_n",
        code=f"df['{metric}'].sort_values(ascending={ascending}).head({n})",
        value=series.tolist(),
        columns_used=[metric],
    )


_AGG_KEYWORDS: dict[str, str] = {
    "average": "mean",
    "mean": "mean",
    "median": "median",
    "sum": "sum",
    "total": "sum",
    "maximum": "max",
    "max": "max",
    "minimum": "min",
    "min": "min",
    "standard deviation": "std",
    "std": "std",
    "variance": "var",
}


def _intent_aggregation(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    func = None
    for keyword, agg in _AGG_KEYWORDS.items():
        if keyword in norm:
            func = agg
            break
    if func is None:
        return None
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metric_matches = _find_columns(question, numeric_cols)
    if not metric_matches:
        return None
    metric = metric_matches[0]

    group_col = None
    group_match = re.search(r"\bby\s+([a-z0-9 _-]+)", norm)
    if group_match:
        candidates = _find_columns(group_match.group(1), [c for c in df.columns if c != metric])
        group_col = candidates[0] if candidates else None

    if group_col:
        work = df[[group_col, metric]].dropna()
        if work.empty:
            return None
        result = work.groupby(group_col)[metric].agg(func).sort_values(ascending=False)
        top = result.head(8)
        lines = "; ".join(f"{idx}: {_fmt_num(val)}" for idx, val in top.items())
        return QAResult(
            answer=f"The {func} of '{metric}' by '{group_col}': {lines}.",
            computed=True,
            method="aggregation",
            code=f"df.groupby('{group_col}')['{metric}'].{func}()",
            value={str(idx): float(val) for idx, val in top.items()},
            columns_used=[metric, group_col],
        )

    series = df[metric].dropna()
    if series.empty:
        return None
    value = getattr(series, func)()
    return QAResult(
        answer=f"The {func} of '{metric}' is {_fmt_num(float(value))}.",
        computed=True,
        method="aggregation",
        code=f"df['{metric}'].{func}()",
        value=float(value),
        columns_used=[metric],
    )


def _intent_shape(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    wants_rows = any(
        k in norm for k in ("how many rows", "how many observations", "how many records", "how many entries", "number of rows")
    )
    wants_cols = any(
        k in norm for k in ("how many columns", "how many variables", "how many fields", "number of columns")
    )
    wants_shape = any(k in norm for k in ("shape of the data", "size of the dataset", "dimensions of"))
    if not (wants_rows or wants_cols or wants_shape):
        return None
    rows, cols = df.shape
    if wants_rows and not wants_cols:
        return QAResult(answer=f"The dataset has {rows:,} rows.", computed=True, method="shape", code="df.shape[0]", value=rows)
    if wants_cols and not wants_rows:
        return QAResult(answer=f"The dataset has {cols:,} columns.", computed=True, method="shape", code="df.shape[1]", value=cols)
    return QAResult(
        answer=f"The dataset has {rows:,} rows and {cols:,} columns.",
        computed=True,
        method="shape",
        code="df.shape",
        value={"rows": int(rows), "columns": int(cols)},
    )


_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


def _intent_time_change(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not any(k in norm for k in ("increase", "decrease", "change", "grew", "declined", "growth", "drop", "fell", "rose")):
        return None
    years = sorted({int(y) for y in _YEAR_PATTERN.findall(question)})
    if len(years) < 2:
        return None
    year_a, year_b = years[0], years[-1]

    time_col = None
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            time_col = c
            break
    if time_col is None:
        for c in df.columns:
            norm_c = _normalize(c)
            if norm_c in ("year", "yr") or norm_c.endswith(" year"):
                time_col = c
                break
    if time_col is None:
        return None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    metric_matches = [c for c in _find_columns(question, numeric_cols) if c != time_col]
    if not metric_matches:
        return None
    metric = metric_matches[0]

    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != time_col]
    label_matches = _find_columns(question, categorical_cols)
    label_col = label_matches[0] if label_matches else (categorical_cols[0] if categorical_cols else None)
    if label_col is None:
        return None

    work = df[[label_col, time_col, metric]].dropna()
    if work.empty:
        return None
    if pd.api.types.is_datetime64_any_dtype(work[time_col]):
        work = work.assign(_year=work[time_col].dt.year)
    else:
        work = work.assign(_year=pd.to_numeric(work[time_col], errors="coerce"))
        work = work.dropna(subset=["_year"])

    subset = work[work["_year"].isin([year_a, year_b])]
    if subset.empty:
        return None
    pivot = subset.pivot_table(index=label_col, columns="_year", values=metric, aggfunc="mean")
    if year_a not in pivot.columns or year_b not in pivot.columns:
        return None
    pivot = pivot.dropna()
    if pivot.empty:
        return None

    diff = (pivot[year_b] - pivot[year_a]).sort_values()
    wants_decrease = any(k in norm for k in ("decrease", "declined", "drop", "fell"))
    ordered = diff if wants_decrease else diff.sort_values(ascending=False)
    top = ordered.head(5)
    direction = "decrease" if wants_decrease else "increase"
    lines = "; ".join(f"{idx} ({_fmt_num(val)})" for idx, val in top.items())
    return QAResult(
        answer=f"The largest {direction} in '{metric}' from {year_a} to {year_b}, by '{label_col}': {lines}.",
        computed=True,
        method="time_change",
        code=f"(pivot[{year_b}] - pivot[{year_a}]).sort_values()  # grouped by {label_col}",
        value={str(idx): float(val) for idx, val in top.items()},
        columns_used=[label_col, time_col, metric],
    )


_INTENTS: list[Callable[[str, pd.DataFrame], QAResult | None]] = [
    _intent_missing,
    _intent_unique,
    _intent_skew,
    _intent_correlation,
    _intent_time_change,
    _intent_condition,
    _intent_topn,
    _intent_aggregation,
    _intent_shape,
]


def _fallback(df: pd.DataFrame, snapshot: dict[str, Any] | None) -> QAResult:
    rows, cols = df.shape
    hint = (
        "Try asking about a specific column, e.g. 'what is the average <column>?', "
        "'how many rows have <column> over <number>?', 'what correlates with <column>?', "
        "or 'which <category> has the highest <column>?'."
    )
    if snapshot:
        msg = (
            "I couldn't map this question to a specific computation. Here's what the current profile shows: "
            f"{snapshot.get('rows', rows):,} observations, {snapshot.get('columns', cols)} variables, "
            f"{snapshot.get('notable_relationships', 0)} notable numeric relationships, and "
            f"{snapshot.get('data_quality_issues', 0)} data-quality issues detected. {hint}"
        )
    else:
        msg = f"I couldn't map this question to a specific computation over this {rows:,}-row dataset. {hint}"
    return QAResult(answer=msg, computed=False, method="fallback", code="", value=None)


def answer_question(question: str, df: pd.DataFrame, snapshot: dict[str, Any] | None = None) -> QAResult:
    """Answer a natural-language question with a real computation over `df` when possible.

    Tries a fixed set of deterministic intent handlers (correlation, thresholds, aggregation,
    ranking, missing data, etc.) in order of specificity; each performs an actual pandas
    computation rather than returning generic advice. Falls back to a grounded snapshot summary,
    clearly marked as not computed, when no intent matches.
    """
    for intent in _INTENTS:
        try:
            result = intent(question, df)
        except Exception:
            result = None
        if result is not None:
            return result
    return _fallback(df, snapshot)
