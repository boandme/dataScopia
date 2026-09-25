from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .schema import coerce_numeric


class TransformError(ValueError):
    """Raised for any invalid transform request - caught by the API layer and surfaced as a
    400 with the message as-is, since these are always meant to be user-facing."""


def _get_column(df: pd.DataFrame, name: str) -> str:
    if not name or name not in df.columns:
        raise TransformError(f"Column '{name}' not found in this dataset.")
    return name


_TRUE_STRINGS = {"true", "1", "yes", "y"}
_FALSE_STRINGS = {"false", "0", "no", "n"}


def _parse_value(series: pd.Series, raw: Any) -> Any:
    # Must come before is_numeric_dtype: pandas treats bool as numeric, so a boolean column
    # would otherwise fall into the float() branch and reject valid values like "True"/"False".
    if pd.api.types.is_bool_dtype(series):
        if isinstance(raw, bool):
            return raw
        normalized = str(raw).strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
        raise TransformError(f"'{raw}' is not a valid true/false value for this column.") from None
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise TransformError(f"'{raw}' is not a valid number for this column.") from None
    return raw


def _guard_not_empty(result: pd.DataFrame, action: str) -> None:
    if result.empty:
        raise TransformError(f"{action} would remove every row - refusing to apply. Try a looser condition.")


_COMPARATORS: dict[str, Callable[[pd.Series, Any], pd.Series]] = {
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
}


def op_filter(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    column = _get_column(df, params.get("column", ""))
    comparator = params.get("comparator", "==")
    if comparator not in _COMPARATORS:
        raise TransformError(f"Unsupported comparator '{comparator}'.")
    value = _parse_value(df[column], params.get("value"))
    mask = _COMPARATORS[comparator](df[column], value)
    result = df[mask].reset_index(drop=True)
    _guard_not_empty(result, "This filter")
    return result, f"Filtered to rows where '{column}' {comparator} {value} ({len(result):,} of {len(df):,} rows kept)."


def op_drop_missing(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    column = params.get("column") or ""
    if not column or column == "__all__":
        result = df.dropna().reset_index(drop=True)
        removed = len(df) - len(result)
        _guard_not_empty(result, "Dropping rows with any missing value")
        return result, f"Dropped {removed:,} row(s) with any missing value."
    _get_column(df, column)
    result = df.dropna(subset=[column]).reset_index(drop=True)
    removed = len(df) - len(result)
    _guard_not_empty(result, f"Dropping rows missing '{column}'")
    return result, f"Dropped {removed:,} row(s) where '{column}' was missing."


def op_fill_missing(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    column = _get_column(df, params.get("column", ""))
    strategy = params.get("strategy", "mean")
    result = df.copy()
    series = result[column]
    missing_count = int(series.isna().sum())
    if missing_count == 0:
        return result, f"'{column}' had no missing values; nothing to fill."

    if strategy == "mean":
        if not pd.api.types.is_numeric_dtype(series):
            raise TransformError(f"'{column}' isn't numeric; mean fill isn't valid.")
        value = series.mean()
    elif strategy == "median":
        if not pd.api.types.is_numeric_dtype(series):
            raise TransformError(f"'{column}' isn't numeric; median fill isn't valid.")
        value = series.median()
    elif strategy == "mode":
        modes = series.mode(dropna=True)
        if modes.empty:
            raise TransformError(f"'{column}' has no mode to fill with.")
        value = modes.iloc[0]
    elif strategy == "constant":
        value = _parse_value(series, params.get("constant_value"))
    else:
        raise TransformError(f"Unsupported fill strategy '{strategy}'.")

    result[column] = series.fillna(value)
    display_value = round(value, 4) if isinstance(value, float) else value
    return result, f"Filled {missing_count:,} missing value(s) in '{column}' with {strategy} ({display_value})."


def op_drop_outliers(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    column = _get_column(df, params.get("column", ""))
    series = df[column]
    # Must come before is_numeric_dtype: pandas treats bool as numeric, so a boolean column
    # would otherwise reach the IQR math below, where `q1 - 1.5 * iqr` on boolean quantile
    # results raises "numpy boolean subtract... not supported" - and a 2-value column has no
    # meaningful notion of an IQR outlier anyway.
    if pd.api.types.is_bool_dtype(series):
        raise TransformError(f"'{column}' is a true/false column; the IQR outlier rule doesn't apply.")
    if not pd.api.types.is_numeric_dtype(series):
        raise TransformError(f"'{column}' isn't numeric; the IQR outlier rule doesn't apply.")
    non_null = series.dropna()
    if non_null.empty:
        raise TransformError(f"'{column}' has no non-missing values to compute outlier bounds from.")
    q1, q3 = non_null.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = series.between(lower, upper) | series.isna()
    result = df[mask].reset_index(drop=True)
    removed = len(df) - len(result)
    _guard_not_empty(result, f"Dropping outliers in '{column}'")
    return result, f"Dropped {removed:,} outlier row(s) in '{column}' outside [{lower:.3f}, {upper:.3f}] (IQR rule)."


def op_rename_column(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    column = _get_column(df, params.get("column", ""))
    new_name = (params.get("new_name") or "").strip()
    if not new_name:
        raise TransformError("A new column name is required.")
    if new_name != column and new_name in df.columns:
        raise TransformError(f"A column named '{new_name}' already exists.")
    result = df.rename(columns={column: new_name})
    return result, f"Renamed '{column}' to '{new_name}'."


def op_cast_type(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    column = _get_column(df, params.get("column", ""))
    target = params.get("target_type", "")
    result = df.copy()
    before_missing = int(result[column].isna().sum())
    try:
        if target == "numeric":
            result[column] = coerce_numeric(result[column])
        elif target == "string":
            result[column] = result[column].astype(str)
        elif target == "datetime":
            result[column] = pd.to_datetime(result[column], errors="coerce")
        elif target == "category":
            result[column] = result[column].astype("category")
        else:
            raise TransformError(f"Unsupported target type '{target}'.")
    except TransformError:
        raise
    except Exception as exc:  # pragma: no cover
        raise TransformError(f"Could not cast '{column}' to {target}: {exc}") from exc
    after_missing = int(result[column].isna().sum())
    newly_missing = after_missing - before_missing
    note = f" ({newly_missing:,} value(s) couldn't convert and became missing)" if newly_missing > 0 else ""
    return result, f"Cast '{column}' to {target}{note}."


def op_drop_column(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    column = _get_column(df, params.get("column", ""))
    if df.shape[1] <= 1:
        raise TransformError("Can't drop the only remaining column.")
    result = df.drop(columns=[column])
    return result, f"Dropped column '{column}'."


def op_combine_columns(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    columns = params.get("columns") or []
    if not isinstance(columns, list) or len(columns) < 2:
        raise TransformError("Pick at least two columns to combine.")
    for c in columns:
        _get_column(df, c)
    operation = params.get("operation", "concat")
    new_name = (params.get("new_name") or "").strip()
    if not new_name:
        raise TransformError("A name for the new combined column is required.")
    if new_name in df.columns:
        raise TransformError(f"A column named '{new_name}' already exists.")

    result = df.copy()
    if operation == "concat":
        separator = params.get("separator", " ")
        # Vectorized Series addition rather than `.agg(sep.join, axis=1)`: the row-wise apply
        # form calls a Python function once per row and is roughly 400x slower at real sample
        # sizes (benchmarked at ~14s vs ~0.03s for 500k rows) for an identical result.
        combined = result[columns[0]].astype(str)
        for extra_col in columns[1:]:
            combined = combined + separator + result[extra_col].astype(str)
        result[new_name] = combined
    elif operation in ("sum", "mean"):
        if not all(pd.api.types.is_numeric_dtype(result[c]) for c in columns):
            raise TransformError("All selected columns must be numeric for sum/mean.")
        block = result[columns]
        result[new_name] = block.sum(axis=1) if operation == "sum" else block.mean(axis=1)
    elif operation == "coalesce":
        result[new_name] = result[columns[0]]
        for c in columns[1:]:
            result[new_name] = result[new_name].fillna(result[c])
    else:
        raise TransformError(f"Unsupported combine operation '{operation}'.")
    return result, f"Combined [{', '.join(columns)}] into '{new_name}' via {operation}."


def op_drop_duplicates(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    result = df.drop_duplicates().reset_index(drop=True)
    removed = len(df) - len(result)
    if removed == 0:
        return result, "No duplicate rows found; nothing removed."
    return result, f"Dropped {removed:,} duplicate row(s), keeping the first occurrence of each."


def op_replace_missing_tokens(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    column = _get_column(df, params.get("column", ""))
    tokens = params.get("values") or []
    if not isinstance(tokens, list) or not tokens:
        raise TransformError("At least one placeholder value to treat as missing is required.")
    tokens = [str(t) for t in tokens]
    result = df.copy()
    series = result[column]
    mask = series.notna() & series.astype(str).str.strip().isin(tokens)
    replaced = int(mask.sum())
    result[column] = series.where(~mask, np.nan)
    shown = ", ".join(repr(t) for t in tokens[:4])
    return result, f"Replaced {replaced:,} placeholder value(s) ({shown}) in '{column}' with real missing values."


def op_standardize_text(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    column = _get_column(df, params.get("column", ""))
    result = df.copy()
    series = result[column]
    text = series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    lower = text.str.lower()
    canonical = text.groupby(lower).agg(lambda v: v.value_counts().index[0])
    standardized = series.where(series.isna(), lower.map(canonical))
    changed = int((series.notna() & (standardized.astype(str) != series.astype(str))).sum())
    result[column] = standardized
    return result, f"Standardized spacing and spelling in '{column}' ({changed:,} value(s) changed); each variant now uses its most common form."


_OPS: dict[str, Callable[[pd.DataFrame, dict[str, Any]], tuple[pd.DataFrame, str]]] = {
    "filter": op_filter,
    "drop_missing": op_drop_missing,
    "fill_missing": op_fill_missing,
    "drop_outliers": op_drop_outliers,
    "rename_column": op_rename_column,
    "cast_type": op_cast_type,
    "drop_column": op_drop_column,
    "combine_columns": op_combine_columns,
    "drop_duplicates": op_drop_duplicates,
    "replace_missing_tokens": op_replace_missing_tokens,
    "standardize_text": op_standardize_text,
}


def apply_transform(df: pd.DataFrame, op_type: str, params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Apply one filter/clean/transform operation to `df` and return (new_df, human description).

    Never mutates `df` in place. Raises TransformError (a ValueError subclass) with a message
    safe to show the user directly for any invalid input or degenerate result.
    """
    if df.empty:
        raise TransformError("The dataset is empty; there's nothing to transform.")
    handler = _OPS.get(op_type)
    if handler is None:
        raise TransformError(f"Unknown transform type '{op_type}'.")
    return handler(df, params or {})


def op_code(df: pd.DataFrame, op_type: str, params: dict[str, Any]) -> str:
    """The pandas code that performs this operation, generated against the dataframe as it stood
    *before* the operation (so typed literals - a number vs a string in a filter - are right).

    This is what makes the notebook export reproduce an in-app cleaning session: every transform
    the workspace applies is stored with this code, and the exported notebook replays it in order.
    Each snippet is written to give the same result as the corresponding ``op_*`` function above;
    the test-suite executes both and compares them.
    """
    params = params or {}
    col = params.get("column")
    if op_type == "filter":
        comparator = params.get("comparator", "==")
        value = _parse_value(df[col], params.get("value"))
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and value.is_integer() and pd.api.types.is_integer_dtype(df[col]):
            value = int(value)
        return f"df = df[df[{col!r}] {comparator} {value!r}].reset_index(drop=True)"
    if op_type == "drop_missing":
        if not col or col == "__all__":
            return "df = df.dropna().reset_index(drop=True)"
        return f"df = df.dropna(subset=[{col!r}]).reset_index(drop=True)"
    if op_type == "fill_missing":
        strategy = params.get("strategy", "mean")
        if strategy == "mean":
            fill = f"df[{col!r}].mean()"
        elif strategy == "median":
            fill = f"df[{col!r}].median()"
        elif strategy == "mode":
            fill = f"df[{col!r}].mode(dropna=True).iloc[0]"
        else:
            fill = repr(_parse_value(df[col], params.get("constant_value")))
        return f"df[{col!r}] = df[{col!r}].fillna({fill})"
    if op_type == "drop_outliers":
        return (
            f"_q1, _q3 = df[{col!r}].dropna().quantile([0.25, 0.75])\n"
            "_iqr = _q3 - _q1\n"
            "_low, _high = _q1 - 1.5 * _iqr, _q3 + 1.5 * _iqr\n"
            f"df = df[df[{col!r}].between(_low, _high) | df[{col!r}].isna()].reset_index(drop=True)"
        )
    if op_type == "rename_column":
        new_name = (params.get("new_name") or "").strip()
        return f"df = df.rename(columns={{{col!r}: {new_name!r}}})"
    if op_type == "cast_type":
        target = params.get("target_type")
        if target == "numeric":
            return (
                f"df[{col!r}] = pd.to_numeric(df[{col!r}].astype(str).str.replace(r'[,$\u20ac\u00a3\u00a5%\\s]', '', regex=True)"
                ".str.replace(r'^\\((.*)\\)$', r'-\\1', regex=True), errors='coerce')"
            )
        if target == "string":
            return f"df[{col!r}] = df[{col!r}].astype(str)"
        if target == "datetime":
            return f"df[{col!r}] = pd.to_datetime(df[{col!r}], errors='coerce')"
        return f"df[{col!r}] = df[{col!r}].astype('category')"
    if op_type == "drop_column":
        return f"df = df.drop(columns=[{col!r}])"
    if op_type == "combine_columns":
        columns = params.get("columns") or []
        new = (params.get("new_name") or "").strip()
        operation = params.get("operation", "concat")
        if operation == "concat":
            separator = params.get("separator", " ")
            joined = f" + {separator!r} + ".join(f"df[{c!r}].astype(str)" for c in columns)
            return f"df[{new!r}] = {joined}"
        if operation in ("sum", "mean"):
            return f"df[{new!r}] = df[{columns!r}].{operation}(axis=1)"
        lines = [f"df[{new!r}] = df[{columns[0]!r}]"] + [f"df[{new!r}] = df[{new!r}].fillna(df[{c!r}])" for c in columns[1:]]
        return "\n".join(lines)
    if op_type == "drop_duplicates":
        return "df = df.drop_duplicates().reset_index(drop=True)"
    if op_type == "replace_missing_tokens":
        tokens = [str(t) for t in (params.get("values") or [])]
        return (
            f"_tokens = {tokens!r}\n"
            f"_mask = df[{col!r}].notna() & df[{col!r}].astype(str).str.strip().isin(_tokens)\n"
            f"df[{col!r}] = df[{col!r}].where(~_mask, np.nan)"
        )
    if op_type == "standardize_text":
        return (
            f"_series = df[{col!r}]\n"
            "_text = _series.astype(str).str.strip().str.replace(r'\\s+', ' ', regex=True)\n"
            "_lower = _text.str.lower()\n"
            "_canonical = _text.groupby(_lower).agg(lambda v: v.value_counts().index[0])\n"
            f"df[{col!r}] = _series.where(_series.isna(), _lower.map(_canonical))"
        )
    raise TransformError(f"No code generator for transform '{op_type}'.")
