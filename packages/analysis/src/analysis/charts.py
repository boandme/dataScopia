"""Server-side chart engine.

Every chart is computed from the *full* analyzed dataframe (after any in-app cleaning), not from
the 300-row preview the browser holds, and comes back as plain Plotly JSON. That matters for
correctness, not just polish: a histogram, correlation matrix or map drawn from the first 300
rows silently misrepresents the data, and a time series drawn from them may cover only the
earliest weeks.

The engine also refuses to draw misleading charts. A scatter needs two numeric fields, a pie
folds its long tail into "Other", overlapping-mark charts cap the number of colour groups, and
every builder that has something statistically meaningful to say (trend, group difference,
association) says it in ``meta["notes"]``.

Colour follows the validated data-viz palette: fixed categorical order, one blue ramp for
magnitude, a blue-red diverging scale with a neutral midpoint for correlations, and a muted grey
for "Other".
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from . import geo as geo_tables
from . import schema as schema_mod

CHART_TYPE_LABELS: dict[str, str] = {
    "histogram": "Histogram",
    "bar": "Bar",
    "stacked_bar": "Stacked bar",
    "pie": "Pie",
    "scatter": "Scatter",
    "line": "Line",
    "area": "Area",
    "timeseries": "Time series",
    "box": "Box plot",
    "violin": "Violin",
    "heatmap": "Correlation matrix",
    "distribution": "Distribution comparison",
    "target": "Target vs feature",
    "treemap": "Treemap",
    "map": "Map",
    "missing": "Missing values",
    "pair": "Pair plot",
}

GRANULARITIES = {"none": None, "day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}

# Dark-surface categorical palette, in its validated fixed order (never cycled or re-sorted).
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
OTHER_COLOR = "#898781"
# One-hue sequential ramp for magnitude: dark = low, light = high on a dark surface.
SEQUENTIAL = [[0.0, "#0d366b"], [0.2, "#184f95"], [0.4, "#256abf"], [0.6, "#3987e5"], [0.8, "#86b6ef"], [1.0, "#cde2fb"]]
# Magnitude ramp used by default so charts stand out: indigo -> violet -> magenta -> orange -> gold.
# Lightness still rises monotonically (dark = low, light = high on the dark surface), so it reads
# as a proper sequential scale even though it passes through several hues.
VIVID = [[0.0, "#5546e0"], [0.25, "#9340d6"], [0.5, "#e2409b"], [0.75, "#ff7d47"], [1.0, "#ffd23a"]]
# Two hues with a neutral midpoint, for signed quantities such as correlation.
DIVERGING = [[0.0, "#2f7bff"], [0.25, "#3a5fb0"], [0.5, "#2a2e45"], [0.75, "#b84a63"], [1.0, "#ff5c7a"]]
INK = "#e5e7eb"
INK_MUTED = "#a3a9b8"
GRID = "rgba(148,163,184,0.14)"
AXIS_LINE = "rgba(148,163,184,0.32)"

MAX_SCATTER_POINTS = 5000
MAX_BOX_POINTS = 1500
MAX_BAR_CATEGORIES = 15
MAX_LINE_SERIES = 8
MAX_OVERLAP_GROUPS = 3  # charts whose marks overlap must keep their colours distinguishable


class ChartError(Exception):
    """A chart request that can't produce a meaningful picture; the message is shown to the user."""


# ------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------


def _clean(value: Any) -> Any:
    """Make anything Plotly-JSON safe: numpy -> python, NaN/inf -> None, timestamps -> ISO."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (pd.Series, pd.Index)):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "M":
            return [_clean(v) for v in pd.Series(value).tolist()]
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if (math.isnan(value) or math.isinf(value)) else float(value)
    if value is pd.NaT or value is pd.NA:
        return None
    return value


def _axis(title: str | None = None, **extra: Any) -> dict[str, Any]:
    axis: dict[str, Any] = {
        "gridcolor": GRID,
        "zerolinecolor": AXIS_LINE,
        "linecolor": AXIS_LINE,
        # A closed, ticked frame (all four sides bordered, small outward tick marks on the labelled
        # sides) is what makes a plot read as "properly axed" rather than numbers floating on a grid -
        # standard practice in scientific/journal-style figures, and it applies equally to x and y so
        # neither axis looks like an afterthought next to the other.
        "showline": True,
        "mirror": True,
        "ticks": "outside",
        "tickcolor": AXIS_LINE,
        "ticklen": 5,
        "tickfont": {"color": INK_MUTED, "size": 11},
        "automargin": True,
    }
    if title:
        axis["title"] = {"text": title, "font": {"color": INK_MUTED, "size": 12.5}}
    axis.update(extra)
    return axis


def _color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def _short(text: Any, limit: int = 28) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _numeric(df: pd.DataFrame, column: str, profiles: dict[str, Any]) -> pd.Series | None:
    """The column as numbers if it is (or can sensibly be read as) numeric, else None."""
    if column not in df.columns:
        return None
    series = df[column]
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    if profiles.get(column, {}).get("notes", {}).get("stored_as_text"):
        coerced = schema_mod.coerce_numeric(series)
        return coerced.astype(float) if coerced.notna().any() else None
    return None


def _role(profiles: dict[str, Any], column: str | None) -> str:
    if not column:
        return "none"
    return profiles.get(column, {}).get("role", "unknown")


def _is_measure(profiles: dict[str, Any], column: str | None) -> bool:
    return _role(profiles, column) in (schema_mod.MEASURE_ROLES | {"geo_lat", "geo_lon", "year"})


def _is_grouper(profiles: dict[str, Any], column: str | None) -> bool:
    return _role(profiles, column) in schema_mod.GROUPING_ROLES


def _category_labels(series: pd.Series, profile: dict[str, Any]) -> pd.Series:
    """Categories as display strings; whole-number floats lose their '.0'."""
    if pd.api.types.is_float_dtype(series):
        finite = series.dropna()
        if len(finite) and (finite == np.floor(finite)).all():
            return series.map(lambda v: None if pd.isna(v) else str(int(v)))
    if pd.api.types.is_bool_dtype(series):
        return series.map({True: "True", False: "False"})
    return series.map(lambda v: None if pd.isna(v) else str(v))


def _ordered_categories(labels: pd.Series, profile: dict[str, Any], values: pd.Series | None = None) -> list[str]:
    """Category order: a known scale (low<medium<high) or numeric order if there is one,
    otherwise by size, so the biggest categories lead."""
    counts = labels.value_counts()
    order = profile.get("notes", {}).get("order")
    if order:
        ranked = {name: i for i, name in enumerate(order)}
        return sorted(counts.index, key=lambda c: ranked.get(str(c).strip().lower(), len(order)))
    if profile.get("role") in {"ordinal_numeric", "binary_numeric", "year"}:
        try:
            return sorted(counts.index, key=lambda c: float(c))
        except (TypeError, ValueError):
            pass
    if values is not None:
        return list(values.sort_values(ascending=False).index)
    return list(counts.index)


def _limit_categories(order: list[str], limit: int) -> tuple[list[str], int]:
    if len(order) <= limit:
        return order, 0
    return order[:limit], len(order) - limit


def _agg_name(agg: str | None, has_measure: bool) -> str:
    if not agg or agg == "none":
        return "mean" if has_measure else "count"
    if agg == "count":
        return "count"
    return agg if has_measure else "count"


def _aggregate(frame: pd.DataFrame, by: list[str], value: str | None, agg: str) -> pd.DataFrame:
    grouped = frame.groupby(by, dropna=True, observed=True)
    if agg == "count" or value is None:
        return grouped.size().rename("value").reset_index()
    return grouped[value].agg(agg).rename("value").reset_index()


def _measure_label(y: str | None, agg: str) -> str:
    if agg == "count" or not y:
        return "Number of rows"
    names = {"mean": "Average", "median": "Median", "sum": "Total", "max": "Maximum", "min": "Minimum"}
    return f"{names.get(agg, agg.title())} {y}"


def _sample_indices(n: int, limit: int, seed: int = 7) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=limit, replace=False))


def _message_figure(message: str) -> dict[str, Any]:
    return {
        "data": [],
        "layout": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [{"text": message, "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
                             "showarrow": False, "font": {"size": 13, "color": INK_MUTED}}],
        },
        "meta": {"notes": [message], "error": message},
    }


def _finish(data: list[dict[str, Any]], layout: dict[str, Any], title: str, notes: list[str], effective: str, **meta: Any) -> dict[str, Any]:
    show_legend = len([t for t in data if t.get("showlegend", True) and t.get("name")]) >= 2
    layout.setdefault("showlegend", show_legend)
    if show_legend:
        layout.setdefault("legend", {"orientation": "h", "y": -0.22, "font": {"color": INK_MUTED}})
    return {"data": data, "layout": layout, "meta": {"title": title, "notes": notes, "effective_type": effective, **meta}}


# ------------------------------------------------------------------------------------
# builders
# ------------------------------------------------------------------------------------


def _histogram(df: pd.DataFrame, cfg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    profiles = schema["profiles"]
    x = cfg.get("x")
    values = _numeric(df, x, profiles) if x else None
    if values is None:
        if x and _role(profiles, x) in {"datetime"}:
            return _bar(df, {**cfg, "type": "bar", "agg": "count"}, schema)
        if x and x in df.columns:
            note = f"'{x}' isn't numeric, so its counts are shown as bars instead of a histogram."
            out = _bar(df, {**cfg, "type": "bar", "agg": "count"}, schema)
            out["meta"]["notes"].insert(0, note)
            return out
        raise ChartError("Pick a numeric field for the histogram.")
    clean = values.dropna()
    if len(clean) < 2:
        raise ChartError(f"'{x}' has too few values to draw a distribution.")

    group = cfg.get("group")
    group_labels = None
    if group and group in df.columns and group != x and _is_grouper(profiles, group):
        group_labels = _category_labels(df[group], profiles.get(group, {}))

    bins = min(50, max(8, len(np.histogram_bin_edges(clean, bins="auto")) - 1))
    edges = np.histogram_bin_edges(clean, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    notes: list[str] = []
    data: list[dict[str, Any]] = []
    shapes: list[dict[str, Any]] = []

    if group_labels is not None:
        order = _ordered_categories(group_labels[values.notna()], profiles.get(group, {}))
        keep, dropped = _limit_categories(order, MAX_OVERLAP_GROUPS)
        for i, name in enumerate(keep):
            sub = values[(group_labels == name) & values.notna()]
            counts, _ = np.histogram(sub, bins=edges)
            data.append({"type": "bar", "x": centers, "y": counts, "width": widths * 0.96, "name": _short(name),
                         "marker": {"color": _color(i)}, "opacity": 0.65,
                         "hovertemplate": f"{_short(name)}<br>%{{customdata[0]:.4g}} to %{{customdata[1]:.4g}}: %{{y}} rows<extra></extra>",
                         "customdata": np.column_stack([edges[:-1], edges[1:]])})
        if dropped:
            notes.append(f"Showing the {MAX_OVERLAP_GROUPS} largest of {len(order)} '{group}' groups so overlapping bars stay distinguishable.")
        barmode = "overlay"
    else:
        counts, _ = np.histogram(clean, bins=edges)
        data.append({"type": "bar", "x": centers, "y": counts, "width": widths * 0.96, "name": str(x),
                     "marker": {"color": counts, "colorscale": VIVID, "showscale": False, "line": {"width": 0}},
                     "hovertemplate": "%{customdata[0]:.4g} to %{customdata[1]:.4g}: %{y} rows<extra></extra>",
                     "customdata": np.column_stack([edges[:-1], edges[1:]]), "showlegend": False})
        mean, median = float(clean.mean()), float(clean.median())
        for value, label, dash in ((mean, "mean", "dash"), (median, "median", "dot")):
            shapes.append({"type": "line", "x0": value, "x1": value, "y0": 0, "y1": 1, "yref": "paper",
                           "line": {"color": INK_MUTED, "width": 1.5, "dash": dash}})
        skew = float(clean.skew()) if len(clean) > 2 else 0.0
        notes.append(f"n = {len(clean):,}; mean {mean:,.4g} (dashed), median {median:,.4g} (dotted), skewness {skew:.2f}.")
        if abs(skew) > 1:
            notes.append("Strongly skewed: the mean is pulled toward the long tail, so the median is the better 'typical' value.")
        barmode = "overlay"

    layout = {"xaxis": _axis(cfg.get("xlabel") or str(x)), "yaxis": _axis(cfg.get("ylabel") or "Number of rows"),
              "barmode": barmode, "bargap": 0.04, "shapes": shapes}
    return _finish(data, layout, f"Distribution of {x}", notes, "histogram")


def _resolve_bar_inputs(df: pd.DataFrame, cfg: dict[str, Any], schema: dict[str, Any]):
    profiles = schema["profiles"]
    x = cfg.get("x")
    if not x or x not in df.columns:
        raise ChartError("Pick a field for the categories.")
    y = cfg.get("y") if cfg.get("y") in df.columns else None
    y_values = _numeric(df, y, profiles) if y else None
    agg = _agg_name(cfg.get("agg"), y_values is not None)
    return x, y, y_values, agg


def _bin_numeric_x(df: pd.DataFrame, x: str, profiles: dict[str, Any], notes: list[str]) -> pd.Series | None:
    """A continuous x can't be one bar per value; bin it (and say so)."""
    values = _numeric(df, x, profiles)
    if values is None or _role(profiles, x) in {"ordinal_numeric", "binary_numeric", "year"}:
        return None
    if values.nunique() <= MAX_BAR_CATEGORIES:
        return None
    binned = pd.cut(values, bins=min(12, max(5, int(math.sqrt(values.notna().sum())))), include_lowest=True)
    notes.append(f"'{x}' is continuous, so its values are grouped into {binned.cat.categories.size} ranges.")
    return binned.map(lambda iv: None if pd.isna(iv) else f"{iv.left:.4g} – {iv.right:.4g}").astype(object)


def _bar(df: pd.DataFrame, cfg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    profiles = schema["profiles"]
    x, y, y_values, agg = _resolve_bar_inputs(df, cfg, schema)
    notes: list[str] = []
    if cfg.get("agg") in (None, "none") and y_values is not None and cfg.get("type") in ("bar", "stacked_bar"):
        notes.append(f"Raw values can't be summed into bars, so the {_measure_label(y, agg).lower()} per category is shown.")

    binned = _bin_numeric_x(df, x, profiles, notes)
    x_labels = binned if binned is not None else _category_labels(df[x], profiles.get(x, {}))
    frame = pd.DataFrame({"x": x_labels})
    if y_values is not None:
        frame["y"] = y_values
    group = cfg.get("group") if cfg.get("group") in df.columns and cfg.get("group") != x else None
    if cfg.get("type") == "stacked_bar" and not group and y is not None and _is_grouper(profiles, y):
        group, y, y_values, agg = y, None, None, "count"  # legacy configs put the stack field in y
        frame = pd.DataFrame({"x": x_labels})
    if group:
        frame["g"] = _category_labels(df[group], profiles.get(group, {}))
    frame = frame.dropna(subset=["x"])

    by = ["x"] + (["g"] if group else [])
    table = _aggregate(frame, by, "y" if y_values is not None else None, agg)
    if table.empty:
        raise ChartError("No rows to chart after removing missing values.")

    totals = table.groupby("x")["value"].sum()
    order = _ordered_categories(frame["x"], profiles.get(x, {}) if binned is None else {}, totals if binned is None else None)
    if binned is not None:
        order = list(binned.dropna().astype(str).unique())
        order = sorted(order, key=lambda s: float(s.split(" – ")[0]))
    sort = cfg.get("sort")
    natural_order = bool(profiles.get(x, {}).get("notes", {}).get("order")) or _role(profiles, x) in {"ordinal_numeric", "binary_numeric", "year"} or binned is not None
    if sort in ("asc", "desc"):
        order = list(totals.sort_values(ascending=(sort == "asc")).index)
    elif not natural_order:
        order = list(totals.sort_values(ascending=False).index)
    order, dropped = _limit_categories(order, MAX_BAR_CATEGORIES)
    if dropped:
        notes.append(f"Showing the {MAX_BAR_CATEGORIES} largest of {len(totals)} categories ({dropped} smaller ones hidden).")
    table = table[table["x"].isin(order)]

    label = _measure_label(y, agg)
    data: list[dict[str, Any]] = []
    if group:
        group_totals = table.groupby("g")["value"].sum().sort_values(ascending=False)
        keep = list(group_totals.index[:5])
        if len(group_totals) > 5:
            notes.append(f"Showing the 5 largest of {len(group_totals)} '{group}' groups.")
        for i, name in enumerate(keep):
            sub = table[table["g"] == name].set_index("x")["value"].reindex(order)
            data.append({"type": "bar", "x": [_short(c) for c in order], "y": sub.fillna(0).tolist(), "name": _short(name),
                         "marker": {"color": _color(i)},
                         "hovertemplate": f"{_short(name)}<br>%{{x}}: %{{y:,.4g}}<extra></extra>"})
        barmode = "stack" if cfg.get("type") == "stacked_bar" else "group"
    else:
        sub = table.groupby("x")["value"].sum().reindex(order)
        data.append({"type": "bar", "x": [_short(c) for c in order], "y": sub.tolist(), "name": label,
                     "marker": {"color": sub.fillna(0).tolist(), "colorscale": VIVID, "showscale": False, "line": {"width": 0}},
                     "showlegend": False,
                     "hovertemplate": "%{x}: %{y:,.4g}<extra></extra>"})
        barmode = "group"

    layout = {"xaxis": _axis(cfg.get("xlabel") or str(x), type="category", tickangle=-30 if max(len(str(c)) for c in order) > 8 else 0),
              "yaxis": _axis(cfg.get("ylabel") or label), "barmode": barmode, "bargap": 0.25}
    title = f"{label} by {x}" if y_values is not None or agg != "count" else f"Rows by {x}"
    return _finish(data, layout, title, notes, "stacked_bar" if barmode == "stack" else "bar")


def _pie(df: pd.DataFrame, cfg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    profiles = schema["profiles"]
    x = cfg.get("x")
    if not x or x not in df.columns:
        raise ChartError("Pick a field for the slices.")
    labels = _category_labels(df[x], profiles.get(x, {})).dropna()
    counts = labels.value_counts()
    notes: list[str] = []
    keep = 8
    if len(counts) > keep:
        other = int(counts.iloc[keep:].sum())
        counts = pd.concat([counts.iloc[:keep], pd.Series({"Other": other})])
        notes.append(f"'{x}' has {len(labels.unique())} distinct values; the {keep} largest are shown and the rest are grouped as Other.")
    colors = [_color(i) for i in range(len(counts))]
    if "Other" in counts.index:
        colors[list(counts.index).index("Other")] = OTHER_COLOR
    data = [{"type": "pie", "labels": [_short(c) for c in counts.index], "values": counts.values, "hole": 0.45,
             "marker": {"colors": colors, "line": {"color": "#0c1020", "width": 2}},
             "textinfo": "percent", "sort": False, "hovertemplate": "%{label}: %{value:,} rows (%{percent})<extra></extra>"}]
    layout = {"legend": {"orientation": "h", "y": -0.1, "font": {"color": INK_MUTED}}, "showlegend": True}
    return _finish(data, layout, f"Share of rows by {x}", notes, "pie")


def _time_axis(df: pd.DataFrame, x: str, profiles: dict[str, Any], granularity: str | None) -> tuple[pd.Series | None, bool]:
    """(x values as timestamps or None, is_time) for a datetime/year column."""
    role = _role(profiles, x)
    if role not in {"datetime", "year"}:
        return None, False
    times = schema_mod.time_values(df, x, profiles[x])
    freq = GRANULARITIES.get(granularity or "none")
    if freq and role == "datetime":
        times = times.dt.to_period(freq).dt.to_timestamp()
    return times, True


def _line(df: pd.DataFrame, cfg: dict[str, Any], schema: dict[str, Any], kind: str = "line") -> dict[str, Any]:
    profiles = schema["profiles"]
    x, y = cfg.get("x"), cfg.get("y")
    if not x or x not in df.columns:
        raise ChartError("Pick a field for the horizontal axis.")
    y_values = _numeric(df, y, profiles) if y and y in df.columns else None
    if y and y_values is None:
        raise ChartError(f"'{y}' isn't numeric, so it can't be plotted over '{x}'.")
    agg = _agg_name(cfg.get("agg"), y_values is not None)
    notes: list[str] = []

    times, is_time = _time_axis(df, x, profiles, cfg.get("granularity"))
    if is_time:
        x_series = times
    else:
        x_num = _numeric(df, x, profiles)
        if x_num is None:
            raise ChartError(f"'{x}' is neither a date nor a number, so a line over it isn't meaningful. Try a bar chart.")
        x_series = x_num
    frame = pd.DataFrame({"x": x_series})
    if y_values is not None:
        frame["y"] = y_values
    group = cfg.get("group") if cfg.get("group") in df.columns and cfg.get("group") != x else None
    if group:
        frame["g"] = _category_labels(df[group], profiles.get(group, {}))
    frame = frame.dropna(subset=["x"] + (["y"] if y_values is not None else []))
    if frame.empty:
        raise ChartError("No rows to chart after removing missing values.")

    duplicated = frame.duplicated(subset=["x"] + (["g"] if group else [])).any()
    if cfg.get("agg") in (None, "none") and not duplicated and y_values is not None:
        table = frame.rename(columns={"y": "value"})
    else:
        table = _aggregate(frame, ["x"] + (["g"] if group else []), "y" if y_values is not None else None, agg)
        if y_values is not None and cfg.get("agg") in (None, "none"):
            notes.append(f"Several rows share the same {x}, so the {agg} of {y} is plotted for each.")
        elif is_time and not group and schema.get("time", {}) and schema["time"].get("is_panel") and schema["time"].get("time_column") == x:
            notes.append(f"The {agg} across all {schema['time']['entity_column']} values is shown for each period.")
    table = table.sort_values("x")

    data: list[dict[str, Any]] = []
    label = _measure_label(y, agg)
    if group:
        totals = table.groupby("g")["value"].sum().abs().sort_values(ascending=False)
        keep = list(totals.index[:MAX_LINE_SERIES])
        if len(totals) > MAX_LINE_SERIES:
            notes.append(f"Showing the {MAX_LINE_SERIES} largest of {len(totals)} '{group}' series.")
        series = [(name, table[table["g"] == name]) for name in keep]
    else:
        series = [(label, table)]
    many_points = len(table) > 60
    for i, (name, sub) in enumerate(series):
        # a wide, faint copy of the line underneath gives it a glow so it stands out on the dark surface
        data.append({"type": "scatter", "mode": "lines", "x": sub["x"], "y": sub["value"], "hoverinfo": "skip",
                     "showlegend": False, "line": {"color": _color(i) + "30", "width": 11}})
        trace: dict[str, Any] = {
            "type": "scatter", "mode": "lines" if many_points else "lines+markers", "x": sub["x"], "y": sub["value"],
            "name": _short(name), "line": {"color": _color(i), "width": 3.2},
            "marker": ({"size": 9, "color": sub["value"], "colorscale": VIVID, "showscale": False, "line": {"width": 2, "color": "#0c1020"}}
                       if len(series) == 1 else {"size": 8, "color": _color(i), "line": {"width": 2, "color": "#0c1020"}}),
            "hovertemplate": f"{_short(name)}<br>%{{x}}: %{{y:,.4g}}<extra></extra>",
        }
        if kind == "area":
            trace["fill"] = "tozeroy"
            trace["fillcolor"] = _color(i) + "40"
        data.append(trace)
    if is_time and cfg.get("granularity") in (None, "none") and schema.get("time") and schema["time"].get("frequency") not in (None, "irregular") and x == schema["time"].get("time_column"):
        notes.append(f"Data is {schema['time']['frequency']}; use 'Date granularity' to roll it up.")
    layout = {"xaxis": _axis(cfg.get("xlabel") or str(x), type="date" if is_time else "linear"), "yaxis": _axis(cfg.get("ylabel") or label)}
    title = f"{label} over {x}" if is_time else f"{label} by {x}"
    return _finish(data, layout, title, notes, "timeseries" if is_time else kind)


def _trend(x: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return {"slope": float(slope), "intercept": float(intercept), "r": r, "r2": r * r}


def _strength(r: float) -> str:
    magnitude = abs(r)
    return "strong" if magnitude >= 0.7 else "moderate" if magnitude >= 0.4 else "weak" if magnitude >= 0.2 else "negligible"


def _scatter(df: pd.DataFrame, cfg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    profiles = schema["profiles"]
    x, y = cfg.get("x"), cfg.get("y")
    xv = _numeric(df, x, profiles) if x else None
    yv = _numeric(df, y, profiles) if y else None
    if xv is None or yv is None:
        if x and y and yv is not None and _is_grouper(profiles, x):
            out = _box(df, {**cfg, "type": "box"}, schema)
            out["meta"]["notes"].insert(0, f"A scatter plot needs two numeric fields; '{x}' is categorical, so the spread of '{y}' within each '{x}' group is shown instead.")
            return out
        raise ChartError("A scatter plot needs two numeric fields for X and Y.")
    frame = pd.DataFrame({"x": xv, "y": yv})
    group = cfg.get("group") if cfg.get("group") in df.columns and cfg.get("group") not in (x, y) and _is_grouper(profiles, cfg.get("group")) else None
    if group:
        frame["g"] = _category_labels(df[group], profiles.get(group, {}))
    frame = frame.dropna(subset=["x", "y"])
    if len(frame) < 2:
        raise ChartError("Too few rows have both X and Y values.")

    notes: list[str] = []
    trend = _trend(frame["x"].to_numpy(), frame["y"].to_numpy())
    if trend:
        try:
            from scipy import stats as sp

            sample_idx = _sample_indices(len(frame), 20000)
            rho = float(sp.spearmanr(frame["x"].to_numpy()[sample_idx], frame["y"].to_numpy()[sample_idx])[0])
            rho_text = f"; Spearman rho = {rho:.2f}"
        except Exception:
            rho_text = ""
        notes.append(f"Pearson r = {trend['r']:.2f} ({_strength(trend['r'])}), R² = {trend['r2']:.2f}, n = {len(frame):,}{rho_text}. Correlation is not causation.")

    idx = _sample_indices(len(frame), MAX_SCATTER_POINTS)
    if len(idx) < len(frame):
        notes.append(f"Showing a random {len(idx):,} of {len(frame):,} points (the trend line uses all of them).")
    shown = frame.iloc[idx]
    data: list[dict[str, Any]] = []
    if group:
        counts = frame["g"].value_counts()
        keep = list(counts.index[:MAX_OVERLAP_GROUPS])
        for i, name in enumerate(keep):
            sub = shown[shown["g"] == name]
            data.append({"type": "scatter", "mode": "markers", "x": sub["x"], "y": sub["y"], "name": _short(name),
                         "marker": {"color": _color(i), "size": 6, "opacity": 0.7, "line": {"width": 0.5, "color": "#0c1020"}}})
        rest = shown[~shown["g"].isin(keep)]
        if len(rest):
            data.append({"type": "scatter", "mode": "markers", "x": rest["x"], "y": rest["y"], "name": "Other",
                         "marker": {"color": OTHER_COLOR, "size": 5, "opacity": 0.45}})
            notes.append(f"Colours show the {MAX_OVERLAP_GROUPS} largest '{group}' groups; the rest are grey.")
    else:
        data.append({"type": "scatter", "mode": "markers", "x": shown["x"], "y": shown["y"], "name": f"{x} vs {y}",
                     "marker": {"color": shown["y"], "colorscale": VIVID, "showscale": False, "size": 8, "opacity": 0.85,
                                "line": {"width": 0.5, "color": "#0c1020"}},
                     "showlegend": False})
    # A straight line fit through skewed/heavy-tailed data (e.g. income-like measures) can shoot far
    # outside the actual data's y-range by the time it reaches the x extremes - Plotly's autorange
    # would then stretch the y-axis to fit that overshoot, squeezing every real point into a sliver
    # of the plot. The axis range is fixed to the data itself (with a little padding) so the trend
    # line can run off the top/bottom of the frame - normal for an extrapolated fit - without
    # distorting the scale the actual points are read against.
    y_lo, y_hi = float(frame["y"].min()), float(frame["y"].max())
    y_pad = (y_hi - y_lo) * 0.06 or max(abs(y_hi), 1.0) * 0.1
    x_lo, x_hi = float(frame["x"].min()), float(frame["x"].max())
    x_pad = (x_hi - x_lo) * 0.04 or max(abs(x_hi), 1.0) * 0.1
    if trend:
        data.append({"type": "scatter", "mode": "lines", "x": [x_lo, x_hi], "y": [trend["intercept"] + trend["slope"] * x_lo, trend["intercept"] + trend["slope"] * x_hi],
                     "name": "Linear trend", "line": {"color": INK, "width": 2, "dash": "dash"}, "hoverinfo": "skip"})
    layout = {"xaxis": _axis(cfg.get("xlabel") or str(x), range=[x_lo - x_pad, x_hi + x_pad]),
              "yaxis": _axis(cfg.get("ylabel") or str(y), range=[y_lo - y_pad, y_hi + y_pad])}
    return _finish(data, layout, f"{y} vs {x}", notes, "scatter", stats=trend)


def _box(df: pd.DataFrame, cfg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    profiles = schema["profiles"]
    x, y = cfg.get("x"), cfg.get("y")
    kind = "violin" if cfg.get("type") == "violin" else "box"
    yv = _numeric(df, y, profiles) if y else None
    if yv is None and x and _numeric(df, x, profiles) is not None and not y:
        yv, x = _numeric(df, x, profiles), None
    if yv is None:
        raise ChartError("Pick a numeric field for the values (Y).")
    notes: list[str] = []
    frame = pd.DataFrame({"y": yv})
    if x and x in df.columns and x != y:
        frame["x"] = _category_labels(df[x], profiles.get(x, {}))
    frame = frame.dropna(subset=["y"])
    if frame.empty:
        raise ChartError("No numeric values to plot.")

    data: list[dict[str, Any]] = []
    if "x" in frame:
        frame = frame.dropna(subset=["x"])
        counts = frame["x"].value_counts()
        natural = bool(profiles.get(x, {}).get("notes", {}).get("order")) or _role(profiles, x) in {"ordinal_numeric", "binary_numeric", "year"}
        order = _ordered_categories(frame["x"], profiles.get(x, {}), counts if not natural else None)
        if not natural:
            medians = frame.groupby("x")["y"].median()
            order = list(medians.reindex(order).sort_values(ascending=False).index) if len(order) > 4 else order
        order, dropped = _limit_categories(order, 12)
        if dropped:
            notes.append(f"Showing the 12 largest of {len(counts)} '{x}' groups.")
        for name in order:
            sub = frame.loc[frame["x"] == name, "y"].to_numpy()
            sub = sub[_sample_indices(len(sub), MAX_BOX_POINTS)]
            data.append(_box_trace(kind, _short(name), sub, x_label=_short(name), n=int((frame["x"] == name).sum()), color=_color(len(data))))
        try:
            from scipy import stats as sp

            samples = [frame.loc[frame["x"] == name, "y"].to_numpy() for name in order if (frame["x"] == name).sum() >= 2]
            if len(samples) >= 2:
                p = float(sp.kruskal(*samples)[1])
                allv = np.concatenate(samples)
                grand = allv.mean()
                ss_between = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
                ss_total = float(((allv - grand) ** 2).sum())
                eta2 = ss_between / ss_total if ss_total else 0.0
                verdict = "differ significantly" if p < 0.05 else "do not differ significantly"
                notes.append(f"'{y}' {verdict} across '{x}' groups (Kruskal-Wallis p = {p:.3g}; group membership explains {eta2:.1%} of the variance).")
        except Exception:
            pass
        xaxis = _axis(cfg.get("xlabel") or str(x), type="category")
    else:
        vals = frame["y"].to_numpy()
        data.append(_box_trace(kind, str(y), vals[_sample_indices(len(vals), MAX_BOX_POINTS)], x_label=str(y), n=len(vals)))
        xaxis = _axis(None, showticklabels=False)
    layout = {"xaxis": xaxis, "yaxis": _axis(cfg.get("ylabel") or str(y)), "boxmode": "group"}
    data_layout_title = f"{y} by {x}" if "x" in frame else f"Distribution of {y}"
    return _finish(data, layout, data_layout_title, notes, kind)


def _box_trace(kind: str, name: str, values: np.ndarray, x_label: str, n: int, color: str | None = None) -> dict[str, Any]:
    color = color or _color(0)
    trace: dict[str, Any] = {"type": kind, "y": values, "name": name, "showlegend": False,
                             "marker": {"color": color}, "line": {"color": color, "width": 2},
                             "hovertemplate": f"{name} (n = {n:,})<br>%{{y:,.4g}}<extra></extra>"}
    if kind == "box":
        trace.update({"boxmean": True, "boxpoints": "outliers", "fillcolor": color + "66"})
    else:
        trace.update({"points": False, "meanline": {"visible": True}, "box": {"visible": True}, "fillcolor": color + "66", "spanmode": "hard"})
    return trace


def _correlation_columns(df: pd.DataFrame, schema: dict[str, Any], limit: int = 14) -> list[str]:
    profiles = schema["profiles"]
    cols = [c for c in schema.get("measures", []) if profiles[c]["role"] in {"numeric", "ordinal_numeric", "binary_numeric"}]
    if len(cols) <= limit:
        return cols
    numeric = pd.DataFrame({c: _numeric(df, c, profiles) for c in cols})
    corr = numeric.corr().abs().fillna(0)
    arr = corr.to_numpy(copy=True)
    np.fill_diagonal(arr, 0)
    ranked = pd.DataFrame(arr, index=corr.index, columns=corr.columns).max().sort_values(ascending=False)
    return list(ranked.index[:limit])


def _heatmap(df: pd.DataFrame, cfg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    profiles = schema["profiles"]
    cols = _correlation_columns(df, schema)
    if len(cols) < 2:
        raise ChartError("A correlation matrix needs at least two numeric columns.")
    numeric = pd.DataFrame({c: _numeric(df, c, profiles) for c in cols})
    corr = numeric.corr().fillna(0.0)
    notes: list[str] = []
    if len(cols) > 2:
        try:
            from scipy.cluster import hierarchy
            from scipy.spatial.distance import squareform

            dist = 1 - corr.abs().to_numpy()
            np.fill_diagonal(dist, 0)
            order = hierarchy.leaves_list(hierarchy.linkage(squareform(np.clip(dist, 0, None), checks=False), "average"))
            corr = corr.iloc[order, order]
            notes.append("Columns are ordered so strongly related ones sit together.")
        except Exception:
            pass
    total = len(schema.get("measures", []))
    if len(cols) < total:
        notes.append(f"Showing the {len(cols)} columns with the strongest relationships out of {total}.")
    labels = [_short(c, 20) for c in corr.columns]
    z = corr.round(2).to_numpy()
    data = [{"type": "heatmap", "z": z, "x": labels, "y": labels, "zmin": -1, "zmax": 1, "colorscale": DIVERGING,
             "hovertemplate": "%{y} × %{x}<br>r = %{z:.2f}<extra></extra>", "xgap": 2, "ygap": 2,
             "colorbar": {"title": {"text": "Pearson r", "font": {"color": INK_MUTED}}, "tickfont": {"color": INK_MUTED}, "thickness": 12}}]
    if len(cols) <= 10:
        data[0]["texttemplate"] = "%{z:.2f}"
        data[0]["textfont"] = {"color": INK, "size": 11}
    upper = np.triu(np.abs(corr.to_numpy()), k=1)
    strongest = np.unravel_index(np.argmax(upper), upper.shape) if upper.size else None
    if strongest is not None and upper[strongest] > 0:
        a, b = corr.columns[strongest[0]], corr.columns[strongest[1]]
        notes.append(f"Strongest pair: {a} and {b} (r = {corr.iloc[strongest]:.2f}, {_strength(corr.iloc[strongest])}).")
    layout = {"xaxis": _axis(None, side="bottom", tickangle=-35, showgrid=False), "yaxis": _axis(None, autorange="reversed", showgrid=False)}
    return _finish(data, layout, "Correlation matrix", notes, "heatmap")


def _missing(df: pd.DataFrame, cfg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    pct = (df.isna().mean() * 100).round(2)
    pct = pct[pct > 0].sort_values(ascending=False)
    if pct.empty:
        return _message_figure("No missing values in this dataset.")
    total = len(pct)
    pct = pct.iloc[:25]
    notes = [f"{total} of {len(df.columns)} columns have missing values; the worst is '{pct.index[0]}' at {pct.iloc[0]:.1f}%."]
    if total > 25:
        notes.append("Showing the 25 columns with the most missing values.")
    counts = df[pct.index].isna().sum()
    data = [{"type": "bar", "orientation": "h", "y": [_short(c, 30) for c in pct.index][::-1], "x": pct.values[::-1], "name": "% missing",
             "marker": {"color": pct.values[::-1], "colorscale": VIVID, "cmin": 0, "cmax": 100, "showscale": False},
             "showlegend": False, "customdata": counts.values[::-1],
             "hovertemplate": "%{y}: %{x:.1f}% missing (%{customdata:,} rows)<extra></extra>"}]
    layout = {"xaxis": _axis(cfg.get("xlabel") or "% of rows missing", range=[0, max(5, float(pct.max()) * 1.1)]),
              "yaxis": _axis(None, type="category"), "bargap": 0.3, "margin": {"l": 150}}
    return _finish(data, layout, "Missing values by column", notes, "missing")


def _pair(df: pd.DataFrame, cfg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    profiles = schema["profiles"]
    cols = [c for c in _correlation_columns(df, schema, limit=12) if profiles[c]["role"] == "numeric"] or _correlation_columns(df, schema, limit=12)
    if len(cols) < 3:
        raise ChartError("A pair plot needs at least three numeric columns.")
    numeric = pd.DataFrame({c: _numeric(df, c, profiles) for c in cols})
    corr = numeric.corr().abs().fillna(0)
    arr = corr.to_numpy(copy=True)
    np.fill_diagonal(arr, 0)
    chosen = list(pd.DataFrame(arr, index=corr.index, columns=corr.columns).sum().sort_values(ascending=False).index[:5])
    frame = numeric[chosen].dropna()
    if len(frame) < 3:
        raise ChartError("Too few complete rows for a pair plot.")
    notes = [f"Comparing the {len(chosen)} columns that relate most strongly to the others."]
    idx = _sample_indices(len(frame), 1200)
    if len(idx) < len(frame):
        notes.append(f"Showing a random {len(idx):,} of {len(frame):,} complete rows.")
    shown = frame.iloc[idx]
    dims = [{"label": _short(c, 14), "values": shown[c]} for c in chosen]
    group = cfg.get("group") if cfg.get("group") in df.columns and _is_grouper(profiles, cfg.get("group")) else None
    data: list[dict[str, Any]] = []
    if group:
        labels = _category_labels(df[group], profiles.get(group, {})).reindex(shown.index)
        keep = list(labels.value_counts().index[:MAX_OVERLAP_GROUPS])
        for i, name in enumerate(keep):
            sub = shown[labels == name]
            data.append({"type": "splom", "dimensions": [{"label": _short(c, 14), "values": sub[c]} for c in chosen], "name": _short(name),
                         "marker": {"color": _color(i), "size": 4, "opacity": 0.6}, "diagonal": {"visible": False}, "showupperhalf": False})
        notes.append(f"Colours show the {MAX_OVERLAP_GROUPS} largest '{group}' groups.")
    else:
        data.append({"type": "splom", "dimensions": dims, "name": "rows", "showlegend": False,
                     "marker": {"color": _color(0), "size": 4, "opacity": 0.55}, "diagonal": {"visible": False}, "showupperhalf": False})
    layout = {"dragmode": "select", "margin": {"t": 44, "r": 20, "l": 60, "b": 60}}
    for i in range(1, len(chosen) + 1):
        suffix = "" if i == 1 else str(i)
        layout[f"xaxis{suffix}"] = _axis(None, showgrid=True, tickfont={"color": INK_MUTED, "size": 9})
        layout[f"yaxis{suffix}"] = _axis(None, showgrid=True, tickfont={"color": INK_MUTED, "size": 9})
    return _finish(data, layout, "Pair plot", notes, "pair")


def _treemap(df: pd.DataFrame, cfg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    profiles = schema["profiles"]
    x = cfg.get("x")
    if not x or x not in df.columns:
        raise ChartError("Pick a categorical field for the treemap tiles.")
    y = cfg.get("y") if cfg.get("y") in df.columns else None
    y_values = _numeric(df, y, profiles) if y else None
    agg = _agg_name(cfg.get("agg"), y_values is not None)
    notes: list[str] = []
    group = cfg.get("group") if cfg.get("group") in df.columns and cfg.get("group") != x else None
    frame = pd.DataFrame({"x": _category_labels(df[x], profiles.get(x, {}))})
    if y_values is not None:
        frame["y"] = y_values
    if group:
        frame["g"] = _category_labels(df[group], profiles.get(group, {}))
    frame = frame.dropna(subset=["x"] + (["g"] if group else []))
    if frame.empty:
        raise ChartError("No rows to chart.")

    if y_values is not None and agg in {"mean", "median", "max", "min"}:
        size_col = "count"
        size_table = _aggregate(frame, ["x"] + (["g"] if group else []), None, "count").rename(columns={"value": "size"})
        color_table = _aggregate(frame, ["x"] + (["g"] if group else []), "y", agg).rename(columns={"value": "color"})
        table = size_table.merge(color_table, on=["x"] + (["g"] if group else []))
        notes.append(f"Tile size = number of rows; tile colour = {_measure_label(y, agg).lower()}.")
    else:
        table = _aggregate(frame, ["x"] + (["g"] if group else []), "y" if y_values is not None else None, agg).rename(columns={"value": "size"})
        if y_values is not None and (table["size"] < 0).any():
            table = _aggregate(frame, ["x"] + (["g"] if group else []), None, "count").rename(columns={"value": "size"})
            notes.append(f"'{y}' has negative values, which can't size a tile, so tiles are sized by row count instead.")
        table["color"] = table["size"]
        size_col = agg

    tiles = table.sort_values("size", ascending=False)
    if len(tiles) > 40:
        notes.append(f"Showing the 40 largest of {len(tiles)} tiles.")
        tiles = tiles.iloc[:40]
    ids, labels, parents, values, colors = [], [], [], [], []
    if group:
        for name, sub in tiles.groupby("g"):
            ids.append(f"g:{name}")
            labels.append(_short(name))
            parents.append("")
            values.append(float(sub["size"].sum()))
            colors.append(float(sub["color"].mean()))
        for _, row in tiles.iterrows():
            ids.append(f"g:{row['g']}/{row['x']}")
            labels.append(_short(row["x"]))
            parents.append(f"g:{row['g']}")
            values.append(float(row["size"]))
            colors.append(float(row["color"]))
    else:
        for _, row in tiles.iterrows():
            ids.append(f"x:{row['x']}")
            labels.append(_short(row["x"]))
            parents.append("")
            values.append(float(row["size"]))
            colors.append(float(row["color"]))
    data = [{"type": "treemap", "ids": ids, "labels": labels, "parents": parents, "values": values, "branchvalues": "total",
             "marker": {"colors": colors, "colorscale": VIVID, "line": {"color": "#0c1020", "width": 2},
                        "colorbar": {"title": {"text": _measure_label(y, agg) if y_values is not None else "Rows", "font": {"color": INK_MUTED}}, "tickfont": {"color": INK_MUTED}, "thickness": 12}},
             "textfont": {"color": INK}, "hovertemplate": "%{label}<br>%{value:,.4g}<extra></extra>", "tiling": {"pad": 3}}]
    layout = {"margin": {"t": 44, "r": 10, "l": 10, "b": 10}}
    return _finish(data, layout, f"{x} breakdown", notes, "treemap")


def _map(df: pd.DataFrame, cfg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    profiles = schema["profiles"]
    geo = schema.get("geo", {})
    x = cfg.get("x")
    y = cfg.get("y") if cfg.get("y") in df.columns else None
    y_values = _numeric(df, y, profiles) if y else None
    agg = _agg_name(cfg.get("agg"), y_values is not None)
    label = _measure_label(y, agg)
    notes: list[str] = []

    role = _role(profiles, x)
    lat_lon = geo.get("lat_lon")
    if role not in {"geo_country", "geo_state"}:
        if x in df.columns:
            # any text column may still be a place column the detector didn't flag
            codes, unresolved = geo_tables.map_locations(df[x], "country")
            if codes.notna().sum() >= 3:
                role = "geo_country"
            else:
                role = ""
        else:
            role = ""
    if role in {"geo_country", "geo_state"}:
        kind = "us_state" if role == "geo_state" else "country"
        codes, unresolved = geo_tables.map_locations(df[x], kind)
        frame = pd.DataFrame({"loc": codes, "name": df[x].astype(str)})
        if y_values is not None:
            frame["y"] = y_values
        frame = frame.dropna(subset=["loc"])
        if frame.empty:
            raise ChartError(f"None of the values in '{x}' could be matched to a {'US state' if kind == 'us_state' else 'country'}.")
        table = _aggregate(frame, ["loc"], "y" if y_values is not None else None, agg)
        names = frame.drop_duplicates("loc").set_index("loc")["name"]
        table["name"] = table["loc"].map(names)
        if unresolved:
            notes.append(f"{unresolved:,} rows had a value that isn't a recognised {'state' if kind == 'us_state' else 'country'} and were left off the map.")
        time = schema.get("time")
        if time and time.get("is_panel") and time.get("entity_column") == x and y_values is not None:
            notes.append(f"Each region shows the {agg} of {y} across all {time['n_periods']} time periods; filter to one period with a dataset editor to compare a single year.")
        data = [{"type": "choropleth", "locations": table["loc"], "z": table["value"], "text": table["name"],
                 "locationmode": "USA-states" if kind == "us_state" else "ISO-3", "colorscale": VIVID,
                 "marker": {"line": {"color": "#0c1020", "width": 0.6}},
                 "colorbar": {"title": {"text": label, "font": {"color": INK_MUTED}}, "tickfont": {"color": INK_MUTED}, "thickness": 12, "len": 0.7},
                 "hovertemplate": "%{text}<br>" + label + ": %{z:,.4g}<extra></extra>"}]
        layout = {"geo": _geo_layout("usa" if kind == "us_state" else "world"), "margin": {"t": 44, "r": 10, "l": 10, "b": 10}}
        notes.append(f"{len(table)} regions drawn; darker = lower, lighter = higher.")
        return _finish(data, layout, f"{label} by {x}", notes, "map")

    if lat_lon:
        lat, lon = lat_lon["lat"], lat_lon["lon"]
        frame = pd.DataFrame({"lat": _numeric(df, lat, profiles), "lon": _numeric(df, lon, profiles)})
        if y_values is not None:
            frame["y"] = y_values
        frame = frame.dropna(subset=["lat", "lon"])
        idx = _sample_indices(len(frame), 3000)
        if len(idx) < len(frame):
            notes.append(f"Showing a random {len(idx):,} of {len(frame):,} points.")
        shown = frame.iloc[idx]
        marker: dict[str, Any] = {"size": 7, "opacity": 0.75}
        if y_values is not None:
            marker.update({"color": shown["y"], "colorscale": VIVID, "showscale": True,
                           "colorbar": {"title": {"text": str(y), "font": {"color": INK_MUTED}}, "tickfont": {"color": INK_MUTED}, "thickness": 12}})
        else:
            marker["color"] = _color(0)
        data = [{"type": "scattergeo", "lat": shown["lat"], "lon": shown["lon"], "mode": "markers", "marker": marker, "showlegend": False,
                 "text": shown["y"] if y_values is not None else None, "hovertemplate": "lat %{lat:.3f}, lon %{lon:.3f}" + ("<br>%{text:,.4g}" if y_values is not None else "") + "<extra></extra>"}]
        layout = {"geo": _geo_layout("world"), "margin": {"t": 44, "r": 10, "l": 10, "b": 10}}
        return _finish(data, layout, f"Locations ({lat}, {lon})", notes, "map")
    raise ChartError("No geographic column found. A map needs a column of countries or US states, or latitude/longitude columns.")


def _geo_layout(scope: str) -> dict[str, Any]:
    return {"scope": scope, "showframe": False, "showcoastlines": False, "showland": True, "landcolor": "#141a2e",
            "showcountries": True, "countrycolor": "#2a3352", "bgcolor": "rgba(0,0,0,0)", "showlakes": False,
            "projection": {"type": "albers usa" if scope == "usa" else "natural earth"}}


def _distribution(df: pd.DataFrame, cfg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    profiles = schema["profiles"]
    x, group = cfg.get("x"), cfg.get("group")
    values = _numeric(df, x, profiles) if x else None
    if values is None:
        raise ChartError("Pick a numeric field to compare (X).")
    if not group or group not in df.columns or not _is_grouper(profiles, group):
        return _histogram(df, {**cfg, "type": "histogram"}, schema)
    labels = _category_labels(df[group], profiles.get(group, {}))
    frame = pd.DataFrame({"v": values, "g": labels}).dropna()
    if frame.empty:
        raise ChartError("No rows have both values.")
    counts = frame["g"].value_counts()
    order = _ordered_categories(frame["g"], profiles.get(group, {}), counts if not profiles.get(group, {}).get("notes", {}).get("order") else None)
    keep, dropped = _limit_categories(order, MAX_OVERLAP_GROUPS)
    notes: list[str] = []
    if dropped:
        notes.append(f"Showing the {MAX_OVERLAP_GROUPS} largest of {len(order)} '{group}' groups so the overlapping curves stay distinguishable.")
    edges = np.histogram_bin_edges(frame["v"], bins="auto")
    if len(edges) > 41:
        edges = np.histogram_bin_edges(frame["v"], bins=40)
    centers = (edges[:-1] + edges[1:]) / 2
    data = []
    for i, name in enumerate(keep):
        sub = frame.loc[frame["g"] == name, "v"].to_numpy()
        density, _ = np.histogram(sub, bins=edges, density=True)
        data.append({"type": "scatter", "mode": "lines", "x": centers, "y": density, "name": f"{_short(name)} (n = {len(sub):,})",
                     "line": {"color": _color(i), "width": 2, "shape": "spline", "smoothing": 0.8}, "fill": "tozeroy", "fillcolor": _color(i) + "30",
                     "hovertemplate": f"{_short(name)}<br>{x} ≈ %{{x:.4g}}<br>density %{{y:.3g}}<extra></extra>"})
    samples = [frame.loc[frame["g"] == name, "v"].to_numpy() for name in order if (frame["g"] == name).sum() >= 2]
    try:
        from scipy import stats as sp

        if len(samples) >= 2:
            p = float(sp.kruskal(*samples)[1])
            means = ", ".join(f"{_short(n)}: {frame.loc[frame['g'] == n, 'v'].mean():,.4g}" for n in keep)
            notes.append(f"Group means - {means}.")
            notes.append(f"The groups {'differ significantly' if p < 0.05 else 'do not differ significantly'} (Kruskal-Wallis p = {p:.3g}).")
    except Exception:
        pass
    layout = {"xaxis": _axis(cfg.get("xlabel") or str(x)), "yaxis": _axis(cfg.get("ylabel") or "Density")}
    return _finish(data, layout, f"{x} by {group}", notes, "distribution")


def _cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    from scipy import stats as sp

    chi2, p, _, _ = sp.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    v = math.sqrt(chi2 / (n * k)) if n and k > 0 else 0.0
    return float(v), float(p)


def _target(df: pd.DataFrame, cfg: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    profiles = schema["profiles"]
    feature, target = cfg.get("x"), cfg.get("y")
    if not feature or not target or feature not in df.columns or target not in df.columns:
        raise ChartError("Pick a feature (X) and a target (Y) to compare.")
    t_role, f_role = _role(profiles, target), _role(profiles, feature)
    t_num, f_num = _numeric(df, target, profiles), _numeric(df, feature, profiles)
    target_is_class = t_role in {"binary", "binary_numeric", "categorical", "ordinal"} or (t_role == "ordinal_numeric")
    feature_is_class = f_role in {"binary", "binary_numeric", "categorical", "ordinal", "ordinal_numeric", "geo_country", "geo_state"}

    if not target_is_class and t_num is not None:
        if f_num is not None and not feature_is_class:
            out = _scatter(df, {**cfg, "type": "scatter", "x": feature, "y": target, "group": None}, schema)
        else:
            out = _box(df, {**cfg, "type": "box", "x": feature, "y": target}, schema)
        out["meta"]["title"] = f"{target} vs {feature}"
        return out

    labels = _category_labels(df[target], profiles.get(target, {}))
    if f_num is not None and not feature_is_class:
        out = _box(df, {**cfg, "type": "box", "x": target, "y": feature}, schema)
        out["meta"]["title"] = f"{feature} by {target}"
        return out

    # class target vs class feature: how the target mix changes across feature categories
    frame = pd.DataFrame({"t": labels, "f": _category_labels(df[feature], profiles.get(feature, {}))}).dropna()
    if frame.empty:
        raise ChartError("No rows have both the feature and the target.")
    f_order, dropped = _limit_categories(_ordered_categories(frame["f"], profiles.get(feature, {}), frame["f"].value_counts() if not profiles.get(feature, {}).get("notes", {}).get("order") else None), MAX_BAR_CATEGORIES)
    frame = frame[frame["f"].isin(f_order)]
    t_order, t_dropped = _limit_categories(_ordered_categories(frame["t"], profiles.get(target, {}), frame["t"].value_counts() if not profiles.get(target, {}).get("notes", {}).get("order") else None), 5)
    notes: list[str] = []
    if dropped:
        notes.append(f"Showing the {MAX_BAR_CATEGORIES} largest '{feature}' categories.")
    if t_dropped:
        notes.append("Showing the 5 largest target classes.")
    ct = pd.crosstab(frame["f"], frame["t"]).reindex(index=f_order)
    share = ct.div(ct.sum(axis=1), axis=0) * 100
    data = []
    for i, name in enumerate(t_order):
        if name not in share.columns:
            continue
        data.append({"type": "bar", "x": [_short(c) for c in share.index], "y": share[name].round(2), "name": f"{target} = {_short(name)}",
                     "marker": {"color": _color(i)}, "customdata": ct[name].to_numpy(),
                     "hovertemplate": f"{target} = {_short(name)}<br>%{{x}}: %{{y:.1f}}% (%{{customdata:,}} rows)<extra></extra>"})
    try:
        v, p = _cramers_v(ct.loc[:, [c for c in t_order if c in ct.columns]])
        notes.append(f"Association between '{feature}' and '{target}': Cramér's V = {v:.2f} ({'strong' if v >= 0.5 else 'moderate' if v >= 0.3 else 'weak' if v >= 0.1 else 'negligible'}), chi-square p = {p:.3g}.")
    except Exception:
        pass
    layout = {"xaxis": _axis(cfg.get("xlabel") or str(feature), type="category"), "yaxis": _axis(cfg.get("ylabel") or f"% of rows", range=[0, 100]), "barmode": "stack", "bargap": 0.25}
    return _finish(data, layout, f"{target} by {feature}", notes, "target")


_BUILDERS = {
    "histogram": _histogram,
    "bar": _bar,
    "stacked_bar": _bar,
    "pie": _pie,
    "scatter": _scatter,
    "line": lambda df, cfg, schema: _line(df, cfg, schema, "line"),
    "area": lambda df, cfg, schema: _line(df, cfg, schema, "area"),
    "timeseries": lambda df, cfg, schema: _line(df, cfg, schema, "line"),
    "box": _box,
    "violin": _box,
    "heatmap": _heatmap,
    "distribution": _distribution,
    "target": _target,
    "treemap": _treemap,
    "map": _map,
    "missing": _missing,
    "pair": _pair,
}


def render_chart(df: pd.DataFrame, config: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Build one chart. Never raises for a bad request: returns a message figure instead, so a
    half-configured chart card shows what's missing rather than an error."""
    chart_type = (config.get("type") or "histogram").strip()
    builder = _BUILDERS.get(chart_type)
    if builder is None:
        return _clean(_message_figure(f"Unknown chart type '{chart_type}'."))
    cfg = {k: (None if v in ("", "none", None) and k not in ("agg", "sort", "granularity") else v) for k, v in config.items()}
    try:
        result = builder(df, cfg, schema)
    except ChartError as exc:
        result = _message_figure(str(exc))
    except Exception as exc:  # a chart bug must not take the workspace down
        result = _message_figure(f"Could not draw this chart ({type(exc).__name__}: {exc}).")
    custom = (config.get("customTitle") or "").strip()
    if custom:
        result.setdefault("meta", {})["title"] = custom
    result.setdefault("meta", {}).setdefault("title", CHART_TYPE_LABELS.get(chart_type, chart_type))
    return _clean(result)
