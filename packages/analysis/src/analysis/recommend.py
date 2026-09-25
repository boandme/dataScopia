"""Visualization recommendations driven by what the data actually contains.

The old recommender picked "the first numeric column" and "the first categorical column" for
every dataset. This one asks what story the data can tell, and only suggests a chart when a
statistic says there is something to show:

* a time axis with a measurable trend      -> time series
* a country / state / lat-lon column       -> map
* a plausible target with related features -> target-vs-feature
* two strongly correlated measures         -> scatter (with trend line)
* a measure whose distribution differs across groups (Kruskal-Wallis, eta-squared) -> distribution comparison
* several correlated measures              -> correlation matrix / pair plot
* a heavily skewed measure                 -> histogram
* meaningful missingness                   -> missing-value chart
* a categorical that dominates / breaks down a measure -> bar / treemap / stacked bar

Each recommendation carries a score, the family of story it tells, and a plain-language reason
containing the number that justified it. Ordering guarantees the first two suggestions (which
become the default chart cards) tell different stories.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from . import schema as schema_mod

_TARGET_HINT = re.compile(
    r"(^|[_\s\-])(target|label|outcome|class|churn|churned|survived|default|fraud|price|salary|income|revenue|sales|"
    r"score|rating|y|response|result|diagnosis|passed|approved|cancel|cancelled|readmitted|attrition)($|[_\s\-])",
    re.I,
)
_ADDITIVE_HINT = re.compile(r"amount|sales|revenue|population|count|total|quantity|qty|volume|units|cost|spend|profit|income", re.I)

SAMPLE_LIMIT = 20000


def _sample(df: pd.DataFrame, limit: int = SAMPLE_LIMIT) -> pd.DataFrame:
    return df if len(df) <= limit else df.sample(limit, random_state=0)


def _numeric_frame(df: pd.DataFrame, columns: list[str], profiles: dict[str, Any]) -> pd.DataFrame:
    data = {}
    for column in columns:
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            data[column] = series.astype(float)
        elif pd.api.types.is_numeric_dtype(series):
            data[column] = series.astype(float)
        else:
            data[column] = schema_mod.coerce_numeric(series)
    return pd.DataFrame(data)


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    grand = values.mean()
    ss_total = float(((values - grand) ** 2).sum())
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for label in np.unique(labels):
        group = values[labels == label]
        ss_between += len(group) * (group.mean() - grand) ** 2
    return float(ss_between / ss_total)


def _kruskal_p(values: np.ndarray, labels: np.ndarray) -> float:
    from scipy import stats

    groups = [values[labels == label] for label in np.unique(labels)]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return 1.0
    try:
        return float(stats.kruskal(*groups)[1])
    except ValueError:  # every value identical
        return 1.0


def _cramers_v(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    from scipy import stats

    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, 1.0
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return (math.sqrt(chi2 / (n * k)) if n and k else 0.0), float(p)


def _low_card_groupers(schema: dict[str, Any], limit: int = 12) -> list[str]:
    profiles = schema["profiles"]
    return [c for c in schema["groupers"]
            if profiles[c]["role"] in {"categorical", "binary", "ordinal", "ordinal_numeric"} and 2 <= profiles[c]["unique_count"] <= limit]


# ------------------------------------------------------------------------------------
# targets
# ------------------------------------------------------------------------------------


def suggest_targets(df: pd.DataFrame, schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Columns that look like something you'd predict, ranked. A column needs either a
    target-like name or a binary shape to qualify: a random low-cardinality number is not a
    label just because it has few values."""
    profiles = schema["profiles"]
    columns = list(df.columns)
    scored: list[tuple[float, str, str, str]] = []
    for position, column in enumerate(columns):
        profile = profiles[column]
        role = profile["role"]
        if role in {"identifier", "text", "constant", "empty", "datetime", "year", "geo_country", "geo_state", "geo_lat", "geo_lon", "unknown"}:
            continue
        hinted = bool(_TARGET_HINT.search(str(column)))
        score = 0.0
        reasons: list[str] = []
        if hinted:
            score += 3
            reasons.append("its name suggests an outcome")
        if role in {"binary", "binary_numeric"}:
            score += 2.5
            reasons.append("it has exactly two values (a classic yes/no label)")
        elif role in {"categorical", "ordinal", "ordinal_numeric"} and profile["unique_count"] <= 10:
            score += 1
            reasons.append(f"it has only {profile['unique_count']} classes")
        elif role == "numeric" and hinted:
            reasons.append("it is a continuous quantity with a target-like name")
        if position == len(columns) - 1 and score > 0:
            score += 0.5
            reasons.append("it is the last column, where labels usually sit")
        if score >= 3:
            kind = "regression" if role == "numeric" else "classification"
            scored.append((score, column, kind, "; ".join(reasons)))
    scored.sort(key=lambda item: -item[0])
    return [{"column": c, "kind": k, "reason": r, "score": s} for s, c, k, r in scored[:6]]


def _feature_associations(df: pd.DataFrame, schema: dict[str, Any], target: str, kind: str) -> list[tuple[float, str, str]]:
    """(strength, feature, description) for each usable feature vs the target, strongest first."""
    profiles = schema["profiles"]
    sample = _sample(df)
    out: list[tuple[float, str, str]] = []
    tgt_num = _numeric_frame(sample, [target], profiles)[target] if kind == "regression" else None
    for feature in df.columns:
        if feature == target:
            continue
        role = profiles[feature]["role"]
        if role in {"identifier", "text", "constant", "empty", "datetime", "year", "geo_lat", "geo_lon", "unknown", "geo_country", "geo_state"}:
            continue
        try:
            if kind == "regression":
                if role in {"numeric", "ordinal_numeric", "binary_numeric"}:
                    f = _numeric_frame(sample, [feature], profiles)[feature]
                    pair = pd.concat([f, tgt_num], axis=1).dropna()
                    if len(pair) < 10 or pair.iloc[:, 0].nunique() < 2:
                        continue
                    r = float(pair.corr().iloc[0, 1])
                    out.append((abs(r), feature, f"correlation r = {r:.2f}"))
                elif profiles[feature]["unique_count"] <= 20:
                    pair = pd.DataFrame({"v": tgt_num, "g": sample[feature].astype(str)}).dropna()
                    if len(pair) < 10:
                        continue
                    eta2 = _eta_squared(pair["v"].to_numpy(), pair["g"].to_numpy())
                    out.append((math.sqrt(eta2), feature, f"explains {eta2:.0%} of the variance"))
            else:
                labels = sample[target].astype(str)
                if role in {"numeric", "ordinal_numeric", "binary_numeric"}:
                    f = _numeric_frame(sample, [feature], profiles)[feature]
                    pair = pd.DataFrame({"v": f, "g": labels}).dropna()
                    if len(pair) < 10 or pair["v"].nunique() < 2:
                        continue
                    eta2 = _eta_squared(pair["v"].to_numpy(), pair["g"].to_numpy())
                    out.append((math.sqrt(eta2), feature, f"separates the classes (eta² = {eta2:.2f})"))
                elif profiles[feature]["unique_count"] <= 20:
                    v, p = _cramers_v(sample[feature].astype(str), labels)
                    out.append((v, feature, f"associated with the target (Cramér's V = {v:.2f})"))
        except Exception:
            continue
    out.sort(key=lambda item: -item[0])
    return out


# ------------------------------------------------------------------------------------
# recommendations
# ------------------------------------------------------------------------------------


def _rec(chart_type: str, family: str, score: float, title: str, reason: str, **config: Any) -> dict[str, Any]:
    return {"type": chart_type, "family": family, "score": round(float(score), 1), "title": title, "reason": reason, **config}


def _best_time_measure(df: pd.DataFrame, schema: dict[str, Any], measures: list[str]) -> tuple[str, float, float] | None:
    """The measure whose per-period average changes most consistently over time.
    Returns (measure, |trend|, change) where change is last/first - 1 of the period means."""
    time = schema["time"]
    profiles = schema["profiles"]
    times = schema_mod.time_values(df, time["time_column"], profiles[time["time_column"]])
    frame = _numeric_frame(df, measures, profiles).assign(_t=times).dropna(subset=["_t"])
    if frame.empty:
        return None
    means = frame.groupby("_t")[measures].mean()
    if len(means) < 4:
        return None
    best = None
    ranks = np.arange(len(means))
    for measure in measures:
        series = means[measure].dropna()
        if len(series) < 4 or series.nunique() < 3:
            continue
        r = abs(float(np.corrcoef(np.arange(len(series)), series.to_numpy())[0, 1]))
        first, last = float(series.iloc[0]), float(series.iloc[-1])
        change = (last / first - 1) if first not in (0.0,) and math.isfinite(first) else float("nan")
        cv = float(series.std() / abs(series.mean())) if series.mean() else 0.0
        strength = r + min(cv, 1.0) * 0.3
        if best is None or strength > best[1]:
            best = (measure, strength, change)
    return best


def recommend_visualizations(
    df: pd.DataFrame,
    schema: dict[str, Any],
    relationships: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    skewed: list[str],
    missing_by_column: dict[str, float],
) -> list[dict[str, Any]]:
    profiles = schema["profiles"]
    recs: list[dict[str, Any]] = []
    continuous = [c for c in schema["measures"] if profiles[c]["role"] == "numeric"]
    all_measures = list(schema["measures"])
    time = schema.get("time")
    geo = schema.get("geo", {})
    used_pairs: set[frozenset[str]] = set()
    sample = _sample(df)

    # ---- time -------------------------------------------------------------------------
    if time and continuous:
        best = _best_time_measure(df, schema, continuous[:12])
        if best:
            measure, strength, change = best
            entity = time.get("entity_column") if time.get("is_panel") and (time.get("n_entities") or 99) <= 6 else None
            granularity = "none"
            if time["kind"] == "datetime":
                if time["frequency"] == "daily" and time["n_periods"] > 180:
                    granularity = "month"
                elif time["frequency"] in {"hourly"} and time["n_periods"] > 200:
                    granularity = "day"
            direction = ""
            if math.isfinite(change):
                direction = f" Its {'average' if not entity else 'level'} moved {change:+.0%} between the first and last period."
            span = f"{time['start'][:10]} to {time['end'][:10]}" if time["kind"] == "datetime" else f"{time['start'][:4]} to {time['end'][:4]}"
            recs.append(_rec("timeseries", "time", 96 + min(strength, 1) * 2, f"{measure} over {time['time_column']}",
                             f"The data has a time axis ({span}, {time['frequency']}, {time['n_periods']} periods) and {measure} is the measure that changes most consistently across it.{direction}",
                             x=time["time_column"], y=measure, agg="mean", group=entity, granularity=granularity))
            others = [m for m in continuous if m != measure][:1]
            for other in others:
                recs.append(_rec("timeseries", "time", 66, f"{other} over {time['time_column']}",
                                 f"A second view of how {other} evolves across the same time axis.",
                                 x=time["time_column"], y=other, agg="mean", group=None, granularity=granularity))

    # ---- geography --------------------------------------------------------------------
    geo_column = None
    geo_label = ""
    if geo.get("country"):
        geo_column, geo_label = geo["country"]["column"], "countries"
    elif geo.get("us_state"):
        geo_column, geo_label = geo["us_state"]["column"], "US states"
    if geo_column and continuous:
        from . import geo as geo_tables

        kind = "us_state" if geo.get("us_state") and geo["us_state"]["column"] == geo_column else "country"
        codes, _ = geo_tables.map_locations(sample[geo_column], kind)
        numeric = _numeric_frame(sample, continuous[:12], profiles).assign(_loc=codes).dropna(subset=["_loc"])
        best_measure, best_cv = None, 0.0
        for measure in continuous[:12]:
            per_loc = numeric.groupby("_loc")[measure].mean().dropna()
            if len(per_loc) < 4 or not per_loc.mean():
                continue
            cv = float(per_loc.std() / abs(per_loc.mean()))
            if cv > best_cv:
                best_measure, best_cv = measure, cv
        if best_measure:
            recs.append(_rec("map", "geo", 94, f"{best_measure} by {geo_column}",
                             f"'{geo_column}' holds {geo_label}, so {best_measure} can be drawn on a map; it varies substantially between regions (coefficient of variation {best_cv:.2f}).",
                             x=geo_column, y=best_measure, agg="mean"))
    elif geo.get("lat_lon"):
        measure = continuous[0] if continuous else None
        recs.append(_rec("map", "geo", 88, "Where the rows are located",
                         "The data has latitude and longitude columns, so each row can be placed on a map.",
                         x=geo["lat_lon"]["lat"], y=measure, agg="mean"))

    # ---- target vs feature ------------------------------------------------------------
    for target in targets[:2]:
        ranked = _feature_associations(df, schema, target["column"], target["kind"])
        strong = [item for item in ranked if item[0] >= 0.1][:2]
        for i, (strength, feature, description) in enumerate(strong):
            recs.append(_rec("target", "target", (90 if target is targets[0] else 74) + strength * 8 - i * 6,
                             f"{target['column']} vs {feature}",
                             f"'{target['column']}' looks like a {'label to classify' if target['kind'] == 'classification' else 'quantity to predict'} ({target['reason']}); {feature} {description}.",
                             x=feature, y=target["column"], agg="none"))
            used_pairs.add(frozenset({feature, target["column"]}))

    # ---- strongest relationship -------------------------------------------------------
    for rel in relationships[:3]:
        pair = frozenset({rel["column_a"], rel["column_b"]})
        r = abs(rel["correlation"])
        if r < 0.3 or pair in used_pairs:
            continue
        if not {rel["column_a"], rel["column_b"]} <= set(continuous):
            continue
        recs.append(_rec("scatter", "relationship", 62 + 30 * r, f"{rel['column_a']} vs {rel['column_b']}",
                         f"These two measures are {'strongly' if r >= 0.7 else 'moderately'} correlated (r = {rel['correlation']:.2f}), so a scatter plot with a trend line shows the relationship and any outliers.",
                         x=rel["column_a"], y=rel["column_b"], agg="none"))
        used_pairs.add(pair)
        break

    # ---- group differences ------------------------------------------------------------
    groupers = _low_card_groupers(schema, 8)
    if groupers and continuous:
        spread = _numeric_frame(sample, continuous, profiles)
        variance_rank = sorted(continuous, key=lambda c: -float(spread[c].std() / abs(spread[c].mean()) if spread[c].mean() else 0))[:6]
        best_pair = None
        for measure in variance_rank:
            values = spread[measure]
            for grouper in groupers[:6]:
                frame = pd.DataFrame({"v": values, "g": sample[grouper].astype(str)}).dropna()
                if len(frame) < 20 or frame["g"].nunique() < 2:
                    continue
                eta2 = _eta_squared(frame["v"].to_numpy(), frame["g"].to_numpy())
                if eta2 < 0.04:
                    continue
                p = _kruskal_p(frame["v"].to_numpy(), frame["g"].to_numpy())
                if p < 0.01 and (best_pair is None or eta2 > best_pair[2]):
                    best_pair = (measure, grouper, eta2, p)
        if best_pair:
            measure, grouper, eta2, p = best_pair
            groups = int(sample[grouper].nunique())
            recs.append(_rec("distribution" if groups <= 4 else "box", "distribution", 78 + min(eta2, 0.5) * 30,
                             f"{measure} by {grouper}",
                             f"{measure} differs significantly across {grouper} groups (Kruskal-Wallis p = {p:.2g}); {grouper} explains {eta2:.0%} of its variance.",
                             x=measure if groups <= 4 else grouper, y=None if groups <= 4 else measure, group=grouper if groups <= 4 else None, agg="none"))

    # ---- correlation structure --------------------------------------------------------
    strong_pairs = [r for r in relationships if abs(r["correlation"]) >= 0.3]
    if len(continuous) >= 3 and len(strong_pairs) >= 2:
        recs.append(_rec("heatmap", "relationship", 74, "Correlation matrix",
                         f"{len(continuous)} numeric columns with {len(strong_pairs)} notable relationships - the matrix shows every pairwise correlation at once, clustered so related columns sit together.",
                         agg="none"))
    if 3 <= len(continuous) and len(strong_pairs) >= 2:
        recs.append(_rec("pair", "relationship", 61, "Pair plot of the most related columns",
                         "A scatter matrix of the columns that relate most strongly to the others shows which relationships are linear, curved or driven by outliers.",
                         agg="none"))

    # ---- distributions ----------------------------------------------------------------
    if skewed:
        column = skewed[0]
        skew = float(pd.to_numeric(df[column], errors="coerce").skew())
        recs.append(_rec("histogram", "distribution", 70 + min(15, abs(skew) * 3), f"Distribution of {column}",
                         f"{column} is strongly skewed (skewness {skew:.1f}); the histogram shows the long tail that pulls its mean away from its median.",
                         x=column, agg="none"))
    elif continuous:
        column = continuous[0]
        recs.append(_rec("histogram", "distribution", 50, f"Distribution of {column}",
                         f"The spread and shape of {column}.", x=column, agg="none"))

    # ---- data quality -----------------------------------------------------------------
    worst = max(missing_by_column.values(), default=0.0)
    if worst >= 5:
        columns_missing = sum(1 for v in missing_by_column.values() if v > 0)
        recs.append(_rec("missing", "quality", 76 + min(15, worst / 4), "Missing values by column",
                         f"{columns_missing} column(s) have missing values, up to {worst:.0f}% - worth understanding before trusting any statistic that uses them.",
                         agg="none"))

    # ---- categorical breakdowns -------------------------------------------------------
    cat_columns = [c for c in schema["groupers"] if profiles[c]["role"] in {"categorical", "ordinal", "binary", "geo_country", "geo_state"}]
    if cat_columns:
        category = cat_columns[0]
        unique = profiles[category]["unique_count"]
        shares = df[category].value_counts(normalize=True)
        top3 = float(shares.iloc[:3].sum())
        if 8 <= unique <= 60 and continuous:
            measure = continuous[0]
            additive = bool(_ADDITIVE_HINT.search(measure)) and float(pd.to_numeric(df[measure], errors="coerce").min() or 0) >= 0
            recs.append(_rec("treemap", "category", 64, f"{measure} across {category}",
                             f"'{category}' has {unique} categories; a treemap shows how the rows and {measure} are distributed across them at a glance.",
                             x=category, y=measure, agg="sum" if additive else "mean"))
        if unique <= 6:
            recs.append(_rec("pie", "category", 46, f"Share of rows by {category}",
                             f"{category} has only {unique} categories, few enough for shares to be readable as slices.", x=category, agg="none"))
        else:
            recs.append(_rec("bar", "category", 58 + (10 if top3 > 0.6 else 0), f"Rows by {category}",
                             f"The three largest {category} categories hold {top3:.0%} of rows." if top3 > 0.5 else f"How rows are spread across the {unique} {category} categories.",
                             x=category, agg="count", sort="desc"))
    pair_candidates = [c for c in _low_card_groupers(schema, 10) if profiles[c]["role"] in {"categorical", "binary", "ordinal"}][:5]
    best_assoc = None
    for i, a in enumerate(pair_candidates):
        for b in pair_candidates[i + 1:]:
            try:
                v, p = _cramers_v(sample[a].astype(str), sample[b].astype(str))
            except Exception:
                continue
            if p < 0.01 and v >= 0.15 and (best_assoc is None or v > best_assoc[2]):
                best_assoc = (a, b, v, p)
    if best_assoc:
        a, b, v, p = best_assoc
        recs.append(_rec("stacked_bar", "category", 62 + v * 20, f"{a} broken down by {b}",
                         f"{a} and {b} are associated (Cramér's V = {v:.2f}, chi-square p = {p:.2g}), so the mix of {b} changes across {a}.",
                         x=a, group=b, agg="count"))

    # Fallback so a dataset with no recognizable structure still gets something honest.
    if not recs and all_measures:
        recs.append(_rec("histogram", "distribution", 40, f"Distribution of {all_measures[0]}", f"The spread of {all_measures[0]}.", x=all_measures[0], agg="none"))
    if not recs and cat_columns:
        recs.append(_rec("bar", "category", 40, f"Rows by {cat_columns[0]}", f"Row counts per {cat_columns[0]} category.", x=cat_columns[0], agg="count"))

    return _order(recs)


def _order(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Best first, but guarantee the second suggestion tells a different story from the first
    (the first two become the default chart cards), and drop exact duplicates."""
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for rec in sorted(recs, key=lambda r: -r["score"]):
        key = (rec["type"], rec.get("x"), rec.get("y"), rec.get("group"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    if len(unique) >= 2 and unique[1]["family"] == unique[0]["family"]:
        for i in range(2, len(unique)):
            if unique[i]["family"] != unique[0]["family"]:
                unique.insert(1, unique.pop(i))
                break
    return unique[:12]


def describe_structure(schema: dict[str, Any]) -> list[str]:
    """Plain-language insight lines about what schema detection found."""
    profiles = schema["profiles"]
    lines: list[str] = []
    time = schema.get("time")
    if time:
        span = f"{time['start'][:10]} to {time['end'][:10]}" if time["kind"] == "datetime" else f"{time['start'][:4]} to {time['end'][:4]}"
        panel = f" across {time['n_entities']} {time['entity_column']} values (panel data)" if time.get("is_panel") else ""
        lines.append(f"Time structure: '{time['time_column']}' spans {span} at {time['frequency']} resolution{panel}.")
    geo = schema.get("geo", {})
    if geo.get("country"):
        g = geo["country"]
        lines.append(f"Geography: '{g['column']}' is a country column ({g['match_rate']:.0%} of values recognised) - it can be drawn on a map and used to join other country data.")
    if geo.get("us_state"):
        g = geo["us_state"]
        lines.append(f"Geography: '{g['column']}' holds US states ({g['match_rate']:.0%} recognised).")
    if geo.get("lat_lon"):
        lines.append(f"Geography: '{geo['lat_lon']['lat']}' / '{geo['lat_lon']['lon']}' are coordinates.")
    identifiers = [c for c, p in profiles.items() if p["role"] == "identifier"]
    if identifiers:
        lines.append(f"Identifier column(s) {', '.join(repr(c) for c in identifiers[:4])} uniquely label rows, so they're left out of statistics and correlations.")
    stored = [c for c, p in profiles.items() if p.get("notes", {}).get("stored_as_text") and p["role"] == "numeric"]
    if stored:
        lines.append(f"{', '.join(repr(c) for c in stored[:4])} look numeric but are stored as text (currency symbols, thousands separators or percent signs); they're read as numbers for analysis.")
    text_cols = [c for c, p in profiles.items() if p["role"] == "text"]
    if text_cols:
        lines.append(f"Free-text column(s) {', '.join(repr(c) for c in text_cols[:3])} aren't charted; they'd need text analysis rather than statistics.")
    return lines
