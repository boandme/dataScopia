from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Statistical narrative generators. Each produces genuine, dataset-specific prose grounded in
# a real computation against the actual dataframe - not templated filler - so the report can
# speak to the significance of what each chart shows, not just restate its axes.
# ---------------------------------------------------------------------------


def _corr_strength_label(r: float) -> str:
    magnitude = abs(r)
    if magnitude >= 0.7:
        return "strong"
    if magnitude >= 0.4:
        return "moderate"
    if magnitude >= 0.2:
        return "weak"
    return "very weak"


def describe_relationship(df: pd.DataFrame, x: str, y: str) -> str:
    """Real Pearson correlation + significance test for a scatter/line/area chart."""
    paired = df[[x, y]].dropna()
    n = len(paired)
    if n < 3 or paired[x].nunique() < 2 or paired[y].nunique() < 2:
        return (
            f"There is not enough variation in the sample to compute a meaningful correlation "
            f"between {x} and {y} ({n} paired observations)."
        )
    r, p_value = stats.pearsonr(paired[x], paired[y])
    strength = _corr_strength_label(r)
    direction = "positive" if r >= 0 else "negative"
    r_squared = r**2
    significant = p_value < 0.05
    sig_phrase = (
        f"statistically significant at the 0.05 level (p = {p_value:.4g})"
        if significant
        else f"not statistically significant at the 0.05 level (p = {p_value:.4g})"
    )
    direction_word = "increase" if r >= 0 else "decrease"
    parts = [
        f"Across {n:,} paired observations, the Pearson correlation between {x} and {y} is "
        f"r = {r:.3f}, indicating a {strength} {direction} relationship. This result is {sig_phrase}."
    ]
    if significant:
        variance_note = (
            ", though the majority of variance remains unexplained by this single variable and other "
            "factors are likely contributing"
            if r_squared < 0.5
            else ", meaning this single variable accounts for a substantial share of the variation observed"
        )
        parts.append(
            f"As {x} increases, {y} tends to {direction_word}. The coefficient of determination "
            f"(R-squared = {r_squared:.3f}) indicates that approximately {r_squared * 100:.1f}% of the "
            f"variance in {y} can be explained by a linear relationship with {x}{variance_note}."
        )
    else:
        parts.append(
            "This suggests any apparent trend in the chart is unlikely to reflect a genuine linear "
            "relationship between these variables and may instead be due to sampling noise."
        )

    # Spearman rank correlation: a meaningful gap from the Pearson value hints that the true
    # relationship is monotonic but non-linear (e.g. curved or accelerating) rather than a
    # straight line, which a scatter plot alone can be hard to judge by eye.
    try:
        rho, _rho_p = stats.spearmanr(paired[x], paired[y])
        rho = float(rho)
        gap = abs(rho - r)
        if gap > 0.15:
            parts.append(
                f"Notably, the Spearman rank correlation (rho = {rho:.3f}) diverges from the Pearson value "
                f"by more than {gap:.2f}, suggesting the true relationship between {x} and {y} may be "
                f"monotonic but non-linear rather than a straight-line trend; a transformation (such as a "
                f"log scale on one axis) may reveal a cleaner fit than the linear one tested here."
            )
        elif significant:
            parts.append(
                f"The Spearman rank correlation (rho = {rho:.3f}) closely matches the Pearson result, "
                f"reinforcing that a straight line is a reasonable description of this relationship rather "
                f"than a more complex curve."
            )
    except Exception:
        pass

    # Outlier-robustness check: does the correlation survive trimming the most extreme 5% of
    # values on either axis? A correlation that collapses without those points is being driven
    # by a handful of extreme observations rather than a pattern spanning the sample.
    try:
        if n >= 20:
            x_lo, x_hi = paired[x].quantile([0.025, 0.975])
            y_lo, y_hi = paired[y].quantile([0.025, 0.975])
            trimmed = paired[(paired[x].between(x_lo, x_hi)) & (paired[y].between(y_lo, y_hi))]
            if len(trimmed) >= 10 and trimmed[x].nunique() > 1 and trimmed[y].nunique() > 1:
                r_trimmed, _ = stats.pearsonr(trimmed[x], trimmed[y])
                r_trimmed = float(r_trimmed)
                if abs(r_trimmed - r) > 0.15:
                    parts.append(
                        f"When the most extreme 5% of values on either axis are excluded, the correlation "
                        f"shifts to r = {r_trimmed:.3f} ({len(trimmed):,} of {n:,} points retained), indicating "
                        f"that a small number of outlying observations meaningfully influence the relationship "
                        f"reported above and it should be interpreted with that in mind."
                    )
                else:
                    parts.append(
                        f"Excluding the most extreme 5% of values on either axis leaves the correlation "
                        f"essentially unchanged (r = {r_trimmed:.3f}), indicating this relationship reflects a "
                        f"broad pattern across the sample rather than the influence of a few outliers."
                    )
    except Exception:
        pass

    parts.append(
        "As with any correlation, this measures linear association only and does not by itself "
        "establish that one variable causes changes in the other; a third factor could influence both."
    )
    return " ".join(parts)


def describe_distribution(df: pd.DataFrame, column: str) -> str:
    series = df[column].dropna()
    n = len(series)
    if n < 3:
        return f"There are too few non-missing values in {column} ({n}) to characterize its distribution."
    mean = float(series.mean())
    median = float(series.median())
    std = float(series.std())
    minimum = float(series.min())
    maximum = float(series.max())
    skew = series.skew()
    skew = 0.0 if pd.isna(skew) else float(skew)
    kurt = series.kurtosis()
    kurt = 0.0 if pd.isna(kurt) else float(kurt)

    if abs(skew) < 0.5:
        shape = "approximately symmetric"
        shape_detail = "with no strong skew in either direction"
    elif skew > 0:
        shape = "right-skewed (positively skewed)"
        shape_detail = (
            f"a relatively small number of unusually high values are pulling the mean "
            f"({mean:,.2f}) above the median ({median:,.2f})"
        )
    else:
        shape = "left-skewed (negatively skewed)"
        shape_detail = (
            f"a relatively small number of unusually low values are pulling the mean "
            f"({mean:,.2f}) below the median ({median:,.2f})"
        )

    if kurt > 1:
        tail_note = (
            f" The distribution is also leptokurtic (excess kurtosis = {kurt:.2f}), meaning it has heavier "
            f"tails and a sharper peak than a normal distribution, so extreme values are more common than a "
            f"bell curve would predict."
        )
    elif kurt < -1:
        tail_note = (
            f" The distribution is platykurtic (excess kurtosis = {kurt:.2f}), meaning it is flatter than a "
            f"normal distribution, with fewer extreme outliers and values spread more uniformly across the range."
        )
    else:
        tail_note = f" Its kurtosis ({kurt:.2f}) is close to that of a normal distribution, without unusually heavy or light tails."

    spread_note = ""
    if mean not in (0, None) and not pd.isna(mean) and mean != 0:
        cv = abs(std / mean)
        if cv < 5:
            level = "high relative variability" if cv > 1 else "moderate relative variability" if cv > 0.3 else "relatively low variability"
            spread_note = f" The coefficient of variation is {cv:.2f}, indicating {level} around the mean."

    outlier_note = ""
    try:
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_outliers = int(((series < lower) | (series > upper)).sum())
            if n_outliers:
                outlier_note = (
                    f" Using the IQR rule, {n_outliers:,} value(s) ({n_outliers / n * 100:.1f}% of observations) "
                    f"fall outside [{lower:,.2f}, {upper:,.2f}] and would typically be flagged as outliers worth "
                    f"reviewing before this field is used in downstream modeling."
                )
    except Exception:
        pass

    return (
        f"Across {n:,} observations, {column} ranges from {minimum:,.2f} to {maximum:,.2f}, with a mean of "
        f"{mean:,.2f} and a median of {median:,.2f} (standard deviation {std:,.2f}). The distribution is "
        f"{shape} (skewness = {skew:.2f}); {shape_detail}.{tail_note}{spread_note}{outlier_note}"
    )


def describe_categorical(df: pd.DataFrame, column: str) -> str:
    counts = df[column].dropna().value_counts()
    n = int(counts.sum())
    if n == 0 or counts.empty:
        return f"No non-missing values are available for {column}."
    top = counts.head(3)
    n_categories = counts.shape[0]
    lead_label, lead_n = str(counts.index[0]), int(counts.iloc[0])
    lead_pct = lead_n / n * 100
    concentration = float(top.sum()) / n * 100
    parts = [
        f"{column} contains {n_categories:,} distinct categories across {n:,} observations. "
        f"The most common is '{lead_label}', accounting for {lead_pct:.1f}% of records ({lead_n:,} observations)."
    ]
    if n_categories > 1:
        top_list = "; ".join(f"'{idx}' ({v / n * 100:.1f}%)" for idx, v in top.items())
        parts.append(f"The top {len(top)} categories together ({top_list}) account for {concentration:.1f}% of all observations.")

    if n_categories > 1:
        probs = (counts / n).to_numpy()
        entropy = float(-(probs * np.log2(probs)).sum())
        max_entropy = float(np.log2(n_categories))
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        if normalized > 0.85:
            balance_note = "close to perfectly even across all categories"
        elif normalized > 0.6:
            balance_note = "fairly well balanced, with some categories more common than others"
        elif normalized > 0.3:
            balance_note = "moderately imbalanced, with a handful of categories dominating"
        else:
            balance_note = "highly imbalanced, dominated by very few categories"
        parts.append(
            f"The normalized diversity of this field (Shannon entropy) is {normalized:.2f} on a 0-1 scale where "
            f"1.0 is perfectly even, indicating a distribution that is {balance_note}."
        )

    if concentration > 80:
        parts.append(
            "This is a highly concentrated distribution, with most records falling into just a few categories; "
            "any chart or downstream statistic based on this field will be dominated by the leading category."
        )
    elif concentration < 40 and n_categories > 5:
        parts.append("Observations are relatively evenly spread across many categories, without strong dominance by any single group.")
    return " ".join(parts)


_AGG_LABELS = {"mean": "average", "median": "median", "sum": "total", "max": "maximum", "min": "minimum", "count": "count"}


def describe_group_comparison(df: pd.DataFrame, x: str, y: str, agg: str) -> str:
    work = df[[x, y]].dropna()
    if work.empty:
        return f"No paired observations are available to compare {y} across {x}."
    agg_func = agg if agg in _AGG_LABELS else "mean"
    grouped = work.groupby(x)[y].agg(agg_func).sort_values(ascending=False)
    if grouped.empty:
        return f"No groups are available to compare {y} across {x}."
    n_groups = grouped.shape[0]
    top_group, top_val = str(grouped.index[0]), float(grouped.iloc[0])
    bottom_group, bottom_val = str(grouped.index[-1]), float(grouped.iloc[-1])
    spread = top_val - bottom_val
    agg_label = _AGG_LABELS.get(agg_func, agg_func)
    sentence = (
        f"Comparing the {agg_label} {y} across {n_groups:,} categories of {x}, '{top_group}' has the "
        f"highest value at {top_val:,.2f}, while '{bottom_group}' has the lowest at {bottom_val:,.2f} "
        f"- a difference of {spread:,.2f}"
    )
    rel_spread = None
    if bottom_val not in (0, None) and not pd.isna(bottom_val) and bottom_val != 0:
        rel_spread = spread / abs(bottom_val) * 100
        sentence += f" ({rel_spread:.1f}% relative difference)."
    else:
        sentence += "."
    parts = [sentence]
    if n_groups > 2:
        ranking = ", ".join(f"{i + 1}. {idx} ({val:,.2f})" for i, (idx, val) in enumerate(grouped.head(5).items()))
        trailer = ", ..." if n_groups > 5 else ""
        parts.append(f"Ranked from highest to lowest: {ranking}{trailer}.")
    if rel_spread is not None and rel_spread > 50:
        parts.append(f"This substantial spread suggests {x} is meaningfully associated with variation in {y}.")
    elif rel_spread is not None and rel_spread < 15:
        parts.append(f"This relatively narrow spread suggests {x} alone does not strongly differentiate {y}.")

    try:
        groups_data = [g[y].to_numpy() for _, g in work.groupby(x) if len(g) >= 2]
        effect_note = None
        if len(groups_data) == 2:
            stat_val, p_value = stats.ttest_ind(groups_data[0], groups_data[1], equal_var=False)
            test_name = "Welch's t-test"
            n1, n2 = len(groups_data[0]), len(groups_data[1])
            pooled_std = (
                ((n1 - 1) * groups_data[0].std(ddof=1) ** 2 + (n2 - 1) * groups_data[1].std(ddof=1) ** 2) / (n1 + n2 - 2)
            ) ** 0.5
            if pooled_std > 0:
                cohens_d = float((groups_data[0].mean() - groups_data[1].mean()) / pooled_std)
                d_abs = abs(cohens_d)
                d_label = "small" if d_abs < 0.5 else "medium" if d_abs < 0.8 else "large"
                effect_note = f"Cohen's d = {cohens_d:.2f} ({d_label} effect size)"
        elif len(groups_data) > 2:
            stat_val, p_value = stats.f_oneway(*groups_data)
            test_name = "one-way ANOVA"
            all_values = np.concatenate(groups_data)
            grand_mean = float(all_values.mean())
            ss_between = sum(len(g) * (float(g.mean()) - grand_mean) ** 2 for g in groups_data)
            ss_total = float(((all_values - grand_mean) ** 2).sum())
            if ss_total > 0:
                eta_sq = ss_between / ss_total
                e_label = "small" if eta_sq < 0.06 else "medium" if eta_sq < 0.14 else "large"
                effect_note = f"eta-squared = {eta_sq:.3f} ({e_label} effect size)"
        else:
            test_name = None
            p_value = None
        if test_name is not None and p_value is not None and not pd.isna(p_value):
            sig = "a statistically significant difference" if p_value < 0.05 else "no statistically significant difference"
            effect_clause = f", with {effect_note}" if effect_note else ""
            parts.append(
                f"A {test_name} across these groups yields p = {p_value:.4g}, indicating {sig} in {y} at the "
                f"0.05 level{effect_clause}."
            )
    except Exception:
        pass
    return " ".join(parts)


def describe_heatmap(df: pd.DataFrame, numeric_cols: list[str]) -> str:
    if len(numeric_cols) < 2:
        return "Not enough numeric columns are available to compute a correlation matrix."
    corr = df[numeric_cols].corr(numeric_only=True)
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1 :]:
            v = corr.loc[a, b]
            if pd.notna(v):
                pairs.append((a, b, float(v)))
    if not pairs:
        return "No valid correlations could be computed among the numeric variables."
    pairs.sort(key=lambda p: -abs(p[2]))
    top = pairs[:3]
    lines = "; ".join(f"{a} & {b} (r = {v:.3f}, {_corr_strength_label(v)} {'positive' if v >= 0 else 'negative'})" for a, b, v in top)
    avg_abs = sum(abs(v) for _, _, v in pairs) / len(pairs)
    moderate_or_stronger = sum(1 for _, _, v in pairs if abs(v) >= 0.4)
    weakest = pairs[-1]
    return (
        f"This heatmap visualizes pairwise Pearson correlations across {len(numeric_cols)} numeric variables "
        f"({len(pairs)} unique pairs). The strongest relationships identified are: {lines}. "
        f"{moderate_or_stronger} of {len(pairs)} pairs ({moderate_or_stronger / len(pairs) * 100:.0f}%) show a "
        f"moderate-or-stronger relationship (|r| >= 0.4), with an average absolute correlation of {avg_abs:.3f} "
        f"across all pairs. The weakest relationship is between {weakest[0]} and {weakest[1]} (r = {weakest[2]:.3f}), "
        f"indicating those two variables are essentially independent of one another. The strongest pairs merit "
        f"closer investigation as potential drivers of variation within the dataset, while weaker correlations "
        f"elsewhere suggest little shared signal between those variable pairs."
    )


_ENGINE_NARRATED_TYPES = {"timeseries", "map", "treemap", "distribution", "target", "missing", "pair"}


def generate_chart_narrative(df: pd.DataFrame, chart: dict[str, Any]) -> str:
    ctype = chart.get("type", "histogram")
    x = chart.get("x")
    y = chart.get("y")
    agg = chart.get("agg", "none")
    numeric_cols = set(df.select_dtypes(include="number").columns)

    try:
        if ctype in _ENGINE_NARRATED_TYPES:
            # These chart types are drawn (and explained) by the chart engine; reuse its notes so the
            # report says exactly what the on-screen chart says, statistics included.
            from .charts import render_chart
            from .schema import build_schema

            meta = render_chart(df, chart, build_schema(df)).get("meta", {})
            notes = meta.get("error") and [meta["error"]] or meta.get("notes") or []
            if notes:
                return " ".join(str(n) for n in notes)
        if ctype == "heatmap":
            return describe_heatmap(df, list(df.select_dtypes(include="number").columns))
        if not x or x not in df.columns:
            return "This chart's configured field could not be found in the analyzed sample."
        if ctype in ("scatter", "line", "area") and agg == "none" and y and y in df.columns and x in numeric_cols and y in numeric_cols:
            return describe_relationship(df, x, y)
        if ctype == "histogram" and x in numeric_cols:
            return describe_distribution(df, x)
        if ctype == "pie" or (ctype in ("bar", "stacked_bar") and agg == "none"):
            return describe_categorical(df, x)
        if y and y in df.columns:
            return describe_group_comparison(df, x, y, agg if agg != "none" else "mean")
        if x in numeric_cols:
            return describe_distribution(df, x)
        return describe_categorical(df, x)
    except Exception as exc:  # pragma: no cover
        return f"An automated analysis could not be generated for this chart ({exc})."
