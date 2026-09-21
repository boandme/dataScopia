from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
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


def _has_any(norm_text: str, phrases: tuple[str, ...]) -> bool:
    """Trigger check that avoids matching short words inside unrelated ones (e.g. 'range'
    inside 'orange', 'min' inside 'mining') by requiring single-word phrases to be a whole
    token, while multi-word phrases still match as a normalized substring. A phrase ending in
    '*' matches any token starting with that stem (e.g. 'correlat*' covers correlate/
    correlated/correlation/correlations without enumerating every inflection)."""
    tokens = norm_text.split()
    token_set = set(tokens)
    for phrase in phrases:
        if phrase.endswith("*"):
            stem = phrase[:-1]
            if any(tok.startswith(stem) for tok in tokens):
                return True
        elif " " in phrase:
            if phrase in norm_text:
                return True
        elif phrase in token_set:
            return True
    return False


def _column_match(col_tokens: list[str], text_tokens: set[str], norm_text: str) -> bool:
    """A column matches if every one of its words appears as a whole token in the text (word-
    boundary safe), OR - for multi-word names only - its exact phrase appears as a substring.
    A single-token name must NEVER fall back to raw substring matching, or short names like
    'age' would match inside unrelated words like 'percentage'."""
    token_match = all(tok in text_tokens or _singular(tok) in text_tokens for tok in col_tokens)
    phrase_match = len(col_tokens) > 1 and " ".join(col_tokens) in norm_text
    return token_match or phrase_match


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
        if _column_match(col_tokens, text_tokens, norm_text):
            scored.append((col, len(col_norm)))
    scored.sort(key=lambda item: -item[1])
    seen: set[str] = set()
    ordered: list[str] = []
    for col, _ in scored:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def _find_columns_by_position(text: str, columns: list[str]) -> list[str]:
    """Like _find_columns, but ordered by where each column is first mentioned in the text
    rather than by name specificity. Word order carries real meaning for phrasings like
    'correlation between X and Y' or 'ratio of X to Y', where the longer column name isn't
    necessarily the one meant first."""
    norm_text = _normalize(text)
    tokens = norm_text.split()
    text_tokens: set[str] = set()
    for tok in tokens:
        text_tokens.add(tok)
        text_tokens.add(_singular(tok))

    first_token_index: dict[str, int] = {}
    for i, tok in enumerate(tokens):
        for key in (tok, _singular(tok)):
            if key not in first_token_index:
                first_token_index[key] = i

    positions: list[tuple[int, str]] = []
    for col in columns:
        col_norm = _normalize(col)
        if not col_norm:
            continue
        col_tokens = col_norm.split()
        if not _column_match(col_tokens, text_tokens, norm_text):
            continue
        candidate_idxs = [first_token_index[t] for t in col_tokens if t in first_token_index]
        if not candidate_idxs:
            candidate_idxs = [
                first_token_index[_singular(t)] for t in col_tokens if _singular(t) in first_token_index
            ]
        if candidate_idxs:
            positions.append((min(candidate_idxs), col))
    positions.sort(key=lambda item: item[0])
    seen: set[str] = set()
    ordered: list[str] = []
    for _, col in positions:
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


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _categorical_cols(df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    numeric = set(_numeric_cols(df))
    return [c for c in df.columns if c not in numeric and c not in exclude]


def _categorical_like_cols(df: pd.DataFrame, exclude: tuple[str, ...] = (), max_unique: int = 20) -> list[str]:
    """Columns usable as a grouping/categorical variable for stats tests: true non-numeric
    columns, plus low-cardinality numeric columns (e.g. a 0/1 'Survived' flag), which are
    routinely used as group labels even though their pandas dtype is numeric."""
    result = []
    for c in df.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique(dropna=True) <= max_unique:
            result.append(c)
    return result


# ---------------------------------------------------------------------------
# Intent handlers. Each takes the raw question + dataframe and returns a
# QAResult if it can confidently answer it, or None to let the next intent
# in the dispatch order try. All are pure/read-only over `df`.
# ---------------------------------------------------------------------------

_ML_TRIGGERS = (
    "build a model", "train a model", "build a classifier", "train a classifier",
    "baseline model", "random forest", "regression model", "machine learning model",
    "build a random forest", "how well can we predict", "feature importance",
    "which features predict", "what predicts", "important features", "predict*",
)


def _intent_ml_baseline(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(norm, _ML_TRIGGERS):
        return None
    columns = _find_columns(question, list(df.columns))
    if not columns:
        return None
    target = columns[0]

    work = df.dropna(subset=[target]).copy()
    if len(work) < 20:
        return None

    is_classification = not pd.api.types.is_numeric_dtype(work[target]) or work[target].nunique(
        dropna=True
    ) <= max(10, int(len(work) * 0.05))

    feature_cols = [c for c in work.columns if c != target]
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(work[c])]
    categorical_features = [
        c for c in feature_cols if not pd.api.types.is_numeric_dtype(work[c]) and work[c].nunique(dropna=True) <= 15
    ]
    if not numeric_features and not categorical_features:
        return None

    x_parts = []
    if numeric_features:
        x_parts.append(work[numeric_features].apply(lambda s: s.fillna(s.median())))
    if categorical_features:
        x_parts.append(pd.get_dummies(work[categorical_features].astype(str)))
    if not x_parts:
        return None
    x = pd.concat(x_parts, axis=1)
    y = work[target]
    if len(x) < 20 or x.shape[1] == 0:
        return None

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.model_selection import train_test_split

    if is_classification:
        y = y.astype(str)
        value_counts = y.value_counts()
        if value_counts.shape[0] < 2 or value_counts.min() < 2:
            return None
        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
        except ValueError:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=12)
        model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))
        baseline = value_counts.max() / value_counts.sum()
        metric_desc = f"accuracy = {accuracy:.3f} (a naive majority-class guess would get {baseline:.3f})"
        task = "classification"
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=12)
        model.fit(x_train, y_train)
        r2 = r2_score(y_test, model.predict(x_test))
        metric_desc = f"R² = {r2:.3f}"
        task = "regression"

    importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
    top_features = importances.head(5)
    features_desc = "; ".join(f"{name} ({value:.3f})" for name, value in top_features.items())
    model_cls = "RandomForestClassifier" if is_classification else "RandomForestRegressor"

    return QAResult(
        answer=(
            f"Trained a baseline Random Forest {task} model predicting '{target}' from {x.shape[1]} feature(s) "
            f"({len(work):,} rows, 75/25 train/test split): {metric_desc}. "
            f"Top predictive features: {features_desc}."
        ),
        computed=True,
        method="ml_baseline",
        code=(
            f"from sklearn.ensemble import {model_cls}\n"
            f"from sklearn.model_selection import train_test_split\n"
            f"# X = numeric + one-hot encoded feature columns, y = df['{target}']\n"
            f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)\n"
            f"model = {model_cls}(n_estimators=100, random_state=42).fit(X_train, y_train)"
        ),
        value={"task": task, "metric": metric_desc, "top_features": {str(k): float(v) for k, v in top_features.items()}},
        columns_used=[target, *numeric_features, *categorical_features],
    )


def _intent_normality(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(norm, ("normally distributed", "normal distribution", "is normal", "shapiro", "follows a normal")):
        return None
    columns = _find_columns(question, _numeric_cols(df))
    if not columns:
        return None
    col = columns[0]
    series = df[col].dropna()
    if len(series) < 8:
        return None
    from scipy import stats

    sample = series if len(series) <= 5000 else series.sample(5000, random_state=42)
    statistic, p_value = stats.shapiro(sample)
    verdict = "does not appear to be normally distributed" if p_value < 0.05 else "is consistent with a normal distribution"
    note = " (tested on a random 5,000-row subsample)" if len(series) > 5000 else ""
    return QAResult(
        answer=(
            f"Shapiro-Wilk test on '{col}'{note}: W = {statistic:.4f}, p = {p_value:.4g}. "
            f"At the 0.05 significance level, this variable {verdict}."
        ),
        computed=True,
        method="normality_test",
        code=f"from scipy import stats\nstats.shapiro(df['{col}'].dropna())",
        value={"statistic": float(statistic), "p_value": float(p_value), "is_normal": bool(p_value >= 0.05)},
        columns_used=[col],
    )


def _intent_distribution_type(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(norm, ("what distribution", "which distribution", "distribution does", "type of distribution")):
        return None
    columns = _find_columns(question, _numeric_cols(df))
    if not columns:
        return None
    col = columns[0]
    series = df[col].dropna()
    if len(series) < 8:
        return None
    from scipy import stats

    sample = series if len(series) <= 5000 else series.sample(5000, random_state=42)
    skew_val = float(series.skew())
    kurt_val = float(series.kurtosis())

    # Distribution objects' .cdf is passed directly rather than a name+args tuple: scipy's
    # string-dispatch path for kstest has a compatibility bug on some builds (TypeError from
    # ndtr internals) that the direct-callable form avoids.
    candidates: dict[str, float] = {}
    mu, sigma = sample.mean(), sample.std()
    if sigma > 0:
        candidates["normal"] = float(stats.kstest(sample, stats.norm(loc=mu, scale=sigma).cdf).statistic)
    if (sample >= 0).all() and sample.mean() > 0:
        candidates["exponential"] = float(
            stats.kstest(sample, stats.expon(loc=sample.min(), scale=sample.mean()).cdf).statistic
        )
    lo, hi = sample.min(), sample.max()
    if hi > lo:
        candidates["uniform"] = float(stats.kstest(sample, stats.uniform(loc=lo, scale=hi - lo).cdf).statistic)
    if (sample > 0).all():
        log_sample = np.log(sample)
        if log_sample.std() > 0:
            candidates["log-normal"] = float(
                stats.kstest(sample, stats.lognorm(s=log_sample.std(), scale=np.exp(log_sample.mean())).cdf).statistic
            )
    if not candidates:
        return None

    best = min(candidates, key=lambda k: candidates[k])
    shape_desc = "right-skewed" if skew_val > 0.5 else ("left-skewed" if skew_val < -0.5 else "roughly symmetric")
    tail_desc = "heavier-tailed than normal" if kurt_val > 1 else ("lighter-tailed than normal" if kurt_val < -1 else "normal-tailed")
    return QAResult(
        answer=(
            f"'{col}' looks closest to a **{best}** distribution (lowest KS statistic among normal/exponential/uniform/"
            f"log-normal fits, tested on {'a subsample of ' if len(series) > 5000 else ''}{len(sample):,} rows). "
            f"It is {shape_desc} (skew = {skew_val:.2f}) and {tail_desc} (excess kurtosis = {kurt_val:.2f})."
        ),
        computed=True,
        method="distribution_fit",
        code=(
            f"from scipy import stats\n"
            f"sample = df['{col}'].dropna()\n"
            "# KS statistic computed against normal/exponential/uniform/log-normal fits; lowest wins"
        ),
        value={"best_fit": best, "skew": skew_val, "kurtosis": kurt_val, "ks_statistics": candidates},
        columns_used=[col],
    )


def _intent_kurtosis(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(norm, ("kurtosis",)):
        return None
    columns = _find_columns(question, _numeric_cols(df))
    if not columns:
        return None
    col = columns[0]
    series = df[col].dropna()
    if len(series) < 4:
        return None
    kurt = series.kurtosis()
    if pd.isna(kurt):
        return None
    kurt = float(kurt)
    if kurt > 0.5:
        desc = "leptokurtic (heavier tails / more outlier-prone than a normal distribution)"
    elif kurt < -0.5:
        desc = "platykurtic (lighter tails / flatter than a normal distribution)"
    else:
        desc = "close to mesokurtic (similar tail weight to a normal distribution)"
    return QAResult(
        answer=f"'{col}' has an excess kurtosis of {kurt:.2f}, which is {desc}.",
        computed=True,
        method="kurtosis",
        code=f"df['{col}'].kurtosis()",
        value=kurt,
        columns_used=[col],
    )


def _intent_group_comparison_test(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(
        norm,
        (
            "significant difference", "significantly different", "statistically significant",
            "t test", "ttest", "anova", "hypothesis test", "p value", "pvalue",
        ),
    ):
        return None
    numeric_cols = _numeric_cols(df)
    metric_matches = _find_columns(question, numeric_cols)
    if not metric_matches:
        return None
    # A low-cardinality numeric column mentioned alongside a continuous one (e.g. a 0/1
    # 'Survived' flag next to 'Fare') is almost always the grouping variable, not the metric
    # being compared - prefer the higher-cardinality match as the outcome.
    metric = next((c for c in metric_matches if df[c].nunique(dropna=True) > 10), metric_matches[0])

    categorical_cols = _categorical_like_cols(df, exclude=(metric,))
    group_matches = _find_columns(question, categorical_cols)
    group_col = group_matches[0] if group_matches else None
    if group_col is None:
        for c in categorical_cols:
            n_unique = df[c].nunique(dropna=True)
            if 2 <= n_unique <= 10:
                group_col = c
                break
    if group_col is None:
        return None

    work = df[[group_col, metric]].dropna()
    grouped = [g[metric].to_numpy() for _, g in work.groupby(group_col) if len(g) >= 2]
    group_labels = [name for name, g in work.groupby(group_col) if len(g) >= 2]
    if len(grouped) < 2:
        return None

    from scipy import stats

    if len(grouped) == 2:
        statistic, p_value = stats.ttest_ind(grouped[0], grouped[1], equal_var=False)
        test_name = "Welch's t-test"
        code = (
            f"from scipy import stats\n"
            f"groups = [g['{metric}'].values for _, g in df.groupby('{group_col}')]\n"
            "stats.ttest_ind(groups[0], groups[1], equal_var=False)"
        )
    else:
        statistic, p_value = stats.f_oneway(*grouped)
        test_name = "one-way ANOVA"
        code = (
            f"from scipy import stats\n"
            f"groups = [g['{metric}'].values for _, g in df.groupby('{group_col}')]\n"
            "stats.f_oneway(*groups)"
        )
    verdict = "a statistically significant difference" if p_value < 0.05 else "no statistically significant difference"
    means = work.groupby(group_col)[metric].mean()
    means_desc = "; ".join(f"{k}: {_fmt_num(v)}" for k, v in means.items())
    return QAResult(
        answer=(
            f"{test_name} on '{metric}' across '{group_col}' groups ({', '.join(str(l) for l in group_labels)}): "
            f"statistic = {statistic:.4f}, p = {p_value:.4g}. This indicates {verdict} at the 0.05 level. "
            f"Group means — {means_desc}."
        ),
        computed=True,
        method="group_comparison_test",
        code=code,
        value={
            "test": test_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "group_means": {str(k): float(v) for k, v in means.items()},
        },
        columns_used=[metric, group_col],
    )


def _intent_chi_square(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(norm, ("chi square", "chi2", "association between", "independent of", "associated with")):
        return None
    categorical_cols = _categorical_like_cols(df)
    matched = _find_columns_by_position(question, categorical_cols)
    if len(matched) < 2:
        return None
    col_a, col_b = matched[0], matched[1]
    contingency = pd.crosstab(df[col_a], df[col_b])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None
    from scipy import stats

    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    verdict = "a statistically significant association" if p_value < 0.05 else "no statistically significant association"
    return QAResult(
        answer=(
            f"Chi-square test of independence between '{col_a}' and '{col_b}': "
            f"chi2 = {chi2:.3f}, df = {dof}, p = {p_value:.4g}. This indicates {verdict} at the 0.05 level."
        ),
        computed=True,
        method="chi_square_test",
        code=f"from scipy import stats\nstats.chi2_contingency(pd.crosstab(df['{col_a}'], df['{col_b}']))",
        value={"chi2": float(chi2), "p_value": float(p_value), "dof": int(dof)},
        columns_used=[col_a, col_b],
    )


def _intent_covariance(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(norm, ("covariance", "co variance")):
        return None
    numeric_cols = _numeric_cols(df)
    matched = _find_columns_by_position(question, numeric_cols)
    if len(matched) < 2:
        return None
    col_a, col_b = matched[0], matched[1]
    cov = df[[col_a, col_b]].cov().iloc[0, 1]
    if pd.isna(cov):
        return None
    return QAResult(
        answer=f"The covariance between '{col_a}' and '{col_b}' is {_fmt_num(cov)}.",
        computed=True,
        method="covariance",
        code=f"df['{col_a}'].cov(df['{col_b}'])",
        value=float(cov),
        columns_used=[col_a, col_b],
    )


def _intent_outliers(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(norm, ("outlier", "outliers")):
        return None
    columns = _find_columns(question, _numeric_cols(df))
    if not columns:
        return None
    col = columns[0]
    series = df[col].dropna()
    if series.empty:
        return None
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    count = int(mask.sum())
    total = len(series)
    pct = count / total * 100 if total else 0.0
    return QAResult(
        answer=(
            f"'{col}' has {count:,} outliers ({_fmt_pct(pct)} of {total:,} values) using the IQR rule "
            f"(outside [{_fmt_num(lower)}, {_fmt_num(upper)}])."
        ),
        computed=True,
        method="outliers",
        code=f"q1, q3 = df['{col}'].quantile([0.25, 0.75]); iqr = q3 - q1\n((df['{col}'] < q1 - 1.5*iqr) | (df['{col}'] > q3 + 1.5*iqr)).sum()",
        value={"count": count, "percentage": round(pct, 2), "lower_bound": float(lower), "upper_bound": float(upper)},
        columns_used=[col],
    )


_PERCENTILE_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:th|st|nd|rd)?\s*percentile")
_QUARTILE_PHRASES: tuple[tuple[str, float], ...] = (
    ("first quartile", 0.25), ("q1", 0.25),
    ("third quartile", 0.75), ("q3", 0.75),
    ("second quartile", 0.5), ("q2", 0.5),
)


def _intent_percentile(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    quantile: float | None = None
    m = _PERCENTILE_RE.search(norm)
    if m:
        pct = _parse_number(m.group(1))
        if pct is not None and 0 <= pct <= 100:
            quantile = pct / 100
    else:
        for phrase, q in _QUARTILE_PHRASES:
            if _has_any(norm, (phrase,)):
                quantile = q
                break

    if quantile is None:
        if not _has_any(norm, ("quartile", "quartiles")):
            return None
        columns = _find_columns(question, _numeric_cols(df))
        if not columns:
            return None
        col = columns[0]
        series = df[col].dropna()
        if series.empty:
            return None
        q1, q2, q3 = series.quantile([0.25, 0.5, 0.75])
        return QAResult(
            answer=f"'{col}' quartiles: Q1 = {_fmt_num(q1)}, Q2 (median) = {_fmt_num(q2)}, Q3 = {_fmt_num(q3)}.",
            computed=True,
            method="percentile",
            code=f"df['{col}'].quantile([0.25, 0.5, 0.75])",
            value={"q1": float(q1), "q2": float(q2), "q3": float(q3)},
            columns_used=[col],
        )

    columns = _find_columns(question, _numeric_cols(df))
    if not columns:
        return None
    col = columns[0]
    series = df[col].dropna()
    if series.empty:
        return None
    value = series.quantile(quantile)
    return QAResult(
        answer=f"The {quantile * 100:g}th percentile of '{col}' is {_fmt_num(value)}.",
        computed=True,
        method="percentile",
        code=f"df['{col}'].quantile({quantile})",
        value=float(value),
        columns_used=[col],
    )


def _intent_iqr(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(norm, ("iqr", "interquartile range")):
        return None
    columns = _find_columns(question, _numeric_cols(df))
    if not columns:
        return None
    col = columns[0]
    series = df[col].dropna()
    if series.empty:
        return None
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return QAResult(
        answer=f"The interquartile range of '{col}' is {_fmt_num(iqr)} (Q1 = {_fmt_num(q1)}, Q3 = {_fmt_num(q3)}).",
        computed=True,
        method="iqr",
        code=f"df['{col}'].quantile(0.75) - df['{col}'].quantile(0.25)",
        value={"q1": float(q1), "q3": float(q3), "iqr": float(iqr)},
        columns_used=[col],
    )


def _intent_range(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(norm, ("range",)):
        return None
    columns = _find_columns(question, _numeric_cols(df))
    if not columns:
        return None
    col = columns[0]
    series = df[col].dropna()
    if series.empty:
        return None
    lo, hi = float(series.min()), float(series.max())
    return QAResult(
        answer=f"'{col}' ranges from {_fmt_num(lo)} to {_fmt_num(hi)} (range = {_fmt_num(hi - lo)}).",
        computed=True,
        method="range",
        code=f"df['{col}'].max() - df['{col}'].min()",
        value={"min": lo, "max": hi, "range": hi - lo},
        columns_used=[col],
    )


def _intent_mode(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    wants_low = _has_any(norm, ("least common", "least frequent", "least frequently", "rarest"))
    wants_high = _has_any(
        norm, ("most common", "most frequent", "most frequently", "mode of", "modal value", "typical value", "appears most")
    )
    if not (wants_low or wants_high):
        return None
    columns = _find_columns(question, list(df.columns))
    if not columns:
        return None
    col = columns[0]
    counts = df[col].value_counts(dropna=True)
    if counts.empty:
        return None
    if wants_low:
        value, n, label = counts.idxmin(), int(counts.min()), "least common"
    else:
        value, n, label = counts.idxmax(), int(counts.max()), "most common"
    pct = n / len(df) * 100 if len(df) else 0.0
    return QAResult(
        answer=f"The {label} value of '{col}' is '{value}', appearing {n:,} times ({_fmt_pct(pct)}).",
        computed=True,
        method="mode",
        code=f"df['{col}'].value_counts().{'idxmin' if wants_low else 'idxmax'}()",
        value={"value": str(value), "count": n},
        columns_used=[col],
    )


def _intent_duplicates(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(norm, ("duplicate*", "repeated rows")):
        return None
    dup_count = int(df.duplicated().sum())
    total = len(df)
    pct = dup_count / total * 100 if total else 0.0
    return QAResult(
        answer=f"There are {dup_count:,} fully duplicated rows ({_fmt_pct(pct)} of {total:,}).",
        computed=True,
        method="duplicates",
        code="df.duplicated().sum()",
        value={"count": dup_count, "percentage": round(pct, 2)},
    )


def _intent_overview(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(
        norm,
        (
            "describe this dataset", "describe the dataset", "summarize this dataset",
            "summarize the dataset", "give me an overview", "overview of this dataset",
            "overview of the dataset", "tell me about this dataset", "summary of this dataset",
            "what is this dataset about",
        ),
    ):
        return None
    rows, cols = df.shape
    numeric_cols = _numeric_cols(df)
    categorical_cols = _categorical_cols(df)
    missing_pct = df.isna().mean().mean() * 100 if rows else 0.0
    dup_count = int(df.duplicated().sum())
    parts = [
        f"{rows:,} rows and {cols} columns ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical/text).",
        f"{_fmt_pct(missing_pct)} of all cells are missing on average, and {dup_count:,} rows are fully duplicated.",
    ]
    if numeric_cols:
        skews = {c: df[c].dropna().skew() for c in numeric_cols if df[c].dropna().shape[0] > 2}
        skewed = [c for c, v in skews.items() if pd.notna(v) and abs(v) > 1]
        if skewed:
            parts.append(f"Highly skewed columns: {', '.join(skewed[:5])}.")
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)
            pairs = []
            for i, a in enumerate(numeric_cols):
                for b in numeric_cols[i + 1 :]:
                    v = corr.loc[a, b]
                    if pd.notna(v):
                        pairs.append((a, b, float(v)))
            if pairs:
                pairs.sort(key=lambda p: -abs(p[2]))
                a, b, v = pairs[0]
                parts.append(f"Strongest correlation: '{a}' & '{b}' (r = {v:.2f}).")
    return QAResult(
        answer=" ".join(parts),
        computed=True,
        method="overview",
        code="df.shape; df.isna().mean().mean(); df.duplicated().sum(); df.corr(numeric_only=True)",
        value={"rows": rows, "columns": cols, "missing_pct": round(missing_pct, 2), "duplicates": dup_count},
    )


def _intent_value_counts(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(
        norm,
        (
            "value counts", "breakdown of", "frequency of", "frequencies of",
            "distribution of categories", "category breakdown", "counts of each",
            "how are the values distributed",
        ),
    ):
        return None
    columns = _find_columns(question, list(df.columns))
    if not columns:
        return None
    col = columns[0]
    counts = df[col].value_counts(dropna=True)
    if counts.empty:
        return None
    total = len(df)
    top = counts.head(10)
    lines = "; ".join(f"{idx}: {int(v):,} ({v / total * 100:.1f}%)" for idx, v in top.items())
    more_note = f" (+{len(counts) - 10} more distinct values)" if len(counts) > 10 else ""
    return QAResult(
        answer=f"Value counts for '{col}': {lines}{more_note}.",
        computed=True,
        method="value_counts",
        code=f"df['{col}'].value_counts()",
        value={str(idx): int(v) for idx, v in top.items()},
        columns_used=[col],
    )


_COLUMN_RANK_STATS: dict[str, str] = {
    "variance": "var",
    "standard deviation": "std",
    "average": "mean",
    "mean": "mean",
    "skew": "skew",
    "skewness": "skew",
    "missing": "missing",
    "unique": "nunique",
}
_COLUMN_RANK_LABELS: dict[str, str] = {
    "var": "variance",
    "std": "standard deviation",
    "mean": "average",
    "skew": "skewness",
    "missing": "missing percentage",
    "nunique": "unique-value count",
}


def _intent_column_ranking(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(norm, ("column", "columns", "variable", "variables", "field", "fields")):
        return None
    if not (_has_any(norm, _SUPERLATIVE_HIGH) or _has_any(norm, _SUPERLATIVE_LOW)):
        return None
    stat_key = None
    for phrase, key in _COLUMN_RANK_STATS.items():
        if _has_any(norm, (phrase,)):
            stat_key = key
            break
    if stat_key is None:
        return None
    ascending = _has_any(norm, _SUPERLATIVE_LOW) and not _has_any(norm, _SUPERLATIVE_HIGH)

    numeric_cols = _numeric_cols(df)
    if stat_key == "missing":
        values = df.isna().mean() * 100
    elif stat_key == "nunique":
        values = df.nunique(dropna=True).astype(float)
    elif stat_key in ("var", "std", "mean", "skew"):
        if not numeric_cols:
            return None
        values = getattr(df[numeric_cols], stat_key)(numeric_only=True)
    else:
        return None

    values = values.dropna().sort_values(ascending=ascending)
    if values.empty:
        return None
    top_col = str(values.index[0])
    top_val = float(values.iloc[0])
    label = _COLUMN_RANK_LABELS[stat_key]
    direction = "lowest" if ascending else "highest"
    suffix = "%" if stat_key == "missing" else ""
    return QAResult(
        answer=f"'{top_col}' has the {direction} {label} ({_fmt_num(top_val)}{suffix}).",
        computed=True,
        method="column_ranking",
        code=(
            "df.isna().mean() * 100" if stat_key == "missing"
            else "df.nunique()" if stat_key == "nunique"
            else f"df.select_dtypes(include='number').{stat_key}()"
        ),
        value={"column": top_col, "value": top_val},
        columns_used=[top_col],
    )


def _intent_ratio(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(norm, ("ratio of", "ratio between")):
        return None
    numeric_cols = _numeric_cols(df)
    matched = _find_columns_by_position(question, numeric_cols)
    if len(matched) < 2:
        return None
    col_a, col_b = matched[0], matched[1]
    sum_a, sum_b = df[col_a].sum(), df[col_b].sum()
    if not sum_b:
        return None
    ratio = sum_a / sum_b
    return QAResult(
        answer=(
            f"The ratio of total '{col_a}' to total '{col_b}' is {ratio:.3f} "
            f"(sum {col_a} = {_fmt_num(float(sum_a))}, sum {col_b} = {_fmt_num(float(sum_b))})."
        ),
        computed=True,
        method="ratio",
        code=f"df['{col_a}'].sum() / df['{col_b}'].sum()",
        value=float(ratio),
        columns_used=[col_a, col_b],
    )


def _intent_trend(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(
        norm, ("trend of", "trend in", "over time", "over the years", "changed over", "change over time")
    ):
        return None
    time_col = None
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            time_col = c
            break
    if time_col is None:
        for c in df.columns:
            norm_c = _normalize(c)
            if norm_c in ("year", "yr", "date") or norm_c.endswith(" year") or norm_c.endswith(" date"):
                time_col = c
                break
    if time_col is None:
        return None
    numeric_cols = [c for c in _numeric_cols(df) if c != time_col]
    matched = _find_columns(question, numeric_cols)
    if not matched:
        return None
    metric = matched[0]
    work = df[[time_col, metric]].dropna().copy()
    if pd.api.types.is_datetime64_any_dtype(work[time_col]):
        work["_x"] = work[time_col].dt.year.astype(float)
    else:
        work["_x"] = pd.to_numeric(work[time_col], errors="coerce")
        work = work.dropna(subset=["_x"])
    if len(work) < 3 or work["_x"].nunique() < 2:
        return None
    slope, intercept = np.polyfit(work["_x"], work[metric].astype(float), 1)
    slope, intercept = float(slope), float(intercept)
    direction = "an increasing" if slope > 0 else ("a decreasing" if slope < 0 else "a flat")
    return QAResult(
        answer=(
            f"'{metric}' shows {direction} trend over '{time_col}': about {_fmt_num(abs(slope))} per unit of "
            f"{time_col} (linear fit slope = {slope:.4f})."
        ),
        computed=True,
        method="trend",
        code=f"import numpy as np\nnp.polyfit(df['{time_col}'], df['{metric}'], 1)  # (slope, intercept)",
        value={"slope": slope, "intercept": intercept},
        columns_used=[time_col, metric],
    )


def _intent_missing(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    if not _has_any(norm, ("missing", "null value", "null values", "nan values", "nulls")):
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
    if not _has_any(
        norm, ("unique", "distinct", "how many different", "how many kinds", "how many categories", "how many types")
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
    if not _has_any(norm, ("skew", "skewed", "skewness")):
        return None
    numeric_cols = _numeric_cols(df)
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
    if not _has_any(norm, ("correlat*", "relationship", "related to", "relate to")):
        return None
    numeric_cols = _numeric_cols(df)
    if len(numeric_cols) < 2:
        return None
    matched = _find_columns_by_position(question, numeric_cols)
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
    (r"(?:not equal to|different from|differs from|!=)", "!="),
    (r"(?:over|above|more than|greater than|exceed(?:ing|s)?)", ">"),
    (r"(?:under|below|less than|fewer than)", "<"),
    (r">", ">"),
    (r"<", "<"),
    (r"(?:equal(?:s|to)?|exactly)", "=="),
]

_NUMBER_PATTERN = r"(-?\d[\d,]*\.?\d*)"
_BETWEEN_RE = re.compile(r"between\s+" + _NUMBER_PATTERN + r"\s+and\s+" + _NUMBER_PATTERN)


def _apply_comparator(series: pd.Series, symbol: str, threshold: float) -> pd.Series:
    if symbol == ">":
        return series > threshold
    if symbol == ">=":
        return series >= threshold
    if symbol == "<":
        return series < threshold
    if symbol == "<=":
        return series <= threshold
    if symbol == "!=":
        return series != threshold
    return series == threshold


def _find_comparator(text: str) -> tuple[str, re.Match[str], float] | None:
    for pattern, symbol in _COMPARATORS:
        m = re.search(pattern + r"\s*" + _NUMBER_PATTERN, text)
        if m:
            threshold = _parse_number(m.group(1))
            if threshold is not None:
                return symbol, m, threshold
    return None


def _intent_condition(question: str, df: pd.DataFrame) -> QAResult | None:
    lower = question.lower()
    numeric_cols = _numeric_cols(df)
    # Position-based, not specificity-based: "age over 30 and salary above 60000" needs 'age'
    # picked for the first clause because it's mentioned first, not because 'salary' is a longer
    # column name.
    columns = _find_columns_by_position(question, numeric_cols)

    between_match = _BETWEEN_RE.search(lower)
    if between_match and columns:
        lo = _parse_number(between_match.group(1))
        hi = _parse_number(between_match.group(2))
        if lo is not None and hi is not None:
            if lo > hi:
                lo, hi = hi, lo
            col = columns[0]
            series = df[col].dropna()
            if not series.empty:
                mask = (series >= lo) & (series <= hi)
                count = int(mask.sum())
                total = len(series)
                pct = count / total * 100 if total else 0.0
                wants_pct = _has_any(_normalize(question), ("percent*", "proportion", "fraction"))
                if wants_pct:
                    answer = f"{_fmt_pct(pct)} of rows have '{col}' between {_fmt_num(lo)} and {_fmt_num(hi)} ({count:,} of {total:,})."
                else:
                    answer = f"{count:,} rows have '{col}' between {_fmt_num(lo)} and {_fmt_num(hi)} ({_fmt_pct(pct)} of {total:,})."
                return QAResult(
                    answer=answer,
                    computed=True,
                    method="condition_filter",
                    code=f"df['{col}'].between({lo}, {hi}).sum()",
                    value={"column": col, "low": lo, "high": hi, "count": count, "percentage": round(pct, 2)},
                    columns_used=[col],
                )

    first = _find_comparator(lower)
    if first is None or not columns:
        return None
    comparator_symbol, match, threshold = first
    col = columns[0]

    # Look for a second "and <column> <comparator> <number>" clause after the first match, so
    # "age over 30 and fare above 50" filters on both conditions together rather than just the
    # first one found.
    and_idx = lower.find(" and ", match.end())
    if and_idx != -1:
        remainder_text = question[and_idx + 5 :]
        remainder_lower = lower[and_idx + 5 :]
        second_candidates = [c for c in numeric_cols if c != col]
        second_cols = _find_columns(remainder_text, second_candidates)
        second = _find_comparator(remainder_lower)
        if second_cols and second is not None:
            symbol2, _, threshold2 = second
            col2 = second_cols[0]
            combined = df[[col, col2]].dropna()
            if not combined.empty:
                mask1 = _apply_comparator(combined[col], comparator_symbol, threshold)
                mask2 = _apply_comparator(combined[col2], symbol2, threshold2)
                mask = mask1 & mask2
                count = int(mask.sum())
                total = len(combined)
                pct = (count / total * 100) if total else 0.0
                answer = (
                    f"{count:,} rows have '{col}' {comparator_symbol} {_fmt_num(threshold)} AND "
                    f"'{col2}' {symbol2} {_fmt_num(threshold2)} ({_fmt_pct(pct)} of {total:,})."
                )
                return QAResult(
                    answer=answer,
                    computed=True,
                    method="condition_filter",
                    code=f"((df['{col}'] {comparator_symbol} {threshold}) & (df['{col2}'] {symbol2} {threshold2})).sum()",
                    value={
                        "conditions": [
                            {"column": col, "comparator": comparator_symbol, "threshold": threshold},
                            {"column": col2, "comparator": symbol2, "threshold": threshold2},
                        ],
                        "count": count,
                        "percentage": round(pct, 2),
                    },
                    columns_used=[col, col2],
                )

    series = df[col].dropna()
    if series.empty:
        return None
    mask = _apply_comparator(series, comparator_symbol, threshold)
    count = int(mask.sum())
    total = len(series)
    pct = (count / total * 100) if total else 0.0
    wants_pct = _has_any(_normalize(question), ("percent*", "proportion", "fraction", "share of"))
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


_SUPERLATIVE_HIGH = ("highest", "largest", "most", "top", "greatest", "biggest")
_SUPERLATIVE_LOW = ("lowest", "smallest", "least", "bottom", "fewest")


def _intent_topn(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    ascending: bool | None = None
    if _has_any(norm, _SUPERLATIVE_LOW):
        ascending = True
    if _has_any(norm, _SUPERLATIVE_HIGH):
        ascending = False if ascending is None else ascending
    if ascending is None:
        return None
    numeric_cols = _numeric_cols(df)
    metric_matches = _find_columns(question, numeric_cols)
    if not metric_matches:
        return None
    metric = metric_matches[0]
    n_match = re.search(r"top\s*(\d+)", norm)
    n = int(n_match.group(1)) if n_match else 1
    n = max(1, min(n, 20))

    categorical_cols = _categorical_cols(df)
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
    "stdev": "std",
    "std": "std",
    "spread": "std",
    "dispersion": "std",
    "variance": "var",
    "count": "count",
}


def _intent_aggregation(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    func = None
    for keyword, agg in _AGG_KEYWORDS.items():
        if _has_any(norm, (keyword,)):
            func = agg
            break
    if func is None:
        return None
    numeric_cols = _numeric_cols(df)
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
    display_name = {"std": "standard deviation", "var": "variance"}.get(func, func)
    return QAResult(
        answer=f"The {display_name} of '{metric}' is {_fmt_num(float(value))}.",
        computed=True,
        method="aggregation",
        code=f"df['{metric}'].{func}()",
        value=float(value),
        columns_used=[metric],
    )


def _intent_shape(question: str, df: pd.DataFrame) -> QAResult | None:
    norm = _normalize(question)
    tokens = set(norm.split())
    context = _has_any(norm, ("how many", "number of", "shape of", "size of the dataset", "dimensions of"))
    if not context:
        return None
    wants_rows = bool(tokens & {"row", "rows", "observation", "observations", "record", "records", "entry", "entries"})
    wants_cols = bool(tokens & {"column", "columns", "variable", "variables", "field", "fields"})
    wants_shape = _has_any(norm, ("shape of the data", "size of the dataset", "dimensions of")) and not (wants_rows or wants_cols)
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
    if not _has_any(norm, ("increase", "decrease", "change", "grew", "declined", "growth", "drop", "fell", "rose")):
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

    numeric_cols = _numeric_cols(df)
    metric_matches = [c for c in _find_columns(question, numeric_cols) if c != time_col]
    if not metric_matches:
        return None
    metric = metric_matches[0]

    categorical_cols = [c for c in _categorical_cols(df) if c != time_col]
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
    wants_decrease = _has_any(norm, ("decrease", "declined", "drop", "fell"))
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
    _intent_overview,
    _intent_ml_baseline,
    _intent_normality,
    _intent_distribution_type,
    _intent_kurtosis,
    _intent_group_comparison_test,
    _intent_chi_square,
    _intent_covariance,
    _intent_ratio,
    _intent_outliers,
    _intent_percentile,
    _intent_iqr,
    _intent_range,
    _intent_duplicates,
    _intent_value_counts,
    _intent_mode,
    _intent_missing,
    _intent_unique,
    _intent_skew,
    _intent_correlation,
    _intent_time_change,
    _intent_trend,
    _intent_condition,
    _intent_column_ranking,
    _intent_topn,
    _intent_aggregation,
    _intent_shape,
]


def _fallback(df: pd.DataFrame, snapshot: dict[str, Any] | None) -> QAResult:
    rows, cols = df.shape
    hint = (
        "Try things like: 'average <column>?', '<column> over <number>?', 'correlation between X and Y', "
        "'is X normally distributed?', 'significant difference in X between Y groups?', "
        "'build a model predicting <column>', 'outliers in <column>', 'describe this dataset', "
        "'value counts of <column>', 'ratio of X to Y', 'trend of <column> over time', or "
        "'which column has the highest variance?'."
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

    Tries a fixed set of deterministic intent handlers, ordered from most to least specific:
    ML baselines, distribution/normality checks, statistical hypothesis tests (t-test/ANOVA/
    chi-square), descriptive statistics (percentile/IQR/range/mode/outliers/covariance/kurtosis),
    then the original core set (missing values, uniqueness, skew, correlation, time-based change,
    threshold filters, ranking, aggregation, shape). Each performs an actual computation rather
    than returning generic advice; falls back to a grounded snapshot summary, clearly marked as
    not computed, only when nothing matches.
    """
    for intent in _INTENTS:
        try:
            result = intent(question, df)
        except Exception:
            result = None
        if result is not None:
            return result
    return _fallback(df, snapshot)
