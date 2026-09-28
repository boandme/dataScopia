"""The statistics behind the PDF report, computed once as plain data.

The report keeps *results* (numbers, tables) separate from *interpretation* (discussion text), so
everything here returns structured facts and none of it makes a judgement beyond standard effect-size
labels. Conventions used throughout:

  * significance level alpha = 0.05, with Bonferroni adjustment inside each family of tests;
  * effect sizes reported next to every p-value (|r|, eta-squared, Cramer's V), because with many rows
    nearly any difference is "significant";
  * every test is exploratory - the pairs are chosen after looking at the data - and the report says so.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import schema as schema_mod

ALPHA = 0.05
MAX_CORR_COLUMNS = 12
NUMERIC_ROLES = {"numeric", "ordinal_numeric"}
GROUP_ROLES = {"categorical", "binary", "ordinal", "geo_country", "geo_state", "ordinal_numeric", "binary_numeric"}


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    series = df[column]
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return schema_mod.coerce_numeric(series).astype(float)


def strength_label(r: float) -> str:
    a = abs(r)
    return "negligible" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong" if a < 0.7 else "very strong"


def eta_label(e: float) -> str:
    return "negligible" if e < 0.01 else "small" if e < 0.06 else "medium" if e < 0.14 else "large"


def v_label(v: float) -> str:
    return "negligible" if v < 0.1 else "small" if v < 0.3 else "medium" if v < 0.5 else "large"


def fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def fmt_num(value: float | None, digits: int = 3) -> str:
    """Compact, readable number: thousands separators for big values, significant digits for small."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "-"
    a = abs(value)
    if a >= 1e9:
        return f"{value / 1e9:.2f}B"
    if a >= 1e6:
        return f"{value / 1e6:.2f}M"
    if a >= 1e4:
        return f"{value:,.0f}"
    if a >= 100:
        return f"{value:,.1f}"
    if a >= 1:
        return f"{value:.2f}"
    if a == 0:
        return "0"
    return f"{value:.{digits}g}"


# ------------------------------------------------------------------ descriptives

def descriptives(df: pd.DataFrame, schema: dict[str, Any], limit: int = 14) -> list[dict[str, Any]]:
    roles = schema["profiles"]
    columns = [c for c in df.columns if roles[c]["role"] in {"numeric"}][:limit]
    rows = []
    for column in columns:
        values = _numeric(df, column).dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        p_norm = None
        if 8 <= len(values) and values.nunique() > 2:
            try:
                p_norm = float(stats.shapiro(values.sample(min(len(values), 5000), random_state=0))[1])
            except Exception:
                p_norm = None
        rows.append({
            "column": column, "n": int(len(values)), "missing_pct": round(float(df[column].isna().mean() * 100), 1),
            "mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(values.median()), "q1": float(q1), "q3": float(q3),
            "min": float(values.min()), "max": float(values.max()),
            "skew": float(values.skew()) if len(values) > 2 else 0.0, "shapiro_p": p_norm,
        })
    return rows


def categorical_summary(df: pd.DataFrame, schema: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    roles = schema["profiles"]
    columns = [c for c in df.columns if roles[c]["role"] in {"categorical", "binary", "ordinal", "geo_country", "geo_state"}]
    columns = sorted(columns, key=lambda c: roles[c]["unique_count"])[:limit]
    rows = []
    for column in columns:
        counts = df[column].dropna().astype(str).value_counts()
        total = int(counts.sum())
        if not total:
            continue
        top = [(str(k), int(v), v / total * 100) for k, v in counts.head(3).items()]
        rows.append({"column": column, "levels": int(len(counts)), "n": total, "top": top,
                     "missing_pct": round(float(df[column].isna().mean() * 100), 1)})
    return rows


# ------------------------------------------------------------------ correlations

def correlations(df: pd.DataFrame, schema: dict[str, Any], top: int = 8) -> dict[str, Any]:
    roles = schema["profiles"]
    columns = [c for c in df.columns if roles[c]["role"] == "numeric"][:MAX_CORR_COLUMNS]
    frame = pd.DataFrame({c: _numeric(df, c) for c in columns})
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            pair = frame[[a, b]].dropna()
            if len(pair) < 10 or pair[a].nunique() < 2 or pair[b].nunique() < 2:
                continue
            r, p = stats.pearsonr(pair[a], pair[b])
            rho, _ = stats.spearmanr(pair[a], pair[b])
            rows.append({"a": a, "b": b, "n": int(len(pair)), "r": float(r), "rho": float(rho), "p": float(p)})
    tests = len(rows)
    for row in rows:
        row["p_adj"] = min(1.0, row["p"] * tests)
        row["strength"] = strength_label(row["r"])
        row["significant"] = row["p_adj"] < ALPHA
    rows.sort(key=lambda r: abs(r["r"]), reverse=True)
    matrix = frame.corr() if len(columns) >= 2 else None
    return {"pairs": rows[:top], "all": rows, "tests": tests, "columns": columns, "matrix": matrix,
            "n_significant": sum(1 for r in rows if r["significant"])}


# ------------------------------------------------------------------ group comparisons

def _group_frame(df: pd.DataFrame, measure: str, group: str) -> pd.DataFrame:
    return pd.DataFrame({"y": _numeric(df, measure), "g": df[group].astype(str).where(df[group].notna())}).dropna()


def dedupe_partitions(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Drop columns that split the rows exactly like an earlier one (e.g. 'species' and 'species_id')."""
    kept: list[str] = []
    for c in columns:
        duplicate = False
        for k in kept:
            joint = df[[c, k]].dropna().drop_duplicates().shape[0]
            if joint == df[c].nunique() == df[k].nunique():
                duplicate = True
                break
        if not duplicate:
            kept.append(c)
    return kept


def group_tests(df: pd.DataFrame, schema: dict[str, Any], recs: list[dict[str, Any]], top: int = 5) -> dict[str, Any]:
    roles = schema["profiles"]
    measures = [c for c in df.columns if roles[c]["role"] == "numeric"][:6]
    groupers = dedupe_partitions(df, [c for c in df.columns if roles[c]["role"] in GROUP_ROLES and 2 <= roles[c]["unique_count"] <= 12])[:4]
    candidates: list[tuple[str, str]] = []
    for rec in recs:  # start from what the workspace recommended, then fill in
        if rec.get("type") == "box" and rec.get("x") in df.columns and rec.get("y") in df.columns:
            candidates.append((rec["y"], rec["x"]))
        elif rec.get("type") == "distribution" and rec.get("x") in df.columns and rec.get("group") in df.columns:
            candidates.append((rec["x"], rec["group"]))
    for m in measures:
        for g in groupers:
            candidates.append((m, g))
    seen, ordered = set(), []
    for pair in candidates:
        if pair not in seen and pair[0] != pair[1]:
            seen.add(pair)
            ordered.append(pair)
    rows: list[dict[str, Any]] = []
    for measure, group in ordered[:16]:
        data = _group_frame(df, measure, group)
        keep = data["g"].value_counts()
        keep = keep[keep >= 3].index[:12]
        data = data[data["g"].isin(keep)]
        samples = [g["y"].to_numpy() for _, g in data.groupby("g")]
        if len(samples) < 2:
            continue
        try:
            h, p = stats.kruskal(*samples)
        except ValueError:
            continue
        k, n = len(samples), sum(len(s) for s in samples)
        eta = max(0.0, float((h - k + 1) / (n - k))) if n > k else 0.0
        medians = data.groupby("g")["y"].median().sort_values()
        rows.append({"measure": measure, "group": group, "k": k, "n": int(n), "H": float(h), "p": float(p), "eta2": eta,
                     "low": (str(medians.index[0]), float(medians.iloc[0])), "high": (str(medians.index[-1]), float(medians.iloc[-1]))})
    tests = len(rows)
    for row in rows:
        row["p_adj"] = min(1.0, row["p"] * tests)
        row["significant"] = row["p_adj"] < ALPHA
        row["effect"] = eta_label(row["eta2"])
    rows.sort(key=lambda r: r["eta2"], reverse=True)
    return {"rows": rows[:top], "tests": tests}


def cramers_v(table: pd.DataFrame) -> tuple[float, float, float]:
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    v = (chi2 / (n * (min(table.shape) - 1))) ** 0.5 if n and min(table.shape) > 1 else 0.0
    return float(chi2), float(p), float(v)


def target_associations(df: pd.DataFrame, schema: dict[str, Any], target: dict[str, Any] | None, top: int = 5) -> dict[str, Any]:
    """Which columns move with the most plausible target column."""
    if not target:
        return {}
    roles = schema["profiles"]
    t = target["column"]
    rows: list[dict[str, Any]] = []
    if target["kind"] == "classification":
        for column in df.columns:
            if column == t or roles[column]["role"] not in {"categorical", "binary", "ordinal", "geo_country"} or not 2 <= roles[column]["unique_count"] <= 25:
                continue
            table = pd.crosstab(df[column], df[t])
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            chi2, p, v = cramers_v(table)
            rows.append({"feature": column, "kind": "categorical", "stat": v, "p": p, "n": int(table.to_numpy().sum()), "effect": v_label(v)})
        target_codes = df[t].astype("category").cat.codes.where(df[t].notna())
        for column in df.columns:
            if column == t or roles[column]["role"] != "numeric":
                continue
            data = pd.DataFrame({"x": _numeric(df, column), "t": df[t].astype(str).where(df[t].notna())}).dropna()
            samples = [g["x"].to_numpy() for _, g in data.groupby("t") if len(g) >= 3]
            if len(samples) < 2:
                continue
            h, p = stats.kruskal(*samples)
            k, n = len(samples), sum(len(s) for s in samples)
            eta = max(0.0, float((h - k + 1) / (n - k))) if n > k else 0.0
            rows.append({"feature": column, "kind": "numeric", "stat": eta, "p": float(p), "n": int(n), "effect": eta_label(eta)})
    else:
        y = _numeric(df, t)
        for column in df.columns:
            if column == t or roles[column]["role"] != "numeric":
                continue
            pair = pd.DataFrame({"x": _numeric(df, column), "y": y}).dropna()
            if len(pair) < 10 or pair["x"].nunique() < 2:
                continue
            rho, p = stats.spearmanr(pair["x"], pair["y"])
            rows.append({"feature": column, "kind": "numeric", "stat": float(rho), "p": float(p), "n": int(len(pair)), "effect": strength_label(rho)})
    tests = len(rows)
    for row in rows:
        row["p_adj"] = min(1.0, row["p"] * tests)
        row["significant"] = row["p_adj"] < ALPHA
    rows.sort(key=lambda r: abs(r["stat"]), reverse=True)
    return {"target": t, "kind": target["kind"], "rows": rows[:top], "tests": tests}


# ------------------------------------------------------------------ trend

def trend(df: pd.DataFrame, schema: dict[str, Any], recs: list[dict[str, Any]]) -> dict[str, Any] | None:
    time = schema.get("time")
    ts = next((r for r in recs if r.get("type") == "timeseries" and r.get("y") in df.columns), None)
    if not time or not ts:
        return None
    t_col, y_col = time["time_column"], ts["y"]
    try:
        when = schema_mod.time_values(df, t_col, schema["profiles"][t_col])
    except Exception:
        return None
    frame = pd.DataFrame({"when": when, "value": _numeric(df, y_col)}).dropna()
    if frame.empty:
        return None
    frame["t"] = frame["when"].dt.year + (frame["when"].dt.dayofyear - 1) / 365.25 if pd.api.types.is_datetime64_any_dtype(frame["when"]) else frame["when"].astype(float)
    by_period = frame.groupby("t")["value"].mean()
    if len(by_period) < 4:
        return None
    fit = stats.linregress(by_period.index.to_numpy(), by_period.to_numpy())
    rho, p_rho = stats.spearmanr(by_period.index, by_period.to_numpy())
    first, last = float(by_period.iloc[0]), float(by_period.iloc[-1])
    return {"time": t_col, "measure": y_col, "periods": int(len(by_period)), "slope": float(fit.slope), "r2": float(fit.rvalue ** 2),
            "p": float(fit.pvalue), "rho": float(rho), "p_rho": float(p_rho), "first": first, "last": last,
            "change_pct": (last - first) / abs(first) * 100 if first else None, "start": float(by_period.index[0]), "end": float(by_period.index[-1])}


# ------------------------------------------------------------------ baseline model

def model_results(df: pd.DataFrame, schema: dict[str, Any], target: dict[str, Any] | None) -> dict[str, Any] | None:
    """A small, honest baseline: does anything beat 'always predict the majority / the mean' on rows held out?"""
    if not target:
        return None
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception:
        return None
    roles = schema["profiles"]
    t = target["column"]
    numeric = [c for c in df.columns if c != t and roles[c]["role"] in {"numeric", "ordinal_numeric", "binary_numeric", "year", "geo_lat", "geo_lon"}][:25]
    categorical = [c for c in df.columns if c != t and roles[c]["role"] in {"categorical", "binary", "ordinal", "geo_country", "geo_state"} and roles[c]["unique_count"] <= 30][:max(0, 25 - len(numeric))]
    if not numeric and not categorical:
        return None
    data = df.dropna(subset=[t])
    if len(data) > 8000:
        data = data.sample(8000, random_state=42)
    X = pd.DataFrame(index=data.index)
    for c in numeric:
        X[c] = _numeric(data, c)
    for c in categorical:
        X[c] = data[c].astype(str)
    classification = target["kind"] == "classification"
    y = data[t].astype(str) if classification else _numeric(data, t)
    if not classification:
        keep = y.notna()
        X, y = X[keep], y[keep]
    if len(X) < 40:
        return None
    stratify = y if classification and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    prep = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical),
    ])
    if classification:
        base, model = DummyClassifier(strategy="most_frequent"), RandomForestClassifier(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=-1)
    else:
        base, model = DummyRegressor(strategy="mean"), RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=-1)
    fitted = {}
    for name, est in (("baseline", base), ("model", model)):
        fitted[name] = Pipeline([("prep", prep), ("est", est)]).fit(X_tr, y_tr)
    out: dict[str, Any] = {"target": t, "kind": target["kind"], "features": len(numeric) + len(categorical), "train": int(len(X_tr)), "test": int(len(X_te))}
    for name, pipe in fitted.items():
        pred = pipe.predict(X_te)
        if classification:
            out[name] = {"accuracy": float(accuracy_score(y_te, pred)), "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
                         "macro_f1": float(f1_score(y_te, pred, average="macro"))}
        else:
            out[name] = {"r2": float(r2_score(y_te, pred)), "mae": float(mean_absolute_error(y_te, pred))}
    scoring = "balanced_accuracy" if classification else "r2"
    try:
        eval_X = X_te.head(1500)
        imp = permutation_importance(fitted["model"], eval_X, y_te.loc[eval_X.index], n_repeats=4, random_state=42, scoring=scoring)
        ranking = sorted(zip(X_te.columns, imp.importances_mean), key=lambda kv: kv[1], reverse=True)[:5]
        out["importance"] = [(str(k), float(v)) for k, v in ranking]
    except Exception:
        out["importance"] = []
    key = "balanced_accuracy" if classification else "r2"
    out["metric"] = key
    out["improvement"] = out["model"][key] - out["baseline"][key]
    return out


# ------------------------------------------------------------------ misc

def data_fingerprint(df: pd.DataFrame) -> str:
    """Short content hash of the analysed table, so the report can be tied to the exact data."""
    try:
        hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
        return hashlib.sha256(hashed.tobytes() + "|".join(map(str, df.columns)).encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "unavailable"


def software_versions() -> str:
    import sys

    import scipy

    parts = [f"Python {sys.version_info.major}.{sys.version_info.minor}", f"pandas {pd.__version__}", f"NumPy {np.__version__}", f"SciPy {scipy.__version__}"]
    try:
        import sklearn

        parts.append(f"scikit-learn {sklearn.__version__}")
    except Exception:
        pass
    return ", ".join(parts)


def compute_all(df: pd.DataFrame, analysis: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    recs = analysis.get("recommendations") or []
    targets = analysis.get("target_details") or []
    target = targets[0] if targets else None
    return {
        "descriptives": descriptives(df, schema),
        "categorical": categorical_summary(df, schema),
        "correlations": correlations(df, schema),
        "groups": group_tests(df, schema, recs),
        "target": target_associations(df, schema, target),
        "trend": trend(df, schema, recs),
        "model": model_results(df, schema, target),
        "fingerprint": data_fingerprint(df),
        "software": software_versions(),
    }
