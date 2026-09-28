"""Deeper analytics for the report: the checks a careful analyst runs *after* seeing a headline number.

A correlation coefficient or a slope alone is a weak finding. This module tests whether each headline
result is trustworthy and says what it does and does not mean:

  * correlations   - confidence intervals, shared variance, robustness to outliers, curvature, whether
                     the pattern holds inside subgroups (Simpson's paradox), multicollinearity (VIF),
                     clusters of related variables, and *explicit null results* with the smallest
                     effect the sample could have detected;
  * trends         - slope, rank trend (Kendall), compound growth, volatility, the largest jump, a
                     single change-point, acceleration, and how many entities/groups follow the trend;
  * group contrasts- pairwise post-hoc tests with Holm adjustment and rank-biserial effect sizes,
                     variance heterogeneity;
  * distributions  - skew, tails, outlier share, suggested transformation;
  * categoricals   - chi-square with the cell that drives it;
  * missing data   - which columns go missing together and whether missingness depends on other values.

Every function returns plain data plus short factual sentences (no markup); the PDF decides layout.
Interpretive text is produced separately (``*_interpretation``) so the report can keep Results and
Discussion apart.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import report_stats as RS
from . import schema as schema_mod

Z_ALPHA = 1.959964
Z_POWER = 0.841621  # 80% power


# ============================================================================ small helpers

def fisher_ci(r: float, n: int) -> tuple[float, float]:
    if n <= 3 or abs(r) >= 1:
        return (r, r)
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    return (math.tanh(z - Z_ALPHA * se), math.tanh(z + Z_ALPHA * se))


def min_detectable_r(n: int) -> float:
    """Smallest |r| this sample size could detect with 80% power at alpha = 0.05."""
    if n <= 4:
        return 1.0
    return math.tanh((Z_ALPHA + Z_POWER) / math.sqrt(n - 3))


def _iqr_mask(values: pd.Series) -> pd.Series:
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def rate_evidence(p_adj: float | None, effect: float, n: int, consistent: bool = True, effect_kind: str = "r") -> str:
    """High / Moderate / Low: needs significance, a non-trivial effect, enough data and a result that
    survives robustness checks. Deliberately conservative - a p-value alone never earns 'High'."""
    if p_adj is None or p_adj >= RS.ALPHA:
        return "Low"
    big = (abs(effect) >= 0.5) if effect_kind == "r" else (effect >= 0.14)
    mid = (abs(effect) >= 0.3) if effect_kind == "r" else (effect >= 0.06)
    if big and n >= 30 and consistent:
        return "High"
    if (mid and n >= 20) or (big and not consistent):
        return "Moderate"
    return "Low"


def _s(value: float, digits: int = 2) -> str:
    return RS.fmt_num(value, digits)


# ============================================================================ correlations

def pair_deep(df: pd.DataFrame, a: str, b: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    frame = pd.DataFrame({"x": RS._numeric(df, a), "y": RS._numeric(df, b)}).dropna()
    n = len(frame)
    if n < 10 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None
    r, p = stats.pearsonr(frame["x"], frame["y"])
    rho, _ = stats.spearmanr(frame["x"], frame["y"])
    lo, hi = fisher_ci(float(r), n)
    slope, intercept = np.polyfit(frame["x"], frame["y"], 1)
    out: dict[str, Any] = {"a": a, "b": b, "n": n, "r": float(r), "p": float(p), "rho": float(rho), "ci": (lo, hi), "r2": float(r) ** 2,
                           "slope": float(slope), "intercept": float(intercept), "min_detectable": min_detectable_r(n),
                           "sd_x": float(frame["x"].std()), "sd_y": float(frame["y"].std())}
    keep = _iqr_mask(frame["x"]) & _iqr_mask(frame["y"])
    out["outliers_removed"] = int((~keep).sum())
    if keep.sum() >= 10 and frame.loc[keep, "x"].nunique() > 1 and frame.loc[keep, "y"].nunique() > 1:
        out["r_robust"] = float(stats.pearsonr(frame.loc[keep, "x"], frame.loc[keep, "y"])[0])
    else:
        out["r_robust"] = None
    try:
        quad = np.polyfit(frame["x"], frame["y"], 2)
        resid = frame["y"] - np.polyval(quad, frame["x"])
        ss_tot = float(((frame["y"] - frame["y"].mean()) ** 2).sum())
        out["r2_quadratic"] = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot else out["r2"]
    except Exception:
        out["r2_quadratic"] = out["r2"]
    out["curvature_gain"] = max(0.0, out["r2_quadratic"] - out["r2"])
    out["monotonic_gap"] = abs(out["rho"]) - abs(out["r"])
    return out


def simpson_check(df: pd.DataFrame, a: str, b: str, schema: dict[str, Any], pooled_r: float) -> dict[str, Any] | None:
    """Does the pooled relationship hold inside subgroups, or is it produced by mixing groups?"""
    profiles = schema["profiles"]
    groupers = [c for c in df.columns if profiles[c]["role"] in {"categorical", "binary", "ordinal", "geo_country"} and 2 <= profiles[c]["unique_count"] <= 8 and c not in (a, b)]
    groupers = RS.dedupe_partitions(df, groupers)
    x, y = RS._numeric(df, a), RS._numeric(df, b)
    for g in groupers[:3]:
        frame = pd.DataFrame({"x": x, "y": y, "g": df[g].astype(str).where(df[g].notna())}).dropna()
        rows = []
        for level, sub in frame.groupby("g"):
            if len(sub) >= 15 and sub["x"].nunique() > 1 and sub["y"].nunique() > 1:
                rows.append((str(level), float(stats.pearsonr(sub["x"], sub["y"])[0]), int(len(sub))))
        if len(rows) >= 2:
            same = sum(1 for _, r_, _ in rows if np.sign(r_) == np.sign(pooled_r) and abs(r_) >= 0.1)
            return {"group": g, "rows": sorted(rows, key=lambda t: -t[2])[:6], "same_sign": same, "total": len(rows),
                    "reverses": any(np.sign(r_) != np.sign(pooled_r) and abs(r_) >= 0.2 for _, r_, _ in rows),
                    "weakest": min(rows, key=lambda t: abs(t[1])), "strongest": max(rows, key=lambda t: abs(t[1]))}
    return None


def vif_table(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if len(columns) < 3:
        return []
    frame = pd.DataFrame({c: RS._numeric(df, c) for c in columns}).dropna()
    if len(frame) < 20:
        return []
    corr = frame.corr().to_numpy()
    try:
        inv = np.linalg.pinv(corr)
    except Exception:
        return []
    rows = []
    for i, c in enumerate(columns):
        v = float(max(1.0, inv[i, i]))
        rows.append({"column": c, "vif": v, "level": "severe" if v >= 10 else "moderate" if v >= 5 else "low"})
    return sorted(rows, key=lambda r: -r["vif"])


def corr_clusters(matrix: pd.DataFrame | None, threshold: float = 0.5) -> list[list[str]]:
    if matrix is None or matrix.shape[0] < 2:
        return []
    cols = list(matrix.columns)
    seen, clusters = set(), []
    for c in cols:
        if c in seen:
            continue
        stack, comp = [c], []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            for other in cols:
                if other not in seen and abs(matrix.loc[cur, other]) >= threshold:
                    stack.append(other)
        if len(comp) >= 2:
            clusters.append(comp)
    return clusters


def correlation_deep(df: pd.DataFrame, schema: dict[str, Any], corr: dict[str, Any]) -> dict[str, Any]:
    """Deep results for the strongest pairs, plus the null results and structure of the whole matrix."""
    all_rows = corr.get("all", [])
    strongest = []
    for row in corr["pairs"][:4]:
        d = pair_deep(df, row["a"], row["b"], schema)
        if not d:
            continue
        d["p_adj"] = row["p_adj"]
        d["simpson"] = simpson_check(df, row["a"], row["b"], schema, d["r"])
        d["consistent"] = (
            np.sign(d["rho"]) == np.sign(d["r"]) and abs(d["rho"]) >= 0.8 * abs(d["r"])
            and (d["r_robust"] is None or abs(d["r_robust"] - d["r"]) < 0.15)
            and not (d["simpson"] and d["simpson"]["reverses"])
        )
        d["rating"] = rate_evidence(row["p_adj"], d["r"], d["n"], bool(d["consistent"]))
        strongest.append(d)
    nulls = []
    for row in all_rows:
        lo, hi = fisher_ci(row["r"], row["n"])
        if abs(row["r"]) < 0.15 and lo <= 0 <= hi:
            nulls.append({"a": row["a"], "b": row["b"], "n": row["n"], "r": row["r"], "ci": (lo, hi), "min_detectable": min_detectable_r(row["n"])})
    nulls.sort(key=lambda r: abs(r["r"]))
    matrix = corr.get("matrix")
    lone = []
    if matrix is not None and matrix.shape[0] >= 3:
        for c in matrix.columns:
            others = matrix.loc[c].drop(c).abs()
            if len(others) and others.max() < 0.3:
                lone.append({"column": c, "max_abs": float(others.max()), "partner": str(others.idxmax())})
    return {"strongest": strongest, "nulls": nulls[:8], "vif": vif_table(df, corr.get("columns", [])), "clusters": corr_clusters(matrix),
            "isolated": lone, "n_pairs": len(all_rows),
            "share_weak": (sum(1 for r in all_rows if abs(r["r"]) < 0.3) / len(all_rows)) if all_rows else 0.0}


def pair_facts(d: dict[str, Any]) -> list[str]:
    """Result sentences (no interpretation) for one pair."""
    a, b = d["a"], d["b"]
    lo, hi = d["ci"]
    out = [f"{a} and {b} correlated at r = {d['r']:.2f} (95% CI {lo:.2f} to {hi:.2f}, n = {d['n']:,}); the Spearman rank correlation was {d['rho']:.2f}. "
           f"The linear fit explained {d['r2']:.0%} of the variance (slope {_s(d['slope'])} {b} per unit of {a})."]
    checks = []
    if d["r_robust"] is not None and d["outliers_removed"]:
        checks.append(f"excluding {d['outliers_removed']:,} rows outside 1.5 x IQR changed r from {d['r']:.2f} to {d['r_robust']:.2f}")
    elif d["r_robust"] is not None:
        checks.append("no rows fell outside 1.5 x IQR on either variable")
    if d["curvature_gain"] >= 0.02:
        checks.append(f"a quadratic fit raised explained variance from {d['r2']:.0%} to {d['r2_quadratic']:.0%}")
    if d.get("simpson"):
        s_ = d["simpson"]
        checks.append(f"within {s_['group']} groups the correlation had the same sign in {s_['same_sign']} of {s_['total']} groups (range {min(r for _, r, _ in s_['rows']):.2f} to {max(r for _, r, _ in s_['rows']):.2f})")
    if checks:
        out.append("Robustness checks: " + "; ".join(checks) + ".")
    return out


def pair_interpretation(d: dict[str, Any], caveat: bool = True) -> list[str]:
    a, b = d["a"], d["b"]
    direction = "positive" if d["r"] > 0 else "negative"
    strength = RS.strength_label(d["r"])
    out = [f"The {strength} {direction} association between {a} and {b} means that, across this dataset, higher {a} tends to accompany {'higher' if d['r'] > 0 else 'lower'} {b}. "
           f"In standardised terms a one-standard-deviation difference in {a} goes with about a {abs(d['r']):.2f}-standard-deviation difference in {b}, and "
           f"{d['r2']:.0%} of the variation in either is shared - the remaining {1 - d['r2']:.0%} is driven by other factors or noise."]
    lo, hi = d["ci"]
    width = hi - lo
    out.append(f"The confidence interval ({lo:.2f} to {hi:.2f}) is {'narrow' if width < 0.2 else 'fairly wide' if width < 0.5 else 'wide'}, so the true strength is "
               f"{'pinned down well' if width < 0.2 else 'reasonably but not precisely known' if width < 0.5 else 'uncertain'}.")
    notes = []
    if d["monotonic_gap"] > 0.1:
        notes.append(f"The rank correlation ({d['rho']:.2f}) exceeds the linear one ({d['r']:.2f}), which indicates a relationship that consistently rises or falls but is not a straight line; a transformation such as a log scale would likely strengthen the linear fit.")
    if d["r_robust"] is not None and abs(d["r_robust"] - d["r"]) >= 0.1:
        notes.append(f"Extreme values matter here: without {d['outliers_removed']:,} outlying rows the correlation moves to {d['r_robust']:.2f}, so the headline figure {'overstates' if abs(d['r_robust']) < abs(d['r']) else 'understates'} the typical relationship.")
    elif d["r_robust"] is not None:
        notes.append("The result is not driven by a few extreme observations: it is essentially unchanged when outliers are removed.")
    if d["curvature_gain"] >= 0.03:
        notes.append(f"A curved fit explains {d['curvature_gain']:.0%} more variance than a straight line, so the effect probably changes strength across the range of {a}.")
    s_ = d.get("simpson")
    if s_:
        if s_["reverses"]:
            notes.append(f"Caution: the relationship reverses inside at least one {s_['group']} group, a sign of Simpson's paradox - the pooled correlation partly reflects differences between groups rather than a common within-group effect.")
        elif s_["same_sign"] == s_["total"]:
            notes.append(f"The relationship holds inside every {s_['group']} group (weakest in {s_['weakest'][0]} at {s_['weakest'][1]:.2f}, strongest in {s_['strongest'][0]} at {s_['strongest'][1]:.2f}), which makes a group-composition artefact unlikely.")
        else:
            notes.append(f"The relationship is not uniform: it has the same sign in only {s_['same_sign']} of {s_['total']} {s_['group']} groups, so it applies to some segments more than others.")
    out += notes
    if caveat:
        out.append(f"This is an observational association. It does not show that {a} influences {b}; the reverse direction, a shared cause, or how the data were collected could each produce it.")
    return out


def null_facts(nulls: list[dict[str, Any]]) -> list[str]:
    if not nulls:
        return []
    lines = []
    for n_ in nulls[:5]:
        lines.append(f"{n_['a']} and {n_['b']}: r = {n_['r']:.2f} (95% CI {n_['ci'][0]:.2f} to {n_['ci'][1]:.2f}, n = {n_['n']:,}); associations larger than |r| = {n_['min_detectable']:.2f} would have been detected with 80% power.")
    return lines


# ============================================================================ trends

def _series_by_period(df: pd.DataFrame, schema: dict[str, Any], y_col: str, mask: pd.Series | None = None) -> pd.Series | None:
    time = schema.get("time")
    if not time:
        return None
    t_col = time["time_column"]
    try:
        when = schema_mod.time_values(df, t_col, schema["profiles"][t_col])
    except Exception:
        return None
    frame = pd.DataFrame({"when": when, "value": RS._numeric(df, y_col)})
    if mask is not None:
        frame = frame[mask.reindex(frame.index, fill_value=False)]
    frame = frame.dropna()
    if frame.empty:
        return None
    if pd.api.types.is_datetime64_any_dtype(frame["when"]):
        frame["t"] = frame["when"].dt.year + (frame["when"].dt.dayofyear - 1) / 365.25
    else:
        frame["t"] = frame["when"].astype(float)
    series = frame.groupby("t")["value"].mean()
    if len(series) < 4:
        return None
    # t is measured in years for dates and year columns; anything else is just "steps"
    series.attrs["unit"] = "year" if pd.api.types.is_datetime64_any_dtype(frame["when"]) or schema["profiles"][t_col]["role"] == "year" else "step"
    return series


def _trend_core(series: pd.Series) -> dict[str, Any]:
    t, v = series.index.to_numpy(dtype=float), series.to_numpy(dtype=float)
    fit = stats.linregress(t, v)
    tau, p_tau = stats.kendalltau(t, v)
    span = float(t[-1] - t[0])
    first, last = float(v[0]), float(v[-1])
    out: dict[str, Any] = {"unit": series.attrs.get("unit", "step"), "periods": int(len(v)), "start": float(t[0]), "end": float(t[-1]), "first": first, "last": last, "slope": float(fit.slope), "r2": float(fit.rvalue ** 2),
                           "p": float(fit.pvalue), "tau": float(tau), "p_tau": float(p_tau), "mean": float(v.mean()),
                           "change_pct": (last - first) / abs(first) * 100 if first else None,
                           "cagr": (last / first) ** (1 / span) - 1 if first > 0 and last > 0 and span > 0 else None,
                           "slope_pct_of_mean": float(fit.slope / abs(v.mean()) * 100) if v.mean() else None}
    pct = pd.Series(v).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(pct):
        out["volatility"] = float(pct.std())
        i_up, i_dn = int(pct.idxmax()), int(pct.idxmin())
        out["max_up"] = (float(t[i_up]), float(pct.iloc[pct.index.get_loc(i_up)]) * 100)
        out["max_down"] = (float(t[i_dn]), float(pct.iloc[pct.index.get_loc(i_dn)]) * 100)
    # single change-point by piecewise-constant fit vs one straight line
    if len(v) >= 8:
        line_sse = float(((v - (fit.intercept + fit.slope * t)) ** 2).sum())
        best = None
        for k in range(3, len(v) - 2):
            sse = float(((v[:k] - v[:k].mean()) ** 2).sum() + ((v[k:] - v[k:].mean()) ** 2).sum())
            if best is None or sse < best[1]:
                best = (k, sse)
        if best and line_sse > 0 and best[1] < line_sse * 0.6:
            k = best[0]
            s1 = stats.linregress(t[:k], v[:k]).slope if k >= 3 else float("nan")
            s2 = stats.linregress(t[k:], v[k:]).slope if len(v) - k >= 3 else float("nan")
            out["changepoint"] = {"at": float(t[k]), "before": float(v[:k].mean()), "after": float(v[k:].mean()), "slope_before": float(s1), "slope_after": float(s2), "gain": 1 - best[1] / line_sse}
    # acceleration: does a curve fit noticeably better than a line?
    if len(v) >= 6:
        quad = np.polyfit(t - t.mean(), v, 2)
        resid = v - np.polyval(quad, t - t.mean())
        ss_tot = float(((v - v.mean()) ** 2).sum())
        r2_q = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot else out["r2"]
        if r2_q - out["r2"] >= 0.02:
            out["acceleration"] = "accelerating" if quad[0] * np.sign(fit.slope or 1) > 0 else "decelerating"
    return out


def trend_deep(df: pd.DataFrame, schema: dict[str, Any], recs: list[dict[str, Any]], limit: int = 3) -> dict[str, Any] | None:
    time = schema.get("time")
    if not time:
        return None
    profiles = schema["profiles"]
    measures = [r["y"] for r in recs if r.get("type") == "timeseries" and r.get("y") in df.columns]
    measures += [c for c in df.columns if profiles[c]["role"] == "numeric" and c != time["time_column"]]
    seen: list[str] = []
    for m in measures:
        if m not in seen:
            seen.append(m)
    results = []
    for m in seen[:limit]:
        series = _series_by_period(df, schema, m)
        if series is None:
            continue
        core = _trend_core(series)
        core["measure"] = m
        # how many entities follow the overall direction
        entity = time.get("entity_column")
        if time.get("is_panel") and entity and entity in df.columns:
            t_full = None
            try:
                when = schema_mod.time_values(df, time["time_column"], profiles[time["time_column"]])
                t_full = when.dt.year + (when.dt.dayofyear - 1) / 365.25 if pd.api.types.is_datetime64_any_dtype(when) else when.astype(float)
            except Exception:
                pass
            if t_full is not None:
                frame = pd.DataFrame({"e": df[entity], "t": t_full, "v": RS._numeric(df, m)}).dropna()
                slopes = []
                for _, sub in list(frame.groupby("e"))[:400]:
                    if len(sub) >= 4 and sub["t"].nunique() >= 4:
                        slopes.append(stats.linregress(sub["t"], sub["v"]).slope)
                if len(slopes) >= 5:
                    slopes = np.array(slopes)
                    core["entities"] = {"column": entity, "n": int(len(slopes)), "share_up": float((slopes > 0).mean()), "share_down": float((slopes < 0).mean()), "median_slope": float(np.median(slopes))}
        # which groups grow fastest
        groupers = [c for c in df.columns if profiles[c]["role"] in {"categorical", "ordinal", "geo_country"} and 2 <= profiles[c]["unique_count"] <= 8 and c != time.get("entity_column")]
        for g in groupers[:1]:
            rows = []
            for level, idx in df.groupby(df[g].astype(str)).groups.items():
                mask = pd.Series(False, index=df.index)
                mask.loc[idx] = True
                s = _series_by_period(df, schema, m, mask)
                if s is not None and len(s) >= 4:
                    c_ = _trend_core(s)
                    rows.append({"group": str(level), "slope": c_["slope"], "pct_per_period": c_["slope_pct_of_mean"], "change_pct": c_["change_pct"], "p": c_["p"]})
            if len(rows) >= 2:
                rows.sort(key=lambda r: -(r["pct_per_period"] if r["pct_per_period"] is not None else 0))
                core["groups"] = {"column": g, "rows": rows}
        results.append(core)
    return {"time": time["time_column"], "frequency": time.get("frequency"), "series": results} if results else None


def trend_facts(t: dict[str, Any], time_name: str) -> list[str]:
    m = t["measure"]
    span = f"{t['start']:.0f} to {t['end']:.0f}" if t["end"] - t["start"] >= 3 else f"{t['start']:.2f} to {t['end']:.2f}"
    lines = [f"Mean {m} moved from {_s(t['first'])} to {_s(t['last'])} across {t['periods']} periods ({span})"
             + (f", a change of {t['change_pct']:+.0f}%" if t.get("change_pct") is not None else "")
             + (f" (compound rate {t['cagr'] * 100:+.1f}% per {t['unit']})" if t.get("cagr") is not None else "")
             + f". The linear trend was {_s(t['slope'])} per {t['unit']} (R² = {t['r2']:.2f}, p {RS.fmt_p(t['p'])}); Kendall tau = {t['tau']:.2f} (p {RS.fmt_p(t['p_tau'])})."]
    extras = []
    if t.get("volatility") is not None:
        extras.append(f"period-to-period volatility was {t['volatility'] * 100:.1f}%")
    if t.get("max_up"):
        if t["max_down"][1] < 0:
            extras.append(f"the largest rise was {t['max_up'][1]:+.0f}% (into {t['max_up'][0]:.0f}) and the largest fall {t['max_down'][1]:+.0f}% (into {t['max_down'][0]:.0f})")
        else:
            extras.append(f"every step was an increase, from {t['max_down'][1]:+.0f}% (into {t['max_down'][0]:.0f}) to {t['max_up'][1]:+.0f}% (into {t['max_up'][0]:.0f})")
    if t.get("acceleration"):
        extras.append(f"the series was {t['acceleration']} (a curved fit beat a straight line)")
    if extras:
        lines.append("Trend detail: " + "; ".join(extras) + ".")
    cp = t.get("changepoint")
    if cp:
        lines.append(f"A single break at about {cp['at']:.0f} fitted the data much better than one line (mean {_s(cp['before'])} before vs {_s(cp['after'])} after; slope {_s(cp['slope_before'])} before vs {_s(cp['slope_after'])} after).")
    e = t.get("entities")
    if e:
        lines.append(f"Across {e['n']:,} {e['column']} values, {e['share_up']:.0%} had an upward and {e['share_down']:.0%} a downward trend (median slope {_s(e['median_slope'])} per {t['unit']}).")
    return lines


def trend_interpretation(t: dict[str, Any], time_name: str) -> list[str]:
    m = t["measure"]
    sig = t["p_tau"] < RS.ALPHA
    direction = "upward" if t["slope"] > 0 else "downward"
    out = []
    if sig and t["r2"] >= 0.8:
        out.append(f"{m} follows a clear, steady {direction} path over {time_name}: the straight line explains {t['r2']:.0%} of the variation in period averages and the rank trend is significant, so the direction is not in doubt.")
    elif sig:
        out.append(f"{m} trends {direction} over {time_name}, but with noticeable scatter around the line (R² = {t['r2']:.2f}); the direction is reliable while the exact pace is less certain.")
    else:
        out.append(f"No reliable monotonic trend in {m} was found over {time_name} (Kendall p {RS.fmt_p(t['p_tau'])}); movements between periods look more like fluctuation than direction.")
    if t.get("cagr") is not None and sig:
        doubling = math.log(2) / math.log(1 + t["cagr"]) if t["cagr"] > 0 else None
        out.append(f"At a compound rate of {t['cagr'] * 100:+.1f}% per {t['unit']} the series would "
                   + (f"double roughly every {doubling:.0f} {t['unit']}s" if doubling and doubling < 60 else "change only slowly")
                   + " if the pattern persisted (an extrapolation, not a forecast).")
    if t.get("changepoint"):
        out.append(f"The break near {t['changepoint']['at']:.0f} suggests the process changed rather than drifting smoothly; it is worth checking what happened then (policy, measurement change, composition of the sample).")
    if t.get("acceleration"):
        out.append(f"Because the series is {t['acceleration']}, a single average slope understates recent movement.")
    e = t.get("entities")
    if e:
        if e["share_up"] >= 0.8 or e["share_down"] >= 0.8:
            out.append(f"The trend is broad-based: most {e['column']} values move the same way, so it is not produced by a few large entities.")
        elif 0.35 <= e["share_up"] <= 0.65:
            out.append(f"The average trend hides divergence: {e['share_up']:.0%} of {e['column']} values rise while {e['share_down']:.0%} fall, so the mean describes few of them well.")
    g = t.get("groups")
    if g and len(g["rows"]) >= 2:
        fast, slow = g["rows"][0], g["rows"][-1]
        out.append(f"By {g['column']}, {fast['group']} changed fastest ({(fast['pct_per_period'] or 0):+.1f}% of its mean per {t['unit']}) and {slow['group']} slowest ({(slow['pct_per_period'] or 0):+.1f}%), so the overall trend is not uniform.")
    return out


# ============================================================================ group contrasts

def posthoc(df: pd.DataFrame, measure: str, group: str, max_groups: int = 8) -> dict[str, Any] | None:
    data = pd.DataFrame({"y": RS._numeric(df, measure), "g": df[group].astype(str).where(df[group].notna())}).dropna()
    counts = data["g"].value_counts()
    levels = [l for l in counts.index if counts[l] >= 5][:max_groups]
    if len(levels) < 2:
        return None
    samples = {l: data.loc[data["g"] == l, "y"].to_numpy() for l in levels}
    pairs = []
    for i, a in enumerate(levels):
        for b in levels[i + 1:]:
            try:
                u, p = stats.mannwhitneyu(samples[a], samples[b], alternative="two-sided")
            except ValueError:
                continue
            rb = 2 * u / (len(samples[a]) * len(samples[b])) - 1  # rank-biserial: >0 means a tends to be larger
            pairs.append({"a": a, "b": b, "p": float(p), "effect": float(rb), "median_a": float(np.median(samples[a])), "median_b": float(np.median(samples[b]))})
    if not pairs:
        return None
    for row, adj in zip(pairs, holm([r["p"] for r in pairs])):
        row["p_adj"] = adj
        row["magnitude"] = "negligible" if abs(row["effect"]) < 0.11 else "small" if abs(row["effect"]) < 0.28 else "medium" if abs(row["effect"]) < 0.43 else "large"
    try:
        levene_p = float(stats.levene(*samples.values())[1])
    except Exception:
        levene_p = None
    medians = {l: float(np.median(v)) for l, v in samples.items()}
    order = sorted(medians, key=medians.get)
    sig = [r for r in pairs if r["p_adj"] < RS.ALPHA]
    return {"measure": measure, "group": group, "levels": len(levels), "pairs": sorted(pairs, key=lambda r: -abs(r["effect"])), "n_pairs": len(pairs), "n_sig": len(sig),
            "levene_p": levene_p, "medians": medians, "lowest": order[0], "highest": order[-1],
            "ratio": (medians[order[-1]] / medians[order[0]]) if medians[order[0]] not in (0, 0.0) else None,
            "similar": [(r["a"], r["b"]) for r in pairs if r["p_adj"] >= RS.ALPHA][:4]}


def posthoc_facts(ph: dict[str, Any]) -> list[str]:
    out = [f"Of {ph['n_pairs']} pairwise comparisons of {ph['measure']} between {ph['group']} groups, {ph['n_sig']} were significant after Holm adjustment. "
           f"Median {ph['measure']} was highest in {ph['highest']} ({_s(ph['medians'][ph['highest']])}) and lowest in {ph['lowest']} ({_s(ph['medians'][ph['lowest']])})"
           + (f", a ratio of {ph['ratio']:.1f}x" if ph.get("ratio") and ph["ratio"] > 0 else "") + "."]
    if ph["pairs"]:
        top = ph["pairs"][0]
        out.append(f"The largest contrast was {top['a']} vs {top['b']} (rank-biserial effect {top['effect']:+.2f}, {top['magnitude']}; adjusted p {RS.fmt_p(top['p_adj'])}).")
    if ph["similar"]:
        out.append("Groups not distinguishable from each other: " + "; ".join(f"{a} and {b}" for a, b in ph["similar"]) + ".")
    if ph["levene_p"] is not None:
        out.append(f"Spread differed between groups (Levene p {RS.fmt_p(ph['levene_p'])})." if ph["levene_p"] < RS.ALPHA else f"Spread was similar across groups (Levene p {RS.fmt_p(ph['levene_p'])}).")
    return out


def posthoc_interpretation(ph: dict[str, Any], g_row: dict[str, Any], caveat: bool = True) -> list[str]:
    m, g = ph["measure"], ph["group"]
    out = []
    share = g_row["eta2"] * 100
    if g_row["significant"]:
        out.append(f"{g} is a meaningful segmenting variable for {m}: group membership accounts for about {share:.0f}% of the variation ({g_row['effect']} effect), and the ordering of groups is "
                   f"{'clear-cut' if ph['n_sig'] >= 0.7 * ph['n_pairs'] else 'only partly clear-cut'} ({ph['n_sig']} of {ph['n_pairs']} pairs differ reliably).")
    else:
        out.append(f"{g} explains little of the variation in {m} ({share:.0f}%), so pooling groups loses little information.")
    if ph.get("ratio") and ph["ratio"] > 1.5:
        out.append(f"The gap between the extreme groups is large in practical terms ({ph['highest']} is about {ph['ratio']:.1f}x {ph['lowest']}), so average values for the whole dataset would be misleading for either.")
    if ph["levene_p"] is not None and ph["levene_p"] < RS.ALPHA:
        out.append("Unequal spread means some groups are far more variable than others; comparisons of averages should be accompanied by their spread.")
    if caveat:
        out.append("Group differences are descriptive here: they may reflect who or what ended up in each group rather than an effect of belonging to it.")
    return out


# ============================================================================ distributions, categoricals, missingness

def distribution_diagnostics(df: pd.DataFrame, schema: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    profiles = schema["profiles"]
    rows = []
    for c in [c for c in df.columns if profiles[c]["role"] == "numeric"][:limit]:
        v = RS._numeric(df, c).dropna()
        if len(v) < 8:
            continue
        skew, kurt = float(v.skew()), float(v.kurt())
        out_pct = float((~_iqr_mask(v)).mean() * 100)
        if abs(skew) > 1:
            suggestion = "log transform" if v.min() > 0 else "log(1 + x) transform" if v.min() >= 0 else "rank-based or robust methods"
        else:
            suggestion = "none needed"
        shape = "right-skewed" if skew > 1 else "left-skewed" if skew < -1 else "roughly symmetric"
        rows.append({"column": c, "skew": skew, "kurtosis": kurt, "outlier_pct": out_pct, "zero_pct": float((v == 0).mean() * 100), "shape": shape,
                     "tails": "heavy-tailed" if kurt > 3 else "light-tailed" if kurt < -1 else "normal-like tails", "suggestion": suggestion})
    return rows


def categorical_associations(df: pd.DataFrame, schema: dict[str, Any], top: int = 4) -> list[dict[str, Any]]:
    profiles = schema["profiles"]
    cols = [c for c in df.columns if profiles[c]["role"] in {"categorical", "binary", "ordinal", "geo_country"} and 2 <= profiles[c]["unique_count"] <= 15][:6]
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            table = pd.crosstab(df[a], df[b])
            if table.shape[0] < 2 or table.shape[1] < 2 or table.to_numpy().sum() < 20:
                continue
            chi2, p, _, exp = stats.chi2_contingency(table, correction=False)
            n = table.to_numpy().sum()
            v = (chi2 / (n * (min(table.shape) - 1))) ** 0.5
            resid = (table.to_numpy() - exp) / np.sqrt(np.where(exp > 0, exp, 1))
            i_, j_ = np.unravel_index(np.abs(resid).argmax(), resid.shape)
            rows.append({"a": a, "b": b, "n": int(n), "chi2": float(chi2), "p": float(p), "v": float(v), "effect": RS.v_label(v),
                         "cell": (str(table.index[i_]), str(table.columns[j_]), float(table.iloc[i_, j_]), float(exp[i_, j_])),
                         "sparse": float((exp < 5).mean())})
    for row in rows:
        row["p_adj"] = min(1.0, row["p"] * len(rows))
    rows.sort(key=lambda r: -r["v"])
    return rows[:top]


def missingness_deep(df: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any] | None:
    miss = [c for c in df.columns if 0.01 < df[c].isna().mean() < 0.99]
    if not miss:
        return None
    profiles = schema["profiles"]
    indicators = df[miss].isna().astype(float)
    co = []
    for i, a in enumerate(miss):
        for b in miss[i + 1:]:
            if indicators[a].std() > 0 and indicators[b].std() > 0:
                co.append((a, b, float(indicators[a].corr(indicators[b]))))
    co.sort(key=lambda t: -abs(t[2]))
    numeric = [c for c in df.columns if profiles[c]["role"] == "numeric"][:6]
    dependent = []
    for c in sorted(miss, key=lambda c: -df[c].isna().mean())[:2]:
        flag = df[c].isna()
        for other in numeric:
            if other == c:
                continue
            vals = RS._numeric(df, other)
            a_, b_ = vals[flag].dropna(), vals[~flag].dropna()
            if len(a_) >= 10 and len(b_) >= 10:
                u, p = stats.mannwhitneyu(a_, b_, alternative="two-sided")
                rb = 2 * u / (len(a_) * len(b_)) - 1
                if p < 0.05 / max(1, len(numeric)) and abs(rb) >= 0.2:
                    dependent.append({"missing": c, "other": other, "effect": float(rb), "median_missing": float(a_.median()), "median_present": float(b_.median())})
    return {"columns": miss, "co_missing": [t for t in co if abs(t[2]) >= 0.3][:3], "dependent": dependent[:4]}


# ============================================================================ per-chart analysis

def _concentration(values: pd.Series) -> dict[str, float]:
    v = values.dropna().astype(float)
    v = v[v > 0]
    if len(v) < 2:
        return {}
    share = v / v.sum()
    s = np.sort(v.to_numpy())
    n = len(s)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(s) / (n * s.sum()))
    return {"n": n, "top1": float(share.max()), "top3": float(share.nlargest(3).sum()), "hhi": float((share ** 2).sum()), "gini": gini, "cv": float(v.std() / v.mean()) if v.mean() else 0.0}


def analyze_chart(df: pd.DataFrame, schema: dict[str, Any], chart: dict[str, Any], corr: dict[str, Any]) -> dict[str, list[str]] | None:
    """Multi-paragraph reading of one chart: what it shows (facts), then what it means (interpretation)."""
    profiles = schema["profiles"]
    ctype, x, y, group = chart.get("type"), chart.get("x"), chart.get("y"), chart.get("group")
    cols = set(df.columns)
    try:
        if ctype in {"scatter"} and x in cols and y in cols and profiles[x]["role"] in {"numeric", "ordinal_numeric", "year"} and profiles[y]["role"] in {"numeric", "ordinal_numeric", "year"}:
            d = pair_deep(df, x, y, schema)
            if d:
                d["simpson"] = simpson_check(df, x, y, schema, d["r"])
                return {"shows": pair_facts(d), "means": pair_interpretation(d)}
        if ctype in {"line", "area", "timeseries"} and schema.get("time"):
            m = y if y in cols and profiles[y]["role"] == "numeric" else None
            m = m or next((c for c in df.columns if profiles[c]["role"] == "numeric"), None)
            series = _series_by_period(df, schema, m) if m else None
            if series is not None:
                t = _trend_core(series)
                t["measure"] = m
                return {"shows": trend_facts(t, schema["time"]["time_column"]), "means": trend_interpretation(t, schema["time"]["time_column"])}
        if ctype in {"box", "violin", "distribution", "target"}:
            measure = y if ctype in {"box", "violin"} else x
            grouper = x if ctype in {"box", "violin"} else (group or y)
            if measure in cols and grouper in cols and profiles[measure]["role"] in {"numeric", "ordinal_numeric"}:
                ph = posthoc(df, measure, grouper)
                if ph:
                    groups = RS.group_tests(df.assign(), schema, [], top=50)["rows"]
                    row = next((g for g in groups if g["measure"] == measure and g["group"] == grouper), None)
                    row = row or {"eta2": 0.0, "significant": False, "effect": "negligible"}
                    return {"shows": posthoc_facts(ph), "means": posthoc_interpretation(ph, row)}
        if ctype == "histogram" and x in cols and profiles[x]["role"] in {"numeric", "ordinal_numeric"}:
            v = RS._numeric(df, x).dropna()
            diag = next((d for d in distribution_diagnostics(df, schema, 40) if d["column"] == x), None)
            if diag:
                mean, med = float(v.mean()), float(v.median())
                shows = [f"{x} is {diag['shape']} with {diag['tails']} (skewness {diag['skew']:.2f}, excess kurtosis {diag['kurtosis']:.2f}). Mean {_s(mean)} vs median {_s(med)}; "
                         f"{diag['outlier_pct']:.1f}% of values lie outside 1.5 x IQR" + (f" and {diag['zero_pct']:.0f}% are exactly zero" if diag["zero_pct"] >= 5 else "") + "."]
                means = []
                if abs(diag["skew"]) > 1:
                    means.append(f"The distribution is dominated by a long tail: the mean is {abs(mean - med) / abs(med) * 100 if med else 0:.0f}% away from the median, so the median is the more honest 'typical' value and mean-based statistics for {x} (including Pearson correlations) are pulled by the tail. "
                                 f"Consider a {diag['suggestion']} before modelling.")
                else:
                    means.append(f"The mean and median agree closely, so summaries based on either describe a typical {x} well; no transformation is needed.")
                if diag["outlier_pct"] > 5:
                    means.append(f"More than {diag['outlier_pct']:.0f}% of values are outliers by the IQR rule, which is more than isolated errors would produce and suggests genuine sub-populations or a heavy-tailed process.")
                return {"shows": shows, "means": means}
        if ctype in {"bar", "stacked_bar", "pie", "treemap", "map"} and x in cols:
            if y in cols and profiles[y]["role"] == "numeric" and chart.get("agg") not in (None, "none", "count"):
                agg = chart.get("agg")
                table = pd.DataFrame({"k": df[x].astype(str), "v": RS._numeric(df, y)}).dropna().groupby("k")["v"].agg(agg)
                label = f"{agg} of {y}"
            else:
                table = df[x].astype(str).value_counts()
                label = "rows"
            table = table.sort_values(ascending=False)
            conc = _concentration(table)
            if conc and len(table) >= 3:
                top = table.index[0]
                shows = [f"Across {len(table):,} {x} categories the largest is {top} ({_s(float(table.iloc[0]))} {label}, {conc['top1']:.0%} of the total); the top three together account for {conc['top3']:.0%}. "
                         f"Concentration (Gini) is {conc['gini']:.2f} and the coefficient of variation {conc['cv']:.2f}; the smallest is {table.index[-1]} ({_s(float(table.iloc[-1]))})."]
                means = []
                if conc["gini"] >= 0.6 or conc["top3"] >= 0.6:
                    means.append(f"The picture is highly concentrated: a few {x} values carry most of the total, so overall figures mostly describe those few and the many small categories contribute little.")
                elif conc["gini"] <= 0.3:
                    means.append(f"Values are spread fairly evenly across {x}, so no single category dominates the totals.")
                else:
                    means.append(f"Values are moderately unequal across {x}: there is a clear leading group, but the rest contribute meaningfully.")
                if ctype == "map":
                    means.append("Geographic patterns can reflect population size or how many records each place contributed rather than a real regional difference; normalise by a per-capita or per-record measure before drawing conclusions.")
                return {"shows": shows, "means": means}
        if ctype == "heatmap":
            matrix = corr.get("matrix")
            if matrix is not None and matrix.shape[0] >= 3:
                clusters = corr_clusters(matrix)
                weak = corr.get("share_weak", 0)
                shows = [f"Of {corr.get('n_pairs', 0)} variable pairs, {weak:.0%} have |r| below 0.3." + (f" Variables that cluster together (|r| >= 0.5): " + "; ".join(", ".join(c) for c in clusters[:3]) + "." if clusters else " No group of variables is strongly interrelated.")]
                means = [("Because several variables are strongly interrelated, they carry overlapping information; using them together as model inputs risks multicollinearity." if clusters
                          else "Most variables move independently of one another, so each contributes distinct information, but none is easily predicted from the others.")]
                return {"shows": shows, "means": means}
    except Exception:
        return None
    return None


# ============================================================================ orchestration

def compute_deep(df: pd.DataFrame, analysis: dict[str, Any], schema: dict[str, Any], S: dict[str, Any]) -> dict[str, Any]:
    recs = analysis.get("recommendations") or []
    corr = S["correlations"]
    out: dict[str, Any] = {"corr": correlation_deep(df, schema, corr), "trends": trend_deep(df, schema, recs),
                           "dist": distribution_diagnostics(df, schema), "cats": categorical_associations(df, schema), "missing": missingness_deep(df, schema)}
    posts = []
    for g in S["groups"]["rows"][:2]:
        ph = posthoc(df, g["measure"], g["group"])
        if ph:
            posts.append((ph, g))
    out["posthoc"] = posts
    return out
