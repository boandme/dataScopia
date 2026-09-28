"""The dataset graph: what can be joined to what.

A *sketch* is a small, privacy-friendly summary of a dataset's join keys - the set of ISO-3 country
codes it covers, its year range, a MinHash signature of its entity/identifier values - plus its column
names. Sketches are cheap to store (a few KB) and can be compared without touching the data, so a
catalogue of thousands of datasets can answer "what else can I combine this with?" instantly.

Matching handles the ways real tables disagree: country names vs ISO codes ("Viet Nam" vs "VNM"),
different year ranges, differently spelled entity values. ``join_frames`` then performs the merge on
the same normalisation, reports how many rows matched, and refuses joins that would explode the table.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import geo
from . import schema as schema_mod

SKETCH_VERSION = 1
MINHASH_SIZE = 64
MAX_SET_VALUES = 400
MAX_MINHASH_VALUES = 50_000
MAX_JOIN_ROWS = 2_000_000

_RNG = np.random.default_rng(20260926)
_A = _RNG.integers(1, 2 ** 63, size=MINHASH_SIZE, dtype=np.uint64) | np.uint64(1)
_B = _RNG.integers(0, 2 ** 63, size=MINHASH_SIZE, dtype=np.uint64)

SINGULAR = {"country": "country", "us_state": "state", "entity": "entity", "identifier": "identifier"}
KIND_WEIGHT = {"country+year": 1.0, "entity+year": 0.9, "us_state+year": 0.95, "country": 0.8, "us_state": 0.8, "entity": 0.7, "identifier": 0.7}


# ============================================================================ normalisation

def _norm_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _key_values(series: pd.Series, kind: str) -> pd.Series:
    """The join-comparable form of a column: ISO-3 for countries, USPS codes for states, int years."""
    if kind == "country":
        cache: dict[Any, Any] = {}
        return series.map(lambda v: cache.setdefault(v, geo.country_to_iso3(v)) if pd.notna(v) else None)
    if kind == "us_state":
        cache = {}
        return series.map(lambda v: cache.setdefault(v, geo.state_to_code(v)) if pd.notna(v) else None)
    if kind == "year":
        if pd.api.types.is_datetime64_any_dtype(series):
            return series.dt.year.astype("Int64")
        return pd.to_numeric(series, errors="coerce").round().astype("Int64")
    return series.map(lambda v: _norm_text(v) if pd.notna(v) else None)


def minhash(values: Iterable[Any]) -> list[int]:
    distinct = list({_norm_text(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))})[:MAX_MINHASH_VALUES]
    if not distinct:
        return []
    hashed = pd.util.hash_array(np.array(distinct, dtype=object)).astype(np.uint64)
    with np.errstate(over="ignore"):
        grid = hashed[None, :] * _A[:, None] + _B[:, None]  # (hashes, values), wraps modulo 2**64
    return [int(x) for x in grid.min(axis=1)]


def jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    if not sig_a or not sig_b:
        return 0.0
    return float(np.mean(np.array(sig_a) == np.array(sig_b)))


def containment_from_jaccard(j: float, n_a: int, n_b: int) -> float:
    """Estimated share of A's distinct values that also occur in B, from Jaccard and the set sizes."""
    if j <= 0 or n_a <= 0:
        return 0.0
    return max(0.0, min(1.0, j * (n_a + n_b) / ((1 + j) * n_a)))


# ============================================================================ sketches

def build_sketch(df: pd.DataFrame, schema: dict[str, Any] | None = None, title: str = "") -> dict[str, Any]:
    schema = schema or schema_mod.build_schema(df)
    profiles = schema["profiles"]
    keys: list[dict[str, Any]] = []
    used: set[str] = set()

    geo_info = schema.get("geo") or {}
    country = geo_info.get("country")
    if country and country["column"] in df.columns:
        values = sorted({v for v in _key_values(df[country["column"]], "country").dropna()})
        if len(values) >= 3:
            keys.append({"kind": "country", "column": country["column"], "values": values[:MAX_SET_VALUES], "n": len(values)})
            used.add(country["column"])
    state = geo_info.get("us_state")
    if state and state["column"] in df.columns:
        values = sorted({v for v in _key_values(df[state["column"]], "us_state").dropna()})
        if len(values) >= 3:
            keys.append({"kind": "us_state", "column": state["column"], "values": values, "n": len(values)})
            used.add(state["column"])

    time = schema.get("time")
    if time and time.get("time_column") in df.columns:
        t_col = time["time_column"]
        years = _key_values(schema_mod.time_values(df, t_col, profiles[t_col]), "year").dropna()
        distinct = sorted({int(y) for y in years})
        if len(distinct) >= 2:
            keys.append({"kind": "year", "column": t_col, "values": distinct[:MAX_SET_VALUES], "min": distinct[0], "max": distinct[-1], "n": len(distinct)})
        entity = time.get("entity_column")
        if entity and entity in df.columns and entity not in used:
            sig = minhash(df[entity].dropna().astype(str).unique())
            if sig:
                keys.append({"kind": "entity", "column": entity, "n": int(df[entity].nunique()), "minhash": sig})
                used.add(entity)
    for column in df.columns:
        if column in used or profiles[column]["role"] != "identifier" or len(keys) > 8:
            continue
        distinct = int(df[column].nunique())
        if distinct >= 10 and profiles[column].get("notes", {}).get("kind") != "code":
            sig = minhash(df[column].dropna().astype(str).unique())
            if sig:
                keys.append({"kind": "identifier", "column": column, "n": distinct, "minhash": sig})
    return {"version": SKETCH_VERSION, "title": title, "rows": int(len(df)), "cols": int(df.shape[1]), "columns": [str(c) for c in df.columns],
            "measures": [str(c) for c in schema.get("measures", [])][:60], "keys": keys}


# ============================================================================ matching

def _strength(overlap: float, weight: float) -> str:
    score = overlap * weight
    return "high" if score >= 0.6 else "medium" if score >= 0.35 else "low"


def _key_match(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    """Overlap of one key of A with one key of B (same kind), or None when they share too little."""
    if a["kind"] != b["kind"]:
        return None
    if "values" in a and "values" in b:
        sa, sb = set(a["values"]), set(b["values"])
        shared = len(sa & sb)
        if shared < (2 if a["kind"] == "year" else 3):
            return None
        return {"shared": shared, "overlap": shared / max(1, len(sa)), "reverse_overlap": shared / max(1, len(sb))}
    if "minhash" in a and "minhash" in b:
        j = jaccard(a["minhash"], b["minhash"])
        if j < 0.08:
            return None
        cont = containment_from_jaccard(j, a["n"], b["n"])
        return {"shared": int(round(cont * a["n"])), "overlap": cont, "reverse_overlap": containment_from_jaccard(j, b["n"], a["n"])}
    return None


def match_sketches(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    """Join options between two datasets, best first. Each names the column(s) on both sides."""
    by_kind_a: dict[str, list[dict[str, Any]]] = {}
    by_kind_b: dict[str, list[dict[str, Any]]] = {}
    for k in a.get("keys", []):
        by_kind_a.setdefault(k["kind"], []).append(k)
    for k in b.get("keys", []):
        by_kind_b.setdefault(k["kind"], []).append(k)

    single: dict[str, dict[str, Any]] = {}
    options: list[dict[str, Any]] = []
    for kind, left_keys in by_kind_a.items():
        for lk in left_keys:
            for rk in by_kind_b.get(kind, []):
                m = _key_match(lk, rk)
                if not m:
                    continue
                opt = {"kind": kind, "left": [lk["column"]], "right": [rk["column"]], "parts": [kind], "shared": m["shared"], "overlap": round(m["overlap"], 3),
                       "reverse_overlap": round(m["reverse_overlap"], 3)}
                if kind == "year":
                    opt["range"] = [max(lk["min"], rk["min"]), min(lk["max"], rk["max"])]
                    if opt["range"][0] > opt["range"][1]:
                        continue
                single[f"{kind}:{lk['column']}:{rk['column']}"] = opt
                options.append(opt)
    # composite keys: a place or entity observed over time on both sides
    years = [o for o in options if o["kind"] == "year"]
    for base_kind in ("country", "us_state", "entity"):
        for o in [o for o in options if o["kind"] == base_kind]:
            for y in years[:1]:
                # a panel joined to a panel must match on place AND time, or rows multiply; rank it above the bare place key
                options.append({"kind": f"{base_kind}+year", "left": o["left"] + y["left"], "right": o["right"] + y["right"], "parts": [base_kind, "year"],
                                "shared": o["shared"], "overlap": round(o["overlap"] * (0.5 + 0.5 * y["overlap"]), 3),
                                "reverse_overlap": round(o["reverse_overlap"] * (0.5 + 0.5 * y["reverse_overlap"]), 3),
                                "year_overlap": round(y["overlap"], 3), "range": y.get("range")})
    scored = []
    for o in options:
        if o["kind"] == "year":
            continue  # a year alone is not a meaningful key; it only refines a place/entity key
        weight = KIND_WEIGHT.get(o["kind"], 0.5)
        o["score"] = round(o["overlap"] * weight, 3)
        o["strength"] = _strength(o["overlap"], weight)
        scored.append(o)
    plain = {o["kind"]: o["score"] for o in scored if "+" not in o["kind"]}
    for o in scored:
        if "+" in o["kind"] and o["parts"][0] in plain:
            o["score"] = round(max(o["score"], plain[o["parts"][0]] + 0.0005), 4)  # matching on place AND time is the right key for panels
    scored.sort(key=lambda o: -o["score"])
    # keep the best option per kind so the list stays short and readable
    seen, out = set(), []
    for o in scored:
        if o["kind"] not in seen:
            seen.add(o["kind"])
            out.append(o)
    return out


def rank_candidates(mine: dict[str, Any], candidates: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    """candidates: [{id, title, source, sketch, ...}] -> those joinable to ``mine``, best first."""
    my_cols = set(mine.get("columns", []))
    ranked = []
    for cand in candidates:
        options = match_sketches(mine, cand["sketch"])
        if not options:
            continue
        best = options[0]
        new_measures = [m for m in cand["sketch"].get("measures", []) if m not in my_cols][:8]
        n_new = len([c for c in cand["sketch"].get("columns", []) if c not in my_cols and c not in best["right"]])
        if n_new == 0:
            continue  # joining would add nothing
        score = best["score"] * (1 + min(len(new_measures), 8) / 16)
        ranked.append({**{k: v for k, v in cand.items() if k != "sketch"}, "options": options, "best": best, "adds_measures": new_measures, "adds_columns": n_new,
                       "score": round(score, 3), "rows": cand["sketch"].get("rows"), "cols": cand["sketch"].get("cols")})
    ranked.sort(key=lambda r: -r["score"])
    return ranked[:limit]


# ============================================================================ joining

def describe_option(option: dict[str, Any]) -> str:
    place = {"country": "countries", "us_state": "US states", "entity": "entities", "identifier": "identifiers"}
    parts = option["kind"].split("+")
    base = place.get(parts[0], parts[0])
    if len(parts) > 1:
        rng = f", years {option['range'][0]}–{option['range'][1]}" if option.get("range") else ""
        return f"{option['shared']} shared {base}{rng} (matched on {base[:-1] if base.endswith('s') else base} and year)"
    return f"covers {option['overlap']:.0%} of your {base} ({option['shared']} shared)"


def join_frames(left: pd.DataFrame, right: pd.DataFrame, option: dict[str, Any], *, kinds: dict[str, str] | None = None, how: str = "left", right_label: str = "other") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge ``right`` onto ``left`` using an option from ``match_sketches``.

    ``kinds`` maps the key column names to their part kind ('country', 'year', ...) so both sides are
    normalised identically before comparison. Returns (merged, stats); raises ValueError if the join is
    unsafe (missing columns, or it would multiply the table beyond MAX_JOIN_ROWS).
    """
    if how not in {"left", "inner"}:
        raise ValueError("Join type must be 'left' or 'inner'.")
    left_cols, right_cols = option["left"], option["right"]
    for c in left_cols:
        if c not in left.columns:
            raise ValueError(f"Column '{c}' is not in your dataset.")
    for c in right_cols:
        if c not in right.columns:
            raise ValueError(f"Column '{c}' is not in the other dataset.")
    parts = option.get("parts") or ["entity"] * len(left_cols)
    kinds_by_side = [p for p in parts]
    lk, rk = left.copy(), right.copy()
    helper = []
    for i, (lc, rc, part) in enumerate(zip(left_cols, right_cols, kinds_by_side)):
        name = f"__key{i}__"
        lk[name] = _key_values(left[lc], part if part in {"country", "us_state", "year"} else "entity")
        rk[name] = _key_values(right[rc], part if part in {"country", "us_state", "year"} else "entity")
        helper.append(name)
    rk = rk.dropna(subset=helper)
    dup_rate = float(len(rk) / max(1, rk[helper].drop_duplicates().shape[0]))
    bring = [c for c in right.columns if c not in right_cols]
    rename = {}
    for c in bring:
        if c in left.columns:
            rename[c] = f"{c}_{right_label}"
    rk = rk[helper + bring].rename(columns=rename)
    # cheap pre-check before materialising a potentially huge table
    est = len(lk) * dup_rate
    if est > MAX_JOIN_ROWS:
        raise ValueError(f"This join would create about {int(est):,} rows because the other dataset has {dup_rate:.1f} rows per key. Aggregate it first.")
    merged = lk.merge(rk, on=helper, how=how, indicator=True)
    matched = int((merged["_merge"] == "both").sum())
    left_matched = int(lk[helper].merge(rk[helper].drop_duplicates(), on=helper, how="left", indicator=True)["_merge"].eq("both").sum())
    merged = merged.drop(columns=helper + ["_merge"])
    if len(merged) > MAX_JOIN_ROWS:
        raise ValueError(f"The joined table would have {len(merged):,} rows, over the {MAX_JOIN_ROWS:,} limit.")
    stats = {"left_rows": int(len(left)), "right_rows": int(len(right)), "result_rows": int(len(merged)), "matched_left_rows": left_matched,
             "matched_left_pct": round(left_matched / max(1, len(left)) * 100, 1), "added_columns": [rename.get(c, c) for c in bring], "rows_per_key": round(dup_rate, 2),
             "how": how, "warnings": []}
    if dup_rate > 1.5:
        stats["warnings"].append(f"The other dataset has about {dup_rate:.1f} rows per key, so matching rows are repeated.")
    if left_matched == 0:
        raise ValueError("No rows matched on these keys.")
    if stats["matched_left_pct"] < 50:
        stats["warnings"].append(f"Only {stats['matched_left_pct']}% of your rows matched.")
    return merged.reset_index(drop=True), stats
