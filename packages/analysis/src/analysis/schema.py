"""Semantic schema detection: what each column *means*, not just what dtype pandas gave it.

A dtype alone misleads in exactly the ways that wreck automatic analysis: a customer-ID column is
"numeric" (so it gets histogrammed and correlated), dates arrive as strings (so no time-series is
ever offered), a 190-value country column is "text" (so it can't be grouped or mapped), and
prices stored as "$1,299" are "categorical". This module assigns every column a *role* using its
name, values and uniqueness together, and derives dataset-level structure from those roles: the
time axis, panel/entity structure, geography, and which columns could join to another dataset.

Everything here is deterministic and reads the data itself; nothing is guessed from the file name.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from . import geo as geo_tables

# Roles that describe measurable quantities (usable in statistics / as chart measures).
MEASURE_ROLES = {"numeric", "ordinal_numeric", "binary_numeric"}
# Roles that can partition rows into groups (x of a bar, colour of a scatter, ...).
GROUPING_ROLES = {"categorical", "binary", "ordinal", "ordinal_numeric", "geo_country", "geo_state"}

ROLE_LABELS = {
    "numeric": "Numeric",
    "ordinal_numeric": "Ordinal (numeric)",
    "binary_numeric": "Binary (0/1)",
    "binary": "Binary",
    "categorical": "Categorical",
    "ordinal": "Ordinal",
    "datetime": "Date / time",
    "year": "Year",
    "identifier": "Identifier",
    "text": "Free text",
    "geo_country": "Country",
    "geo_state": "US state",
    "geo_lat": "Latitude",
    "geo_lon": "Longitude",
    "constant": "Constant",
    "empty": "Empty",
    "unknown": "Unknown",
}

_ID_NAME_PATTERN = re.compile(r"(^|[_\s\-\.])(id|uuid|guid|pk|key|idx|index|rowid|row_id|rownum|record)($|[_\s\-\.])", re.I)
_CODE_NAME_PATTERN = re.compile(r"(^|[_\s\-])(code|num|no|number|zip|zipcode|postal|fips|isbn|ssn|phone)$", re.I)
_YEAR_NAME_PATTERN = re.compile(r"(^|[_\s\-\.])(year|yr|yyyy|fiscal_?year|season|annee)($|[_\s\-\.])|year$", re.I)
_RANK_NAME_PATTERN = re.compile(r"rank|position|place|order|score|rating|grade|level|stars?", re.I)
_DATE_NAME_PATTERN = re.compile(r"date|time|timestamp|datetime|day|created|updated|_at$|_on$", re.I)

_ISO_DATE = re.compile(r"^\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?\s?(Z|[+-]\d{2}:?\d{2}|[AaPp][Mm])?)?$")
_SLASH_DATE = re.compile(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}([ T]\d{1,2}:\d{2}(:\d{2})?\s?([AaPp][Mm])?)?$")
_MONTH_NAME_DATE = re.compile(
    r"^(\d{1,2}[\s\-/]?)?(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s\-/,.]*(\d{1,2}[\s\-/,.]*)?(\d{2,4})?$", re.I
)

_ORDINAL_SCALES = [
    ["very low", "low", "medium", "high", "very high"],
    ["low", "medium", "high"],
    ["low", "moderate", "high"],
    ["small", "medium", "large"],
    ["small", "medium", "large", "extra large"],
    ["xs", "s", "m", "l", "xl", "xxl"],
    ["poor", "fair", "good", "very good", "excellent"],
    ["bad", "average", "good"],
    ["never", "rarely", "sometimes", "often", "always"],
    ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    ["none", "mild", "moderate", "severe"],
    ["bronze", "silver", "gold", "platinum"],
    ["freshman", "sophomore", "junior", "senior"],
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"],
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    ["q1", "q2", "q3", "q4"],
]


def _safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.astype(str).nunique(dropna=True))


def _is_integer_valued(values: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(values):
        return True
    if not pd.api.types.is_float_dtype(values):
        return False
    finite = values[np.isfinite(values)]
    return bool(len(finite)) and bool((finite == np.floor(finite)).all())


def _looks_like_id_name(name: str) -> bool:
    text = str(name)
    if _ID_NAME_PATTERN.search(text) or text.lower() in {"id", "index"} or text.lower().startswith("unnamed"):
        return True
    return bool(re.search(r"[a-z0-9]Id$", text) or re.search(r"[a-z]ID$", text))


def parse_datetime_strings(series: pd.Series) -> tuple[pd.Series | None, float]:
    """Try to read a text column as dates. Returns (parsed_series_or_None, matched_fraction).

    Guarded by a regex screen on the raw strings first: pandas will happily "parse" almost
    anything (plain integers become nanosecond timestamps), so a column is only treated as dates
    when its values *look* like dates in a recognisable layout.
    """
    non_null = series.dropna()
    if non_null.empty:
        return None, 0.0
    text = non_null.astype(str).str.strip()
    sample = text.head(300)
    looks_like_date = sample.map(lambda v: bool(_ISO_DATE.match(v) or _SLASH_DATE.match(v) or _MONTH_NAME_DATE.match(v)))
    if looks_like_date.mean() < 0.9:
        return None, float(looks_like_date.mean())
    try:
        from pandas.tseries.api import guess_datetime_format

        fmt = guess_datetime_format(sample.iloc[0])
    except Exception:
        fmt = None
    try:
        if fmt:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        else:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except Exception:
        return None, 0.0
    fraction = float(parsed.notna().sum() / max(1, series.notna().sum()))
    if fraction < 0.9:
        # A guessed single format can reject a few odd rows; retry permissively before giving up.
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            fraction = float(parsed.notna().sum() / max(1, series.notna().sum()))
        except Exception:
            return None, 0.0
    if fraction < 0.9:
        return None, fraction
    return parsed, fraction


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Read '$1,299', '12%', '3 000' as numbers (values that can't be read become NaN)."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = series.astype(str).str.replace(r"[,$€£¥%\s]", "", regex=True).str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _numeric_text_fraction(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    sample = non_null.head(500)
    parsed = coerce_numeric(sample)
    return float(parsed.notna().mean())


def _ordinal_scale(values: pd.Series) -> list[str] | None:
    lowered = {str(v).strip().lower() for v in values.dropna().unique()}
    if len(lowered) < 3:
        return None
    for scale in _ORDINAL_SCALES:
        if lowered <= set(scale):
            observed = [s for s in scale if s in lowered]
            return observed
    return None


def _classify_numeric(name: str, series: pd.Series, non_null: pd.Series, unique: int, geo_info: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    n = len(non_null)
    integer_valued = _is_integer_valued(non_null)
    unique_ratio = unique / n if n else 0.0
    notes: dict[str, Any] = {}

    lat_lon = geo_info.get("lat_lon") or {}
    if name == lat_lon.get("lat"):
        return "geo_lat", notes
    if name == lat_lon.get("lon"):
        return "geo_lon", notes

    if unique == 2:
        return "binary_numeric", notes

    if integer_valued:
        lo, hi = float(non_null.min()), float(non_null.max())
        year_name = bool(_YEAR_NAME_PATTERN.search(str(name)))
        plausible_year_range = 1800 <= lo and hi <= 2100
        if plausible_year_range and (year_name or (unique >= 3 and unique <= 130 and n >= 10 and 1900 <= lo)):
            return "year", notes

        # "order_id" contains "order" but is an identifier, not a ranking
        rank_like = bool(_RANK_NAME_PATTERN.search(str(name))) and not re.search(r"(^|[_\s\-])(id|key|uuid|guid)$", str(name), re.I)
        if _CODE_NAME_PATTERN.search(str(name)) and not rank_like and unique >= 3:
            return "identifier", {"kind": "code"}
        sorted_unique = np.sort(non_null.unique())
        sequential = len(sorted_unique) >= 20 and bool((np.diff(sorted_unique) == 1).all()) and unique == n
        if not rank_like and n >= 5 and unique_ratio >= 0.9 and (_looks_like_id_name(name) or sequential):
            return "identifier", notes
        if _looks_like_id_name(name) and unique_ratio >= 0.5 and not rank_like:
            return "identifier", notes

        if unique <= 10 and n >= 20 or (unique <= 20 and unique_ratio <= 0.02 and n >= 100):
            notes["levels"] = [float(v) for v in sorted_unique[:20]]
            return "ordinal_numeric", notes
    return "numeric", notes


def classify_column(name: str, series: pd.Series, geo_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assign one column its role plus the evidence behind it."""
    geo_info = geo_info or {}
    dtype = str(series.dtype)
    non_null = series.dropna()
    n = int(len(non_null))
    profile: dict[str, Any] = {
        "name": name,
        "dtype": dtype,
        "role": "unknown",
        "missing_pct": round(float(series.isna().mean() * 100), 2) if len(series) else 0.0,
        "unique_count": 0,
        "unique_ratio": 0.0,
        "notes": {},
    }
    if n == 0:
        profile["role"] = "empty"
        return profile

    unique = _safe_nunique(non_null)
    profile["unique_count"] = unique
    profile["unique_ratio"] = round(unique / n, 4)

    if unique <= 1:
        profile["role"] = "constant"
        return profile

    if pd.api.types.is_datetime64_any_dtype(series):
        profile["role"] = "datetime"
        return profile

    if pd.api.types.is_bool_dtype(series):
        profile["role"] = "binary"
        return profile

    if pd.api.types.is_numeric_dtype(series):
        role, notes = _classify_numeric(name, series, non_null, unique, geo_info)
        profile["role"] = role
        profile["notes"] = notes
        return profile

    # ---- object / string columns ----
    country = geo_info.get("country") or {}
    us_state = geo_info.get("us_state") or {}
    if name == country.get("column"):
        profile["role"] = "geo_country"
        profile["notes"] = {"code_kind": country.get("kind"), "match_rate": country.get("match_rate")}
        return profile
    if name == us_state.get("column"):
        profile["role"] = "geo_state"
        profile["notes"] = {"code_kind": us_state.get("kind"), "match_rate": us_state.get("match_rate")}
        return profile

    parsed, fraction = parse_datetime_strings(series)
    if parsed is not None:
        profile["role"] = "datetime"
        profile["notes"] = {"stored_as_text": True, "parsed_fraction": round(fraction, 3)}
        return profile

    if unique == 2:
        profile["role"] = "binary"
        profile["notes"] = {"levels": [str(v) for v in non_null.unique()[:2]]}
        return profile

    text = non_null.astype(str)
    avg_len = float(text.str.len().mean())
    avg_words = float(text.str.split().str.len().mean())

    numeric_fraction = _numeric_text_fraction(series)
    if numeric_fraction >= 0.95:
        profile["role"] = "numeric"
        profile["notes"] = {"stored_as_text": True, "parsed_fraction": round(numeric_fraction, 3)}
        return profile

    scale = _ordinal_scale(non_null)
    if scale:
        profile["role"] = "ordinal"
        profile["notes"] = {"order": scale}
        return profile

    unique_ratio = unique / n
    if avg_len > 60 or avg_words >= 8:
        profile["role"] = "text"
        profile["notes"] = {"avg_length": round(avg_len, 1), "avg_words": round(avg_words, 1)}
        return profile
    if unique >= 20 and unique_ratio >= 0.9:
        profile["role"] = "identifier"
        return profile
    if _looks_like_id_name(name) and unique_ratio >= 0.5:
        profile["role"] = "identifier"
        return profile
    if unique <= 50 or (unique <= 200 and unique_ratio < 0.3):
        profile["role"] = "categorical"
        profile["notes"] = {"cardinality": "low" if unique <= 12 else "medium"}
        return profile
    if unique_ratio >= 0.5:
        profile["role"] = "identifier"
        return profile
    profile["role"] = "categorical"
    profile["notes"] = {"cardinality": "high"}
    return profile


# Legacy coarse type kept for older consumers (Column Explorer badges, reports, saved analyses).
def legacy_type(role: str, dtype: str, notes: dict[str, Any]) -> str:
    if role in {"numeric", "ordinal_numeric", "binary_numeric", "geo_lat", "geo_lon", "year"}:
        return "numeric" if not notes.get("stored_as_text") else "numeric"
    if role == "datetime":
        return "datetime"
    if role == "binary":
        return "boolean" if dtype == "bool" else "categorical"
    if role in {"categorical", "ordinal", "geo_country", "geo_state"}:
        return "categorical"
    if role == "identifier":
        return "identifier"
    if role == "text":
        return "text"
    if role == "constant":
        return "numeric" if dtype.startswith(("int", "float", "uint")) else "categorical"
    return "unknown"


def classify_columns(df: pd.DataFrame, geo_info: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    geo_info = geo_info if geo_info is not None else geo_tables.detect_geo(df)
    profiles: dict[str, dict[str, Any]] = {}
    for column in df.columns:
        try:
            profile = classify_column(column, df[column], geo_info)
        except Exception:
            profile = {"name": column, "dtype": str(df[column].dtype), "role": "unknown", "missing_pct": 0.0,
                       "unique_count": 0, "unique_ratio": 0.0, "notes": {}}
        profile["type"] = legacy_type(profile["role"], profile["dtype"], profile.get("notes", {}))
        profile["role_label"] = ROLE_LABELS.get(profile["role"], profile["role"])
        profiles[column] = profile
    return profiles


def columns_with_roles(profiles: dict[str, dict[str, Any]], roles: set[str]) -> list[str]:
    return [name for name, p in profiles.items() if p["role"] in roles]


# --------------------------------------------------------------------------------------
# Dataset-level structure
# --------------------------------------------------------------------------------------

_FREQUENCIES = [
    (1 / 24 * 0.75, 1 / 24 * 1.5, "hourly"),
    (0.75, 1.5, "daily"),
    (6, 8, "weekly"),
    (13, 16, "biweekly"),
    (27, 32, "monthly"),
    (85, 95, "quarterly"),
    (175, 190, "semiannual"),
    (355, 375, "yearly"),
]


def _guess_frequency(times: pd.Series) -> str:
    unique_sorted = pd.Series(pd.unique(times.dropna())).sort_values()
    if len(unique_sorted) < 3:
        return "irregular"
    days = unique_sorted.diff().dropna().dt.total_seconds() / 86400.0
    median = float(days.median())
    for low, high, label in _FREQUENCIES:
        if low <= median <= high:
            regular = float(((days >= low * 0.5) & (days <= high * 2.5)).mean())
            return label if regular >= 0.7 else "irregular"
    return "irregular"


def time_values(df: pd.DataFrame, column: str, profile: dict[str, Any]) -> pd.Series:
    """The column as real timestamps, whichever way it's stored."""
    series = df[column]
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if profile["role"] == "year":
        years = pd.to_numeric(series, errors="coerce")
        return pd.to_datetime(years.round().astype("Int64").astype(str) + "-01-01", errors="coerce")
    parsed, _ = parse_datetime_strings(series)
    return parsed if parsed is not None else pd.Series(pd.NaT, index=series.index)


def detect_time_structure(df: pd.DataFrame, profiles: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Find the dataset's time axis, its frequency, and whether it is panel data (many entities
    each observed over time), which changes which charts are meaningful."""
    candidates = [name for name, p in profiles.items() if p["role"] in {"datetime", "year"}]
    if not candidates:
        return None
    # Prefer real timestamps over bare years, and columns whose name says "date".
    candidates.sort(key=lambda c: (profiles[c]["role"] != "datetime", not _DATE_NAME_PATTERN.search(str(c)), -profiles[c]["unique_count"]))
    time_column = candidates[0]
    profile = profiles[time_column]
    times = time_values(df, time_column, profile)
    valid = times.dropna()
    if valid.empty:
        return None
    periods = int(valid.nunique())
    if periods < 3:
        return None

    frequency = "yearly" if profile["role"] == "year" else _guess_frequency(valid)
    structure: dict[str, Any] = {
        "time_column": time_column,
        "kind": profile["role"],
        "frequency": frequency,
        "start": valid.min().isoformat(),
        "end": valid.max().isoformat(),
        "n_periods": periods,
        "is_sorted": bool(valid.is_monotonic_increasing),
        "entity_column": None,
        "n_entities": None,
        "is_panel": False,
        "other_time_columns": [c for c in candidates[1:]],
    }

    # Panel structure: an entity column whose values each recur across several time points.
    n_rows = len(df)
    best: tuple[float, str, int] | None = None
    for name, p in profiles.items():
        if name == time_column or p["role"] not in {"categorical", "geo_country", "geo_state", "identifier"}:
            continue
        entities = p["unique_count"]
        if entities < 2 or entities > max(2, n_rows // 3):
            continue
        per_entity = n_rows / entities
        if per_entity < 3:
            continue
        try:
            observed = df.assign(_t=times).dropna(subset=["_t", name]).groupby(name)["_t"].nunique()
        except Exception:
            continue
        if observed.empty or float(observed.median()) < 3:
            continue
        score = float(observed.median()) + (5 if p["role"].startswith("geo") else 0)
        if best is None or score > best[0]:
            best = (score, name, entities)
    if best is not None:
        structure["entity_column"] = best[1]
        structure["n_entities"] = best[2]
        structure["is_panel"] = True
    return structure


def detect_join_keys(df: pd.DataFrame, profiles: dict[str, dict[str, Any]], geo_info: dict[str, Any], time_structure: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Columns (or pairs) that could be matched against another dataset. Advisory only: it lists
    what looks like a key and why, and never joins anything itself."""
    keys: list[dict[str, Any]] = []
    n_rows = len(df)

    country = geo_info.get("country")
    if country:
        column = country["column"]
        keys.append({
            "columns": [column], "kind": "country", "strength": "high",
            "reason": f"'{column}' holds countries ({country['matched']} recognised). Values normalise to ISO-3 codes, "
                      "so it can match another dataset even when the two spell country names differently.",
        })
    us_state = geo_info.get("us_state")
    if us_state:
        column = us_state["column"]
        keys.append({"columns": [column], "kind": "us_state", "strength": "high",
                     "reason": f"'{column}' holds US states, which normalise to 2-letter codes."})

    if time_structure:
        time_column = time_structure["time_column"]
        entity = time_structure.get("entity_column")
        if entity:
            pair = [entity, time_column]
            try:
                unique_pairs = int(df[pair].dropna().drop_duplicates().shape[0])
                is_unique = unique_pairs >= 0.98 * len(df[pair].dropna())
            except Exception:
                is_unique = False
            kind = "country+time" if (country and entity == country["column"]) else "entity+time"
            keys.append({
                "columns": pair, "kind": kind, "strength": "high" if is_unique else "medium",
                "reason": f"Each '{entity}' is observed over time in '{time_column}'"
                          + (" and the combination is unique per row" if is_unique else ", though the pair repeats in some rows")
                          + ". This is the natural key for joining time-varying data from another source.",
            })
        keys.append({"columns": [time_column], "kind": "time", "strength": "medium",
                     "reason": f"'{time_column}' is the time axis ({time_structure['frequency']}, {time_structure['n_periods']} periods) - "
                               "datasets on the same time axis can be aligned on it."})

    for name, p in profiles.items():
        if p["role"] == "identifier" and n_rows and p["unique_ratio"] >= 0.98:
            keys.append({"columns": [name], "kind": "identifier", "strength": "high",
                         "reason": f"'{name}' is unique for every row, so it identifies a record and can key a join."})
    strength_rank = {"high": 0, "medium": 1, "low": 2}
    keys.sort(key=lambda k: strength_rank[k["strength"]])
    return keys[:8]


def build_schema(df: pd.DataFrame) -> dict[str, Any]:
    """All schema knowledge about a dataframe in one call."""
    geo_info = geo_tables.detect_geo(df)
    profiles = classify_columns(df, geo_info)
    time_structure = detect_time_structure(df, profiles)
    join_keys = detect_join_keys(df, profiles, geo_info, time_structure)
    role_counts: dict[str, int] = {}
    for profile in profiles.values():
        role_counts[profile["role"]] = role_counts.get(profile["role"], 0) + 1
    return {
        "profiles": profiles,
        "geo": geo_info,
        "time": time_structure,
        "join_keys": join_keys,
        "role_counts": role_counts,
        "measures": columns_with_roles(profiles, MEASURE_ROLES),
        "groupers": columns_with_roles(profiles, GROUPING_ROLES),
    }
