from __future__ import annotations

import re

import pandas as pd

_BULLET_PATTERN = re.compile(r"^[\-\*•]\s*[`*_]{0,2}([A-Za-z0-9_ .]{1,40}?)[`*_]{0,2}\s*[:–—]\s*(.+)$")
_BOLD_LINE_PATTERN = re.compile(r"^[`*_]{2}([A-Za-z0-9_ .]{1,40}?)[`*_]{2}\s*[:–—]\s*(.+)$")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def extract_column_descriptions(description: str, columns: list[str]) -> dict[str, str]:
    """Best-effort extraction of per-column descriptions from a dataset's free-text Kaggle
    description.

    Kaggle's public API does not reliably expose structured column metadata: the `columns`
    field returned by the file-listing endpoint was empty across every dataset we tested,
    including ones whose web page visibly documents each field (Titanic, World Happiness,
    the Olist e-commerce dataset, etc). What authors actually do instead is describe columns
    in prose within the dataset's free-text description - as a markdown table, a bullet list
    ("- **ColumnName**: meaning"), or bolded lines. This scans for those common formats and
    keeps only lines whose leading name matches a real column in the loaded dataset, so it
    degrades gracefully (returns {}) for datasets that don't document columns this way rather
    than returning noise.
    """
    if not description:
        return {}
    lookup = {_slug(c): c for c in columns}
    results: dict[str, str] = {}

    for raw_line in description.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("|") and line.count("|") >= 3:
            cells = [c.strip().strip("`*_") for c in line.strip("|").split("|")]
            if len(cells) >= 2 and not set(cells[0]) <= {"-", " ", ":"}:
                key = _slug(cells[0])
                if key in lookup and cells[1] and not set(cells[1]) <= {"-", " ", ":"}:
                    results.setdefault(lookup[key], cells[1])
            continue

        for pattern in (_BULLET_PATTERN, _BOLD_LINE_PATTERN):
            m = pattern.match(line)
            if m:
                key = _slug(m.group(1))
                if key in lookup:
                    results.setdefault(lookup[key], m.group(2).strip())
                break

    return results


# ---------------------------------------------------------------------------
# Name-based description inference. Used as a fallback for any column the dataset's own
# documentation doesn't cover: a deterministic, explainable guess from the column's name alone
# (never from its values), built the same way as the Q&A engine's trigger-word matching -
# whole-token matches against a vocabulary of common data-field terms - so results are
# reproducible and free of any external model call.
# ---------------------------------------------------------------------------

_BOOLEAN_NAME_PREFIXES = ("is_", "has_", "was_", "can_", "should_", "did_")

# Ordered most-specific-first: a column matching an earlier group is described by it even if a
# later group's keyword also appears in its name (e.g. "country" must not fall through to a
# generic catch-all just because a later group happens to share a substring).
_NAME_PATTERN_GROUPS: list[tuple[tuple[str, ...], str]] = [
    (("target", "label", "outcome", "churn", "churned", "fraud", "default", "defaulted", "approved", "survived"), "the outcome or target variable this dataset appears to be organized around"),
    (("id", "uuid", "guid", "identifier", "pk"), "a unique identifier for each record"),
    (("password", "secret", "token", "apikey"), "a credential or security token - treat as sensitive data"),
    (("email",), "an email address"),
    (("phone", "tel", "mobile"), "a phone number"),
    (("url", "link", "href", "website", "domain"), "a web address (URL) related to the record"),
    (("image", "photo", "picture", "avatar", "thumbnail"), "a reference to an image (e.g. a file path or URL)"),
    (("ip", "ipaddress"), "an IP address"),
    (("address", "street"), "a physical address or street-level location"),
    (("city", "town"), "the city associated with the record"),
    (("state", "province"), "the state or province associated with the record"),
    (("country", "nation"), "the country associated with the record"),
    (("zip", "postal", "postcode"), "a postal/ZIP code"),
    (("lat", "latitude"), "a geographic latitude coordinate"),
    (("lon", "lng", "longitude"), "a geographic longitude coordinate"),
    (("price", "cost", "amount", "revenue", "salary", "income", "fee", "fare", "wage", "budget"), "a monetary value"),
    (("age",), "an age, typically in years"),
    (("year",), "a calendar year"),
    (("month",), "a calendar month"),
    (("day",), "a day of the month or week"),
    (("date", "timestamp", "datetime"), "a date or point in time"),
    (("duration", "elapsed"), "a duration or length of time"),
    (("weight", "mass"), "a physical weight/mass measurement"),
    (("height",), "a height measurement"),
    (("width",), "a width measurement"),
    (("count", "qty", "quantity", "num", "number", "total"), "a count or total quantity"),
    (("percent", "pct", "ratio", "rate"), "a percentage or rate"),
    (("score", "rating", "rank"), "a score, rating, or rank"),
    (("gender", "sex"), "a gender/sex category"),
    (("category", "type", "class", "kind", "group", "segment", "tier"), "a categorical grouping or classification"),
    (("status", "flag"), "a status indicating a particular condition"),
    (("code",), "a code or short identifier"),
    (("description", "desc", "notes", "comment", "summary", "text", "message", "body", "content", "review"), "free-text descriptive content"),
    (("color", "colour"), "a color"),
    (("size",), "a size measurement or category"),
    (("name", "title"), "the name or title associated with each record"),
]


def _tokenize_column_name(name: str) -> set[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    spaced = re.sub(r"[_\-.]+", " ", spaced)
    return {token.lower() for token in spaced.split() if token}


def _series_type_label(series: pd.Series) -> str:
    # Order matters: pandas treats bool as a numeric subtype, so is_numeric_dtype(bool_series)
    # is True - checking it first would misclassify every boolean column as numeric.
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def infer_column_description(column_name: str, series: pd.Series) -> str:
    """A deterministic, name-based approximation of what a column represents, for use only when
    the dataset's documentation doesn't describe it. Never inspects the column's values - just
    its name and inferred type - so the result is reproducible and clearly a guess rather than
    an authoritative claim."""
    column_type = _series_type_label(series)
    tokens = _tokenize_column_name(column_name)
    lower_name = column_name.lower().replace(" ", "_")

    if column_type == "boolean" or any(lower_name.startswith(prefix) for prefix in _BOOLEAN_NAME_PREFIXES):
        matched_prefix = next((p for p in _BOOLEAN_NAME_PREFIXES if lower_name.startswith(p)), None)
        if matched_prefix:
            verb = matched_prefix.rstrip("_")
            subject = lower_name[len(matched_prefix):].replace("_", " ").strip()
            if subject:
                return f"Likely a true/false flag, based on its name, indicating whether each record {verb} {subject}."
        return "Likely a true/false flag, based on its name, indicating whether each record meets a particular condition."

    for keywords, meaning in _NAME_PATTERN_GROUPS:
        if tokens & set(keywords):
            return f"Based on its name, this column likely represents {meaning}."

    if column_type == "numeric":
        return "A numeric measurement whose exact meaning isn't documented; its name and value range are the best guide to what it captures."
    if column_type == "datetime":
        return "A date/time field whose exact meaning isn't documented beyond its inferred type."
    return "A categorical or text field whose exact meaning isn't documented; its name and the values it takes on are the best guide to what it represents."
