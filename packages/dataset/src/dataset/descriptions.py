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

# Grammatical filler stripped entirely when building the "subject" phrase (never itself part of
# what a column is about).
_STOPWORDS = {"of", "the", "a", "an", "for", "in", "on", "at", "and", "per", "to", "by"}

# Unit suffixes: dropped the same way as stopwords, since a unit qualifies the measurement rather
# than naming a subject ("temperature_celsius" should read as "the temperature", not "the
# temperature of the celsius").
_UNIT_WORDS = {
    "celsius", "fahrenheit", "kelvin", "kg", "kgs", "lb", "lbs", "cm", "mm", "km", "mi", "miles",
    "meters", "meter", "m", "ft", "feet", "in", "inches", "usd", "eur", "gbp", "mph", "kmh", "kph",
    "hrs", "hours", "mins", "minutes", "secs", "seconds", "ms", "ml", "l", "liters", "litres",
    "oz", "hz", "kb", "mb", "gb", "px", "percent", "pct",
}

# Adjective-like qualifiers: kept, but prepended to the attribute noun ("average rating") rather
# than treated as the thing the column is about ("rating of the average", which reads backwards).
_MODIFIER_WORDS = {
    "average", "avg", "mean", "median", "total", "sum", "min", "minimum", "max", "maximum",
    "monthly", "annual", "yearly", "daily", "weekly", "quarterly", "estimated", "final",
    "initial", "current", "previous", "prior", "first", "last", "new", "old", "net", "gross",
    "shipping", "billing", "delivery", "booking", "contact", "mailing", "home", "work", "primary",
    "secondary",
}

# Adjacent-pair compounds where treating the second word as "the attribute" and the first as
# "the subject" would read backwards or redundantly ("the code of the zip" instead of "a postal
# code"). Keyed by (token_before, matched_token); both tokens are folded into the result instead
# of one becoming a spurious subject.
_COMPOUND_OVERRIDES: dict[tuple[str, str], tuple[str, str]] = {
    ("zip", "code"): ("a postal/ZIP code", "postal code"),
    ("postal", "code"): ("a postal/ZIP code", "postal code"),
    ("post", "code"): ("a postal/ZIP code", "postal code"),
    ("area", "code"): ("an area code", "area code"),
    ("first", "name"): ("a person's first name", "first name"),
    ("last", "name"): ("a person's last/family name", "last name"),
    ("full", "name"): ("a person's full name", "full name"),
    ("user", "name"): ("a username", "username"),
    ("screen", "name"): ("a display/screen name", "screen name"),
    ("file", "name"): ("a file name", "file name"),
    ("id", "number"): ("a unique identifier for each record", "identifier"),
    ("phone", "number"): ("a phone number", "phone number"),
    ("flight", "number"): ("a flight number", "flight number"),
    ("invoice", "number"): ("an invoice number", "invoice number"),
    ("order", "number"): ("an order number", "order number"),
    ("tracking", "number"): ("a tracking number", "tracking number"),
    ("reference", "number"): ("a reference number", "reference number"),
    ("account", "number"): ("an account number", "account number"),
    ("country", "code"): ("a country code", "country code"),
    ("credit", "score"): ("a credit score", "credit score"),
    ("gpa", "score"): ("a GPA", "GPA"),
}

# Each entry is (keywords, article_label, short_noun):
#   article_label - a full noun phrase with its article ("an email address"), used verbatim when
#     the column name is just the attribute itself (e.g. "email") with nothing left over.
#   short_noun - a bare singular noun ("email address"), used to build "school name"-style
#     combinations ("the name of the school", "the average rating") once other words in the
#     column name are folded in as a subject or modifier.
# Ordered most-specific-first: a column matching an earlier group is described by it even if a
# later group's keyword also appears in its name (e.g. "id" must win over a coincidental "code").
_ATTRIBUTE_GROUPS: list[tuple[tuple[str, ...], str, str]] = [
    (("target", "label", "outcome", "churn", "churned", "fraud", "default", "defaulted", "approved", "survived"), "the outcome or target variable this dataset appears to be organized around", "outcome"),
    (("id", "uuid", "guid", "identifier", "pk"), "a unique identifier for each record", "identifier"),
    (("password", "secret", "token", "apikey"), "a credential or security token - treat as sensitive data", "credential"),
    (("email",), "an email address", "email address"),
    (("phone", "tel", "mobile"), "a phone number", "phone number"),
    (("url", "link", "href", "website", "domain"), "a web address (URL) related to the record", "web address"),
    (("image", "photo", "picture", "avatar", "thumbnail"), "a reference to an image (e.g. a file path or URL)", "image reference"),
    (("ip", "ipaddress"), "an IP address", "IP address"),
    (("address", "street"), "a physical address or street-level location", "address"),
    (("city", "town"), "the city associated with the record", "city"),
    (("state", "province"), "the state or province associated with the record", "state/province"),
    (("country", "nation"), "the country associated with the record", "country"),
    (("zip", "postal", "postcode"), "a postal/ZIP code", "postal code"),
    (("lat", "latitude"), "a geographic latitude coordinate", "latitude"),
    (("lon", "lng", "longitude"), "a geographic longitude coordinate", "longitude"),
    (("price", "cost", "amount", "revenue", "salary", "income", "fee", "fare", "wage", "budget", "spend", "expense", "profit", "margin", "discount", "tax", "balance"), "a monetary amount", "amount"),
    (("age",), "an age, typically in years", "age"),
    (("year",), "a calendar year", "year"),
    (("month",), "a calendar month", "month"),
    (("day",), "a day of the month or week", "day"),
    (("date", "timestamp", "datetime", "login", "logout"), "a date or point in time", "date"),
    (("duration", "elapsed", "time"), "a duration or length of time", "duration"),
    (("weight", "mass"), "a physical weight/mass measurement", "weight"),
    (("height",), "a height measurement", "height"),
    (("width",), "a width measurement", "width"),
    (("length",), "a length measurement", "length"),
    (("distance", "mileage"), "a distance measurement", "distance"),
    (("speed", "velocity"), "a speed measurement", "speed"),
    (("temperature", "temp"), "a temperature measurement", "temperature"),
    (("count", "qty", "quantity", "num", "total"), "a count or total quantity", "count"),
    (("percent", "pct", "percentage", "ratio", "rate", "growth"), "a percentage or rate", "rate"),
    (("score", "rating", "rank", "grade"), "a score, rating, or rank", "rating"),
    (("gender", "sex"), "a gender/sex category", "gender"),
    (("author", "creator", "publisher", "owner"), "the name of the author, creator, or owner", "author"),
    (("category", "type", "class", "kind", "group", "segment", "tier", "method", "mode", "channel", "model", "brand", "manufacturer"), "a categorical grouping or classification", "category"),
    (("status", "flag"), "a status indicating a particular condition", "status"),
    (("code", "sku", "barcode", "number"), "a code or short identifier", "code"),
    (("description", "desc", "notes", "comment", "summary", "text", "message", "body", "content", "review"), "free-text descriptive content", "description"),
    (("color", "colour"), "a color", "color"),
    (("size",), "a size measurement or category", "size"),
    (("value", "metric", "measure", "level", "index", "indicator"), "a measured value", "value"),
    (("name", "title"), "the name or title associated with each record", "name"),
]


def _tokenize_column_name(name: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    spaced = re.sub(r"[_\-.]+", " ", spaced)
    return [token.lower() for token in spaced.split() if token]


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
    an authoritative claim.

    Rather than matching the column name against a vocabulary and stopping at the first hit (so
    "school_name" would match only "name" and describe every such column identically as "the
    name ... of each record", discarding "school" entirely), this identifies the attribute the
    name is naming (name, id, price, date, ...) and folds every other meaningful word in as the
    subject it belongs to, so "school_name" reads as "the name of the school" and "employee_id"
    as "a unique identifier associated with the employee".
    """
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

    # A column name's LAST word is, by far, the dominant real-world convention for naming the
    # attribute itself ("school_name", "product_price", "order_date") with earlier words
    # narrowing what it belongs to - so tokens are scanned from the end, and the first one that
    # matches any group wins the attribute role. Scanning by group-priority instead (checking
    # every group against the whole name) would let an earlier, unrelated word decide the
    # attribute purely by list order and produce backwards results like "city_name" -> "the city
    # of the name" instead of "the name of the city".
    token_to_group: dict[str, tuple[tuple[str, ...], str, str]] = {}
    for group in _ATTRIBUTE_GROUPS:
        for kw in group[0]:
            token_to_group.setdefault(kw, group)

    for index in range(len(tokens) - 1, -1, -1):
        matched_token = tokens[index]
        if matched_token not in token_to_group:
            continue
        keywords, article_label, short_noun = token_to_group[matched_token]
        keyword_set = set(keywords)

        if index > 0 and (tokens[index - 1], matched_token) in _COMPOUND_OVERRIDES:
            article_label, short_noun = _COMPOUND_OVERRIDES[(tokens[index - 1], matched_token)]
            keyword_set = keyword_set | {tokens[index - 1]}

        remaining = [t for t in tokens if t not in keyword_set and t not in _STOPWORDS and t not in _UNIT_WORDS]
        if not remaining:
            return f"Based on its name, this column likely represents {article_label}."

        modifiers = [t for t in remaining if t in _MODIFIER_WORDS]
        entities = [t for t in remaining if t not in _MODIFIER_WORDS]
        if entities:
            subject = " ".join(modifiers + entities)
            return f"Based on its name, this column likely represents the {short_noun} of the {subject}."
        modifier_prefix = f"{' '.join(modifiers)} " if modifiers else ""
        return f"Based on its name, this column likely represents the {modifier_prefix}{short_noun}."

    readable = " ".join(tokens) if tokens else column_name
    if column_type == "numeric":
        return f"Based on its name, this appears to be a numeric field related to '{readable}'; its exact unit or definition isn't documented."
    if column_type == "datetime":
        return f"Based on its name, this appears to be a date/time field related to '{readable}'; its exact meaning isn't documented beyond that."
    return f"Based on its name, this appears to be a categorical or text field related to '{readable}'; its exact meaning isn't documented."
