from __future__ import annotations

import re

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
