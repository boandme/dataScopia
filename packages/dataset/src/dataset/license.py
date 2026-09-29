"""Plain-language reading of a Kaggle dataset license.

Kaggle exposes only a license *name* ("CC0: Public Domain", "Other (specified in description)").
The user must be shown what that means and consent before a file is downloaded, so each known
license family maps to a short summary and a level: ``open`` (few conditions), ``conditions``
(attribution / share-alike / non-commercial terms apply), ``restricted`` (read the terms first)
or ``unknown`` (no license recorded, so no permission should be assumed).
"""

from __future__ import annotations

from typing import Any

_NC_RULE: tuple[tuple[str, ...], str, str] = (("nc", "non-commercial", "noncommercial"), "restricted",
                                                "Non-commercial: you may not use the data for commercial purposes.")
_ND_RULE: tuple[tuple[str, ...], str, str] = (("nd", "no derivative", "noderiv"), "restricted",
                                                "No derivatives: you may share the data as-is but not modify or adapt it.")

_RULES: list[tuple[tuple[str, ...], str, str]] = [
    (("cc0", "public domain", "pddl", "cc-zero"), "open",
     "Public domain: you can copy, modify and use the data for any purpose, including commercial, without asking."),
    (("odc-by", "odc attribution", "cc by 4", "cc-by-4", "cc by 3", "cc-by 3", "attribution 4", "attribution 3"), "conditions",
     "Attribution required: reuse (including commercial) is allowed if you credit the source."),
    (("sa", "share alike", "sharealike", "odbl", "gpl", "share-alike"), "conditions",
     "Share-alike: reuse is allowed with credit, but anything you build from it must carry the same license."),
    _NC_RULE,
    _ND_RULE,
    (("copyright", "all rights reserved", "proprietary", "other", "specified in description", "unknown"), "restricted",
     "Custom or restrictive terms: read the dataset's description or owner terms before using or sharing this data."),
    (("mit", "apache", "apache 2", "bsd", "isc", "unlicense"), "open",
     "Permissive open-source license: reuse, modification and commercial use are allowed with minimal conditions (typically just keeping the license notice)."),
]


def describe_license(name: str | None) -> dict[str, Any]:
    """{name, level, summary} for a license name; never raises."""
    clean = (name or "").strip()
    if not clean:
        return {"name": "Not specified", "level": "unknown",
                "summary": "No license is recorded for this dataset. Without one you have no explicit permission to reuse or redistribute it."}
    lowered = f" {clean.lower().replace('-', ' ').replace('_', ' ')} "
    # Restrictions win over permissions: "CC BY-NC-SA" is non-commercial even though it says BY.
    for tokens, level, summary in (_NC_RULE, _ND_RULE):
        if any(f" {t.replace('-', ' ')} " in lowered for t in tokens):
            return {"name": clean, "level": level, "summary": summary}
    for keys, level, summary in _RULES:
        if any((f" {k} " in lowered) or (len(k) > 3 and k in lowered) for k in keys):
            return {"name": clean, "level": level, "summary": summary}
    return {"name": clean, "level": "conditions",
            "summary": "Check the license terms on the dataset's Kaggle page before reusing or redistributing this data."}
