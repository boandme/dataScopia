"""The dataset graph: find datasets that can be combined with the one you are analysing, preview the join,
and create the combined dataset.

Candidates come from three places: the public catalogue (every Kaggle dataset anyone has analysed here is
sketched and added, so it grows with use), the built-in library, and your own saved projects. Matching uses
compact sketches of join keys (see analysis.joinkeys) - no raw data is compared or exposed.
"""

from __future__ import annotations

import json
import os
import secrets
from typing import Any

import pandas as pd
from flask import Blueprint, jsonify, request

import services
import store
from accounts import current_user
from analysis import joinkeys
from analysis import schema as schema_mod
from analysis.library import LIBRARY, load_library_frame
from errors import ApiError

bp = Blueprint("graph", __name__, url_prefix="/api/graph")

_sketch_cache: dict[str, tuple[int, dict[str, Any]]] = {}
SAMPLE_BYTES = int(float(os.getenv("ANALYSIS_SAMPLE_MB", "5")) * 1024 * 1024)
ALLOWED_PARTS = {"country", "us_state", "year", "entity", "identifier"}


# ============================================================================ catalogue

INDEX = "dataset_index"


def _index_key(slug: str) -> str:
    return slug.replace("/", "~")  # document ids cannot contain a slash


def index_dataset(slug: str, title: str, license_name: str | None, df: pd.DataFrame, source: str = "kaggle") -> None:
    """Add (or refresh) a dataset in the public catalogue. Best-effort: indexing never breaks an analysis."""
    try:
        sketch = joinkeys.build_sketch(df, title=title or slug)
        if not sketch["keys"]:
            return
        key = _index_key(slug)
        existing = store.docs().get(INDEX, key) or {}
        store.docs().put(INDEX, key, {"slug": slug, "title": title or slug, "license": license_name or "", "source": source, "rows": int(len(df)), "cols": int(df.shape[1]),
                                      "sketch": sketch, "indexed_at": store.now(), "uses": int(existing.get("uses", 0)) + 1}, public=True)
    except Exception:  # noqa: BLE001
        pass


def _my_sketch(dataset_id: str, df: pd.DataFrame, title: str) -> dict[str, Any]:
    cached = _sketch_cache.get(dataset_id)
    if cached and cached[0] == id(df):
        return cached[1]
    sketch = joinkeys.build_sketch(df, title=title)
    if len(_sketch_cache) > 40:
        _sketch_cache.clear()
    _sketch_cache[dataset_id] = (id(df), sketch)
    return sketch


def _candidates(exclude_slug: str | None, exclude_project: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in sorted(store.docs().find(INDEX, public=True, limit=2000), key=lambda d: -d.get("uses", 0)):
        if r["slug"] != exclude_slug:
            out.append({"id": f"kaggle:{r['slug']}", "type": "kaggle", "title": r["title"], "license": r["license"], "url": f"https://www.kaggle.com/datasets/{r['slug']}", "sketch": r["sketch"]})
    user = current_user()
    if user:
        for r in store.docs().find("projects", owner=store.owner_of(user)):
            if r["id"] != exclude_project and r.get("sketch"):
                out.append({"id": f"project:{r['id']}", "type": "project", "title": f"{r['name']} (your project)", "license": "", "url": "", "sketch": r["sketch"]})
    for key, meta in LIBRARY.items():
        lib_id = key.split("/", 1)[1]
        out.append({"id": f"library:{lib_id}", "type": "library", "title": meta["title"], "license": meta["license"], "url": "", "sketch": _library_sketch(lib_id)})
    return out


_library_sketches: dict[str, dict[str, Any]] = {}


def _library_sketch(lib_id: str) -> dict[str, Any]:
    if lib_id not in _library_sketches:
        _library_sketches[lib_id] = joinkeys.build_sketch(load_library_frame(f"library/{lib_id}"), title=LIBRARY[f"library/{lib_id}"]["title"])
    return _library_sketches[lib_id]


def _load_candidate(candidate_id: str) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """(frame, title, info) for a candidate id such as 'library:gapminder', 'project:<id>' or 'kaggle:<owner/name>'."""
    kind, _, ref = candidate_id.partition(":")
    if kind == "library" and f"library/{ref}" in LIBRARY:
        return load_library_frame(f"library/{ref}"), LIBRARY[f"library/{ref}"]["title"], {"type": "library", "license": LIBRARY[f"library/{ref}"]["license"]}
    if kind == "project":
        user = current_user()
        if not user:
            raise ApiError(401, "Sign in to join with your projects.")
        row = store.docs().get("projects", ref)
        if not row or row.get("owner") != store.owner_of(user):
            raise ApiError(404, "Project not found.")
        frame, _, _ = services.replay(store.load_frame(row["data_path"]), (row.get("state") or {}).get("transform_log", []))
        return frame, row["name"], {"type": "project", "license": ""}
    if kind == "kaggle" and "/" in ref:
        try:
            from dataset.streaming import load_dataset_sample

            loaded = load_dataset_sample(ref, max_sample_bytes=SAMPLE_BYTES)
        except Exception as exc:  # noqa: BLE001 - credentials, network, missing file...
            raise ApiError(502, f"Could not load that Kaggle dataset right now: {exc}") from exc
        return loaded["dataframe"], ref, {"type": "kaggle", "license": "", "sample": loaded.get("sample")}
    raise ApiError(400, "Unknown dataset to join.")


# ============================================================================ routes

@bp.get("/stats")
def stats():
    return jsonify({"indexed": len(store.docs().find(INDEX, public=True, limit=5000)), "library": len(LIBRARY)})


@bp.post("/joinable")
def joinable():
    body = request.get_json(silent=True) or {}
    dataset_id = str(body.get("dataset_id") or "")
    current, _ = services.session_frames(dataset_id)
    if current is None:
        raise ApiError(400, "This dataset is no longer loaded on the server. Re-open it first.")
    name = str(body.get("dataset_name") or "dataset")
    mine = _my_sketch(dataset_id, current, name)
    slug = dataset_id if "/" in dataset_id and not dataset_id.startswith(("local-", "upload-")) else None
    project = dataset_id[5:] if dataset_id.startswith("proj-") else None
    ranked = joinkeys.rank_candidates(mine, _candidates(slug, project), limit=12)
    for cand in ranked:
        for opt in cand["options"]:
            opt["description"] = joinkeys.describe_option(opt)
    n = len(store.docs().find(INDEX, public=True, limit=5000))
    return jsonify({"keys": [{"kind": k["kind"], "column": k["column"], "n": k.get("n")} for k in mine["keys"]], "candidates": ranked, "indexed": n})


def _run_join(body: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    dataset_id = str(body.get("dataset_id") or "")
    current, _ = services.session_frames(dataset_id)
    if current is None:
        raise ApiError(400, "This dataset is no longer loaded on the server. Re-open it first.")
    option = body.get("option") if isinstance(body.get("option"), dict) else {}
    left, right = option.get("left"), option.get("right")
    parts = option.get("parts")
    if not (isinstance(left, list) and isinstance(right, list) and left and len(left) == len(right) == len(parts or [])) or any(p not in ALLOWED_PARTS for p in parts):
        raise ApiError(400, "Choose a join key first.")
    other, title, info = _load_candidate(str(body.get("candidate_id") or ""))
    how = "inner" if body.get("how") == "inner" else "left"
    label = "".join(ch for ch in title.lower().replace(" ", "_") if ch.isalnum() or ch == "_")[:14] or "other"
    try:
        merged, stats = joinkeys.join_frames(current, other, {"left": [str(c) for c in left], "right": [str(c) for c in right], "parts": list(parts)}, how=how, right_label=label)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    return merged, stats, title, info, option


@bp.post("/preview")
def preview():
    merged, stats, title, info, option = _run_join(request.get_json(silent=True) or {})
    new_cols = stats["added_columns"]
    shown = merged[new_cols].notna().any(axis=1) if new_cols else pd.Series(True, index=merged.index)
    sample = merged[shown].head(6) if shown.any() else merged.head(6)
    return jsonify({"stats": stats, "title": title, "columns": list(sample.columns), "rows": json.loads(sample.to_json(orient="records", date_format="iso")), "candidate_license": info.get("license", "")})


@bp.post("/join")
def create_join():
    body = request.get_json(silent=True) or {}
    merged, stats, title, info, option = _run_join(body)
    mine = str(body.get("dataset_name") or "dataset")
    dataset_id = f"join-{secrets.token_hex(5)}"
    name = f"{mine} + {title.split(':')[0][:28]}".strip()
    payload = services.open_in_workspace(merged, dataset_id=dataset_id, name=name, transform_log=[], meta={"sample": info.get("sample")})
    provenance = {"left": mine, "right": title, "right_id": body.get("candidate_id"), "keys": {"left": option.get("left"), "right": option.get("right")}, "how": stats["how"], "stats": stats,
                  "right_license": info.get("license", "")}
    payload["dataset"]["derived_from"] = provenance
    return jsonify({**payload, "provenance": provenance})
