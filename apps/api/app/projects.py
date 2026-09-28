"""Projects: an analysis saved to your account - the original data, your cleaning steps, findings and chart
setup - so it can be reopened later from any device with everything reproduced exactly."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request

import services
import store
from accounts import current_user, login_required
from analysis import joinkeys
from analysis import report_stats as RS
from errors import ApiError

bp = Blueprint("projects", __name__, url_prefix="/api/projects")

COLL = "projects"
MAX_FINDINGS, MAX_STEPS, MAX_CHARTS, MAX_TEXT = 100, 200, 60, 2000


def clean_steps(raw: Any) -> list[dict[str, Any]]:
    steps = []
    for entry in (raw if isinstance(raw, list) else [])[:MAX_STEPS]:
        if isinstance(entry, dict) and entry.get("type"):
            steps.append({"type": str(entry["type"])[:60], "params": entry.get("params") if isinstance(entry.get("params"), dict) else {},
                          "description": str(entry.get("description") or "")[:300]})
    return steps


def clean_findings(raw: Any) -> list[dict[str, Any]]:
    out = []
    for f in (raw if isinstance(raw, list) else [])[:MAX_FINDINGS]:
        if isinstance(f, dict):
            out.append({k: (str(v)[:MAX_TEXT] if isinstance(v, str) else v) for k, v in f.items() if k in {"type", "title", "summary", "question", "answer", "method", "code", "chart"}})
    return out


def clean_charts(raw: Any) -> list[dict[str, Any]]:
    keys = {"label", "type", "x", "y", "agg", "group", "sort", "granularity", "customTitle"}
    return [{k: str(v)[:200] for k, v in c.items() if k in keys} for c in (raw if isinstance(raw, list) else [])[:MAX_CHARTS] if isinstance(c, dict)]


def clean_dataset_meta(raw: Any) -> dict[str, Any]:
    """The bits of dataset metadata worth keeping: where it came from and its license/description."""
    raw = raw if isinstance(raw, dict) else {}
    info = raw.get("kaggle_info") if isinstance(raw.get("kaggle_info"), dict) else {}
    kept = {"title": str(info.get("title") or "")[:200], "license": str(info.get("license") or "")[:120], "description": str(info.get("description") or "")[:4000],
            "subtitle": str(info.get("subtitle") or "")[:300], "url": str(info.get("url") or "")[:300]} if info else {}
    return {"kaggle_info": kept, "file_type": str(raw.get("file_type") or "")[:20], "sample": raw.get("sample") if isinstance(raw.get("sample"), dict) else None,
            "derived_from": raw.get("derived_from") if isinstance(raw.get("derived_from"), dict) else None}


def build_state(body: dict[str, Any]) -> dict[str, Any]:
    return {"findings": clean_findings(body.get("findings")), "transform_log": clean_steps(body.get("transform_log")), "chart_configs": clean_charts(body.get("chart_configs")),
            "dataset_meta": clean_dataset_meta(body.get("dataset")), "notes": str(body.get("notes") or "")[:MAX_TEXT]}


def _owner() -> str:
    return store.owner_of(current_user())


def _owned(project_id: str) -> dict[str, Any]:
    doc = store.docs().get(COLL, project_id)
    if not doc or doc.get("owner") != _owner():
        raise ApiError(404, "Project not found.")
    return doc


def _save(doc: dict[str, Any]) -> None:
    try:
        store.docs().put(COLL, doc["id"], doc, owner=doc["owner"])
    except ValueError as exc:
        raise ApiError(413, str(exc)) from exc


def summary(doc: dict[str, Any]) -> dict[str, Any]:
    state = doc.get("state") or {}
    return {"id": doc["id"], "name": doc["name"], "dataset_name": doc["dataset_name"], "source": doc["source"], "dataset_slug": doc.get("dataset_slug"), "rows": doc["rows"], "cols": doc["cols"],
            "created_at": doc["created_at"], "updated_at": doc["updated_at"], "findings": len(state.get("findings", [])), "steps": len(state.get("transform_log", [])),
            "charts": len(state.get("chart_configs", [])), "license": (state.get("dataset_meta", {}).get("kaggle_info") or {}).get("license", "")}


@bp.get("")
@login_required
def list_projects():
    docs = sorted(store.docs().find(COLL, owner=_owner()), key=lambda d: d["updated_at"], reverse=True)
    return jsonify({"projects": [summary(d) for d in docs], "limit": store.MAX_PROJECTS_PER_USER})


def create_project_from_frames(owner: str, name: str, dataset_name: str, original: Any, state: dict[str, Any], *, source: str = "upload", slug: str | None = None) -> dict[str, Any]:
    if len(store.docs().find(COLL, owner=owner, limit=store.MAX_PROJECTS_PER_USER + 1)) >= store.MAX_PROJECTS_PER_USER:
        raise ApiError(409, f"You have reached the limit of {store.MAX_PROJECTS_PER_USER} projects. Delete one to save another.")
    project_id = store.new_id()
    try:
        path = store.save_frame("projects", project_id, original)
    except ValueError as exc:
        raise ApiError(413, str(exc)) from exc
    try:
        sketch = joinkeys.build_sketch(original, title=dataset_name)
    except Exception:  # a sketch is a convenience; never block saving on it
        sketch = None
    doc = {"id": project_id, "owner": owner, "name": name, "dataset_name": dataset_name, "source": source, "dataset_slug": slug, "rows": int(len(original)), "cols": int(original.shape[1]),
           "fingerprint": RS.data_fingerprint(original), "state": state, "sketch": sketch, "data_path": path, "created_at": store.now(), "updated_at": store.now()}
    try:
        store.docs().put(COLL, project_id, doc, owner=owner)
    except ValueError as exc:
        store.delete_frame(path)
        raise ApiError(413, str(exc)) from exc
    return doc


@bp.post("")
@login_required
def create_project():
    body = request.get_json(silent=True) or {}
    dataset_id = str(body.get("dataset_id") or "")
    current, original = services.session_frames(dataset_id)
    if original is None:
        raise ApiError(400, "This dataset is no longer loaded on the server. Re-open it, then save.")
    name = str(body.get("name") or "").strip()[:120] or f"{body.get('dataset_name') or 'Analysis'} project"
    dataset_meta = body.get("dataset") if isinstance(body.get("dataset"), dict) else {}
    slug = dataset_id if "/" in dataset_id and not dataset_id.startswith(("local-", "upload-")) else None
    source = "kaggle" if slug else ("join" if dataset_id.startswith("join-") else ("library" if dataset_id.startswith("library") else "upload"))
    doc = create_project_from_frames(_owner(), name, str(body.get("dataset_name") or dataset_meta.get("name") or "dataset"), original, build_state(body), source=source, slug=slug)
    return jsonify({"project": summary(doc)}), 201


@bp.get("/<project_id>")
@login_required
def get_project(project_id: str):
    doc = _owned(project_id)
    return jsonify({"project": summary(doc), "state": doc.get("state") or {}})


@bp.put("/<project_id>")
@login_required
def update_project(project_id: str):
    doc = _owned(project_id)
    body = request.get_json(silent=True) or {}
    state = doc.get("state") or {}
    if any(k in body for k in ("findings", "transform_log", "chart_configs", "notes")):
        new = build_state({"findings": body.get("findings", state.get("findings")), "transform_log": body.get("transform_log", state.get("transform_log")),
                           "chart_configs": body.get("chart_configs", state.get("chart_configs")), "notes": body.get("notes", state.get("notes")), "dataset": None})
        state.update({k: new[k] for k in ("findings", "transform_log", "chart_configs", "notes")})
    doc.update({"state": state, "name": str(body.get("name") or "").strip()[:120] or doc["name"], "updated_at": store.now()})
    _save(doc)
    return jsonify({"project": summary(doc)})


@bp.delete("/<project_id>")
@login_required
def delete_project(project_id: str):
    doc = _owned(project_id)
    store.docs().delete(COLL, project_id)
    store.delete_frame(doc.get("data_path"))
    return jsonify({"ok": True})


@bp.post("/<project_id>/duplicate")
@login_required
def duplicate_project(project_id: str):
    doc = _owned(project_id)
    original = store.load_frame(doc["data_path"])
    created = create_project_from_frames(_owner(), f"{doc['name']} (copy)", doc["dataset_name"], original, doc.get("state") or {}, source=doc["source"], slug=doc.get("dataset_slug"))
    return jsonify({"project": summary(created)}), 201


@bp.post("/<project_id>/open")
@login_required
def open_project(project_id: str):
    """Load the stored data into the workspace, replay the saved cleaning steps and return everything the UI needs."""
    doc = _owned(project_id)
    state = doc.get("state") or {}
    original = store.load_frame(doc["data_path"])
    dataset_id = f"proj-{project_id}"
    payload = services.open_in_workspace(original, dataset_id=dataset_id, name=doc["dataset_name"], transform_log=state.get("transform_log", []), meta=state.get("dataset_meta", {}))
    payload["dataset"]["project_id"] = project_id
    if doc.get("dataset_slug"):
        payload["dataset"]["slug"] = doc["dataset_slug"]
    return jsonify({**payload, "project": summary(doc), "state": {"findings": state.get("findings", []), "chart_configs": state.get("chart_configs", []), "notes": state.get("notes", "")}})
