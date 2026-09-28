"""Verified, shareable analyses.

Publishing freezes an analysis: the original table, the ordered cleaning steps, the headline metrics and an
evidence table are stored under a public link. Anyone can *re-run verification* - the server replays the
steps on the stored data and recomputes every number - and the page shows whether they still reproduce.
Signed-in readers can fork it into their own project. Raw data is never exposed through the public API;
the notebook download embeds a data snapshot only when the license and the author allow reuse.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
from flask import Blueprint, Response, jsonify, request

import projects as projects_mod
import services
import store
import workspace_cache
from accounts import current_user, login_required
from analysis import report_deep as D
from analysis import report_stats as RS
from analysis import schema as schema_mod
from analysis import verify as V
from analysis.charts import render_chart
from analysis.notebook_export import build_notebook
from analysis.profiler import analyze_dataframe
from analysis.report_pdf import build_report_pdf
from dataset.license import describe_license
from errors import ApiError

bp = Blueprint("shares", __name__, url_prefix="/api/shares")

_last_verify: dict[str, float] = {}
VERIFY_COOLDOWN = 4.0
MAX_TITLE, MAX_DESCRIPTION = 140, 1200


# ============================================================================ snapshot

def evidence_rows(final: pd.DataFrame, schema: dict[str, Any], analysis: dict[str, Any]) -> list[dict[str, str]]:
    """The headline findings with a statistic, what they were checked against and a High/Moderate/Low rating."""
    recs = analysis.get("recommendations") or []
    corr = RS.correlations(final, schema)
    groups = RS.group_tests(final, schema, recs)
    deep_corr = D.correlation_deep(final, schema, corr)
    rows: list[dict[str, str]] = []
    for d in deep_corr["strongest"][:3]:
        rob = []
        if d["r_robust"] is not None:
            rob.append(f"r {d['r_robust']:.2f} without outliers")
        if d.get("simpson"):
            rob.append(f"same sign in {d['simpson']['same_sign']}/{d['simpson']['total']} {d['simpson']['group']} groups")
        rows.append({"finding": f"{d['a']} ~ {d['b']}", "statistic": f"r = {d['r']:.2f} ({d['ci'][0]:.2f} to {d['ci'][1]:.2f}); adj p {RS.fmt_p(d['p_adj'])}",
                     "robustness": "; ".join(rob) or "-", "rating": d["rating"]})
    for g in groups["rows"][:2]:
        ph = D.posthoc(final, g["measure"], g["group"])
        consistent = bool(ph) and ph["n_sig"] >= 0.5 * ph["n_pairs"]
        rows.append({"finding": f"{g['measure']} by {g['group']}", "statistic": f"eta² = {g['eta2']:.2f}; adj p {RS.fmt_p(g['p_adj'])}",
                     "robustness": f"{ph['n_sig']}/{ph['n_pairs']} pairs differ (Holm)" if ph else "-", "rating": D.rate_evidence(g["p_adj"], g["eta2"], g["n"], consistent, "eta")})
    trends = D.trend_deep(final, schema, recs, limit=2)
    for t in (trends or {"series": []})["series"][:2]:
        e = t.get("entities")
        broad = e is None or max(e["share_up"], e["share_down"]) >= 0.7
        rows.append({"finding": f"{t['measure']} over {trends['time']}", "statistic": f"tau = {t['tau']:.2f}; R² = {t['r2']:.2f}; p {RS.fmt_p(t['p_tau'])}",
                     "robustness": f"{e['share_up']:.0%} of {e['column']} rising" if e else "period means only",
                     "rating": D.rate_evidence(t["p_tau"], t["r2"] ** 0.5, t["periods"] * 3, broad)})
    for n_ in deep_corr["nulls"][:2]:
        rows.append({"finding": f"{n_['a']} ~ {n_['b']}", "statistic": f"r = {n_['r']:.2f} ({n_['ci'][0]:.2f} to {n_['ci'][1]:.2f})",
                     "robustness": f"detects |r| > {n_['min_detectable']:.2f} (80% power)", "rating": "Null result"})
    return rows


def default_charts(recs: list[dict[str, Any]], workspace: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    out, seen = [], set()
    for cfg in list(workspace) + list(recs):
        key = (cfg.get("type"), cfg.get("x"), cfg.get("y"), cfg.get("group"))
        if cfg.get("type") and key not in seen:
            seen.add(key)
            out.append({k: cfg.get(k) for k in ("type", "x", "y", "agg", "group", "sort", "granularity", "customTitle", "title") if cfg.get(k) not in (None, "")})
        if len(out) >= limit:
            break
    return out


def data_policy(dataset_meta: dict[str, Any], source: str, requested: bool) -> tuple[bool, str]:
    """May the stored data be copied by others (fork) or embedded in downloads? Author consent AND license/privacy."""
    info = dataset_meta.get("kaggle_info") or {}
    if source == "kaggle":
        level = describe_license(info.get("license"))["level"]
        if level not in {"open", "conditions"}:
            return False, "The dataset's license does not clearly allow redistribution."
    return bool(requested), "" if requested else "The author chose not to allow copying of the data."


# ============================================================================ helpers

COLL = "shares"


def _doc(share_id: str) -> dict[str, Any]:
    doc = store.docs().get(COLL, share_id)
    if not doc:
        raise ApiError(404, "This analysis does not exist or has been removed.")
    return doc


def _save(doc: dict[str, Any]) -> None:
    store.docs().put(COLL, doc["id"], doc, owner=doc["owner"], public=doc["visibility"] == "public")


def _is_owner(doc: dict[str, Any]) -> bool:
    user = current_user()
    return bool(user) and store.owner_of(user) == doc["owner"]


def _final_frame(doc: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    snap = doc["snapshot"]
    key = f"share::{doc['id']}"
    original = store.load_frame(doc["data_path"])
    final = workspace_cache.get(key)
    if final is None:
        final = V.replay_steps(original, snap["steps"])
        workspace_cache.put(key, final)
    return original, final, snap["steps"]


def public(doc: dict[str, Any]) -> dict[str, Any]:
    snap = doc["snapshot"]
    verification = doc.get("verification")
    return {"id": doc["id"], "title": doc["title"], "description": doc["description"], "visibility": doc["visibility"], "author": doc["author"], "created_at": doc["created_at"],
            "forks": doc.get("forks", 0), "views": doc.get("views", 0), "repro_hash": doc["repro_hash"], "verified_at": doc.get("verified_at"),
            "verification": ({k: verification[k] for k in ("verified", "checked", "passed", "ran_at", "engine", "software", "error") if k in verification} if verification else None),
            "dataset": snap["dataset"], "headline": snap["headline"], "evidence": snap["evidence"], "steps": snap["steps"], "charts": snap["charts"], "findings": snap["findings"],
            "allow_data_reuse": snap["allow_data_reuse"], "data_note": snap.get("data_note", ""), "is_owner": _is_owner(doc)}


def _summary(doc: dict[str, Any]) -> dict[str, Any]:
    snap = doc["snapshot"]
    verification = doc.get("verification") or {}
    top = next((e for e in snap["evidence"] if e["rating"] in {"High", "Moderate"}), (snap["evidence"] or [None])[0])
    return {"id": doc["id"], "title": doc["title"], "description": doc["description"][:220], "author": doc["author"], "dataset_title": snap["dataset"].get("title"), "source": snap["dataset"].get("source"),
            "license": snap["dataset"].get("license"), "verified": bool(verification.get("verified")), "forks": doc.get("forks", 0), "views": doc.get("views", 0), "created_at": doc["created_at"],
            "rows": snap["headline"].get("rows"), "cols": snap["headline"].get("columns"), "quality_score": snap["headline"].get("quality_score"),
            "top_finding": f"{top['finding']}: {top['statistic']}" if top else "", "visibility": doc["visibility"]}


# ============================================================================ publish + browse

@bp.post("")
@login_required
def publish():
    body = request.get_json(silent=True) or {}
    dataset_id = str(body.get("dataset_id") or "")
    current, original = services.session_frames(dataset_id)
    if original is None:
        raise ApiError(400, "This dataset is no longer loaded on the server. Re-open it, then publish.")
    title = str(body.get("title") or "").strip()[:MAX_TITLE]
    if not title:
        raise ApiError(400, "Give the analysis a title before publishing.")
    visibility = "unlisted" if body.get("visibility") == "unlisted" else "public"
    steps = projects_mod.clean_steps(body.get("transform_log"))
    state = projects_mod.build_state(body)
    try:
        final = V.replay_steps(original, steps)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(400, f"The cleaning steps could not be replayed on the original data: {exc}") from exc
    name = str(body.get("dataset_name") or "dataset")
    slug = dataset_id if "/" in dataset_id and not dataset_id.startswith(("local-", "upload-")) else None
    source = "kaggle" if slug else ("join" if dataset_id.startswith("join-") else ("library" if dataset_id.startswith("library") else "upload"))
    allow, note = data_policy(state["dataset_meta"], source, bool(body.get("allow_data_reuse")))

    analysis = analyze_dataframe(final, dataset_name=name)
    schema = schema_mod.build_schema(final)
    metrics = V.compute_metrics(final, name)
    fingerprint = RS.data_fingerprint(original)
    repro = V.make_hash(fingerprint, steps, metrics)
    info = state["dataset_meta"].get("kaggle_info") or {}
    snapshot = {
        "dataset": {"name": name, "title": info.get("title") or name, "source": source, "slug": slug, "license": info.get("license") or "", "license_info": describe_license(info.get("license")) if info else None,
                    "description": (info.get("description") or "")[:900], "url": info.get("url") or (f"https://www.kaggle.com/datasets/{slug}" if slug else ""), "rows": int(len(original)), "cols": int(original.shape[1]),
                    "file_type": state["dataset_meta"].get("file_type"), "sample": state["dataset_meta"].get("sample"), "fingerprint": fingerprint},
        "steps": steps, "metrics": metrics, "engine": V.ENGINE,
        "headline": {"rows": int(len(final)), "columns": int(final.shape[1]), "quality_score": int(analysis["data_quality"]["score"]), "quality_label": analysis["data_quality"]["label"],
                     "missing_pct": metrics["missing_pct"], "insights": (analysis.get("insights") or [])[:4]},
        "evidence": evidence_rows(final, schema, analysis),
        "charts": default_charts(analysis.get("recommendations") or [], state["chart_configs"]),
        "findings": state["findings"][:10], "allow_data_reuse": allow, "data_note": note,
    }
    share_id = store.new_id(5)
    try:
        path = store.save_frame("shares", share_id, original)
    except ValueError as exc:
        raise ApiError(413, str(exc)) from exc
    verification = V.verify(original, steps, metrics, repro, fingerprint, name)
    user = current_user()
    doc = {"id": share_id, "owner": store.owner_of(user), "author": user["display_name"], "title": title, "description": str(body.get("description") or "").strip()[:MAX_DESCRIPTION],
           "visibility": visibility, "snapshot": snapshot, "data_path": path, "repro_hash": repro, "verification": verification, "verified_at": store.now(), "forks": 0, "views": 0,
           "created_at": store.now()}
    try:
        _save(doc)
    except ValueError as exc:
        store.delete_frame(path)
        raise ApiError(413, str(exc)) from exc
    return jsonify({"id": share_id, "url": f"/#/a/{share_id}", "verified": verification["verified"]}), 201


@bp.get("")
def explore():
    q = str(request.args.get("q") or "").strip().lower()
    sort = request.args.get("sort", "new")
    docs = store.docs().find(COLL, public=True, limit=300)
    key = {"forks": lambda d: (d.get("forks", 0), d["created_at"]), "views": lambda d: (d.get("views", 0), d["created_at"])}.get(sort, lambda d: d["created_at"])
    items = [_summary(d) for d in sorted(docs, key=key, reverse=True)]
    if q:
        items = [i for i in items if q in f"{i['title']} {i['description']} {i['dataset_title'] or ''} {i['author']}".lower()]
    if request.args.get("verified") == "1":
        items = [i for i in items if i["verified"]]
    return jsonify({"analyses": items[:60], "total": len(items)})


@bp.get("/mine")
@login_required
def mine():
    docs = sorted(store.docs().find(COLL, owner=store.owner_of(current_user())), key=lambda d: d["created_at"], reverse=True)
    return jsonify({"analyses": [_summary(d) for d in docs]})


@bp.get("/<share_id>")
def view(share_id: str):
    doc = _doc(share_id)
    doc["views"] = doc.get("views", 0) + 1
    _save(doc)
    return jsonify({"analysis": public(doc)})


@bp.put("/<share_id>")
@login_required
def update(share_id: str):
    doc = _doc(share_id)
    if not _is_owner(doc):
        raise ApiError(403, "Only the author can edit this analysis.")
    body = request.get_json(silent=True) or {}
    doc["title"] = str(body.get("title") or doc["title"]).strip()[:MAX_TITLE] or doc["title"]
    doc["description"] = str(body.get("description") if body.get("description") is not None else doc["description"]).strip()[:MAX_DESCRIPTION]
    if body.get("visibility") in {"public", "unlisted"}:
        doc["visibility"] = body["visibility"]
    _save(doc)
    return jsonify({"analysis": public(doc)})


@bp.delete("/<share_id>")
@login_required
def unpublish(share_id: str):
    doc = _doc(share_id)
    if not _is_owner(doc):
        raise ApiError(403, "Only the author can remove this analysis.")
    store.docs().delete(COLL, share_id)
    store.delete_frame(doc["data_path"])
    workspace_cache.put(f"share::{share_id}", pd.DataFrame())  # drop the cached copy
    return jsonify({"ok": True})


# ============================================================================ verify, fork, charts, downloads

@bp.post("/<share_id>/verify")
def reverify(share_id: str):
    doc = _doc(share_id)
    now = time.time()
    if now - _last_verify.get(share_id, 0) < VERIFY_COOLDOWN and doc.get("verification"):
        return jsonify({"verification": doc["verification"], "cached": True})
    _last_verify[share_id] = now
    snap = doc["snapshot"]
    original = store.load_frame(doc["data_path"])
    result = V.verify(original, snap["steps"], snap["metrics"], doc["repro_hash"], snap["dataset"]["fingerprint"], snap["dataset"]["name"])
    doc["verification"], doc["verified_at"] = result, store.now()
    _save(doc)
    return jsonify({"verification": result, "cached": False})


@bp.get("/<share_id>/verification")
def verification_detail(share_id: str):
    return jsonify({"verification": _doc(share_id).get("verification")})


@bp.post("/<share_id>/fork")
@login_required
def fork(share_id: str):
    doc = _doc(share_id)
    snap = doc["snapshot"]
    if not snap["allow_data_reuse"]:
        raise ApiError(403, f"This analysis cannot be forked. {snap.get('data_note') or ''}".strip())
    original = store.load_frame(doc["data_path"])
    state = {"findings": snap["findings"], "transform_log": snap["steps"], "chart_configs": [{"label": c.get("title") or c.get("type"), **{k: v for k, v in c.items() if k != "title"}} for c in snap["charts"]],
             "dataset_meta": {"kaggle_info": {"title": snap["dataset"]["title"], "license": snap["dataset"]["license"], "description": snap["dataset"]["description"], "url": snap["dataset"]["url"]},
                              "file_type": snap["dataset"].get("file_type"), "sample": snap["dataset"].get("sample"), "derived_from": {"share": share_id, "title": doc["title"], "author": doc["author"]}},
             "notes": f"Forked from '{doc['title']}' by {doc['author']}."}
    project = projects_mod.create_project_from_frames(store.owner_of(current_user()), f"Fork of {doc['title']}"[:120], snap["dataset"]["name"], original, state,
                                                      source=snap["dataset"]["source"] or "upload", slug=snap["dataset"].get("slug"))
    doc["forks"] = doc.get("forks", 0) + 1
    _save(doc)
    return jsonify({"project_id": project["id"]}), 201


@bp.get("/<share_id>/chart/<int:index>")
def chart(share_id: str, index: int):
    doc = _doc(share_id)
    snap = doc["snapshot"]
    if not 0 <= index < len(snap["charts"]):
        raise ApiError(404, "No such chart.")
    _, final, _ = _final_frame(doc)
    schema = schema_mod.build_schema(final)
    return jsonify(render_chart(final, snap["charts"][index], schema))


def _safe_name(doc: dict[str, Any]) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in doc["title"]).strip()[:80] or doc["id"]


@bp.get("/<share_id>/report.pdf")
def report(share_id: str):
    doc = _doc(share_id)
    snap = doc["snapshot"]
    _, final, steps = _final_frame(doc)
    info = snap["dataset"]
    dataset_info = {"dataset_id": info.get("slug") or "", "slug": info.get("slug"), "file_type": info.get("file_type"), "sample": info.get("sample"),
                    "kaggle_info": {"title": info["title"], "license": info["license"], "description": info["description"]}, "license_info": info.get("license_info")}
    pdf = build_report_pdf(info["name"], final, {"sample": info.get("sample") or {}}, info["description"] or None, [], snap["findings"], dataset_info=dataset_info, transform_log=steps,
                           options={"title": doc["title"], "author": doc["author"], "detail": "standard", "notes": doc["description"],
                                    "footer": f"Published analysis {share_id} - reproducibility hash {doc['repro_hash']}"})
    return Response(pdf, mimetype="application/pdf", headers={"Content-Disposition": f'attachment; filename="{_safe_name(doc)}.pdf"'})


@bp.get("/<share_id>/notebook.ipynb")
def notebook(share_id: str):
    doc = _doc(share_id)
    snap = doc["snapshot"]
    original, final, steps = _final_frame(doc)
    info = snap["dataset"]
    dataset = {"name": info["name"], "dataset_id": info.get("slug") or "", "kaggle_info": {"title": info["title"], "license": info["license"], "description": info["description"], "url": info["url"]},
               "sample": info.get("sample"), "file_type": info.get("file_type")}
    nb = build_notebook(doc["title"], snap["findings"], analyze_dataframe(final, dataset_name=info["name"]), df=final, original_df=original, transform_log=steps, dataset=dataset, charts=snap["charts"],
                        options={"author": doc["author"], "notes": f"Published analysis {share_id} - reproducibility hash {doc['repro_hash']}. {doc['description']}".strip(),
                                 "embed_rows": 10000 if snap["allow_data_reuse"] else 0})
    return Response(json.dumps(nb, indent=1), mimetype="application/json", headers={"Content-Disposition": f'attachment; filename="{_safe_name(doc)}.ipynb"'})
