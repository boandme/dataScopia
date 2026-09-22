from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS_SRC = PROJECT_ROOT / "packages" / "analysis" / "src"
DATASET_SRC = PROJECT_ROOT / "packages" / "dataset" / "src"
APP_DIR = Path(__file__).resolve().parent
for path in (ANALYSIS_SRC, DATASET_SRC, APP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from analysis.profiler import analyze_dataframe
from analysis.notebook_export import build_notebook
from analysis.qa_engine import answer_question
from analysis.report import build_report_pdf
from analysis.transforms import apply_transform, TransformError
from dataset.kaggle_search import (
    download_kaggle_dataset,
    get_dataset_description,
    kaggle_credential_status,
    search_kaggle_datasets,
)
from dataset.descriptions import extract_column_descriptions, infer_column_description
from dataset.loader import load_dataset, load_dataframe_from_bytes
from dataset.streaming import DEFAULT_SAMPLE_BYTES, load_dataset_sample
import workspace_cache

app = FastAPI(title=os.getenv("APP_TITLE", "DataScopia"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# How much of a dataset file to stream for sample-based analysis before analysis falls back to
# asking the user to run a full analysis or full download. Configurable since it trades off
# analysis fidelity against bandwidth/time for very large files.
SAMPLE_BYTES = int(float(os.getenv("ANALYSIS_SAMPLE_MB", "5")) * 1024 * 1024) or DEFAULT_SAMPLE_BYTES


def _serializable_dataset(dataset: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in dataset.items() if key not in {"dataframe", "path"}}


def _attach_column_descriptions(dataset: dict[str, object], df: pd.DataFrame, documented: dict[str, str] | None = None) -> None:
    """Attach both documented (from the dataset's own text, when available) and inferred
    (deterministic, name-based heuristic) column descriptions, so the UI always has something to
    show for every column without ever presenting a guess as if it were the author's own words."""
    documented = documented or {}
    dataset["column_descriptions"] = documented
    dataset["inferred_column_descriptions"] = {
        column: infer_column_description(column, df[column]) for column in df.columns if column not in documented
    }


def _attach_kaggle_description(dataset: dict[str, object], df: pd.DataFrame, slug: str) -> None:
    """Best-effort: fetch the dataset's Kaggle description and extract any column meanings the
    author wrote into it, then fill in the rest with name-based inference. Never fails the
    request - a bad/missing description just means fewer columns get a documented description."""
    try:
        info = get_dataset_description(slug)
    except Exception:
        info = {}
    dataset["kaggle_info"] = info
    documented = extract_column_descriptions(info.get("description", ""), list(df.columns))
    _attach_column_descriptions(dataset, df, documented)


def _cache_dataset(dataset: dict[str, object], df: pd.DataFrame, dataset_id: str) -> None:
    """Cache the dataframe server-side (in-memory, TTL-bounded) so follow-up questions about
    this dataset can compute real answers without re-fetching it from Kaggle each time."""
    workspace_cache.put(dataset_id, df)
    dataset["dataset_id"] = dataset_id


def _analysis_with_preview(df: pd.DataFrame, dataset_name: str, sample_info: dict[str, object] | None = None) -> dict[str, object]:
    analysis = analyze_dataframe(df, dataset_name=dataset_name, sample_info=sample_info)
    analysis["preview"] = json.loads(df.head(300).to_json(orient="records", date_format="iso"))
    return analysis


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/kaggle")
def kaggle_auth_status() -> dict[str, str | bool]:
    return kaggle_credential_status()


@app.get("/datasets/search")
def search_datasets(query: str = "", limit: int = 60) -> dict[str, object]:
    if not query:
        return {"results": []}
    limit = max(1, min(limit, 300))
    return {"results": search_kaggle_datasets(query, max_results=limit)}


@app.post("/datasets/load")
def load_dataset_from_path(payload: dict[str, str]) -> dict[str, object]:
    path = payload.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="A dataset path is required.")

    try:
        dataset = load_dataset(path)
        df = dataset["dataframe"]
        _cache_dataset(dataset, df, dataset_id=f"local-{dataset['name']}")
        _attach_column_descriptions(dataset, df)
        analysis = _analysis_with_preview(df, dataset["name"])
        return {"dataset": _serializable_dataset(dataset), "analysis": analysis}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/datasets/analyze")
def analyze_dataset_from_kaggle(payload: dict[str, object]) -> dict[str, object]:
    """Profile a Kaggle dataset by streaming only a bounded sample of its primary data file.

    This never downloads the dataset in full: for CSV/TSV it stops reading once enough bytes
    have been streamed, and for Parquet it reads only as many row groups as needed via HTTP
    range requests. Use /datasets/analyze-full or /datasets/download-file when a complete copy
    is actually required.
    """
    slug = payload.get("slug") or payload.get("dataset_slug")
    if not slug:
        raise HTTPException(status_code=400, detail="A Kaggle dataset slug is required.")
    sample_bytes = int(payload.get("sample_bytes") or SAMPLE_BYTES)

    try:
        dataset = load_dataset_sample(str(slug), max_sample_bytes=sample_bytes)
        df = dataset["dataframe"]
        _cache_dataset(dataset, df, dataset_id=str(slug))
        _attach_kaggle_description(dataset, df, str(slug))
        analysis = _analysis_with_preview(df, dataset["name"], sample_info=dataset.get("sample"))
        return {"dataset": _serializable_dataset(dataset), "analysis": analysis}
    except Exception as exc:  # pragma: no cover
        detail = str(exc)
        status_code = 404 if "Resource not found" in detail or "404" in detail else 500
        raise HTTPException(status_code=status_code, detail=detail) from exc


@app.post("/datasets/analyze-full")
def analyze_dataset_full(payload: dict[str, str]) -> dict[str, object]:
    """Download a Kaggle dataset in full to a temporary server workspace, analyze it, then
    delete the workspace. Opt-in for cases where sample-based analysis isn't precise enough;
    never required for the default research snapshot."""
    slug = payload.get("slug") or payload.get("dataset_slug")
    if not slug:
        raise HTTPException(status_code=400, detail="A Kaggle dataset slug is required.")

    try:
        with tempfile.TemporaryDirectory(prefix="kaggle-analysis-") as workspace:
            dataset_path = download_kaggle_dataset(slug, destination=workspace)
            dataset = load_dataset(dataset_path)
            df = dataset["dataframe"]
            _cache_dataset(dataset, df, dataset_id=str(slug))
            _attach_kaggle_description(dataset, df, str(slug))
            analysis = _analysis_with_preview(df, dataset["name"])
            return {"dataset": _serializable_dataset(dataset), "analysis": analysis}
    except Exception as exc:  # pragma: no cover
        detail = str(exc)
        status_code = 404 if "Resource not found" in detail or "404" in detail else 500
        raise HTTPException(status_code=status_code, detail=detail) from exc


@app.post("/datasets/download-file")
def download_full_dataset(payload: dict[str, str], background_tasks: BackgroundTasks) -> FileResponse:
    slug = payload.get("slug") or payload.get("dataset_slug")
    if not slug:
        raise HTTPException(status_code=400, detail="A Kaggle dataset slug is required.")

    workspace = tempfile.mkdtemp(prefix="kaggle-export-")
    try:
        dataset_path = download_kaggle_dataset(slug, destination=workspace)
        archive_base = str(Path(workspace) / Path(slug.replace("/", "-")))
        archive_path = Path(shutil.make_archive(archive_base, "zip", dataset_path))
        background_tasks.add_task(shutil.rmtree, workspace, ignore_errors=True)
        return FileResponse(archive_path, filename=f"{slug.replace('/', '-')}.zip", media_type="application/zip", background=background_tasks)
    except Exception as exc:
        shutil.rmtree(workspace, ignore_errors=True)
        detail = str(exc)
        status_code = 404 if "Resource not found" in detail or "404" in detail else 500
        raise HTTPException(status_code=status_code, detail=detail) from exc


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, object]:
    """Analyze a file the user picks from their own computer. Parsed entirely in memory — never
    written to the server's disk — since the whole point of an upload is a file the user already
    has locally; the server shouldn't need its own copy to profile it."""
    filename = file.filename or "uploaded_dataset.csv"
    try:
        contents = await file.read()
        df = load_dataframe_from_bytes(filename, contents)
        name = Path(filename).stem
        suffix = Path(filename).suffix.lstrip(".").lower()
        dataset: dict[str, object] = {
            "name": name,
            "file_type": suffix,
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns": list(df.columns),
            "dataframe": df,
        }
        _cache_dataset(dataset, df, dataset_id=f"upload-{uuid.uuid4().hex[:10]}")
        _attach_column_descriptions(dataset, df)
        analysis = _analysis_with_preview(df, name)
        return {"dataset": _serializable_dataset(dataset), "analysis": analysis}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _original_cache_key(dataset_id: str) -> str:
    return f"{dataset_id}::original"


@app.post("/datasets/transform")
def transform_dataset(payload: dict[str, object]) -> dict[str, object]:
    """Apply one filter/clean/transform step to the cached dataframe for `dataset_id`, in place
    of the working copy - so it composes with further transforms, charts, and Q&A. The very
    first pre-transform snapshot is kept under a shadow cache key so /datasets/transform/reset
    can undo everything in one step."""
    dataset_id = payload.get("dataset_id")
    op_type = payload.get("type")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    dataset_name = str(payload.get("dataset_name") or "dataset")
    if not dataset_id or not op_type:
        raise HTTPException(status_code=400, detail="dataset_id and type are required.")

    df = workspace_cache.get(str(dataset_id))
    if df is None:
        raise HTTPException(
            status_code=400,
            detail="No cached data found for this dataset in this session. Re-analyze it first.",
        )

    original_key = _original_cache_key(str(dataset_id))
    if workspace_cache.get(original_key) is None:
        workspace_cache.put(original_key, df)

    try:
        new_df, description = apply_transform(df, str(op_type), params)
    except TransformError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    workspace_cache.put(str(dataset_id), new_df)
    analysis = _analysis_with_preview(new_df, dataset_name)
    return {
        "analysis": analysis,
        "description": description,
        "shape": {"rows": int(new_df.shape[0]), "columns": int(new_df.shape[1])},
    }


@app.post("/datasets/transform/reset")
def reset_dataset_transform(payload: dict[str, object]) -> dict[str, object]:
    """Restore the dataset to its state before the first transform was applied this session."""
    dataset_id = payload.get("dataset_id")
    dataset_name = str(payload.get("dataset_name") or "dataset")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required.")

    original_key = _original_cache_key(str(dataset_id))
    original_df = workspace_cache.get(original_key)
    if original_df is None:
        raise HTTPException(status_code=400, detail="No original snapshot to reset to for this dataset.")

    workspace_cache.put(str(dataset_id), original_df)
    analysis = _analysis_with_preview(original_df, dataset_name)
    return {
        "analysis": analysis,
        "shape": {"rows": int(original_df.shape[0]), "columns": int(original_df.shape[1])},
    }


@app.post("/analysis/notebook")
def export_analysis_notebook(payload: dict[str, object]) -> dict[str, object]:
    analysis = payload.get("analysis") or {}
    title = str(payload.get("title") or analysis.get("dataset_name") or "Dataset analysis")
    findings = payload.get("findings") or []
    return build_notebook(title, findings, analysis)


@app.post("/analysis/question")
def answer_analysis_question(payload: dict[str, object]) -> dict[str, object]:
    """Answer a question with a real computation over the dataset, not a canned summary.

    Uses the server-side cached dataframe for the dataset (populated by the analyze/upload/load
    endpoints) when available, so repeated questions don't require re-fetching from Kaggle. Falls
    back to the up-to-300-row preview already known to the frontend if the cache has expired or
    was never populated (e.g. an old session), clearly flagging the lower-fidelity answer.

    This is deterministic/rule-based today (no external LLM call). The engine in
    analysis.qa_engine exposes a stable (question, dataframe) -> result contract so it can be
    swapped for an LLM-backed implementation later without changing this endpoint or the frontend.
    """
    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="A research question is required.")

    analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
    dataset_id = payload.get("dataset_id")
    snapshot = analysis.get("research_snapshot", {})

    df = workspace_cache.get(str(dataset_id)) if dataset_id else None
    used_preview_fallback = False
    if df is None:
        preview = analysis.get("preview")
        if preview:
            df = pd.DataFrame(preview)
            used_preview_fallback = True

    if df is None or df.empty:
        raise HTTPException(
            status_code=400,
            detail="No dataset data is available to answer this question. Analyze a dataset first.",
        )

    result = answer_question(question, df, snapshot=snapshot)
    answer_text = result.answer
    if used_preview_fallback and result.computed:
        answer_text += " (Computed from the loaded preview rows only; re-analyze for a full-sample answer.)"

    return {
        "question": question,
        "answer": answer_text,
        "computed": result.computed,
        "method": result.method,
        "code": result.code,
        "value": result.value,
        "columns_used": result.columns_used,
        "evidence": analysis.get("insights", []),
    }


@app.post("/analysis/report")
def export_analysis_report(payload: dict[str, object]) -> Response:
    """Generate a professional, dataset-specific PDF research report server-side.

    Unlike a client-side dump of the on-screen summary stats, every narrative paragraph here is
    computed fresh against the full cached dataframe (real scipy correlation/significance tests,
    distribution shape, group comparisons) so the report's claims about significance and
    correlation are genuine rather than restated UI copy.
    """
    dataset_id = payload.get("dataset_id")
    dataset_name = str(payload.get("dataset_name") or "Dataset")
    analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
    charts = payload.get("charts") if isinstance(payload.get("charts"), list) else []
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []

    df = workspace_cache.get(str(dataset_id)) if dataset_id else None
    if df is None:
        preview = analysis.get("preview")
        if preview:
            df = pd.DataFrame(preview)

    if df is None or df.empty:
        raise HTTPException(
            status_code=400,
            detail="No dataset data is available to build a report. Analyze a dataset first.",
        )

    kaggle_description = None
    dataset_payload = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    kaggle_info = dataset_payload.get("kaggle_info") if isinstance(dataset_payload.get("kaggle_info"), dict) else {}
    if kaggle_info.get("description"):
        kaggle_description = str(kaggle_info["description"])

    try:
        pdf_bytes = build_report_pdf(
            dataset_name=dataset_name,
            df=df,
            analysis=analysis,
            kaggle_description=kaggle_description,
            charts=charts,
            findings=findings,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not generate the report: {exc}") from exc

    filename = f"{dataset_name.replace(' ', '_').replace('/', '-')}_report.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/datasets/demo")
def demo_dataset() -> dict[str, object]:
    df = pd.DataFrame(
        {
            "country": ["USA", "Canada", "Mexico", "France", "Germany", "Japan", "India", "Brazil"],
            "gdp_growth": [2.1, 1.8, 2.0, 1.2, 1.6, 0.9, 6.3, 2.4],
            "tourism_share": [0.12, 0.09, 0.14, 0.08, 0.06, 0.1, 0.18, 0.11],
            "population_millions": [331, 38, 128, 65, 83, 125, 1396, 216],
            "year": [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023],
        }
    )
    workspace_cache.put("demo", df)
    dataset: dict[str, object] = {"name": "demo_dataset", "dataset_id": "demo", "shape": {"rows": len(df), "columns": len(df.columns)}}
    _attach_column_descriptions(dataset, df)
    return {
        "dataset": dataset,
        "analysis": analyze_dataframe(df, dataset_name="demo_dataset"),
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "DataScopia API"}
