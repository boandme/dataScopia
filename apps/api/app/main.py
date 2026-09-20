from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS_SRC = PROJECT_ROOT / "packages" / "analysis" / "src"
DATASET_SRC = PROJECT_ROOT / "packages" / "dataset" / "src"
for path in (ANALYSIS_SRC, DATASET_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from analysis.profiler import analyze_dataframe
from analysis.notebook_export import build_notebook
from dataset.kaggle_search import (
    download_kaggle_dataset,
    kaggle_credential_status,
    search_kaggle_datasets,
)
from dataset.loader import load_dataset

app = FastAPI(title=os.getenv("APP_TITLE", "Kaggle Dataset Analysis Workspace"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _serializable_dataset(dataset: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in dataset.items() if key != "dataframe"}


def _analysis_with_preview(df: pd.DataFrame, dataset_name: str) -> dict[str, object]:
    analysis = analyze_dataframe(df, dataset_name=dataset_name)
    analysis["preview"] = json.loads(df.head(300).to_json(orient="records", date_format="iso"))
    return analysis


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/kaggle")
def kaggle_auth_status() -> dict[str, str | bool]:
    return kaggle_credential_status()


@app.get("/datasets/search")
def search_datasets(query: str = "", limit: int = 10) -> dict[str, object]:
    if not query:
        return {"results": []}
    return {"results": search_kaggle_datasets(query, max_results=limit)}


@app.post("/datasets/load")
def load_dataset_from_path(payload: dict[str, str]) -> dict[str, object]:
    path = payload.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="A dataset path is required.")

    try:
        dataset = load_dataset(path)
        df = dataset["dataframe"]
        analysis = _analysis_with_preview(df, dataset["name"])
        return {"dataset": _serializable_dataset(dataset), "analysis": analysis}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/datasets/download")
def download_dataset_from_kaggle(payload: dict[str, str]) -> dict[str, object]:
    slug = payload.get("slug") or payload.get("dataset_slug")
    if not slug:
        raise HTTPException(status_code=400, detail="A Kaggle dataset slug is required.")

    try:
        dataset_path = download_kaggle_dataset(slug)
        dataset = load_dataset(dataset_path)
        df = dataset["dataframe"]
        analysis = _analysis_with_preview(df, dataset["name"])
        return {
            "dataset": _serializable_dataset(dataset),
            "analysis": analysis,
            "local_path": dataset_path,
        }
    except Exception as exc:  # pragma: no cover
        detail = str(exc)
        status_code = 404 if "Resource not found" in detail or "404" in detail else 500
        raise HTTPException(status_code=status_code, detail=detail) from exc


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, object]:
    try:
        contents = await file.read()
        temp_dir = Path("/tmp") / "kaggle_dataset_analysis"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / file.filename
        temp_path.write_bytes(contents)

        dataset = load_dataset(temp_path)
        df = dataset["dataframe"]
        analysis = _analysis_with_preview(df, dataset["name"])
        return {"dataset": _serializable_dataset(dataset), "analysis": analysis}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analysis/notebook")
def export_analysis_notebook(payload: dict[str, object]) -> dict[str, object]:
    analysis = payload.get("analysis") or {}
    title = str(payload.get("title") or analysis.get("dataset_name") or "Dataset analysis")
    findings = payload.get("findings") or []
    return build_notebook(title, findings, analysis)


@app.post("/analysis/question")
def answer_analysis_question(payload: dict[str, object]) -> dict[str, object]:
    question = str(payload.get("question") or "").strip()
    analysis = payload.get("analysis") or {}
    if not question:
        raise HTTPException(status_code=400, detail="A research question is required.")
    snapshot = analysis.get("research_snapshot", {})
    answer = (
        f"The active dataset contains {snapshot.get('rows', 0)} observations and "
        f"{snapshot.get('columns', 0)} variables. Based on the current profile, "
        f"{snapshot.get('notable_relationships', 0)} notable numeric relationships and "
        f"{snapshot.get('data_quality_issues', 0)} data-quality issues were detected."
    )
    return {"question": question, "answer": answer, "evidence": analysis.get("insights", [])}


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
    return {"dataset": {"name": "demo_dataset", "shape": {"rows": len(df), "columns": len(df.columns)}}, "analysis": analyze_dataframe(df, dataset_name="demo_dataset")}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Kaggle Dataset Analysis Workspace API"}
