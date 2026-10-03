"""API tests through Flask's test client: no server, no network, no Kaggle credentials."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import main  # noqa: E402
import workspace_cache  # noqa: E402


@pytest.fixture
def client():
    main.app.config["TESTING"] = True
    return main.app.test_client()


CSV = "country,year,gdp,group\n" + "\n".join(
    f"{c},{y},{100 + i * 3 + j},{'a' if i % 2 else 'b'}"
    for i, c in enumerate(["France", "Germany", "Japan", "Brazil", "India", "Kenya"])
    for j, y in enumerate(range(2000, 2012))
)


def upload(client):
    response = client.post("/datasets/upload", data={"file": (io.BytesIO(CSV.encode()), "econ.csv")},
                           content_type="multipart/form-data")
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_root_serves_frontend(client):
    response = client.get("/")
    assert response.status_code == 200 and b"DataScopia" in response.data


def test_upload_returns_schema_and_recommendations(client):
    body = upload(client)
    analysis = body["analysis"]
    assert analysis["shape"] == {"rows": 72, "columns": 4}
    assert analysis["schema"]["geo"]["country"]["column"] == "country"
    assert analysis["schema"]["time"]["time_column"] == "year"
    assert {r["type"] for r in analysis["recommendations"]} >= {"timeseries", "map"}
    assert analysis["data_quality"]["score"] >= 0
    assert body["dataset"]["dataset_id"].startswith("upload-")


def test_upload_requires_file(client):
    assert client.post("/datasets/upload", data={}, content_type="multipart/form-data").status_code == 400


def test_upload_over_the_size_cap_is_rejected_cleanly(client):
    # MAX_CONTENT_LENGTH is set at app-creation time from DATASCOPIA_MAX_UPLOAD_MB; temporarily
    # shrink it rather than actually uploading 150MB+ to prove the same behavior cheaply.
    original = main.app.config["MAX_CONTENT_LENGTH"]
    main.app.config["MAX_CONTENT_LENGTH"] = 100
    try:
        oversized = io.BytesIO(b"a,b\n" + b"1,2\n" * 50)
        response = client.post("/datasets/upload", data={"file": (oversized, "big.csv")}, content_type="multipart/form-data")
        assert response.status_code == 413
    finally:
        main.app.config["MAX_CONTENT_LENGTH"] = original


def test_charts_render_all_recommendations(client):
    body = upload(client)
    dataset_id = body["dataset"]["dataset_id"]
    for rec in body["analysis"]["recommendations"]:
        response = client.post("/charts/render", json={"dataset_id": dataset_id, "config": rec})
        assert response.status_code == 200
        chart = response.get_json()
        assert "data" in chart and "layout" in chart
        assert not chart["meta"].get("error"), (rec, chart["meta"])
        json.dumps(chart, allow_nan=False)  # strict JSON: no NaN/Infinity reaches the browser


def test_chart_render_unknown_dataset(client):
    assert client.post("/charts/render", json={"dataset_id": "nope", "config": {"type": "bar"}}).status_code == 404


def test_chart_render_bad_config_is_a_message_not_a_500(client):
    dataset_id = upload(client)["dataset"]["dataset_id"]
    response = client.post("/charts/render", json={"dataset_id": dataset_id, "config": {"type": "scatter", "x": "nope"}})
    assert response.status_code == 200
    assert response.get_json()["meta"].get("error")


def test_transform_returns_code_and_reset_restores(client):
    dataset_id = upload(client)["dataset"]["dataset_id"]
    response = client.post("/datasets/transform", json={"dataset_id": dataset_id, "type": "filter",
                                                        "params": {"column": "year", "comparator": ">=", "value": "2006"}})
    body = response.get_json()
    assert response.status_code == 200
    assert body["shape"]["rows"] == 36 and "df = df[" in body["code"]
    reset = client.post("/datasets/transform/reset", json={"dataset_id": dataset_id}).get_json()
    assert reset["shape"]["rows"] == 72


def test_transform_validation(client):
    assert client.post("/datasets/transform", json={}).status_code == 400
    dataset_id = upload(client)["dataset"]["dataset_id"]
    assert client.post("/datasets/transform", json={"dataset_id": dataset_id, "type": "nope"}).status_code == 400


def test_question_endpoint(client):
    body = upload(client)
    response = client.post("/analysis/question", json={"question": "What is the average gdp?",
                                                       "dataset_id": body["dataset"]["dataset_id"],
                                                       "analysis": body["analysis"]})
    assert response.status_code == 200 and response.get_json()["answer"]
    assert client.post("/analysis/question", json={"question": ""}).status_code == 400


def test_notebook_endpoint_replays_transforms(client):
    import nbformat

    body = upload(client)
    dataset_id = body["dataset"]["dataset_id"]
    step = {"type": "filter", "params": {"column": "year", "comparator": ">=", "value": "2006"}}
    client.post("/datasets/transform", json={"dataset_id": dataset_id, **step})
    response = client.post("/analysis/notebook", json={"dataset_id": dataset_id, "analysis": body["analysis"],
                                                       "dataset": body["dataset"], "transform_log": [step],
                                                       "charts": [{"type": "line", "x": "year", "y": "gdp"}]})
    assert response.status_code == 200
    nb = response.get_json()
    nbformat.validate(nbformat.from_dict(nb))
    source = " ".join("".join(c["source"]) for c in nb["cells"])
    assert "df = df[df['year'] >= 2006]" in source
    assert nb["metadata"]["datascopia"]["transform_steps"] == 1


def test_quality_report_download(client):
    body = upload(client)
    response = client.post("/analysis/quality-report", json={"analysis": body["analysis"], "dataset_name": "econ"})
    assert response.status_code == 200 and response.mimetype == "text/markdown"
    assert b"Data quality report" in response.data
    assert client.post("/analysis/quality-report", json={}).status_code == 400


def test_report_pdf(client):
    body = upload(client)
    response = client.post("/analysis/report", json={"dataset_id": body["dataset"]["dataset_id"], "dataset_name": "econ",
                                                     "analysis": body["analysis"]})
    assert response.status_code == 200 and response.data.startswith(b"%PDF")


def test_demo_dataset(client):
    body = client.get("/datasets/demo").get_json()
    assert body["dataset"]["dataset_id"] == "demo"
    # the "preview" key is what the frontend reads to know which columns exist and build chart
    # slots (fields = Object.keys(analysis.preview[0])) - without it "Load demo data" shows the
    # Visualization page as if no dataset were loaded at all, even though analysis itself is fine.
    assert body["analysis"]["preview"] and len(body["analysis"]["preview"]) == body["analysis"]["shape"]["rows"]
    assert workspace_cache.get("demo") is not None


def test_search_without_query(client):
    assert client.get("/datasets/search").get_json() == {"results": []}


def test_license_endpoint(client, monkeypatch):
    monkeypatch.setattr(main, "get_dataset_description", lambda slug: {"title": "T", "license": "CC BY-NC-SA 4.0", "url": "https://x"})
    body = client.get("/datasets/license?slug=owner/name").get_json()
    assert body["license"]["level"] == "restricted" and body["title"] == "T"
    assert client.get("/datasets/license?slug=bad").status_code == 400


def test_license_endpoint_survives_kaggle_failure(client, monkeypatch):
    def boom(slug):
        raise RuntimeError("offline")
    monkeypatch.setattr(main, "get_dataset_description", boom)
    body = client.get("/datasets/license?slug=owner/name").get_json()
    assert body["license"]["level"] == "unknown" and "could not be fetched" in body["license"]["summary"]


def test_hf_search_without_query(client):
    assert client.get("/datasets/hf/search").get_json() == {"results": []}


def test_hf_search_delegates_to_provider(client, monkeypatch):
    monkeypatch.setattr(main, "search_hf_datasets", lambda q, max_results=60: [{"id": "hf:org/name", "title": "org/name"}])
    body = client.get("/datasets/hf/search?query=titanic").get_json()
    assert body["results"] == [{"id": "hf:org/name", "title": "org/name"}]


def test_hf_license_endpoint(client, monkeypatch):
    monkeypatch.setattr(main, "get_hf_dataset_description", lambda ref: {"title": "T", "license": "mit", "url": "https://x"})
    body = client.get("/datasets/hf/license?ref=org/name").get_json()
    assert body["license"]["level"] == "open" and body["title"] == "T"
    assert client.get("/datasets/hf/license?ref=bad").status_code == 400
    # accepts either the bare ref or the "hf:" namespaced form the frontend stores it as
    body2 = client.get("/datasets/hf/license?ref=hf:org/name").get_json()
    assert body2["ref"] == "org/name"


def _fake_hf_dataset(ref: str) -> dict:
    import pandas as pd

    df = pd.DataFrame({"country": ["France", "Germany", "Japan"], "value": [1, 2, 3]})
    return {"path": None, "name": ref.replace("/", "-"), "file_type": "hf", "shape": {"rows": 3, "columns": 2},
            "columns": list(df.columns), "dataframe": df,
            "sample": {"file_name": f"{ref.replace('/', '-')}.parquet", "config": "default", "split": "train", "is_sample": False, "sample_rows": 3, "estimated_total_rows": 3, "total_bytes": None}}


def test_hf_analyze_caches_under_namespaced_dataset_id(client, monkeypatch):
    monkeypatch.setattr(main, "load_hf_dataset_sample", lambda ref, max_rows=2000: _fake_hf_dataset(ref))
    monkeypatch.setattr(main, "get_hf_dataset_description", lambda ref: {"title": "Titanic", "license": "mit", "description": "d", "url": "https://x"})
    response = client.post("/datasets/hf/analyze", json={"ref": "phihung/titanic"})
    assert response.status_code == 200, response.get_data(as_text=True)
    body = response.get_json()
    assert body["dataset"]["dataset_id"] == "hf:phihung/titanic"
    assert body["dataset"]["hf_info"]["title"] == "Titanic"
    assert body["dataset"]["license_info"]["level"] == "open"
    assert body["analysis"]["shape"] == {"rows": 3, "columns": 2}
    # a Kaggle slug with the same "owner/name" shape must not collide with the hf: namespaced id
    assert workspace_cache.get("phihung/titanic") is None
    assert workspace_cache.get("hf:phihung/titanic") is not None


def test_hf_analyze_requires_a_ref(client):
    assert client.post("/datasets/hf/analyze", json={}).status_code == 400
    assert client.post("/datasets/hf/analyze", json={"ref": "no-slash"}).status_code == 400


def test_hf_analyze_reports_provider_errors_as_404(client, monkeypatch):
    from dataset.huggingface import HFError

    def boom(ref, max_rows=2000):
        raise HFError(f"'{ref}' is private or gated.")
    monkeypatch.setattr(main, "load_hf_dataset_sample", boom)
    response = client.post("/datasets/hf/analyze", json={"ref": "org/gated-dataset"})
    assert response.status_code == 404 and "gated" in response.get_json()["detail"]


def test_hf_analyze_full_uses_namespaced_id_too(client, monkeypatch):
    monkeypatch.setattr(main, "download_hf_dataset_full", lambda ref: _fake_hf_dataset(ref))
    monkeypatch.setattr(main, "get_hf_dataset_description", lambda ref: {"title": "Titanic", "license": "", "description": "", "url": "https://x"})
    body = client.post("/datasets/hf/analyze-full", json={"ref": "hf:phihung/titanic"}).get_json()
    assert body["dataset"]["dataset_id"] == "hf:phihung/titanic"


def test_frontend_is_a_multi_page_app_with_static_assets(client):
    html = client.get("/").get_data(as_text=True)
    for route in ("datasets", "overview", "visualization", "ask", "export", "changelog"):
        assert f'data-page="{route}"' in html and f'href="#/{route}"' in html
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/styles.css").status_code == 200


def test_changelog_endpoint(client):
    body = client.get("/api/changelog").get_json()
    releases = body["releases"]
    assert body["latest"] == releases[0]["version"]
    assert body["stats"]["releases"] == len(releases) and body["stats"]["changes"] > len(releases)
    assert {r["tag"] for r in releases} == set(body["tags"])
    assert all(r["items"] and r["date"] and r["name"] for r in releases)


def test_report_honours_filename_and_options(client):
    body = upload(client)
    response = client.post("/analysis/report", json={"dataset_id": body["dataset"]["dataset_id"], "dataset_name": "econ", "analysis": body["analysis"],
                                                     "filename": "../../My Board Report!.pdf", "options": {"title": "Board Pack", "detail": "brief", "page_size": "a4"}})
    assert response.status_code == 200 and response.data.startswith(b"%PDF")
    disposition = response.headers["Content-Disposition"]
    assert 'filename="My Board Report.pdf"' in disposition or "My_Board_Report" in disposition
    assert ".." not in disposition and "/" not in disposition.split("filename=")[1]


def test_notebook_honours_options_and_filename(client):
    import nbformat

    body = upload(client)
    response = client.post("/analysis/notebook", json={"dataset_id": body["dataset"]["dataset_id"], "analysis": body["analysis"], "dataset": body["dataset"], "filename": "my analysis",
                                                       "options": {"author": "Dr. Ada", "notes": "For class", "embed_rows": 0, "sections": {"modelling": False, "statistics": False, "quality": False},
                                                                   "chart_source": "recommended", "max_charts": 2}})
    nb = response.get_json()
    nbformat.validate(nbformat.from_dict(nb))
    text = " ".join("".join(c["source"]) for c in nb["cells"])
    assert "Dr. Ada" in text and "For class" in text and "_SNAPSHOT_GZ_B64" not in text and "No data is embedded" in text
    assert "Predictive modelling" not in text and "Statistical analysis" not in text and "Data quality before cleaning" not in text
    headings = [c["source"][0] for c in nb["cells"] if c["cell_type"] == "markdown" and c["source"][0].startswith("## ")]
    numbers = [int(h.split(".")[0].replace("## ", "")) for h in headings if h.split(".")[0].replace("## ", "").isdigit()]
    assert numbers == list(range(1, len(numbers) + 1))  # renumbered without gaps
    assert response.headers["X-Filename"] == "my analysis.ipynb"


def test_quality_export_formats_and_filters(client):
    body = upload(client)
    payload = {"analysis": {"data_quality": body["analysis"]["data_quality"], "dataset_name": "econ"}, "dataset_name": "econ"}
    md = client.post("/analysis/quality-report", json={**payload, "options": {"title": "My Audit", "include_why": False}, "filename": "audit"})
    assert md.status_code == 200 and b"# My Audit" in md.data and b"Why it matters" not in md.data and md.headers["X-Filename"] == "audit.md"
    js = client.post("/analysis/quality-report", json={**payload, "options": {"format": "json", "severities": ["high"]}})
    data = js.get_json()
    assert js.mimetype == "application/json" and all(i["severity"] == "high" for i in data["issues"]) and js.headers["X-Filename"].endswith(".json")
    csv_response = client.post("/analysis/quality-report", json={**payload, "options": {"format": "csv"}})
    assert csv_response.mimetype == "text/csv" and csv_response.data.startswith(b"severity,category")
