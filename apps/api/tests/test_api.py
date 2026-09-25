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


def test_frontend_is_a_multi_page_app_with_static_assets(client):
    html = client.get("/").get_data(as_text=True)
    for route in ("datasets", "overview", "workspace", "ask", "export"):
        assert f'data-page="{route}"' in html and f'href="#/{route}"' in html
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/styles.css").status_code == 200
