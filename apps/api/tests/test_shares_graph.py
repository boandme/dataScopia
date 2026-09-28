"""Verified sharing (publish, public view, re-verification, tamper detection, fork rules, downloads) and the
dataset graph (indexing, finding joinable datasets, preview and creating the combined dataset)."""
from __future__ import annotations

import io
import json

import pytest

import main
import store

PASSWORD = "correct horse battery"
COUNTRIES = ["France", "Germany", "Japan", "Brazil", "India", "Kenya", "Peru", "Egypt"]
CSV = "country,year,co2,trees\n" + "\n".join(f"{c},{y},{10 + i * 2 + j * 0.5},{50 - i - j}" for i, c in enumerate(COUNTRIES) for j, y in enumerate(range(2000, 2008)))


def make_client():
    main.app.config["TESTING"] = True
    return main.app.test_client()


def signup(client, email="ada@example.com", name="Ada"):
    r = client.post("/api/auth/register", json={"email": email, "password": PASSWORD, "display_name": name})
    assert r.status_code == 201
    return r.get_json()["user"]


def load(client, csv=CSV, filename="climate.csv"):
    r = client.post("/datasets/upload", data={"file": (io.BytesIO(csv.encode()), filename)}, content_type="multipart/form-data")
    assert r.status_code == 200
    return r.get_json()


def publish(client, body, **extra):
    payload = {"dataset_id": body["dataset"]["dataset_id"], "dataset_name": "climate", "dataset": body["dataset"], "title": "CO2 and forests", "description": "A test analysis.",
               "transform_log": [{"type": "filter", "params": {"column": "year", "comparator": ">=", "value": "2002"}, "description": "Kept 2002 onwards"}], **extra}
    return client.post("/api/shares", json=payload)


@pytest.fixture
def author():
    client = make_client()
    signup(client)
    return client


# ------------------------------------------------------------------ publishing

def test_publish_requires_login_title_and_live_session():
    anon = make_client()
    body = load(anon)
    assert publish(anon, body).status_code == 401
    client = make_client()
    signup(client)
    body = load(client)
    assert publish(client, body, title=" ").status_code == 400
    assert client.post("/api/shares", json={"dataset_id": "gone", "title": "x"}).status_code == 400


def test_published_analysis_is_public_verified_and_hides_raw_data(author):
    body = load(author)
    response = publish(author, body)
    assert response.status_code == 201 and response.get_json()["verified"] is True
    share_id = response.get_json()["id"]

    reader = make_client()  # not signed in
    data = reader.get(f"/api/shares/{share_id}").get_json()["analysis"]
    assert data["title"] == "CO2 and forests" and data["author"] == "Ada" and data["is_owner"] is False
    assert data["verification"]["verified"] and data["verification"]["passed"] == data["verification"]["checked"] > 10
    assert data["headline"]["rows"] == 48 and data["dataset"]["rows"] == 64  # 6 years x 8 countries after the filter vs 8 x 8 original
    assert data["steps"][0]["type"] == "filter" and data["evidence"]
    text = json.dumps(data)
    assert "Kenya" not in text or "Kenya" in json.dumps(data["evidence"])  # no row-level data leaks into the public payload
    assert "preview" not in data and "data_path" not in text and "pw_hash" not in text


def test_reverification_passes_and_tampering_is_caught(author):
    share_id = publish(author, load(author)).get_json()["id"]
    reader = make_client()
    ok = reader.post(f"/api/shares/{share_id}/verify").get_json()["verification"]
    assert ok["verified"] and ok["checks"][0]["name"] == "Original data fingerprint"

    # someone edits a published number in the database -> verification must fail on that exact metric
    doc = store.docs().get("shares", share_id)
    doc["snapshot"]["metrics"]["correlations"][0]["r"] = 0.123456
    store.docs().put("shares", share_id, doc, owner=doc["owner"], public=True)
    import shares as shares_mod

    shares_mod._last_verify.clear()
    bad = reader.post(f"/api/shares/{share_id}/verify").get_json()["verification"]
    assert bad["verified"] is False and any(not c["ok"] and c["name"].startswith("Correlation") for c in bad["checks"])
    assert reader.get(f"/api/shares/{share_id}").get_json()["analysis"]["verification"]["verified"] is False


def test_corrupted_data_fails_verification(author):
    share_id = publish(author, load(author)).get_json()["id"]
    path = store.docs().get("shares", share_id)["data_path"]

    frame = store.load_frame(path)
    frame.loc[0, "co2"] = 999.0
    frame.to_parquet(store.CONFIG["data_dir"] / path, index=False)
    import shares as shares_mod

    shares_mod._last_verify.clear()
    result = make_client().post(f"/api/shares/{share_id}/verify").get_json()["verification"]
    assert result["verified"] is False and not result["checks"][0]["ok"]


def test_explore_lists_only_public_analyses(author):
    public_id = publish(author, load(author), title="Visible one").get_json()["id"]
    hidden_id = publish(author, load(author), title="Link only", visibility="unlisted").get_json()["id"]
    reader = make_client()
    ids = [a["id"] for a in reader.get("/api/shares").get_json()["analyses"]]
    assert public_id in ids and hidden_id not in ids
    assert reader.get(f"/api/shares/{hidden_id}").status_code == 200  # unlisted still opens by link
    assert [a["id"] for a in reader.get("/api/shares?q=visible").get_json()["analyses"]] == [public_id]
    assert reader.get("/api/shares?q=zzz").get_json()["analyses"] == []
    mine = author.get("/api/shares/mine").get_json()["analyses"]
    assert {a["id"] for a in mine} == {public_id, hidden_id}


def test_only_the_author_can_edit_or_remove(author):
    share_id = publish(author, load(author)).get_json()["id"]
    stranger = make_client()
    signup(stranger, "eve@example.com", "Eve")
    assert stranger.put(f"/api/shares/{share_id}", json={"title": "Hacked"}).status_code == 403
    assert stranger.delete(f"/api/shares/{share_id}").status_code == 403
    edited = author.put(f"/api/shares/{share_id}", json={"title": "Better title", "visibility": "unlisted"}).get_json()["analysis"]
    assert edited["title"] == "Better title" and edited["visibility"] == "unlisted" and edited["is_owner"] is True
    assert author.delete(f"/api/shares/{share_id}").status_code == 200
    assert make_client().get(f"/api/shares/{share_id}").status_code == 404


# ------------------------------------------------------------------ fork + downloads

def test_fork_is_blocked_unless_the_author_allows_data_reuse(author):
    locked = publish(author, load(author)).get_json()["id"]
    reader = make_client()
    signup(reader, "bob@example.com", "Bob")
    assert reader.post(f"/api/shares/{locked}/fork").status_code == 403
    open_id = publish(author, load(author), allow_data_reuse=True).get_json()["id"]
    assert make_client().post(f"/api/shares/{open_id}/fork").status_code == 401
    forked = reader.post(f"/api/shares/{open_id}/fork")
    assert forked.status_code == 201
    project_id = forked.get_json()["project_id"]
    opened = reader.post(f"/api/projects/{project_id}/open").get_json()
    assert opened["dataset"]["shape"]["rows"] == 48  # the author's cleaning steps came along
    assert reader.get("/api/projects").get_json()["projects"][0]["name"].startswith("Fork of")
    assert make_client().get(f"/api/shares/{open_id}").get_json()["analysis"]["forks"] == 1


def test_public_charts_report_and_notebook(author):
    share_id = publish(author, load(author), allow_data_reuse=True).get_json()["id"]
    reader = make_client()
    detail = reader.get(f"/api/shares/{share_id}").get_json()["analysis"]
    assert detail["charts"]
    chart = reader.get(f"/api/shares/{share_id}/chart/0").get_json()
    assert chart["data"] and not chart["meta"].get("error")
    assert reader.get(f"/api/shares/{share_id}/chart/99").status_code == 404
    pdf = reader.get(f"/api/shares/{share_id}/report.pdf")
    assert pdf.status_code == 200 and pdf.data.startswith(b"%PDF")
    nb = json.loads(reader.get(f"/api/shares/{share_id}/notebook.ipynb").data)
    text = " ".join("".join(c["source"]) for c in nb["cells"])
    assert "_SNAPSHOT_GZ_B64" in text and "reproducibility hash" in text  # data reuse allowed -> snapshot embedded
    locked = publish(author, load(author)).get_json()["id"]
    nb2 = json.loads(reader.get(f"/api/shares/{locked}/notebook.ipynb").data)
    assert "_SNAPSHOT_GZ_B64" not in " ".join("".join(c["source"]) for c in nb2["cells"])  # author did not allow it


def test_restricted_kaggle_license_blocks_data_reuse(author):
    body = load(author)
    body["dataset"]["kaggle_info"] = {"title": "Restricted", "license": "CC BY-NC-SA 4.0", "description": "x"}
    body["dataset"]["dataset_id"] = body["dataset"]["dataset_id"]
    import shares as shares_mod

    allowed, note = shares_mod.data_policy({"kaggle_info": {"license": "CC BY-NC-SA 4.0"}}, "kaggle", True)
    assert allowed is False and "license" in note
    assert shares_mod.data_policy({"kaggle_info": {"license": "CC0: Public Domain"}}, "kaggle", True)[0] is True
    assert shares_mod.data_policy({}, "upload", False)[0] is False


# ------------------------------------------------------------------ dataset graph

def test_joinable_finds_the_builtin_gapminder_library():
    client = make_client()
    body = load(client)
    result = client.post("/api/graph/joinable", json={"dataset_id": body["dataset"]["dataset_id"], "dataset_name": "climate"}).get_json()
    ids = [c["id"] for c in result["candidates"]]
    assert "library:gapminder" in ids
    cand = next(c for c in result["candidates"] if c["id"] == "library:gapminder")
    assert cand["best"]["kind"] == "country+year" and cand["best"]["strength"] in {"high", "medium"}
    assert "life_expectancy" in cand["adds_measures"] and cand["options"][0]["description"]


def test_preview_and_create_combined_dataset():
    client = make_client()
    body = load(client)
    cand = client.post("/api/graph/joinable", json={"dataset_id": body["dataset"]["dataset_id"]}).get_json()["candidates"][0]
    request_body = {"dataset_id": body["dataset"]["dataset_id"], "dataset_name": "climate", "candidate_id": cand["id"], "option": cand["best"], "how": "left"}
    preview = client.post("/api/graph/preview", json=request_body).get_json()
    assert preview["stats"]["matched_left_pct"] == 25.0 and "life_expectancy" in preview["columns"] and preview["stats"]["result_rows"] == 64
    created = client.post("/api/graph/join", json=request_body).get_json()
    assert created["dataset"]["dataset_id"].startswith("join-") and "life_expectancy" in created["dataset"]["columns"]
    assert created["provenance"]["right"].startswith("Gapminder") and created["analysis"]["shape"]["rows"] == 64
    # the combined dataset is a normal dataset: charts render on it
    chart = client.post("/charts/render", json={"dataset_id": created["dataset"]["dataset_id"], "config": {"type": "scatter", "x": "life_expectancy", "y": "co2"}})
    assert chart.status_code == 200 and not chart.get_json()["meta"].get("error")


def test_join_rejects_bad_requests():
    client = make_client()
    body = load(client)
    base = {"dataset_id": body["dataset"]["dataset_id"], "candidate_id": "library:gapminder"}
    assert client.post("/api/graph/join", json={**base, "option": {}}).status_code == 400
    assert client.post("/api/graph/join", json={**base, "option": {"left": ["nope"], "right": ["country"], "parts": ["country"]}}).status_code == 400
    assert client.post("/api/graph/join", json={**base, "candidate_id": "bogus:1", "option": {"left": ["country"], "right": ["country"], "parts": ["country"]}}).status_code == 400
    assert client.post("/api/graph/join", json={"dataset_id": "gone", "option": {}}).status_code == 400


def test_index_grows_and_joins_across_users_projects(monkeypatch):
    import graph as graph_mod
    import pandas as pd

    frame = pd.DataFrame({"Country": COUNTRIES * 3, "Year": [2000] * 8 + [2001] * 8 + [2002] * 8, "internet": [round(i * 1.37 + (i % 5) * 0.4, 2) for i in range(24)]})
    graph_mod.index_dataset("someone/internet-usage", "Internet usage", "CC0: Public Domain", frame)
    assert make_client().get("/api/graph/stats").get_json()["indexed"] == 1
    client = make_client()
    body = load(client)
    found = client.post("/api/graph/joinable", json={"dataset_id": body["dataset"]["dataset_id"]}).get_json()
    kaggle = [c for c in found["candidates"] if c["type"] == "kaggle"]
    assert kaggle and kaggle[0]["id"] == "kaggle:someone/internet-usage" and "internet" in kaggle[0]["adds_measures"]
    # joining a catalogue entry loads its sample from Kaggle; stub the loader so no network/credentials are needed
    import dataset.streaming as streaming

    monkeypatch.setattr(streaming, "load_dataset_sample", lambda slug, max_sample_bytes=0: {"dataframe": frame, "sample": {"is_sample": False}})
    request_body = {"dataset_id": body["dataset"]["dataset_id"], "dataset_name": "climate", "candidate_id": kaggle[0]["id"], "option": kaggle[0]["best"]}
    preview = client.post("/api/graph/preview", json=request_body).get_json()
    assert "internet" in preview["columns"] and preview["stats"]["matched_left_rows"] > 0


def test_joining_your_own_projects():
    client = make_client()
    signup(client)
    a = load(client)
    project = client.post("/api/projects", json={"dataset_id": a["dataset"]["dataset_id"], "dataset_name": "climate", "name": "Climate"}).get_json()["project"]
    other_csv = "nation,yr,gini\n" + "\n".join(f"{c},{y},{30 + i + j}" for i, c in enumerate(COUNTRIES) for j, y in enumerate(range(2000, 2008)))
    b = load(client, other_csv, "inequality.csv")
    found = client.post("/api/graph/joinable", json={"dataset_id": b["dataset"]["dataset_id"], "dataset_name": "inequality"}).get_json()
    mine = [c for c in found["candidates"] if c["id"] == f"project:{project['id']}"]
    assert mine and "co2" in mine[0]["adds_measures"]
    body = {"dataset_id": b["dataset"]["dataset_id"], "dataset_name": "inequality", "candidate_id": mine[0]["id"], "option": mine[0]["best"]}
    created = client.post("/api/graph/join", json=body).get_json()
    assert "co2" in created["dataset"]["columns"] and created["provenance"]["stats"]["matched_left_pct"] == 100.0
    # another user cannot join against someone else's project
    stranger = make_client()
    signup(stranger, "eve@example.com", "Eve")
    c = load(stranger, other_csv, "inequality.csv")
    assert stranger.post("/api/graph/join", json={"dataset_id": c["dataset"]["dataset_id"], "candidate_id": f"project:{project['id']}", "option": mine[0]["best"]}).status_code == 404
