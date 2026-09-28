"""Accounts and cloud projects: sign-up/sign-in, session cookie behaviour, isolation between users,
saving a session as a project, reopening it with the cleaning steps replayed, duplicating and deleting."""
from __future__ import annotations

import io

import pytest

import main
import store

CSV = "country,year,gdp\n" + "\n".join(f"{c},{y},{100 + i * 3 + j}" for i, c in enumerate(["France", "Germany", "Japan", "Brazil", "India", "Kenya"]) for j, y in enumerate(range(2000, 2012)))
PASSWORD = "correct horse battery"


@pytest.fixture
def client():
    main.app.config["TESTING"] = True
    return main.app.test_client()


def make_client():
    return main.app.test_client()


def register(client, email="ada@example.com", name="Ada"):
    response = client.post("/api/auth/register", json={"email": email, "password": PASSWORD, "display_name": name})
    assert response.status_code == 201, response.get_data(as_text=True)
    return response.get_json()["user"]


def upload(client):
    response = client.post("/datasets/upload", data={"file": (io.BytesIO(CSV.encode()), "econ.csv")}, content_type="multipart/form-data")
    assert response.status_code == 200
    return response.get_json()


def save_project(client, name="My econ project", steps=None):
    body = upload(client)
    response = client.post("/api/projects", json={"dataset_id": body["dataset"]["dataset_id"], "dataset_name": "econ", "name": name, "dataset": body["dataset"],
                                                  "transform_log": steps or [], "findings": [{"type": "question", "title": "Q", "question": "q", "answer": "a"}]})
    return body, response


# ------------------------------------------------------------------ accounts

def test_register_login_logout_and_me(client):
    assert client.get("/api/auth/me").get_json() == {"user": None}
    user = register(client)
    assert user["email"] == "ada@example.com" and "pw_hash" not in user
    assert client.get("/api/auth/me").get_json()["user"]["display_name"] == "Ada"
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").get_json()["user"] is None
    assert client.post("/api/auth/login", json={"email": "ADA@example.com", "password": PASSWORD}).status_code == 200  # email is case-insensitive
    assert client.get("/api/auth/me").get_json()["user"]["id"] == user["id"]


def test_session_cookie_is_httponly_and_samesite(client):
    response = client.post("/api/auth/register", json={"email": "a@b.co", "password": PASSWORD})
    cookie = response.headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie and PASSWORD not in cookie


def test_password_is_hashed_and_token_is_not_stored_raw(client):
    register(client)
    with store.db() as conn:
        pw = conn.execute("SELECT pw_hash FROM users").fetchone()[0]
        token_hash = conn.execute("SELECT token_hash FROM sessions").fetchone()[0]
    assert pw.startswith("scrypt$") and PASSWORD not in pw
    assert client.get_cookie("ds_session").value != token_hash


def test_registration_validation(client):
    assert client.post("/api/auth/register", json={"email": "nope", "password": PASSWORD}).status_code == 400
    assert client.post("/api/auth/register", json={"email": "a@b.co", "password": "short"}).status_code == 400
    register(client)
    assert client.post("/api/auth/register", json={"email": "ada@example.com", "password": PASSWORD}).status_code == 409


def test_wrong_password_and_throttle(client):
    register(client)
    fresh = make_client()
    for _ in range(6):
        assert fresh.post("/api/auth/login", json={"email": "ada@example.com", "password": "wrong-password"}).status_code == 401
    assert fresh.post("/api/auth/login", json={"email": "ada@example.com", "password": PASSWORD}).status_code == 429


def test_cross_origin_writes_are_rejected(client):
    response = client.post("/api/auth/register", json={"email": "x@y.zz", "password": PASSWORD}, headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    same = client.post("/api/auth/register", json={"email": "x@y.zz", "password": PASSWORD}, headers={"Origin": "http://localhost"})
    assert same.status_code == 201


def test_profile_password_change_and_delete(client):
    register(client)
    other_device = make_client()
    other_device.post("/api/auth/login", json={"email": "ada@example.com", "password": PASSWORD})
    assert client.put("/api/auth/profile", json={"display_name": "Countess"}).get_json()["user"]["display_name"] == "Countess"
    assert client.put("/api/auth/profile", json={"display_name": " "}).status_code == 400
    assert client.post("/api/auth/password", json={"current_password": "bad", "new_password": "new password 1"}).status_code == 403
    assert client.post("/api/auth/password", json={"current_password": PASSWORD, "new_password": "new password 1"}).status_code == 200
    assert other_device.get("/api/auth/me").get_json()["user"] is None  # other sessions signed out
    assert client.get("/api/auth/me").get_json()["user"] is not None
    assert client.post("/api/auth/delete", json={"password": "wrong"}).status_code == 403
    assert client.post("/api/auth/delete", json={"password": "new password 1"}).status_code == 200
    assert make_client().post("/api/auth/login", json={"email": "ada@example.com", "password": "new password 1"}).status_code == 401


def test_login_required_endpoints(client):
    for method, path in (("get", "/api/projects"), ("post", "/api/projects"), ("get", "/api/shares/mine"), ("post", "/api/shares"), ("put", "/api/auth/profile")):
        assert getattr(client, method)(path, json={}).status_code == 401, path


# ------------------------------------------------------------------ projects

def test_save_list_open_project_replays_cleaning(client):
    register(client)
    step = {"type": "filter", "params": {"column": "year", "comparator": ">=", "value": "2006"}, "description": "Kept year >= 2006"}
    body, response = save_project(client, steps=[step])
    assert response.status_code == 201
    project = response.get_json()["project"]
    assert project["rows"] == 72 and project["steps"] == 1 and project["findings"] == 1
    listing = client.get("/api/projects").get_json()
    assert [p["id"] for p in listing["projects"]] == [project["id"]]

    fresh = make_client()  # a different device: sign in and reopen
    fresh.post("/api/auth/login", json={"email": "ada@example.com", "password": PASSWORD})
    opened = fresh.post(f"/api/projects/{project['id']}/open").get_json()
    assert opened["dataset"]["shape"]["rows"] == 36  # the filter was replayed on the stored original
    assert opened["dataset"]["dataset_id"] == f"proj-{project['id']}" and opened["transform_log"][0]["type"] == "filter"
    assert opened["state"]["findings"][0]["title"] == "Q" and opened["analysis"]["shape"]["rows"] == 36
    # the reset-to-original shadow copy came back too
    reset = fresh.post("/datasets/transform/reset", json={"dataset_id": opened["dataset"]["dataset_id"]}).get_json()
    assert reset["shape"]["rows"] == 72
    # and charts work against the reopened dataset
    chart = fresh.post("/charts/render", json={"dataset_id": opened["dataset"]["dataset_id"], "config": {"type": "histogram", "x": "gdp"}})
    assert chart.status_code == 200


def test_update_rename_duplicate_delete(client):
    register(client)
    _, response = save_project(client)
    pid = response.get_json()["project"]["id"]
    updated = client.put(f"/api/projects/{pid}", json={"name": "Renamed", "findings": [{"type": "question", "title": "A"}, {"type": "question", "title": "B"}]}).get_json()["project"]
    assert updated["name"] == "Renamed" and updated["findings"] == 2
    copy = client.post(f"/api/projects/{pid}/duplicate").get_json()["project"]
    assert copy["name"] == "Renamed (copy)" and copy["findings"] == 2 and copy["id"] != pid
    assert client.delete(f"/api/projects/{pid}").status_code == 200
    assert [p["id"] for p in client.get("/api/projects").get_json()["projects"]] == [copy["id"]]
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_projects_are_private_to_their_owner(client):
    register(client)
    _, response = save_project(client)
    pid = response.get_json()["project"]["id"]
    intruder = make_client()
    register(intruder, "eve@example.com", "Eve")
    assert intruder.get("/api/projects").get_json()["projects"] == []
    for method, path in (("get", f"/api/projects/{pid}"), ("post", f"/api/projects/{pid}/open"), ("delete", f"/api/projects/{pid}"), ("post", f"/api/projects/{pid}/duplicate")):
        assert getattr(intruder, method)(path).status_code == 404


def test_project_requires_a_live_session_and_respects_limits(client, monkeypatch):
    register(client)
    assert client.post("/api/projects", json={"dataset_id": "gone", "name": "x"}).status_code == 400
    monkeypatch.setattr(store, "MAX_PROJECTS_PER_USER", 1)
    assert save_project(client)[1].status_code == 201
    assert save_project(client)[1].status_code == 409


def test_deleting_account_removes_project_files(client):
    register(client)
    _, response = save_project(client)
    path = store.docs().find("projects")[0]["data_path"]
    assert (store.CONFIG["data_dir"] / path).exists()
    client.post("/api/auth/delete", json={"password": PASSWORD})
    assert not (store.CONFIG["data_dir"] / path).exists()


def test_frames_roundtrip_including_mixed_types(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "mixed": [1, "x", None], "when": pd.to_datetime(["2020-01-01", "2020-01-02", None])})
    rel = store.save_frame("projects", "t1", df)
    back = store.load_frame(rel)
    assert list(back.columns) == ["a", "mixed", "when"] and len(back) == 3
