"""Firebase integration: ID-token sign-in, and the Realtime Database/Cloud Storage backends.

No network or credentials are used. Google's token verification is stubbed, and the Realtime Database and
Cloud Storage REST APIs are replaced by an in-memory fake that implements just the calls the app makes -
checking the request shapes (JSON-tree paths, orderBy/equalTo queries, object URLs) as it goes.
"""
from __future__ import annotations

import io
import json
import time
import urllib.parse

import pytest

import accounts
import docstore
import firebase
import main
import store

PASSWORD = "correct horse battery"
DB_BASE = "https://fake-db.firebaseio.com"
CSV = "country,year,gdp,forest\n" + "\n".join(f"{c},{y},{100 + i * 3 + j},{40 - i}" for i, c in enumerate(["France", "Germany", "Japan", "Brazil", "India", "Kenya"]) for j, y in enumerate(range(2000, 2008)))


class FakeResponse:
    def __init__(self, status: int = 200, payload=None, content: bytes = b"") -> None:
        self.status_code, self._payload, self.content = status, payload, content
        self.text = json.dumps(payload) if payload is not None else content.decode("utf-8", "ignore")

    def json(self):
        return self._payload


class FakeGoogle:
    """In-memory Realtime Database (a JSON tree) + Cloud Storage (objects) behind a requests-like session."""

    def __init__(self) -> None:
        self.tree: dict[str, dict[str, dict]] = {}  # collection -> {doc_id: {"owner", "public", "doc", "updated"}}
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []

    # ---- Realtime Database: paths look like https://fake-db.firebaseio.com/<collection>[/<id>].json
    @staticmethod
    def _rtdb_parts(url: str) -> list[str]:
        path = url[len(DB_BASE) + 1 :].rsplit(".json", 1)[0]
        return [urllib.parse.unquote(p) for p in path.split("/")]

    def get(self, url, params=None, **kw):
        self.calls.append(("GET", url))
        if url.startswith(DB_BASE):
            parts = self._rtdb_parts(url)
            if len(parts) == 1:
                records = dict(self.tree.get(parts[0], {}))
                if params and "orderBy" in params:
                    field, value = json.loads(params["orderBy"]), json.loads(params["equalTo"])
                    records = {k: v for k, v in records.items() if v.get(field) == value}
                return FakeResponse(200, records or None)
            record = self.tree.get(parts[0], {}).get(parts[1])
            return FakeResponse(200, record)  # RTDB returns 200 + null for a missing path, never 404
        name = urllib.parse.unquote(url.split("/o/", 1)[1])
        assert params == {"alt": "media"}
        return FakeResponse(200, content=self.objects[name]) if name in self.objects else FakeResponse(404, {"error": {"message": "missing"}})

    def put(self, url, json=None, **kw):
        self.calls.append(("PUT", url))
        assert url.startswith(DB_BASE) and set(json) == {"owner", "public", "doc", "updated"}
        collection, doc_id = self._rtdb_parts(url)
        self.tree.setdefault(collection, {})[doc_id] = json
        return FakeResponse(200, json)

    def delete(self, url, **kw):
        self.calls.append(("DELETE", url))
        if url.startswith(DB_BASE):
            collection, doc_id = self._rtdb_parts(url)
            self.tree.get(collection, {}).pop(doc_id, None)
        else:
            self.objects.pop(urllib.parse.unquote(url.split("/o/", 1)[1]), None)
        return FakeResponse(200, {})

    def post(self, url, json=None, params=None, data=None, headers=None, **kw):
        self.calls.append(("POST", url))
        assert "/upload/storage/v1/b/" in url and params["uploadType"] == "media"
        self.objects[params["name"]] = data
        return FakeResponse(200, {"name": params["name"]})


@pytest.fixture
def cloud():
    """A store wired to the fake Realtime Database/Storage instead of SQLite documents and local files."""
    fake = FakeGoogle()
    store.configure(store.CONFIG["db_path"], store.CONFIG["data_dir"], docs=docstore.RealtimeDocs(firebase.RealtimeDatabase(fake, DB_BASE), store._json_default),
                    blobs=docstore.StorageBlobs(firebase.Storage(fake, "demo-bucket")))
    return fake


@pytest.fixture
def client():
    main.app.config["TESTING"] = True
    return main.app.test_client()


def make_client():
    main.app.config["TESTING"] = True
    return main.app.test_client()


def register(client, email="ada@example.com", name="Ada"):
    assert client.post("/api/auth/register", json={"email": email, "password": PASSWORD, "display_name": name}).status_code == 201


def load(client):
    r = client.post("/datasets/upload", data={"file": (io.BytesIO(CSV.encode()), "econ.csv")}, content_type="multipart/form-data")
    return r.get_json()


# ------------------------------------------------------------------ Realtime Database + Storage backends

def test_realtime_database_documents_roundtrip_and_query(cloud):
    docs = store.docs()
    docs.put("projects", "p1", {"name": "one", "n": 1}, owner="u1")
    docs.put("projects", "p2", {"name": "two"}, owner="u2")
    docs.put("shares", "s1", {"title": "pub"}, owner="u1", public=True)
    assert docs.get("projects", "p1") == {"name": "one", "n": 1, "id": "p1"}
    assert docs.get("projects", "nope") is None
    assert [d["id"] for d in docs.find("projects", owner="u1")] == ["p1"]
    assert [d["id"] for d in docs.find("shares", public=True)] == ["s1"] and docs.find("shares", owner="u1", public=False) == []
    docs.put("projects", "p1", {"name": "renamed"}, owner="u1")  # upsert replaces
    assert docs.get("projects", "p1")["name"] == "renamed"
    docs.delete("projects", "p1")
    docs.delete("projects", "p1")  # deleting twice is fine
    assert docs.get("projects", "p1") is None
    with pytest.raises(ValueError):
        docs.put("projects", "big", {"text": "x" * 950_000}, owner="u1")


def test_cloud_storage_frames_roundtrip(cloud):
    import pandas as pd

    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None]})
    rel = store.save_frame("projects", "abc", frame)
    assert rel in cloud.objects and store.load_frame(rel).equals(frame)
    store.delete_frame(rel)
    assert rel not in cloud.objects
    store.delete_frame(rel)  # already gone: no error


def test_realtime_database_error_is_reported_not_swallowed(cloud):
    class Broken(FakeGoogle):
        def get(self, url, params=None, **kw):
            return FakeResponse(403, {"error": "Missing or insufficient permissions."})

    docs = firebase.RealtimeDatabase(Broken(), DB_BASE)
    with pytest.raises(firebase.FirebaseError, match="insufficient permissions"):
        docs.get("projects", "x")


def test_projects_shares_and_graph_work_on_the_cloud_backend(cloud, client):
    register(client)
    body = load(client)
    step = {"type": "filter", "params": {"column": "year", "comparator": ">=", "value": "2004"}, "description": "since 2004"}
    project = client.post("/api/projects", json={"dataset_id": body["dataset"]["dataset_id"], "dataset_name": "econ", "name": "Cloud project", "transform_log": [step]}).get_json()["project"]
    assert any(k.startswith("projects/") for k in cloud.objects) and cloud.tree.get("projects")
    opened = client.post(f"/api/projects/{project['id']}/open").get_json()
    assert opened["dataset"]["shape"]["rows"] == 24  # 6 countries x 4 years
    published = client.post("/api/shares", json={"dataset_id": body["dataset"]["dataset_id"], "dataset_name": "econ", "title": "In the cloud", "dataset": body["dataset"],
                                                 "transform_log": [step], "allow_data_reuse": True})
    assert published.status_code == 201 and published.get_json()["verified"] is True
    share_id = published.get_json()["id"]
    reader = make_client()
    assert reader.get(f"/api/shares/{share_id}").get_json()["analysis"]["verification"]["verified"] is True
    assert [a["id"] for a in reader.get("/api/shares").get_json()["analyses"]] == [share_id]
    assert reader.post(f"/api/shares/{share_id}/verify").get_json()["verification"]["verified"] is True
    bob = make_client()
    register(bob, "bob@example.com", "Bob")
    assert bob.post(f"/api/shares/{share_id}/fork").status_code == 201
    assert [p["name"] for p in bob.get("/api/projects").get_json()["projects"]] == ["Fork of In the cloud"]
    # the data really lives in Cloud Storage, not on local disk
    assert not any((store.CONFIG["data_dir"] / "shares").glob("*")) if (store.CONFIG["data_dir"] / "shares").exists() else True
    # catalogue + joinable candidates go through the Realtime Database too
    import graph

    import pandas as pd

    graph.index_dataset("someone/forests", "Forests", "CC0: Public Domain", pd.DataFrame({"Nation": ["France", "Germany", "Japan", "Brazil"] * 2, "yr": [2000] * 4 + [2001] * 4, "trees": [3.1, 4.7, 5.2, 8.8, 3.3, 4.9, 5.1, 9.0]}))
    assert cloud.tree.get("dataset_index")
    found = client.post("/api/graph/joinable", json={"dataset_id": body["dataset"]["dataset_id"]}).get_json()
    assert "kaggle:someone/forests" in [c["id"] for c in found["candidates"]]


def test_deleting_an_account_removes_cloud_documents_and_files(cloud, client):
    register(client)
    body = load(client)
    client.post("/api/projects", json={"dataset_id": body["dataset"]["dataset_id"], "dataset_name": "econ", "name": "p"})
    assert cloud.tree and cloud.objects
    assert client.post("/api/auth/delete", json={"password": PASSWORD}).status_code == 200
    assert not cloud.tree.get("projects") and not [k for k in cloud.objects if k.startswith("projects/")]


# ------------------------------------------------------------------ Firebase Authentication (ID tokens)

@pytest.fixture
def firebase_auth(monkeypatch):
    accounts.set_mode("firebase")
    claims = {"user_id": "uid-123", "email": "grace@example.com", "email_verified": True, "name": "Grace Hopper", "auth_time": time.time(), "firebase": {"sign_in_provider": "password"}}
    seen = {}

    def fake_verify(token, audience):
        seen["audience"] = audience
        if token == "bad":
            raise ValueError("Token expired")
        return dict(claims)

    monkeypatch.setattr(firebase, "_verify_raw", fake_verify)
    return claims, seen


def test_config_exposes_public_web_config_only(client, firebase_auth):
    data = client.get("/api/auth/config").get_json()
    assert data["mode"] == "firebase" and data["firebase"]["projectId"] == "datascopia" and data["firebase"]["apiKey"]
    assert "serviceAccount" not in json.dumps(data) and "private_key" not in json.dumps(data)
    accounts.set_mode("local")
    assert client.get("/api/auth/config").get_json() == {**client.get("/api/auth/config").get_json(), "mode": "local", "firebase": None}


def test_id_token_is_exchanged_for_a_session(client, firebase_auth):
    _, seen = firebase_auth
    response = client.post("/api/auth/firebase", json={"id_token": "good-token"})
    assert response.status_code == 200 and "HttpOnly" in response.headers["Set-Cookie"]
    user = response.get_json()["user"]
    assert user["display_name"] == "Grace Hopper" and user["email"] == "grace@example.com" and user["managed_by"] == "firebase"
    assert seen["audience"] == "datascopia"  # the token must have been issued for THIS project
    assert client.get("/api/auth/me").get_json()["user"]["id"] == user["id"]
    again = client.post("/api/auth/firebase", json={"id_token": "good-token", "display_name": "Rear Admiral Hopper"}).get_json()["user"]
    assert again["id"] == user["id"] and again["display_name"] == "Rear Admiral Hopper"  # same account, refreshed name


def test_google_sign_in_captures_and_refreshes_the_profile_photo(client, monkeypatch):
    accounts.set_mode("firebase")
    photo = "https://lh3.googleusercontent.com/a/one.jpg"
    claims = {"user_id": "uid-google-1", "email": "ada@example.com", "email_verified": True, "name": "Ada Lovelace",
              "picture": photo, "auth_time": time.time(), "firebase": {"sign_in_provider": "google.com"}}
    monkeypatch.setattr(firebase, "_verify_raw", lambda token, audience: dict(claims))

    user = client.post("/api/auth/firebase", json={"id_token": "t"}).get_json()["user"]
    assert user["photo_url"] == photo
    assert client.get("/api/auth/me").get_json()["user"]["photo_url"] == photo

    # Google can rotate the photo URL on a later sign-in; the stored one should follow it.
    claims["picture"] = "https://lh3.googleusercontent.com/a/two.jpg"
    again = client.post("/api/auth/firebase", json={"id_token": "t"}).get_json()["user"]
    assert again["id"] == user["id"] and again["photo_url"] == "https://lh3.googleusercontent.com/a/two.jpg"


def test_invalid_or_missing_tokens_are_rejected(client, firebase_auth):
    assert client.post("/api/auth/firebase", json={"id_token": "bad"}).status_code == 401
    assert client.post("/api/auth/firebase", json={}).status_code == 401
    assert client.get("/api/auth/me").get_json()["user"] is None


def test_builtin_password_endpoints_are_disabled_in_firebase_mode(client, firebase_auth):
    assert client.post("/api/auth/register", json={"email": "a@b.co", "password": PASSWORD}).status_code == 400
    assert client.post("/api/auth/login", json={"email": "a@b.co", "password": PASSWORD}).status_code == 400
    client.post("/api/auth/firebase", json={"id_token": "t"})
    assert client.post("/api/auth/password", json={"current_password": "x", "new_password": "yyyyyyyy"}).status_code == 400


def test_firebase_users_own_their_data_by_uid(cloud, client, firebase_auth):
    client.post("/api/auth/firebase", json={"id_token": "t"})
    body = load(client)
    client.post("/api/projects", json={"dataset_id": body["dataset"]["dataset_id"], "dataset_name": "econ", "name": "Grace's project"})
    assert store.docs().find("projects", owner="uid-123")[0]["name"] == "Grace's project"
    # another Firebase user cannot see it
    claims, _ = firebase_auth
    other = make_client()
    claims.update({"user_id": "uid-999", "email": "other@example.com", "name": "Other"})
    other.post("/api/auth/firebase", json={"id_token": "t"})
    assert other.get("/api/projects").get_json()["projects"] == []


def test_account_deletion_needs_a_fresh_sign_in(client, firebase_auth):
    claims, _ = firebase_auth
    client.post("/api/auth/firebase", json={"id_token": "t"})
    body = load(client)
    client.post("/api/projects", json={"dataset_id": body["dataset"]["dataset_id"], "dataset_name": "econ", "name": "p"})
    claims["auth_time"] = time.time() - 3600  # signed in an hour ago
    stale = client.post("/api/auth/delete", json={"id_token": "t"})
    assert stale.status_code == 403 and "confirm your password" in stale.get_json()["detail"]
    claims["user_id"] = "someone-else"
    assert client.post("/api/auth/delete", json={"id_token": "t"}).status_code == 403  # a different account's token
    claims.update({"user_id": "uid-123", "auth_time": time.time()})
    assert client.post("/api/auth/delete", json={"id_token": "t"}).status_code == 200
    assert store.docs().find("projects", owner="uid-123") == [] and client.get("/api/auth/me").get_json()["user"] is None


def test_existing_local_account_is_linked_by_email(client, firebase_auth):
    accounts.set_mode("local")
    register(client, "grace@example.com", "Grace (old)")
    old_id = client.get("/api/auth/me").get_json()["user"]["id"]
    accounts.set_mode("firebase")
    linked = make_client().post("/api/auth/firebase", json={"id_token": "t"}).get_json()["user"]
    assert linked["id"] == old_id and linked["managed_by"] == "firebase"


def test_verify_id_token_normalises_claims_and_rejects_garbage(monkeypatch):
    monkeypatch.setattr(firebase, "_verify_raw", lambda token, audience: {"sub": "u1", "email": "a@b.co", "firebase": {"sign_in_provider": "google.com"}})
    claims = firebase.verify_id_token("x")
    assert claims["uid"] == "u1" and claims["provider"] == "google.com" and claims["email_verified"] is False
    monkeypatch.setattr(firebase, "_verify_raw", lambda token, audience: {})
    with pytest.raises(firebase.FirebaseError):
        firebase.verify_id_token("x")
    with pytest.raises(firebase.FirebaseError):
        firebase.verify_id_token("")


def test_service_account_loading(monkeypatch, tmp_path):
    for var in ("FIREBASE_SERVICE_ACCOUNT", "FIREBASE_SERVICE_ACCOUNT_FILE", "GOOGLE_APPLICATION_CREDENTIALS", "FIREBASE_DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)
    assert firebase.has_credentials() is False and store.data_backend() == "local"
    key = {"type": "service_account", "project_id": "datascopia", "client_email": "x@datascopia.iam.gserviceaccount.com", "private_key": "not-a-real-key"}
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT", json.dumps(key))
    assert firebase.has_credentials() and store.data_backend() == "firebase"
    import base64

    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT", base64.b64encode(json.dumps(key).encode()).decode())
    assert firebase.has_credentials()
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT", "{ not json")
    assert firebase.has_credentials() is False
    monkeypatch.setenv("DATASCOPIA_DATA", "local")
    assert store.data_backend() == "local"


def test_database_url_is_required_and_configure_falls_back_when_missing(monkeypatch, tmp_path):
    key = {"type": "service_account", "project_id": "datascopia", "client_email": "x@datascopia.iam.gserviceaccount.com",
           "private_key": "not-a-real-key", "token_uri": "https://oauth2.googleapis.com/token"}
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT", json.dumps(key))
    monkeypatch.delenv("FIREBASE_DATABASE_URL", raising=False)
    with pytest.raises(firebase.FirebaseError, match="FIREBASE_DATABASE_URL"):
        firebase.database_url()
    # configure() must not crash the app when Firebase is misconfigured - it logs and falls back to local storage
    store.configure(tmp_path / "t.db", tmp_path / "data")
    assert store.CONFIG["backend"] == "local"
    monkeypatch.setenv("FIREBASE_DATABASE_URL", "https://datascopia-default-rtdb.firebaseio.com/")
    assert firebase.database_url() == "https://datascopia-default-rtdb.firebaseio.com"  # trailing slash stripped
