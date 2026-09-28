"""Storage backends behind one small interface, so the rest of the app never knows where data lives.

  Documents (JSON records: projects, published analyses, dataset catalogue)
      SqliteDocs     - a table in the local SQLite file            (default, zero setup)
      RealtimeDocs   - the Firebase Realtime Database                (when a Firebase service account is configured)

  Blobs (dataframes as Parquet / gzipped CSV)
      LocalBlobs     - files under the data directory
      StorageBlobs   - Cloud Storage for Firebase

Every document is a dict with an ``id``; ``owner`` (a string) and ``public`` are stored alongside so lists
can be filtered without reading every record.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

import firebase

MAX_DOC_BYTES = 900_000  # comfortably under a single Realtime Database node's practical size


def _dump(doc: dict[str, Any], default: Callable[[Any], Any]) -> str:
    text = json.dumps(doc, ensure_ascii=False, default=default)
    if len(text.encode("utf-8")) > MAX_DOC_BYTES:
        raise ValueError("This record is too large to store. Remove some findings or notes and try again.")
    return text


# ============================================================================= documents

class SqliteDocs:
    def __init__(self, connect: Callable[[], sqlite3.Connection], default: Callable[[Any], Any]) -> None:
        self._connect, self._default = connect, default

    def get(self, coll: str, doc_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT doc FROM docs WHERE coll = ? AND id = ?", (coll, doc_id)).fetchone()
        finally:
            conn.close()
        return json.loads(row[0]) if row else None

    def put(self, coll: str, doc_id: str, doc: dict[str, Any], *, owner: str | None = None, public: bool = False) -> None:
        doc = {**doc, "id": doc_id}
        text = _dump(doc, self._default)
        conn = self._connect()
        try:
            with conn:
                conn.execute("INSERT INTO docs(coll, id, owner, public, updated, doc) VALUES (?,?,?,?,?,?) ON CONFLICT(coll, id) DO UPDATE SET owner = excluded.owner, "
                             "public = excluded.public, updated = excluded.updated, doc = excluded.doc", (coll, doc_id, owner or "", 1 if public else 0, time.time(), text))
        finally:
            conn.close()

    def delete(self, coll: str, doc_id: str) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute("DELETE FROM docs WHERE coll = ? AND id = ?", (coll, doc_id))
        finally:
            conn.close()

    def find(self, coll: str, *, owner: str | None = None, public: bool | None = None, limit: int = 500) -> list[dict[str, Any]]:
        sql, params = "SELECT doc FROM docs WHERE coll = ?", [coll]
        if owner is not None:
            sql += " AND owner = ?"
            params.append(owner)
        if public is not None:
            sql += " AND public = ?"
            params.append(1 if public else 0)
        conn = self._connect()
        try:
            rows = conn.execute(sql + " LIMIT ?", params + [limit]).fetchall()
        finally:
            conn.close()
        return [json.loads(r[0]) for r in rows]


class RealtimeDocs:
    def __init__(self, client: firebase.RealtimeDatabase, default: Callable[[Any], Any]) -> None:
        self.client, self._default = client, default

    def get(self, coll: str, doc_id: str) -> dict[str, Any] | None:
        return self.client.get(coll, doc_id)

    def put(self, coll: str, doc_id: str, doc: dict[str, Any], *, owner: str | None = None, public: bool = False) -> None:
        doc = {**doc, "id": doc_id}
        self.client.put(coll, doc_id, _dump(doc, self._default), owner, public, time.time())

    def delete(self, coll: str, doc_id: str) -> None:
        self.client.delete(coll, doc_id)

    def find(self, coll: str, *, owner: str | None = None, public: bool | None = None, limit: int = 500) -> list[dict[str, Any]]:
        return self.client.query(coll, owner=owner, public=public, limit=limit)


# ============================================================================= blobs

class LocalBlobs:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path(self, name: str) -> Path:
        path = (self.directory / name).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError("Invalid data path.")
        return path

    def put(self, name: str, data: bytes) -> None:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, name: str) -> bytes:
        return self._path(name).read_bytes()

    def delete(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)

    def exists(self, name: str) -> bool:
        return self._path(name).exists()


class StorageBlobs:
    def __init__(self, client: firebase.Storage) -> None:
        self.client = client

    def put(self, name: str, data: bytes) -> None:
        self.client.put(name, data)

    def get(self, name: str) -> bytes:
        return self.client.get(name)

    def delete(self, name: str) -> None:
        self.client.delete(name)

    def exists(self, name: str) -> bool:
        try:
            self.client.get(name)
            return True
        except FileNotFoundError:
            return False
