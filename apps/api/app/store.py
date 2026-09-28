"""Persistent storage: user records and sessions (local SQLite) plus documents and dataframes, which live
either beside the database or in Firebase (the Realtime Database + Cloud Storage) - see ``docstore``.

Configuration (environment variables, all optional):
  DATASCOPIA_DB        path of the SQLite file          (default: <project>/data/datascopia.db)
  DATASCOPIA_DATA_DIR  directory for dataframe files    (default: same folder as the database)
  DATASCOPIA_DATA      "auto" | "firebase" | "local"    (auto = Firebase when a service account is configured)

Sessions are deliberately local: they are cheap, disposable cookies. If the disk is wiped (e.g. a redeploy on a
free host) users simply sign in again - with Firebase Authentication that happens silently.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

import docstore
import firebase

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG: dict[str, Any] = {"db_path": None, "data_dir": None, "docs": None, "blobs": None, "backend": "local"}
_init_lock = threading.Lock()

SESSION_DAYS = 30
MAX_PROJECTS_PER_USER = int(os.getenv("DATASCOPIA_MAX_PROJECTS", "40"))
MAX_FRAME_BYTES = int(float(os.getenv("DATASCOPIA_MAX_PROJECT_MB", "80")) * 1024 * 1024)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT NOT NULL UNIQUE COLLATE NOCASE,
  display_name TEXT NOT NULL,
  pw_hash TEXT NOT NULL DEFAULT '',
  firebase_uid TEXT UNIQUE,
  created_at REAL NOT NULL,
  last_login REAL
);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at REAL NOT NULL,
  expires_at REAL NOT NULL,
  user_agent TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS docs (
  coll TEXT NOT NULL,
  id TEXT NOT NULL,
  owner TEXT NOT NULL DEFAULT '',
  public INTEGER NOT NULL DEFAULT 0,
  updated REAL NOT NULL,
  doc TEXT NOT NULL,
  PRIMARY KEY (coll, id)
);
CREATE INDEX IF NOT EXISTS idx_docs_owner ON docs(coll, owner);
CREATE INDEX IF NOT EXISTS idx_docs_public ON docs(coll, public);
"""


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if obj != obj else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=_json_default)


def data_backend() -> str:
    choice = (os.getenv("DATASCOPIA_DATA") or "auto").lower()
    if choice == "firebase" or (choice == "auto" and firebase.has_credentials()):
        return "firebase"
    return "local"


def configure(db_path: str | os.PathLike[str] | None = None, data_dir: str | os.PathLike[str] | None = None, *, docs: Any = None, blobs: Any = None) -> None:
    """(Re)point the store at a database and data directory; docs/blobs may be injected (used by tests)."""
    path = Path(db_path or os.getenv("DATASCOPIA_DB") or PROJECT_ROOT / "data" / "datascopia.db")
    directory = Path(data_dir or os.getenv("DATASCOPIA_DATA_DIR") or path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    with _init_lock:
        CONFIG["db_path"], CONFIG["data_dir"] = path, directory
        with sqlite3.connect(path) as conn:
            conn.executescript(SCHEMA)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
            if "firebase_uid" not in cols:  # databases created before Firebase support
                conn.execute("ALTER TABLE users ADD COLUMN firebase_uid TEXT")
                conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_firebase ON users(firebase_uid)")
        if docs is not None or blobs is not None:
            CONFIG["docs"], CONFIG["blobs"], CONFIG["backend"] = docs, blobs, "injected"
        elif data_backend() == "firebase":
            try:
                session = firebase.authorized_session()
                CONFIG["docs"] = docstore.RealtimeDocs(firebase.RealtimeDatabase(session, firebase.database_url()), _json_default)
                CONFIG["blobs"] = docstore.StorageBlobs(firebase.Storage(session))
                CONFIG["backend"] = "firebase"
            except firebase.FirebaseError as exc:
                print(f"[datascopia] Firebase storage unavailable ({exc}); falling back to local storage.")
                CONFIG["docs"] = docstore.SqliteDocs(_connect, _json_default)
                CONFIG["blobs"] = docstore.LocalBlobs(directory)
                CONFIG["backend"] = "local"
        else:
            CONFIG["docs"] = docstore.SqliteDocs(_connect, _json_default)
            CONFIG["blobs"] = docstore.LocalBlobs(directory)
            CONFIG["backend"] = "local"
        if CONFIG["docs"] is None:
            CONFIG["docs"] = docstore.SqliteDocs(_connect, _json_default)
        if CONFIG["blobs"] is None:
            CONFIG["blobs"] = docstore.LocalBlobs(directory)


def _ensure() -> None:
    if CONFIG["db_path"] is None:
        configure()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(CONFIG["db_path"], timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    """A short-lived connection to the user/session tables; commits on success, rolls back on error."""
    _ensure()
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def docs() -> Any:
    _ensure()
    return CONFIG["docs"]


def blobs() -> Any:
    _ensure()
    return CONFIG["blobs"]


def new_id(nbytes: int = 6) -> str:
    return secrets.token_hex(nbytes)


def now() -> float:
    return time.time()


# ============================================================================= users

def owner_of(user: Any) -> str:
    """The stable string that owns a user's documents: their Firebase uid, or the local id for built-in accounts."""
    return user["firebase_uid"] or str(user["id"])


def upsert_firebase_user(claims: dict[str, Any], display_name: str = "") -> Any:
    """Create or refresh the local record for a Firebase user (sessions and ownership hang off it)."""
    email = claims["email"] or f"{claims['uid']}@users.firebase.local"
    name = (display_name or claims.get("name") or email.split("@")[0]).strip()[:60]
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE firebase_uid = ?", (claims["uid"],)).fetchone()
        if row:
            conn.execute("UPDATE users SET email = ?, display_name = CASE WHEN ? != '' THEN ? ELSE display_name END, last_login = ? WHERE id = ?",
                         (email, display_name.strip()[:60], name, now(), row["id"]))
        else:
            # a built-in account with the same email is linked rather than duplicated
            existing = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            if existing and not existing["firebase_uid"]:
                conn.execute("UPDATE users SET firebase_uid = ?, last_login = ? WHERE id = ?", (claims["uid"], now(), existing["id"]))
            elif existing:
                email = f"{claims['uid']}@users.firebase.local"
                conn.execute("INSERT INTO users(email, display_name, pw_hash, firebase_uid, created_at, last_login) VALUES (?,?,?,?,?,?)", (email, name, "", claims["uid"], now(), now()))
            else:
                conn.execute("INSERT INTO users(email, display_name, pw_hash, firebase_uid, created_at, last_login) VALUES (?,?,?,?,?,?)", (email, name, "", claims["uid"], now(), now()))
        return conn.execute("SELECT * FROM users WHERE firebase_uid = ?", (claims["uid"],)).fetchone()


# ============================================================================= passwords / sessions

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    n, r, p = 2 ** 14, 8, 1
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        candidate = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(candidate, bytes.fromhex(digest))
    except Exception:
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(user_id: int, user_agent: str = "") -> str:
    token = secrets.token_urlsafe(32)
    ts = now()
    with db() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (ts,))
        conn.execute("INSERT INTO sessions(token_hash, user_id, created_at, expires_at, user_agent) VALUES (?,?,?,?,?)",
                     (_token_hash(token), user_id, ts, ts + SESSION_DAYS * 86400, user_agent[:200]))
    return token


def user_for_token(token: str | None) -> sqlite3.Row | None:
    if not token:
        return None
    with db() as conn:
        row = conn.execute("SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ? AND s.expires_at > ?", (_token_hash(token), now())).fetchone()
        if row:  # sliding expiry
            conn.execute("UPDATE sessions SET expires_at = ? WHERE token_hash = ?", (now() + SESSION_DAYS * 86400, _token_hash(token)))
        return row


def delete_session(token: str | None) -> None:
    if token:
        with db() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


# ============================================================================= dataframes

def _rel(kind: str, ident: str, ext: str) -> str:
    return f"{kind}/{ident}.{ext}"


def save_frame(kind: str, ident: str, df: pd.DataFrame) -> str:
    """Store a table and return its key. Parquet when the table allows it, gzipped CSV otherwise."""
    _ensure()
    buffer = io.BytesIO()
    try:
        df.to_parquet(buffer, index=False)
        rel = _rel(kind, ident, "parquet")
    except Exception:
        # mixed-type object columns: CSV round-trips anything
        buffer = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
        import gzip

        compressed = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode="wb") as handle:
            handle.write(buffer.getvalue())
        buffer = compressed
        rel = _rel(kind, ident, "csv.gz")
    data = buffer.getvalue()
    if len(data) > MAX_FRAME_BYTES:
        raise ValueError(f"This dataset is too large to store ({len(data) / 1e6:.0f} MB; the limit is {MAX_FRAME_BYTES / 1e6:.0f} MB).")
    blobs().put(rel, data)
    return rel


def load_frame(rel: str) -> pd.DataFrame:
    data = blobs().get(rel)
    if rel.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data), compression="gzip", low_memory=False)


def delete_frame(rel: str | None) -> None:
    if not rel:
        return
    try:
        blobs().delete(rel)
    except Exception:  # noqa: BLE001 - a leftover file must never block a delete
        pass
