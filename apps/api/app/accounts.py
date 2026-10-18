"""Accounts: sign in/out, profile, account deletion - through Firebase Authentication or built-in accounts.

Two modes (DATASCOPIA_AUTH):
  firebase (default)  Firebase Authentication handles identity: email + password, Google sign-in, password
                      reset and email verification. The browser sends its Firebase ID token to
                      ``POST /api/auth/firebase``; the server verifies it against Google's public keys and
                      starts a session cookie. No password ever reaches this server.
  local               built-in email + password accounts (scrypt), useful offline and in tests.

Sessions are random tokens held in an HttpOnly cookie; only a SHA-256 of the token is stored
server-side. Every state-changing API request must come from the app's own origin - except the one
origin named in DATASCOPIA_TRUSTED_ORIGIN, if set, which exists for a frontend hosted separately
from this API (e.g. a GitHub Pages build of apps/web - GitHub Pages only serves static files, so
it can't run this server itself). Left unset (the default), nothing about CORS, cookies or CSRF
changes from a same-origin deployment: the cookie stays SameSite=Lax and CORS stays wide open for
the (non-credentialed) public read endpoints, same as before this existed.
"""

from __future__ import annotations

import os
import re
import threading
import time
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, Flask, g, jsonify, request

import firebase
import store
from errors import ApiError

COOKIE = "ds_session"
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s.]{2,}$")
MIN_PASSWORD = 8
MAX_ATTEMPTS, WINDOW_SECONDS = 6, 15 * 60

bp = Blueprint("auth", __name__, url_prefix="/api/auth")
_mode_override: list[str | None] = [None]
FRESH_LOGIN_SECONDS = 600


def auth_mode() -> str:
    if _mode_override[0]:
        return _mode_override[0]
    return "local" if (os.getenv("DATASCOPIA_AUTH") or "firebase").lower() == "local" else "firebase"


def trusted_origin() -> str | None:
    """The one extra origin (scheme+host, no trailing slash) allowed to make credentialed
    cross-origin requests - e.g. https://you.github.io for a GitHub Pages frontend. None by
    default, which is what keeps every CORS/cookie/CSRF behavior below unchanged for a normal
    same-origin deployment."""
    value = (os.getenv("DATASCOPIA_TRUSTED_ORIGIN") or "").strip()
    return value.rstrip("/") or None


def set_mode(mode: str | None) -> None:
    """Force a mode (used by tests); None restores the environment-driven default."""
    _mode_override[0] = mode


def _require_local() -> None:
    if auth_mode() != "local":
        raise ApiError(400, "Accounts are managed by Firebase sign-in; use the sign-in form.")
_attempts: dict[str, list[float]] = {}
_attempts_lock = threading.Lock()


# ---------------------------------------------------------------------------------- helpers

def public_user(row: Any) -> dict[str, Any]:
    return {"id": row["id"], "email": "" if row["email"].endswith("@users.firebase.local") else row["email"], "display_name": row["display_name"], "created_at": row["created_at"],
            "managed_by": "firebase" if row["firebase_uid"] else "local", "photo_url": row["photo_url"] if "photo_url" in row.keys() else ""}


def current_user() -> Any | None:
    if "user" not in g:
        g.user = store.user_for_token(request.cookies.get(COOKIE))
    return g.user


def login_required(fn: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if current_user() is None:
            raise ApiError(401, "Sign in to use this feature.")
        return fn(*args, **kwargs)

    return wrapper


def _set_cookie(response: Any, token: str | None) -> Any:
    secure = request.is_secure or request.headers.get("X-Forwarded-Proto") == "https"
    # A cross-site fetch (a trusted-origin frontend on a different domain) only ever receives this
    # cookie back if it's SameSite=None - but browsers refuse SameSite=None without Secure, so this
    # only activates over HTTPS; a trusted origin configured without HTTPS simply won't get a
    # working session; the request to the API still fails closed via enforce_same_origin either way.
    cross_site = bool(trusted_origin()) and request.headers.get("Origin") == trusted_origin()
    samesite = "None" if cross_site else "Lax"
    if token:
        response.set_cookie(COOKIE, token, max_age=store.SESSION_DAYS * 86400, httponly=True, samesite=samesite, secure=(secure or cross_site), path="/")
    else:
        response.delete_cookie(COOKIE, path="/", samesite=samesite, secure=(secure or cross_site))
    return response


def _json() -> dict[str, Any]:
    return request.get_json(silent=True) or {}


def _throttle_key(email: str) -> str:
    return f"{request.headers.get('X-Forwarded-For', request.remote_addr)}|{email.lower()}"


def _check_throttle(key: str) -> None:
    with _attempts_lock:
        recent = [t for t in _attempts.get(key, []) if t > time.time() - WINDOW_SECONDS]
        _attempts[key] = recent
        if len(recent) >= MAX_ATTEMPTS:
            raise ApiError(429, "Too many sign-in attempts. Please wait a few minutes and try again.")


def _record_failure(key: str) -> None:
    with _attempts_lock:
        _attempts.setdefault(key, []).append(time.time())


def reset_throttle() -> None:
    with _attempts_lock:
        _attempts.clear()


def _validate_credentials(email: str, password: str) -> None:
    if not EMAIL_RE.match(email):
        raise ApiError(400, "Enter a valid email address.")
    if len(password) < MIN_PASSWORD:
        raise ApiError(400, f"Choose a password of at least {MIN_PASSWORD} characters.")
    if len(password) > 200:
        raise ApiError(400, "That password is too long.")


def enforce_same_origin(app: Flask) -> None:
    """Reject state-changing API calls that come from another site (CSRF defence in depth on top of
    the session cookie's SameSite attribute) - except the configured trusted origin, if any."""

    @app.before_request
    def _origin_check() -> None:
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.path.startswith("/api/"):
            origin = request.headers.get("Origin")
            if origin and origin.split("://", 1)[-1] != request.host and origin != trusted_origin():
                raise ApiError(403, "Cross-site requests are not allowed.")


# ---------------------------------------------------------------------------------- routes

@bp.post("/register")
def register():
    _require_local()
    body = _json()
    email = str(body.get("email") or "").strip()
    password = str(body.get("password") or "")
    name = str(body.get("display_name") or "").strip()[:60] or email.split("@")[0]
    _validate_credentials(email, password)
    with store.db() as conn:
        if conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
            raise ApiError(409, "An account with that email already exists. Try signing in.")
        cur = conn.execute("INSERT INTO users(email, display_name, pw_hash, created_at, last_login) VALUES (?,?,?,?,?)",
                           (email, name, store.hash_password(password), store.now(), store.now()))
        user_id = cur.lastrowid
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    token = store.create_session(user_id, request.headers.get("User-Agent", ""))
    return _set_cookie(jsonify({"user": public_user(row)}), token), 201


@bp.post("/login")
def login():
    _require_local()
    body = _json()
    email = str(body.get("email") or "").strip()
    password = str(body.get("password") or "")
    key = _throttle_key(email)
    _check_throttle(key)
    with store.db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        ok = bool(row) and store.verify_password(password, row["pw_hash"])
        if not ok:
            if not row:
                store.hash_password(password or "x")  # keep timing similar whether or not the account exists
            _record_failure(key)
            raise ApiError(401, "Incorrect email or password.")
        conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (store.now(), row["id"]))
    token = store.create_session(row["id"], request.headers.get("User-Agent", ""))
    return _set_cookie(jsonify({"user": public_user(row)}), token)


@bp.get("/config")
def config():
    """What the browser needs to sign in: the mode and (for Firebase) the public web configuration."""
    return jsonify({"mode": auth_mode(), "firebase": firebase.web_config() if auth_mode() == "firebase" else None, "data_backend": store.CONFIG.get("backend") if store.CONFIG.get("db_path") else store.data_backend()})


@bp.post("/firebase")
def firebase_sign_in():
    """Exchange a Firebase ID token for a session cookie."""
    if auth_mode() != "firebase":
        raise ApiError(400, "Firebase sign-in is not enabled on this server.")
    body = _json()
    key = _throttle_key(request.headers.get("X-Forwarded-For", request.remote_addr or "-"))
    _check_throttle(key)
    try:
        claims = firebase.verify_id_token(str(body.get("id_token") or ""))
    except firebase.FirebaseError as exc:
        _record_failure(key)
        raise ApiError(401, str(exc)) from exc
    user = store.upsert_firebase_user(claims, str(body.get("display_name") or "").strip())
    token = store.create_session(user["id"], request.headers.get("User-Agent", ""))
    return _set_cookie(jsonify({"user": public_user(user)}), token)


@bp.post("/logout")
def logout():
    store.delete_session(request.cookies.get(COOKIE))
    return _set_cookie(jsonify({"ok": True}), None)


@bp.get("/me")
def me():
    user = current_user()
    return jsonify({"user": public_user(user) if user else None})


@bp.put("/profile")
@login_required
def update_profile():
    name = str(_json().get("display_name") or "").strip()[:60]
    if not name:
        raise ApiError(400, "Display name cannot be empty.")
    with store.db() as conn:
        conn.execute("UPDATE users SET display_name = ? WHERE id = ?", (name, current_user()["id"]))
        row = conn.execute("SELECT * FROM users WHERE id = ?", (current_user()["id"],)).fetchone()
    return jsonify({"user": public_user(row)})


@bp.post("/password")
@login_required
def change_password():
    _require_local()
    body = _json()
    user = current_user()
    if not store.verify_password(str(body.get("current_password") or ""), user["pw_hash"]):
        raise ApiError(403, "Your current password is incorrect.")
    new = str(body.get("new_password") or "")
    if len(new) < MIN_PASSWORD:
        raise ApiError(400, f"Choose a password of at least {MIN_PASSWORD} characters.")
    keep = request.cookies.get(COOKIE)
    with store.db() as conn:
        conn.execute("UPDATE users SET pw_hash = ? WHERE id = ?", (store.hash_password(new), user["id"]))
        conn.execute("DELETE FROM sessions WHERE user_id = ? AND token_hash != ?", (user["id"], store._token_hash(keep or "")))  # sign out other devices
    return jsonify({"ok": True})


def delete_user_data(owner: str) -> int:
    """Remove every project and published analysis (and its stored data) owned by ``owner``."""
    removed = 0
    for coll in ("projects", "shares"):
        for doc in store.docs().find(coll, owner=owner, limit=1000):
            store.docs().delete(coll, doc["id"])
            store.delete_frame(doc.get("data_path"))
            removed += 1
    return removed


@bp.post("/delete")
@login_required
def delete_account():
    user = current_user()
    body = _json()
    if auth_mode() == "firebase" and user["firebase_uid"]:
        try:
            claims = firebase.verify_id_token(str(body.get("id_token") or ""))
        except firebase.FirebaseError as exc:
            raise ApiError(403, str(exc)) from exc
        if claims["uid"] != user["firebase_uid"]:
            raise ApiError(403, "That sign-in belongs to a different account.")
        if not claims.get("auth_time") or time.time() - float(claims["auth_time"]) > FRESH_LOGIN_SECONDS:
            raise ApiError(403, "For your security, please confirm your password again before deleting your account.")
    elif not store.verify_password(str(body.get("password") or ""), user["pw_hash"]):
        raise ApiError(403, "Your password is incorrect.")
    delete_user_data(store.owner_of(user))
    with store.db() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user["id"],))  # cascades to sessions
    return _set_cookie(jsonify({"ok": True}), None)
