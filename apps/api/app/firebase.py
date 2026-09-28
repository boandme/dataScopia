"""Firebase integration without the Admin SDK (no gRPC, no heavy dependencies).

* **Identity** - the browser signs users in with Firebase Authentication (email/password, Google). It sends the
  resulting ID token to the server, which verifies Google's signature, audience and issuer with public
  certificates - no secret needed - and starts a normal session cookie.
* **Data** - when a service account is configured, projects, published analyses and the dataset catalogue live in
  the Firebase Realtime Database and dataframes in Cloud Storage, reached through their REST APIs with that
  service account. The server is the only client; the bundled security rules deny direct access from browsers
  (a service-account request is authenticated as an admin and bypasses rules, same as with the Admin SDK).

Configuration (environment variables):
  FIREBASE_PROJECT_ID           default "datascopia"
  FIREBASE_WEB_API_KEY, FIREBASE_AUTH_DOMAIN, FIREBASE_STORAGE_BUCKET, FIREBASE_MESSAGING_SENDER_ID, FIREBASE_APP_ID
                                the public web config (defaults are this project's values; none of them is a secret)
  FIREBASE_DATABASE_URL         the Realtime Database URL, e.g. https://datascopia-default-rtdb.firebaseio.com
                                (Firebase console -> Build -> Realtime Database -> top of the Data tab) - required
                                to store projects/analyses/catalogue in Firebase rather than locally
  FIREBASE_SERVICE_ACCOUNT      the service-account key JSON (or base64 of it) - enables Realtime Database + Storage
  FIREBASE_SERVICE_ACCOUNT_FILE / GOOGLE_APPLICATION_CREDENTIALS   path to that key file instead
  DATASCOPIA_AUTH               "firebase" (default) or "local" (built-in email + password accounts)
  DATASCOPIA_DATA               "auto" (default: Firebase when credentials exist), "firebase" or "local"
"""

from __future__ import annotations

import base64
import json
import os
import urllib.parse
from typing import Any

PROJECT_ID_DEFAULT = "datascopia"
FIREBASE_SCOPES = ["https://www.googleapis.com/auth/firebase.database", "https://www.googleapis.com/auth/userinfo.email",
                   "https://www.googleapis.com/auth/devstorage.read_write", "https://www.googleapis.com/auth/cloud-platform"]


class FirebaseError(Exception):
    """A Firebase/Google API call failed; the message is safe to show."""


def project_id() -> str:
    return os.getenv("FIREBASE_PROJECT_ID") or PROJECT_ID_DEFAULT


def web_config() -> dict[str, str]:
    """The browser-side config. These values identify the project and are public by design."""
    pid = project_id()
    return {
        "apiKey": os.getenv("FIREBASE_WEB_API_KEY", "AIzaSyBInOQR6qMQqvH2oT-zOan7gwkIwxeVYLU"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", f"{pid}.firebaseapp.com"),
        "projectId": pid,
        "storageBucket": storage_bucket(),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", "118502814569"),
        "appId": os.getenv("FIREBASE_APP_ID", "1:118502814569:web:d3d97cec18a78f1b4003f8"),
    }


def storage_bucket() -> str:
    return os.getenv("FIREBASE_STORAGE_BUCKET") or f"{project_id()}.firebasestorage.app"


# ----------------------------------------------------------------------------- identity

def _verify_raw(token: str, audience: str) -> dict[str, Any]:
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    return id_token.verify_firebase_token(token, google_requests.Request(), audience=audience)


def _diagnose_verify_failure(exc: Exception, audience: str) -> str:
    """A specific, safe-to-show reason when possible (these describe the token/request, not a secret),
    falling back to a generic message for anything unrecognised. The real exception is always logged too."""
    text = str(exc)
    lower = text.lower()
    if isinstance(exc, ImportError):
        return "The server is missing a required package (google-auth). Run: pip install -r requirements.txt, then restart the server."
    if "audience" in lower or "aud" in lower and "wrong" in lower:
        return (f"This sign-in was issued for a different Firebase project than this server expects "
                f"(expected project '{audience}'). If FIREBASE_PROJECT_ID is set in your environment, check it "
                f"matches your Firebase project ID.")
    if "used too early" in lower or "not yet valid" in lower or "iat" in lower and "future" in lower:
        return "Your sign-in could not be verified because this computer's clock is set incorrectly (it appears to be behind). Fix the system date/time and try again."
    if "expired" in lower or "used too late" in lower:
        return "Your sign-in expired before it reached the server (or this computer's clock is set incorrectly - check the system date/time). Please sign in again."
    if "issuer" in lower:
        return "This sign-in was not issued by Firebase for this project. Please sign in again."
    if any(kind in type(exc).__module__ + "." + type(exc).__name__ for kind in ("Connection", "Timeout", "SSL", "URLError", "Transport")) or \
       any(word in lower for word in ("connection", "certificate", "ssl", "timed out", "timeout", "name or service not known", "getaddrinfo")):
        return "The server could not reach Google to verify your sign-in - check its internet connection (or a firewall/proxy/antivirus may be blocking outbound HTTPS to googleapis.com)."
    return "Your sign-in could not be verified. Please sign in again."


def verify_id_token(token: str) -> dict[str, Any]:
    """Verify a Firebase ID token (signature, expiry, audience = our project, issuer) and return normalised claims."""
    if not token or not isinstance(token, str):
        raise FirebaseError("Missing sign-in token.")
    audience = project_id()
    try:
        claims = _verify_raw(token, audience)
    except Exception as exc:  # noqa: BLE001 - any failure means the token cannot be trusted
        print(f"[datascopia] Firebase ID token verification failed ({type(exc).__name__}): {exc}")
        raise FirebaseError(_diagnose_verify_failure(exc, audience)) from exc
    if not claims:
        raise FirebaseError("Your sign-in could not be verified. Please sign in again.")
    uid = claims.get("user_id") or claims.get("sub")
    if not uid:
        raise FirebaseError("The sign-in token has no user.")
    email = claims.get("email") or ""
    return {"uid": str(uid), "email": email, "email_verified": bool(claims.get("email_verified")), "name": claims.get("name") or "", "picture": claims.get("picture") or "",
            "provider": (claims.get("firebase") or {}).get("sign_in_provider", ""), "auth_time": claims.get("auth_time")}


# ----------------------------------------------------------------------------- credentials

def _load_service_account() -> dict[str, Any] | None:
    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if raw:
        raw = raw.strip()
        try:
            return json.loads(raw if raw.startswith("{") else base64.b64decode(raw).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise FirebaseError("FIREBASE_SERVICE_ACCOUNT is not valid JSON (or base64 of JSON).") from exc
    path = os.getenv("FIREBASE_SERVICE_ACCOUNT_FILE") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    return None


def has_credentials() -> bool:
    try:
        return _load_service_account() is not None
    except FirebaseError:
        return False


def database_url() -> str:
    url = os.getenv("FIREBASE_DATABASE_URL")
    if not url:
        raise FirebaseError(
            "FIREBASE_DATABASE_URL is not set. Copy it from the Firebase console (Build -> Realtime Database -> "
            "top of the Data tab, e.g. https://datascopia-default-rtdb.firebaseio.com) and set it as an environment variable."
        )
    return url.rstrip("/")


def authorized_session() -> Any:
    """A requests session that signs every call with the service account (tokens refresh automatically)."""
    info = _load_service_account()
    if not info:
        raise FirebaseError("No Firebase service account is configured.")
    try:
        from google.auth.transport.requests import AuthorizedSession
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_info(info, scopes=FIREBASE_SCOPES)
    except ImportError as exc:
        raise FirebaseError("The server is missing a required package (google-auth). Run: pip install -r requirements.txt, then restart the server.") from exc
    except Exception as exc:  # noqa: BLE001 - a malformed key must not crash the app; it's reported like any other FirebaseError
        raise FirebaseError(f"FIREBASE_SERVICE_ACCOUNT is not a valid service-account key: {exc}") from exc
    return AuthorizedSession(credentials)


# ----------------------------------------------------------------------------- Realtime Database (REST)

class RealtimeDatabase:
    """Just enough of the Realtime Database REST API for a document store: get/put/delete/equality queries.

    Each collection is a top-level node keyed by document id, storing ``{owner, public, updated, doc}`` where
    ``doc`` is the record's JSON encoded as a string. A request authenticated with the service account's OAuth2
    token is treated as an admin and bypasses the database's security rules, the same way the Admin SDK does -
    so the rules can simply deny all direct (browser) access. Equality queries use RTDB's ``orderBy``/``equalTo``
    (which needs a ``.indexOn`` rule to stay fast as a collection grows; see database.rules.json).
    """

    def __init__(self, session: Any, database_url: str) -> None:
        self.session = session
        self.base = database_url.rstrip("/")

    def _url(self, *parts: str) -> str:
        return f"{self.base}/{'/'.join(urllib.parse.quote(p, safe='') for p in parts)}.json"

    def _check(self, response: Any) -> Any:
        if response.status_code >= 300:
            raise FirebaseError(f"Realtime Database request failed ({response.status_code}): {self._detail(response)}")
        return response

    @staticmethod
    def _detail(response: Any) -> str:
        try:
            payload = response.json()
            return payload.get("error") if isinstance(payload, dict) else str(payload)
        except Exception:  # noqa: BLE001
            return response.text[:200]

    def get(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        record = self._check(self.session.get(self._url(collection, doc_id))).json()
        return json.loads(record["doc"]) if record else None

    def put(self, collection: str, doc_id: str, doc_json: str, owner: str | None, public: bool, updated: float) -> None:
        payload = {"owner": owner or "", "public": bool(public), "doc": doc_json, "updated": float(updated)}
        self._check(self.session.put(self._url(collection, doc_id), json=payload))

    def delete(self, collection: str, doc_id: str) -> None:
        self._check(self.session.delete(self._url(collection, doc_id)))

    def query(self, collection: str, *, owner: str | None = None, public: bool | None = None, limit: int = 500) -> list[dict[str, Any]]:
        params: dict[str, str] | None = None
        if owner is not None:
            params = {"orderBy": json.dumps("owner"), "equalTo": json.dumps(owner)}
        elif public is not None:
            params = {"orderBy": json.dumps("public"), "equalTo": json.dumps(bool(public))}
        records = self._check(self.session.get(self._url(collection), params=params)).json() or {}
        rows = list(records.values())
        if owner is not None and public is not None:
            rows = [r for r in rows if r.get("public") == bool(public)]
        return [json.loads(r["doc"]) for r in rows[:limit]]


# ----------------------------------------------------------------------------- Cloud Storage (JSON API)

class Storage:
    def __init__(self, session: Any, bucket: str | None = None) -> None:
        self.session = session
        self.bucket = bucket or storage_bucket()

    def _object_url(self, name: str) -> str:
        return f"https://storage.googleapis.com/storage/v1/b/{self.bucket}/o/{urllib.parse.quote(name, safe='')}"

    def put(self, name: str, data: bytes) -> None:
        response = self.session.post(f"https://storage.googleapis.com/upload/storage/v1/b/{self.bucket}/o", params={"uploadType": "media", "name": name}, data=data,
                                     headers={"Content-Type": "application/octet-stream"})
        if response.status_code >= 300:
            raise FirebaseError(f"Cloud Storage upload failed ({response.status_code}): {response.text[:200]}")

    def get(self, name: str) -> bytes:
        response = self.session.get(self._object_url(name), params={"alt": "media"})
        if response.status_code == 404:
            raise FileNotFoundError(name)
        if response.status_code >= 300:
            raise FirebaseError(f"Cloud Storage download failed ({response.status_code}): {response.text[:200]}")
        return response.content

    def delete(self, name: str) -> None:
        response = self.session.delete(self._object_url(name))
        if response.status_code >= 300 and response.status_code != 404:
            raise FirebaseError(f"Cloud Storage delete failed ({response.status_code}): {response.text[:200]}")
