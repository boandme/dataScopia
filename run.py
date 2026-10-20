#!/usr/bin/env python3
"""Single-command launcher for DataScopia.

Starts the Flask server - which also serves the frontend from the same origin, see the "/" route
in apps/api/app/main.py - and opens the app in your default browser once it's actually listening.
This replaces manually starting a server in one terminal and separately opening
apps/web/index.html as a file:// page in a browser.

Usage:
    python run.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_PORT_ATTEMPTS = 10

# A PyInstaller build runs this exact script, but as compiled bytecode with no real file on disk -
# __file__ isn't a usable path there, so sys._MEIPASS (the bundle's extracted/collected root,
# set by PyInstaller in both onefile and onedir builds) stands in for it instead. Either way this
# resolves to the directory actually holding apps/ and packages/ as real files, which main.py's
# own dynamic import (see _load_app below) and its own PROJECT_ROOT computation both need.
PROJECT_ROOT = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
MAIN_PATH = PROJECT_ROOT / "apps" / "api" / "app" / "main.py"


def _default_desktop_data_dir() -> Path:
    """Where a packaged desktop build keeps its database and uploaded files by default - never
    inside the install directory itself, which a Program Files-style install makes read-only to
    a normal user and which a reinstall/upgrade could wipe. %LOCALAPPDATA% is the standard
    per-user writable location for exactly this on Windows."""
    base = os.getenv("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "DataScopia"


def _apply_desktop_defaults() -> None:
    """Only takes effect in a frozen build, and only for variables the user/host hasn't already
    set - a source checkout or a hosted deployment (Render, etc.) keeps using whatever
    DATASCOPIA_DB/DATASCOPIA_DATA_DIR already resolve to (see store.py)."""
    if not getattr(sys, "frozen", False):
        return
    data_dir = _default_desktop_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DATASCOPIA_DB", str(data_dir / "datascopia.db"))
    os.environ.setdefault("DATASCOPIA_DATA_DIR", str(data_dir))

    # Lets a user drop a .env with their Kaggle/Firebase credentials right next to the installed
    # .exe - the obvious place to look - rather than inside the bundle's internal data folder,
    # which is where main.py's own PROJECT_ROOT/.env lookup resolves to once frozen. Harmless
    # no-op if the file doesn't exist; main.py's own load_dotenv(..., override=True) runs right
    # after this and won't clear anything this already set, since it has nothing to override with.
    from dotenv import load_dotenv

    load_dotenv(Path(sys.executable).resolve().parent / ".env", override=False)


def _load_app():
    """Import apps/api/app/main.py by file path rather than a dotted module string, so this
    works regardless of whether apps/ is set up as an importable package (no __init__.py files
    required) and matches exactly how the app has been loaded throughout development."""
    spec = importlib.util.spec_from_file_location("datascopia_main", MAIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the API app from {MAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


class _PortBindFailure(Exception):
    """Raised in place of the real SystemExit that werkzeug's BaseWSGIServer.__init__ calls
    directly on a bind failure (address already in use) - it catches the OSError itself, prints
    a message, and calls sys.exit(1) rather than letting the exception propagate normally, which
    would otherwise kill this whole launcher on the very first occupied port instead of just
    that one attempt. Swapping in this catchable exception for the duration of a single bind
    attempt keeps the failure local and retryable."""


def _bind_server(app, make_server, host: str, start_port: int, attempts: int):
    """Try `start_port`, then a few after it. werkzeug's make_server() performs a real,
    synchronous bind - unlike some ASGI servers, there's no separate "probe, then bind" race to
    worry about here: the attempt to bind *is* the definitive answer, so trying ports in order is
    both correct and simple, once sys.exit() is neutralized for the duration of the attempt (see
    _PortBindFailure)."""

    def _guarded_exit(code: int = 0) -> None:
        raise _PortBindFailure(f"exit({code}) during bind")

    last_error: Exception | None = None
    original_exit = sys.exit
    sys.exit = _guarded_exit
    try:
        for port in range(start_port, start_port + attempts):
            try:
                return make_server(host, port, app, threaded=True), port
            except (OSError, _PortBindFailure) as exc:
                last_error = exc
    finally:
        sys.exit = original_exit
    raise RuntimeError(
        f"Could not find a free port in {start_port}-{start_port + attempts - 1}: {last_error}"
    )


def start_server():
    """Set up and bind the real server (shared by this CLI entry point and desktop_app.py's
    native-window entry point) without starting to serve yet - the caller decides how: block the
    main thread (this file's main(), for a terminal/dev run) or run it in a background thread
    behind a native window (desktop_app.py, for the packaged app)."""
    try:
        from werkzeug.serving import BaseWSGIServer, ThreadedWSGIServer, make_server
    except ImportError:
        print("Flask/werkzeug isn't installed. Run: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1)

    # werkzeug's server classes default allow_reuse_address = True (SO_REUSEADDR). On Windows,
    # SO_REUSEADDR is far more permissive than on Unix: it lets a *second* process bind to a
    # port a *first* process is already actively listening on, without error - both then believe
    # they own the port, and which one actually receives a given request becomes undefined. That
    # silently defeats the whole point of trying a real bind to detect "is this port free" -
    # disabling it restores the expected "fails if something's already listening" behavior.
    BaseWSGIServer.allow_reuse_address = False
    ThreadedWSGIServer.allow_reuse_address = False

    _apply_desktop_defaults()
    app = _load_app()
    server, port = _bind_server(app, make_server, HOST, DEFAULT_PORT, MAX_PORT_ATTEMPTS)
    url = f"http://{HOST}:{port}"
    if port != DEFAULT_PORT:
        print(f"Port {DEFAULT_PORT} was busy - using {port} instead.", flush=True)
    return server, url


def main() -> None:
    server, url = start_server()
    print(f"DataScopia is running at {url}", flush=True)

    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
