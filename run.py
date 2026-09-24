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
import sys
import threading
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_PORT_ATTEMPTS = 10

PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_PATH = PROJECT_ROOT / "apps" / "api" / "app" / "main.py"


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


def main() -> None:
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

    app = _load_app()
    server, port = _bind_server(app, make_server, HOST, DEFAULT_PORT, MAX_PORT_ATTEMPTS)
    url = f"http://{HOST}:{port}"
    if port != DEFAULT_PORT:
        print(f"Port {DEFAULT_PORT} was busy - using {port} instead.", flush=True)
    print(f"DataScopia is running at {url}", flush=True)

    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
