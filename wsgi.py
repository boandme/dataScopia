"""WSGI entrypoint for production servers (waitress, gunicorn, etc.).

Local development uses run.py, which binds a socket directly and retries on the next port if
the default is busy - a nicety that doesn't apply to a hosted deployment, where the platform
(Render, etc.) always assigns the port via the $PORT environment variable and runs a real
production WSGI server rather than Flask's own dev server. This file exists so that server can
point at a plain `module:app` reference (e.g. `waitress-serve wsgi:app`).

Loads apps/api/app/main.py by file path, the same way run.py does, so apps/ doesn't need to be
set up as a formal importable Python package (no __init__.py files required).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_PATH = PROJECT_ROOT / "apps" / "api" / "app" / "main.py"

_spec = importlib.util.spec_from_file_location("datascopia_main", MAIN_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load the API app from {MAIN_PATH}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

app = _module.app
