"""Every API test gets its own empty database and data directory, and a clean sign-in throttle."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@pytest.fixture(autouse=True)
def isolated_store(tmp_path):
    import accounts
    import store

    store.configure(tmp_path / "test.db", tmp_path / "data")
    accounts.reset_throttle()
    accounts.set_mode("local")   # built-in accounts unless a test opts into Firebase mode
    yield
    accounts.set_mode(None)
