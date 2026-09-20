from __future__ import annotations

import threading
import time

import pandas as pd

# Lets a dataset's dataframe be reused across follow-up questions in the same research
# session without re-fetching it from Kaggle each time, while staying purely in-process
# memory (never written to disk, never shared beyond this server run).
_TTL_SECONDS = 30 * 60
_MAX_ENTRIES = 20

_lock = threading.Lock()
_store: dict[str, tuple[pd.DataFrame, float]] = {}


def put(dataset_id: str, df: pd.DataFrame) -> None:
    with _lock:
        _evict_expired()
        if dataset_id not in _store and len(_store) >= _MAX_ENTRIES:
            oldest_id = min(_store, key=lambda key: _store[key][1])
            _store.pop(oldest_id, None)
        _store[dataset_id] = (df, time.time())


def get(dataset_id: str) -> pd.DataFrame | None:
    with _lock:
        _evict_expired()
        entry = _store.get(dataset_id)
        if entry is None:
            return None
        df, _ = entry
        _store[dataset_id] = (df, time.time())  # refresh TTL on access
        return df


def _evict_expired() -> None:
    now = time.time()
    expired = [key for key, (_, ts) in _store.items() if now - ts > _TTL_SECONDS]
    for key in expired:
        _store.pop(key, None)
