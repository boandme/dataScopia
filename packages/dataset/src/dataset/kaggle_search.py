from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv
import requests


KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"
PROJECT_ENV_FILE = Path(__file__).resolve().parents[4] / ".env"


def _ensure_kaggle_config_file() -> bool:
    load_dotenv(PROJECT_ENV_FILE, override=True)

    token = os.getenv("KAGGLE_API_TOKEN") or os.getenv("KAGGLE_KEY")
    if token and token.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = token
        token_path = Path.home() / ".kaggle" / "access_token"
        try:
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(token, encoding="utf-8")
        except OSError:
            pass
        return True

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            try:
                data = json.loads(kaggle_json.read_text(encoding="utf-8"))
                username = data.get("username")
                key = data.get("key")
            except Exception:
                username, key = None, None
        if not username or not key:
            return False

    access_token_path = Path.home() / ".kaggle" / "access_token"
    if not os.getenv("KAGGLE_API_TOKEN") and access_token_path.exists():
        try:
            access_token_path.unlink()
        except OSError:
            pass

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    payload = {"username": username, "key": key}
    try:
        existing = {}
        if kaggle_json.exists():
            existing = json.loads(kaggle_json.read_text(encoding="utf-8") or "{}")
        if existing != payload:
            kaggle_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    return True


def validate_kaggle_credentials() -> bool:
    return _ensure_kaggle_config_file()


def kaggle_credential_status() -> dict[str, str | bool]:
    configured = _ensure_kaggle_config_file()
    token = os.getenv("KAGGLE_API_TOKEN")
    username = os.getenv("KAGGLE_USERNAME")
    source = "modern API token" if token else "legacy kaggle.json"
    return {
        "configured": configured,
        "source": source if configured else "none",
        "username": username or "",
        "env_file": str(PROJECT_ENV_FILE),
    }


def _kaggle_request_kwargs() -> dict[str, Any]:
    token = os.getenv("KAGGLE_API_TOKEN") or os.getenv("KAGGLE_KEY")
    if token and token.startswith("KGAT_"):
        return {"headers": {"Authorization": f"Bearer {token}"}, "timeout": 120}
    return {
        "auth": (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]),
        "timeout": 120,
    }


def _download_dataset_files_fallback(dataset_slug: str, target_dir: Path) -> None:
    owner, slug = dataset_slug.split("/", 1)
    files_response = requests.get(
        f"{KAGGLE_API_BASE}/datasets/list/{quote(owner)}/{quote(slug)}",
        **_kaggle_request_kwargs(),
    )
    files_response.raise_for_status()
    files = files_response.json().get("datasetFiles", [])
    if not files:
        raise ValueError(f"Kaggle returned no files for dataset '{dataset_slug}'.")

    for file_info in files:
        file_name = file_info.get("name")
        if not file_name:
            continue
        # file_name may itself contain "/" (files nested under a folder, common for
        # notebook-sourced datasets), so it must be fully percent-encoded (safe="") rather than
        # treated as literal path segments, or Kaggle 404s on the request.
        file_response = requests.get(
            f"{KAGGLE_API_BASE}/datasets/download/{quote(owner)}/{quote(slug)}/{quote(file_name, safe='')}",
            **_kaggle_request_kwargs(),
        )
        file_response.raise_for_status()
        output_path = target_dir / Path(file_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(file_response.content)

    if not any(target_dir.rglob("*")):
        raise ValueError(f"Kaggle returned no downloadable files for '{dataset_slug}'.")


def download_kaggle_dataset(dataset_slug: str, destination: str | None = None) -> str:
    if not dataset_slug:
        raise ValueError("A Kaggle dataset slug is required.")
    if not validate_kaggle_credentials():
        raise ValueError("Kaggle credentials are not configured.")

    base_dir = Path(destination) if destination else Path.home() / ".kaggle_datasets"
    base_dir.mkdir(parents=True, exist_ok=True)
    safe_slug = dataset_slug.strip().replace("/", "-")
    target_dir = base_dir / safe_slug
    if not target_dir.exists() or not any(target_dir.iterdir()):
        target_dir.mkdir(parents=True, exist_ok=True)
        import kagglehub

        previous_cache = os.environ.get("KAGGLEHUB_CACHE")
        os.environ["KAGGLEHUB_CACHE"] = str(target_dir / ".kagglehub-cache")
        try:
            try:
                kagglehub.dataset_download(
                    dataset_slug.strip(),
                    output_dir=str(target_dir),
                    force_download=True,
                )
            except Exception:
                for child in target_dir.iterdir():
                    if child.is_file():
                        child.unlink()
                _download_dataset_files_fallback(dataset_slug.strip(), target_dir)
        finally:
            if previous_cache is None:
                os.environ.pop("KAGGLEHUB_CACHE", None)
            else:
                os.environ["KAGGLEHUB_CACHE"] = previous_cache
    return str(target_dir)


def search_kaggle_datasets(query: str, page_size: int = 10, max_results: int = 10) -> list[dict[str, Any]]:
    if not validate_kaggle_credentials():
        return [
            {
                "id": "demo-dataset",
                "title": "Kaggle credentials not available",
                "subtitle": f"Add a legacy kaggle.json or API token. Checked: {PROJECT_ENV_FILE}",
                "url": "",
                "is_demo": True,
            }
        ]

    try:
        from kaggle import KaggleApi

        api = KaggleApi()
        api.authenticate()
        datasets = api.dataset_list(search=query, max_size=page_size, file_type=None, sort_by="hottest")
        payload = []
        for item in datasets[:max_results]:
            metadata = getattr(item, "__dict__", {})
            dataset_ref = getattr(item, "ref", None) or getattr(item, "dataset", None) or metadata.get("ref")
            payload.append(
                {
                    "id": dataset_ref,
                    "slug": dataset_ref,
                    "title": getattr(item, "title", None) or metadata.get("title"),
                    "subtitle": getattr(item, "subtitle", None) or metadata.get("subtitle"),
                    "url": getattr(item, "url", None) or metadata.get("url"),
                    "size": getattr(item, "size", None) or metadata.get("size"),
                    "is_demo": False,
                }
            )
        return payload
    except Exception as exc:
        return [
            {
                "id": "demo-fallback",
                "title": "Kaggle search unavailable",
                "subtitle": f"The Kaggle API could not be reached or credentials are invalid: {exc}",
                "url": "",
                "is_demo": True,
            }
        ]
