from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.huggingface import _flatten_cell, _license_from, _rows_to_frame  # noqa: E402


def test_license_from_card_data_wins_over_tags():
    assert _license_from(["license:other"], {"license": "mit"}) == "mit"


def test_license_from_card_data_list_takes_first():
    assert _license_from([], {"license": ["cc-by-4.0", "mit"]}) == "cc-by-4.0"


def test_license_from_tags_when_no_card_data():
    assert _license_from(["size_categories:n<1K", "license:apache-2.0", "region:us"], None) == "apache-2.0"


def test_license_from_neither_is_empty():
    assert _license_from(["region:us"], {}) == ""


def test_flatten_cell_leaves_scalars_alone():
    assert _flatten_cell(1) == 1
    assert _flatten_cell(1.5) == 1.5
    assert _flatten_cell("text") == "text"
    assert _flatten_cell(None) is None


def test_flatten_cell_stringifies_nested_structures():
    assert _flatten_cell([1, 2, 3]) == "[1, 2, 3]"
    assert _flatten_cell({"a": 1}) == '{"a": 1}'


def test_rows_to_frame_builds_a_scalar_dataframe():
    rows = [
        {"row_idx": 0, "row": {"name": "Alice", "tags": ["x", "y"], "age": 30}},
        {"row_idx": 1, "row": {"name": "Bob", "tags": [], "age": None}},
    ]
    df = _rows_to_frame(rows)
    assert list(df.columns) == ["name", "tags", "age"]
    assert df.loc[0, "tags"] == "[\"x\", \"y\"]"
    assert df.loc[1, "tags"] == "[]"
    assert df.loc[1, "age"] is None or df["age"].isna().iloc[1]
