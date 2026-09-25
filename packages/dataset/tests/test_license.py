from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.license import describe_license  # noqa: E402


@pytest.mark.parametrize("name,level", [
    ("CC0: Public Domain", "open"),
    ("CC BY-SA 4.0", "conditions"),
    ("Attribution 4.0 International (CC BY 4.0)", "conditions"),
    ("CC BY-NC-SA 4.0", "restricted"),
    ("CC BY-ND 4.0", "restricted"),
    ("Other (specified in description)", "restricted"),
    ("", "unknown"),
    (None, "unknown"),
])
def test_license_levels(name, level):
    info = describe_license(name)
    assert info["level"] == level and info["summary"]
