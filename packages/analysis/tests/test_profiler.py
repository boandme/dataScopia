from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.profiler import analyze_dataframe  # noqa: E402


class TestBooleanColumnClassification:
    """Regression coverage for a real crash: pandas' is_numeric_dtype returns True for bool
    dtype, so checking it before is_bool_dtype silently misclassified boolean columns as
    numeric and sent them into IQR outlier math, where `q1 - 1.5 * iqr` on boolean quantile
    results raises "numpy boolean subtract... not supported"."""

    def test_boolean_column_classified_as_boolean_not_numeric(self):
        df = pd.DataFrame({"id": range(20), "is_active": [True, False] * 10, "score": range(20)})
        result = analyze_dataframe(df, dataset_name="bool-check")
        assert result["type_summary"]["is_active"] == "boolean"
        assert "is_active" not in result["numeric_columns"]
        assert "score" in result["numeric_columns"]

    def test_analysis_does_not_crash_with_boolean_column(self):
        df = pd.DataFrame({"id": range(30), "flag": [True, False, None] * 10, "amount": range(30)})
        # Should complete without raising - this call itself is the regression check.
        result = analyze_dataframe(df, dataset_name="bool-check-2")
        assert result["shape"] == {"rows": 30, "columns": 3}

    def test_all_boolean_dataset_does_not_crash(self):
        df = pd.DataFrame({"a": [True, False, True, True], "b": [False, False, True, False]})
        result = analyze_dataframe(df, dataset_name="all-bool")
        assert result["numeric_columns"] == []
        assert result["type_summary"] == {"a": "boolean", "b": "boolean"}

    def test_outlier_count_ignores_boolean_columns(self):
        df = pd.DataFrame({"flag": [True] * 19 + [False], "value": list(range(19)) + [1000]})
        result = analyze_dataframe(df, dataset_name="outlier-check")
        # Only 'value' (the real numeric column) should contribute to the outlier count.
        assert result["numeric_columns"] == ["value"]
        assert result["outlier_count"] >= 0  # computed at all, i.e. did not crash first
