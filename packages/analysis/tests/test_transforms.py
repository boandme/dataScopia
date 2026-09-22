from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.transforms import TransformError, apply_transform  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: several structurally different datasets, so "does the filter work"
# is checked against variety (numeric-heavy, categorical-heavy, mixed with
# dates/constant columns, and a tiny edge-case table) rather than one shape.
# ---------------------------------------------------------------------------


@pytest.fixture
def sales_df() -> pd.DataFrame:
    """Numeric-heavy: floats/ints, scattered missing values, one injected outlier."""
    rng = np.random.default_rng(42)
    n = 200
    revenue = rng.normal(500, 120, n)
    revenue[-1] = 50_000.0  # deliberate outlier
    return pd.DataFrame(
        {
            "order_id": range(1, n + 1),
            "revenue": revenue,
            "units": rng.integers(1, 20, n),
            "discount_pct": [None if i % 17 == 0 else round(v, 2) for i, v in enumerate(rng.uniform(0, 0.3, n))],
            "region": rng.choice(["North", "South", "East", "West"], n),
        }
    )


@pytest.fixture
def survey_df() -> pd.DataFrame:
    """Categorical-heavy: strings, missing values, low cardinality."""
    rng = np.random.default_rng(7)
    n = 150
    satisfaction = rng.choice(
        ["Very Satisfied", "Satisfied", "Neutral", "Unsatisfied", None], n, p=[0.25, 0.3, 0.2, 0.15, 0.1]
    )
    return pd.DataFrame(
        {
            "respondent_id": [f"R{i:04d}" for i in range(n)],
            "satisfaction": satisfaction,
            "age_group": rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n),
            "would_recommend": rng.choice(["Yes", "No"], n),
            "years_customer": rng.integers(0, 15, n).astype(float),
        }
    )


@pytest.fixture
def events_df() -> pd.DataFrame:
    """Mixed types: date-like strings, a constant column, name columns for merge tests."""
    n = 100
    dates = pd.date_range("2022-01-01", periods=n, freq="3D").astype(str)
    return pd.DataFrame(
        {
            "event_date": dates,
            "attendees": np.random.default_rng(1).integers(10, 500, n),
            "venue": ["Main Hall"] * n,  # constant column - a real-world edge case
            "city": np.random.default_rng(2).choice(["Austin", "Denver", "Seattle"], n),
            "first_name": [f"Guest{i}" for i in range(n)],
            "last_name": [f"Surname{i}" for i in range(n)],
        }
    )


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Small edge case: only 3 rows, easy to fully exhaust with a filter."""
    return pd.DataFrame({"score": [10, 20, 30], "label": ["a", "b", "c"]})


ALL_FIXTURES = ["sales_df", "survey_df", "events_df", "tiny_df"]


# ---------------------------------------------------------------------------
# Filter - the headline operation, exercised across every fixture and every
# comparator, plus immutability and error-path checks.
# ---------------------------------------------------------------------------


class TestFilter:
    def test_greater_than_keeps_correct_rows(self, sales_df):
        result, desc = apply_transform(sales_df, "filter", {"column": "units", "comparator": ">", "value": "10"})
        assert (result["units"] > 10).all()
        assert len(result) == int((sales_df["units"] > 10).sum())
        assert "units" in desc and ">" in desc

    def test_does_not_mutate_input(self, sales_df):
        original = sales_df.copy(deep=True)
        apply_transform(sales_df, "filter", {"column": "units", "comparator": ">", "value": "5"})
        pd.testing.assert_frame_equal(sales_df, original)

    @pytest.mark.parametrize(
        "comparator,expected_fn",
        [
            (">", lambda s, v: s > v),
            (">=", lambda s, v: s >= v),
            ("<", lambda s, v: s < v),
            ("<=", lambda s, v: s <= v),
            ("==", lambda s, v: s == v),
            ("!=", lambda s, v: s != v),
        ],
    )
    def test_all_comparators_on_numeric_column(self, sales_df, comparator, expected_fn):
        threshold = float(sales_df["units"].median())
        result, _ = apply_transform(sales_df, "filter", {"column": "units", "comparator": comparator, "value": str(threshold)})
        expected_count = int(expected_fn(sales_df["units"], threshold).sum())
        assert len(result) == expected_count

    def test_filter_on_string_column_equality(self, survey_df):
        result, _ = apply_transform(survey_df, "filter", {"column": "would_recommend", "comparator": "==", "value": "Yes"})
        assert (result["would_recommend"] == "Yes").all()
        assert len(result) == int((survey_df["would_recommend"] == "Yes").sum())

    def test_filter_resets_index(self, sales_df):
        result, _ = apply_transform(sales_df, "filter", {"column": "units", "comparator": ">", "value": "1"})
        assert list(result.index) == list(range(len(result)))

    def test_unknown_column_raises(self, sales_df):
        with pytest.raises(TransformError, match="not found"):
            apply_transform(sales_df, "filter", {"column": "does_not_exist", "comparator": ">", "value": "1"})

    def test_unsupported_comparator_raises(self, sales_df):
        with pytest.raises(TransformError, match="[Cc]omparator"):
            apply_transform(sales_df, "filter", {"column": "units", "comparator": "~=", "value": "1"})

    def test_non_numeric_value_on_numeric_column_raises(self, sales_df):
        with pytest.raises(TransformError, match="not a valid number"):
            apply_transform(sales_df, "filter", {"column": "units", "comparator": ">", "value": "not-a-number"})

    def test_filter_that_empties_dataset_raises(self, tiny_df):
        with pytest.raises(TransformError, match="remove every row"):
            apply_transform(tiny_df, "filter", {"column": "score", "comparator": ">", "value": "1000"})

    @pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
    def test_filter_smoke_across_all_fixtures(self, request, fixture_name):
        df = request.getfixturevalue(fixture_name)
        numeric_cols = df.select_dtypes(include="number").columns
        assert len(numeric_cols) > 0, f"fixture {fixture_name} needs a numeric column for this test"
        col = numeric_cols[0]
        median = float(df[col].dropna().median())
        result, desc = apply_transform(df, "filter", {"column": col, "comparator": ">=", "value": str(median)})
        assert isinstance(result, pd.DataFrame)
        assert 0 < len(result) <= len(df)
        assert isinstance(desc, str) and desc


# ---------------------------------------------------------------------------
# Drop / fill missing values
# ---------------------------------------------------------------------------


class TestDropMissing:
    def test_drops_rows_missing_specific_column(self, sales_df):
        before_missing = int(sales_df["discount_pct"].isna().sum())
        result, _ = apply_transform(sales_df, "drop_missing", {"column": "discount_pct"})
        assert result["discount_pct"].isna().sum() == 0
        assert len(result) == len(sales_df) - before_missing

    def test_drops_rows_missing_any_column(self, survey_df):
        result, _ = apply_transform(survey_df, "drop_missing", {"column": ""})
        assert result.isna().sum().sum() == 0

    def test_no_missing_values_still_succeeds(self, tiny_df):
        result, _ = apply_transform(tiny_df, "drop_missing", {"column": "score"})
        assert len(result) == len(tiny_df)

    @pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
    def test_smoke_across_all_fixtures(self, request, fixture_name):
        df = request.getfixturevalue(fixture_name)
        result, _ = apply_transform(df, "drop_missing", {"column": ""})
        assert len(result) <= len(df)


class TestFillMissing:
    def test_mean_fill(self, sales_df):
        result, _ = apply_transform(sales_df, "fill_missing", {"column": "discount_pct", "strategy": "mean"})
        assert result["discount_pct"].isna().sum() == 0
        assert len(result) == len(sales_df)

    def test_median_fill(self, sales_df):
        result, _ = apply_transform(sales_df, "fill_missing", {"column": "discount_pct", "strategy": "median"})
        assert result["discount_pct"].isna().sum() == 0

    def test_mode_fill_categorical(self, survey_df):
        result, _ = apply_transform(survey_df, "fill_missing", {"column": "satisfaction", "strategy": "mode"})
        assert result["satisfaction"].isna().sum() == 0

    def test_constant_fill(self, survey_df):
        result, _ = apply_transform(
            survey_df, "fill_missing", {"column": "satisfaction", "strategy": "constant", "constant_value": "No Response"}
        )
        assert (result.loc[survey_df["satisfaction"].isna(), "satisfaction"] == "No Response").all()

    def test_mean_on_non_numeric_raises(self, survey_df):
        with pytest.raises(TransformError, match="numeric"):
            apply_transform(survey_df, "fill_missing", {"column": "satisfaction", "strategy": "mean"})

    def test_shape_unchanged(self, sales_df):
        result, _ = apply_transform(sales_df, "fill_missing", {"column": "discount_pct", "strategy": "mean"})
        assert result.shape == sales_df.shape


class TestDropOutliers:
    def test_removes_injected_outlier(self, sales_df):
        result, _ = apply_transform(sales_df, "drop_outliers", {"column": "revenue"})
        assert len(result) < len(sales_df)
        assert result["revenue"].max() < sales_df["revenue"].max()

    def test_non_numeric_raises(self, survey_df):
        with pytest.raises(TransformError, match="numeric"):
            apply_transform(survey_df, "drop_outliers", {"column": "satisfaction"})

    @pytest.mark.parametrize("fixture_name", ["sales_df", "survey_df", "events_df"])
    def test_smoke_across_fixtures(self, request, fixture_name):
        df = request.getfixturevalue(fixture_name)
        numeric_cols = df.select_dtypes(include="number").columns
        assert len(numeric_cols) > 0
        result, _ = apply_transform(df, "drop_outliers", {"column": numeric_cols[0]})
        assert len(result) <= len(df)


# ---------------------------------------------------------------------------
# Rename / cast / drop column
# ---------------------------------------------------------------------------


class TestRenameColumn:
    def test_renames_column(self, sales_df):
        result, _ = apply_transform(sales_df, "rename_column", {"column": "units", "new_name": "quantity"})
        assert "quantity" in result.columns and "units" not in result.columns
        assert list(result["quantity"]) == list(sales_df["units"])

    def test_collision_raises(self, sales_df):
        with pytest.raises(TransformError, match="already exists"):
            apply_transform(sales_df, "rename_column", {"column": "units", "new_name": "revenue"})

    def test_blank_name_raises(self, sales_df):
        with pytest.raises(TransformError, match="required"):
            apply_transform(sales_df, "rename_column", {"column": "units", "new_name": "   "})


class TestCastType:
    def test_numeric_to_string(self, survey_df):
        result, _ = apply_transform(survey_df, "cast_type", {"column": "years_customer", "target_type": "string"})
        # Modern pandas may use its dedicated StringDtype instead of legacy `object` for
        # astype(str) - check string semantics rather than a specific dtype object.
        assert pd.api.types.is_string_dtype(result["years_customer"])

    def test_string_to_numeric_reports_new_missing(self, events_df):
        result, desc = apply_transform(events_df, "cast_type", {"column": "city", "target_type": "numeric"})
        assert result["city"].isna().all()
        assert "missing" in desc.lower()

    def test_datetime_cast(self, events_df):
        result, _ = apply_transform(events_df, "cast_type", {"column": "event_date", "target_type": "datetime"})
        assert pd.api.types.is_datetime64_any_dtype(result["event_date"])

    def test_category_cast(self, sales_df):
        result, _ = apply_transform(sales_df, "cast_type", {"column": "region", "target_type": "category"})
        assert isinstance(result["region"].dtype, pd.CategoricalDtype)

    def test_unsupported_target_raises(self, sales_df):
        with pytest.raises(TransformError, match="Unsupported target"):
            apply_transform(sales_df, "cast_type", {"column": "region", "target_type": "complex128"})


class TestDropColumn:
    def test_drops_column(self, sales_df):
        result, _ = apply_transform(sales_df, "drop_column", {"column": "discount_pct"})
        assert "discount_pct" not in result.columns
        assert result.shape[1] == sales_df.shape[1] - 1

    def test_cannot_drop_only_column(self):
        single_col_df = pd.DataFrame({"only": [1, 2, 3]})
        with pytest.raises(TransformError, match="only remaining"):
            apply_transform(single_col_df, "drop_column", {"column": "only"})


# ---------------------------------------------------------------------------
# Combine / merge columns
# ---------------------------------------------------------------------------


class TestCombineColumns:
    def test_concat_two_text_columns(self, events_df):
        result, _ = apply_transform(
            events_df,
            "combine_columns",
            {"columns": ["first_name", "last_name"], "operation": "concat", "separator": " ", "new_name": "full_name"},
        )
        assert "full_name" in result.columns
        assert result.loc[0, "full_name"] == f"{events_df.loc[0, 'first_name']} {events_df.loc[0, 'last_name']}"

    def test_sum_two_numeric_columns(self, sales_df):
        df = sales_df.assign(bonus_units=sales_df["units"] * 2)
        result, _ = apply_transform(
            df, "combine_columns", {"columns": ["units", "bonus_units"], "operation": "sum", "new_name": "total_units"}
        )
        assert np.allclose(result["total_units"], df["units"] + df["bonus_units"])

    def test_coalesce_prefers_first_non_null(self):
        df = pd.DataFrame({"a": [1, None, None], "b": [None, 2, None], "c": [None, None, 3]})
        result, _ = apply_transform(df, "combine_columns", {"columns": ["a", "b", "c"], "operation": "coalesce", "new_name": "merged"})
        assert list(result["merged"]) == [1.0, 2.0, 3.0]

    def test_sum_on_non_numeric_raises(self, events_df):
        with pytest.raises(TransformError, match="numeric"):
            apply_transform(events_df, "combine_columns", {"columns": ["first_name", "last_name"], "operation": "sum", "new_name": "x"})

    def test_needs_at_least_two_columns(self, sales_df):
        with pytest.raises(TransformError, match="at least two"):
            apply_transform(sales_df, "combine_columns", {"columns": ["units"], "operation": "concat", "new_name": "x"})

    def test_name_collision_raises(self, sales_df):
        with pytest.raises(TransformError, match="already exists"):
            apply_transform(sales_df, "combine_columns", {"columns": ["units", "revenue"], "operation": "sum", "new_name": "revenue"})


# ---------------------------------------------------------------------------
# Chaining - dataset-editor cards apply transforms sequentially on top of the
# previous result, so this exercises that composition explicitly.
# ---------------------------------------------------------------------------


class TestChainedTransforms:
    def test_filter_then_drop_column_then_fill(self, sales_df):
        step1, _ = apply_transform(sales_df, "filter", {"column": "units", "comparator": ">", "value": "1"})
        step2, _ = apply_transform(step1, "drop_column", {"column": "order_id"})
        step3, _ = apply_transform(step2, "fill_missing", {"column": "discount_pct", "strategy": "median"})
        assert "order_id" not in step3.columns
        assert step3["discount_pct"].isna().sum() == 0
        assert (step3["units"] > 1).all()
        assert len(step3) <= len(sales_df)

    def test_unknown_op_type_raises(self, sales_df):
        with pytest.raises(TransformError, match="Unknown transform"):
            apply_transform(sales_df, "not_a_real_op", {})

    def test_empty_dataframe_raises(self):
        with pytest.raises(TransformError, match="empty"):
            apply_transform(pd.DataFrame(), "filter", {"column": "x", "comparator": ">", "value": "1"})
