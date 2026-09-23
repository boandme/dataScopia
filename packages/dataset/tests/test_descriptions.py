from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.descriptions import infer_column_description  # noqa: E402

_NUMERIC = pd.Series([1, 2, 3])
_TEXT = pd.Series(["a", "b", "c"])
_BOOL = pd.Series([True, False, True])


class TestSubjectAwareInference:
    """The core requirement: a compound name like "school_name" must name the entity it belongs
    to (school), not just the generic attribute category (name) - the bug this replaced always
    described every "*_name" column identically regardless of what came before it."""

    def test_school_name_mentions_school(self):
        desc = infer_column_description("school_name", _TEXT)
        assert "school" in desc.lower()
        assert "name" in desc.lower()

    @pytest.mark.parametrize(
        "column_name,expected_words",
        [
            ("employee_id", ["employee"]),
            ("customer_email", ["customer", "email"]),
            ("product_price", ["product"]),
            ("order_date", ["order", "date"]),
            ("patient_age", ["patient", "age"]),
            ("city_name", ["city", "name"]),
        ],
    )
    def test_subject_word_is_preserved(self, column_name, expected_words):
        desc = infer_column_description(column_name, _TEXT if "name" in column_name or "date" in column_name or "email" in column_name else _NUMERIC).lower()
        for word in expected_words:
            assert word in desc

    def test_rightmost_token_wins_the_attribute_role(self):
        # "city_name" must read as "name of the city", not "city of the name" - the last word is
        # the conventional attribute in "{subject}_{attribute}" naming.
        desc = infer_column_description("city_name", _TEXT).lower()
        assert "name of the city" in desc

    def test_modifier_word_is_prepended_not_treated_as_subject(self):
        desc = infer_column_description("average_rating", _NUMERIC).lower()
        assert "average rating" in desc


class TestCompoundOverrides:
    def test_zip_code_not_described_as_code_of_the_zip(self):
        desc = infer_column_description("zip_code", _TEXT).lower()
        assert "code of the zip" not in desc
        assert "postal" in desc or "zip" in desc

    def test_first_name_recognized_as_a_name_field(self):
        desc = infer_column_description("first_name", _TEXT).lower()
        assert "name" in desc

    def test_phone_number_not_swallowed_by_generic_code_group(self):
        desc = infer_column_description("phone_number", _TEXT).lower()
        assert "phone" in desc


class TestBooleanNames:
    def test_is_prefix_reads_grammatically(self):
        desc = infer_column_description("is_active", _BOOL)
        assert desc == "Likely a true/false flag, based on its name, indicating whether each record is active."

    def test_has_prefix_reads_grammatically(self):
        desc = infer_column_description("has_subscription", _BOOL)
        assert "has subscription" in desc


class TestDeterminism:
    def test_same_input_always_produces_same_output(self):
        results = {infer_column_description("customer_id", _NUMERIC) for _ in range(20)}
        assert len(results) == 1

    def test_never_raises_on_empty_or_unusual_names(self):
        for name in ("", "___", "123", "a", "a_b_c_d_e_f_g"):
            infer_column_description(name, _TEXT)  # must not raise


class TestAccuracyBar:
    """Empirically locks in the >=80% specific-description bar the algorithm was built to hit,
    against a representative sample of real-world Kaggle-style column names spanning many
    domains (e-commerce, travel, sports, finance, health, tech)."""

    COMMON_COLUMN_NAMES = [
        "school_name", "student_id", "teacher_name", "employee_id", "customer_email",
        "product_price", "order_date", "total_amount", "average_rating", "customer_rating",
        "user_name", "company_name", "hospital_name", "patient_age", "doctor_id",
        "movie_title", "book_author", "song_duration", "album_name", "artist_name",
        "city_name", "country_code", "zip_code", "shipping_address", "billing_address",
        "order_status", "payment_method", "discount_percent", "tax_amount", "product_category",
        "review_text", "review_score", "num_reviews", "purchase_count", "signup_date",
        "account_balance", "credit_score", "loan_amount", "interest_rate", "flight_number",
        "departure_city", "arrival_city", "ticket_price", "passenger_name", "hotel_name",
        "room_type", "booking_status", "guest_count", "restaurant_name", "cuisine_type",
        "driver_id", "engine_size", "fuel_type", "manufacture_year", "team_name",
        "player_name", "match_date", "wind_speed", "unemployment_rate", "population_total",
        "literacy_rate", "is_active", "has_discount", "customer_id", "transaction_id",
        "invoice_number", "first_name", "last_name", "phone_number", "postal_code",
        "salary_usd", "job_title", "project_name", "task_status", "due_date",
        "release_year", "imdb_rating", "word_count", "like_count", "follower_count",
        "row_id", "record_id", "session_id", "device_type", "browser_name", "error_code",
    ]

    def test_at_least_80_percent_get_a_specific_not_generic_description(self):
        generic_markers = ("isn't documented",)
        specific_count = 0
        for name in self.COMMON_COLUMN_NAMES:
            is_texty = any(
                w in name
                for w in ("date", "name", "city", "title", "type", "status", "method", "code", "brand", "cuisine", "model", "fuel")
            )
            series = _TEXT if is_texty else _NUMERIC
            desc = infer_column_description(name, series)
            if not any(marker in desc for marker in generic_markers):
                specific_count += 1
        accuracy = specific_count / len(self.COMMON_COLUMN_NAMES)
        assert accuracy >= 0.80, f"Only {accuracy:.1%} of common column names got a specific description (need >=80%)"
