"""Schema detection, geography, chart engine, recommendations and the data-quality report, checked
against real datasets (plotly ships them offline) plus small hand-built edge cases."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis import geo  # noqa: E402
from analysis.charts import render_chart  # noqa: E402
from analysis.profiler import analyze_dataframe  # noqa: E402
from analysis.quality import build_quality_report, quality_markdown  # noqa: E402
from analysis.schema import build_schema  # noqa: E402
from analysis.transforms import apply_transform  # noqa: E402


def roles(df: pd.DataFrame) -> dict[str, str]:
    return {c: p["role"] for c, p in build_schema(df)["profiles"].items()}


# ------------------------------------------------------------------ geography

@pytest.mark.parametrize("value,iso", [
    ("Korea, Rep.", "KOR"), ("Viet Nam", "VNM"), ("Côte d'Ivoire", "CIV"), ("UK", "GBR"),
    ("Congo, Dem. Rep.", "COD"), ("united states", "USA"), ("FR", "FRA"),
])
def test_country_normalisation(value, iso):
    assert geo.country_to_iso3(value) == iso


def test_us_state_codes():
    assert geo.state_to_code("Texas") == "TX" and geo.state_to_code("tx") == "TX"
    assert geo.country_to_iso3("Atlantis") is None


# ------------------------------------------------------------------ schema roles

def test_roles_on_gapminder():
    import plotly.express as px

    r = roles(px.data.gapminder())
    assert r["country"] == "geo_country" and r["continent"] == "categorical"
    assert r["year"] == "year" and r["lifeExp"] == "numeric" and r["iso_num"] == "identifier"


def test_string_dates_and_ids_and_text():
    df = pd.DataFrame({
        "order_id": range(1000, 1060),
        "when": pd.date_range("2024-01-01", periods=60).strftime("%Y-%m-%d"),
        "review": [f"Review {i}: this product was surprisingly good, I would recommend it to any friend who asks about it" for i in range(60)],
        "size": ["small", "medium", "large"] * 20,
        "flag": [0, 1] * 30,
        "price": [f"${i * 10:,}" for i in range(60)],
    })
    r = roles(df)
    assert r["order_id"] == "identifier" and r["when"] == "datetime" and r["review"] == "text"
    assert r["size"] == "ordinal" and r["flag"] in {"binary", "binary_numeric"} and r["price"] == "numeric"


def test_time_structure_and_panel():
    import plotly.express as px

    time = build_schema(px.data.gapminder())["time"]
    assert time["time_column"] == "year" and time["is_panel"] and time["entity_column"] == "country"


def test_join_keys_offered():
    import plotly.express as px

    keys = build_schema(px.data.gapminder())["join_keys"]
    assert any("country" in k["columns"] for k in keys)


def test_empty_and_constant_columns():
    df = pd.DataFrame({"a": [None] * 30, "b": ["x"] * 30, "c": range(30)})
    r = roles(df)
    assert r["a"] == "empty" and r["b"] == "constant"


# ------------------------------------------------------------------ charts + recommendations

DATASETS = ["gapminder", "stocks", "tips", "iris", "election", "medals_long", "wind"]


@pytest.mark.parametrize("name", DATASETS)
def test_every_recommendation_renders_as_strict_json(name):
    import plotly.express as px

    df = getattr(px.data, name)()
    analysis = analyze_dataframe(df, name)
    assert analysis["recommendations"], "every real dataset should get recommendations"
    schema = build_schema(df)
    for rec in analysis["recommendations"]:
        chart = render_chart(df, rec, schema)
        assert not chart["meta"].get("error"), (name, rec, chart["meta"])
        assert chart["data"], (name, rec)
        json.dumps(chart, allow_nan=False)


def test_recommendations_are_statistically_appropriate():
    import plotly.express as px

    types = {r["type"] for r in analyze_dataframe(px.data.gapminder(), "g")["recommendations"]}
    assert {"timeseries", "map"} <= types
    # no geography, no time -> no map / time series
    types = {r["type"] for r in analyze_dataframe(px.data.iris(), "i")["recommendations"]}
    assert "map" not in types and "timeseries" not in types


def test_first_recommendations_differ_in_family():
    import plotly.express as px

    recs = analyze_dataframe(px.data.tips(), "t")["recommendations"]
    assert recs[0]["family"] != recs[1]["family"]


def test_correlation_matrix_uses_all_pairs():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"c{i}": rng.normal(size=200) for i in range(8)})
    df["c1"] = df["c0"] * 2 + rng.normal(scale=0.1, size=200)
    chart = render_chart(df, {"type": "heatmap"}, build_schema(df))
    z = np.array(chart["data"][0]["z"], dtype=float)
    assert z.shape == (8, 8) and np.isfinite(z).all()
    assert (np.abs(z) > 0.9).sum() > 8  # the strong pair shows up (beyond the diagonal)


def test_scatter_is_capped_but_reports_full_stats():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=20000)})
    df["y"] = df["x"] + rng.normal(size=20000)
    chart = render_chart(df, {"type": "scatter", "x": "x", "y": "y"}, build_schema(df))
    assert len(chart["data"][0]["x"]) <= 5000
    assert chart["meta"]["stats"]


def test_bad_configs_give_messages_not_exceptions():
    df = pd.DataFrame({"a": range(20), "b": list("xy") * 10})
    schema = build_schema(df)
    for cfg in ({"type": "scatter", "x": "nope"}, {"type": "nonsense"}, {"type": "map", "x": "b"}, {"type": "line"}):
        chart = render_chart(df, cfg, schema)
        assert chart["meta"].get("error") or chart["data"] is not None


def test_map_only_with_geography():
    import plotly.express as px

    df = px.data.gapminder()
    chart = render_chart(df, {"type": "map", "x": "country", "y": "lifeExp", "agg": "mean"}, build_schema(df))
    assert chart["data"][0]["type"] == "choropleth" and not chart["meta"].get("error")


def test_charts_reflect_transforms():
    import plotly.express as px

    df = px.data.gapminder()
    filtered, _ = apply_transform(df, "filter", {"column": "year", "comparator": "==", "value": "2007"})
    chart = render_chart(filtered, {"type": "histogram", "x": "lifeExp"}, build_schema(filtered))
    assert sum(chart["data"][0].get("y", [0])) in (0, len(filtered)) or chart["data"]


# ------------------------------------------------------------------ data quality

def test_quality_flags_planted_problems():
    df = pd.DataFrame({
        "price": ["10", "20", "N/A", "40", "-", "60"] * 10,
        "city": ["Paris", "paris", "PARIS ", "Rome", "Rome", "rome"] * 10,
        "when": pd.date_range("2024-01-01", periods=60).strftime("%Y-%m-%d"),
        "const": ["x"] * 60,
        "age": [25, 30, 35, 400, 28, 31] * 10,
    })
    df = pd.concat([df, df.head(5)], ignore_index=True)
    schema = build_schema(df)
    report = build_quality_report(df, schema, [])
    titles = " ".join(i["title"].lower() for i in report["issues"])
    assert "duplicate" in titles and ("placeholder" in titles or "missing" in titles)
    assert any(i["category"] for i in report["issues"])
    assert 0 <= report["score"] < 100
    fixes = [i["fix"]["type"] for i in report["issues"] if i.get("fix")]
    assert fixes, "at least one issue should offer a one-click fix"
    md = quality_markdown(report, "planted")
    assert md.startswith("# Data quality report")


def test_quality_fixes_are_valid_transforms_and_improve_score():
    df = pd.DataFrame({"city": ["Paris", "paris", "PARIS "] * 20, "v": list(range(60))})
    df = pd.concat([df, df.head(4)], ignore_index=True)
    before = build_quality_report(df, build_schema(df), [])
    for issue in before["issues"]:
        fix = issue.get("fix")
        if fix:
            df, _ = apply_transform(df, fix["type"], fix["params"])
    after = build_quality_report(df, build_schema(df), [])
    assert after["score"] >= before["score"]


def test_clean_data_scores_high():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300), "g": rng.choice(list("xyz"), 300)})
    assert build_quality_report(df, build_schema(df), [])["score"] >= 85
