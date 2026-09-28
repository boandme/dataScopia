"""The PDF report: the statistics behind it, and the assembled document."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis import report_stats as RS  # noqa: E402
from analysis.profiler import analyze_dataframe  # noqa: E402
from analysis.report_pdf import build_report_pdf  # noqa: E402
from analysis.schema import build_schema  # noqa: E402


def _text(pdf: bytes) -> str:
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    return "\n".join(page.extract_text() for page in reader.pages)


def _pages(pdf: bytes) -> int:
    pypdf = pytest.importorskip("pypdf")
    return len(pypdf.PdfReader(io.BytesIO(pdf)).pages)


# ------------------------------------------------------------------ statistics

def test_correlations_are_adjusted_and_ranked():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300), "c": rng.normal(size=300)})
    df["d"] = df["a"] * 2 + rng.normal(scale=0.2, size=300)
    out = RS.correlations(df, build_schema(df))
    top = out["pairs"][0]
    assert {top["a"], top["b"]} == {"a", "d"} and top["significant"] and top["strength"] == "very strong"
    assert out["tests"] == 6 and all(r["p_adj"] >= r["p"] for r in out["pairs"])


def test_group_tests_report_effect_size():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"g": np.repeat(list("abc"), 60), "y": np.concatenate([rng.normal(m, 1, 60) for m in (0, 0, 3)])})
    row = RS.group_tests(df, build_schema(df), [])["rows"][0]
    assert row["significant"] and row["eta2"] > 0.3 and row["effect"] == "large" and row["high"][0] == "c"


def test_model_beats_baseline_on_signal_and_not_on_noise():
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    df = pd.DataFrame({"x": x, "noise": rng.normal(size=400), "label": np.where(x > 0, "yes", "no")})
    schema = build_schema(df)
    model = RS.model_results(df, schema, {"column": "label", "kind": "classification"})
    assert model and model["improvement"] > 0.3 and model["importance"][0][0] == "x"


def test_fingerprint_changes_with_data():
    a = pd.DataFrame({"x": [1, 2, 3]})
    assert RS.data_fingerprint(a) == RS.data_fingerprint(a.copy())
    assert RS.data_fingerprint(a) != RS.data_fingerprint(pd.DataFrame({"x": [1, 2, 4]}))


def test_number_formatting():
    assert RS.fmt_num(1_320_000_000.0) == "1.32B" and RS.fmt_num(29_600_000.0) == "29.60M"
    assert RS.fmt_p(0.0002) == "< 0.001" and RS.fmt_p(0.0412) == "0.041"


# ------------------------------------------------------------------ document

@pytest.mark.parametrize("name", ["gapminder", "election", "tips", "iris", "stocks"])
def test_report_builds_for_real_datasets(name):
    import plotly.express as px

    df = getattr(px.data, name)()
    pdf = build_report_pdf(name, df, analyze_dataframe(df, name), None, [], [])
    assert pdf.startswith(b"%PDF") and len(pdf) > 8_000
    assert 3 <= _pages(pdf) <= 12  # compact by design


def test_report_has_the_expected_sections_and_numbered_tables():
    import plotly.express as px

    df = px.data.gapminder()
    pdf = build_report_pdf("gapminder", df, analyze_dataframe(df, "gapminder"), "An example dataset.", [], [],
                           dataset_info={"dataset_id": "owner/gapminder", "kaggle_info": {"title": "Gapminder", "license": "CC0: Public Domain"}},
                           transform_log=[{"type": "drop_duplicates", "description": "Dropped 5 duplicate row(s)."}])
    text = _text(pdf)
    for heading in ("Executive Summary", "Key findings", "Recommended actions", "Introduction and Objectives", "Hypotheses", "Data and Methods",
                    "Data Quality Assessment", "Results", "Discussion", "Limitations and sources of error", "What the data do not show", "Evidence summary",
                    "Conclusions and Recommendations", "References", "Appendix A. Variable dictionary", "Appendix B. Reproducibility record"):
        assert heading in text, heading
    assert "Table 1." in text and "Table 2." in text and "Figure 1." in text
    assert "Kruskal" in text and "Bonferroni" in text and "Data fingerprint" in text
    assert "Dropped 5 duplicate row(s)." in text and "CC0: Public Domain" in text
    # results are separated from interpretation: interpretive language lives in the Discussion
    results = text.split("4. Results")[1].split("5. Discussion")[0]
    assert "does not show that either variable causes" not in results


def test_report_survives_odd_data():
    df = pd.DataFrame({"only_text": ["a", "b", "c", "d"] * 3, "café 中": range(12)})
    pdf = build_report_pdf("odd", df, analyze_dataframe(df, "odd"), None, [], [])
    assert pdf.startswith(b"%PDF")
    tiny = pd.DataFrame({"a": [1, 2]})
    assert build_report_pdf("tiny", tiny, analyze_dataframe(tiny, "tiny"), None, [], []).startswith(b"%PDF")


def test_report_marks_partial_samples():
    import plotly.express as px

    df = px.data.tips()
    analysis = analyze_dataframe(df, "tips", sample_info={"is_sample": True, "estimated_total_rows": 5000, "file_name": "tips.csv"})
    text = _text(build_report_pdf("tips", df, analysis, None, [], []))
    assert "partial sample" in text


def _build(options=None, name="gapminder"):
    import plotly.express as px

    df = getattr(px.data, name)()
    return build_report_pdf(name, df, analyze_dataframe(df, name), None, [], [], options=options)


def test_report_goes_deep_on_correlations_trends_and_nulls():
    text = _text(_build())
    for phrase in ("95% CI", "Simpson", "Variance inflation", "no detectable association", "Kendall", "compound rate", "Post-hoc pairwise",
                   "Holm", "Evidence: ", "Distribution diagnostics", "Min. detectable"):
        assert phrase in text or phrase.lower() in text.lower(), phrase


def test_detail_levels_change_length():
    brief, deep = _pages(_build({"detail": "brief"})), _pages(_build({"detail": "deep"}))
    assert brief < deep


def test_report_options_title_author_sections_theme():
    pdf = _build({"title": "Custom Study Title", "author": "Dr. Ada", "organization": "Lab X", "subtitle": "Wave 2", "notes": "Prepared for the board.",
                  "footer": "Confidential draft", "sections": {"references": False, "appendices": False}, "theme": "teal", "page_size": "a4", "font_size": "compact"})
    text = _text(pdf)
    assert "Custom Study Title" in text and "Dr. Ada" in text and "Lab X" in text and "Wave 2" in text
    assert "Prepared for the board." in text and "Confidential draft" in text
    assert "Appendix A" not in text and "References" not in text.split("Conclusions and Recommendations")[-1]
    import pypdf

    box = pypdf.PdfReader(io.BytesIO(pdf)).pages[0].mediabox
    assert abs(float(box.width) - 595.3) < 2  # A4


def test_report_options_are_validated():
    from analysis.report_pdf import normalize_options

    o = normalize_options({"detail": "nonsense", "theme": "neon", "page_size": "tabloid", "sections": {"bogus": True, "methods": 0}, "title": "x" * 500})
    assert o["detail"] == "deep" and o["theme"] == "indigo" and o["page_size"] == "letter" and o["sections"]["methods"] is False and "bogus" not in o["sections"] and len(o["title"]) == 140
