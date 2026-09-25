"""The notebook must reproduce in-app cleaning exactly, so every generated snippet is executed
and compared with the real transform, and the whole exported notebook is executed end to end."""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.notebook_export import build_notebook  # noqa: E402
from analysis.profiler import analyze_dataframe  # noqa: E402
from analysis.transforms import apply_transform, op_code  # noqa: E402


@pytest.fixture
def messy() -> pd.DataFrame:
    base = pd.DataFrame({
        "id": range(1, 41),
        "price": ["$1,200", "$950", "n/a", "$2,000", "$1,100"] * 8,
        "city": ["Paris", "paris ", "PARIS", "Rome", "rome"] * 8,
        "score": [1.0, 2.0, np.nan, 4.0, 5.0, 100.0, 2.5, 3.5] * 5,
        "a": list(range(40)),
        "b": list(range(40, 80)),
    })
    return pd.concat([base, base.head(3)], ignore_index=True)


OPS = [
    ("filter", {"column": "a", "comparator": ">", "value": "5"}),
    ("filter", {"column": "city", "comparator": "==", "value": "Rome"}),
    ("drop_missing", {"column": "score"}),
    ("drop_missing", {"column": "__all__"}),
    ("fill_missing", {"column": "score", "strategy": "mean"}),
    ("fill_missing", {"column": "score", "strategy": "median"}),
    ("fill_missing", {"column": "city", "strategy": "mode"}),
    ("fill_missing", {"column": "score", "strategy": "constant", "constant_value": "0"}),
    ("drop_outliers", {"column": "score"}),
    ("rename_column", {"column": "a", "new_name": "alpha"}),
    ("cast_type", {"column": "price", "target_type": "numeric"}),
    ("cast_type", {"column": "a", "target_type": "string"}),
    ("drop_column", {"column": "b"}),
    ("combine_columns", {"columns": ["a", "b"], "new_name": "ab", "operation": "sum"}),
    ("combine_columns", {"columns": ["city", "id"], "new_name": "ci", "operation": "concat", "separator": "-"}),
    ("drop_duplicates", {}),
    ("replace_missing_tokens", {"column": "price", "values": ["n/a"]}),
    ("standardize_text", {"column": "city"}),
]


@pytest.mark.parametrize("op_type,params", OPS, ids=[f"{o}-{i}" for i, (o, _) in enumerate(OPS)])
def test_generated_code_matches_transform(messy, op_type, params):
    expected, _ = apply_transform(messy, op_type, params)
    scope = {"df": messy.copy(), "pd": pd, "np": np}
    exec(op_code(messy, op_type, params), scope)
    pd.testing.assert_frame_equal(scope["df"].reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)


def _run_notebook(nb: dict) -> dict:
    import plotly.graph_objects as go

    go.Figure.show = lambda self, *a, **k: None  # no browser in tests
    scope: dict = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile("".join(cell["source"]), cell["id"], "exec"), scope)
    return scope


def test_notebook_replays_cleaning_and_runs():
    import nbformat
    import plotly.express as px

    original = px.data.gapminder()
    log = [
        {"type": "drop_duplicates", "params": {}},
        {"type": "filter", "params": {"column": "year", "comparator": ">=", "value": "1970"}},
        {"type": "rename_column", "params": {"column": "pop", "new_name": "population"}},
    ]
    current = original
    for step in log:
        current, _ = apply_transform(current, step["type"], step["params"])
    nb = build_notebook("Gapminder", [], analyze_dataframe(current, "gapminder"), df=current,
                        original_df=original, transform_log=log, dataset={"name": "gapminder"})
    nbformat.validate(nbformat.from_dict(nb))
    scope = _run_notebook(nb)
    pd.testing.assert_frame_equal(scope["df"].reset_index(drop=True), current.reset_index(drop=True), check_dtype=False)
    text = " ".join("".join(c["source"]) for c in nb["cells"])
    for heading in ("Data source and license", "Loading the dataset", "Data quality", "Cleaning steps",
                    "Summary statistics", "Visualizations", "Statistical analysis", "Predictive modelling"):
        assert heading in text


def test_notebook_has_model_when_target_exists():
    import plotly.express as px

    df = px.data.election()
    nb = build_notebook("Election", [], analyze_dataframe(df, "election"), df=df, original_df=df, dataset={"name": "election"})
    text = " ".join("".join(c["source"]) for c in nb["cells"])
    assert "RandomForest" in text
    scope = _run_notebook(nb)
    assert "results" in scope


def test_notebook_falls_back_to_preview():
    df = pd.DataFrame({"a": range(30), "b": list("xyz") * 10})
    summary = analyze_dataframe(df, "small")
    summary["preview"] = df.to_dict(orient="records")
    nb = build_notebook("Small", [], summary)
    assert any("preview" in "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    _run_notebook(nb)


def test_notebook_includes_saved_findings_and_charts():
    df = pd.DataFrame({"g": list("abcd") * 25, "v": np.arange(100.0)})
    findings = [{"type": "question", "title": "Q", "question": "mean?", "answer": "49.5", "code": "df['v'].mean()"},
                {"type": "chart", "title": "Bar", "chart": {"type": "bar", "x": "g", "y": "v", "agg": "mean"}}]
    nb = build_notebook("T", findings, analyze_dataframe(df, "t"), df=df, original_df=df,
                        charts=[{"type": "box", "x": "g", "y": "v"}], dataset={"name": "t"})
    _run_notebook(nb)
