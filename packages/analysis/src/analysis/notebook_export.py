from __future__ import annotations

import json
from typing import Any


def _lines(*parts: str) -> list[str]:
    return [p if p.endswith("\n") else p + "\n" for p in parts if p]


def _markdown_cell(*parts: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(*parts)}


def _code_cell(*parts: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(*parts)}


def _dataset_loading_cell(analysis_summary: dict[str, Any]) -> dict[str, Any]:
    preview = analysis_summary.get("preview") or []
    dataset_name = analysis_summary.get("dataset_name", "dataset")
    sample = analysis_summary.get("sample") or {}
    parts = ["import pandas as pd\n\n"]
    if sample.get("file_name"):
        parts.append(f"# Original source file: {sample['file_name']}\n")
    note = f"# {len(preview):,} rows captured from the analyzed sample of '{dataset_name}'.\n"
    if sample.get("is_sample"):
        note += "# This was a partial sample, not the full dataset - re-run 'Run full analysis' in the workspace for complete data.\n"
    parts.append(note)
    parts.append("records = " + json.dumps(preview) + "\n")
    parts.append("df = pd.DataFrame(records)\n")
    parts.append("df.head()\n")
    return _code_cell(*parts)


def _snapshot_markdown(analysis_summary: dict[str, Any]) -> dict[str, Any]:
    snap = analysis_summary.get("research_snapshot") or {}
    return _markdown_cell(
        "## Research Snapshot\n\n",
        f"- **{snap.get('rows', 0):,}** observations, **{snap.get('columns', 0)}** variables\n",
        f"- **{snap.get('missing_percentage', 0)}%** missing data across **{snap.get('missing_columns', 0)}** columns\n",
        f"- **{snap.get('duplicate_rows', 0)}** duplicate rows\n",
        f"- **{snap.get('highly_skewed_fields', 0)}** highly skewed variables\n",
        f"- **{snap.get('potential_targets', 0)}** potential target variables\n",
        f"- **{snap.get('notable_relationships', 0)}** notable correlations\n",
        f"- **{snap.get('data_quality_issues', 0)}** data-quality issues flagged\n",
    )


def _is_directly_runnable(code: str) -> bool:
    stripped = code.strip()
    return stripped.startswith("df") or stripped.startswith("(df")


def _chart_finding_code(chart: dict[str, Any]) -> str:
    """Best-effort translation of a chart's on-screen config (type/x/y/aggregate/group/sort)
    into runnable plotly express code, so a saved chart reproduces in the exported notebook
    instead of just being described in prose."""
    ctype = chart.get("type") or "histogram"
    x = chart.get("x")
    y = chart.get("y")
    agg = chart.get("agg") or "none"
    group = chart.get("group") or None
    sort = chart.get("sort") or "none"
    title = chart.get("customTitle") or ""

    lines = ["import plotly.express as px", ""]
    working_df = "df"
    plot_x, plot_y = x, y

    if agg != "none" and x:
        group_cols = [x] + ([group] if group else [])
        if agg == "count":
            lines.append(f"agg_df = df.groupby({group_cols!r})[{group_cols[0]!r}].count()")
            lines.append("agg_df = agg_df.rename('count').reset_index()")
            plot_y = "count"
        else:
            lines.append(f"agg_df = df.groupby({group_cols!r})[{y!r}].{agg}().reset_index()")
            plot_y = y
        if sort in ("asc", "desc"):
            lines.append(f"agg_df = agg_df.sort_values({plot_y!r}, ascending={sort == 'asc'})")
        working_df = "agg_df"

    color_kw = f", color={group!r}" if (group and agg == "none") else ""
    title_kw = f", title={title!r}" if title else ""

    if ctype == "histogram":
        lines.append(f"fig = px.histogram({working_df}, x={plot_x!r}{title_kw})")
    elif ctype == "pie":
        values_kw = f", values={plot_y!r}" if agg != "none" else ""
        lines.append(f"fig = px.pie({working_df}, names={plot_x!r}{values_kw}{title_kw})")
    elif ctype in ("box", "violin"):
        lines.append(f"fig = px.{ctype}({working_df}, x={plot_x!r}, y={plot_y!r}{color_kw}{title_kw})")
    elif ctype == "heatmap":
        lines.append("corr = df.select_dtypes(include='number').corr()")
        lines.append(f"fig = px.imshow(corr, color_continuous_scale='Viridis'{title_kw})")
    elif ctype in ("bar", "stacked_bar"):
        lines.append(f"fig = px.bar({working_df}, x={plot_x!r}, y={plot_y!r}{color_kw}{title_kw})")
    elif ctype == "line":
        lines.append(f"fig = px.line({working_df}, x={plot_x!r}, y={plot_y!r}{color_kw}{title_kw})")
    elif ctype == "area":
        lines.append(f"fig = px.area({working_df}, x={plot_x!r}, y={plot_y!r}{color_kw}{title_kw})")
    else:
        lines.append(f"fig = px.scatter({working_df}, x={plot_x!r}, y={plot_y!r}{color_kw}{title_kw})")
    lines.append("fig.show()")
    return "\n".join(lines) + "\n"


def build_notebook(title: str, findings: list[dict[str, Any]], analysis_summary: dict[str, Any]) -> dict[str, Any]:
    notebook: dict[str, Any] = {
        "cells": [
            _markdown_cell(
                f"# {title}\n\n",
                "This notebook was generated from the Kaggle Dataset Analysis workspace. It reproduces the "
                "dataset sample, summary statistics, and saved findings from that session.\n",
            ),
            _dataset_loading_cell(analysis_summary),
            _snapshot_markdown(analysis_summary),
            _code_cell("df.describe(include='all').T\n"),
            _code_cell("df.isna().sum().sort_values(ascending=False)\n"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    if findings:
        notebook["cells"].append(_markdown_cell("## Saved Findings\n"))

    for index, finding in enumerate(findings, start=1):
        ftype = finding.get("type")
        title_text = finding.get("title") or f"Finding {index}"

        if ftype == "question":
            notebook["cells"].append(
                _markdown_cell(
                    f"### Finding {index}: {title_text}\n\n",
                    f"**Question:** {finding.get('question', '')}\n\n",
                    f"**Answer:** {finding.get('answer', '')}\n",
                )
            )
            code = finding.get("code") or ""
            if code and _is_directly_runnable(code):
                notebook["cells"].append(
                    _code_cell("# Reproduces the computed answer above\n", f"result = {code}\n", "result\n")
                )
            else:
                notebook["cells"].append(
                    _code_cell(
                        f"# Computed via method '{finding.get('method', 'unknown')}' on the loaded dataframe.\n",
                        "# See the Answer above for the result; this method used intermediate steps not\n",
                        "# reducible to a single expression.\n",
                    )
                )
        elif ftype == "chart" and finding.get("chart"):
            notebook["cells"].append(_markdown_cell(f"### Finding {index}: {title_text}\n"))
            notebook["cells"].append(_code_cell(_chart_finding_code(finding["chart"])))
        else:
            notebook["cells"].append(
                _markdown_cell(f"### Finding {index}: {title_text}\n\n", f"{finding.get('summary', '')}\n")
            )
            notebook["cells"].append(_code_cell(f"print({json.dumps(finding.get('summary', ''))})\n"))

    return notebook
