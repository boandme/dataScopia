"""Build a complete, runnable Jupyter notebook for a dataset session.

The notebook is meant to stand on its own: someone who has never seen the workspace can open it,
run every cell top to bottom, and get the same cleaned dataframe, the same charts and the same
statistics the workspace showed - with explanatory text in between. Sections:

  1. Overview and how to run          6. Summary statistics
  2. Data source and license          7. Visualizations
  3. Loading the dataset              8. Statistical analysis
  4. Data quality (before cleaning)   9. Predictive modelling (when there is a plausible target)
  5. Cleaning steps (replayed)       10. Findings saved in the workspace, and conclusions

Design choices worth knowing:
  * Cleaning is *replayed*: every in-app transform is stored as (type, params); here each one is
    re-generated as pandas code (``transforms.op_code``) against the frame as it stood before the
    step, so the notebook reproduces the exact cleaned data - not just the raw sample.
  * The data is either read from a local file (drop the original file next to the notebook) or,
    failing that, from an embedded gzip+base64 CSV snapshot of the original rows, so the notebook
    always runs even offline. A snapshot larger than the embed limit is a head sample and says so.
  * Charts use the same aggregation / top-N / trend logic as the workspace (``chart_code``).
"""

from __future__ import annotations

import base64
import gzip
import json
import re
from datetime import datetime
from typing import Any

import pandas as pd

from . import quality as quality_mod
from . import report_deep as report_deep_mod
from . import report_stats as report_stats_mod
from . import schema as schema_mod
from .chart_code import chart_code
from .profiler import analyze_dataframe
from .transforms import TransformError, apply_transform, op_code

MAX_EMBED_ROWS = 10_000
MAX_EMBED_BYTES = 1_500_000
MAX_ML_FEATURES = 25
MAX_NOTEBOOK_CHARTS = 12

_NUMERIC_ROLES = {"numeric", "ordinal_numeric", "binary_numeric", "year", "geo_lat", "geo_lon"}
_CATEGORICAL_ROLES = {"categorical", "binary", "ordinal", "geo_country", "geo_state"}
_SKIP_FEATURE_ROLES = {"identifier", "text", "constant", "empty", "datetime"}

DEFAULT_SECTIONS = {"quality": True, "cleaning": True, "structure": True, "summary": True, "visualizations": True,
                    "statistics": True, "modelling": True, "findings": True, "conclusions": True}
EMBED_CHOICES = (0, 1000, 5000, 10000)


def normalize_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Validated, defaulted notebook options (the browser may send anything)."""
    o = options if isinstance(options, dict) else {}
    ml = o.get("ml") if isinstance(o.get("ml"), dict) else {}

    def num(value: Any, default: float, low: float, high: float) -> float:
        try:
            return max(low, min(high, float(value)))
        except (TypeError, ValueError):
            return default

    embed = o.get("embed_rows")
    return {
        "author": str(o.get("author") or "").strip()[:120],
        "notes": str(o.get("notes") or "").strip()[:1500],
        "sections": {**DEFAULT_SECTIONS, **{k: bool(v) for k, v in (o.get("sections") or {}).items() if k in DEFAULT_SECTIONS}},
        "embed_rows": int(embed) if isinstance(embed, (int, float)) and int(embed) in EMBED_CHOICES else MAX_EMBED_ROWS,
        "chart_source": o.get("chart_source") if o.get("chart_source") in {"both", "workspace", "recommended"} else "both",
        "max_charts": int(num(o.get("max_charts"), MAX_NOTEBOOK_CHARTS, 1, 30)),
        "ml": {"test_size": num(ml.get("test_size"), 0.25, 0.1, 0.5), "seed": int(num(ml.get("seed"), 42, 0, 99999)), "trees": int(num(ml.get("trees"), 200, 20, 600))},
    }


STEP_TITLES = {
    "filter": "Filter rows",
    "drop_missing": "Drop rows with missing values",
    "fill_missing": "Fill missing values",
    "drop_outliers": "Remove outliers",
    "rename_column": "Rename a column",
    "cast_type": "Change a column's type",
    "drop_column": "Drop a column",
    "combine_columns": "Combine columns into a new one",
    "drop_duplicates": "Remove duplicate rows",
    "replace_missing_tokens": "Turn placeholder text into real missing values",
    "standardize_text": "Standardize spelling and spacing",
}


# ----------------------------------------------------------------------------- cell helpers

class _Cells:
    """Accumulates cells and gives each a stable id (required by nbformat 4.5)."""

    def __init__(self) -> None:
        self.cells: list[dict[str, Any]] = []

    @staticmethod
    def _source(text: str) -> list[str]:
        return text.strip("\n").splitlines(keepends=True) or [""]

    def md(self, text: str) -> None:
        self.cells.append({"cell_type": "markdown", "id": f"cell-{len(self.cells):03d}", "metadata": {},
                           "source": self._source(text)})

    def code(self, text: str) -> None:
        self.cells.append({"cell_type": "code", "id": f"cell-{len(self.cells):03d}", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": self._source(text)})


def _role(roles: dict[str, Any], column: str) -> str:
    entry = roles.get(column)
    return entry.get("role", "unknown") if isinstance(entry, dict) else "unknown"


def _numeric_read(roles: dict[str, Any], column: str) -> str:
    entry = roles.get(column) or {}
    if isinstance(entry, dict) and (entry.get("notes") or {}).get("stored_as_text"):
        return f"pd.to_numeric(df[{column!r}].astype(str).str.replace(r'[,$\u20ac\u00a3\u00a5%\\s]', '', regex=True), errors='coerce')"
    return f"df[{column!r}]"


def _md_escape(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


# ----------------------------------------------------------------------------- loading

def _read_call(file_name: str) -> str:
    lower = file_name.lower()
    if lower.endswith((".tsv", ".tab")):
        return "pd.read_csv(DATA_FILE, sep='\\t', encoding_errors='replace', low_memory=False)"
    if lower.endswith((".xlsx", ".xls")):
        return "pd.read_excel(DATA_FILE)"
    if lower.endswith(".parquet"):
        return "pd.read_parquet(DATA_FILE)"
    if lower.endswith(".json"):
        return "pd.read_json(DATA_FILE)"
    return "pd.read_csv(DATA_FILE, encoding_errors='replace', low_memory=False)"


def _embed_snapshot(original: pd.DataFrame, max_rows: int = MAX_EMBED_ROWS) -> tuple[str, int, bool]:
    """gzip+base64 CSV of (up to ``max_rows`` of) the original rows -> (blob, rows, truncated)."""
    if max_rows <= 0:
        return "", 0, True
    frame = original.head(max_rows)
    blob = ""
    while True:
        csv_bytes = frame.to_csv(index=False).encode("utf-8")
        blob = base64.b64encode(gzip.compress(csv_bytes, 9)).decode("ascii")
        if len(blob) <= MAX_EMBED_BYTES or len(frame) <= 200:
            break
        frame = frame.head(max(200, len(frame) // 2))
    return blob, len(frame), len(frame) < len(original)


def _wrap_blob(blob: str, width: int = 100) -> str:
    chunks = [blob[i:i + width] for i in range(0, len(blob), width)]
    body = "\n".join(f"    {chunk!r}" for chunk in chunks)
    return f"_SNAPSHOT_GZ_B64 = (\n{body}\n)"


def _kaggle_slug(dataset: dict[str, Any]) -> str | None:
    dataset_id = str(dataset.get("dataset_id") or "")
    if "/" in dataset_id and not dataset_id.startswith(("local-", "upload-")):
        return dataset_id
    return None


# ----------------------------------------------------------------------------- sections

def _intro(cells: _Cells, title: str, name: str, analysis: dict[str, Any], slug: str | None, opts: dict[str, Any] | None = None, fingerprint: str | None = None) -> None:
    snap = analysis.get("research_snapshot") or {}
    opts = opts or {}
    byline = ""
    if opts.get("author"):
        byline = f"*Prepared by {opts['author']} on {datetime.now().strftime('%d %B %Y')}.*\n\n"
    if opts.get("notes"):
        byline += f"> {opts['notes']}\n\n"
    if fingerprint:
        byline += (f"**Reproducibility fingerprint:** `{fingerprint}` - a hash of this dataset's contents. It matches the fingerprint "
                   "shown on a PDF export of the same analysis, so you can confirm both describe the same data.\n\n")
    cells.md(
        f"# {title}\n\n"
        f"{byline}"
        f"An exploratory analysis of **{name}** ({snap.get('rows', 0):,} rows x {snap.get('columns', 0)} columns), "
        "exported from DataScopia.\n\n"
        "**How to read this notebook.** It runs top to bottom: it loads the data, checks its quality, replays every "
        "cleaning step applied in the workspace, summarizes and visualizes the cleaned data, tests the main "
        "relationships statistically and - when the data suggests a target - fits a baseline predictive model. "
        "Every number and chart below is computed by the code in its own cell, so you can change a cell and re-run it.\n\n"
        "**Requirements:** `pip install pandas numpy scipy scikit-learn plotly`"
        + (" (plus `kagglehub` to download the full dataset)." if slug else ".")
    )
    cells.code(
        "import gzip, base64, io, os\n"
        "import warnings\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import plotly.express as px\n"
        "import plotly.graph_objects as go\n"
        "from scipy import stats\n"
        "\n"
        "warnings.filterwarnings('ignore')              # keep the output readable\n"
        "pd.set_option('display.max_columns', 50)       # show wide tables in full\n"
        "pd.set_option('display.float_format', lambda v: f'{v:,.4g}')"
    )


def _source_and_license(cells: _Cells, name: str, dataset: dict[str, Any]) -> None:
    info = dataset.get("kaggle_info") if isinstance(dataset.get("kaggle_info"), dict) else {}
    slug = _kaggle_slug(dataset)
    lines = ["## 1. Data source and license\n"]
    if slug:
        lines.append(f"- **Source:** Kaggle dataset [`{slug}`](https://www.kaggle.com/datasets/{slug})")
    elif dataset.get("dataset_id", "").startswith("upload-") if isinstance(dataset.get("dataset_id"), str) else False:
        lines.append(f"- **Source:** a file uploaded by the analyst (`{name}`)")
    else:
        lines.append(f"- **Source:** `{name}`")
    if info.get("title"):
        lines.append(f"- **Title:** {info['title']}")
    if info.get("license"):
        lines.append(f"- **License:** {info['license']} - check the license terms before redistributing this data or derived work.")
    else:
        lines.append("- **License:** not recorded. Check the dataset's page before reusing it.")
    if info.get("description"):
        summary = " ".join(str(info["description"]).split())
        lines.append(f"\n> {summary[:600]}{'...' if len(summary) > 600 else ''}")
    cells.md("\n".join(lines))


def _loading(cells: _Cells, original: pd.DataFrame, dataset: dict[str, Any], analysis: dict[str, Any], name: str, embed_rows: int = MAX_EMBED_ROWS) -> None:
    sample = dataset.get("sample") or analysis.get("sample") or {}
    file_name = str(sample.get("file_name") or dataset.get("file_name") or f"{name}.{dataset.get('file_type') or 'csv'}")
    slug = _kaggle_slug(dataset)
    blob, embedded_rows, truncated = _embed_snapshot(original, embed_rows)
    total_rows = len(original)
    is_partial = bool(sample.get("is_sample"))

    note = [
        "## 2. Loading the dataset\n",
        f"The data comes from `{file_name}`. The cell below uses the first source it can find:\n",
        f"1. **A local copy** - put `{file_name}` in the same folder as this notebook to analyze the complete file.",
    ]
    if slug:
        note.append(f"2. **Kaggle** - set `USE_KAGGLE = True` to download `{slug}` with `kagglehub` (needs Kaggle credentials).")
        if embedded_rows:
            note.append(f"3. **The embedded snapshot** - {embedded_rows:,} rows saved inside this notebook, so it always runs.")
    elif embedded_rows:
        note.append(f"2. **The embedded snapshot** - {embedded_rows:,} rows saved inside this notebook, so it always runs.")
    if not embedded_rows:
        note.append("\n> **No data is embedded in this notebook** (chosen at export time to keep it small and private), so the local file or Kaggle download is required.")
    if truncated or is_partial:
        why = []
        if is_partial:
            why.append("the workspace analyzed a partial sample of a large file")
        if truncated:
            why.append(f"only the first {embedded_rows:,} of {total_rows:,} rows fit in the notebook")
        note.append(f"\n> **Heads up:** {' and '.join(why)}. Results from the snapshot describe that sample; "
                    "use the local file or Kaggle download for the complete data.")
    cells.md("\n".join(note))

    if embedded_rows:
        cells.code("# Embedded snapshot of the original rows (gzip-compressed CSV, base64-encoded). Safe to skip over.\n" + _wrap_blob(blob))

    kaggle_block = ""
    if slug:
        kaggle_block = (
            f"USE_KAGGLE = False          # set True to download the full dataset from Kaggle\n"
            f"KAGGLE_SLUG = {slug!r}\n"
            "if USE_KAGGLE:\n"
            "    import kagglehub\n"
            "    DATA_FILE = os.path.join(kagglehub.dataset_download(KAGGLE_SLUG), os.path.basename(DATA_FILE))\n"
            "\n"
        )
    cells.code(
        f"DATA_FILE = {file_name!r}\n"
        f"{kaggle_block}"
        "if os.path.exists(DATA_FILE):\n"
        f"    df_raw = {_read_call(file_name)}\n"
        "    print(f'Loaded {DATA_FILE}')\n"
        "else:\n"
        + ("    df_raw = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(''.join(_SNAPSHOT_GZ_B64)))), low_memory=False)\n"
           "    print('Local file not found - using the snapshot embedded in this notebook.')\n" if embedded_rows else
           "    raise FileNotFoundError(f'{DATA_FILE} not found. Place the data file next to this notebook (no snapshot is embedded).')\n")
        + ""
        "\n"
        "df = df_raw.copy()   # 'df' is the working copy; df_raw stays untouched for comparison\n"
        "print(f'{df.shape[0]:,} rows x {df.shape[1]} columns')\n"
        "df.head()"
    )
    cells.md("### Column overview\n\nData types and how many values are present in each column.")
    cells.code(
        "overview = pd.DataFrame({\n"
        "    'dtype': df.dtypes.astype(str),\n"
        "    'non_null': df.notna().sum(),\n"
        "    'missing_%': (df.isna().mean() * 100).round(2),\n"
        "    'unique': df.nunique(),\n"
        "})\n"
        "overview"
    )


def _quality_before(cells: _Cells, report: dict[str, Any]) -> None:
    lines = ["## 3. Data quality before cleaning\n",
             f"An automatic audit of the loaded data scored it **{report['label']} ({report['score']}/100)**: "
             f"{report['counts']['high']} high, {report['counts']['medium']} medium and {report['counts']['low']} low severity findings.\n"]
    issues = report.get("issues") or []
    if issues:
        lines += ["| Severity | Where | Finding | Suggested fix |", "|---|---|---|---|"]
        for issue in issues[:20]:
            cols = ", ".join(issue.get("columns") or [])[:60] or "whole table"
            fix = (issue.get("fix") or {}).get("type", "")
            fix = STEP_TITLES.get(fix, fix.replace("_", " ")) if fix else "review manually"
            lines.append(f"| {issue['severity']} | {_md_escape(cols)} | {_md_escape(issue['title'])} | {fix} |")
        if len(issues) > 20:
            lines.append(f"\n...and {len(issues) - 20} more lower-priority findings.")
    else:
        lines.append("No data-quality issues were detected.")
    cells.md("\n".join(lines))
    cells.code(
        "# Reproduce the headline quality checks on the raw data\n"
        "quality = pd.Series({\n"
        "    'rows': len(df_raw),\n"
        "    'duplicate rows': int(df_raw.duplicated().sum()),\n"
        "    'columns with missing values': int((df_raw.isna().sum() > 0).sum()),\n"
        "    'missing cells': int(df_raw.isna().sum().sum()),\n"
        "    'constant columns': int((df_raw.nunique(dropna=True) <= 1).sum()),\n"
        "})\n"
        "print(quality.to_string())\n"
        "df_raw.isna().sum().loc[lambda s: s > 0].sort_values(ascending=False).head(15)"
    )


def _cleaning(cells: _Cells, original: pd.DataFrame, transform_log: list[dict[str, Any]], final_df: pd.DataFrame) -> pd.DataFrame:
    """Replays the log against the original frame, emitting one documented step at a time."""
    cells.md("## 4. Cleaning steps")
    if not transform_log:
        cells.md("No cleaning steps were applied in the workspace, so the analysis below uses the data exactly as loaded. "
                 "The cell below is where you would add your own.")
        cells.code("# No cleaning steps were applied in the workspace.\n"
                   "print(f'Working data: {df.shape[0]:,} rows x {df.shape[1]} columns')")
        return original
    cells.md(f"{len(transform_log)} cleaning step(s) were applied in the workspace. Each one is repeated below in the same order "
             "with the pandas code that performs it, followed by the shape of the data afterwards.")
    frame = original
    for index, entry in enumerate(transform_log, start=1):
        op_type, params = str(entry.get("type")), entry.get("params") or {}
        try:
            code = op_code(frame, op_type, params)
            new_frame, description = apply_transform(frame, op_type, params)
        except (TransformError, KeyError, IndexError) as exc:
            cells.md(f"### Step {index}: {STEP_TITLES.get(op_type, op_type)}\n\n"
                     f"This step could not be replayed automatically ({exc}); it is skipped here.")
            continue
        cells.md(f"### Step {index}: {STEP_TITLES.get(op_type, op_type)}\n\n{description}")
        cells.code(f"# {description}\n{code}\n"
                   f"print(f'After step {index}: {{df.shape[0]:,}} rows x {{df.shape[1]}} columns')")
        frame = new_frame
    cells.md("### Result of cleaning\n\nHow the cleaned data compares with the original.")
    cells.code(
        "print(f'Before: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns, {int(df_raw.isna().sum().sum()):,} missing cells')\n"
        "print(f'After:  {df.shape[0]:,} rows x {df.shape[1]} columns, {int(df.isna().sum().sum()):,} missing cells')\n"
        "df.head()"
    )
    return frame


def _structure(cells: _Cells, analysis: dict[str, Any]) -> None:
    schema = analysis.get("schema") or {}
    roles = schema.get("roles") or {}
    lines = ["## 5. Structure of the cleaned data\n",
             "DataScopia infers what each column *means* (not just its dtype), because that decides which charts and tests are valid.\n"]
    from . import recommend as recommend_mod  # local import: recommend imports profiler helpers lazily
    for sentence in recommend_mod.describe_structure({"profiles": {c: {**v, "type": ""} for c, v in roles.items()}, **{k: schema.get(k) for k in ("geo", "time", "join_keys", "role_counts")}}):
        lines.append(f"- {sentence}")
    cells.md("\n".join(lines))
    table = ["| Column | Role | Notes |", "|---|---|---|"]
    for column, info in list(roles.items())[:60]:
        notes = ", ".join(f"{k}: {v}" for k, v in (info.get("notes") or {}).items() if isinstance(v, (str, int, float, bool)))
        table.append(f"| {_md_escape(column)} | {info.get('role_label') or info.get('role')} | {_md_escape(notes)[:80]} |")
    cells.md("\n".join(table))


def _summary_stats(cells: _Cells, analysis: dict[str, Any]) -> None:
    roles = (analysis.get("schema") or {}).get("roles") or {}
    measures = [c for c in analysis.get("numeric_columns", []) if _role(roles, c) in {"numeric", "ordinal_numeric"}]
    categoricals = [c for c in analysis.get("categorical_columns", []) if c in roles][:8]
    cells.md("## 6. Summary statistics\n\nCentral tendency and spread for the numeric columns, then the most common values of the "
             "categorical ones. A mean far from the median, or a maximum far above the 75th percentile, signals skew or outliers.")
    if measures:
        reads = ",\n    ".join(f"{c!r}: {_numeric_read(roles, c)}" for c in measures[:30])
        cells.code(
            "numeric = pd.DataFrame({\n"
            f"    {reads}\n"
            "})\n"
            "summary = numeric.describe(percentiles=[0.25, 0.5, 0.75]).T\n"
            "summary['skew'] = numeric.skew()\n"
            "summary['missing_%'] = numeric.isna().mean() * 100\n"
            "summary"
        )
    else:
        cells.code("df.describe(include='all').T")
    if categoricals:
        cells.code(
            f"for column in {categoricals!r}:\n"
            "    counts = df[column].value_counts(dropna=False).head(8)\n"
            "    print(f'\\n{column}  ({df[column].nunique()} distinct values)')\n"
            "    print((counts / len(df) * 100).round(1).astype(str).add('%').to_string())"
        )


def _chart_key(cfg: dict[str, Any]) -> tuple:
    return (cfg.get("type"), cfg.get("x"), cfg.get("y"), cfg.get("group"))


def _visualizations(cells: _Cells, analysis: dict[str, Any], on_screen: list[dict[str, Any]], chart_source: str = "both", max_charts: int = MAX_NOTEBOOK_CHARTS) -> None:
    roles = (analysis.get("schema") or {}).get("roles") or {}
    measures = list(analysis.get("numeric_columns", []))
    cells.md("## 7. Visualizations\n\nCharts are chosen from the column roles and only where they are statistically meaningful "
             "(for example a map only when a real geographic column exists, a time series only when there is a time column). "
             "Where a chart summarizes a relationship, the code also prints the supporting statistic.")
    seen: set[tuple] = set()
    plan: list[tuple[dict[str, Any], str]] = []
    for cfg in (on_screen if chart_source in {"both", "workspace"} else []):
        if isinstance(cfg, dict) and cfg.get("type") and _chart_key(cfg) not in seen:
            seen.add(_chart_key(cfg))
            plan.append((cfg, "Chart from your workspace"))
    for rec in (analysis.get("recommendations") or [] if chart_source in {"both", "recommended"} else []):
        if _chart_key(rec) not in seen:
            seen.add(_chart_key(rec))
            plan.append((rec, rec.get("reason") or ""))
    for number, (cfg, blurb) in enumerate(plan[:max_charts], start=1):
        title = cfg.get("customTitle") or cfg.get("title") or f"{cfg.get('type', 'chart').title()} {number}"
        cells.md(f"### 7.{number} {title}\n\n{blurb}".rstrip())
        cells.code(chart_code(cfg, roles, measures))
    if not plan:
        cells.md("No charts were recommended for this dataset.")


def _statistics(cells: _Cells, analysis: dict[str, Any], deep_ctx: dict[str, Any] | None = None) -> None:
    schema = analysis.get("schema") or {}
    roles = schema.get("roles") or {}
    measures = [c for c in analysis.get("numeric_columns", []) if _role(roles, c) in {"numeric", "ordinal_numeric"}]
    recs = analysis.get("recommendations") or []
    targets = analysis.get("target_details") or []
    cells.md("## 8. Statistical analysis\n\nEvery test below reports an effect size next to its p-value. With large samples almost any "
             "difference is 'significant', so the effect size (how big the difference is) matters more than the p-value alone.")

    if len(measures) >= 2:
        use = measures[:14]
        reads = ",\n    ".join(f"{c!r}: {_numeric_read(roles, c)}" for c in use)
        cells.md("### 8.1 Correlations\n\nPearson measures straight-line association, Spearman measures any consistently rising or falling "
                 "relationship (robust to outliers). The adjusted p-value multiplies by the number of pairs tested (Bonferroni), "
                 "because testing many pairs produces some false positives by chance alone.")
        cells.code(
            "num = pd.DataFrame({\n"
            f"    {reads}\n"
            "})\n"
            "rows = []\n"
            "cols = list(num.columns)\n"
            "for i, a in enumerate(cols):\n"
            "    for b in cols[i + 1:]:\n"
            "        pair = num[[a, b]].dropna()\n"
            "        if len(pair) < 10 or pair[a].nunique() < 2 or pair[b].nunique() < 2:\n"
            "            continue          # too little data (or no variation) for a meaningful correlation\n"
            "        r, p = stats.pearsonr(pair[a], pair[b])\n"
            "        rho, _ = stats.spearmanr(pair[a], pair[b])\n"
            "        rows.append({'a': a, 'b': b, 'n': len(pair), 'pearson_r': r, 'spearman_rho': rho, 'p': p})\n"
            "corr_table = pd.DataFrame(rows)\n"
            "if len(corr_table):\n"
            "    corr_table['p_adjusted'] = (corr_table['p'] * len(corr_table)).clip(upper=1)\n"
            "    corr_table = corr_table.reindex(corr_table['pearson_r'].abs().sort_values(ascending=False).index)\n"
            "corr_table.head(10)"
        )
        cells.md("### 8.2 Shape of each distribution\n\nSkewness far from 0 means a long tail (medians and rank-based tests are safer than "
                 "means there). The Shapiro-Wilk test checks normality on up to 5,000 values.")
        cells.code(
            f"rows = []\n"
            f"for column in {use[:8]!r}:\n"
            "    values = num[column].dropna()\n"
            "    if len(values) < 8 or values.nunique() < 3:\n"
            "        continue\n"
            "    sample = values.sample(min(len(values), 5000), random_state=0)\n"
            "    w, p = stats.shapiro(sample)\n"
            "    rows.append({'column': column, 'skew': values.skew(), 'excess_kurtosis': values.kurt(), 'shapiro_p': p,\n"
            "                 'looks_normal': p > 0.05 and abs(values.skew()) < 0.5})\n"
            "pd.DataFrame(rows)"
        )

    comparisons: list[tuple[str, str]] = []
    for rec in recs:
        if rec.get("type") == "box" and rec.get("x") and rec.get("y"):
            comparisons.append((rec["y"], rec["x"]))
        elif rec.get("type") == "distribution" and rec.get("x") and rec.get("group"):
            comparisons.append((rec["x"], rec["group"]))
    comparisons = list(dict.fromkeys(comparisons))[:4]
    if comparisons:
        cells.md("### 8.3 Do groups differ?\n\nKruskal-Wallis compares a numeric column across groups without assuming a normal shape. "
                 "**eta-squared** is the share of the variation the grouping explains (about 0.01 small, 0.06 medium, 0.14 large).")
        cells.code(
            "def compare_groups(measure, group):\n"
            "    data = pd.DataFrame({'value': pd.to_numeric(df[measure], errors='coerce'), 'group': df[group]}).dropna()\n"
            "    data = data[data['group'].isin(data['group'].value_counts().index[:12])]   # the 12 largest groups\n"
            "    samples = [g['value'].to_numpy() for _, g in data.groupby('group') if len(g) >= 3]\n"
            "    if len(samples) < 2:\n"
            "        return None\n"
            "    h, p = stats.kruskal(*samples)\n"
            "    k, n = len(samples), sum(len(s) for s in samples)\n"
            "    eta_sq = max(0.0, (h - k + 1) / (n - k))\n"
            "    return {'measure': measure, 'grouped_by': group, 'groups': k, 'n': n, 'H': h, 'p': p, 'eta_squared': eta_sq}\n"
            "\n"
            f"results = [compare_groups(m, g) for m, g in {comparisons!r}]\n"
            "pd.DataFrame([r for r in results if r])"
        )

    target = targets[0] if targets else None
    if target:
        t_col = target["column"]
        categorical_features = [c for c in analysis.get("categorical_columns", []) if c != t_col and _role(roles, c) in _CATEGORICAL_ROLES][:6]
        if target.get("kind") == "classification" and categorical_features:
            cells.md(f"### 8.4 Which categorical columns are associated with `{t_col}`?\n\nChi-square tests independence; **Cramer's V** "
                     "measures strength (0 = none, 1 = perfect; about 0.1 small, 0.3 medium, 0.5 large).")
            cells.code(
                "rows = []\n"
                f"for feature in {categorical_features!r}:\n"
                f"    table = pd.crosstab(df[feature], df[{t_col!r}])\n"
                "    if table.shape[0] < 2 or table.shape[1] < 2:\n"
                "        continue\n"
                "    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)\n"
                "    v = (chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))) ** 0.5\n"
                "    rows.append({'feature': feature, 'chi2': chi2, 'p': p, 'cramers_v': v})\n"
                "pd.DataFrame(rows).sort_values('cramers_v', ascending=False)"
            )
        elif target.get("kind") == "regression" and len(measures) >= 2:
            cells.md(f"### 8.4 Which numeric columns move with `{t_col}`?\n\nSpearman correlation of each numeric column with the target.")
            cells.code(
                "rows = []\n"
                f"target_values = pd.to_numeric(df[{t_col!r}], errors='coerce')\n"
                f"for feature in {[m for m in measures if m != t_col][:12]!r}:\n"
                "    pair = pd.DataFrame({'x': pd.to_numeric(df[feature], errors='coerce'), 'y': target_values}).dropna()\n"
                "    if len(pair) < 10 or pair['x'].nunique() < 2:\n"
                "        continue\n"
                "    rho, p = stats.spearmanr(pair['x'], pair['y'])\n"
                "    rows.append({'feature': feature, 'spearman_rho': rho, 'p': p, 'n': len(pair)})\n"
                "pd.DataFrame(rows).sort_values('spearman_rho', key=abs, ascending=False)"
            )

    time = schema.get("time")
    ts = next((r for r in recs if r.get("type") == "timeseries" and r.get("y")), None)
    if time and ts:
        t_col, y_col = time.get("time_column"), ts["y"]
        cells.md(f"### 8.5 Trend over time\n\nHow `{y_col}` moves with `{t_col}`. The slope is the average change per year; "
                 "Spearman's rho says whether the movement is consistently upward or downward, not just noisy.")
        parse = f"pd.to_datetime(df[{t_col!r}], errors='coerce')" if _role(roles, t_col) == "datetime" else f"pd.to_numeric(df[{t_col!r}], errors='coerce')"
        cells.code(
            f"when = {parse}\n"
            f"series = pd.DataFrame({{'when': when, 'value': pd.to_numeric(df[{y_col!r}], errors='coerce')}}).dropna()\n"
            "if pd.api.types.is_datetime64_any_dtype(series['when']):\n"
            "    series['t'] = series['when'].dt.year + (series['when'].dt.dayofyear - 1) / 365.25\n"
            "else:\n"
            "    series['t'] = series['when'].astype(float)\n"
            "trend = series.groupby('t')['value'].mean()\n"
            "if len(trend) >= 4:\n"
            "    fit = stats.linregress(trend.index, trend.values)\n"
            "    rho, p_rho = stats.spearmanr(trend.index, trend.values)\n"
            "    print(f'Average change per year: {fit.slope:,.4g} (R^2 = {fit.rvalue**2:.2f}, p = {fit.pvalue:.3g})')\n"
            "    print(f'Spearman rho = {rho:.2f} (p = {p_rho:.3g}) across {len(trend)} periods')\n"
            "else:\n"
            "    print('Fewer than 4 distinct periods - too few to fit a trend.')"
        )

    if deep_ctx:
        deep = deep_ctx["deep"]
        strongest = deep["corr"]["strongest"]
        cells.md("### 8.6 Deep statistical interpretation\n\n"
                 "This goes beyond the tables above: confidence intervals, robustness checks (outlier removal, curvature, "
                 "Simpson's paradox) and an **evidence rating** that combines significance, effect size, sample size and "
                 "consistency across those checks. **High** = significant, a large effect, at least 30 observations, and "
                 "unchanged by the robustness checks; **Moderate** and **Low** relax one or more of those; **Null** = no "
                 "detectable effect at the sample size available. (Computed by DataScopia's analysis engine, the same one "
                 "behind the PDF export - not re-derivable from a single cell, so it is presented as narrative here.)")
        if strongest:
            lines = ["**Relationships between variables**\n"]
            for i, d in enumerate(strongest[:3]):
                lines.append(f"- **{d['a']} and {d['b']}** (evidence: {d['rating']}). "
                             + " ".join(report_deep_mod.pair_interpretation(d, caveat=(i == 0))))
            cells.md("\n".join(lines))
            clusters = deep["corr"].get("clusters") or []
            if clusters:
                cells.md("**Overlapping variables (multicollinearity).** "
                         + "; ".join(", ".join(c) for c in clusters[:4])
                         + " move together closely enough (|r| >= 0.5) that they carry largely the same information - treating "
                           "them as independent evidence, or as separate model inputs, overstates what the data say.")
        trends_series = ((deep.get("trends") or {}) or {}).get("series") or []
        if trends_series:
            time_name = deep["trends"]["time"]
            lines = [f"**Change over time (by {time_name})**\n"]
            for t_ in trends_series[:2]:
                lines.append(f"- **{t_['measure']}.** " + " ".join(report_deep_mod.trend_interpretation(t_, time_name)))
            cells.md("\n".join(lines))
        posts = deep.get("posthoc") or []
        if posts:
            lines = ["**Differences between groups**\n"]
            for i, (ph, g_row) in enumerate(posts[:2]):
                lines.append(f"- **{ph['measure']} by {ph['group']}.** "
                             + " ".join(report_deep_mod.posthoc_interpretation(ph, g_row, caveat=(i == 0))))
            cells.md("\n".join(lines))


def _model_features(analysis: dict[str, Any], target: str, df: pd.DataFrame) -> tuple[list[str], list[str]]:
    roles = (analysis.get("schema") or {}).get("roles") or {}
    numeric, categorical = [], []
    for column, info in roles.items():
        if column == target or column not in df.columns or info.get("role") in _SKIP_FEATURE_ROLES:
            continue
        if info.get("role") in _NUMERIC_ROLES:
            numeric.append(column)
        elif info.get("role") in _CATEGORICAL_ROLES and int(info.get("unique_count", 0)) <= 30:
            categorical.append(column)
    budget = MAX_ML_FEATURES
    return numeric[:budget], categorical[: max(0, budget - len(numeric[:budget]))]


def _modelling(cells: _Cells, analysis: dict[str, Any], df: pd.DataFrame, ml: dict[str, Any] | None = None) -> None:
    ml = ml or {"test_size": 0.25, "seed": 42, "trees": 200}
    targets = analysis.get("target_details") or []
    cells.md("## 9. Predictive modelling")
    if not targets:
        cells.md("No column looked like an obvious prediction target (a label, outcome or price-like column), so no model is fitted. "
                 "To try one, set `target` in a copy of the cell below.")
        return
    target = targets[0]
    t_col, kind = target["column"], target["kind"]
    roles = (analysis.get("schema") or {}).get("roles") or {}
    numeric, categorical = _model_features(analysis, t_col, df)
    if not numeric and not categorical:
        cells.md(f"`{t_col}` looks like a target ({target.get('reason', '')}), but there are no usable feature columns left to predict it from.")
        return
    cells.md(
        f"`{t_col}` looks like the outcome of interest ({target.get('reason', '')}), so this section fits a **{kind}** baseline.\n\n"
        "Models are compared against a trivial baseline (always predict the most common class / the average). A model only "
        "*means* something if it beats that baseline on **held-out** rows it never saw while training.\n\n"
        "Identifier and free-text columns are excluded as features, numeric gaps are filled with the median and categories are one-hot encoded."
    )
    num_reads = "".join(f"X[{c!r}] = {_numeric_read(roles, c).replace('df[', 'data[')}\n" for c in numeric)
    classification = kind == "classification"
    cells.code(
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
        + ("from sklearn.dummy import DummyClassifier\n"
           "from sklearn.ensemble import RandomForestClassifier\n"
           "from sklearn.linear_model import LogisticRegression\n"
           "from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score\n"
           if classification else
           "from sklearn.dummy import DummyRegressor\n"
           "from sklearn.ensemble import RandomForestRegressor\n"
           "from sklearn.linear_model import Ridge\n"
           "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n")
        + "\n"
        f"target = {t_col!r}\n"
        f"numeric_features = {numeric!r}\n"
        f"categorical_features = {categorical!r}\n"
        "\n"
        "# Keep rows where the target is known; cap the size so training stays quick.\n"
        "data = df.dropna(subset=[target])\n"
        "if len(data) > 20000:\n"
        f"    data = data.sample(20000, random_state={ml['seed']})\n"
        "X = pd.DataFrame(index=data.index)\n"
        f"{num_reads}"
        "for column in categorical_features:\n"
        "    X[column] = data[column].astype(str)   # missing values become their own 'nan' category\n"
        + ("y = data[target].astype(str)\n" if classification else "y = pd.to_numeric(data[target], errors='coerce')\nkeep = y.notna()\nX, y = X[keep], y[keep]\n")
        + "\n"
        "preprocess = ColumnTransformer([\n"
        "    ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric_features),\n"
        "    ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), categorical_features),\n"
        "])\n"
        + ("stratify = y if y.value_counts().min() >= 2 else None\n" if classification else "stratify = None\n")
        + f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={ml['test_size']}, random_state={ml['seed']}, stratify=stratify)\n"
        "print(f'Training on {len(X_train):,} rows, testing on {len(X_test):,} held-out rows')"
    )
    if classification:
        cells.md("### Baseline vs. real models\n\nAccuracy can look good just because one class dominates, so balanced accuracy (average recall per class) "
                 "and macro F1 are reported too.")
        cells.code(
            "models = {\n"
            "    'Baseline (most common class)': DummyClassifier(strategy='most_frequent'),\n"
            "    'Logistic regression': LogisticRegression(max_iter=1000),\n"
            f"    'Random forest': RandomForestClassifier(n_estimators={ml['trees']}, min_samples_leaf=2, random_state={ml['seed']}, n_jobs=-1),\n"
            "}\n"
            "rows, fitted = [], {}\n"
            "for name, model in models.items():\n"
            "    pipe = Pipeline([('prep', preprocess), ('model', model)]).fit(X_train, y_train)\n"
            "    pred = pipe.predict(X_test)\n"
            "    fitted[name] = pipe\n"
            "    rows.append({'model': name, 'accuracy': accuracy_score(y_test, pred),\n"
            "                 'balanced_accuracy': balanced_accuracy_score(y_test, pred),\n"
            "                 'macro_f1': f1_score(y_test, pred, average='macro')})\n"
            "results = pd.DataFrame(rows).set_index('model')\n"
            "results"
        )
        cells.code(
            "best = results.drop(index='Baseline (most common class)')['balanced_accuracy'].idxmax()\n"
            "print(f'Best model: {best}')\n"
            "print(classification_report(y_test, fitted[best].predict(X_test), zero_division=0))"
        )
        scorer = "balanced_accuracy"
    else:
        cells.md("### Baseline vs. real models\n\nR-squared is the share of variation explained on held-out rows (0 = no better than predicting the average; "
                 "negative = worse). MAE is the typical miss in the target's own units.")
        cells.code(
            "models = {\n"
            "    'Baseline (predict the mean)': DummyRegressor(strategy='mean'),\n"
            "    'Ridge regression': Ridge(alpha=1.0),\n"
            f"    'Random forest': RandomForestRegressor(n_estimators={ml['trees']}, min_samples_leaf=2, random_state={ml['seed']}, n_jobs=-1),\n"
            "}\n"
            "rows, fitted = [], {}\n"
            "for name, model in models.items():\n"
            "    pipe = Pipeline([('prep', preprocess), ('model', model)]).fit(X_train, y_train)\n"
            "    pred = pipe.predict(X_test)\n"
            "    fitted[name] = pipe\n"
            "    rows.append({'model': name, 'R2': r2_score(y_test, pred), 'MAE': mean_absolute_error(y_test, pred),\n"
            "                 'RMSE': mean_squared_error(y_test, pred) ** 0.5})\n"
            "results = pd.DataFrame(rows).set_index('model')\n"
            "results"
        )
        cells.code(
            "best = results.drop(index='Baseline (predict the mean)')['R2'].idxmax()\n"
            "print(f'Best model: {best} (R2 = {results.loc[best, \"R2\"]:.2f} on held-out rows)')"
        )
        scorer = "r2"
    cells.md("### Which columns matter most?\n\nPermutation importance: shuffle one column at a time and measure how much the score drops on held-out "
             "rows. A bigger drop means the model leans on that column more. This shows what the model *uses*, not causation.")
    cells.code(
        "from sklearn.inspection import permutation_importance\n"
        "\n"
        "eval_X = X_test.head(2000)\n"
        "eval_y = y_test.loc[eval_X.index]\n"
        f"importance = permutation_importance(fitted[best], eval_X, eval_y, n_repeats=5, random_state={ml['seed']}, scoring={scorer!r})\n"
        "ranking = pd.Series(importance.importances_mean, index=X_test.columns).sort_values(ascending=False)\n"
        "fig = px.bar(ranking.head(12)[::-1], orientation='h', labels={'value': 'Score drop when shuffled', 'index': 'column'},\n"
        f"             title='What the model relies on to predict {t_col}')\n"
        "fig.update_layout(showlegend=False)\n"
        "fig.show()\n"
        "ranking.head(12)"
    )


def _findings(cells: _Cells, findings: list[dict[str, Any]], analysis: dict[str, Any]) -> None:
    roles = (analysis.get("schema") or {}).get("roles") or {}
    measures = list(analysis.get("numeric_columns", []))
    cells.md("## 10. Findings saved in the workspace" if findings else "## 10. Findings")
    if not findings:
        cells.md("No findings were saved in the workspace session. (Use the *Save* buttons on charts and answers to collect them here.)")
    for index, finding in enumerate(findings, start=1):
        ftype = finding.get("type")
        title_text = finding.get("title") or f"Finding {index}"
        if ftype == "question":
            cells.md(f"### Finding {index}: {title_text}\n\n**Question:** {finding.get('question', '')}\n\n**Answer:** {finding.get('answer', '')}")
            code = str(finding.get("code") or "")
            if code.strip().startswith(("df", "(df")):
                cells.code(f"# Reproduces the computed answer above\nresult = {code}\nresult")
            else:
                cells.code(f"# Computed via method '{finding.get('method', 'unknown')}'. The answer above is the result; this method used\n"
                           "# intermediate steps that don't reduce to a single expression.")
        elif ftype == "chart" and finding.get("chart"):
            cells.md(f"### Finding {index}: {title_text}")
            cells.code(chart_code(finding["chart"], roles, measures))
        else:
            cells.md(f"### Finding {index}: {title_text}\n\n{finding.get('summary', '')}")


def _conclusions(cells: _Cells, analysis: dict[str, Any], report: dict[str, Any], removed_rows: int, deep_ctx: dict[str, Any] | None = None) -> None:
    lines = ["## 11. Conclusions and next steps\n", "**What the analysis showed**\n"]
    if deep_ctx:
        deep = deep_ctx["deep"]
        strongest = deep["corr"]["strongest"]
        if strongest:
            d = strongest[0]
            direction = "positive" if d["r"] > 0 else "negative"
            lines.append(f"- The strongest relationship found is between **{d['a']}** and **{d['b']}** (r = {d['r']:.2f}, "
                         f"evidence: {d['rating']}), a {direction} association - see 8.6 for the full robustness check.")
        trends_series = ((deep.get("trends") or {}) or {}).get("series") or []
        if trends_series:
            t_ = trends_series[0]
            reliable = t_["p_tau"] < report_stats_mod.ALPHA
            change = f" ({t_['change_pct']:+.0f}% first period to last)" if t_.get("change_pct") is not None else ""
            lines.append(f"- **{t_['measure']}** " + ("shows a reliable trend" if reliable else "shows no reliable trend")
                         + f" over {deep['trends']['time']}{change}.")
        posts = deep.get("posthoc") or []
        if posts:
            ph, g_row = posts[0]
            lines.append(f"- **{ph['group']}** is " + ("a meaningful" if g_row.get("significant") else "not a strong")
                         + f" way to segment **{ph['measure']}** ({ph['highest']} highest, {ph['lowest']} lowest).")
    for insight in (analysis.get("insights") or [])[:8]:
        lines.append(f"- {insight}")
    quality = analysis.get("data_quality") or report
    lines.append(f"\n**Data quality:** {quality.get('label', '')} ({quality.get('score', '')}/100)"
                 + (f"; cleaning removed {removed_rows:,} row(s)." if removed_rows else "."))
    questions = analysis.get("research_questions") or []
    if questions:
        lines.append("\n**Questions worth pursuing next**\n")
        lines += [f"- {q}" for q in questions[:5]]
    lines.append("\n**Caveats.** Correlation is not causation; group comparisons describe this dataset, not necessarily the wider world; "
                 "and if the data was a partial sample, results should be re-checked on the full file.")
    cells.md("\n".join(lines))


def _renumber(cells: list[dict[str, Any]]) -> None:
    """Sections can be switched off, so renumber the '## N.' headings (and '### N.x' sub-headings) to stay gapless."""
    mapping: dict[str, str] = {}
    counter = 0
    for cell in cells:
        if cell["cell_type"] != "markdown" or not cell["source"]:
            continue
        m = re.match(r"^## (\d+)\. ", cell["source"][0])
        if m:
            counter += 1
            mapping[m.group(1)] = str(counter)
    for cell in cells:
        if cell["cell_type"] != "markdown" or not cell["source"]:
            continue
        first = cell["source"][0]
        m = re.match(r"^## (\d+)\. ", first) or re.match(r"^### (\d+)\.\d+ ", first)
        if m and m.group(1) in mapping:
            cell["source"][0] = first.replace(m.group(1) + ".", mapping[m.group(1)] + ".", 1)


# ----------------------------------------------------------------------------- public API

def build_notebook(
    title: str,
    findings: list[dict[str, Any]],
    analysis_summary: dict[str, Any],
    *,
    df: pd.DataFrame | None = None,
    original_df: pd.DataFrame | None = None,
    transform_log: list[dict[str, Any]] | None = None,
    dataset: dict[str, Any] | None = None,
    charts: list[dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the notebook (a plain nbformat-4.5 dict).

    ``df`` is the current (cleaned) dataframe and ``original_df`` the frame before any transform;
    with ``transform_log`` they let the notebook replay cleaning exactly. When neither frame is
    available (an expired session) the notebook falls back to the ~300-row preview, clearly
    labelled, so the export never hard-fails.
    """
    dataset = dataset or {}
    transform_log = [t for t in (transform_log or []) if isinstance(t, dict) and t.get("type")]
    name = str(analysis_summary.get("dataset_name") or dataset.get("name") or "dataset")

    preview_only = df is None and original_df is None
    if preview_only:
        preview = pd.DataFrame(analysis_summary.get("preview") or [])
        original, final = preview, preview
        transform_log = []
    else:
        final = df if df is not None else original_df
        original = original_df if original_df is not None else final
        if original_df is None:
            transform_log = []        # can't replay without the pre-transform frame
    analysis = analyze_dataframe(final, dataset_name=name, sample_info=analysis_summary.get("sample")) if not final.empty else analysis_summary
    # analysis_summary may carry a sample flag from the original upload; keep it authoritative
    sample = analysis_summary.get("sample") or dataset.get("sample")
    if sample:
        dataset = {**dataset, "sample": sample}
    report_before = quality_mod.build_quality_report(original, schema_mod.build_schema(original), analysis.get("target_details") or []) if not original.empty else analysis.get("data_quality") or {"score": 0, "label": "", "counts": {"high": 0, "medium": 0, "low": 0}, "issues": []}

    # Same deep-analysis engine as the PDF export (Fisher CIs, robustness checks, evidence ratings), run once and
    # reused for the "deep interpretation" subsection, the conclusions and the reproducibility fingerprint - so a
    # notebook and a PDF of the same dataset carry the same statistical depth, not a lighter, code-only version.
    deep_ctx: dict[str, Any] | None = None
    if not final.empty and len(final) >= 8:
        try:
            schema_full = schema_mod.build_schema(final)
            S_full = report_stats_mod.compute_all(final, analysis, schema_full)
            deep_full = report_deep_mod.compute_deep(final, analysis, schema_full, S_full)
            deep_ctx = {"S": S_full, "deep": deep_full, "fingerprint": report_stats_mod.data_fingerprint(final)}
        except Exception:
            deep_ctx = None

    cells = _Cells()
    slug = _kaggle_slug(dataset)
    opt = normalize_options(options)
    on = opt["sections"]
    _intro(cells, title, name, analysis, slug, opt, fingerprint=(deep_ctx or {}).get("fingerprint"))
    _source_and_license(cells, name, dataset)
    if preview_only:
        cells.md("> **Note:** the server no longer holds this dataset's data (the session expired), so this notebook embeds only the "
                 "~300-row preview. Re-open the dataset in the workspace and export again for the full analysis.")
    _loading(cells, original, dataset, analysis, name, opt["embed_rows"])
    if on["quality"]:
        _quality_before(cells, report_before)
    if on["cleaning"]:
        cleaned = _cleaning(cells, original, transform_log, final)
    else:
        # cleaning was excluded from the notebook, so the analysis below runs on the data as loaded
        cleaned = original
        transform_log = []
    if on["structure"]:
        _structure(cells, analysis)
    if on["summary"]:
        _summary_stats(cells, analysis)
    if on["visualizations"]:
        _visualizations(cells, analysis, list(charts or []), opt["chart_source"], opt["max_charts"])
    if on["statistics"]:
        _statistics(cells, analysis, deep_ctx)
    if on["modelling"]:
        _modelling(cells, analysis, final, opt["ml"])
    if on["findings"]:
        _findings(cells, findings, analysis)
    if on["conclusions"]:
        _conclusions(cells, analysis, report_before, max(0, len(original) - len(cleaned)), deep_ctx)
    _renumber(cells.cells)

    return {
        "cells": cells.cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "datascopia": {"dataset": name, "transform_steps": len(transform_log)},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
