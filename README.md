# DataScopia

*A Kaggle Dataset Analysis & AI Research Workspace.*

## Run DataScopia

From the project root:

```powershell
cd "C:\Users\vyom\OneDrive\Desktop\code\kaggle_datasetAnalysis"
python run.py
```

That's it — this starts the server and opens the app in your default browser automatically once it's ready (if port 8000 is busy, it falls back to the next free port and tells you which one). On Windows you can also just double-click `start_datascopia.bat`, which installs dependencies on first run if needed. The frontend is served directly by the same server at `/`, so there's no separate "open the HTML file" step and no second terminal to manage.

Close the terminal window (or press Ctrl+C) to stop the server.

<details>
<summary>Running the API directly (for development)</summary>

```powershell
python -m flask --app apps/api/app/main.py run --debug --host 127.0.0.1 --port 8000
```

`--debug` restarts the server on code changes, which `run.py` intentionally doesn't do (it prioritizes a single stable process for normal use). The frontend is served from the same URL either way.
</details>

## Deploying to Render (a real hosted URL)

`python run.py` is for local use — it binds a socket directly and runs Flask's own dev server, which explicitly refuses to be a production server. A hosted deployment needs a real WSGI server pointed at a fixed port the host controls, which is what `wsgi.py` + `Procfile` are for.

1. Push this repo to GitHub (already done if you're reading this from there).
2. In the [Render dashboard](https://dashboard.render.com), choose **New → Blueprint** and select this repo. Render reads `render.yaml` and configures the service automatically (build command, start command, health check).
3. When prompted, set the two required environment variables — your Kaggle API credentials from [kaggle.com/settings → API → Create New Token](https://www.kaggle.com/settings), which downloads a `kaggle.json` containing them:
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`
4. Deploy. Render builds with `pip install -r requirements.txt` and starts the app with `python -m waitress --host=0.0.0.0 --port=$PORT --threads=8 wsgi:app` — the same production WSGI server (`waitress`), verified locally, that Render's Linux container will run. You get a public `https://<your-app>.onrender.com` URL serving the full app, frontend and API together, exactly like `run.py` does locally.

No manual dashboard configuration is required — everything Render needs is already in `render.yaml`. If you'd rather configure it by hand (or use a different host like Railway or Fly.io that also runs a plain Python web service), the build command is `pip install -r requirements.txt` and the start command is the one in `Procfile`.

**One architectural thing worth knowing:** dataset analysis is cached in-memory (a plain Python dict, see `workspace_cache.py`), not in a database. That's fine for a single always-on process — which is what the free tier and the Procfile above both give you — but it means the app can't currently be scaled to multiple worker processes/instances without a shared cache (e.g. Redis) to replace it, since a request could otherwise land on a process that never analyzed that dataset. Not a concern for personal or small-audience use; worth revisiting before scaling up traffic.

Render's free tier spins the service down after ~15 minutes idle; the first request after a quiet period takes roughly 30-60 seconds while it wakes back up. That's a Render free-tier characteristic, not something this app can work around.

### Storage model

Dataset analysis does not require downloading the full dataset anywhere — not to the browser, and not to the API server. By default, `/datasets/analyze` streams only a bounded sample (`ANALYSIS_SAMPLE_MB`, default 5 MB) of the dataset's primary file directly from Kaggle: CSV/TSV files are read up to that byte limit and truncated at the last complete row, and Parquet files use HTTP range requests via `pyarrow`/`fsspec` to read only as many row groups as needed, using the file's footer metadata for an exact row count. Nothing is written to disk.

Two opt-in paths exist for when a full copy is actually needed:

- **Run full analysis** (`/datasets/analyze-full`) downloads the dataset to a temporary server workspace, profiles it in full, then deletes the workspace — useful when sample-based statistics aren't precise enough.
- **Download full dataset** (`/datasets/download-file`) returns a ZIP archive of the raw files for the user to keep.

Neither full-dataset path is required to get a Research Snapshot, recommended visualizations, or AI analysis.

A research-focused open-source platform for exploring, analyzing, and understanding Kaggle datasets with the help of intelligent automated analysis, interactive visualizations, and an AI-assisted research workflow.

## Purpose

This project is designed to help students, researchers, analysts, and data science practitioners move quickly from “I found a Kaggle dataset” to “I understand the dataset, can evaluate its quality, and can run meaningful analyses on it.”

Instead of presenting a generic dashboard full of charts, the app focuses on data understanding: identifying the dataset’s structure, highlighting what matters, recommending meaningful analyses, and enabling users to explore and document findings in a research-oriented workspace.

## Target Users

- Students working on data analysis or research assignments
- Analysts evaluating unfamiliar public datasets
- Kaggle competitors exploring benchmark data
- Researchers needing a structured exploration workflow
- Developers building data products and analysis pipelines
- Data science teams comparing multiple dataset sources

## Core Goals

- Search and add Kaggle datasets into a local workspace
- Automatically profile datasets and detect schema characteristics
- Generate a research snapshot before deeper analysis
- Recommend the most relevant visualizations for each dataset
- Allow users to edit, filter, and refine those visualizations
- Support AI-guided analysis grounded in actual data
- Save findings in a structured notebook-like workspace
- Export the resulting analysis as a reproducible Jupyter Notebook
- Manage multiple datasets in a single research environment

## Example Workflow

1. Search for a Kaggle dataset.
2. Preview metadata, files, and basic schema.
3. Load the dataset into the workspace.
4. Automatically identify column types, missingness, outliers, and likely targets.
5. Review the Research Snapshot and recommended analyses.
6. Explore visualizations and modify them interactively.
7. Ask the AI dataset analyst to compute relationships, run tests, or suggest hypotheses.
8. Save insights into the findings notebook.
9. Export the final analysis to a notebook for sharing or further development.

## Core Product Philosophy

This is not a generic “upload CSV and generate charts” application.

The project is intentionally designed around the idea of understanding dataset context and relevance:

- identify the dataset’s real structure
- detect what is useful to analyze
- recommend high-value evidence
- let the user investigate, refine, and save findings
- produce reproducible research artifacts

## Feature Highlights

### Dataset Discovery and Workspace Management
- Kaggle dataset search and metadata preview
- Multi-dataset workspace support
- Dataset switching and comparison
- Basic dataset compatibility assessment for joins or merges

### Intelligent EDA
- Column-type inference
- Missingness and duplicate detection
- Skewness and outlier detection
- Correlation and relationship analysis
- Time-series and geographic structure detection
- Target and feature candidate identification

### Research Snapshot
- Dataset size and schema overview
- Missingness and duplicate summaries
- Key distribution notes
- Potential target variables
- Notable correlations and relationships
- Recommended visualizations and analysis paths

### Visualization System
- Automatically generated visualizations based on data structure
- Editable axes, grouping, scaling, filters, and aggregation
- Chart-type switching and formatting controls
- AI-assisted visualization modification

### AI Dataset Analyst
- Question answering grounded in the actual dataset
- Data-driven calculations and summaries
- Visualization generation from natural language requests
- Statistical and modeling workflows for common analytical tasks
- Results documented and saved as findings

### Saved Findings and Notebook Workspace
- Persistent notebook-style analysis journal
- Stored statistics, charts, and AI explanations
- Reproducible code snippets and analysis context
- Multi-step research histories for each dataset

### Jupyter Export
- Export findings and analyses to a .ipynb notebook
- Reproduce visualizations, code, and summary sections
- Suitable for academic and portfolio use

## Planned Architecture

The application will be organized as a modular system with separation between data access, analysis logic, AI orchestration, and UI concerns.

### High-level layers

- Dataset ingestion and metadata layer
- Data profiling and schema analysis layer
- Visualization recommendation and editing layer
- AI analysis agent orchestration layer
- Research notebook / findings persistence layer
- Export services for notebook generation
- Frontend workspace UI

### Suggested initial structure

```text
kaggle_datasetAnalysis/
├── README.md
├── PROJECT_SPEC.md
├── docs/
│   └── architecture.md
├── apps/
│   ├── web/
│   └── api/
├── packages/
│   ├── analysis/
│   ├── dataset/
│   └── ui/
├── .env.example
├── .gitignore
├── LICENSE
└── requirements.txt or package.json
```

## Version History

- **v0.1** (2026-09-19) — Initial plan: product spec, architecture doc, and repo scaffolding.
- **v0.3** (2026-09-19 – 2026-09-20) — First working backend: Kaggle auth + search, dataset loading via `kagglehub`, FastAPI wired to a basic profiler and a Plotly frontend.
- **v0.35** (2026-09-20) — Workspace overhaul: sample-based streaming analysis (no full download required by default, via HTTP range/truncated reads), server-side dataframe caching, a deterministic natural-language Q&A engine grounded in real pandas/scipy computation, CSV/Parquet/JSON/Excel upload, deeper Kaggle search (had been silently capped at ~3–5 results by a parameter bug), reproducible `.ipynb` export, and a first UX pass on the chart editor.
- **v0.38** (2026-09-20) — Q&A engine expansion (dataset overview, duplicate detection, value counts, column-level ranking, ratios, trend detection, multi-condition filters) plus real column-matching bug fixes found through testing; Kaggle dataset/column description surfacing; UI polish (wider layout, capped Research Snapshot lists, sidebar sizing).
- **v0.4** (2026-09-21) — In-app dataset transformation (item 4 on the roadmap below): filter/clean/merge-columns applied server-side on the full analyzed sample, composable, reset-able, and reflected live in every chart and question. localStorage-backed saved analyses with a "Past Analyses" panel and named saves. Per-chart export with format/scale controls. Unified card-based workspace redesign so chart and dataset-editor cards share one grid and visual system instead of conflicting layouts. Added a 55-test unit suite plus a benchmark script that caught a real ~440x performance regression in the column-merge operation (row-wise Python apply instead of vectorized pandas) and fixed it.
- **v0.42** (2026-09-21) — Full **DataScopia** brand identity and page redesign. Server-side PDF research report generation (`analysis/report.py`): real scipy-backed statistical narrative per chart (Pearson/Spearman correlation, distribution shape, group-comparison significance tests) plus a synthesizing conclusion and a "Next Steps" section, replacing an earlier client-side version that just restated on-screen stats. New Column Explorer panel with a name-based column-description inference engine as a fallback wherever the dataset's own documentation doesn't describe a column. Enlarged/restyled Research Snapshot and Research Questions lists. Boolean-dtype crash class (`numpy boolean subtract`) fixed across the profiler and transform engine, with regression tests.
- **v0.45** (2026-09-22) — Rebuilt the column-description inference into a subject-aware algorithm (e.g. `school_name` → "the name of the school" rather than a generic "name" guess that discarded half the column name), reaching ~88% specific (non-generic) descriptions on a representative name battery. Root-caused and fixed a column/value-garbling bug affecting a meaningful fraction of real Kaggle CSVs: automatic text-encoding detection (cp1252/Latin-1/UTF-16 vs. a hardcoded UTF-8 assumption) and delimiter sniffing (semicolon/tab-delimited files no longer collapse into one garbled column). Collapsible dataset-search sidebar. Workspace (charts/editors) reprioritized to render before the Overview/Column Explorer detail panels so it's visible without scrolling. Fixed "Analyze current dataset," which had been hard-wired to always reload the demo dataset regardless of what was actually loaded.
- **v0.5** (2026-09-23) — Closed out the garbled-text bug class for good: detects and correctly decodes BOM-less UTF-16 files (a NUL-byte pattern that silently "succeeds" under a naive UTF-8 decode, producing the box-glyph-riddled text reported as a live bug) and ZIP/GZIP-wrapped files served under a plain `.csv` name, plus a final sanity check that raises a clear error instead of ever silently returning garbage column names. Migrated the backend from FastAPI/uvicorn to **Flask**, at the user's request after a uvicorn install failure; along the way, found and fixed two real Windows-specific bugs in the new launcher (`SO_REUSEADDR` silently letting two server instances share one port, and `sys.exit()` calls inside both uvicorn's and werkzeug's startup paths bypassing normal exception handling). The whole app now launches with a single `python run.py` (or double-clicking `start_datascopia.bat` on Windows) — it starts the server, serves the frontend from the same origin, opens the browser automatically, and retries on the next port if the default is busy. "Ask about this dataset" moved to its own compact panel at the very top of the page, above the Workspace, so it's reachable without scrolling.
- **v0.6** (2026-09-24) — Four themes. **(1) Reproducible notebook export**: the "Export notebook" button now builds a complete notebook — source and license, loading (local file, optional Kaggle download, or an embedded snapshot so it always runs), a data-quality audit, the exact in-app cleaning steps *replayed* as pandas code (`transforms.op_code`, tested against the real transforms), summary statistics, the recommended and on-screen charts as runnable code, statistical tests with effect sizes (correlations with multiple-comparison adjustment, normality, Kruskal-Wallis, Cramér's V, trend), a baseline-vs-model section when a target is detected, saved findings and conclusions. **(2) Licenses and consent**: license shown in search results and after load, plain-language license summary, (shown, never blocking — a consent dialog was tried and removed as too intrusive). **(3) Deeper schema detection and chart recommendations** (`schema.py`, `geo.py`, `recommend.py`, `charts.py`): columns get semantic roles (identifier, year, string dates, country/US state/lat-lon, ordinal, text, numbers stored as text, …), time-series/panel structure and candidate join keys are detected, and charts are drawn server-side from every row (the old client renderer only saw a 300-row preview) — maps, time series, treemaps, pair plots, missing-value charts, a true clustered correlation matrix, group distributions and target-vs-feature, each recommended only when statistically justified, with the reason and supporting statistic shown. **(4) Hygiene**: Flask test-client API tests, GitHub Actions CI, and `index.html` split into `styles.css` + `app.js`. Also new: a scored **Data Quality Report** with one-click fixes (which flow into the notebook) and a Markdown export.
- **v0.7** (2026-09-24) — DataScopia is now a multi-page site: a hash-routed single-page app with a top menu — **Datasets** (search, upload, demo, saved analyses), **Overview**, **Workspace**, **Ask** (suggested questions from the data plus a session history) and **Export** (notebook, PDF, data-quality report, saved findings) — with working back/forward and deep links, and a "load a dataset first" gate on pages that need data. The license consent dialog was removed; the license is still shown in search results and on the loaded dataset.
- **v0.8** (2026-09-26) — The Workspace tab is now **Visualization**; the menu moved into a full-width sticky top bar with larger type and an animated active-tab indicator (old `#/workspace` links still work). Charts are colorful by default: a vivid indigo→gold magnitude ramp colors bars, histograms, scatter points, maps, treemaps and missing-value charts by value, box plots give each group its own color, lines get a glow, and recommendation cards, stat cards and chart frames carry color accents. New dynamic **Changelog** page served from `/api/changelog` (data in `apps/api/app/changelog.json`) with live-computed stats, category filters and search.
- **v0.81** (2026-09-26) — First-run guidance: after analyzing a dataset, a welcome guide and a short spotlight tour point users to Visualization, Ask, Export and Save (shown once per browser; a "Quick tour" button replays it).
- **v0.9** (2026-09-26) — PDF report rebuilt (`analysis/report_pdf.py`, statistics in `analysis/report_stats.py`) following data-analysis report practice (executive summary, key findings, recommended actions, methods, results, discussion, references, appendices) and the scientific-manuscript conventions (hypotheses, results separated from interpretation, mean ± SD with effect sizes, captioned/numbered tables and figures). Analysis-first: descriptive statistics, adjusted correlations, group tests with effect sizes, a held-out baseline model, quality audit, reproducibility record.
- **v0.92** (2026-09-26) — Deep analysis reports and an Export studio. The PDF now runs the checks an analyst would (`analysis/report_deep.py`): correlation confidence intervals, outlier/rank/curvature/subgroup robustness, VIF and clusters, explicit null results with detectable-effect size, trend/growth/change-point analysis, Holm-adjusted post-hoc contrasts, distribution and missingness diagnostics, per-chart two-part analysis, and an evidence-rating table. Every export has a custom file name plus an options panel (PDF, notebook, quality report), validated server-side and remembered in the browser.
- **v1.0** (2026-09-27) — Platform release. **Accounts** (scrypt passwords, HttpOnly session cookies, rate-limited sign-in, same-origin write checks, account deletion), **cloud projects** (original data + cleaning steps + findings + charts, auto-saved and reopened with everything replayed), **verified, shareable analyses** (public link, reproducibility hash, server-side re-verification that recomputes every published number from the stored data, forks, PDF/notebook downloads, an Explore page), and the **dataset graph** (join-key sketches - country/state/year/entity MinHash - matched across a self-growing catalogue, a built-in library and your own projects, with join preview and one-click combined datasets). Storage is SQLite + Parquet (`DATASCOPIA_DB`, `DATASCOPIA_DATA_DIR`); `tools/index_kaggle.py` grows the catalogue from Kaggle. On Render, attach a persistent disk (see `render.yaml`) or the data is lost on each deploy.
- **v1.1** (2026-09-27) — Firebase. **Accounts** now use Firebase Authentication (email/password, Google, password reset, email verification; the server verifies ID tokens with Google's public keys and starts a session cookie), and **storage** can move to the Firebase Realtime Database + Cloud Storage via a service account (`apps/api/app/firebase.py`, `docstore.py`; REST APIs, no Admin SDK). Everything falls back to local SQLite/files, and `DATASCOPIA_AUTH=local` keeps the built-in accounts. Setup: [docs/FIREBASE_SETUP.md](docs/FIREBASE_SETUP.md).
- **v1.25** (2026-09-27) — Redesigned sign-in/account-settings UI (split visual panel, icon-led fields, password-visibility toggle, redesigned account menu). Notebook export now runs the same deep-statistics engine as the PDF (`report_deep.py`/`report_stats.py`): a "Deep statistical interpretation" subsection and evidence-led conclusions, plus a reproducibility fingerprint shared with the PDF so both exports of the same analysis can be confirmed to match. The PDF report gained a clickable Contents page and a full section outline in the PDF viewer's sidebar (fixed a `NumberedCanvas` bug where deferred page commits bound every bookmark to page 1). Every chart (`analysis/charts.py`) now draws a bordered, ticked frame on both axes for a proper scientific-figure look, and a real bug is fixed where a scatter chart's extrapolated trend line could stretch the y-axis far past the actual data, squeezing every real point into a sliver of the chart — the axis range is now fixed to the data itself. Export titles default to whatever the analysis was already named (the saved/cloud project name) instead of a generic guess, and the file name reliably keeps following the title as it's edited.

## Roadmap: what's still needed to be a genuinely useful tool

The MVP checklist above is functionally complete for a *single* dataset, single-session workflow. The gaps below are what stand between that and something students, researchers, and analysts would actually adopt for real work — not just a demo.

1. **True multi-dataset workspace.** Load, switch between, and compare two or more datasets at once; surface likely join keys (e.g. Country + Year) between them. This was explicit MVP scope and is currently entirely unbuilt — the app only ever holds one active dataset in memory.
2. ~~**Persistent projects.**~~ Done (v0.4) — localStorage-backed saved analyses with a "Past Analyses" panel, named saves, and full resume (dataset, charts, findings, transform log).
3. **A genuinely agentic AI layer.** The Q&A engine answers one question at a time against a fixed dataframe, with no memory between questions and no chaining ("filter to 2020, then check the correlation"). This is also the natural place to plug in a real LLM backend behind the existing `(question, dataframe) -> QAResult` contract, instead of today's deterministic intent-matching.
4. ~~**In-app data cleaning and transformation.**~~ Done (v0.4) — filter/clean/merge-columns, composable and reset-able, applied server-side on the full analyzed sample.
5. ~~**Shareable exports beyond a notebook.**~~ Done (v0.4/v0.42) — per-chart image export and a server-side PDF research report with real statistical narrative, not just a stats dump.
6. ~~**A first-class, standalone Data Quality Report.**~~ Done (v0.6) — scored report, one-click fixes, Markdown export. (Original note:) Data-quality narrative is now woven into the PDF export's "Data Quality Assessment" section, and license text is surfaced in the dataset description panel — but there's still no dedicated, exportable data-quality artifact independent of the full report.
7. **Onboarding.** No guided first run and no visible "what can I ask" reference beyond a one-line hint — most of what the Q&A engine can do (statistical tests, ML baselines, trend detection) is undiscoverable unless you already know to ask for it.
8. ~~**Actual hosted deployment.**~~ Deployment-ready (v0.5): `wsgi.py`, `Procfile`, and `render.yaml` are set up for a one-Blueprint deploy to Render (or any host that runs a plain Python web service), with credentials handled via environment variables instead of a committed `.env` — see [Deploying to Render](#deploying-to-render-a-real-hosted-url). Actually clicking through Render's dashboard to stand up a live URL is still a manual, one-time step outside this repo. Caveat worth tracking as usage grows: the in-memory `workspace_cache` (no database) means this is currently a single-process deployment, not one that scales to multiple workers/instances without further work.

## Open-Source Goals

This project is intended to be community-friendly and contributor-driven. The design should emphasize:

- clear module boundaries
- readable architecture
- environment-variable configuration
- reproducibility
- minimal vendor lock-in where practical
- support for multiple AI providers and dataset sources
- maintainable documentation

## Licensing and Responsibility

The platform must respect dataset licensing, Kaggle API rules, and platform usage restrictions. The software should be built to support compliant usage and give users clear information about whether data can be downloaded, redistributed, or exported.

## Status

Past the planning phase — see [Version History](#version-history) above. The single-dataset MVP loop (search → sample-based analyze → Q&A → charts → save findings → export notebook) works end-to-end against live Kaggle data. See [Roadmap](#roadmap-whats-still-needed-to-be-a-genuinely-useful-tool) for what's next, in priority order.
