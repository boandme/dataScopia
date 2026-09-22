# DataScopia

*A Kaggle Dataset Analysis & AI Research Workspace.*

## Run the API

From the project root, start the Python server:

```powershell
cd "C:\Users\vyom\OneDrive\Desktop\code\kaggle_datasetAnalysis"
python -m uvicorn apps.api.app.main:app --reload --host 127.0.0.1 --port 8000
```

Then open `apps/web/index.html` with VS Code Live Server. Keep the API terminal running while using the frontend.

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

## Roadmap: what's still needed to be a genuinely useful tool

The MVP checklist above is functionally complete for a *single* dataset, single-session workflow. The gaps below are what stand between that and something students, researchers, and analysts would actually adopt for real work — not just a demo.

1. **True multi-dataset workspace.** Load, switch between, and compare two or more datasets at once; surface likely join keys (e.g. Country + Year) between them. This was explicit MVP scope and is currently entirely unbuilt — the app only ever holds one active dataset in memory.
2. **Persistent projects.** Findings, saved charts, and the loaded dataset currently live only in browser memory and disappear on refresh. Needs save/resume — at minimum local-storage-backed, ideally a real saved-project file or account.
3. **A genuinely agentic AI layer.** The Q&A engine answers one question at a time against a fixed dataframe, with no memory between questions and no chaining ("filter to 2020, then check the correlation"). This is also the natural place to plug in a real LLM backend behind the existing `(question, dataframe) -> QAResult` contract, instead of today's deterministic intent-matching.
4. **In-app data cleaning and transformation.** Filtering rows, handling missing values, excluding outliers, renaming/recasting columns — needed before this is useful for real domain analysis rather than just profiling a dataset as-is.
5. **Shareable exports beyond a notebook.** A PDF/HTML report and per-chart image export, so a non-Python domain expert — the actual "researcher in biology/economics/policy" persona this is aimed at — can use the output without opening Jupyter.
6. **A first-class Data Quality Report, and clearer license/redistribution visibility.** Both are explicit in the original spec; today they're informal bullet insights rather than a dedicated exportable artifact, and a dataset's license isn't surfaced prominently before someone downloads or redistributes it.
7. **Onboarding.** No guided first run and no visible "what can I ask" reference beyond a one-line hint — most of what the Q&A engine can do (statistical tests, ML baselines, trend detection) is undiscoverable unless you already know to ask for it.
8. **Actual deployment.** The app currently only runs via a local `uvicorn` process plus opening a local HTML file by hand. Nothing is hosted — for students/researchers to reach it without cloning the repo and running Python, it needs a real deployment target: containerized, hosted somewhere, secrets handled outside `.env`. (The backend is FastAPI, not Flask — worth a quick sanity check on whether that's a hard requirement or shorthand for "a real deployed web app," since a framework swap would be a lateral rewrite with no functional gain over just productionizing what's already there.)

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
