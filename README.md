# Kaggle Dataset Analysis & AI Research Workspace

## Run the API

From the project root, start the Python server:

```powershell
cd "C:\Users\vyom\OneDrive\Desktop\code\kaggle_datasetAnalysis"
python -m uvicorn apps.api.app.main:app --reload --host 127.0.0.1 --port 8000
```

Then open `apps/web/index.html` with VS Code Live Server. Keep the API terminal running while using the frontend.

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

## Development Roadmap

### Phase 1: Planning and specification
- Define the product direction and MVP scope
- Confirm architecture and core workflows
- Document requirements and technical constraints

### Phase 2: MVP foundation
- Dataset search and loading backend
- Schema profiling and research snapshot generation
- Basic visualization recommendation system
- Multi-dataset workspace support

### Phase 3: AI analysis layer
- Dataset-aware analysis agent
- Natural-language question handling
- Computation and notebook-logging workflows

### Phase 4: Research notebook and export
- Save findings and code
- Generate Jupyter Notebooks
- Improve reproducibility and documentation

### Phase 5: Advanced features
- More sophisticated statistical testing
- ML workflows and model comparison
- Data join recommendations and cross-dataset analysis
- Expanded export and dataset management tools

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

This repository is currently in the planning phase. The next steps are to finalize the product specification, confirm the MVP architecture, and then begin implementation incrementally.
