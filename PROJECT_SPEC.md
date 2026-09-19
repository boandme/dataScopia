# Project Specification: Kaggle Dataset Analysis & AI Research Workspace

## 1. Product Overview

This project is an open-source research and analysis application for exploring Kaggle datasets, profiling them intelligently, and helping users uncover meaningful insights with a dataset-aware AI assistant. The application is designed to help users move from raw dataset discovery to reliable analysis and reproducible research outputs.

The core experience should feel like a combination of:

- Kaggle dataset discovery
- Notebook-style analysis
- Data profiling and research snapshots
- Visualization-based exploration
- AI dataset analysis and interpretation
- Structured findings management

## 2. Problem Statement

Many users find a public dataset and immediately have to assemble their own EDA pipeline: checking column types, profiling missingness, creating charts, exploring relationships, writing analysis notes, and deciding what to investigate next. This creates friction and slows down analytical work.

This project reduces that friction by providing a dataset-aware workspace that profiles the data automatically, recommends useful analyses, supports editing and iteration, and gives the user an AI collaborator that works from the real dataset rather than a generic generic chatbot.

## 3. Target Users

- Students in research or data analysis classes
- Analysts evaluating unfamiliar data sources
- Kaggle participants exploring competition data
- Researchers who need a structured research workspace
- Data practitioners who want to understand and document a dataset before building models

## 4. Core Goals

1. Enable users to search and open Kaggle datasets.
2. Load one or more datasets into a shared workspace.
3. Automatically detect dataset structure and quality issues.
4. Generate a Research Snapshot summarizing the most relevant details.
5. Recommend the best visualizations for the specific dataset.
6. Support interactive modification of visualizations.
7. Allow AI-driven analysis based on actual data, not generic advice.
8. Save useful findings in a persistent research notebook.
9. Export an analysis to Jupyter Notebook for reproducibility.
10. Respect licensing, dataset restrictions, and API usage requirements.

## 5. Functional Requirements

### 5.1 Dataset Search and Import

The system shall allow users to:

- search Kaggle datasets from the application
- preview basic dataset metadata
- review dataset description, size, file list, and update history
- add a dataset to the current workspace
- load dataset files into an analysis environment
- support multiple datasets in a single workspace

#### Required metadata

- dataset name
- author
- description
- number of files
- file types
- dataset size
- update date
- usage/popularity indicators when available

#### Constraints

- Kaggle access and usage must follow Kaggle API requirements and authentications
- The app must not assume redistribution rights for every dataset
- Data licensing and download restrictions must be made visible to the user

### 5.2 Dataset Workspace

The system shall support:

- loading multiple datasets simultaneously
- switching between datasets in the UI
- displaying active/open datasets clearly
- comparing datasets side-by-side where relevant
- identifying likely compatible join keys between datasets
- allowing questions spanning multiple datasets

#### Example multi-dataset workflow

- dataset A: tourism data
- dataset B: GDP data
- dataset C: climate or region metadata

The system should help the user understand how these datasets may relate and support cross-dataset analysis when feasible.

### 5.3 Automatic Schema Detection

The system should inspect each dataset and classify columns by type and data characteristics, including:

- numerical
- categorical
- binary
- datetime / timestamp
- geographic / location
- text fields
- IDs or unique keys
- ordinal variables
- high-cardinality columns
- missing data patterns
- outliers
- skewness
- target-like columns
- time-series structure
- candidate join keys

The system should build a schema summary that is informative and dataset-specific rather than generic.

### 5.4 Research Snapshot

Every dataset must have a concise, informative Research Snapshot designed to orient the user immediately.

#### Snapshot should include

- observation count
- variable count
- column type overview
- missingness summary
- duplicate summary
- likely target variables
- relevant distributions and skewness
- notable correlations
- notable data-quality issues
- recommended analyses and visualizations

#### Example structure

- 15,243 observations
- 20 variables
- 3.2% missing data
- 4 highly skewed variables
- 2 potential target variables
- 6 notable correlations
- 3 data-quality concerns
- 5 recommended visualizations

The snapshot should be factual and derived from actual data values. It should help a user quickly understand what matters before deeper exploration.

### 5.5 Intelligent EDA and Visualization Selection

The system shall not generate a fixed template chart pack for every dataset. Instead, it must use data structure and field relationships to determine what is relevant.

#### Visualization logic should consider

- column types
- cardinality
- dataset size
- relationships between variables
- time structure
- geographic structure
- target variables
- missingness patterns
- outliers
- skewness
- possible correlations

#### Example chart families supported

- histograms
- box plots
- violin plots
- bar charts
- stacked bars
- scatter plots
- line charts
- area charts
- heatmaps
- correlation matrix
- time-series plots
- geographic maps
- missing-value visualizations
- pair plots
- treemaps
- distribution comparisons
- target-vs-feature analyses

Only generate visualizations when they are meaningful and statistically appropriate.

### 5.6 Visualization Editor

Each generated visualization shall be editable in a structured way.

#### Editing capabilities

- X and Y field selection
- grouping variable
- aggregation method
- filter conditions
- sort order
- chart type change
- color grouping
- date granularity
- statistical measure
- axis labels and titles

#### Natural language modification

Users should be able to ask the AI to modify a chart using natural language, such as:

> “Change this to show median income by state only for states with more than 100 observations.”

The system should re-render or regenerate the visualization to match the request.

### 5.7 AI Dataset Analyst

The application must include a dataset-aware AI analysis agent that can operate on the loaded data and current workspace context.

The AI must not merely reply with generic advice. It must have access to:

- actual dataset values
- schema summary
- summary statistics
- generated visualizations
- current filters
- saved findings
- prior analysis steps
- other workspace datasets
- a computation environment

#### Example supported questions

- What percentage of countries contribute more than 10% of GDP from tourism?
- Which countries saw the largest decline between 2015 and 2020?
- Find the strongest relationships among numerical variables.
- Why is this distribution skewed?
- Compare these two datasets.
- Create a visualization for this question.
- Run a statistical test.
- Build a baseline model predicting this target.

The AI should perform actual computation, not just explain how it could be done.

### 5.8 AI as an Analysis Agent

The AI should function as an analysis agent with a structured workflow:

1. understand the objective
2. inspect relevant dataset(s)
3. determine appropriate fields and methods
4. clean or transform data when needed
5. run computation or analysis
6. generate relevant outputs
7. explain results clearly
8. record useful findings
9. generate notebook code for reproducibility

#### Examples

- “Determine whether tourism revenue is correlated with GDP growth and visualize the relationship.”
- “Build a model predicting tourist arrivals.”

In both cases, the AI should execute and document the analysis, rather than only giving instructions.

### 5.9 Research Notebook / Saved Findings

The app shall include a persistent analysis notebook where users can save important findings.

#### Supported saved artifacts

- charts
- statistics
- AI-generated explanations
- tables
- data transformations
- ML results
- research questions
- notes
- dataset metadata and context

Each saved finding should include enough context to understand how it was produced.

#### Example saved finding structure

- Question
- Result
- Analysis summary
- Dataset or workspace context
- Filters used
- Reproducible code snippet

Users should be able to organize findings into a coherent research narrative.

### 5.10 Jupyter Notebook Export

The system should export the user’s analysis as a Jupyter Notebook .ipynb file.

#### Exported notebook should include

- dataset loading
- data cleaning steps
- summary statistics
- visualizations
- statistical analysis
- ML models where relevant
- results
- explanatory markdown where appropriate
- comments explaining important steps

This notebook must satisfy reproducibility and research usability goals.

### 5.11 Multi-Dataset Research Features

The system should allow the user to work with multiple datasets at once.

The interface should clearly show active datasets and support:

- dataset switching
- dataset comparison
- cross-dataset analysis
- joined or related analysis when appropriate
- AI questions that refer to multiple datasets

The system should make basic join-key suggestions such as:

- country + year
- region + date
- entity + timestamp

These suggestions should be advisory and not over-automated.

### 5.12 Data Quality Report

The system should built a data-quality report that highlights issues relevant to research and reuse.

#### It should detect

- missingness
- duplicates
- inconsistent formats
- suspicious values
- outliers
- high-cardinality columns
- constant or near-constant columns
- leakage-risk indicators
- likely schema problems

This should be presented as a readable, fact-based report rather than a dump of raw metrics.

### 5.13 Dataset Insights

The system should generate dataset-specific factual observations, such as:

- “82% of observations belong to three categories.”
- “Income is strongly right-skewed.”
- “Three columns show substantial missingness.”

These should be grounded in the actual data and shown in a concise research-friendly format.

### 5.14 Research Questions

The product should recommend meaningful research questions based on patterns in the dataset, such as:

- Does tourism dependence correlate with GDP growth?
- Are certain regions disproportionately represented?
- How has this metric changed over time?

The user should remain in control of selecting which questions to pursue.

## 6. Non-Functional Requirements

### 6.1 Open Source and Maintainability

The project should be designed as an open-source application with:

- modular architecture
- clear responsibility boundaries
- readable code structure
- good documentation
- reproducible local setup
- environment-variable-based configuration
- abstraction for providers and integrations

### 6.2 Extensibility

The project should be buildable for future extension with:

- additional visualization heuristics
- new statistical analyses
- new AI tools
- additional dataset providers
- new export formats
- additional chart types

### 6.3 Security and Compliance

The system should handle API credentials and dataset access responsibly.

It must:

- use environment variables for secrets
- avoid hardcoded credentials
- respect Kaggle API policies and restrictions
- display licensing and redistribution constraints
- ensure user consent before exporting or downloading restricted files

## 7. MVP Scope

The first version should not attempt to solve every advanced feature at once. The MVP should include:

1. Kaggle dataset search
2. dataset loading
3. automatic schema detection
4. research snapshot
5. dataset overview and statistics
6. intelligent visualization recommendations
7. interactive visualization editing
8. basic AI dataset analysis agent
9. saved findings / analysis notebook
10. Jupyter Notebook export
11. dataset export/download
12. multi-dataset workspace foundation

### Advanced features deferred

- full statistical workflow orchestration
- sophisticated model-building agents
- complex join optimization
- deep research agent loops
- advanced feature engineering workflows

These can be added iteratively after the MVP is stable.

## 8. Technical Considerations

### Data Layer

The data layer should support:

- CSV, parquet, Excel, and common tabular formats
- schema inference
- missingness and quality checks
- multiple dataset sources
- in-memory or local store for analysis artifacts

### Analysis Layer

The analysis layer should provide:

- profiling functions
- relationship detection
- aggregation helpers
- statistical summaries
- visualization recommendation logic
- notebook generation

### AI Layer

The AI layer should be provider-agnostic where practical, with integration points for:

- LLM-based analysis assistant
- local or hosted model support
- dataset/tool execution runtime
- notebook code generation

### UI Layer

The UI should prioritize a lightweight but informative workflow:

- dataset list and workspace context
- main analysis panel
- insights and snapshot panel
- visualization editor
- findings notebook panel
- export tools

## 9. Product Workflow

The intended workflow is:

Find Dataset → Understand Dataset → Explore → Investigate → Save Findings → Export Notebook → Reproduce

This must remain the visual and conceptual core of the product.

## 10. Success Criteria

The product is successful if it helps users:

- understand unfamiliar datasets quickly
- identify meaningful analysis directions
- produce relevant, evidence-based summaries
- interactively explore and refine visualizations
- ask data-grounded questions to an AI assistant
- save structured research progress
- export a reproducible notebook from an interactive workflow

## 11. Initial Scope Recommendation

The project should begin with a strong foundation and advanced features layered in carefully.

The first implementation should prioritize:

- reliable dataset ingestion
- profiling and insight generation
- useful visualization recommendations
- AI-backed analysis of actual data
- research notebook persistence
- Jupyter export
- clear architecture and documentation

## 12. Summary

This project aims to create a serious, research-oriented dataset analysis environment that helps users discover what matters in a dataset, understand it quickly, and perform meaningful analysis without rebuilding the same tooling from scratch. The combination of automated understanding, dataset-aware AI, and reproducible notebook export gives it a clear focus beyond generic dashboarding.
