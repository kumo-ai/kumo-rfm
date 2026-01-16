# kumo-rfm

Examples, demos, and benchmarks for KumoRFM - a foundation model for business data prediction.

## Overview

This repository provides reference implementations, demo applications, and benchmark scripts for the cloud-based KumoRFM model. It is not a library - the actual model is accessed via the `kumoai` SDK from PyPI. Primary consumers are developers evaluating KumoRFM, building integrations, or reproducing benchmark results.

## Directory Structure

```
kumo-rfm/
├── notebooks/           # Jupyter tutorials and examples (start here)
│   ├── quickstart.ipynb       # Basic setup and first prediction
│   ├── handbook.ipynb         # Comprehensive API guide
│   ├── predictive_query.ipynb # PQL syntax tutorial
│   ├── explanations.ipynb     # Model interpretability
│   └── *_agent.ipynb          # MCP integration examples (CrewAI, LangGraph, OpenAI)
├── apps/
│   ├── short-form-demo/       # Fashion e-commerce Streamlit app
│   └── support_assistant/     # Customer support Streamlit app
└── benchmarks/
    └── relbench/              # RelBench evaluation scripts
```

## Key Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `Instant_Predict.py` | `apps/short-form-demo/` | Minimal prediction example - loads data, builds graph, runs PQL query |
| `display.py` | `apps/short-form-demo/` | Streamlit UI for product recommendations |
| `complete.py` | `apps/short-form-demo/` | Full demo with churn detection, recommendations, returns |
| `support_assistant.py` | `apps/support_assistant/` | Customer support dashboard (churn, LTV, recommendations) |
| `relbench_classification.py` | `benchmarks/relbench/` | Classification benchmark runner (AUROC, F1) |
| `relbench_regression.py` | `benchmarks/relbench/` | Regression benchmark runner (MAE) |
| `pql.py` | `benchmarks/relbench/` | PQL-based benchmark (loads from S3, no relbench dependency) |
| `relbench_from_s3.py` | `benchmarks/relbench/` | S3-based benchmark with custom context tables |

## Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Parquet/CSV    │───▶│   LocalTable    │───▶│   LocalGraph    │
│  (S3 or local)  │    │  + metadata     │    │  + foreign keys │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Predictions    │◀───│   KumoRFM API   │◀───│  PQL Query      │
│  (DataFrame)    │    │   (cloud)       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Core pattern (from `Instant_Predict.py`):**
```python
rfm.init(api_key=API_KEY)
graph = rfm.LocalGraph.from_data(data)  # dict of DataFrames
model = rfm.KumoRFM(graph)
pred_df = model.predict("PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) RANK TOP 10 FOR users.user_id=50")
```

## Integration Points

| Kumo Repository | Relationship | Usage |
|-----------------|--------------|-------|
| `kumoai` (PyPI) | Required SDK | Provides `rfm.KumoRFM`, `rfm.LocalGraph`, `rfm.LocalTable` |
| `kumo-rfm-mcp` | MCP server | Enables agentic workflows (see `*_agent.ipynb` notebooks) |

**External dependencies:**
- `relbench`: Optional, for benchmark dataset loading
- `streamlit`: Demo app framework
- `sklearn`: Benchmark metrics (AUROC, F1, MAE)

## Key Patterns & Conventions

- **PQL (Predictive Query Language)**: SQL-like syntax for predictions
  - Classification: `PREDICT COUNT(orders.*, 0, 90, days)=0 FOR users.user_id={id}`
  - Regression: `PREDICT SUM(orders.price, 0, 90, days) FOR users.user_id={id}`
  - Ranking: `PREDICT LIST_DISTINCT(...) RANK TOP K FOR ...`
- **Graph construction**: `LocalGraph.from_data(dict)` auto-infers metadata; use `graph.link()` for foreign keys
- **Batch predictions**: Use `with model.batch_mode(batch_size=N)` context manager
- **Environment**: API key via `KUMO_API_KEY` env var or `.env` file
- **Sample data**: `s3://kumo-sdk-public/rfm-datasets/online-shopping/` (users, orders, items parquet files)

## Entry Points

| Task | Start Here |
|------|------------|
| Learn KumoRFM basics | `notebooks/quickstart.ipynb` |
| Understand PQL syntax | `notebooks/predictive_query.ipynb` |
| See minimal code example | `apps/short-form-demo/Instant_Predict.py` (27 lines) |
| Run a demo app | `cd apps/short-form-demo && streamlit run display.py` |
| Reproduce benchmarks | `python benchmarks/relbench/relbench_classification.py --dataset=rel-hm` |
| Build MCP agent | `notebooks/ecom_agent.ipynb` (CrewAI), `notebooks/insurance_agent.ipynb` (LangGraph) |
| Model explainability | `notebooks/explanations.ipynb` |

## PQL Quick Reference

| Prediction Type | PQL Template |
|-----------------|--------------|
| Binary churn | `PREDICT COUNT(orders.*, 0, 90, days)=0 FOR users.user_id={id}` |
| Revenue prediction | `PREDICT SUM(orders.price, 0, 90, days) FOR users.user_id={id}` |
| Top-K recommendations | `PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) RANK TOP {k} FOR users.user_id={id}` |
| Batch prediction | `PREDICT ... FOR table.column IN ({ids})` or `FOR EACH table.column` |
| Conditional | `PREDICT ... WHERE COUNT(orders.*, -91, 0, days)>0` |
