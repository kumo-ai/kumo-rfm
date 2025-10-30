#!/usr/bin/env bash
# Orchestration script: end-to-end pipeline for FRED txt -> structured -> DB -> search -> Kumo -> Monolith
# Usage: ./run_pipeline.sh

set -euo pipefail

# 1) Parse txt into structured data
python3 parse_fred_txt.py --input txt --output data --format all

# 2) Initialize PostgreSQL schema and load data
python3 03_load_to_postgres.py --init --data data/fred_series_metadata.parquet --format parquet

# 3) Export for Kumo (optional artifact generation)
python3 03_load_to_postgres.py --export-kumo data/kumo

# 4) Create embeddings and run vector search demo
python3 vector_search.py --data data/fred_series_metadata.parquet --create --save data/embeddings
python3 vector_search.py --demo

# 5) Run Kumo demo (requires KUMO_API_KEY env var)
python3 kumo_rfm_integration.py --data data/fred_series_metadata.parquet --demo || true

# 6) Prepare Monolith features (uses embeddings if present)
python3 prepare_monolith_features.py --data data/fred_series_metadata.parquet \
  --embeddings data/embeddings/embeddings.npy \
  --output data/monolith --format parquet --n-interactions 1000

echo "\nPipeline complete. Outputs:"
echo "- Structured data: data/fred_series_metadata.*"
echo "- PostgreSQL: localhost:5432/fred"
echo "- Kumo exports: data/kumo/"
echo "- Embeddings: data/embeddings/"
echo "- Monolith features: data/monolith/"
