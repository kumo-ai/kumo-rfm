# FRED Data Processing & Integration Pipeline

Comprehensive pipeline for parsing, analyzing, and transforming FRED economic series data for various use cases including Kumo RFM, vector search, and ByteDance Monolith.

## Overview

This pipeline processes FRED (Federal Reserve Economic Data) series from text files into structured formats, enabling:

- **SQL/PostgreSQL querying** for relational analysis
- **Kumo RFM integration** for predictive modeling without feature engineering
- **Vector search** for semantic discovery of economic indicators
- **Workflow integrations** with ElasticSearch, Pinecone, Weaviate, Neo4j
- **Monolith preparation** for ByteDance's recommendation algorithm

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install manually:
pip install pandas numpy psycopg2-binary pyarrow python-dotenv
pip install sentence-transformers faiss-cpu
pip install matplotlib seaborn networkx

# Optional packages
pip install kumoai                    # Kumo RFM integration (requires API key)
pip install tensorflow                # TFRecord export for Monolith

# Workflow integrations (optional)
pip install elasticsearch pinecone-client weaviate-client neo4j
```

## Quick Start

### Option 1: Run Full Pipeline

```bash
# Make pipeline scripts executable
chmod +x run_pipeline.sh
chmod +x pipeline.py

# Run full pipeline (bash)
./run_pipeline.sh

# Or using Python orchestrator
python3 pipeline.py --full
```

### Option 2: Run Individual Steps

```bash
# 1. Parse txt files
python3 parse_fred_txt.py --input txt --output data --format all

# 2. Load into DuckDB
python3 load_to_duckdb.py --init --data data/fred_series_metadata.parquet

# 3. Create vector embeddings
python3 vector_search.py --data data/fred_series_metadata.parquet --create

# 4. Run Kumo RFM demo
export KUMO_API_KEY='your-api-key'
python3 kumo_rfm_integration.py --demo

# 5. Prepare Monolith features
python3 prepare_monolith_features.py --data data/fred_series_metadata.parquet \
  --embeddings data/embeddings/embeddings.npy --output data/monolith
```

## Scripts Overview

### 1. `parse_fred_txt.py`
Parse FRED series text files into structured DataFrames.

```bash
# Parse with all output formats
python3 parse_fred_txt.py --input txt --output data --format all

# Parse to CSV only
python3 parse_fred_txt.py --input txt --output data --format csv
```

**Outputs:**
- `data/fred_series_metadata.parquet`
- `data/fred_series_metadata.csv`
- `data/fred_series_metadata.json`

### 2. `create_tables.sql` & `03_load_to_postgres.py`
Create SQL schema and load data into PostgreSQL.

```bash
# Set PostgreSQL connection details in .env
cp .env.example .env
# Edit .env with your PostgreSQL credentials

# Initialize schema and load data
python3 03_load_to_postgres.py --init --data data/fred_series_metadata.parquet

# Query the database using psql
psql -h localhost -U postgres -d fred -c "SELECT category, COUNT(*) FROM series_metadata GROUP BY category"

# Export for Kumo
python3 03_load_to_postgres.py --export-kumo data/kumo
```

**Outputs:**
- PostgreSQL database tables
- `data/kumo/` - Kumo-formatted exports

### 3. `kumo_rfm_integration.py`
Integrate with Kumo RFM for predictive queries.

```bash
# Set API key
export KUMO_API_KEY='your-api-key'

# Run demo
python3 kumo_rfm_integration.py --demo
```

**Use cases:**
- Predict series popularity
- Recommend related indicators
- SQL-style predictive queries

### 4. `vector_search.py`
Semantic search using sentence transformers.

```bash
# Create embeddings
python3 vector_search.py --data data/fred_series_metadata.parquet \
  --create --save data/embeddings

# Search by query
python3 vector_search.py --load data/embeddings \
  --search "unemployment and inflation" --top-k 10

# Find similar series
python3 vector_search.py --load data/embeddings \
  --similar PAYEMS --top-k 5

# Run demo
python3 vector_search.py --demo
```

**Outputs:**
- `data/embeddings/embeddings.npy` - Vector embeddings
- `data/embeddings/faiss.index` - FAISS index for fast search

### 5. `workflow_integrations.py`
Examples for integrating with search and database systems.

```bash
# View integration examples
python3 workflow_integrations.py --demo
```

**Integrations:**
- **ElasticSearch**: Full-text search and aggregations
- **Pinecone**: Cloud vector database
- **Weaviate**: Semantic search
- **Neo4j**: Graph relationships

### 6. `prepare_monolith_features.py`
Prepare data for ByteDance Monolith recommendation algorithm.

```bash
# Prepare features with embeddings
python3 prepare_monolith_features.py \
  --data data/fred_series_metadata.parquet \
  --embeddings data/embeddings/embeddings.npy \
  --output data/monolith \
  --format parquet \
  --n-interactions 1000

# Export as TFRecord
python3 prepare_monolith_features.py \
  --data data/fred_series_metadata.parquet \
  --output data/monolith \
  --format tfrecord
```

**Outputs:**
- `data/monolith/item_features.parquet` - Series features
- `data/monolith/user_features.parquet` - User behavior sequences
- `data/monolith/training_samples.parquet` - Training data
- `data/monolith/feature_config.json` - Feature configuration
- `data/monolith/train.tfrecord` - TensorFlow format (optional)

### 7. `pipeline.py`
Orchestrate the entire pipeline.

```bash
# Run full pipeline
python3 pipeline.py --full

# Run specific steps
python3 pipeline.py --steps parse duckdb embeddings monolith

# Custom configuration
python3 pipeline.py --full --txt-dir my_txt --data-dir my_data
```

## Use Cases

### 1. SQL Analysis with PostgreSQL

```sql
-- Connect to database
psql -h localhost -U postgres -d fred

-- Find employment-related series
SELECT series_id, title, popularity
FROM series_metadata
WHERE category LIKE '%Employment%'
ORDER BY popularity DESC
LIMIT 10;

-- Aggregate by category
SELECT category, COUNT(*) as series_count, AVG(popularity) as avg_popularity
FROM series_metadata
GROUP BY category
ORDER BY series_count DESC;
```

### 2. Kumo RFM Predictive Queries

```python
from kumo_rfm_integration import KumoFREDIntegration
import pandas as pd

# Initialize
kumo = KumoFREDIntegration(api_key='your-key')
df = pd.read_parquet('data/fred_series_metadata.parquet')

# Predict popularity
results = kumo.predict_series_popularity(df)

# Recommend series
recommendations = kumo.recommend_series_for_analysis(df, 'inflation', top_k=5)
```

### 3. Semantic Search

```python
from vector_search import FREDVectorSearch

search = FREDVectorSearch()
search.load('data/embeddings')

# Search by natural language
results = search.search("housing market indicators", top_k=10)

# Find similar series
similar = search.find_similar_series("PAYEMS", top_k=5)
```

### 4. Monolith Feature Engineering

```python
from prepare_monolith_features import MonolithFeaturePreparation

prep = MonolithFeaturePreparation()

# Create features
monolith_data = prep.prepare_monolith_format(series_df, interactions_df, embeddings)

# Export for training
prep.export_to_tfrecord(monolith_data, 'data/monolith')
```

## Data Formats

### Parsed Metadata Structure

```
series_id          : str    - FRED series identifier
title              : str    - Full series title
frequency          : str    - Data frequency (D, W, M, Q, A)
frequency_name     : str    - Full frequency name
popularity         : int    - Popularity score (0-100)
notes              : str    - Series description and metadata
source_file        : str    - Original source filename
category           : str    - Extracted category
has_notes          : bool   - Whether notes exist
notes_length       : int    - Length of notes field
```

### Monolith Feature Format

**Item Features:**
- Categorical: series_id, category_id, frequency_id
- Numerical: popularity, notes_length, has_notes
- Dense: embedding vectors (if provided)

**User Features:**
- Sequences: series_sequence, category_sequence
- Scalars: series_sequence_length, latest_category

**Training Samples:**
- user_id, series_id, timestamp, label

## Advanced Usage

### Custom Embeddings

```python
from sentence_transformers import SentenceTransformer
from vector_search import FREDVectorSearch

# Use custom model
search = FREDVectorSearch(model_name='all-mpnet-base-v2')
search.create_embeddings(df)
```

### Graph Analysis with Neo4j

```python
from workflow_integrations import Neo4jIntegration

neo4j = Neo4jIntegration(uri="bolt://localhost:7687")
neo4j.create_series_nodes(df)
neo4j.create_category_relationships()

# Find related series
related = neo4j.find_related_series("PAYEMS", max_depth=2)
```

### ElasticSearch Full-Text Search

```python
from workflow_integrations import ElasticSearchIntegration

es = ElasticSearchIntegration()
es.create_index()
es.index_series(df)

# Search
results = es.search("consumer price index")
```

## Troubleshooting

### Missing Dependencies
```bash
# If imports fail, install missing packages
pip install pandas numpy duckdb pyarrow sentence-transformers
```

### Kumo API Key Not Found
```bash
export KUMO_API_KEY='your-api-key-here'
# Or pass directly: python3 kumo_rfm_integration.py --api-key 'your-key'
```

### TensorFlow Not Available
```bash
# For TFRecord export
pip install tensorflow
# Or use Parquet format instead: --format parquet
```

### Memory Issues with Large Datasets
```bash
# Process in batches or use sampling
python3 prepare_monolith_features.py --n-interactions 500
```

## Performance Tips

1. **Vector Search**: Use FAISS for datasets > 10K series
2. **DuckDB**: Create indexes on frequently queried columns
3. **Embeddings**: Cache embeddings to disk after first creation
4. **Monolith**: Use Parquet for faster I/O vs TFRecord

## License & Credits

Data source: Federal Reserve Economic Data (FRED)
- FRED API: https://fred.stlouisfed.org/

Kumo RFM: https://kumorfm.ai/
ByteDance Monolith: https://arxiv.org/abs/2209.07663
