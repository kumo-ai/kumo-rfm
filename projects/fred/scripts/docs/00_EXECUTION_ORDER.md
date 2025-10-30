# FRED Data Pipeline - Execution Order

All scripts have been numbered in the recommended execution order.

##  Execution Sequence

### **STEP 1: Fetch Data from FRED API**
**File:** `01_fetch_fred_data.py`

Fetch economic series metadata from the Federal Reserve database.

```bash
# Set your FRED API key first
export FRED_API_KEY='your-api-key-here'

# Option A: Fetch all indicators (recommended for first run)
python3 01_fetch_fred_data.py --all

# Option B: Fetch specific categories
python3 01_fetch_fred_data.py --category core
python3 01_fetch_fred_data.py --category monetary

# Option C: Search specific terms
python3 01_fetch_fred_data.py --search "GDP" "inflation" "unemployment"
```

**Output:** Creates `.txt` files in `txt/` directory with series metadata

---

### **STEP 2: Parse TXT Files to Structured Data**
**File:** `02_parse_fred_txt.py`

Convert raw text files into structured formats (CSV, Parquet, JSON).

```bash
# Parse all txt files
python3 02_parse_fred_txt.py --input txt --output data --format all

# Or parse to specific format
python3 02_parse_fred_txt.py --input txt --output data --format parquet
```

**Output:** 
- `data/fred_series_metadata.parquet`
- `data/fred_series_metadata.csv`
- `data/fred_series_metadata.json`

---

### **STEP 3: Load Data into PostgreSQL**
**File:** `03_load_to_postgres.py` (uses `create_tables.sql`)

Create PostgreSQL database for relational queries.

```bash
# Set PostgreSQL credentials in .env file
cp .env.example .env
# Edit .env with your PostgreSQL details

# Initialize schema and load data
python3 03_load_to_postgres.py --init --data data/fred_series_metadata.parquet

# Optional: Export for Kumo RFM
python3 03_load_to_postgres.py --export-kumo data/kumo
```

**Output:** 
- PostgreSQL database tables in 'fred' database
- `data/kumo/` - Kumo-formatted exports (optional)

**Query the database:**
```bash
psql -h localhost -U postgres -d fred -c "SELECT category, COUNT(*) FROM series_metadata GROUP BY category"
```

---

### **STEP 4: Create Vector Embeddings**
**File:** `04_vector_search.py`

Generate semantic embeddings for similarity search.

```bash
# Create embeddings and save
python3 04_vector_search.py --data data/fred_series_metadata.parquet \
  --create --save data/embeddings

# Run demo searches
python3 04_vector_search.py --demo

# Search for specific terms
python3 04_vector_search.py --load data/embeddings \
  --search "unemployment and labor market" --top-k 10

# Find similar series
python3 04_vector_search.py --load data/embeddings \
  --similar PAYEMS --top-k 5
```

**Output:** 
- `data/embeddings/embeddings.npy` - Vector embeddings
- `data/embeddings/faiss.index` - FAISS index
- `data/embeddings/series_data.parquet` - Metadata

---

### **STEP 5: Kumo RFM Integration** (Optional)
**File:** `05_kumo_rfm_integration.py`

Run predictive queries using Kumo RFM (requires API key).

```bash
# Set Kumo API key
export KUMO_API_KEY='your-kumo-key'

# Run demo
python3 05_kumo_rfm_integration.py --demo

# Or provide key directly
python3 05_kumo_rfm_integration.py --api-key 'your-key' --demo
```

**Output:** Predictive analysis results (console output)

---

### **STEP 6: Prepare Monolith Features**
**File:** `06_prepare_monolith_features.py`

Transform data for ByteDance Monolith recommendation algorithm.

```bash
# Prepare with embeddings (recommended)
python3 06_prepare_monolith_features.py \
  --data data/fred_series_metadata.parquet \
  --embeddings data/embeddings/embeddings.npy \
  --output data/monolith \
  --format parquet \
  --n-interactions 1000

# Or export as TFRecord
python3 06_prepare_monolith_features.py \
  --data data/fred_series_metadata.parquet \
  --output data/monolith \
  --format tfrecord
```

**Output:** 
- `data/monolith/item_features.parquet` - Series features
- `data/monolith/user_features.parquet` - User sequences
- `data/monolith/training_samples.parquet` - Training data
- `data/monolith/feature_config.json` - Configuration
- `data/monolith/train.tfrecord` - TensorFlow format (optional)

---

### **STEP 7: Workflow Integrations** (Optional)
**File:** `07_workflow_integrations.py`

View examples for ElasticSearch, Pinecone, Weaviate, Neo4j integrations.

```bash
# View integration examples
python3 07_workflow_integrations.py --demo
```

**Output:** Example code and documentation (console output)

---

### **STEP 99: Full Pipeline Automation**
**Files:** `99_pipeline.py` or `99_run_pipeline.sh`

Run the entire pipeline end-to-end (after you've tested individual steps).

```bash
# Option A: Python orchestrator (recommended)
python3 99_pipeline.py --full

# Option B: Bash script
./99_run_pipeline.sh

# Run specific steps only
python3 99_pipeline.py --steps parse duckdb embeddings monolith
```

---

##  Recommended First-Time Workflow

For your first complete run:

```bash
# 1. Set API keys
export FRED_API_KEY='your-fred-key'
export KUMO_API_KEY='your-kumo-key'  # Optional

# 2. Fetch data (if you haven't already)
python3 01_fetch_fred_data.py --all

# 3. Run full pipeline
python3 99_pipeline.py --full
```

##  Expected Directory Structure After Full Run

```
/home/david/Desktop/kumo-rfm/projects/fred/map/
├── 00_EXECUTION_ORDER.md          # This file
├── 01_fetch_fred_data.py          # Data fetcher
├── 02_parse_fred_txt.py           # Parser
├── 03_load_to_duckdb.py           # DB loader
├── 04_vector_search.py            # Vector search
├── 05_kumo_rfm_integration.py     # Kumo integration
├── 06_prepare_monolith_features.py # Monolith prep
├── 07_workflow_integrations.py    # Integration examples
├── 99_pipeline.py                 # Full pipeline
├── 99_run_pipeline.sh             # Bash pipeline
├── create_tables.sql              # SQL schema
├── README.md                      # Full documentation
├── txt/                           # Raw FRED data
│   ├── gdp_series_20251029.txt
│   ├── inflation_series_20251029.txt
│   └── ...
├── data/                          # Processed data
│   ├── fred_series_metadata.parquet
│   ├── fred_series_metadata.csv
│   ├── fred_series_metadata.json
│   ├── embeddings/
│   │   ├── embeddings.npy
│   │   ├── faiss.index
│   │   └── series_data.parquet
│   ├── kumo/
│   │   ├── series_metadata.parquet
│   │   └── series_relationships.parquet
│   └── monolith/
│       ├── item_features.parquet
│       ├── user_features.parquet
│       ├── training_samples.parquet
│       └── feature_config.json
└── fred.db                        # DuckDB database
```

##  Required API Keys

### FRED API Key (Required for Step 1)
- Get free at: https://fred.stlouisfed.org/docs/api/api_key.html
- Set as: `export FRED_API_KEY='your-key'`

### Kumo API Key (Optional, for Step 5)
- Get at: https://kumorfm.ai
- Set as: `export KUMO_API_KEY='your-key'`

##  Tips

1. **Start Small**: Test with `--category core` before running `--all`
2. **Check Dependencies**: Install required packages from `README.md`
3. **Skip Optional Steps**: Steps 5 and 7 are optional demonstrations
4. **Reuse Embeddings**: Step 4 embeddings are cached for reuse
5. **Monitor API Limits**: FRED has rate limits (adjust `--delay` if needed)

##  Common Issues

### "FRED_API_KEY not found"
```bash
export FRED_API_KEY='your-key-here'
```

### "requests library not found"
```bash
pip install requests
```

### "sentence-transformers not installed"
```bash
pip install sentence-transformers faiss-cpu
```

### "TensorFlow not available"
Use Parquet format instead:
```bash
python3 06_prepare_monolith_features.py --format parquet
```

##  Quick Commands Cheat Sheet

```bash
# Complete fresh start
export FRED_API_KEY='your-key'
python3 01_fetch_fred_data.py --all
python3 99_pipeline.py --full

# Query database (requires PostgreSQL running)
psql -h localhost -U postgres -d fred

# Search series
python3 04_vector_search.py --load data/embeddings --search "your query"

# List categories
python3 01_fetch_fred_data.py --list-categories
```
