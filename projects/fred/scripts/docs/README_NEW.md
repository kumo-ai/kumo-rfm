# FRED Series Recommendation System

**Use Kumo RFM to discover relevant FRED economic series based on queries, similar series, or categories.**

## Purpose

This project recommends FRED (Federal Reserve Economic Data) series using two approaches:

1. **Kumo RFM** (Primary) - Graph-based ML recommendation engine
2. **Baseline ML** (Validation) - Traditional ML for comparison/validation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with FRED_API_KEY and KUMO_API_KEY

# Fetch FRED data
python3 01_fetch_fred_data.py --search "GDP" "unemployment" "inflation"

# Parse to structured format
python3 02_parse_fred_txt.py --input txt --output data --format parquet

# Create embeddings (for baseline)
python3 04_vector_search.py --data data/fred_series_metadata.parquet --create
```

## Usage

### Baseline Recommendations (Traditional ML)

```bash
# Text query
python3 recommend_series.py --text "housing market inflation" --top-k 10

# Similar series
python3 recommend_series.py --series PAYEMS --top-k 10

# Category exploration
python3 recommend_series.py --category Employment --top-k 10

# Multi-series context
python3 recommend_series.py --multi GDPC1 UNRATE --top-k 10
```

### Kumo RFM Recommendations (Primary)

```bash
# Set API key
export KUMO_API_KEY='your-key'

# Run recommendations
python3 05_kumo_rfm_integration.py --recommend "inflation housing" --top-k 10

# Similar series via Kumo
python3 05_kumo_rfm_integration.py --similar PAYEMS --top-k 10

# Category recommendations
python3 05_kumo_rfm_integration.py --category Employment --top-k 10
```

## Comparison: Baseline vs Kumo RFM

| Feature | Baseline ML | Kumo RFM |
|---------|-------------|----------|
| **Method** | Vector similarity, keyword matching | Graph-based relational learning |
| **Setup** | Requires embeddings generation | Builds graph from metadata |
| **Strengths** | Fast, interpretable, no API | Better at complex relationships |
| **Use Case** | Validation, fallback | Primary recommendations |
| **API Required** | No | Yes (KUMO_API_KEY) |

## Architecture

```
User Query
    ↓
┌─────────────────────┬──────────────────────┐
│                     │                      │
│  Baseline ML        │  Kumo RFM           │
│  (recommend_series) │  (05_kumo_rfm)      │
│                     │                      │
│  - Embeddings       │  - Graph structure   │
│  - Keyword match    │  - PQL queries       │
│  - Popularity       │  - Predictions       │
│                     │                      │
└─────────────────────┴──────────────────────┘
    ↓                       ↓
Compare Results → Validate Quality
```

## Project Structure

```
Core Pipeline:
├── 01_fetch_fred_data.py       # Fetch from FRED API
├── 02_parse_fred_txt.py         # Parse to structured data
├── 04_vector_search.py          # Generate embeddings (baseline)
├── recommend_series.py          # Baseline ML recommender 
├── 05_kumo_rfm_integration.py  # Kumo RFM recommender 
└── 06_prepare_monolith_features.py # Feature tracking for Kumo

Support:
├── 03_load_to_postgres.py      # Optional: PostgreSQL storage
├── 051_kumo_rfm_cli.py         # Interactive CLI
└── 99_pipeline.py              # Full pipeline orchestrator

Configuration:
├── requirements.txt            # Dependencies
├── .env.example                # Environment template
└── create_tables.sql           # PostgreSQL schema

Examples & Archive:
├── examples/                   # Advanced demos
└── archive/                    # Unused scripts
```

## Recommendation Methods

### 1. Text Query → Series

**Input:** "inflation impact on housing market"

**Baseline approach:**
- Keyword matching in title/category
- Semantic similarity via embeddings
- Popularity boost

**Kumo approach:**
- Graph relationships between terms and series
- Predictive relevance scoring

### 2. Series → Similar Series

**Input:** Series ID (e.g., "PAYEMS")

**Baseline approach:**
- Vector similarity from embeddings
- Category/frequency matching

**Kumo approach:**
- Graph-based similarity
- Relationship discovery

### 3. Category → Top Series

**Input:** Category name (e.g., "Employment")

**Baseline approach:**
- Filter by category
- Rank by popularity

**Kumo approach:**
- Predict best series for category
- Consider user context

### 4. Multi-Series → Additional Context

**Input:** Multiple series (e.g., ["GDPC1", "UNRATE"])

**Baseline approach:**
- Average similarity to group
- Common category members

**Kumo approach:**
- Find series that complete the graph
- Predictive context addition

## Key Features

### Baseline ML (recommend_series.py)
-  No API key required
-  Fast inference
-  Works offline with cached embeddings
-  Transparent scoring
-  Multiple methods (keyword, embedding, hybrid)

### Kumo RFM (05_kumo_rfm_integration.py)
-  Graph-based reasoning
-  No feature engineering required
-  Complex relationship discovery
-  Predictive queries (PQL)
-  Better at implicit patterns

### Monolith Features (06_prepare_monolith_features.py)
- Track which features Kumo RFM uses
- Monitor feature importance
- Validate predictions
- Export for analysis

## Validation Workflow

1. **Generate recommendations with both methods**
   ```bash
   python3 recommend_series.py --text "inflation" > baseline.txt
   python3 05_kumo_rfm_integration.py --text "inflation" > kumo.txt
   ```

2. **Compare results**
   - Overlap in recommendations?
   - Which finds better series?
   - Speed differences?

3. **Use Monolith to track**
   - What features drove Kumo's choices?
   - Which metadata was most predictive?

## Example Session

```bash
# 1. Get baseline recommendations
$ python3 recommend_series.py --text "GDP unemployment" --top-k 5
Top 5 Recommendations:
- GDPC1: Real Gross Domestic Product (similarity: 0.89)
- UNRATE: Unemployment Rate (similarity: 0.87)
- PAYEMS: All Employees: Total Nonfarm (similarity: 0.82)
...

# 2. Compare with Kumo RFM
$ python3 05_kumo_rfm_integration.py --recommend "GDP unemployment" --top-k 5
Kumo RFM Recommendations:
- GDPC1: Real Gross Domestic Product (confidence: 0.94)
- UNRATE: Unemployment Rate (confidence: 0.91)
- INDPRO: Industrial Production Index (confidence: 0.88)
...

# 3. Analyze why Kumo recommended differently
$ python3 06_prepare_monolith_features.py --analyze
Feature Importance:
- category_id: 0.45
- frequency_id: 0.32
- popularity_vs_category: 0.23
...
```

## API Keys

### FRED API Key (Required)
- Free at: https://fred.stlouisfed.org/docs/api/api_key.html
- Used by: `01_fetch_fred_data.py`
- Set in `.env`: `FRED_API_KEY=your_key`

### Kumo API Key (For Kumo RFM)
- Get at: https://kumorfm.ai
- Used by: `05_kumo_rfm_integration.py`
- Set in `.env`: `KUMO_API_KEY=your_key`

## Performance

| Operation | Baseline ML | Kumo RFM |
|-----------|-------------|----------|
| Cold start (no cache) | 2-3s | 3-5s |
| With cache | <1s | 1-2s |
| Batch (100 queries) | ~30s | ~60s |

## Next Steps

1.  Run baseline recommendations
2.  Get Kumo API key
3.  Compare recommendation quality
4.  Tune parameters for your use case
5.  Build application on top of recommender

## Troubleshooting

**"No KUMO_API_KEY"**
```bash
export KUMO_API_KEY='your-key'
# Or add to .env file
```

**"sentence-transformers not found"**
```bash
pip install sentence-transformers
```

**"No embeddings found"**
```bash
python3 04_vector_search.py --data data/fred_series_metadata.parquet --create
```

## References

- FRED API: https://fred.stlouisfed.org/
- Kumo RFM: https://kumorfm.ai/
- Sentence Transformers: https://www.sbert.net/
