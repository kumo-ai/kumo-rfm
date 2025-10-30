# Quick Start Guide - FRED Series Recommender

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

## Get Data

```bash
# Fetch from FRED API
python3 01_fetch_fred_data.py --search "GDP" "unemployment" "inflation"

# Parse to parquet
python3 02_parse_fred_txt.py --input txt --output data --format parquet

# (Optional) Create embeddings for baseline
python3 04_vector_search.py --data data/fred_series_metadata.parquet --create
```

## Recommendation Methods

### Baseline ML (No API Key Required)

```bash
# Text query
python3 recommend_series.py --text "housing inflation" --top-k 10

# Similar series
python3 recommend_series.py --series PAYEMS --top-k 10

# Category
python3 recommend_series.py --category Employment --top-k 10

# Multi-series
python3 recommend_series.py --multi GDPC1 UNRATE --top-k 10
```

### Kumo RFM (Requires KUMO_API_KEY)

```bash
export KUMO_API_KEY='your-key'

# Text query
python3 05_kumo_rfm_integration.py --recommend "housing inflation" --top-k 10

# Similar series
python3 05_kumo_rfm_integration.py --similar PAYEMS --top-k 10

# Category
python3 05_kumo_rfm_integration.py --category Employment --top-k 10

# Multi-series
python3 05_kumo_rfm_integration.py --multi GDPC1 UNRATE --top-k 10

# Explain
python3 05_kumo_rfm_integration.py --explain PAYEMS UNRATE
```

## Compare Both

```bash
# Run both
python3 recommend_series.py --text "inflation" > baseline.txt
python3 05_kumo_rfm_integration.py --recommend "inflation" > kumo.txt

# Compare
diff baseline.txt kumo.txt
```

## Key Files

- `recommend_series.py` - Baseline recommender (embeddings + keywords)
- `05_kumo_rfm_integration.py` - Kumo RFM recommender (graph-based)
- `06_prepare_monolith_features.py` - Feature tracking
- `PROJECT_VISION.md` - Architecture overview
- `README_NEW.md` - Full documentation

## Next Steps

1. Test both recommenders
2. Compare results
3. Analyze feature importance
4. Build application on top

## Need Help?

- Vision: `PROJECT_VISION.md`
- Full docs: `README_NEW.md`
- Enhancement details: `ENHANCEMENT_COMPLETE.md`
