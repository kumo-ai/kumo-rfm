# Data Enrichment Guide

## How to Get More Data for Advanced KumoAI Demos

The advanced demo works best with rich, diverse economic data. Here's how to expand your dataset using `01_fetch_fred_data.py`.

## Quick Start

### 1. Get a FRED API Key (Free)

Visit: https://fred.stlouisfed.org/docs/api/api_key.html
- Sign up for a free account
- Generate your API key
- Set it as an environment variable:

```bash
export FRED_API_KEY='your-api-key-here'
```

### 2. Fetch More Data

The script has predefined categories that are perfect for KumoAI analysis:

```bash
# List available categories
python3 01_fetch_fred_data.py --list-categories

# Fetch specific category
python3 01_fetch_fred_data.py --category core

# Fetch all indicators (comprehensive)
python3 01_fetch_fred_data.py --all

# Custom search
python3 01_fetch_fred_data.py --search "cryptocurrency" "blockchain" "fintech"
```

## Data Categories for KumoAI

### Core Economic Indicators (Best for Multi-Table Demo)
```bash
python3 01_fetch_fred_data.py --category core
```
Includes: GDP, inflation, deflation, CPI, unemployment, employment, wages

**Use case**: Demo 1 (Multi-Table Relationships) - creates rich category hierarchies

### Monetary Policy (Best for Temporal Demo)
```bash
python3 01_fetch_fred_data.py --category monetary
```
Includes: Interest rates, money supply, bank reserves, credit

**Use case**: Demo 2 (Temporal Predictions) - frequent updates, time-series patterns

### Housing Market (Best for Anomaly Detection)
```bash
python3 01_fetch_fred_data.py --category housing
```
Includes: Home sales, prices, starts, permits, mortgages

**Use case**: Demo 3 (Anomaly Detection) - volatile, regional variations

### Market Data (Best for Feature Importance)
```bash
python3 01_fetch_fred_data.py --category markets
```
Includes: Stock indices, volatility, profits, yields

**Use case**: Demo 4 (Feature Importance) - clear popularity patterns

## Advanced Data Collection Strategies

### Strategy 1: Comprehensive Coverage
Get everything for maximum demo variety:

```bash
python3 01_fetch_fred_data.py --all --limit 1000
```

This fetches:
- 9 categories
- ~90 search terms
- Up to 1000 series per term
- Estimated: 50,000+ series

**Best for**: All demos, production-quality analysis

### Strategy 2: Time-Series Focus
Get data with frequent updates for temporal analysis:

```bash
python3 01_fetch_fred_data.py \
  --search "daily" "weekly" "high frequency" \
  --limit 2000
```

**Best for**: Demo 2 (Temporal Predictions)

### Strategy 3: Regional Data
Get state/city level data for relationship analysis:

```bash
python3 01_fetch_fred_data.py \
  --search "state" "county" "MSA" "metro" \
  --limit 5000
```

**Best for**: Demo 1 (Multi-Table), Demo 6 (Causal Inference)

### Strategy 4: Industry-Specific
Focus on specific economic sectors:

```bash
python3 01_fetch_fred_data.py \
  --search "manufacturing" "agriculture" "technology" "retail" \
  --limit 1000
```

**Best for**: Demo 3 (Anomaly Detection), Demo 7 (Automated Insights)

## After Fetching Data

### 1. Parse the New Data

```bash
# Parse all txt files in the txt/ directory
python3 02_parse_fred_txt.py --input txt --output data --format parquet
```

This creates/updates:
- `data/fred_series_metadata.parquet` (for KumoAI demos)
- `data/fred_series_metadata.csv`
- `data/fred_series_metadata.json`

### 2. Load to PostgreSQL (Optional)

```bash
python3 03_load_to_postgres.py --init --data data/fred_series_metadata.parquet
```

Enables CLI queries and advanced SQL analysis.

### 3. Create Embeddings

```bash
python3 04_vector_search.py --data data/fred_series_metadata.parquet --create
```

Required for Demo 2 and NetworkX similarity analysis.

### 4. Run Advanced Demos

```bash
export KUMO_API_KEY='your-kumo-key'
python3 10_advanced_kumo_demo.py
```

## Data Quality Improvements

### Add Metadata Richness

The advanced demos work better with rich metadata. Fetch series with:

1. **Detailed Notes**: More context for analysis
```bash
python3 01_fetch_fred_data.py --search "labor force participation" --limit 500
```

2. **Multiple Frequencies**: Better temporal patterns
```bash
python3 01_fetch_fred_data.py --search "monthly" "quarterly" "annual"
```

3. **Historical Data**: Longer time series
```bash
python3 01_fetch_fred_data.py --search "historical" "long-term" "vintage"
```

### Fetch Related Series

For Demo 6 (Causal Inference), fetch related indicators:

```bash
# Unemployment + related factors
python3 01_fetch_fred_data.py --search \
  "unemployment rate" \
  "job openings" \
  "labor force" \
  "wages" \
  "education attainment"

# Housing + factors
python3 01_fetch_fred_data.py --search \
  "home prices" \
  "mortgage rate" \
  "housing supply" \
  "construction spending" \
  "median income"
```

## Custom Search Queries

### By Frequency
```bash
# Daily data
python3 01_fetch_fred_data.py --search "daily" --limit 2000

# Weekly data  
python3 01_fetch_fred_data.py --search "weekly" --limit 1000

# High frequency
python3 01_fetch_fred_data.py --search "high frequency" --limit 500
```

### By Geography
```bash
# State-level
python3 01_fetch_fred_data.py --search "state" --limit 5000

# County-level
python3 01_fetch_fred_data.py --search "county" --limit 3000

# Metro areas
python3 01_fetch_fred_data.py --search "MSA" "metropolitan" --limit 2000
```

### By Source
```bash
# Bureau of Labor Statistics
python3 01_fetch_fred_data.py --search "BLS" --limit 3000

# Census Bureau
python3 01_fetch_fred_data.py --search "Census" --limit 2000

# Federal Reserve
python3 01_fetch_fred_data.py --search "Federal Reserve" --limit 1000
```

## Optimizing for Specific Demos

### Demo 1: Multi-Table Relationships
**Goal**: Rich category hierarchies

```bash
# Get diverse categories
python3 01_fetch_fred_data.py --all

# Then ensure good category distribution:
python3 02_parse_fred_txt.py --input txt --output data --format parquet
```

### Demo 2: Temporal Predictions
**Goal**: Frequent updates, time patterns

```bash
# Get daily/weekly data
python3 01_fetch_fred_data.py \
  --search "daily" "weekly" "real-time" \
  --limit 2000

# Focus on volatile indicators
python3 01_fetch_fred_data.py \
  --search "stock" "exchange rate" "commodity" \
  --limit 1000
```

### Demo 3: Anomaly Detection
**Goal**: Outliers and unusual patterns

```bash
# Get diverse popularity levels
python3 01_fetch_fred_data.py --all

# Add niche indicators
python3 01_fetch_fred_data.py \
  --search "discontinued" "experimental" "vintage" \
  --limit 500
```

### Demo 4: Feature Importance
**Goal**: Clear popularity gradients

```bash
# Mix popular and obscure
python3 01_fetch_fred_data.py --category core
python3 01_fetch_fred_data.py --category markets
python3 01_fetch_fred_data.py \
  --search "county level" "regional" \
  --limit 2000
```

### Demo 5: What-If Analysis
**Goal**: Comparable series with varying metadata

```bash
# Get series with similar structure but different notes
python3 01_fetch_fred_data.py \
  --search "unemployment rate" \
  --limit 1000

# Regional variations
python3 01_fetch_fred_data.py \
  --search "state unemployment" \
  --limit 5000
```

### Demo 6: Causal Inference
**Goal**: Related indicators for causal analysis

```bash
# Unemployment ecosystem
python3 01_fetch_fred_data.py \
  --search "unemployment" "job openings" "labor force" "wages" \
  --limit 500

# Housing ecosystem
python3 01_fetch_fred_data.py \
  --search "home prices" "mortgage" "construction" "rent" \
  --limit 500
```

### Demo 7: Automated Insights
**Goal**: Comprehensive coverage

```bash
# Everything
python3 01_fetch_fred_data.py --all --limit 1000
```

## Rate Limits and Best Practices

FRED API limits:
- 120 requests per minute
- Default delay: 0.5s between requests (safe)
- Increase delay if hitting limits: `--delay 1.0`

```bash
# Safe for large fetches
python3 01_fetch_fred_data.py --all --delay 1.0
```

## Verification

After fetching new data, verify:

```bash
# Check file count
ls -1 txt/*.txt | wc -l

# Check total series
python3 -c "import pandas as pd; df = pd.read_parquet('data/fred_series_metadata.parquet'); print(f'Total series: {len(df)}')"

# Check categories
python3 -c "import pandas as pd; df = pd.read_parquet('data/fred_series_metadata.parquet'); print(f'Categories: {df.category.nunique()}')"
```

## Example: Full Enrichment Workflow

```bash
# 1. Set API keys
export FRED_API_KEY='your-fred-key'
export KUMO_API_KEY='your-kumo-key'

# 2. Fetch comprehensive data
python3 01_fetch_fred_data.py --all --limit 1000

# 3. Parse to structured format
python3 02_parse_fred_txt.py --input txt --output data --format parquet

# 4. Create embeddings for similarity
python3 04_vector_search.py --data data/fred_series_metadata.parquet --create

# 5. Run advanced demos
python3 10_advanced_kumo_demo.py

# 6. Optional: Network analysis
python3 08_network_visualizations.py \
  --data data/fred_series_metadata.parquet \
  --embeddings data/embeddings/embeddings.npy
```

## Troubleshooting

### "No results found"
- Check search term spelling
- Try broader terms: "employment" instead of "nonfarm employment"
- Check FRED website first: https://fred.stlouisfed.org/

### "API rate limit exceeded"
- Increase delay: `--delay 2.0`
- Fetch in smaller batches
- Wait 60 seconds and retry

### "Parsing errors"
- Some series have unusual formatting
- Parser handles most cases automatically
- Check `02_parse_fred_txt.py` for detailed logs

## Next Steps

After enriching your data:

1. Re-run all 7 advanced demos to see improved results
2. Use NetworkX visualizations with richer data
3. Load to PostgreSQL for SQL analysis
4. Build production pipelines with the patterns learned
