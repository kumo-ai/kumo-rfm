# Network Visualization Summary

## What You Can Do Now

Your FRED project now has comprehensive network visualization capabilities combining NetworkX and KumoAI:

### Visualizations Available

1. **Category Network** - See how economic categories relate to each other
2. **Series Similarity Network** - Discover which indicators are similar
3. **Community Detection** - Automatically find clusters of related data
4. **Centrality Analysis** - Identify the most important economic indicators
5. **Interactive CLI** - Query and visualize from the command line

### Technologies

- **NetworkX**: Graph creation, analysis, and basic visualization
- **KumoAI**: Predictive queries and relationship discovery
- **Matplotlib/Seaborn**: High-quality static visualizations
- **Export options**: JSON for D3.js, GEXF for Gephi

## Quick Examples

### 1. Basic Network Visualization
```bash
python3 08_network_visualizations.py --data data/fred_series_metadata.parquet
```

Creates 4 visualizations:
- Category network (how categories connect)
- Series network (which series are similar)
- Community detection (automatic clustering)
- Centrality analysis (most important series)

### 2. With Semantic Embeddings
```bash
# First, create embeddings
python3 04_vector_search.py --create

# Then visualize with semantic similarity
python3 08_network_visualizations.py \
  --data data/fred_series_metadata.parquet \
  --embeddings data/embeddings/embeddings.npy \
  --max-nodes 100 \
  --similarity 0.75
```

### 3. KumoAI Predictions
```bash
export KUMO_API_KEY='your-key'
python3 example_networkx_kumo.py --example 3
```

Demonstrates:
- Predicting series popularity
- Getting recommendations
- Batch predictions for multiple series

### 4. Combined Workflow
```bash
python3 example_networkx_kumo.py --example 4
```

Shows the power of combining both:
1. NetworkX finds central economic indicators
2. KumoAI predicts their future importance
3. Visualization shows the results

### 5. Interactive CLI
```bash
python3 051_kumo_rfm_cli.py
# Select mode 4 (visualize)
```

Then:
```sql
viz> query SELECT * FROM series_metadata WHERE category LIKE '%Employment%' LIMIT 50
viz> plot graph
```

## What Each Visualization Shows

### Category Network
- **Nodes**: Economic categories (Employment, Prices, etc.)
- **Size**: Number of series in category
- **Color**: Average popularity
- **Edges**: Categories that share data frequencies

**Use for**: Understanding the structure of economic data

### Series Similarity Network
- **Nodes**: Individual economic series
- **Size**: Popularity
- **Color**: Category
- **Edges**: High similarity (semantic or metadata)

**Use for**: Finding related indicators for analysis

### Community Detection
- **Nodes**: Economic series
- **Color**: Detected community
- **Automatic**: Uses Louvain algorithm to find clusters

**Use for**: Discovering natural groupings in the data

### Centrality Analysis
Shows 3 types of importance:
- **Degree**: Most connected (popular indicators)
- **Betweenness**: Bridges between clusters (linking indicators)
- **Closeness**: Central to everything (core economic metrics)

**Use for**: Identifying key economic indicators

## Files Created

### New Scripts
- `08_network_visualizations.py` - Main visualization module
- `example_networkx_kumo.py` - 5 example workflows
- `NETWORKX_KUMO_GUIDE.md` - Complete usage guide

### Enhanced Scripts
- `051_kumo_rfm_cli.py` - Already has NetworkX imported (line 24)
- `05_kumo_rfm_integration.py` - KumoAI integration ready

### Outputs (in visualizations/)
When you run the scripts, they create:
- `category_network_*.png` - Category relationships
- `series_network_*.png` - Series similarity
- `community_detection_*.png` - Detected clusters
- `communities_*.csv` - Community assignments
- `centrality_analysis_*.png` - Centrality measures
- `centrality_data_*.csv` - Centrality scores
- `*.json` - Network data for web viz

## Integration Points

### With Existing Pipeline
The network visualizations integrate with your existing scripts:

```bash
# Existing pipeline
python3 01_fetch_fred_data.py
python3 02_parse_fred_txt.py
python3 03_load_to_postgres.py
python3 04_vector_search.py --create

# NEW: Network analysis
python3 08_network_visualizations.py --embeddings data/embeddings/embeddings.npy
```

### With KumoAI
```python
# Use NetworkX to find important nodes
from module_08_network_visualizations import FREDNetworkVisualizer
viz = FREDNetworkVisualizer()
G = viz.build_series_network(df, max_nodes=100)

import networkx as nx
central_nodes = nx.degree_centrality(G)

# Use KumoAI to predict their future
from module_05_kumo_rfm_integration import KumoFREDIntegration
kumo = KumoFREDIntegration()
predictions = kumo.batch_predict_popularity(list(central_nodes.keys())[:10])
```

### With PostgreSQL
The CLI tool (`051_kumo_rfm_cli.py`) combines all three:
1. Query PostgreSQL for data
2. Visualize with NetworkX
3. Predict with KumoAI

## Real-World Use Cases

### 1. Economic Research
"Which employment indicators are most central to the economy?"
```bash
python3 08_network_visualizations.py --viz centrality --max-nodes 200
# Check centrality_data_*.csv for top indicators
```

### 2. Data Discovery
"Find all indicators related to inflation"
```bash
python3 08_network_visualizations.py --viz community --max-nodes 150
# Check communities_*.csv and filter by your series of interest
```

### 3. Dashboard Creation
"Create an interactive web dashboard"
```bash
python3 example_networkx_kumo.py --example 5
# Opens network_viewer.html with D3.js template
```

### 4. Automated Analysis
"Continuously monitor indicator relationships"
```python
# In your cron job or monitoring script
from module_08_network_visualizations import FREDNetworkVisualizer
viz = FREDNetworkVisualizer()
df = viz.load_data('data/fred_series_metadata.parquet')
G = viz.build_series_network(df)

# Alert if network structure changes significantly
density = nx.density(G)
if density < threshold:
    alert("Economic indicator relationships weakening!")
```

## Next Steps

### Try It Out
```bash
# Quick test (basic visualization)
python3 example_networkx_kumo.py --example 1

# Full test (with embeddings, requires prior setup)
python3 example_networkx_kumo.py --example 2

# All examples
python3 example_networkx_kumo.py
```

### Customize
Edit `08_network_visualizations.py` to:
- Change color schemes
- Adjust layout algorithms
- Add custom network metrics
- Create new visualization types

### Extend
Build on the foundation:
- Add time-series network analysis
- Create animated network evolution
- Integrate with your own ML models
- Build a Flask/FastAPI web app

## Performance Notes

- **Small datasets** (<100 nodes): All visualizations work well
- **Medium datasets** (100-500 nodes): Use higher similarity thresholds
- **Large datasets** (>500 nodes): Export to Gephi or use web visualization

For best performance with large networks:
1. Pre-compute and cache graphs
2. Use embeddings for better similarity
3. Export to JSON and use D3.js for interactive viz
4. Consider using Neo4j for very large graphs

## Dependencies

Your project already has:
- NetworkX (imported in CLI)
- Pandas, NumPy
- Matplotlib, Seaborn
- PostgreSQL connectivity

Only need to add:
- KumoAI (optional): `pip install kumoai`
- Scikit-learn (for similarity): `pip install scikit-learn`

## Documentation

- **Complete guide**: `NETWORKX_KUMO_GUIDE.md`
- **This summary**: `VISUALIZATION_SUMMARY.md`
- **Main README**: `README.md` (already exists)
- **Code examples**: `example_networkx_kumo.py`

## Questions?

The guide has answers for:
- How to export for Gephi?
- How to create web visualizations?
- What's the difference between centrality measures?
- How to combine with KumoAI predictions?
- How to handle large networks?

See `NETWORKX_KUMO_GUIDE.md` for details.
