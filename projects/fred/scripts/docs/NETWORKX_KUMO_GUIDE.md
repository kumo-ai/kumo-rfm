# NetworkX + KumoAI Visualization Guide

Complete guide for visualizing FRED economic series relationships using NetworkX and KumoAI.

## Quick Start

```bash
# 1. Create basic network visualizations
python3 08_network_visualizations.py --data data/fred_series_metadata.parquet

# 2. With embeddings for semantic similarity
python3 08_network_visualizations.py \
  --data data/fred_series_metadata.parquet \
  --embeddings data/embeddings/embeddings.npy \
  --max-nodes 100 \
  --similarity 0.75

# 3. Run examples
python3 example_networkx_kumo.py --example 1  # Basic NetworkX
python3 example_networkx_kumo.py --example 2  # With embeddings
python3 example_networkx_kumo.py --example 3  # KumoAI predictions
python3 example_networkx_kumo.py --example 4  # Combined workflow
python3 example_networkx_kumo.py --example 5  # Export for web
```

## Features

### NetworkX Visualizations

#### 1. Category Network
Shows relationships between economic categories based on shared characteristics.

```python
from module_08_network_visualizations import FREDNetworkVisualizer

viz = FREDNetworkVisualizer()
df = viz.load_data('data/fred_series_metadata.parquet')

# Create category network
viz.visualize_category_network(df, layout='spring', save=True)
```

**What it shows:**
- Nodes = Economic categories
- Node size = Number of series in category
- Node color = Average popularity
- Edge width = Shared frequencies

#### 2. Series Similarity Network
Shows which economic indicators are similar to each other.

```python
# Without embeddings (uses category/frequency)
viz.visualize_series_network(
    df, 
    embeddings=None,
    similarity_threshold=0.6,
    max_nodes=50
)

# With embeddings (semantic similarity)
import numpy as np
embeddings = np.load('data/embeddings/embeddings.npy')

viz.visualize_series_network(
    df,
    embeddings=embeddings,
    similarity_threshold=0.75,
    max_nodes=100
)
```

**What it shows:**
- Nodes = Economic series
- Node size = Popularity
- Node color = Category
- Edges = High similarity (above threshold)

#### 3. Community Detection
Automatically discovers clusters of related economic indicators.

```python
viz.visualize_community_detection(
    df,
    embeddings=embeddings,
    max_nodes=100,
    save=True
)
```

**What it shows:**
- Automatically detected communities using Louvain algorithm
- Node color = Community membership
- Saves community assignments to CSV

#### 4. Centrality Analysis
Identifies the most important/influential economic indicators.

```python
viz.visualize_centrality_analysis(
    df,
    embeddings=embeddings,
    max_nodes=100,
    save=True
)
```

**Shows 3 centrality measures:**
- **Degree**: How many connections (popular indicators)
- **Betweenness**: Bridge between clusters (linking indicators)
- **Closeness**: Close to all nodes (central to the economy)

### KumoAI Integration

#### Setup

```bash
export KUMO_API_KEY='your-api-key-here'
pip install kumoai
```

#### Predictive Queries

```python
from module_05_kumo_rfm_integration import KumoFREDIntegration

kumo = KumoFREDIntegration()
df = kumo.load_data('data/fred_series_metadata.parquet')
kumo.build_graph(df)

# Predict series popularity
result = kumo.predict_series_popularity('PAYEMS')

# Batch predictions
predictions = kumo.batch_predict_popularity(['PAYEMS', 'UNRATE', 'CPIAUCSL'])

# Get recommendations
recommendations = kumo.recommend_series_for_analysis(df, 'inflation', top_k=10)
```

### Combined NetworkX + KumoAI

#### Workflow Example

```python
import networkx as nx
from module_08_network_visualizations import FREDNetworkVisualizer
from module_05_kumo_rfm_integration import KumoFREDIntegration

# 1. Build network with NetworkX
viz = FREDNetworkVisualizer()
df = viz.load_data('data/fred_series_metadata.parquet')
G = viz.build_series_network(df, max_nodes=100)

# 2. Find central nodes
degree_centrality = nx.degree_centrality(G)
top_central = sorted(degree_centrality.items(), 
                     key=lambda x: x[1], 
                     reverse=True)[:10]

# 3. Use KumoAI to analyze central nodes
kumo = KumoFREDIntegration()
kumo.build_graph(df)

top_series_ids = [sid for sid, _ in top_central[:5]]
predictions = kumo.batch_predict_popularity(top_series_ids)

# 4. Visualize results
viz.visualize_centrality_analysis(df, save=True)
```

## CLI Integration

The NetworkX visualizations are also integrated into the CLI tool:

```bash
python3 051_kumo_rfm_cli.py
```

Then select mode `4. visualize` and use:

```sql
viz> query SELECT * FROM series_metadata LIMIT 50
viz> plot graph
```

The CLI will automatically create a network graph from your query results if the data contains relationships (e.g., series_id, category, or edge data).

## Export Options

### For Web Visualization (D3.js, Cytoscape.js)

```python
viz = FREDNetworkVisualizer()
df = viz.load_data('data/fred_series_metadata.parquet')

# Build network
G = viz.build_category_network(df)

# Export to JSON
viz.export_graph_to_json('visualizations/network.json')
```

Use the exported JSON with:
- **D3.js** - Force-directed graphs
- **Cytoscape.js** - Interactive network analysis
- **Vis.js** - Quick network visualization
- **Gephi** - Advanced network analysis (import via networkx)

### For Gephi

```python
import networkx as nx

G = viz.build_series_network(df)
nx.write_gexf(G, 'visualizations/network.gexf')
```

Then open in Gephi for advanced analysis.

## Layout Algorithms

Available layouts for visualization:

```python
# Spring/Force-directed (default)
viz.visualize_category_network(df, layout='spring')

# Circular
viz.visualize_category_network(df, layout='circular')

# Kamada-Kawai (force-directed with optimal node distance)
viz.visualize_category_network(df, layout='kamada_kawai')
```

**Best practices:**
- `spring`: Good for general purpose, shows clusters
- `circular`: Good for seeing all nodes clearly
- `kamada_kawai`: Good for aesthetically pleasing layouts

## Advanced NetworkX Analysis

### Custom Network Analysis

```python
import networkx as nx

# Build graph
G = viz.build_series_network(df, max_nodes=100)

# Calculate various metrics
degree = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
pagerank = nx.pagerank(G)

# Find communities
from networkx.algorithms import community
communities = community.greedy_modularity_communities(G)

# Find shortest paths
path = nx.shortest_path(G, source='PAYEMS', target='UNRATE')

# Calculate clustering coefficient
clustering = nx.clustering(G)

# Find bridges (critical connections)
bridges = list(nx.bridges(G))
```

### Network Statistics

```python
# Basic stats
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.4f}")
print(f"Average clustering: {nx.average_clustering(G):.4f}")

# Connected components
components = list(nx.connected_components(G))
print(f"Connected components: {len(components)}")

# Diameter (if connected)
if nx.is_connected(G):
    diameter = nx.diameter(G)
    print(f"Diameter: {diameter}")
```

## Use Cases

### 1. Economic Policy Analysis
Find which indicators are most central to the economy:
```bash
python3 08_network_visualizations.py --viz centrality --max-nodes 200
```

### 2. Data Discovery
Find related series for a research topic:
```python
# Find communities of related indicators
viz.visualize_community_detection(df, max_nodes=150)

# Check the saved communities CSV for your topic
communities_df = pd.read_csv('visualizations/communities_*.csv')
```

### 3. Anomaly Detection
Find series that don't fit with others (low centrality):
```python
G = viz.build_series_network(df)
degree_cent = nx.degree_centrality(G)

# Find isolated series
isolated = [node for node, cent in degree_cent.items() if cent < 0.01]
```

### 4. Trend Analysis
Combine with KumoAI to predict future relationships:
```python
# Current network structure
G = viz.build_series_network(df)
central_nodes = [n for n, _ in 
                sorted(nx.degree_centrality(G).items(), 
                       key=lambda x: x[1], reverse=True)[:10]]

# Predict which will remain central
kumo = KumoFREDIntegration()
kumo.build_graph(df)
predictions = kumo.batch_predict_popularity(central_nodes)
```

## Troubleshooting

### Graph too sparse
```python
# Lower similarity threshold
viz.visualize_series_network(df, similarity_threshold=0.5)

# Increase max nodes
viz.visualize_series_network(df, max_nodes=200)
```

### Out of memory
```python
# Reduce max_nodes
viz.visualize_series_network(df, max_nodes=50)

# Use sampling
df_sample = df.sample(n=1000, random_state=42)
viz.visualize_series_network(df_sample)
```

### Cluttered visualization
```python
# Increase similarity threshold
viz.visualize_series_network(df, similarity_threshold=0.8)

# Show fewer nodes
viz.visualize_series_network(df, max_nodes=30)

# Use different layout
viz.visualize_series_network(df, layout='circular')
```

## Performance Tips

1. **Use embeddings** for semantic similarity (more accurate than category/frequency)
2. **Limit nodes** for interactive exploration (50-100 nodes)
3. **Pre-compute** and cache network graphs for repeated analysis
4. **Export to web** for large networks (better performance than matplotlib)

## Example Workflows

### Discovery Workflow
```bash
# 1. Create embeddings
python3 04_vector_search.py --create

# 2. Find communities
python3 08_network_visualizations.py --viz community --max-nodes 150

# 3. Analyze specific community
# (Use community CSV to filter series)
```

### Analysis Workflow
```bash
# 1. Build full network
python3 08_network_visualizations.py --viz all --max-nodes 200

# 2. Export centrality data
# (Automatically saved as CSV)

# 3. Use KumoAI for predictions
python3 05_kumo_rfm_integration.py --demo
```

### Integration Workflow
```bash
# 1. Run combined example
python3 example_networkx_kumo.py --example 4

# 2. Export for web
python3 example_networkx_kumo.py --example 5

# 3. Open in browser
# visualizations/network_viewer.html
```

## Further Resources

- **NetworkX Documentation**: https://networkx.org/documentation/stable/
- **KumoAI Documentation**: https://docs.kumo.ai/
- **D3.js Network Examples**: https://d3-graph-gallery.com/network.html
- **Gephi**: https://gephi.org/
