# Example Scripts

These scripts demonstrate advanced features and integrations. They are not required for the core pipeline.

## Files

### `07_workflow_integrations.py`
**Purpose:** Examples for integrating with external systems
- ElasticSearch - Full-text search
- Pinecone - Cloud vector database
- Weaviate - Semantic search
- Neo4j - Graph database

**Usage:**
```bash
python3 examples/07_workflow_integrations.py --demo
```

### `08_network_visualizations.py`
**Purpose:** NetworkX graph visualizations of FRED series relationships

**Features:**
- Category networks
- Series similarity networks
- Community detection
- Centrality analysis

**Usage:**
```bash
# Import in your scripts
import sys
sys.path.append('examples/')
from network_visualizations import FREDNetworkVisualizer

viz = FREDNetworkVisualizer()
viz.visualize_category_network(df)
```

### `10_advanced_kumo_demo.py`
**Purpose:** Advanced KumoAI RFM demonstrations

**Features:**
- Multi-table relationships
- Temporal predictions
- What-if analysis
- Feature importance

**Requirements:** KUMO_API_KEY

**Usage:**
```bash
export KUMO_API_KEY='your-key'
python3 examples/10_advanced_kumo_demo.py --demo
```

### `example_networkx_kumo.py`
**Purpose:** Combined NetworkX + KumoAI workflow example

**Features:**
- Network visualization without ML
- Vector embedding visualization
- KumoAI predictions
- Combined analysis workflow

**Usage:**
```bash
python3 examples/example_networkx_kumo.py --example 1  # Basic network viz
python3 examples/example_networkx_kumo.py --example 2  # With embeddings
python3 examples/example_networkx_kumo.py --example 3  # Kumo predictions
```

## Note on Imports

These example scripts import from the numbered core modules. If you encounter import errors, you can either:

1. **Run from project root:**
   ```bash
   cd /path/to/fred/map
   python3 examples/example_networkx_kumo.py
   ```

2. **Add to Python path:**
   ```python
   import sys
   sys.path.insert(0, '..')
   ```

## Dependencies

Some examples require additional packages:
```bash
pip install networkx matplotlib seaborn
pip install elasticsearch pinecone-client weaviate-client neo4j  # For workflow integrations
```
