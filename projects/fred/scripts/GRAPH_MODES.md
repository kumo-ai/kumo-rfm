# Graph Visualization Modes

The system provides **4 different ways** to visualize the series relationship graph, each optimized for different analysis needs.

---

## The Four Modes

### 1. **Full Mode** (Default)
```bash
python 05_kumo_rfm_integration.py --visualize-graph --graph-mode full
```

**What it shows:**
- Complete relationship graph
- Top 150 most connected nodes
- Color-coded by degree centrality (# of connections)
- **Best for:** Understanding the overall network structure

**File size:** ~6.5MB  
**Use when:** You want to see the big picture of how all series relate

**Visual style:**
- Yellow/Orange/Red gradient = degree centrality
- Larger nodes = more connections
- All nodes treated equally

---

### 2. **Highlight Mode** (Recommended with `--recommend`)
```bash
python 05_kumo_rfm_integration.py --recommend "housing" --visualize-graph --graph-mode highlight
```

**What it shows:**
- Full graph (top 150 nodes)
- **Recommended series highlighted in RED**
- Other series in blue
- Recommended series are larger and labeled

**Best for:** Seeing WHERE recommendations fit in the overall network

**File size:** ~6.5MB  
**Use when:** You want to understand the context of recommendations

**Visual style:**
- Red nodes = Recommended series (larger, labeled)
- Blue nodes = Other series
- Black borders on recommended nodes

**Example insight:** "These 10 housing series are all clustered in the Bank Reserves category"

---

### 3. **Subgraph Mode** (Neighborhood View)
```bash
python 05_kumo_rfm_integration.py --recommend "housing" --visualize-graph --graph-mode subgraph
```

**What it shows:**
- ONLY the recommended series + their 2-hop neighborhood
- Focused view on "what's related to recommendations"
- All neighbors up to 2 connections away

**Best for:** Deep dive into relationships around recommended series

**File size:** ~7.3MB (more nodes but focused)  
**Use when:** Exploring "what else is related to my recommendations?"

**Visual style:**
- Red nodes = Recommended series
- Blue nodes = Neighboring series (1-2 hops away)
- Shows the local network structure

**Example insight:** "Housing series connect to mortgage data, which connects to consumer credit"

**Size comparison:**
- Full graph: 428 total nodes → filtered to 150
- Subgraph: 220 nodes (all relevant to recommendations)

---

### 4. **Paths Mode** (Connection Analysis)
```bash
python 05_kumo_rfm_integration.py --recommend "housing" --visualize-graph --graph-mode paths
```

**What it shows:**
- ONLY nodes that connect recommended series to each other
- The "bridges" between recommendations
- Shortest paths between all pairs

**Best for:** Understanding HOW recommendations are related

**File size:** ~805KB (smallest - very focused)  
**Use when:** Answering "what connects these recommendations?"

**Visual style:**
- Red nodes = Recommended series (endpoints)
- Blue nodes = Bridge nodes (connecting paths)
- **Thicker edges** = part of connecting paths
- Minimal, focused view

**Example insight:** "All housing recommendations connect through the 'Bank Reserves' category hub"

**Size comparison:**
- Only 12 nodes, 6 edges (from 10 recommendations)
- Shows the essential skeleton

---

## Visual Comparison

| Mode | Nodes Shown | Focus | Red Nodes | File Size | Best For |
|------|-------------|-------|-----------|-----------|----------|
| **Full** | 150 | Overview | None | 6.5MB | Network structure |
| **Highlight** | 150 | Context | 10 recs | 6.5MB | **Where recs fit** |
| **Subgraph** | 220 | Neighborhood | 10 recs | 7.3MB | **Related series** |
| **Paths** | 12 | Connections | 10 recs | 805KB | **How recs connect** |

---

## Usage Examples

### Generate All 4 Modes at Once
```bash
python 05_kumo_rfm_integration.py --recommend "housing interest rates" --visualize-graph --graph-mode all
```

**Output:** 4 PNG files
- `relationship_graph_full_TIMESTAMP.png`
- `relationship_graph_highlight_TIMESTAMP.png`
- `relationship_graph_subgraph_TIMESTAMP.png`
- `relationship_graph_paths_TIMESTAMP.png`

### Compare Different Queries
```bash
# Housing query
python 05_kumo_rfm_integration.py --recommend "housing" --graph-mode subgraph --visualize-graph

# Employment query
python 05_kumo_rfm_integration.py --recommend "employment" --graph-mode subgraph --visualize-graph
```

Compare the two subgraphs to see how different topics connect differently!

---

## When to Use Each Mode

### Use **Full** when:
- Initial exploration of the dataset
- Understanding overall network topology
- Identifying major hubs/clusters
- No specific recommendations yet

### Use **Highlight** when:
- You have recommendations and want context
- Checking if recommendations are clustered or scattered
- Understanding recommendation diversity
- **Most versatile mode for presentations**

### Use **Subgraph** when:
- Exploring "related series" deeply
- Finding series similar to recommendations
- Understanding local network structure
- Building domain knowledge around a topic

### Use **Paths** when:
- Understanding relationships BETWEEN recommendations
- Finding common themes connecting results
- Identifying key "hub" series
- Minimal, focused analysis

---

## Real-World Scenarios

### Scenario 1: "What should I explore after GDP?"
```bash
python 05_kumo_rfm_integration.py --recommend "GDP" --graph-mode subgraph --visualize-graph
```
**Result:** See all series related to GDP within 2 connections

### Scenario 2: "Are housing and employment related?"
```bash
python 05_kumo_rfm_integration.py --recommend "housing employment" --graph-mode paths --visualize-graph
```
**Result:** See the shortest paths connecting housing and employment series

### Scenario 3: "Show me the full picture with highlights"
```bash
python 05_kumo_rfm_integration.py --recommend "inflation" --graph-mode highlight --visualize-graph
```
**Result:** Full network with inflation-related series in red

### Scenario 4: "Generate everything for my presentation"
```bash
python 05_kumo_rfm_integration.py --recommend "mortgage rates" --graph-mode all --visualize-graph
```
**Result:** 4 complementary views of mortgage-related series

---

## Output Organization

All visualizations go to:
```
outputs/visualizations/YYYY-MM-DD/
├── relationship_graph_full_TIMESTAMP.png
├── relationship_graph_highlight_TIMESTAMP.png
├── relationship_graph_subgraph_TIMESTAMP.png
└── relationship_graph_paths_TIMESTAMP.png
```

**Naming convention:** `relationship_graph_{MODE}_{YYYYMMDD_HHMMSS}.png`

---

## Technical Details

### Node Colors
- **Full mode:** Yellow→Orange→Red gradient by degree centrality
- **Other modes:** Red = recommended, Blue = others

### Node Sizes
- **Full mode:** Proportional to degree (# connections)
- **Other modes:** Recommended nodes 2x larger

### Edge Styling
- **Full/Highlight/Subgraph:** Thin, transparent edges
- **Paths:** Thicker edges (part of connecting paths)

### Labels
- **Full mode:** High-degree nodes (≥5 connections)
- **Other modes:** All recommended series + high-degree neighbors

### Max Nodes
- Default: 150 nodes (configurable)
- **Subgraph:** Shows all relevant nodes (may exceed 150)
- **Paths:** Minimal (only connecting nodes)

---

## Interpretation Guide

### Full Mode - Look for:
- **Dense clusters** = related series grouped together
- **Hub nodes** = series connecting many others
- **Isolated nodes** = unique or niche series

### Highlight Mode - Look for:
- **Clustering** = recommendations in same area (similar types)
- **Spread** = diverse recommendations across network
- **Position** = are recommendations central or peripheral?

### Subgraph Mode - Look for:
- **Neighborhood size** = how many series relate?
- **Intermediate nodes** = "bridge" series worth exploring
- **Cluster patterns** = sub-communities around recommendations

### Paths Mode - Look for:
- **Hub nodes** = series connecting multiple recommendations
- **Path length** = how closely related are recommendations?
- **Common nodes** = themes shared across recommendations

---

## Disk Space

Generating all modes:
- **Full:** ~6.5MB
- **Highlight:** ~6.5MB
- **Subgraph:** ~7.3MB
- **Paths:** ~800KB
- **Total:** ~21MB per run

**Tip:** Use `--graph-mode paths` for quick checks (smallest file)

---

## Quick Reference

```bash
# Single mode
--graph-mode full       # Overview
--graph-mode highlight  # Recommended (best with --recommend)
--graph-mode subgraph   # Neighborhood exploration
--graph-mode paths      # Connection analysis

# All modes at once
--graph-mode all        # Generate all 4 visualizations
```

**Remember:** Modes are most useful WITH recommendations!
```bash
python 05_kumo_rfm_integration.py --recommend "YOUR_QUERY" --visualize-graph --graph-mode all
```
