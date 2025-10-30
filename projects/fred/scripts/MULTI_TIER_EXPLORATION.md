# Multi-Tier Exploration System

Automatically discover related series through iterative, intelligent follow-up recommendations.

---

## What It Does

Instead of just getting recommendations for "housing", the system:

1. **Tier 1:** Searches for "housing" → Gets 10 results
2. **Analyzes results** → Identifies themes (inflation, prices, permits)
3. **Tier 2:** Searches for "housing inflation" → Gets 10 more results
4. **Tier 3:** Searches for "housing prices" → Gets 10 more results

**Result:** 30 diverse results across 3 related queries instead of 10 from one query!

---

## Basic Usage

### Auto-Generated Follow-Ups (Recommended)
```bash
python 05_kumo_rfm_integration.py --recommend "housing" --explore-tiers
```

**What happens:**
1. Tier 1: Returns "housing" recommendations
2. System analyzes results and generates intelligent follow-up queries
3. Tier 2 & 3: Automatically explore those follow-ups

### Custom Follow-Ups
```bash
python 05_kumo_rfm_integration.py --recommend "housing" --explore-tiers \
  --tier-queries "mortgage rates" "home sales" "construction"
```

**What happens:**
1. Tier 1: "housing"
2. Tier 2: "mortgage rates"
3. Tier 3: "home sales"
4. (Tier 4: "construction" if --max-tiers 4)

---

## Real Example

### Command:
```bash
python 05_kumo_rfm_integration.py --recommend "housing" --explore-tiers --max-tiers 3 --top-k 5
```

### Output:

```
================================================================================
TIER 1: housing
================================================================================

Top 5 recommendations:
--------------------------------------------------------------------------------

PCU49314931
  Producer Price Index by Industry: Warehousing and Storage
  Category: Inflation | Frequency: Monthly
  Score: 7.96 | Popularity: 29

FLBP1FHSA
  New Private Housing Units Authorized by Building Permits: 1-Unit
  Category: Bank Reserves | Frequency: Monthly
  Score: 6.53 | Popularity: 12

... 3 more

Analyzing results to generate follow-up queries...

Follow-up exploration paths:
   Tier 2: housing inflation
   Tier 3: housing prices

================================================================================
TIER 2: housing inflation
================================================================================

Top 5 recommendations:
--------------------------------------------------------------------------------

T10YIE
  10-Year Breakeven Inflation Rate
  Category: Interest Rates | Frequency: Daily
  Score: 4.79 | Popularity: 89

T5YIFR
  5-Year, 5-Year Forward Inflation Expectation Rate
  Category: Interest Rates | Frequency: Daily
  Score: 4.72 | Popularity: 81

... 3 more

================================================================================
TIER 3: housing prices
================================================================================

Top 5 recommendations:
--------------------------------------------------------------------------------

MCOILBRENTEU
  Crude Oil Prices: Brent - Europe
  Category: Commodities | Frequency: Monthly
  Score: 4.13 | Popularity: 50

... 4 more

================================================================================
EXPLORATION SUMMARY
================================================================================
Total tiers explored: 3
Total unique series found: 15
```

---

## How Follow-Ups Are Generated

The system uses **3 intelligent strategies**:

### Strategy 1: Category Expansion
If results span multiple categories, drill into each:
- **Tier 1 has:** Bank Reserves (7), Inflation (2), PCE (1)
- **Follow-up:** "housing inflation" (explore the minority category)

### Strategy 2: Title Keyword Extraction
Extract meaningful themes from recommendation titles:
- Sees "price" in many titles → Suggests "housing prices"
- Sees "permit" frequently → Suggests "housing construction"
- Sees "mortgage" → Suggests "housing mortgage"

### Strategy 3: Domain Knowledge
Pre-defined related concepts for common queries:

| Query | Auto-Suggests |
|-------|---------------|
| housing | mortgage rates, home sales, construction |
| employment | unemployment, wages, labor force |
| inflation | prices, consumer price index, producer price |
| GDP | economic growth, productivity, output |
| interest | federal funds rate, treasury rates, yields |

---

## Options

### `--explore-tiers`
Enable multi-tier exploration mode.
```bash
--explore-tiers
```

### `--max-tiers N`
Maximum exploration depth (default: 3).
```bash
--max-tiers 5    # Go 5 levels deep
```

### `--tier-queries Q1 Q2 Q3`
Custom follow-up queries (overrides auto-generation).
```bash
--tier-queries "mortgage rates" "home sales" "construction employment"
```

### `--top-k N`
Results per tier (default: 10).
```bash
--top-k 5    # Get 5 results per tier
```

---

## Use Cases

### Use Case 1: Comprehensive Topic Coverage
**Goal:** Get all aspects of housing data

```bash
python 05_kumo_rfm_integration.py --recommend "housing" --explore-tiers --max-tiers 4
```

**Result:** Covers permits, prices, sales, construction, mortgages

---

### Use Case 2: Planned Deep Dive
**Goal:** Specific exploration path you already know

```bash
python 05_kumo_rfm_integration.py --recommend "GDP" --explore-tiers \
  --tier-queries "GDP growth rate" "GDP per capita" "GDP by sector"
```

**Result:** Systematic exploration of GDP variations

---

### Use Case 3: Discovery Mode
**Goal:** Let the system surprise you with connections

```bash
python 05_kumo_rfm_integration.py --recommend "employment" --explore-tiers --max-tiers 5
```

**Result:** May discover connections like:
- Tier 1: Employment → job statistics
- Tier 2: Employment wages → income data
- Tier 3: Employment labor force → participation rates
- Tier 4: Employment unemployment → claims data
- Tier 5: Employment jobs → industry breakdown

---

### Use Case 4: Comparing Queries
**Goal:** See how different starting points lead to different explorations

```bash
# Starting from "housing"
python 05_kumo_rfm_integration.py --recommend "housing" --explore-tiers

# Starting from "real estate"
python 05_kumo_rfm_integration.py --recommend "real estate" --explore-tiers
```

**Compare:** Do they converge on similar series or diverge?

---

## Combining with Visualizations

### Visualize Each Tier
```bash
python 05_kumo_rfm_integration.py --recommend "housing" --explore-tiers \
  --visualize-graph --graph-mode all
```

**Result:**
- Recommendations across 3 tiers
- 4 graph visualizations showing Tier 1 results
- See how first tier fits in network

### Visualize Final Tier Only
```bash
python 05_kumo_rfm_integration.py --recommend "housing" --explore-tiers \
  --max-tiers 3 --visualize-graph --graph-mode highlight
```

**Shows:** Where the deepest exploration tier sits in the network

---

## Output Structure

### Console Output
- **Tier headers** with query names
- **Top 5 results per tier** (compact view)
- **Follow-up paths** announced before execution
- **Summary** with total unique series found

### Result Storage
All tier results are stored in the returned dictionary:
```python
{
    "Tier 1: housing": DataFrame(10 rows),
    "Tier 2: housing inflation": DataFrame(10 rows),
    "Tier 3: housing prices": DataFrame(10 rows)
}
```

---

## Tips & Best Practices

### 1. Start Broad, Let System Narrow
```bash
# Good: Broad starting query
--recommend "housing"

# Less optimal: Too specific
--recommend "30-year fixed mortgage rates in California"
```

### 2. Use Custom Queries for Known Paths
```bash
# When you know what you want
--tier-queries "mortgage rates" "home prices" "construction"
```

### 3. Adjust Depth Based on Specificity
```bash
# Broad topics → more tiers
--recommend "economy" --max-tiers 5

# Specific topics → fewer tiers
--recommend "30-year mortgage" --max-tiers 2
```

### 4. Combine with Graph Modes
```bash
# See how tiers relate
--explore-tiers --visualize-graph --graph-mode subgraph
```

---

## Technical Details

### Follow-Up Generation Algorithm

1. **Extract categories** from Tier 1 results
   - Count: How many in each category?
   - Select: Minority categories (interesting alternatives)

2. **Extract keywords** from titles
   - Filter: Words > 3 chars, not common words
   - Identify themes: mortgage, construction, sales, etc.

3. **Apply domain knowledge**
   - Match: Does query match known concepts?
   - Suggest: Pre-defined related terms

4. **Combine & deduplicate**
   - Ensure: No duplicate follow-ups
   - Limit: To max_tiers - 1

### Performance

- **Time:** ~2-3 seconds per tier (Kumo API call)
- **API calls:** 1 per tier (max_tiers total)
- **Results:** top_k × max_tiers total series

**Example:** 3 tiers × 10 results = 30 series total

---

## Comparison: Single vs Multi-Tier

### Single Query
```bash
python 05_kumo_rfm_integration.py --recommend "housing" --top-k 10
```
**Result:** 10 series, one perspective

### Multi-Tier
```bash
python 05_kumo_rfm_integration.py --recommend "housing" --explore-tiers
```
**Result:** 30 series (3 tiers × 10), three perspectives

**Benefit:** 3x more results, covering different aspects

---

## Advanced Examples

### Example 1: Exhaustive Exploration
```bash
python 05_kumo_rfm_integration.py \
  --recommend "inflation" \
  --explore-tiers \
  --max-tiers 5 \
  --top-k 20 \
  --visualize-graph \
  --graph-mode all \
  --save-to-db
```

**Does:**
- 5 tiers of inflation exploration
- 20 results per tier = 100 total series
- All 4 graph visualizations
- Saves to database

### Example 2: Guided Exploration with Visualization
```bash
python 05_kumo_rfm_integration.py \
  --recommend "employment" \
  --explore-tiers \
  --tier-queries "unemployment" "labor force participation" \
  --visualize-graph \
  --graph-mode paths
```

**Shows:**
- How employment connects to unemployment
- How unemployment connects to participation
- Path visualization showing connections

### Example 3: Fast Discovery
```bash
python 05_kumo_rfm_integration.py \
  --recommend "housing" \
  --explore-tiers \
  --max-tiers 2 \
  --top-k 5
```

**Quick:** 2 tiers, 5 each = 10 series in ~5 seconds

---

## Quick Reference

```bash
# Basic auto-exploration
--recommend "QUERY" --explore-tiers

# Custom path
--recommend "QUERY" --explore-tiers --tier-queries "Q1" "Q2"

# Deep dive
--recommend "QUERY" --explore-tiers --max-tiers 5

# With visualization
--recommend "QUERY" --explore-tiers --visualize-graph --graph-mode all

# Fast mode
--recommend "QUERY" --explore-tiers --max-tiers 2 --top-k 5
```

---

## When to Use Multi-Tier

### Use Multi-Tier When:
- You want comprehensive topic coverage
- You're exploring an unfamiliar domain
- You need to discover related concepts
- You want diverse perspectives on a topic

### Use Single Query When:
- You know exactly what you want
- You need quick results
- You're testing/debugging
- The topic is very specific already

---

## Output Files

With visualization enabled:
```
outputs/visualizations/YYYY-MM-DD/
├── relationship_graph_full_TIMESTAMP.png
├── relationship_graph_highlight_TIMESTAMP.png  ← Tier 1 results highlighted
├── relationship_graph_subgraph_TIMESTAMP.png
└── relationship_graph_paths_TIMESTAMP.png
```

Note: Only Tier 1 results are visualized (for clarity)

---

## Summary

**Multi-tier exploration transforms:**
- "Show me housing series" (narrow, 10 results)
- "Explore housing comprehensively" (broad, 30+ results)

**By automatically:**
- Generating intelligent follow-up queries
- Executing multiple searches
- Summarizing total unique findings

**Result:** Faster, more comprehensive research!
