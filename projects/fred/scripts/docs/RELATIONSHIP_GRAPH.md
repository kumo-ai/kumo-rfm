# Kumo RFM Relationship Graph

## Overview

The **Relationship Graph** (06_kumo_relationship_graph.py) creates a knowledge graph showing **connections between FRED datasets** based on:
- Domain knowledge (e.g., interest rates affect housing)
- Category relationships (series in same category)
- Economic theory (employment affects GDP, inflation affects rates)

## What It Does

Instead of just finding series by keyword, the relationship graph shows **WHY datasets are connected** and **HOW they relate to each other**.

### Example: Interest Rates and Housing

```bash
python 06_kumo_relationship_graph.py --connect "interest rates housing" --top-k 5
```

**Output shows connections like:**
```
FEDFUNDS -> MORTGAGE30US
  From: Federal Funds Effective Rate
        Category: Inflation, Popularity: 99
  To:   30-Year Fixed Rate Mortgage Average in the United States
        Category: Mortgage Rate, Popularity: 91
  Relationship: monetary_policy_impact (strength: 0.90)
  Why: Interest rates directly impact housing affordability
```

## How It Works

### 1. Domain Relationship Rules

The system encodes economic relationships:

```python
# Interest rates affect housing
{
    'source_categories': ['Mortgage Rate', 'Interest Rates', 'Bank Reserves'],
    'target_categories': ['Housing Starts', 'Building Permits', 'Home Sales'],
    'relationship_type': 'monetary_policy_impact',
    'strength': 0.9,
    'description': 'Interest rates directly impact housing affordability'
}
```

### 2. Built-in Relationships

**Monetary Policy Impact** (strength: 0.90)
- Interest Rates -> Housing Market
- Federal Funds Rate -> Mortgage Rates -> Housing Starts

**Economic Indicators** (strength: 0.85)
- Employment -> GDP
- Unemployment Rate -> Economic Output

**Monetary Policy Response** (strength: 0.95)
- Inflation -> Interest Rates
- CPI -> Federal Funds Rate

**Economic Drivers** (strength: 0.80)
- Consumer Spending -> GDP
- Retail Sales -> Economic Growth

**Industry Linkages** (strength: 0.75)
- Housing Construction -> Employment
- Building Permits -> Construction Jobs

### 3. Category Relationships (strength: 0.70)
- Automatically connects top series within each category
- Example: All housing series connected to each other

## Graph Structure

The system creates a **Kumo RFM LocalGraph** with:

**Two Tables:**
1. **series**: All FRED series (58,595 nodes)
2. **relationships**: Connections between series (14,456 edges)

**Edges:**
- `relationships.source_id` -> `series.series_id`
- `relationships.target_id` -> `series.series_id`

This creates a proper graph that Kumo RFM can traverse and learn from.

## Usage Examples

### Find Interest Rate -> Housing Connections
```bash
python 06_kumo_relationship_graph.py --connect "interest rates housing" --top-k 10
```

### Find Employment -> GDP Connections
```bash
python 06_kumo_relationship_graph.py --connect "employment GDP" --top-k 10
```

### Find Inflation -> Consumer Spending Connections
```bash
python 06_kumo_relationship_graph.py --connect "inflation consumer" --top-k 10
```

### Disable Domain Knowledge (only category-based)
```bash
python 06_kumo_relationship_graph.py --connect "interest housing" --top-k 10 --no-domain
```

### Disable Category Relationships (only domain knowledge)
```bash
python 06_kumo_relationship_graph.py --connect "interest housing" --top-k 10 --no-category
```

## Output Format

For each connection, you get:

1. **Source Series**: ID, title, category, popularity
2. **Target Series**: ID, title, category, popularity
3. **Relationship Type**: What kind of connection
4. **Strength**: How strong the relationship is (0-1)
5. **Description**: WHY they're connected

## Relationship Types

| Type | Description | Example |
|------|-------------|---------|
| `monetary_policy_impact` | Interest rates affect market | Fed Rate -> Housing |
| `economic_indicator` | Indicator signals economic state | Employment -> GDP |
| `monetary_policy_response` | Policy responds to conditions | Inflation -> Rate Changes |
| `economic_driver` | Direct economic cause | Consumer Spending -> GDP |
| `industry_linkage` | Industry connections | Housing -> Construction Jobs |
| `same_category` | Same FRED category | All housing series together |

## Adding New Relationships

To add your own domain knowledge, edit the `domain_rules` list in `create_domain_relationships()`:

```python
domain_rules.append({
    'source_categories': ['Trade', 'Exports'],
    'target_categories': ['Exchange Rate', 'Currency'],
    'relationship_type': 'trade_impact',
    'strength': 0.85,
    'description': 'Trade balance affects currency strength'
})
```

## Comparison: Before vs After

### Before (Simple Search)
```bash
python 05_kumo_rfm_integration.py --recommend 'housing' --top-k 10
```
Returns: Series matching "housing" ranked by popularity
- Good for discovery
- No context about WHY

### After (Relationship Graph)
```bash
python 06_kumo_relationship_graph.py --connect "interest housing" --top-k 10
```
Returns: Series connected through economic relationships
- Shows WHY datasets are related
- Explains HOW they influence each other
- Based on domain knowledge + ML

## Future Enhancements

1. **Time-series correlations**: Add relationships based on actual data correlations
2. **User co-viewing**: "Users who viewed X also viewed Y"
3. **Temporal lags**: "Series X leads series Y by 6 months"
4. **Strength learning**: Learn relationship strengths from data instead of hardcoding

## Integration with Existing Tools

The relationship graph complements existing tools:

1. **Search/Filter** (05_kumo_rfm_integration.py): Find individual series
2. **Relationship Graph** (06_kumo_relationship_graph.py): Understand connections
3. **Chat Assistant** (search_term_assistant.py): Describe your needs naturally
4. **Database CLI** (051_kumo_rfm_cli.py): Explore and visualize

## Technical Details

**Graph Statistics:**
- Nodes: 58,595 series
- Edges: 14,456 relationships
- Relationship types: 6
- Average connections per series: ~0.5

**Performance:**
- Graph build time: ~5 seconds
- Query time: <1 second
- Memory usage: ~50MB for graph

**Kumo RFM Benefits:**
- Proper foreign key relationships
- Can traverse multi-hop connections
- ML-powered relevance scoring
- Scalable to millions of nodes/edges

## Example Workflow

```bash
# 1. Find connected datasets
python 06_kumo_relationship_graph.py --connect "interest housing" --top-k 5

# 2. Pick series IDs from connections
# Example: FEDFUNDS and MORTGAGE30US

# 3. Use recommend for deeper analysis
python 05_kumo_rfm_integration.py --multi FEDFUNDS MORTGAGE30US --top-k 10

# 4. Or explore in database CLI
python 051_kumo_rfm_cli.py
sql> SELECT * FROM series WHERE series_id IN ('FEDFUNDS', 'MORTGAGE30US');
```

This gives you a complete workflow from **discovery** to **understanding connections** to **detailed analysis**.
