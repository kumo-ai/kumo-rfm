# Project Vision: FRED Series Recommendation with Kumo RFM

## Core Purpose

**Use Kumo RFM to recommend relevant FRED economic series based on different types of user inputs.**

This project is NOT about:
- Building a general FRED data processing pipeline
- Vector search or embeddings as primary features
- Database management
- Monolith recommendation systems

This project IS about:
- **Kumo RFM-powered recommendations** for economic series
- Accepting diverse input types (text queries, series IDs, categories, etc.)
- Helping users discover relevant economic indicators

## User Stories

### 1. Text Query → Series Recommendations
**As a researcher**, I want to search "inflation impact on housing" and get recommended FRED series that capture these relationships.

### 2. Series → Related Series
**As an economist**, I want to input "PAYEMS" (employment) and discover related economic indicators I should also analyze.

### 3. Category → Top Series
**As an analyst**, I want to explore "Employment" category and get Kumo's recommendations for the most relevant series to track.

### 4. Multi-series → Additional Context
**As a data scientist**, I'm analyzing GDP and unemployment, and I want Kumo to recommend what other series would add valuable context.

### 5. Metadata-based Discovery
**As a policy maker**, I want to find highly popular, frequently updated series in a specific domain.

## How Kumo RFM Fits

Kumo RFM enables:
1. **No feature engineering**: Just define relationships, Kumo handles the ML
2. **Graph-based reasoning**: Understands relationships between series, categories, frequencies
3. **Predictive queries**: Can predict popularity, category, or recommend similar items
4. **Flexible inputs**: Accept various query types and return relevant results

## Core Architecture

```
User Input → Input Processor → Kumo RFM Graph → Recommendations → User
            (text/series/     (query builder)   (series IDs +
             category/etc)                       relevance scores)
```

### Essential Components

1. **FRED Data Fetcher** (`01_fetch_fred_data.py`)
   - Get series metadata from FRED API
   - Store in parquet format

2. **Data Parser** (`02_parse_fred_txt.py`)
   - Convert FRED text to structured format
   - Extract metadata (category, frequency, popularity)

3. **Kumo RFM Integration** (`05_kumo_rfm_integration.py`)  **CORE**
   - Build Kumo graph from FRED metadata
   - Define series, category, frequency relationships
   - Implement recommendation queries
   - Handle different input types

4. **Recommendation API/CLI** (`051_kumo_rfm_cli.py` or new)
   - User-facing interface
   - Accept various input formats
   - Return ranked recommendations
   - Explain why series were recommended

### Supporting Components (Optional)

- **PostgreSQL** (`03_load_to_postgres.py`)
  - Store metadata for complex queries
  - Not required if Kumo RFM can handle all queries

- **Vector Search** (`04_vector_search.py`)
  - Semantic similarity for text queries
  - Could enhance Kumo recommendations
  - Optional fallback if Kumo API unavailable

- **Monolith Features** (`06_prepare_monolith_features.py`)
  - REMOVE or ARCHIVE - not relevant to Kumo RFM focus

## Key Questions to Answer

1. **What should we recommend?**
   - FRED series IDs with relevance scores
   - Include: title, category, popularity, frequency
   - Why was it recommended? (reasoning)

2. **What inputs should we accept?**
   - Free text query: "housing market indicators"
   - Series ID: "PAYEMS"
   - Category name: "Employment"
   - Multiple series: ["GDPC1", "UNRATE"]
   - Metadata filters: {frequency: "Monthly", popularity: >50}

3. **How do we use Kumo RFM?**
   ```python
   # Build graph with series, categories, frequencies
   graph = build_kumo_graph(fred_metadata)
   
   # Recommendation queries
   - PREDICT series.popularity FOR series WHERE <conditions>
   - FIND similar series to series_id='PAYEMS'
   - RECOMMEND series FOR category='Employment' ORDER BY relevance
   ```

4. **What's the user experience?**
   - CLI: `kumo-fred recommend "inflation housing"`
   - Python API: `recommender.recommend("inflation housing", top_k=10)`
   - Web API: `POST /recommend {"query": "...", "top_k": 10}`

## Success Metrics

 User inputs query → Gets relevant FRED series recommendations  
 Recommendations are better than simple keyword matching  
 Can explain why series were recommended  
 Works with multiple input types  
 Fast response time (<2 seconds)  

## What to Refactor

### Keep & Enhance
-  `01_fetch_fred_data.py` - Essential data source
-  `02_parse_fred_txt.py` - Essential parsing
-  `05_kumo_rfm_integration.py` - **CORE** - needs major enhancement
-  `051_kumo_rfm_cli.py` - Refactor as recommendation interface

### Simplify or Remove
-  `03_load_to_postgres.py` - Only if needed for complex queries
-  `04_vector_search.py` - Optional enhancement, not core
-  `06_prepare_monolith_features.py` - REMOVE (wrong algorithm)
-  `99_pipeline.py` - Simplify for Kumo-focused workflow

### Archive (Already done)
-  Research assistant, foreign key checker, etc.

## Next Actions

1. **Refactor `05_kumo_rfm_integration.py`**
   - Focus on recommendation use cases
   - Build proper graph structure
   - Implement diverse query types
   - Add reasoning/explanation

2. **Create `recommend_series.py`** (new main script)
   - Unified interface for all recommendation types
   - CLI and Python API
   - Clean, user-friendly output

3. **Simplify `051_kumo_rfm_cli.py`**
   - Focus on recommendation interactions
   - Remove PostgreSQL-specific features
   - Add recommendation commands

4. **Update Documentation**
   - README focused on recommendations
   - Example queries and outputs
   - How to interpret results

5. **Evaluate Support Components**
   - PostgreSQL: Keep only if Kumo can't handle certain queries
   - Vector search: Maybe useful for text → series mapping
   - Monolith: REMOVE completely
