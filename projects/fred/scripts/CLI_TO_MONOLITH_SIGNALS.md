# CLI Signals for Monolith Feed

## Overview
The CLI captures real user behavior that can dramatically improve the Monolith recommendation system. Currently, the monolith uses **synthetic interactions** - we should replace these with real usage patterns.

---

## High-Value Signals to Capture

### 1. **Query Patterns** (User-Item Interactions)
Currently tracked in: `kumo_rfm_queries`

**What to feed Monolith:**
```python
{
    'user_id': 'user_123',  # Or session_id if anonymous
    'series_id': 'GDP',      # Extracted from query
    'query_text': 'SELECT * FROM economic_studies WHERE series_id = "GDP"',
    'timestamp': 1234567890,
    'context': {
        'time_of_day': 'morning',
        'session_length': 450,  # seconds
        'query_complexity': 'simple'  # based on joins, where clauses
    }
}
```

**How to extract series_id from queries:**
- Parse SQL: `WHERE series_id = 'GDP'`
- Parse SQL: `FROM economic_study_series WHERE series_id IN (...)`
- Track which series appear in results

---

### 2. **Co-occurrence Patterns** (Sequence Features)
Currently synthetic in monolith

**What to feed Monolith:**
```python
{
    'user_id': 'user_123',
    'series_sequence': ['GDP', 'UNRATE', 'CPIAUCSL'],  # Order of queries
    'time_between_queries': [30, 120],  # seconds
    'session_pattern': 'exploratory'  # vs 'focused'
}
```

**Signals:**
- Which series are queried together in same session?
- What's the typical query order? (GDP → UNRATE suggests interest in economic correlation)
- Are users doing joins? (Multi-series interest)

---

### 3. **Implicit Feedback** (User Engagement)
Currently missing!

**What to feed Monolith:**
```python
{
    'series_id': 'GDP',
    'engagement_signals': {
        'result_count': 150,           # How many rows returned
        'viz_created': True,           # Did user visualize?
        'viz_type': 'line',            # What type?
        'export_format': 'csv',        # Did they export?
        'query_refinement': True,      # Did they rerun with modifications?
        'time_spent': 180              # Seconds on this series
    }
}
```

**Strong positive signals:**
- Created visualization
- Exported data
- Queried multiple times (interest)
- Long session time

**Negative signals:**
- Quick exit after query
- No follow-up actions
- Error in query

---

### 4. **Join/Relationship Patterns** (Graph Features)
Currently in: `kumo_rfm_relationships`

**What to feed Monolith:**
```sql
-- Example: User joined GDP with UNRATE
SELECT * FROM economic_study_series ess1
JOIN economic_study_series ess2 
  ON ess1.study_id = ess2.study_id
WHERE ess1.series_id = 'GDP' AND ess2.series_id = 'UNRATE'
```

**Extract as:**
```python
{
    'source_series': 'GDP',
    'target_series': 'UNRATE',
    'relationship_type': 'user_joined',
    'strength': 0.85,  # Based on frequency
    'context': 'same_study'
}
```

---

### 5. **Feature Engineering from Queries** (Category Preferences)

**Current synthetic categories in monolith:**
- category_id, frequency_id, units_id

**What we can learn from CLI:**
```python
{
    'user_id': 'user_123',
    'category_preference': {
        'Employment': 0.6,      # 60% of queries
        'Prices': 0.3,          # 30% of queries
        'Production': 0.1       # 10% of queries
    },
    'frequency_preference': {
        'Monthly': 0.8,
        'Quarterly': 0.2
    },
    'temporal_pattern': {
        'active_hours': [9, 10, 14, 15],  # When do they query?
        'days_active': ['Monday', 'Wednesday', 'Friday']
    }
}
```

---

### 6. **Error Patterns** (Negative Signals)
Currently missing!

**What to track:**
```python
{
    'user_id': 'user_123',
    'series_id': 'GDP',
    'error_type': 'syntax_error',
    'recovered': True,  # Did they fix and retry?
    'abandon_reason': 'no_results'  # Or 'complexity', 'wrong_data'
}
```

**Use for:**
- Filter out series that cause errors
- Recommend simpler alternatives
- Improve documentation

---

## Implementation Plan

### Phase 1: Basic Tracking (Week 1)
```python
class KumoRFMCLI:
    def execute_sql(self, query: str):
        start_time = time.time()
        
        # Extract series mentioned in query
        series_ids = self._extract_series_from_query(query)
        
        # Execute query
        results = self.cursor.execute(query)
        
        # Log interaction
        for series_id in series_ids:
            self._log_interaction(
                series_id=series_id,
                query=query,
                result_count=len(results),
                execution_time=time.time() - start_time
            )
```

### Phase 2: Session Tracking (Week 2)
- Track session_id for anonymous users
- Sequence of queries
- Time between queries
- Session goals (exploration vs retrieval)

### Phase 3: Engagement Signals (Week 3)
- Viz creation tracking
- Export tracking
- Query refinement patterns
- Time spent per series

### Phase 4: Export to Monolith Format (Week 4)
```python
def export_cli_interactions_for_monolith():
    """
    Convert CLI logs to Monolith training format.
    Replaces synthetic interactions with real user behavior.
    """
    # Query from kumo_rfm_queries
    interactions = query_db("""
        SELECT 
            user_id,
            series_id,
            executed_at as timestamp,
            result_count,
            metadata->>'viz_created' as viz_created,
            metadata->>'exported' as exported
        FROM kumo_rfm_queries q
        -- Extract series_id from query_text or metadata
    """)
    
    # Score each interaction (implicit feedback)
    interactions['score'] = compute_engagement_score(
        viz_created=interactions.viz_created,
        exported=interactions.exported,
        result_count=interactions.result_count
    )
    
    # Export in Monolith format
    return format_for_monolith(interactions)
```

---

## Expected Impact

### Current Monolith (Synthetic Data):
- Random user interactions weighted by popularity
- No temporal patterns
- No co-occurrence learning
- No engagement signals

### Improved Monolith (Real CLI Data):
- **Better recommendations**: Based on actual user behavior
- **Personalization**: Learn user preferences from session history
- **Co-occurrence**: Recommend series commonly queried together
- **Quality signals**: Rank series by engagement (viz, export, time spent)
- **Cold start**: New series get exposure based on similarity to popular ones

---

## Key Tables to Update

### 1. Add to `kumo_rfm_queries`
```sql
ALTER TABLE kumo_rfm_queries ADD COLUMN series_ids TEXT[];
ALTER TABLE kumo_rfm_queries ADD COLUMN engagement_score FLOAT;
ALTER TABLE kumo_rfm_queries ADD COLUMN session_id VARCHAR(100);
```

### 2. Create `user_series_interactions` (for Monolith)
```sql
CREATE TABLE user_series_interactions (
    interaction_id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    series_id VARCHAR(50) NOT NULL,
    interaction_type VARCHAR(50), -- 'query', 'viz', 'export', 'join'
    engagement_score FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    context JSONB -- query complexity, time of day, etc.
);

CREATE INDEX idx_interactions_user ON user_series_interactions(user_id);
CREATE INDEX idx_interactions_series ON user_series_interactions(series_id);
CREATE INDEX idx_interactions_session ON user_series_interactions(session_id);
```

### 3. Create `series_cooccurrence` (for recommendations)
```sql
CREATE TABLE series_cooccurrence (
    series_id_1 VARCHAR(50),
    series_id_2 VARCHAR(50),
    cooccurrence_count INTEGER,
    avg_time_between_seconds FLOAT,
    strength FLOAT,  -- Normalized score
    PRIMARY KEY (series_id_1, series_id_2)
);
```

---

## Quick Win: Immediate Implementation

Add this to `execute_sql()` in CLI:

```python
def execute_sql(self, query: str, store_results: bool = True):
    """Execute SQL query and log interaction for Monolith."""
    import time
    import re
    
    start_time = time.time()
    
    try:
        self.cursor.execute(query)
        
        # Extract series IDs from query
        series_ids = re.findall(r"series_id\s*[=IN]\s*'([A-Z0-9]+)'", query, re.IGNORECASE)
        
        if self.cursor.description:
            results = self.cursor.fetchall()
            result_count = len(results)
            
            # Log interaction for each series
            for series_id in set(series_ids):
                self._log_series_interaction(
                    series_id=series_id,
                    query=query,
                    result_count=result_count,
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )
        
        # ... rest of execute_sql ...
        
    except Exception as e:
        # Log error for negative signal
        for series_id in set(series_ids):
            self._log_series_error(series_id, str(e))
        raise

def _log_series_interaction(self, series_id, query, result_count, execution_time_ms):
    """Log user-series interaction for Monolith feed."""
    try:
        self.cursor.execute("""
            INSERT INTO user_series_interactions 
            (user_id, session_id, series_id, interaction_type, engagement_score, context)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            self.user_id or 'anonymous',
            self.session_id,
            series_id,
            'query',
            self._compute_engagement_score(result_count, execution_time_ms),
            json.dumps({
                'query': query[:100],
                'result_count': result_count,
                'execution_time_ms': execution_time_ms
            })
        ))
        self.conn.commit()
    except Exception as e:
        # Don't fail query on logging error
        print(f"Warning: Failed to log interaction: {e}")
```

---

## Summary

**From the CLI process, we can capture:**
1. Real user-series interactions (not synthetic!)
2. Query sequences and co-occurrence patterns
3. Engagement signals (viz, export, time spent)
4. Join patterns (which series go together)
5. Category/frequency preferences
6. Error patterns (negative signals)

**This transforms Monolith from:**
- Synthetic demo with fake interactions
- Production system trained on real user behavior

**Result:** Better recommendations, personalization, and series discovery! 
