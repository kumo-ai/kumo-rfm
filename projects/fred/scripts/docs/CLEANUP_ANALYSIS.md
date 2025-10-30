# Project Cleanup Analysis

**Date:** 2025-10-29
**Status:** Analysis Complete

## Summary

The project has **inconsistencies between documentation and implementation**:
- Documentation refers to DuckDB 25+ times
- Actual implementation uses PostgreSQL (`03_load_to_postgres.py`)
- Database file `fred.db` is actually DuckDB format
- SQL schema `create_tables.sql` is PostgreSQL syntax

## Key Findings

###  What's Working
- Data exists: txt files parsed, embeddings created, monolith features generated
- Core parsing script (`02_parse_fred_txt.py`) works
- Vector search embeddings exist (168MB)
- Monolith features prepared
- 16 Python scripts total

###  Major Issues

#### 1. **Database Mismatch**
- **Problem:** Docs say "DuckDB", code says "PostgreSQL", file is DuckDB
- **Files affected:**
  - `README.md` - mentions DuckDB 10+ times
  - `00_EXECUTION_ORDER.md` - shows DuckDB commands
  - `03_load_to_postgres.py` - implements PostgreSQL
  - `create_tables.sql` - PostgreSQL syntax
  - `fred.db` - is actually a DuckDB file

#### 2. **Import Errors**
- `example_networkx_kumo.py` uses hacky importlib workaround for numbered modules
- Numbered file convention (01_, 02_, etc.) causes import issues
- Should use proper module structure or rename files

#### 3. **Documentation Confusion**
- Multiple overlapping docs: README.md, 00_EXECUTION_ORDER.md, 00_POSTGRES_SETUP.md
- Conflicting instructions
- Outdated references

## Detailed File Analysis

### Core Pipeline (KEEP - Essential)
```
 01_fetch_fred_data.py         - FRED API fetcher (works)
 02_parse_fred_txt.py           - Parser (tested, works)
? 03_load_to_postgres.py         - PostgreSQL loader (conflicts with DuckDB usage)
 04_vector_search.py            - Vector embeddings (working, data exists)
 05_kumo_rfm_integration.py    - Kumo integration (needs API key to test)
 06_prepare_monolith_features.py - Monolith prep (working, data exists)
```

### Support Scripts (EVALUATE)
```
? 00_research_assistant.py       - 758 lines, unclear purpose/usage
? 07_workflow_integrations.py   - 454 lines, integration examples
? 08_network_visualizations.py  - 610 lines, NetworkX visualizations
? 10_advanced_kumo_demo.py      - 688 lines, advanced demos
? 11_ml_schema_recommender.py   - 448 lines, ML schema recommendations
? 12_complete_fred_api.py       - 391 lines, complete API wrapper
```

### Utility Scripts (KEEP)
```
 99_pipeline.py                - Pipeline orchestrator
 99_run_pipeline.sh            - Bash orchestrator
 051_kumo_rfm_cli.py           - 997 lines, interactive CLI
```

### Special Cases
```
? check_foreign_keys.py         - 296 lines, debugging tool
 example_networkx_kumo.py      - Example script (has import issues)
```

### SQL/Config (REVIEW)
```
? create_tables.sql             - PostgreSQL schema but used with DuckDB?
 .env.example                  - Environment template (just added)
```

### Documentation (CONSOLIDATE)
```
 README.md                     - Main docs (needs DuckDB/Postgres clarification)
 00_EXECUTION_ORDER.md         - Execution guide (needs update)
? 00_POSTGRES_SETUP.md          - Postgres setup (conflicts with DuckDB usage)
? NETWORKX_KUMO_GUIDE.md        - NetworkX guide
? DATA_ENRICHMENT_GUIDE.md      - Data enrichment
? VISUALIZATION_SUMMARY.md      - Visualization summary
```

## Recommendations

### Priority 1: Fix Database Confusion
**Option A: Go Full DuckDB** (Recommended - simpler, file-based)
- Rename `03_load_to_postgres.py` → `03_load_to_duckdb.py`
- Update implementation to use DuckDB library
- Update `create_tables.sql` for DuckDB syntax
- Update all documentation

**Option B: Go Full PostgreSQL**
- Delete `fred.db` (DuckDB file)
- Update documentation to remove DuckDB references
- Keep PostgreSQL implementation
- Add proper Postgres setup guide

### Priority 2: Fix Import Issues
**Rename numbered files to be importable:**
```bash
01_fetch_fred_data.py       → fetch_fred_data.py
02_parse_fred_txt.py        → parse_fred_txt.py
03_load_to_postgres.py      → load_to_database.py
04_vector_search.py         → vector_search.py
05_kumo_rfm_integration.py  → kumo_integration.py
06_prepare_monolith_features.py → prepare_monolith.py
07_workflow_integrations.py → workflow_integrations.py
08_network_visualizations.py → network_visualizations.py
```

Keep orchestrators numbered:
```bash
99_pipeline.py              → run_pipeline.py (or keep as is)
99_run_pipeline.sh          → run_pipeline.sh (or keep as is)
```

### Priority 3: Consolidate Documentation
**Merge into single comprehensive README:**
- Main README.md (keep, update)
- Merge 00_EXECUTION_ORDER.md content
- Remove or archive: 00_POSTGRES_SETUP.md, NETWORKX_KUMO_GUIDE.md, DATA_ENRICHMENT_GUIDE.md, VISUALIZATION_SUMMARY.md
- Create separate docs/ folder if needed

### Priority 4: Evaluate Unused Scripts
**Scripts that may be removable:**
- `00_research_assistant.py` - No clear usage, 758 lines
- `check_foreign_keys.py` - Debugging tool, not production
- `11_ml_schema_recommender.py` - Unclear if used
- `12_complete_fred_api.py` - Duplicate of 01_fetch_fred_data.py?

**Keep but mark as examples/demos:**
- `10_advanced_kumo_demo.py` - Move to examples/
- `07_workflow_integrations.py` - Move to examples/
- `08_network_visualizations.py` - Move to examples/
- `example_networkx_kumo.py` - Already an example

### Priority 5: Add Proper Project Structure
```
/fred-pipeline/
├── src/
│   ├── fetch_fred_data.py
│   ├── parse_fred_txt.py
│   ├── load_to_database.py
│   ├── vector_search.py
│   ├── kumo_integration.py
│   └── prepare_monolith.py
├── examples/
│   ├── advanced_kumo_demo.py
│   ├── workflow_integrations.py
│   ├── network_visualizations.py
│   └── networkx_kumo_example.py
├── cli/
│   └── kumo_cli.py (051_kumo_rfm_cli.py)
├── data/
│   ├── embeddings/
│   ├── monolith/
│   └── fred_series_metadata.parquet
├── txt/
│   └── *.txt (FRED data files)
├── tests/
│   └── test_*.py
├── run_pipeline.py (99_pipeline.py)
├── run_pipeline.sh (99_run_pipeline.sh)
├── requirements.txt
├── .env.example
└── README.md
```

## Action Items

### Immediate (Fix Blockers)
1. [ ] Choose database: DuckDB or PostgreSQL
2. [ ] Update implementation to match choice
3. [ ] Fix import issues in example_networkx_kumo.py
4. [ ] Update .env.example with correct variables

### Short-term (Clean Up)
5. [ ] Rename numbered files for proper imports
6. [ ] Consolidate documentation
7. [ ] Move demo/example scripts to examples/
8. [ ] Remove or archive unused scripts
9. [ ] Create requirements.txt with actual dependencies

### Long-term (Refactor)
10. [ ] Reorganize into src/ structure
11. [ ] Add proper tests/
12. [ ] Create unified CLI entry point
13. [ ] Add logging throughout
14. [ ] Document all functions/classes

## Dependencies to Verify
```bash
# Core
pandas
numpy
python-dotenv

# Database (choose one)
duckdb  # OR psycopg2-binary

# Vector search
sentence-transformers
faiss-cpu

# Optional
kumoai
tensorflow
networkx
matplotlib
seaborn

# Workflow integrations (optional)
elasticsearch
pinecone-client
weaviate-client
neo4j
```

## Estimated Cleanup Time
- **Quick fix (Priority 1-2):** 2-3 hours
- **Full cleanup (Priority 1-4):** 1 day
- **Complete refactor (All priorities):** 2-3 days

## Next Steps

**Recommended immediate action:**
1. Decide: DuckDB or PostgreSQL?
2. Run: Fix database implementation
3. Run: Fix imports in example_networkx_kumo.py
4. Create: requirements.txt
5. Test: Run full pipeline end-to-end

**Quick wins:**
- Archive unused scripts to archive/ folder
- Move examples to examples/ folder
- Create single comprehensive README
