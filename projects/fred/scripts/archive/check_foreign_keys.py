#!/usr/bin/env python3
"""
Check Foreign Keys in KumoAI Graph

This script inspects the KumoAI graph to see what relationships and
foreign keys are actually detected.
"""

import os
import pandas as pd
import numpy as np

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

try:
    import kumoai.experimental.rfm as rfm
    KUMO_AVAILABLE = True
except ImportError:
    KUMO_AVAILABLE = False
    print("Error: kumoai not installed")
    exit(1)


def check_demo1_foreign_keys():
    """Check foreign keys in Demo 1 multi-table setup."""
    print("="*70)
    print("CHECKING DEMO 1: Multi-Table Foreign Keys")
    print("="*70)
    
    # Load data
    df = pd.read_parquet('data/fred_series_metadata.parquet')
    
    # Create category table
    categories = df.groupby('category').agg({
        'series_id': 'count',
        'popularity': 'mean'
    }).reset_index()
    categories.columns = ['category_name', 'series_count', 'avg_popularity']
    categories['category_id'] = range(len(categories))
    
    # Create frequency table
    frequencies = df.groupby(['frequency', 'frequency_name']).size().reset_index()
    frequencies.columns = ['freq_code', 'freq_name', 'count']
    frequencies['frequency_id'] = range(len(frequencies))
    
    # Add foreign keys to series
    series_enhanced = df.copy()
    cat_map = dict(zip(categories['category_name'], categories['category_id']))
    freq_map = dict(zip(frequencies['freq_code'], frequencies['frequency_id']))
    series_enhanced['category_id'] = series_enhanced['category'].map(cat_map)
    series_enhanced['frequency_id'] = series_enhanced['frequency'].map(freq_map)
    
    print(f"\nData Summary:")
    print(f"  Series: {len(series_enhanced)}")
    print(f"  Categories: {len(categories)}")
    print(f"  Frequencies: {len(frequencies)}")
    
    # Check foreign key mappings
    print(f"\nForeign Key Statistics:")
    print(f"  series.category_id -> categories.category_id")
    print(f"    Unique values in series: {series_enhanced['category_id'].nunique()}")
    print(f"    Null values: {series_enhanced['category_id'].isna().sum()}")
    print(f"    Range: {series_enhanced['category_id'].min():.0f} to {series_enhanced['category_id'].max():.0f}")
    
    print(f"\n  series.frequency_id -> frequencies.frequency_id")
    print(f"    Unique values in series: {series_enhanced['frequency_id'].nunique()}")
    print(f"    Null values: {series_enhanced['frequency_id'].isna().sum()}")
    print(f"    Range: {series_enhanced['frequency_id'].min():.0f} to {series_enhanced['frequency_id'].max():.0f}")
    
    # Check referential integrity
    print(f"\nReferential Integrity Check:")
    cat_ids_in_series = set(series_enhanced['category_id'].dropna().astype(int))
    cat_ids_in_categories = set(categories['category_id'])
    orphans_cat = cat_ids_in_series - cat_ids_in_categories
    print(f"  Category FK orphans: {len(orphans_cat)}")
    if orphans_cat:
        print(f"    Orphan IDs: {sorted(orphans_cat)[:10]}")
    
    freq_ids_in_series = set(series_enhanced['frequency_id'].dropna().astype(int))
    freq_ids_in_frequencies = set(frequencies['frequency_id'])
    orphans_freq = freq_ids_in_series - freq_ids_in_frequencies
    print(f"  Frequency FK orphans: {len(orphans_freq)}")
    if orphans_freq:
        print(f"    Orphan IDs: {sorted(orphans_freq)[:10]}")
    
    # Create KumoAI tables
    print(f"\nCreating KumoAI tables...")
    
    series_table = rfm.LocalTable(
        series_enhanced[['series_id', 'category_id', 'frequency_id', 'popularity', 'notes_length']],
        name="series",
        primary_key="series_id"
    )
    
    category_table = rfm.LocalTable(
        categories,
        name="categories",
        primary_key="category_id"
    )
    
    frequency_table = rfm.LocalTable(
        frequencies,
        name="frequencies",
        primary_key="frequency_id"
    )
    
    # Set semantic types (marks as foreign keys)
    print(f"\nSetting semantic types...")
    series_table['category_id'].stype = "ID"
    series_table['frequency_id'].stype = "ID"
    category_table['category_id'].stype = "ID"
    frequency_table['frequency_id'].stype = "ID"
    
    print(f"  series.category_id -> stype: {series_table['category_id'].stype}")
    print(f"  series.frequency_id -> stype: {series_table['frequency_id'].stype}")
    print(f"  categories.category_id -> stype: {category_table['category_id'].stype}")
    print(f"  frequencies.frequency_id -> stype: {frequency_table['frequency_id'].stype}")
    
    # Build graph
    print(f"\nBuilding KumoAI graph...")
    graph = rfm.LocalGraph(
        tables=[series_table, category_table, frequency_table]
    )
    
    # Inspect graph structure
    print(f"\nGraph Structure:")
    print(f"  Number of tables: {len(graph.tables)}")
    print(f"  Table names: {graph.tables}")
    
    # Check for edges (relationships)
    print(f"\nGraph Edges/Relationships:")
    try:
        # Try to access graph edges if available
        if hasattr(graph, 'edges'):
            print(f"  Edges: {graph.edges}")
        elif hasattr(graph, 'relationships'):
            print(f"  Relationships: {graph.relationships}")
        elif hasattr(graph, '_edges'):
            print(f"  Internal edges: {graph._edges}")
        else:
            print("  (Edge information not directly accessible)")
            print("  KumoAI infers relationships from ID semantic types at query time")
    except Exception as e:
        print(f"  Could not access edge info: {e}")
    
    # Try to inspect table schemas
    print(f"\nTable Schemas:")
    for table in graph.tables.values():
        print(f"\n  {table.name}:")
        print(f"    Primary key: {table.primary_key}")
        try:
            # Get columns
            if hasattr(table, 'columns'):
                print(f"    Columns: {list(table.columns)}")
            elif hasattr(table, '_data'):
                print(f"    Columns: {list(table._data.columns)}")
            
            # Get column types
            if hasattr(table, '_column_stypes'):
                print(f"    Semantic types:")
                for col, stype in table._column_stypes.items():
                    print(f"      {col}: {stype}")
        except Exception as e:
            print(f"    Could not access column info: {e}")
    
    # Test if relationships work with predictions
    print(f"\n\nTesting Relationship Inference:")
    print("When you query series data, KumoAI should automatically:")
    print("  1. See series.category_id (type: ID)")
    print("  2. Look for matching categories.category_id (type: ID)")
    print("  3. Infer the foreign key relationship")
    print("  4. Use category features in predictions")
    
    return graph


def check_demo2_temporal():
    """Check temporal table structure."""
    print("\n\n" + "="*70)
    print("CHECKING DEMO 2: Temporal Table")
    print("="*70)
    
    df = pd.read_parquet('data/fred_series_metadata.parquet')
    
    # Create synthetic temporal data
    time_series = []
    for idx, row in df.head(10).iterrows():
        base_pop = row['popularity']
        for month in range(12):
            from datetime import datetime, timedelta
            time_series.append({
                'series_id': row['series_id'],
                'month': month,
                'date': (datetime.now() - timedelta(days=30*(12-month))).strftime('%Y-%m-%d'),
                'popularity': base_pop + np.random.randint(-10, 20),
                'views': np.random.randint(100, 10000)
            })
    
    ts_df = pd.DataFrame(time_series)
    ts_df['date'] = pd.to_datetime(ts_df['date'])
    
    print(f"\nTemporal Data Summary:")
    print(f"  Total observations: {len(ts_df)}")
    print(f"  Unique series: {ts_df['series_id'].nunique()}")
    print(f"  Time range: {ts_df['date'].min()} to {ts_df['date'].max()}")
    print(f"  Months: {ts_df['month'].unique()}")
    
    # Create table (composite primary key as single string)
    ts_table = rfm.LocalTable(
        ts_df,
        name="popularity_trends",
        primary_key="series_id"  # Use single column as primary key
    )
    
    # Set semantic types
    ts_table['series_id'].stype = "ID"
    ts_table['month'].stype = "numerical"
    ts_table['date'].stype = "timestamp"
    ts_table['popularity'].stype = "numerical"
    ts_table['views'].stype = "numerical"
    
    print(f"\nSemantic Types:")
    print(f"  series_id: {ts_table['series_id'].stype}")
    print(f"  month: {ts_table['month'].stype}")
    print(f"  date: {ts_table['date'].stype}")
    print(f"  popularity: {ts_table['popularity'].stype}")
    print(f"  views: {ts_table['views'].stype}")
    
    # Build graph
    graph = rfm.LocalGraph(tables=[ts_table])
    
    print(f"\nGraph Structure:")
    print(f"  Tables: {[t.name for t in graph.tables.values()]}")
    print(f"  Primary key: {ts_table.primary_key}")


def main():
    """Run all foreign key checks."""
    api_key = os.getenv('KUMO_API_KEY')
    if not api_key:
        print("Error: KUMO_API_KEY must be set")
        print("export KUMO_API_KEY='your-key'")
        return
    
    rfm.init(api_key=api_key)
    
    print("\nKUMOAI FOREIGN KEY DIAGNOSTIC")
    print("="*70)
    print("\nThis script checks what foreign keys and relationships are")
    print("created in the KumoAI graph structure.\n")
    
    # Check Demo 1
    try:
        graph1 = check_demo1_foreign_keys()
    except Exception as e:
        print(f"\nDemo 1 check failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Check Demo 2
    try:
        check_demo2_temporal()
    except Exception as e:
        print(f"\nDemo 2 check failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
KumoAI uses implicit foreign key relationships:

1. When you set a column's stype to "ID", KumoAI marks it as a potential key
2. At query time, KumoAI looks for matching ID columns across tables
3. If it finds category_id (ID) in table A and category_id (ID) in table B,
   it infers a foreign key relationship
4. This allows queries to automatically use joined information

The relationships are NOT stored explicitly in the graph structure.
They are inferred dynamically based on:
- Column names matching
- Both columns having stype="ID"
- One being a primary key

This is different from traditional databases where FKs are explicitly defined.
    """)


if __name__ == '__main__':
    main()
