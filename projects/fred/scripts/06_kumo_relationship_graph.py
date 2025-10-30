#!/usr/bin/env python3
"""
Enhanced Kumo RFM with Relationship Graph
Builds a graph showing connections between FRED series based on:
- Category relationships
- Co-occurrence patterns
- Semantic similarity
- Domain knowledge (interest rates -> housing, etc.)
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import kumoai.experimental.rfm as rfm
    KUMO_AVAILABLE = True
except ImportError:
    KUMO_AVAILABLE = False
    print("Warning: kumoai not installed. Install with: pip install kumoai")


class KumoRelationshipGraph:
    """Build and query a relationship graph between FRED series."""
    
    def __init__(self, api_key: Optional[str] = None):
        if not KUMO_AVAILABLE:
            raise ImportError("kumoai package required")
        
        self.api_key = api_key or os.getenv('KUMO_API_KEY')
        if not self.api_key:
            raise ValueError("KUMO_API_KEY required")
        
        rfm.init(api_key=self.api_key)
        self.graph = None
        self.model = None
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load FRED series metadata."""
        if data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported format: {data_path}")
    
    def create_domain_relationships(self, series_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create relationship edges based on domain knowledge.
        
        Examples of domain relationships:
        - Interest rates <-> Housing (mortgage rates affect housing)
        - Employment <-> GDP (employment affects economic output)
        - Inflation <-> Interest rates (Fed uses rates to control inflation)
        """
        relationships = []
        rel_id = 0
        
        # Define domain relationship rules
        domain_rules = [
            # Interest rates affect housing market
            {
                'source_categories': ['Mortgage Rate', 'Interest Rates', 'Bank Reserves'],
                'target_categories': ['Housing Starts', 'Building Permits', 'Home Sales', 'Mortgage'],
                'relationship_type': 'monetary_policy_impact',
                'strength': 0.9,
                'description': 'Interest rates directly impact housing affordability'
            },
            # Employment affects GDP
            {
                'source_categories': ['Unemployment', 'Employment'],
                'target_categories': ['GDP', 'Gross Domestic Product'],
                'relationship_type': 'economic_indicator',
                'strength': 0.85,
                'description': 'Employment levels indicate economic activity'
            },
            # Inflation and interest rates
            {
                'source_categories': ['Inflation'],
                'target_categories': ['Interest Rates', 'Bank Reserves'],
                'relationship_type': 'monetary_policy_response',
                'strength': 0.95,
                'description': 'Fed adjusts rates to control inflation'
            },
            # Consumer spending affects GDP
            {
                'source_categories': ['Consumer Spending', 'Retail Sales'],
                'target_categories': ['GDP', 'Gross Domestic Product'],
                'relationship_type': 'economic_driver',
                'strength': 0.8,
                'description': 'Consumer spending is major GDP component'
            },
            # Housing and construction
            {
                'source_categories': ['Housing Starts', 'Building Permits'],
                'target_categories': ['Construction', 'Employment'],
                'relationship_type': 'industry_linkage',
                'strength': 0.75,
                'description': 'Housing construction creates jobs'
            },
        ]
        
        print("Creating domain-based relationships...")
        
        for rule in domain_rules:
            # Find series in source categories
            source_mask = series_df['category'].str.contains('|'.join(rule['source_categories']), case=False, na=False, regex=True)
            source_series = series_df[source_mask]
            
            # Find series in target categories
            target_mask = series_df['category'].str.contains('|'.join(rule['target_categories']), case=False, na=False, regex=True)
            target_series = series_df[target_mask]
            
            # Create relationships between top series in each group
            for _, source in source_series.nlargest(10, 'popularity').iterrows():
                for _, target in target_series.nlargest(10, 'popularity').iterrows():
                    if source['series_id'] != target['series_id']:
                        relationships.append({
                            'relationship_id': f"rel_{rel_id}",
                            'source_id': source['series_id'],
                            'target_id': target['series_id'],
                            'relationship_type': rule['relationship_type'],
                            'strength': rule['strength'],
                            'description': rule['description'],
                            'source_category': source['category'],
                            'target_category': target['category']
                        })
                        rel_id += 1
        
        print(f"  Created {len(relationships)} domain relationships")
        
        return pd.DataFrame(relationships) if relationships else pd.DataFrame(
            columns=['relationship_id', 'source_id', 'target_id', 'relationship_type', 'strength', 'description', 'source_category', 'target_category']
        )
    
    def create_category_relationships(self, series_df: pd.DataFrame, max_per_category: int = 20) -> pd.DataFrame:
        """Create relationships between series in the same category."""
        relationships = []
        rel_id = 0
        
        print("Creating category-based relationships...")
        
        for category, group in series_df.groupby('category'):
            # Connect top series within each category
            top_series = group.nlargest(min(max_per_category, len(group)), 'popularity')['series_id'].tolist()
            
            for i, source in enumerate(top_series):
                for target in top_series[i+1:]:
                    relationships.append({
                        'relationship_id': f"cat_rel_{rel_id}",
                        'source_id': source,
                        'target_id': target,
                        'relationship_type': 'same_category',
                        'strength': 0.7,
                        'description': f'Both in {category}',
                        'source_category': category,
                        'target_category': category
                    })
                    rel_id += 1
        
        print(f"  Created {len(relationships)} category relationships")
        
        return pd.DataFrame(relationships) if relationships else pd.DataFrame(
            columns=['relationship_id', 'source_id', 'target_id', 'relationship_type', 'strength', 'description', 'source_category', 'target_category']
        )
    
    def build_graph(self, series_df: pd.DataFrame, include_domain: bool = True, include_category: bool = True) -> rfm.LocalGraph:
        """Build a graph with series and their relationships."""
        print("Building Kumo relationship graph...")
        
        # Create series table
        series_table = rfm.LocalTable(
            series_df[['series_id', 'title', 'category', 'frequency', 'popularity']].copy(),
            name="series",
            primary_key="series_id"
        )
        series_table['series_id'].stype = "ID"
        series_table['category'].stype = "categorical"
        series_table['frequency'].stype = "categorical"
        series_table['popularity'].stype = "numerical"
        
        # Create relationship tables
        all_relationships = []
        
        if include_domain:
            domain_rels = self.create_domain_relationships(series_df)
            if not domain_rels.empty:
                all_relationships.append(domain_rels)
        
        if include_category:
            category_rels = self.create_category_relationships(series_df, max_per_category=15)
            if not category_rels.empty:
                all_relationships.append(category_rels)
        
        if not all_relationships:
            print("Warning: No relationships created. Using simple graph.")
            graph = rfm.LocalGraph(tables=[series_table])
            print(f"Graph created with {len(series_df)} series and 0 relationships")
        else:
            # Combine all relationships
            relationships_df = pd.concat(all_relationships, ignore_index=True)
            relationships_df = relationships_df.drop_duplicates(subset=['source_id', 'target_id'], keep='first')
            
            # Create relationship table
            rel_table = rfm.LocalTable(
                relationships_df,
                name="relationships",
                primary_key="relationship_id"
            )
            rel_table['relationship_id'].stype = "ID"
            rel_table['source_id'].stype = "ID"
            rel_table['target_id'].stype = "ID"
            rel_table['relationship_type'].stype = "categorical"
            rel_table['strength'].stype = "numerical"
            
            # Create graph
            graph = rfm.LocalGraph(tables=[series_table, rel_table])
            
            # Add edges manually
            graph.link(src_table="relationships", fkey="source_id", dst_table="series")
            graph.link(src_table="relationships", fkey="target_id", dst_table="series")
            
            print(f"Graph created with {len(series_df)} series and {len(relationships_df)} relationships")
            print(f"  Relationship types: {relationships_df['relationship_type'].unique().tolist()}")
        
        self.graph = graph
        self.model = rfm.KumoRFM(graph)
        
        return graph
    
    def find_connected_series(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Find series connected through the relationship graph.
        
        Example: query="interest rates housing" will find:
        1. Series matching "interest rates"
        2. Series matching "housing"
        3. Relationships connecting them
        """
        if self.graph is None:
            raise ValueError("Build graph first with build_graph()")
        
        series_df = self.graph.tables['series']._data
        
        # Extract keywords
        terms = query.lower().split()
        
        # Find series matching each term
        matching_series = {}
        for term in terms:
            mask = series_df['title'].str.lower().str.contains(term, na=False) | \
                   series_df['category'].str.lower().str.contains(term, na=False)
            matches = series_df[mask].nlargest(20, 'popularity')
            matching_series[term] = matches['series_id'].tolist()
        
        if len(matching_series) < 2:
            print(f"Query must have at least 2 terms to find connections. Found matches for: {list(matching_series.keys())}")
            return pd.DataFrame()
        
        # Find relationships connecting these series
        if 'relationships' in self.graph.tables:
            rel_df = self.graph.tables['relationships']._data
            
            # Find relationships where source matches one term and target matches another
            connections = []
            terms_list = list(matching_series.keys())
            
            for i, term1 in enumerate(terms_list):
                for term2 in terms_list[i+1:]:
                    # Forward connections: term1 -> term2
                    forward = rel_df[
                        rel_df['source_id'].isin(matching_series[term1]) &
                        rel_df['target_id'].isin(matching_series[term2])
                    ]
                    
                    # Backward connections: term2 -> term1
                    backward = rel_df[
                        rel_df['source_id'].isin(matching_series[term2]) &
                        rel_df['target_id'].isin(matching_series[term1])
                    ]
                    
                    connections.append(forward)
                    connections.append(backward)
            
            if connections:
                all_connections = pd.concat(connections, ignore_index=True)
                all_connections = all_connections.drop_duplicates(subset=['source_id', 'target_id'])
                
                # Sort by strength
                all_connections = all_connections.nlargest(top_k, 'strength')
                
                # Enrich with series information
                result = all_connections.merge(
                    series_df[['series_id', 'title', 'category', 'popularity']],
                    left_on='source_id',
                    right_on='series_id',
                    suffixes=('', '_source')
                ).merge(
                    series_df[['series_id', 'title', 'category', 'popularity']],
                    left_on='target_id',
                    right_on='series_id',
                    suffixes=('_source', '_target')
                )
                
                return result[[
                    'source_id', 'title_source', 'category_source', 'popularity_source',
                    'target_id', 'title_target', 'category_target', 'popularity_target',
                    'relationship_type', 'strength', 'description'
                ]]
            else:
                print(f"No direct relationships found between {' and '.join(terms_list)}")
                return pd.DataFrame()
        else:
            print("No relationships table in graph")
            return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description='Kumo RFM Relationship Graph for FRED data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find connections between interest rates and housing
  python 06_kumo_relationship_graph.py --connect "interest rates housing" --top-k 10
  
  # Find connections between employment and GDP
  python 06_kumo_relationship_graph.py --connect "employment GDP" --top-k 10
  
  # Find connections between inflation and consumer spending
  python 06_kumo_relationship_graph.py --connect "inflation consumer" --top-k 10
        """
    )
    
    parser.add_argument('--data', type=str, default='data/fred_series_metadata.parquet',
                       help='Path to FRED series data')
    parser.add_argument('--api-key', type=str, help='Kumo API key')
    parser.add_argument('--connect', type=str, required=True,
                       help='Query to find connections (e.g., "interest rates housing")')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of connections to return')
    parser.add_argument('--no-domain', action='store_true',
                       help='Disable domain knowledge relationships')
    parser.add_argument('--no-category', action='store_true',
                       help='Disable category-based relationships')
    
    args = parser.parse_args()
    
    # Set API key
    if args.api_key:
        os.environ['KUMO_API_KEY'] = args.api_key
    
    try:
        # Load data
        print(f"Loading data from {args.data}...")
        df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data)
        print(f"Loaded {len(df)} series\n")
        
        # Build relationship graph
        kumo = KumoRelationshipGraph(api_key=args.api_key)
        kumo.build_graph(
            df,
            include_domain=not args.no_domain,
            include_category=not args.no_category
        )
        
        # Find connections
        print(f"\nFinding connections for: '{args.connect}'\n")
        connections = kumo.find_connected_series(args.connect, top_k=args.top_k)
        
        if not connections.empty:
            print("=" * 120)
            print(f"Top {len(connections)} Connections:")
            print("=" * 120)
            
            for idx, row in connections.iterrows():
                print(f"\n{row['source_id']} -> {row['target_id']}")
                print(f"  From: {row['title_source'][:70]}")
                print(f"        Category: {row['category_source']}, Popularity: {row['popularity_source']:.0f}")
                print(f"  To:   {row['title_target'][:70]}")
                print(f"        Category: {row['category_target']}, Popularity: {row['popularity_target']:.0f}")
                print(f"  Relationship: {row['relationship_type']} (strength: {row['strength']:.2f})")
                print(f"  Why: {row['description']}")
        else:
            print("No connections found. Try different search terms.")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    main()
