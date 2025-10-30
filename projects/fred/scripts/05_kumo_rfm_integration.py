#!/usr/bin/env python3
"""
Kumo RFM integration for FRED economic series data.
Uses KumoRFM to make predictions and discover relationships in economic data.
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import json
import networkx as nx

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
    print("Warning: kumoai not installed. Install with: pip install kumoai")


class KumoFREDIntegration:
    """Integration layer between FRED data and Kumo RFM."""
    
    def __init__(self, api_key: Optional[str] = None):
        if not KUMO_AVAILABLE:
            raise ImportError("kumoai package is required. Install with: pip install kumoai")
        
        self.api_key = api_key or os.getenv('KUMO_API_KEY')
        if not self.api_key:
            raise ValueError("KUMO_API_KEY must be provided or set as environment variable")
        
        # Initialize Kumo RFM client
        rfm.init(api_key=self.api_key)
        self.graph = None
        self.model = None
        
        # Track exploration history
        self.exploration_history = []
        
        # Setup visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load FRED series metadata."""
        if data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    def create_series_table(self, series_df: pd.DataFrame) -> 'rfm.LocalTable':
        """Create a LocalTable from FRED series data with proper metadata."""
        # Prepare the dataframe
        df = series_df.copy()
        
        # Create LocalTable with metadata
        table = rfm.LocalTable(
            df,
            name="series",
            primary_key="series_id"
        )
        
        # Set semantic types
        table['series_id'].stype = "ID"
        table['category'].stype = "categorical"
        table['frequency'].stype = "categorical"
        if 'popularity' in df.columns:
            table['popularity'].stype = "numerical"
        if 'notes_length' in df.columns:
            table['notes_length'].stype = "numerical"
        
        return table
    
    def build_graph(self, series_df: pd.DataFrame, include_edges: bool = True) -> 'rfm.LocalGraph':
        """Build a LocalGraph from FRED series data with relationships.
        
        Args:
            series_df: DataFrame with series metadata
            include_edges: Whether to create relationship edges between series
        """
        print("Building Kumo RFM graph...")
        
        # Create series table  
        series_table = self.create_series_table(series_df)
        
        # Create relationship edges if requested
        if include_edges:
            relationships_df = self._create_relationships(series_df)
            
            if len(relationships_df) > 0:
                # Create relationships table
                relationships_table = rfm.LocalTable(
                    relationships_df,
                    name="series_relationships",
                    primary_key="relationship_id"
                )
                
                # Set semantic types
                relationships_table['source_id'].stype = "ID"
                relationships_table['target_id'].stype = "ID"
                relationships_table['relationship_type'].stype = "categorical"
                relationships_table['strength'].stype = "numerical"
                
                # Define foreign key relationships
                relationships_table['source_id'].references = series_table['series_id']
                relationships_table['target_id'].references = series_table['series_id']
                
                # Create graph with both tables
                graph = rfm.LocalGraph(tables=[series_table, relationships_table])
                
                print(f"Graph created with {len(series_df)} series and {len(relationships_df)} relationships")
            else:
                graph = rfm.LocalGraph(tables=[series_table])
                print(f"Graph created with {len(series_df)} series (no relationships)")
        else:
            graph = rfm.LocalGraph(tables=[series_table])
            print(f"Graph created with {len(series_df)} series (edges disabled)")
        
        print(f"  Categories: {series_df['category'].nunique()}")
        print(f"  Frequencies: {series_df['frequency'].nunique()}")
        
        self.graph = graph
        self.model = rfm.KumoRFM(graph)
        
        return graph
    
    def _create_relationships(self, series_df: pd.DataFrame, max_relationships: int = 50000) -> pd.DataFrame:
        """Create relationship edges between related series.
        
        Creates multiple types of relationships:
        1. Same category relationships (strong signal)
        2. Same frequency relationships (temporal pattern)
        3. Keyword similarity (title/notes overlap)
        4. Cross-category for popular series (discovery)
        """
        relationships = []
        rel_id = 0
        
        # 1. Category-based relationships (strongest signal)
        print("  Creating category-based relationships...")
        for category, group in series_df.groupby('category'):
            if len(group) < 2:
                continue
                
            # Connect top series within category
            top_series = group.nlargest(min(15, len(group)), 'popularity')
            
            for i in range(len(top_series)):
                for j in range(i+1, len(top_series)):
                    source = top_series.iloc[i]
                    target = top_series.iloc[j]
                    
                    # Strength based on popularity similarity
                    pop_similarity = 1.0 - abs(source['popularity'] - target['popularity']) / max(source['popularity'], target['popularity'], 1)
                    strength = 0.7 + (0.3 * pop_similarity)  # 0.7 to 1.0
                    
                    relationships.append({
                        'relationship_id': f"rel_{rel_id}",
                        'source_id': source['series_id'],
                        'target_id': target['series_id'],
                        'relationship_type': 'same_category',
                        'strength': strength
                    })
                    rel_id += 1
                    
                    if rel_id >= max_relationships * 0.6:  # 60% of budget for category
                        break
                if rel_id >= max_relationships * 0.6:
                    break
            if rel_id >= max_relationships * 0.6:
                break
        
        print(f"    Category relationships: {len(relationships)}")
        
        # 2. Frequency-based relationships (temporal patterns)
        if rel_id < max_relationships:
            print("  Creating frequency-based relationships...")
            freq_start = len(relationships)
            
            for freq, group in series_df.groupby('frequency'):
                if len(group) < 2:
                    continue
                    
                # Connect popular series with same frequency
                top_freq = group.nlargest(min(10, len(group)), 'popularity')
                
                for i in range(len(top_freq)):
                    for j in range(i+1, min(i+5, len(top_freq))):  # Limit connections per series
                        source = top_freq.iloc[i]
                        target = top_freq.iloc[j]
                        
                        relationships.append({
                            'relationship_id': f"rel_{rel_id}",
                            'source_id': source['series_id'],
                            'target_id': target['series_id'],
                            'relationship_type': 'same_frequency',
                            'strength': 0.5
                        })
                        rel_id += 1
                        
                        if rel_id >= max_relationships * 0.8:  # 80% budget
                            break
                    if rel_id >= max_relationships * 0.8:
                        break
                if rel_id >= max_relationships * 0.8:
                    break
            
            print(f"    Frequency relationships: {len(relationships) - freq_start}")
        
        # 3. Keyword similarity (title overlap)
        if rel_id < max_relationships:
            print("  Creating keyword similarity relationships...")
            keyword_start = len(relationships)
            
            # Extract common economic keywords
            keywords = ['rate', 'price', 'index', 'gdp', 'employment', 'inflation', 
                       'housing', 'retail', 'trade', 'debt', 'credit', 'consumer', 
                       'producer', 'income', 'wage', 'bank']
            
            for keyword in keywords:
                if rel_id >= max_relationships * 0.95:  # 95% budget
                    break
                    
                # Find series with this keyword
                matches = series_df[series_df['title'].str.contains(keyword, case=False, na=False)]
                
                if len(matches) >= 2:
                    # Connect top 5 series with this keyword
                    top_matches = matches.nlargest(min(5, len(matches)), 'popularity')
                    
                    for i in range(len(top_matches)):
                        for j in range(i+1, len(top_matches)):
                            source = top_matches.iloc[i]
                            target = top_matches.iloc[j]
                            
                            relationships.append({
                                'relationship_id': f"rel_{rel_id}",
                                'source_id': source['series_id'],
                                'target_id': target['series_id'],
                                'relationship_type': f'keyword_{keyword}',
                                'strength': 0.6
                            })
                            rel_id += 1
                            
                            if rel_id >= max_relationships * 0.95:
                                break
                        if rel_id >= max_relationships * 0.95:
                            break
            
            print(f"    Keyword relationships: {len(relationships) - keyword_start}")
        
        # 4. Cross-category discovery (popular series connections)
        if rel_id < max_relationships:
            print("  Creating cross-category discovery relationships...")
            discovery_start = len(relationships)
            
            # Connect top 20 most popular series across all categories
            top_global = series_df.nlargest(20, 'popularity')
            
            for i in range(len(top_global)):
                for j in range(i+1, len(top_global)):
                    source = top_global.iloc[i]
                    target = top_global.iloc[j]
                    
                    # Only if different categories
                    if source['category'] != target['category']:
                        relationships.append({
                            'relationship_id': f"rel_{rel_id}",
                            'source_id': source['series_id'],
                            'target_id': target['series_id'],
                            'relationship_type': 'cross_category',
                            'strength': 0.4
                        })
                        rel_id += 1
                        
                        if rel_id >= max_relationships:
                            break
                if rel_id >= max_relationships:
                    break
            
            print(f"    Cross-category relationships: {len(relationships) - discovery_start}")
        
        print(f"  Total relationships created: {len(relationships)}")
        
        if len(relationships) == 0:
            return pd.DataFrame(columns=['relationship_id', 'source_id', 'target_id', 'relationship_type', 'strength'])
        
        return pd.DataFrame(relationships)
    
    def predict_series_popularity(self, series_id: str) -> pd.DataFrame:
        """
        Predict popularity score for a specific series based on metadata.
        
        Args:
            series_id: The series ID to predict popularity for
        
        Example use case: Predict which new economic indicators will be popular.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_graph() first")
        
        print(f"Predicting popularity for series: {series_id}")
        
        # PQL query to predict popularity for a specific series
        query = f"PREDICT series.popularity FOR series.series_id='{series_id}'"
        
        try:
            result = self.model.predict(query)
            return result
        except Exception as e:
            print(f"Kumo prediction failed: {e}")
            print("This is a demo - ensure you have valid API access and data format")
            return pd.DataFrame()
    
    def predict_category(self, series_id: str) -> pd.DataFrame:
        """
        Predict the category of a series (useful for series with missing categories).
        
        Args:
            series_id: The series ID to predict category for
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_graph() first")
        
        print(f"Predicting category for series: {series_id}")
        
        # PQL query to predict category
        query = f"PREDICT series.category FOR series.series_id='{series_id}'"
        
        try:
            result = self.model.predict(query)
            return result
        except Exception as e:
            print(f"Category prediction failed: {e}")
            return pd.DataFrame()
    
    def generate_followup_queries(self, initial_query: str, results: pd.DataFrame, max_queries: int = 3) -> List[str]:
        """Generate intelligent follow-up queries based on initial recommendations.
        
        Args:
            initial_query: The original search query
            results: DataFrame with recommendation results
            max_queries: Maximum number of follow-up queries to generate
            
        Returns:
            List of follow-up query strings
        """
        followups = []
        
        if len(results) == 0:
            return followups
        
        # Strategy 1: Extract categories from recommendations
        categories = results['category'].value_counts()
        
        # If results span multiple categories, drill into each
        if len(categories) > 1:
            # Get top 2 most common categories (excluding the most obvious one)
            for cat in categories.index[1:min(3, len(categories))]:
                # Generate query combining category with original terms
                followups.append(f"{initial_query} {cat.lower()}")
        
        # Strategy 2: Extract keywords from titles
        title_words = set()
        for title in results['title'].head(10):
            # Extract meaningful words (longer than 3 chars, not common words)
            words = [w.lower() for w in title.split() 
                    if len(w) > 3 and w.lower() not in ['rate', 'index', 'total', 'united', 'states']]
            title_words.update(words[:2])  # Top 2 words per title
        
        # Find common themes
        common_themes = []
        theme_keywords = {
            'mortgage': ['mortgage', 'loan', 'lending'],
            'construction': ['construction', 'building', 'permit'],
            'prices': ['price', 'cost', 'value'],
            'sales': ['sales', 'sold', 'transaction'],
            'inventory': ['inventory', 'stock', 'supply'],
            'employment': ['employment', 'worker', 'labor', 'jobs']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(kw in title_words for kw in keywords):
                if theme.lower() not in initial_query.lower():
                    common_themes.append(theme)
        
        # Add theme-based queries
        for theme in common_themes[:max_queries - len(followups)]:
            followups.append(f"{initial_query} {theme}")
        
        # Strategy 3: If we have few followups, suggest related economic indicators
        if len(followups) < max_queries:
            related_concepts = {
                'housing': ['mortgage rates', 'home sales', 'construction'],
                'employment': ['unemployment', 'wages', 'labor force'],
                'inflation': ['prices', 'consumer price index', 'producer price'],
                'gdp': ['economic growth', 'productivity', 'output'],
                'interest': ['federal funds rate', 'treasury rates', 'yields']
            }
            
            query_lower = initial_query.lower()
            for concept, related in related_concepts.items():
                if concept in query_lower:
                    for rel in related:
                        if rel not in query_lower and len(followups) < max_queries:
                            followups.append(rel)
        
        return followups[:max_queries]
    
    def explore_tiers(self, initial_query: str, top_k: int = 10, max_tiers: int = 3, 
                     custom_queries: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Perform multi-tier exploration starting from an initial query.
        
        Args:
            initial_query: Starting search query
            top_k: Number of recommendations per tier
            max_tiers: Maximum exploration depth
            custom_queries: Optional list of custom follow-up queries
            
        Returns:
            Dictionary mapping query to results for each tier
        """
        all_results = {}
        
        # Tier 1: Initial query
        print(f"\n{'='*80}")
        print(f"TIER 1: {initial_query}")
        print(f"{'='*80}")
        
        tier1_results = self.recommend_by_text(initial_query, top_k)
        all_results[f"Tier 1: {initial_query}"] = tier1_results
        
        if len(tier1_results) == 0:
            print("No results found. Stopping exploration.")
            return all_results
        
        # Display tier 1 results
        self._display_tier_results(tier1_results, 1)
        
        # Generate or use custom follow-up queries
        if custom_queries:
            followup_queries = custom_queries[:max_tiers-1]
        else:
            print("\n[INFO] Analyzing results to generate follow-up queries...")
            followup_queries = self.generate_followup_queries(initial_query, tier1_results, max_tiers-1)
        
        if not followup_queries:
            print("\n[INFO] No additional exploration paths identified.")
            return all_results
        
        print("\nFollow-up exploration paths:")
        for i, q in enumerate(followup_queries, 2):
            print(f"   Tier {i}: {q}")
        
        # Execute follow-up tiers
        for tier_num, query in enumerate(followup_queries, 2):
            if tier_num > max_tiers:
                break
                
            print(f"\n{'='*80}")
            print(f"TIER {tier_num}: {query}")
            print(f"{'='*80}")
            
            tier_results = self.recommend_by_text(query, top_k)
            all_results[f"Tier {tier_num}: {query}"] = tier_results
            
            if len(tier_results) > 0:
                self._display_tier_results(tier_results, tier_num)
            else:
                print(f"No results found for '{query}'.")
        
        # Summary
        print(f"\n{'='*80}")
        print("EXPLORATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total tiers explored: {len(all_results)}")
        print(f"Total unique series found: {len(self._get_unique_series(all_results))}")
        
        return all_results
    
    def _display_tier_results(self, results: pd.DataFrame, tier_num: int):
        """Display results for a single tier."""
        print(f"\nTop {len(results)} recommendations:")
        print("-" * 80)
        
        for idx, row in results.head(5).iterrows():  # Show top 5 per tier
            print(f"\n{row['series_id']}")
            print(f"  {row['title'][:70]}")
            print(f"  Category: {row['category']} | Frequency: {row['frequency_name']}")
            if 'kumo_score' in row:
                print(f"  Score: {row['kumo_score']:.2f} | Popularity: {row['actual_popularity']:.0f}")
        
        if len(results) > 5:
            print(f"\n... and {len(results) - 5} more")
    
    def _get_unique_series(self, all_results: Dict[str, pd.DataFrame]) -> set:
        """Get unique series IDs across all tiers."""
        unique = set()
        for results in all_results.values():
            if len(results) > 0:
                unique.update(results['series_id'].tolist())
        return unique
    
    def recommend_by_text(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Recommend series using Kumo RFM based on text query.
        
        Uses Kumo's graph to find series most relevant to the query terms.
        
        Args:
            query: Free text query (e.g., "inflation housing market")
            top_k: Number of recommendations to return
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_graph() first")
        
        print(f"Kumo RFM recommendations for: '{query}'")
        
        # Extract keywords from query
        terms = query.lower().split()
        
        # Strategy: Use Kumo to predict popularity for series matching query terms
        # Filter series that match any term
        series_df = self.graph.tables['series']._data
        
        # Find matching series
        mask = pd.Series(False, index=series_df.index)
        for term in terms:
            mask |= series_df['title'].str.lower().str.contains(term, na=False)
            mask |= series_df['category'].str.lower().str.contains(term, na=False)
        
        matching_ids = series_df[mask]['series_id'].tolist()
        
        if not matching_ids:
            print(f"No series found matching: {query}")
            return pd.DataFrame()
        
        # Use Kumo to predict popularity for matching series and rank
        try:
            ids_str = ', '.join([f"'{sid}'" for sid in matching_ids[:100]])  # Limit to 100 for API
            pql_query = f"PREDICT series.popularity FOR series.series_id IN ({ids_str})"
            
            result = self.model.predict(pql_query)
            
            # Debug: Check result structure
            if result.empty:
                raise ValueError("Prediction returned empty DataFrame")
            
            # Check what columns are actually in result
            if 'series_id' not in result.columns:
                # Try to find the ID column - Kumo often returns 'ENTITY'
                if 'ENTITY' in result.columns:
                    result = result.rename(columns={'ENTITY': 'series_id'})
                else:
                    id_cols = [c for c in result.columns if 'id' in c.lower() or c == 'series.series_id']
                    if id_cols:
                        result = result.rename(columns={id_cols[0]: 'series_id'})
                    else:
                        raise KeyError(f"No series_id column found. Available columns: {result.columns.tolist()}")
            
            # Merge with original metadata and deduplicate
            recommendations = series_df[series_df['series_id'].isin(result['series_id'])].copy()
            
            # Handle popularity column naming - Kumo often returns 'TARGET_PRED'
            pop_col = 'popularity'
            if 'popularity' not in result.columns:
                if 'TARGET_PRED' in result.columns:
                    pop_col = 'TARGET_PRED'
                else:
                    pop_cols = [c for c in result.columns if 'popularity' in c.lower() or 'target' in c.lower() or 'pred' in c.lower()]
                    if pop_cols:
                        pop_col = pop_cols[0]
            
            # Rename Kumo column before merge to avoid suffix confusion
            result_for_merge = result[['series_id', pop_col]].rename(columns={pop_col: 'kumo_prediction'})
            
            recommendations = recommendations.merge(result_for_merge, on='series_id')
            
            # Remove duplicates (keep first occurrence)
            recommendations = recommendations.drop_duplicates(subset=['series_id'], keep='first')
            
            # Rank by Kumo prediction
            recommendations = recommendations.nlargest(top_k, 'kumo_prediction')
            
            return recommendations[['series_id', 'title', 'category', 'frequency_name', 
                                   'popularity', 'kumo_prediction']].rename(
                columns={'popularity': 'actual_popularity', 
                        'kumo_prediction': 'kumo_score'})
            
        except Exception as e:
            print(f"Kumo prediction failed: {e}")
            # Fallback to simple filtering with deduplication
            fallback = series_df[mask].drop_duplicates(subset=['series_id'], keep='first')
            return fallback.nlargest(top_k, 'popularity')[
                ['series_id', 'title', 'category', 'frequency_name', 'popularity']
            ]
    
    def recommend_similar_series(self, series_id: str, top_k: int = 10) -> pd.DataFrame:
        """
        Recommend series similar to a given series using Kumo RFM graph relationships.
        
        Args:
            series_id: Series ID to find similar series for
            top_k: Number of recommendations to return
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_graph() first")
        
        print(f"Finding similar series to: {series_id}")
        
        series_df = self.graph.tables['series']._data
        
        # Get source series info
        source = series_df[series_df['series_id'] == series_id]
        if len(source) == 0:
            raise ValueError(f"Series not found: {series_id}")
        
        source = source.iloc[0]
        
        # Strategy: Find series in same category and use Kumo to predict which are most relevant
        candidates = series_df[
            (series_df['category'] == source['category']) &
            (series_df['series_id'] != series_id)
        ]
        
        if len(candidates) == 0:
            print(f"No similar series found in category: {source['category']}")
            return pd.DataFrame()
        
        # Use Kumo to rank candidates
        try:
            candidate_ids = candidates['series_id'].tolist()[:100]  # Limit for API
            ids_str = ', '.join([f"'{sid}'" for sid in candidate_ids])
            pql_query = f"PREDICT series.popularity FOR series.series_id IN ({ids_str})"
            
            result = self.model.predict(pql_query)
            
            # Merge and rank with deduplication
            recommendations = candidates[candidates['series_id'].isin(result['series_id'])].copy()
            recommendations = recommendations.merge(result[['series_id', 'popularity']], 
                                                   on='series_id', 
                                                   suffixes=('_actual', '_predicted'))
            
            recommendations = recommendations.drop_duplicates(subset=['series_id'], keep='first')
            recommendations = recommendations.nlargest(top_k, 'popularity_predicted')
            
            return recommendations[['series_id', 'title', 'category', 'frequency_name', 
                                   'popularity_actual', 'popularity_predicted']].rename(
                columns={'popularity_actual': 'actual_popularity', 
                        'popularity_predicted': 'kumo_score'})
            
        except Exception as e:
            print(f"Kumo prediction failed: {e}")
            # Fallback to popularity with deduplication
            candidates = candidates.drop_duplicates(subset=['series_id'], keep='first')
            return candidates.nlargest(top_k, 'popularity')[
                ['series_id', 'title', 'category', 'frequency_name', 'popularity']
            ]
    
    def recommend_by_category(self, category: str, top_k: int = 10) -> pd.DataFrame:
        """
        Recommend top series in a category using Kumo RFM predictions.
        
        Args:
            category: Category name (partial match)
            top_k: Number of recommendations
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_graph() first")
        
        print(f"Kumo RFM recommendations for category: {category}")
        
        series_df = self.graph.tables['series']._data
        
        # Find matching categories
        matches = series_df[series_df['category'].str.contains(category, case=False, na=False)]
        
        if len(matches) == 0:
            print(f"No series found in category: {category}")
            return pd.DataFrame()
        
        # Use Kumo to predict best series in category
        try:
            match_ids = matches['series_id'].tolist()[:100]  # Limit for API
            ids_str = ', '.join([f"'{sid}'" for sid in match_ids])
            pql_query = f"PREDICT series.popularity FOR series.series_id IN ({ids_str})"
            
            result = self.model.predict(pql_query)
            
            # Merge and rank with deduplication
            recommendations = matches[matches['series_id'].isin(result['series_id'])].copy()
            recommendations = recommendations.merge(result[['series_id', 'popularity']], 
                                                   on='series_id', 
                                                   suffixes=('_actual', '_predicted'))
            
            recommendations = recommendations.drop_duplicates(subset=['series_id'], keep='first')
            recommendations = recommendations.nlargest(top_k, 'popularity_predicted')
            
            return recommendations[['series_id', 'title', 'category', 'frequency_name', 
                                   'popularity_actual', 'popularity_predicted']].rename(
                columns={'popularity_actual': 'actual_popularity', 
                        'popularity_predicted': 'kumo_score'})
            
        except Exception as e:
            print(f"Kumo prediction failed: {e}")
            # Fallback to popularity with deduplication
            matches = matches.drop_duplicates(subset=['series_id'], keep='first')
            return matches.nlargest(top_k, 'popularity')[
                ['series_id', 'title', 'category', 'frequency_name', 'popularity']
            ]
    
    def recommend_multi_series(self, series_ids: List[str], top_k: int = 10) -> pd.DataFrame:
        """
        Recommend additional series based on multiple input series.
        Uses Kumo to find series that "complete" the set.
        
        Args:
            series_ids: List of series IDs user is analyzing
            top_k: Number of recommendations
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_graph() first")
        
        print(f"Finding complementary series for: {', '.join(series_ids)}")
        
        series_df = self.graph.tables['series']._data
        
        # Get categories of input series
        input_series = series_df[series_df['series_id'].isin(series_ids)]
        categories = input_series['category'].unique()
        
        # Find candidates from same categories (but not in input list)
        candidates = series_df[
            series_df['category'].isin(categories) &
            (~series_df['series_id'].isin(series_ids))
        ]
        
        if len(candidates) == 0:
            print("No complementary series found")
            return pd.DataFrame()
        
        # Use Kumo to predict which would be most valuable additions
        try:
            candidate_ids = candidates['series_id'].tolist()[:100]  # Limit for API
            ids_str = ', '.join([f"'{sid}'" for sid in candidate_ids])
            pql_query = f"PREDICT series.popularity FOR series.series_id IN ({ids_str})"
            
            result = self.model.predict(pql_query)
            
            # Merge and rank with deduplication
            recommendations = candidates[candidates['series_id'].isin(result['series_id'])].copy()
            recommendations = recommendations.merge(result[['series_id', 'popularity']], 
                                                   on='series_id', 
                                                   suffixes=('_actual', '_predicted'))
            
            recommendations = recommendations.drop_duplicates(subset=['series_id'], keep='first')
            recommendations = recommendations.nlargest(top_k, 'popularity_predicted')
            
            return recommendations[['series_id', 'title', 'category', 'frequency_name', 
                                   'popularity_actual', 'popularity_predicted']].rename(
                columns={'popularity_actual': 'actual_popularity', 
                        'popularity_predicted': 'kumo_score'})
            
        except Exception as e:
            print(f"Kumo prediction failed: {e}")
            # Fallback to popularity with deduplication
            candidates = candidates.drop_duplicates(subset=['series_id'], keep='first')
            return candidates.nlargest(top_k, 'popularity')[
                ['series_id', 'title', 'category', 'frequency_name', 'popularity']
            ]
    
    def explain_recommendation(self, series_id: str, recommended_id: str) -> Dict:
        """
        Explain why a series was recommended.
        
        Args:
            series_id: Original series (if any)
            recommended_id: Recommended series to explain
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_graph() first")
        
        series_df = self.graph.tables['series']._data
        
        original = series_df[series_df['series_id'] == series_id].iloc[0] if series_id else None
        recommended = series_df[series_df['series_id'] == recommended_id].iloc[0]
        
        explanation = {
            'recommended_series': recommended_id,
            'title': recommended['title'],
            'category': recommended['category'],
            'popularity': recommended['popularity'],
            'reasons': []
        }
        
        if original is not None:
            # Compare with original
            if original['category'] == recommended['category']:
                explanation['reasons'].append(f"Same category: {recommended['category']}")
            if original['frequency'] == recommended['frequency']:
                explanation['reasons'].append(f"Same frequency: {recommended['frequency']}")
            
        # Add Kumo-specific reasoning
        explanation['reasons'].append(f"Kumo RFM graph relationships")
        explanation['reasons'].append(f"Predicted relevance score")
        
        return explanation
    
    def batch_predict_popularity(self, series_ids: List[str]) -> pd.DataFrame:
        """
        Predict popularity for multiple series at once.
        
        Args:
            series_ids: List of series IDs to predict popularity for
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_graph() first")
        
        print(f"Predicting popularity for {len(series_ids)} series...")
        
        # Format series IDs for PQL IN clause
        ids_str = ', '.join([f"'{sid}'" for sid in series_ids])
        query = f"PREDICT series.popularity FOR series.series_id IN ({ids_str})"
        
        try:
            result = self.model.predict(query)
            return result
        except Exception as e:
            print(f"Batch prediction failed: {e}")
            return pd.DataFrame()
    
    def visualize_category_distribution(self, series_df: pd.DataFrame, output_path: str = None):
        """
        Visualize the distribution of series across categories.
        
        Args:
            series_df: DataFrame with series metadata
            output_path: Optional path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top categories by count
        top_categories = series_df['category'].value_counts().head(15)
        ax1.barh(range(len(top_categories)), top_categories.values)
        ax1.set_yticks(range(len(top_categories)))
        ax1.set_yticklabels(top_categories.index, fontsize=9)
        ax1.set_xlabel('Number of Series')
        ax1.set_title('Top 15 Categories by Series Count')
        ax1.invert_yaxis()
        
        # Category popularity distribution
        category_popularity = series_df.groupby('category')['popularity'].mean().sort_values(ascending=False).head(15)
        ax2.barh(range(len(category_popularity)), category_popularity.values, color='coral')
        ax2.set_yticks(range(len(category_popularity)))
        ax2.set_yticklabels(category_popularity.index, fontsize=9)
        ax2.set_xlabel('Average Popularity')
        ax2.set_title('Top 15 Categories by Average Popularity')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved category distribution to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_frequency_analysis(self, series_df: pd.DataFrame, output_path: str = None):
        """
        Visualize series distribution by update frequency.
        
        Args:
            series_df: DataFrame with series metadata
            output_path: Optional path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Frequency distribution
        freq_counts = series_df['frequency_name'].value_counts()
        colors = sns.color_palette('husl', len(freq_counts))
        ax1.pie(freq_counts.values, labels=freq_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Series Distribution by Update Frequency')
        
        # Popularity by frequency
        freq_popularity = series_df.groupby('frequency_name')['popularity'].mean().sort_values(ascending=False)
        ax2.bar(range(len(freq_popularity)), freq_popularity.values, color=colors[:len(freq_popularity)])
        ax2.set_xticks(range(len(freq_popularity)))
        ax2.set_xticklabels(freq_popularity.index, rotation=45, ha='right')
        ax2.set_ylabel('Average Popularity')
        ax2.set_title('Average Popularity by Frequency')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved frequency analysis to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_popularity_distribution(self, series_df: pd.DataFrame, output_path: str = None):
        """
        Visualize the distribution of series popularity scores.
        
        Args:
            series_df: DataFrame with series metadata
            output_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram of popularity
        axes[0, 0].hist(series_df['popularity'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Popularity Score')
        axes[0, 0].set_ylabel('Number of Series')
        axes[0, 0].set_title('Distribution of Popularity Scores')
        axes[0, 0].axvline(series_df['popularity'].median(), color='red', linestyle='--', label=f'Median: {series_df["popularity"].median():.2f}')
        axes[0, 0].legend()
        
        # Log-scale popularity (for heavily skewed data)
        log_pop = np.log10(series_df['popularity'] + 1)  # +1 to avoid log(0)
        axes[0, 1].hist(log_pop, bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Log10(Popularity + 1)')
        axes[0, 1].set_ylabel('Number of Series')
        axes[0, 1].set_title('Popularity Distribution (Log Scale)')
        
        # Box plot by top categories
        top_cats = series_df['category'].value_counts().head(8).index
        filtered = series_df[series_df['category'].isin(top_cats)]
        filtered.boxplot(column='popularity', by='category', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Category')
        axes[1, 0].set_ylabel('Popularity')
        axes[1, 0].set_title('Popularity Distribution by Top 8 Categories')
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.sca(axes[1, 0])
        plt.xticks(rotation=45, ha='right')
        
        # Cumulative distribution
        sorted_pop = np.sort(series_df['popularity'])
        cumulative = np.arange(1, len(sorted_pop) + 1) / len(sorted_pop) * 100
        axes[1, 1].plot(sorted_pop, cumulative, linewidth=2)
        axes[1, 1].set_xlabel('Popularity Score')
        axes[1, 1].set_ylabel('Cumulative Percentage (%)')
        axes[1, 1].set_title('Cumulative Distribution of Popularity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved popularity distribution to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_rfm_segments(self, series_df: pd.DataFrame, output_path: str = None):
        """
        Create RFM-style segmentation visualization for series metadata.
        Uses popularity, frequency, and category as proxies for RFM dimensions.
        
        Args:
            series_df: DataFrame with series metadata
            output_path: Optional path to save the figure
        """
        # Create RFM-style scores
        df = series_df.copy()
        
        # Recency proxy: Use first letter of series_id or random for demo
        df['recency_score'] = pd.qcut(df['popularity'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
        
        # Frequency proxy: Map frequency to numeric score
        freq_map = {'Daily': 5, 'Weekly': 4, 'Monthly': 3, 'Quarterly': 2, 'Annual': 1}
        df['frequency_score'] = df['frequency_name'].map(freq_map).fillna(3)
        
        # Monetary proxy: Use popularity
        df['monetary_score'] = pd.qcut(df['popularity'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
        
        # Convert to numeric
        df['recency_score'] = pd.to_numeric(df['recency_score'])
        df['monetary_score'] = pd.to_numeric(df['monetary_score'])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # RFM Score distribution
        df['rfm_score'] = df['recency_score'] + df['frequency_score'] + df['monetary_score']
        axes[0, 0].hist(df['rfm_score'], bins=20, edgecolor='black', alpha=0.7, color='purple')
        axes[0, 0].set_xlabel('RFM Score')
        axes[0, 0].set_ylabel('Number of Series')
        axes[0, 0].set_title('Distribution of RFM Scores')
        
        # Scatter: Recency vs Monetary
        scatter = axes[0, 1].scatter(df['recency_score'], df['monetary_score'], 
                                     c=df['frequency_score'], cmap='viridis', alpha=0.6, s=30)
        axes[0, 1].set_xlabel('Recency Score')
        axes[0, 1].set_ylabel('Monetary Score')
        axes[0, 1].set_title('RFM Segment Map (colored by Frequency)')
        plt.colorbar(scatter, ax=axes[0, 1], label='Frequency Score')
        
        # Segment categories
        df['segment'] = 'Regular'
        df.loc[(df['recency_score'] >= 4) & (df['monetary_score'] >= 4), 'segment'] = 'Champions'
        df.loc[(df['recency_score'] >= 3) & (df['monetary_score'] >= 3) & (df['segment'] != 'Champions'), 'segment'] = 'Loyal'
        df.loc[(df['recency_score'] <= 2) & (df['monetary_score'] >= 4), 'segment'] = 'At Risk'
        df.loc[(df['recency_score'] <= 2) & (df['monetary_score'] <= 2), 'segment'] = 'Lost'
        
        segment_counts = df['segment'].value_counts()
        axes[1, 0].bar(range(len(segment_counts)), segment_counts.values, color=sns.color_palette('Set2', len(segment_counts)))
        axes[1, 0].set_xticks(range(len(segment_counts)))
        axes[1, 0].set_xticklabels(segment_counts.index, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Number of Series')
        axes[1, 0].set_title('Series Count by RFM Segment')
        
        # Heatmap: Average popularity by frequency and category (top categories)
        top_cats = df['category'].value_counts().head(5).index
        heatmap_data = df[df['category'].isin(top_cats)].pivot_table(
            values='popularity', 
            index='category', 
            columns='frequency_name', 
            aggfunc='mean'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': 'Avg Popularity'})
        axes[1, 1].set_title('Popularity Heatmap: Top 5 Categories vs Frequency')
        axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].set_ylabel('Category')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved RFM segmentation to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return df[['series_id', 'segment', 'rfm_score']]
    
    def visualize_relationship_graph(self, relationships_df: pd.DataFrame, 
                                    series_df: pd.DataFrame = None,
                                    max_nodes: int = 100,
                                    output_path: str = None,
                                    recommended_series: List[str] = None,
                                    mode: str = 'full'):
        """Visualize the series relationship graph using NetworkX.
        
        Args:
            relationships_df: DataFrame with relationship edges
            series_df: Optional DataFrame with series metadata for node labels
            max_nodes: Maximum number of nodes to display
            output_path: Optional path to save the figure
            recommended_series: List of recommended series IDs to highlight
            mode: Visualization mode - 'full', 'highlight', 'subgraph', or 'paths'
                  - 'full': Show full graph (default)
                  - 'highlight': Full graph with recommended series highlighted
                  - 'subgraph': Only show neighborhood of recommended series
                  - 'paths': Show paths connecting recommended series
        """
        mode_names = {
            'full': 'Full Graph',
            'highlight': 'Highlighted Recommendations',
            'subgraph': 'Recommendation Subgraph',
            'paths': 'Recommendation Paths'
        }
        print(f"\nVisualizing relationship graph ({mode_names.get(mode, mode)})...")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add edges from relationships
        for _, row in relationships_df.iterrows():
            G.add_edge(row['source_id'], row['target_id'], 
                      relationship_type=row['relationship_type'],
                      strength=row['strength'])
        
        print(f"Full graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        # Apply mode-specific filtering
        if mode == 'subgraph' and recommended_series:
            # Filter to neighborhood of recommended series
            print(f"Filtering to {len(recommended_series)} recommended series + neighbors...")
            subgraph_nodes = set(recommended_series)
            
            # Add 2-hop neighborhood
            for rec_id in recommended_series:
                if rec_id in G:
                    neighbors = nx.single_source_shortest_path_length(G, rec_id, cutoff=2)
                    subgraph_nodes.update(neighbors.keys())
            
            G = G.subgraph(subgraph_nodes).copy()
            print(f"Subgraph: {len(G.nodes())} nodes, {len(G.edges())} edges")
            
        elif mode == 'paths' and recommended_series and len(recommended_series) >= 2:
            # Show paths between recommended series
            print(f"Computing paths between {len(recommended_series)} recommended series...")
            path_nodes = set(recommended_series)
            path_edges = set()
            
            # Find shortest paths between all pairs of recommendations
            for i, source in enumerate(recommended_series):
                for target in recommended_series[i+1:]:
                    if source in G and target in G:
                        try:
                            path = nx.shortest_path(G, source, target)
                            path_nodes.update(path)
                            # Add edges in path
                            for j in range(len(path)-1):
                                path_edges.add((path[j], path[j+1]))
                        except nx.NetworkXNoPath:
                            pass
            
            # Create graph from paths
            G_paths = nx.Graph()
            G_paths.add_nodes_from(path_nodes)
            G_paths.add_edges_from(path_edges)
            G = G_paths
            print(f"Path graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        # General size filtering (if still too large)
        elif len(G.nodes()) > max_nodes:
            print(f"Filtering to top {max_nodes} most connected nodes...")
            # Prioritize recommended series
            if recommended_series:
                # Keep all recommended series
                priority_nodes = set([n for n in recommended_series if n in G])
                # Add most connected remaining nodes
                degrees = dict(G.degree())
                remaining_nodes = [n for n in G.nodes() if n not in priority_nodes]
                top_remaining = sorted(remaining_nodes, key=lambda x: degrees.get(x, 0), reverse=True)
                top_nodes = list(priority_nodes) + top_remaining[:max_nodes - len(priority_nodes)]
            else:
                degrees = dict(G.degree())
                top_nodes_list = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                top_nodes = [node for node, _ in top_nodes_list]
            
            G = G.subgraph(top_nodes).copy()
            print(f"Filtered graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left plot: Full graph
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Node sizes based on degree
        node_sizes = [300 + (G.degree(node) * 50) for node in G.nodes()]
        
        # Color nodes based on mode
        if mode in ['highlight', 'subgraph', 'paths'] and recommended_series:
            # Highlight recommended series
            node_colors = []
            for node in G.nodes():
                if node in recommended_series:
                    node_colors.append('#ff4444')  # Red for recommended
                else:
                    node_colors.append('#4444ff')  # Blue for others
            
            # Make recommended nodes larger
            node_sizes = [600 if node in recommended_series else 300 + (G.degree(node) * 30) 
                         for node in G.nodes()]
            
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                                  alpha=0.8, ax=ax1, edgecolors='black', linewidths=2)
        else:
            # Color by degree (default)
            degrees_list = [G.degree(node) for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=degrees_list,
                                  cmap='YlOrRd', alpha=0.8, ax=ax1)
        
        # Edge styling based on mode
        if mode == 'paths' and recommended_series:
            # Thicker edges for paths
            nx.draw_networkx_edges(G, pos, alpha=0.5, width=2.0, edge_color='#666666', ax=ax1)
        else:
            nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax1)
        
        # Add labels - prioritize recommended series
        if recommended_series:
            # Always label recommended series
            labels = {node: node for node in G.nodes() if node in recommended_series}
            # Add high-degree nodes if not too many
            if len(labels) < 20:
                high_degree = [node for node in G.nodes() if G.degree(node) >= 8 and node not in recommended_series]
                labels.update({node: node for node in high_degree[:20-len(labels)]})
        else:
            # Default: label high-degree nodes
            high_degree_nodes = [node for node in G.nodes() if G.degree(node) >= 5]
            labels = {node: node for node in high_degree_nodes}
        
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_weight='bold', ax=ax1)
        
        # Title based on mode
        title = f"Series Relationship Graph - {mode_names.get(mode, mode)}\n{len(G.nodes())} nodes, {len(G.edges())} edges"
        if recommended_series:
            title += f" | {len([n for n in recommended_series if n in G.nodes()])} recommended series"
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Right plot: Statistics
        ax2.axis('off')
        
        # Compute graph statistics
        stats_text = []
        stats_text.append("GRAPH STATISTICS")
        stats_text.append("=" * 50)
        stats_text.append(f"Nodes: {len(G.nodes())}")
        stats_text.append(f"Edges: {len(G.edges())}")
        stats_text.append(f"Density: {nx.density(G):.4f}")
        stats_text.append(f"Average degree: {sum(dict(G.degree()).values()) / len(G.nodes()):.2f}")
        
        # Connected components
        components = list(nx.connected_components(G))
        stats_text.append(f"\nConnected components: {len(components)}")
        if len(components) > 0:
            largest_cc = max(components, key=len)
            stats_text.append(f"Largest component: {len(largest_cc)} nodes")
        
        # Clustering coefficient
        avg_clustering = nx.average_clustering(G)
        stats_text.append(f"\nAverage clustering: {avg_clustering:.4f}")
        
        # Degree distribution
        stats_text.append("\nTOP 10 MOST CONNECTED SERIES:")
        stats_text.append("-" * 50)
        top_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
        for node, degree in top_degree:
            # Get title if series_df provided
            if series_df is not None:
                title = series_df[series_df['series_id'] == node]['title'].values
                title_str = title[0][:30] if len(title) > 0 else node
            else:
                title_str = node
            stats_text.append(f"{node:15s} ({degree:2d} connections)")
            if series_df is not None and len(title) > 0:
                stats_text.append(f"  {title_str}")
        
        # Recommended series info
        if recommended_series:
            stats_text.append("\nRECOMMENDED SERIES:")
            stats_text.append("-" * 50)
            visible_recs = [r for r in recommended_series if r in G.nodes()]
            stats_text.append(f"Total recommended: {len(recommended_series)}")
            stats_text.append(f"Visible in graph: {len(visible_recs)}")
            stats_text.append("\nTop 5 Recommended:")
            for rec_id in visible_recs[:5]:
                if series_df is not None:
                    title = series_df[series_df['series_id'] == rec_id]['title'].values
                    if len(title) > 0:
                        stats_text.append(f"  {rec_id}: {title[0][:35]}...")
                    else:
                        stats_text.append(f"  {rec_id}")
                else:
                    stats_text.append(f"  {rec_id}")
        
        # Relationship type distribution
        stats_text.append("\nRELATIONSHIP TYPES:")
        stats_text.append("-" * 50)
        rel_types = relationships_df['relationship_type'].value_counts().head(10)
        for rel_type, count in rel_types.items():
            stats_text.append(f"{rel_type:25s}: {count:5d}")
        
        # Display statistics
        ax2.text(0.1, 0.95, '\n'.join(stats_text), 
                transform=ax2.transAxes, 
                fontsize=9, 
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved relationship graph to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return G
    
    def visualize_prediction_comparison(self, predictions_df: pd.DataFrame, actual_col: str = 'actual', 
                                       predicted_col: str = 'predicted', output_path: str = None):
        """
        Visualize comparison between predicted and actual values.
        
        Args:
            predictions_df: DataFrame with actual and predicted values
            actual_col: Column name for actual values
            predicted_col: Column name for predicted values
            output_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot: Predicted vs Actual
        axes[0].scatter(predictions_df[actual_col], predictions_df[predicted_col], alpha=0.5)
        
        # Perfect prediction line
        min_val = min(predictions_df[actual_col].min(), predictions_df[predicted_col].min())
        max_val = max(predictions_df[actual_col].max(), predictions_df[predicted_col].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Predicted vs Actual Values')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = predictions_df[predicted_col] - predictions_df[actual_col]
        axes[1].scatter(predictions_df[actual_col], residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Actual Values')
        axes[1].set_ylabel('Residuals (Predicted - Actual)')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved prediction comparison to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_visualization_dashboard(self, series_df: pd.DataFrame, output_dir: str = 'visualizations'):
        """
        Create a complete dashboard of all visualizations.
        
        Args:
            series_df: DataFrame with series metadata
            output_dir: Directory to save all visualizations
        """
        # Create date-based subdirectory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        date_dir = datetime.now().strftime('%Y-%m-%d')
        full_output_dir = os.path.join(output_dir, date_dir)
        os.makedirs(full_output_dir, exist_ok=True)
        
        print(f"\nGenerating visualization dashboard in {full_output_dir}/...")
        
        # Generate all visualizations
        print("  • Creating category distribution...")
        self.visualize_category_distribution(series_df, f"{full_output_dir}/category_distribution_{timestamp}.png")
        
        print("  • Creating frequency analysis...")
        self.visualize_frequency_analysis(series_df, f"{full_output_dir}/frequency_analysis_{timestamp}.png")
        
        print("  • Creating popularity distribution...")
        self.visualize_popularity_distribution(series_df, f"{full_output_dir}/popularity_distribution_{timestamp}.png")
        
        print("  • Creating RFM segmentation...")
        rfm_segments = self.visualize_rfm_segments(series_df, f"{full_output_dir}/rfm_segmentation_{timestamp}.png")
        
        # Save segment data
        rfm_segments.to_csv(f"{full_output_dir}/rfm_segments_{timestamp}.csv", index=False)
        print(f"  • Saved RFM segment data to {full_output_dir}/rfm_segments_{timestamp}.csv")
        
        print(f"\n[OK] Dashboard created successfully in {full_output_dir}/")
        print(f"   Date: {date_dir}")
        print(f"   Time: {timestamp}")
        return full_output_dir
    
    def save_relationships_to_db(self, relationships_df: pd.DataFrame, 
                                series_df: pd.DataFrame,
                                db_connection) -> bool:
        """Save relationship graph to PostgreSQL for monolith feed.
        
        Creates two tables:
        1. monolith_series_relationships - The edge table
        2. monolith_series_features - Node features for recommendation
        
        Args:
            relationships_df: DataFrame with relationship edges
            series_df: DataFrame with series metadata
            db_connection: psycopg2 connection object
        
        Returns:
            True if successful
        """
        try:
            import psycopg2
            cursor = db_connection.cursor()
            
            print("\nSaving relationships to PostgreSQL...")
            
            # Create relationships table
            print("  Creating monolith_series_relationships table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS monolith_series_relationships (
                    relationship_id VARCHAR(50) PRIMARY KEY,
                    source_id VARCHAR(50) NOT NULL,
                    target_id VARCHAR(50) NOT NULL,
                    relationship_type VARCHAR(50) NOT NULL,
                    strength FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_id, target_id, relationship_type)
                );
                
                CREATE INDEX IF NOT EXISTS idx_monolith_rel_source 
                    ON monolith_series_relationships(source_id);
                CREATE INDEX IF NOT EXISTS idx_monolith_rel_target 
                    ON monolith_series_relationships(target_id);
                CREATE INDEX IF NOT EXISTS idx_monolith_rel_type 
                    ON monolith_series_relationships(relationship_type);
                CREATE INDEX IF NOT EXISTS idx_monolith_rel_strength 
                    ON monolith_series_relationships(strength DESC);
            """)
            
            # Create series features table
            print("  Creating monolith_series_features table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS monolith_series_features (
                    series_id VARCHAR(50) PRIMARY KEY,
                    title TEXT,
                    category VARCHAR(100),
                    frequency VARCHAR(50),
                    popularity FLOAT,
                    degree_centrality FLOAT,
                    betweenness_centrality FLOAT,
                    clustering_coefficient FLOAT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_monolith_feat_category 
                    ON monolith_series_features(category);
                CREATE INDEX IF NOT EXISTS idx_monolith_feat_popularity 
                    ON monolith_series_features(popularity DESC);
                CREATE INDEX IF NOT EXISTS idx_monolith_feat_degree 
                    ON monolith_series_features(degree_centrality DESC);
            """)
            
            db_connection.commit()
            
            # Insert relationships (upsert to handle updates)
            print(f"  Inserting {len(relationships_df)} relationships...")
            for _, row in relationships_df.iterrows():
                cursor.execute("""
                    INSERT INTO monolith_series_relationships 
                    (relationship_id, source_id, target_id, relationship_type, strength)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (source_id, target_id, relationship_type) 
                    DO UPDATE SET 
                        strength = EXCLUDED.strength,
                        created_at = CURRENT_TIMESTAMP
                """, (
                    row['relationship_id'],
                    row['source_id'],
                    row['target_id'],
                    row['relationship_type'],
                    float(row['strength'])
                ))
            
            # Compute graph centrality metrics
            print("  Computing graph metrics...")
            G = nx.Graph()
            for _, row in relationships_df.iterrows():
                G.add_edge(row['source_id'], row['target_id'], weight=row['strength'])
            
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
            clustering = nx.clustering(G)
            
            # Insert series features
            print(f"  Inserting features for {len(series_df)} series...")
            for _, row in series_df.iterrows():
                series_id = row['series_id']
                
                cursor.execute("""
                    INSERT INTO monolith_series_features 
                    (series_id, title, category, frequency, popularity, 
                     degree_centrality, betweenness_centrality, clustering_coefficient)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (series_id) 
                    DO UPDATE SET 
                        title = EXCLUDED.title,
                        category = EXCLUDED.category,
                        frequency = EXCLUDED.frequency,
                        popularity = EXCLUDED.popularity,
                        degree_centrality = EXCLUDED.degree_centrality,
                        betweenness_centrality = EXCLUDED.betweenness_centrality,
                        clustering_coefficient = EXCLUDED.clustering_coefficient,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    series_id,
                    row.get('title', ''),
                    row.get('category', ''),
                    row.get('frequency_name', row.get('frequency', '')),
                    float(row.get('popularity', 0)),
                    float(degree_centrality.get(series_id, 0)),
                    float(betweenness_centrality.get(series_id, 0)),
                    float(clustering.get(series_id, 0))
                ))
            
            db_connection.commit()
            
            print("\n[OK] Successfully saved to database!")
            print("\nYou can now query:")
            print("  SELECT * FROM monolith_series_relationships;")
            print("  SELECT * FROM monolith_series_features ORDER BY degree_centrality DESC LIMIT 10;")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Error saving to database: {e}")
            db_connection.rollback()
            return False


def demo_workflow(data_path: str):
    """Demonstrate various Kumo RFM workflows with FRED data."""
    
    print("="*60)
    print("KUMO RFM + FRED DATA INTEGRATION DEMO")
    print("="*60)
    
    # Check for API key
    api_key = os.getenv('KUMO_API_KEY')
    if not api_key:
        print("\nNote: KUMO_API_KEY not found in environment.")
        print("Set it with: export KUMO_API_KEY='your-api-key'")
        print("Or use the authentication widget: rfm.authenticate()")
        print("\nRunning in demo mode with limited functionality...\n")
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
    print(f"Loaded {len(df)} series")
    
    # Demo 1: Build graph and make predictions
    print("\n" + "="*60)
    print("DEMO 1: Build Graph and Predict Popularity")
    print("="*60)
    
    if api_key and KUMO_AVAILABLE:
        try:
            kumo = KumoFREDIntegration(api_key)
            kumo.build_graph(df)
            
            # Predict popularity for a sample series
            sample_series = df.iloc[0]['series_id']
            print(f"\nPredicting popularity for series: {sample_series}")
            result = kumo.predict_series_popularity(sample_series)
            if not result.empty:
                print(result)
        except Exception as e:
            print(f"Error in RFM prediction: {e}")
            print("Falling back to basic recommendations...")
    
    # Demo 2: Recommend series based on category
    print("\n" + "="*60)
    print("DEMO 2: Series Recommendations (Rule-based)")
    print("="*60)
    
    recommendations = df[df['category'].str.contains('Employment', case=False, na=False)]\
        .nlargest(5, 'popularity')[['series_id', 'title', 'category', 'popularity']]
    
    print("\nTop 5 employment-related series:")
    print(recommendations.to_string(index=False))
    
    # Demo 3: Query by frequency and category
    print("\n" + "="*60)
    print("DEMO 3: Filtered Query")
    print("="*60)
    
    monthly_inflation = df[
        (df['category'].str.contains('Inflation', case=False, na=False)) &
        (df['frequency'] == 'M')
    ]
    print(f"\nMonthly inflation series: {len(monthly_inflation)}")
    if len(monthly_inflation) > 0:
        print(monthly_inflation[['series_id', 'title', 'popularity']].head().to_string(index=False))
    
    # Demo 4: Category statistics
    print("\n" + "="*60)
    print("DEMO 4: Series Statistics by Category")
    print("="*60)
    
    # Group series by category
    category_stats = df.groupby('category').agg({
        'series_id': 'count',
        'popularity': 'mean'
    }).round(2).sort_values('series_id', ascending=False)
    
    print("\nSeries count and avg popularity by category:")
    print(category_stats.head(10).to_string())


def main():
    parser = argparse.ArgumentParser(description='Kumo RFM integration for FRED data')
    parser.add_argument('--data', type=str, default='data/fred_series_metadata.parquet',
                       help='Path to FRED series data')
    parser.add_argument('--api-key', type=str, help='Kumo API key (or set KUMO_API_KEY env var)')
    
    # Recommendation modes
    parser.add_argument('--recommend', type=str, help='Text query for recommendations')
    parser.add_argument('--similar', type=str, help='Series ID to find similar series')
    parser.add_argument('--category', type=str, help='Category to explore')
    parser.add_argument('--multi', type=str, nargs='+', help='Multiple series IDs for context')
    parser.add_argument('--explain', type=str, nargs=2, metavar=('SOURCE', 'RECOMMENDED'),
                       help='Explain why series was recommended')
    
    # Multi-tier exploration
    parser.add_argument('--explore-tiers', action='store_true',
                       help='Generate follow-up recommendations based on initial results')
    parser.add_argument('--max-tiers', type=int, default=3,
                       help='Maximum number of exploration tiers (default: 3)')
    parser.add_argument('--tier-queries', type=str, nargs='+',
                       help='Custom follow-up queries for each tier')
    
    parser.add_argument('--top-k', type=int, default=10, help='Number of recommendations')
    parser.add_argument('--demo', action='store_true', help='Run demo examples')
    parser.add_argument('--output-dir', type=str, default='outputs/visualizations',
                       help='Output directory for visualizations')
    
    # New features
    parser.add_argument('--visualize-graph', action='store_true', 
                       help='Visualize relationship graph with NetworkX')
    parser.add_argument('--graph-mode', type=str, default='full',
                       choices=['full', 'highlight', 'subgraph', 'paths', 'all'],
                       help='Graph visualization mode (use "all" to generate all modes)')
    parser.add_argument('--save-to-db', action='store_true',
                       help='Save relationships to PostgreSQL database')
    parser.add_argument('--db-host', type=str, default='localhost',
                       help='PostgreSQL host')
    parser.add_argument('--db-port', type=str, default='5432',
                       help='PostgreSQL port')
    parser.add_argument('--db-name', type=str, default='postgres',
                       help='PostgreSQL database name')
    parser.add_argument('--db-user', type=str, default='postgres',
                       help='PostgreSQL username')
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ['KUMO_API_KEY'] = args.api_key
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data)
    print(f"Loaded {len(df)} series")
    
    # Handle recommendation modes or graph operations
    if args.recommend or args.similar or args.category or args.multi or args.explain or args.visualize_graph or args.save_to_db:
        try:
            # Initialize Kumo integration
            kumo = KumoFREDIntegration(api_key=args.api_key)
            kumo.build_graph(df)
            
            # Get relationships DataFrame for visualization/storage
            relationships_df = kumo._create_relationships(df)
            
            # Text query recommendations
            if args.recommend:
                if args.explore_tiers:
                    # Multi-tier exploration mode
                    all_tier_results = kumo.explore_tiers(
                        initial_query=args.recommend,
                        top_k=args.top_k,
                        max_tiers=args.max_tiers,
                        custom_queries=args.tier_queries
                    )
                    # Use tier 1 results for visualization
                    results = all_tier_results.get(f"Tier 1: {args.recommend}", pd.DataFrame())
                else:
                    # Single query mode
                    print(f"\n Kumo RFM recommendations for: '{args.recommend}'\n")
                    results = kumo.recommend_by_text(args.recommend, args.top_k)
                
            # Similar series
            elif args.similar:
                print(f"\n Similar series to: {args.similar}\n")
                results = kumo.recommend_similar_series(args.similar, args.top_k)
                
            # Category recommendations
            elif args.category:
                print(f"\n Top series in category: {args.category}\n")
                results = kumo.recommend_by_category(args.category, args.top_k)
                
            # Multi-series recommendations
            elif args.multi:
                print(f"\n Complementary series for: {', '.join(args.multi)}\n")
                results = kumo.recommend_multi_series(args.multi, args.top_k)
                
            # Explain recommendation
            elif args.explain:
                print(f"\n Explanation for recommendation\n")
                explanation = kumo.explain_recommendation(args.explain[0], args.explain[1])
                print(json.dumps(explanation, indent=2))
                return
            
            # Visualize graph if requested
            if args.visualize_graph:
                print("\n" + "="*60)
                print("VISUALIZING RELATIONSHIP GRAPH")
                print("="*60)
                
                # Create date-based subdirectory
                date_dir = datetime.now().strftime('%Y-%m-%d')
                full_output_dir = os.path.join(args.output_dir, date_dir)
                os.makedirs(full_output_dir, exist_ok=True)
                
                # Get recommended series IDs if we have recommendations
                recommended_ids = None
                if 'results' in locals() and len(results) > 0:
                    recommended_ids = results['series_id'].tolist()
                    print(f"\nVisualizing with {len(recommended_ids)} recommended series")
                
                # Generate visualization(s)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                if args.graph_mode == 'all':
                    # Generate all four modes
                    modes = ['full', 'highlight', 'subgraph', 'paths']
                    for mode in modes:
                        print(f"\nGenerating {mode} mode...")
                        output_path = f"{full_output_dir}/relationship_graph_{mode}_{timestamp}.png"
                        
                        G = kumo.visualize_relationship_graph(
                            relationships_df, 
                            series_df=df,
                            max_nodes=150,
                            output_path=output_path,
                            recommended_series=recommended_ids,
                            mode=mode
                        )
                        print(f"  Saved: {output_path}")
                else:
                    # Single mode
                    mode_suffix = f"_{args.graph_mode}" if args.graph_mode != 'full' else ""
                    output_path = f"{full_output_dir}/relationship_graph{mode_suffix}_{timestamp}.png"
                    
                    G = kumo.visualize_relationship_graph(
                        relationships_df, 
                        series_df=df,
                        max_nodes=150,
                        output_path=output_path,
                        recommended_series=recommended_ids,
                        mode=args.graph_mode
                    )
                    print(f"\n[OK] Graph saved to: {output_path}")
            
            # Save to database if requested
            if args.save_to_db:
                print("\n" + "="*60)
                print("SAVING TO DATABASE")
                print("="*60)
                
                try:
                    import psycopg2
                    from getpass import getpass
                    
                    # Get password
                    db_password = os.getenv('PGPASSWORD')
                    if not db_password:
                        db_password = getpass(f"PostgreSQL password for {args.db_user}@{args.db_host}: ")
                    
                    # Connect to database
                    conn = psycopg2.connect(
                        host=args.db_host,
                        port=args.db_port,
                        database=args.db_name,
                        user=args.db_user,
                        password=db_password
                    )
                    
                    # Save relationships
                    success = kumo.save_relationships_to_db(relationships_df, df, conn)
                    
                    conn.close()
                    
                    if success:
                        print("\nDatabase storage complete!")
                    
                except ImportError:
                    print("\n[ERROR] psycopg2 not installed. Install with: pip install psycopg2-binary")
                except Exception as e:
                    print(f"\n[ERROR] Database error: {e}")
            
            # Display recommendation results (if any)
            if args.recommend or args.similar or args.category or args.multi:
                if 'results' in locals() and len(results) > 0:
                    print("\n" + "="*100)
                    print(f"Top {len(results)} Recommendations:")
                    print("="*100)
                    for idx, row in results.iterrows():
                        print(f"\n{row['series_id']}")
                        print(f"  Title: {row['title'][:80]}")
                        print(f"  Category: {row['category']}")
                        print(f"  Frequency: {row['frequency_name']}")
                        if 'kumo_score' in row:
                            print(f"  Kumo Score: {row['kumo_score']:.2f}")
                            print(f"  Actual Popularity: {row['actual_popularity']:.0f}")
                        elif 'popularity' in row:
                            print(f"  Popularity: {row['popularity']:.0f}")
                else:
                    if not args.visualize_graph and not args.save_to_db:
                        print("No recommendations found.")
                
        except Exception as e:
            print(f"\n Error: {e}")
            print("\nMake sure:")
            print("1. KUMO_API_KEY is set (export KUMO_API_KEY='your-key')")
            print("2. kumoai package is installed (pip install kumoai)")
            print("3. Data file exists and is valid")
            return
    
    # Run demo
    elif args.demo:
        demo_workflow(args.data)
    
    # No arguments - show help
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("Quick Examples:")
        print("="*60)
        print("\n# Text query recommendations:")
        print("python3 05_kumo_rfm_integration.py --recommend 'inflation housing' --top-k 10")
        print("\n# Multi-tier exploration (auto-generate follow-ups):")
        print("python3 05_kumo_rfm_integration.py --recommend 'housing' --explore-tiers --max-tiers 3")
        print("\n# Multi-tier with custom queries:")
        print("python3 05_kumo_rfm_integration.py --recommend 'housing' --explore-tiers --tier-queries 'mortgage rates' 'home sales'")
        print("\n# Similar series:")
        print("python3 05_kumo_rfm_integration.py --similar PAYEMS --top-k 10")
        print("\n# Category exploration:")
        print("python3 05_kumo_rfm_integration.py --category Employment --top-k 10")
        print("\n# Multi-series context:")
        print("python3 05_kumo_rfm_integration.py --multi GDPC1 UNRATE --top-k 10")
        print("\n# Explain recommendation:")
        print("python3 05_kumo_rfm_integration.py --explain PAYEMS UNRATE")
        print("\n# Run demo:")
        print("python3 05_kumo_rfm_integration.py --demo")
        print("\n# Visualize relationship graph:")
        print("python3 05_kumo_rfm_integration.py --visualize-graph")
        print("\n# Visualize with recommendations (highlight mode):")
        print("python3 05_kumo_rfm_integration.py --recommend 'housing' --visualize-graph --graph-mode highlight")
        print("\n# Generate all 4 graph modes at once:")
        print("python3 05_kumo_rfm_integration.py --recommend 'housing' --visualize-graph --graph-mode all")
        print("\n# Save to database:")
        print("python3 05_kumo_rfm_integration.py --save-to-db --db-user postgres")
        print("\n# Combined: recommend + visualize + save:")
        print("python3 05_kumo_rfm_integration.py --recommend 'housing' --visualize-graph --graph-mode all --save-to-db")


if __name__ == '__main__':
    main()
