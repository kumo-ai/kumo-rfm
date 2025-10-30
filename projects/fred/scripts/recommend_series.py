#!/usr/bin/env python3
"""
FRED Series Recommendation - ML Baseline

This module provides traditional ML-based recommendations as a baseline/validation
for Kumo RFM recommendations. Uses:
- Vector similarity (embeddings)
- Collaborative filtering
- Content-based filtering
- Popularity-based ranking

Purpose: Compare against Kumo RFM to validate recommendation quality.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")


class BaselineRecommender:
    """Traditional ML-based recommender for validation against Kumo RFM."""
    
    def __init__(self, data_path: str, embeddings_path: Optional[str] = None):
        """
        Initialize baseline recommender.
        
        Args:
            data_path: Path to FRED series metadata
            embeddings_path: Optional path to pre-computed embeddings
        """
        self.df = self._load_data(data_path)
        self.embeddings = None
        self.similarity_matrix = None
        
        if embeddings_path and Path(embeddings_path).exists():
            self.embeddings = np.load(embeddings_path)
            print(f"Loaded embeddings: {self.embeddings.shape}")
            self._compute_similarity_matrix()
        
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load FRED series metadata."""
        if data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported format: {data_path}")
    
    def _compute_similarity_matrix(self):
        """Compute cosine similarity matrix from embeddings."""
        if self.embeddings is not None:
            print("Computing similarity matrix...")
            self.similarity_matrix = cosine_similarity(self.embeddings)
            print(f"Similarity matrix: {self.similarity_matrix.shape}")
    
    def recommend_by_text(self, query: str, top_k: int = 10, method: str = "hybrid") -> pd.DataFrame:
        """
        Recommend series based on text query.
        
        Args:
            query: Free text query (e.g., "inflation housing market")
            top_k: Number of recommendations
            method: "keyword", "embedding", or "hybrid"
        
        Returns:
            DataFrame with recommended series
        """
        if method == "keyword":
            return self._recommend_keyword(query, top_k)
        elif method == "embedding" and self.embeddings is not None:
            return self._recommend_embedding(query, top_k)
        elif method == "hybrid":
            # Combine keyword and embedding approaches
            keyword_recs = self._recommend_keyword(query, top_k * 2)
            if self.embeddings is not None:
                embedding_recs = self._recommend_embedding(query, top_k * 2)
                # Merge and re-rank
                combined = pd.concat([keyword_recs, embedding_recs]).drop_duplicates(subset=['series_id'])
                return combined.head(top_k)
            return keyword_recs.head(top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _recommend_keyword(self, query: str, top_k: int) -> pd.DataFrame:
        """Simple keyword matching."""
        query_lower = query.lower()
        terms = query_lower.split()
        
        # Score based on keyword matches in title and category
        scores = []
        for idx, row in self.df.iterrows():
            title = str(row['title']).lower()
            category = str(row['category']).lower()
            notes = str(row.get('notes', '')).lower()
            
            score = 0
            for term in terms:
                if term in title:
                    score += 3
                if term in category:
                    score += 2
                if term in notes:
                    score += 1
            
            # Boost by popularity
            score *= (1 + np.log1p(row['popularity']) / 10)
            scores.append(score)
        
        self.df['match_score'] = scores
        results = self.df[self.df['match_score'] > 0].nlargest(top_k, 'match_score')
        
        return results[['series_id', 'title', 'category', 'frequency_name', 'popularity', 'match_score']]
    
    def _recommend_embedding(self, query: str, top_k: int) -> pd.DataFrame:
        """Embedding-based semantic similarity."""
        if not EMBEDDINGS_AVAILABLE:
            raise RuntimeError("sentence-transformers required for embedding method")
        
        # Encode query
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])[0]
        
        # Compute similarity to all series
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        self.df['similarity_score'] = similarities
        results = self.df.nlargest(top_k, 'similarity_score')
        
        return results[['series_id', 'title', 'category', 'frequency_name', 'popularity', 'similarity_score']]
    
    def recommend_by_series(self, series_id: str, top_k: int = 10) -> pd.DataFrame:
        """
        Recommend similar series based on a given series ID.
        
        Args:
            series_id: FRED series ID
            top_k: Number of recommendations
        
        Returns:
            DataFrame with recommended series
        """
        if self.similarity_matrix is None:
            # Fallback to category/frequency matching
            return self._recommend_by_metadata(series_id, top_k)
        
        # Find index of series
        try:
            idx = self.df[self.df['series_id'] == series_id].index[0]
        except IndexError:
            raise ValueError(f"Series not found: {series_id}")
        
        # Get similarity scores
        similarities = self.similarity_matrix[idx]
        
        # Get top similar (excluding itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = self.df.iloc[similar_indices].copy()
        results['similarity'] = similarities[similar_indices]
        
        return results[['series_id', 'title', 'category', 'frequency_name', 'popularity', 'similarity']]
    
    def _recommend_by_metadata(self, series_id: str, top_k: int) -> pd.DataFrame:
        """Fallback: recommend by matching category/frequency."""
        source = self.df[self.df['series_id'] == series_id].iloc[0]
        
        # Find series in same category and frequency
        matches = self.df[
            (self.df['category'] == source['category']) &
            (self.df['frequency'] == source['frequency']) &
            (self.df['series_id'] != series_id)
        ]
        
        # Rank by popularity
        return matches.nlargest(top_k, 'popularity')[
            ['series_id', 'title', 'category', 'frequency_name', 'popularity']
        ]
    
    def recommend_by_category(self, category: str, top_k: int = 10, 
                             min_popularity: int = 0) -> pd.DataFrame:
        """
        Recommend top series in a category.
        
        Args:
            category: Category name (partial match)
            top_k: Number of recommendations
            min_popularity: Minimum popularity threshold
        
        Returns:
            DataFrame with recommended series
        """
        matches = self.df[
            self.df['category'].str.contains(category, case=False, na=False) &
            (self.df['popularity'] >= min_popularity)
        ]
        
        if len(matches) == 0:
            print(f"No series found in category: {category}")
            return pd.DataFrame()
        
        return matches.nlargest(top_k, 'popularity')[
            ['series_id', 'title', 'category', 'frequency_name', 'popularity']
        ]
    
    def recommend_multi_series(self, series_ids: List[str], top_k: int = 10) -> pd.DataFrame:
        """
        Recommend series based on multiple input series.
        Finds series similar to the group.
        
        Args:
            series_ids: List of series IDs
            top_k: Number of recommendations
        
        Returns:
            DataFrame with recommended series
        """
        if self.similarity_matrix is None:
            # Fallback: find common categories
            categories = self.df[self.df['series_id'].isin(series_ids)]['category'].unique()
            results = self.df[
                self.df['category'].isin(categories) &
                (~self.df['series_id'].isin(series_ids))
            ]
            return results.nlargest(top_k, 'popularity')[
                ['series_id', 'title', 'category', 'frequency_name', 'popularity']
            ]
        
        # Get indices of input series
        indices = []
        for sid in series_ids:
            try:
                idx = self.df[self.df['series_id'] == sid].index[0]
                indices.append(idx)
            except IndexError:
                print(f"Warning: Series not found: {sid}")
        
        if not indices:
            return pd.DataFrame()
        
        # Average similarity across all input series
        avg_similarities = self.similarity_matrix[indices].mean(axis=0)
        
        # Get top similar (excluding input series)
        mask = ~self.df.index.isin(indices)
        candidates = self.df[mask].copy()
        candidates['similarity'] = avg_similarities[mask]
        
        results = candidates.nlargest(top_k, 'similarity')
        
        return results[['series_id', 'title', 'category', 'frequency_name', 'popularity', 'similarity']]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'total_series': len(self.df),
            'categories': self.df['category'].nunique(),
            'avg_popularity': self.df['popularity'].mean(),
            'has_embeddings': self.embeddings is not None,
            'top_categories': self.df['category'].value_counts().head(10).to_dict()
        }


def main():
    parser = argparse.ArgumentParser(description='FRED Series Baseline Recommender')
    parser.add_argument('--data', type=str, default='data/fred_series_metadata.parquet',
                       help='Path to FRED series metadata')
    parser.add_argument('--embeddings', type=str, default='data/embeddings/embeddings.npy',
                       help='Path to embeddings file')
    
    # Recommendation modes
    parser.add_argument('--text', type=str, help='Text query for recommendations')
    parser.add_argument('--series', type=str, help='Series ID to find similar series')
    parser.add_argument('--category', type=str, help='Category to explore')
    parser.add_argument('--multi', type=str, nargs='+', help='Multiple series IDs')
    
    parser.add_argument('--top-k', type=int, default=10, help='Number of recommendations')
    parser.add_argument('--method', type=str, default='hybrid', 
                       choices=['keyword', 'embedding', 'hybrid'],
                       help='Recommendation method for text queries')
    parser.add_argument('--stats', action='store_true', help='Show dataset statistics')
    
    args = parser.parse_args()
    
    # Initialize recommender
    print(f"Loading data from {args.data}...")
    recommender = BaselineRecommender(args.data, args.embeddings)
    
    if args.stats:
        stats = recommender.get_statistics()
        print("\n=== Dataset Statistics ===")
        for key, value in stats.items():
            if key != 'top_categories':
                print(f"{key}: {value}")
        print("\nTop Categories:")
        for cat, count in stats['top_categories'].items():
            print(f"  {cat}: {count}")
        return
    
    # Generate recommendations based on mode
    if args.text:
        print(f"\n Text Query: '{args.text}'")
        print(f"Method: {args.method}\n")
        results = recommender.recommend_by_text(args.text, args.top_k, args.method)
    
    elif args.series:
        print(f"\n Similar to: {args.series}\n")
        results = recommender.recommend_by_series(args.series, args.top_k)
    
    elif args.category:
        print(f"\n Category: {args.category}\n")
        results = recommender.recommend_by_category(args.category, args.top_k)
    
    elif args.multi:
        print(f"\n Multi-series input: {', '.join(args.multi)}\n")
        results = recommender.recommend_multi_series(args.multi, args.top_k)
    
    else:
        parser.print_help()
        return
    
    # Display results
    if len(results) > 0:
        print("=" * 100)
        print(f"Top {len(results)} Recommendations:")
        print("=" * 100)
        for idx, row in results.iterrows():
            print(f"\n{row['series_id']}")
            print(f"  Title: {row['title'][:80]}")
            print(f"  Category: {row['category']}")
            print(f"  Frequency: {row['frequency_name']}")
            print(f"  Popularity: {row['popularity']}")
            
            # Show score if available
            if 'match_score' in row:
                print(f"  Match Score: {row['match_score']:.2f}")
            elif 'similarity_score' in row:
                print(f"  Similarity: {row['similarity_score']:.4f}")
            elif 'similarity' in row:
                print(f"  Similarity: {row['similarity']:.4f}")
    else:
        print("No recommendations found.")


if __name__ == '__main__':
    main()
