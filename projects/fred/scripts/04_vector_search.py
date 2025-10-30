#!/usr/bin/env python3
"""
Vector search and semantic embedding for FRED series discovery.
Uses sentence transformers to enable semantic search across economic indicators.
"""

import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not installed. Install with: pip install faiss-cpu")


class FREDVectorSearch:
    """Vector-based semantic search for FRED series."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize vector search with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")
        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device='cpu')
        self.embeddings = None
        self.series_df = None
        self.index = None
        
    def create_embeddings(self, series_df: pd.DataFrame, text_columns: List[str] = None) -> np.ndarray:
        """
        Create embeddings for all series in the dataset.
        
        Args:
            series_df: DataFrame with series metadata
            text_columns: Columns to use for embedding (default: title + notes)
        
        Returns:
            numpy array of embeddings
        """
        if text_columns is None:
            text_columns = ['title', 'notes']
        
        print(f"Creating embeddings for {len(series_df)} series...")
        
        # Combine text columns
        texts = []
        for _, row in series_df.iterrows():
            parts = []
            for col in text_columns:
                if col in row and pd.notna(row[col]) and row[col]:
                    parts.append(str(row[col]))
            
            # Add category and frequency for better context
            if 'category' in row and pd.notna(row['category']):
                parts.append(f"Category: {row['category']}")
            if 'frequency_name' in row and pd.notna(row['frequency_name']):
                parts.append(f"Frequency: {row['frequency_name']}")
            
            texts.append(' '.join(parts))
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        self.embeddings = embeddings
        self.series_df = series_df
        
        print(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search."""
        if not FAISS_AVAILABLE:
            print("Warning: FAISS not available. Using brute-force search instead.")
            return
        
        if self.embeddings is None:
            raise ValueError("Embeddings must be created first. Call create_embeddings()")
        
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        
        # Use flat L2 index for exact search (good for smaller datasets)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Search for series matching the query using semantic similarity.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
        
        Returns:
            DataFrame with matching series and similarity scores
        """
        if self.embeddings is None:
            raise ValueError("Embeddings must be created first. Call create_embeddings()")
        
        print(f"Searching for: '{query}'")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        if self.index is not None and FAISS_AVAILABLE:
            # FAISS search
            distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = self.series_df.iloc[indices[0]].copy()
            results['similarity_score'] = 1 / (1 + distances[0])  # Convert distance to similarity
        else:
            # Brute-force cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = self.series_df.iloc[top_indices].copy()
            results['similarity_score'] = similarities[top_indices]
        
        return results[['series_id', 'title', 'category', 'frequency_name', 'popularity', 'similarity_score']]
    
    def find_similar_series(self, series_id: str, top_k: int = 10) -> pd.DataFrame:
        """
        Find series similar to a given series.
        
        Args:
            series_id: ID of the reference series
            top_k: Number of similar series to return
        
        Returns:
            DataFrame with similar series
        """
        if self.embeddings is None:
            raise ValueError("Embeddings must be created first")
        
        # Find the series
        series_idx = self.series_df[self.series_df['series_id'] == series_id].index
        if len(series_idx) == 0:
            raise ValueError(f"Series {series_id} not found")
        
        series_idx = series_idx[0]
        series_embedding = self.embeddings[series_idx:series_idx+1]
        
        if self.index is not None and FAISS_AVAILABLE:
            distances, indices = self.index.search(series_embedding.astype('float32'), top_k + 1)
            
            # Skip the first result (the series itself)
            results = self.series_df.iloc[indices[0][1:]].copy()
            results['similarity_score'] = 1 / (1 + distances[0][1:])
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(series_embedding, self.embeddings)[0]
            
            # Get top-k+1 (excluding itself)
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            results = self.series_df.iloc[top_indices].copy()
            results['similarity_score'] = similarities[top_indices]
        
        return results[['series_id', 'title', 'category', 'similarity_score']]
    
    def save(self, save_dir: str):
        """Save embeddings and index to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Save embeddings
        np.save(save_path / 'embeddings.npy', self.embeddings)
        
        # Save series data
        self.series_df.to_parquet(save_path / 'series_data.parquet')
        
        # Save FAISS index
        if self.index is not None and FAISS_AVAILABLE:
            faiss.write_index(self.index, str(save_path / 'faiss.index'))
        
        print(f"Saved embeddings to {save_dir}")
    
    def load(self, load_dir: str):
        """Load embeddings and index from disk."""
        load_path = Path(load_dir)
        
        # Load embeddings
        self.embeddings = np.load(load_path / 'embeddings.npy')
        
        # Load series data
        self.series_df = pd.read_parquet(load_path / 'series_data.parquet')
        
        # Load FAISS index
        index_path = load_path / 'faiss.index'
        if index_path.exists() and FAISS_AVAILABLE:
            self.index = faiss.read_index(str(index_path))
        
        print(f"Loaded embeddings from {load_dir}")


def demo_search():
    """Demonstrate vector search capabilities."""
    
    print("="*60)
    print("FRED VECTOR SEARCH DEMO")
    print("="*60)
    
    # Load data
    data_path = 'data/fred_series_metadata.parquet'
    if not Path(data_path).exists():
        print(f"\nError: Data file not found at {data_path}")
        print("Run parse_fred_txt.py first to create the data file.")
        return
    
    print(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} series")
    
    # Initialize search
    search = FREDVectorSearch()
    
    # Create embeddings
    search.create_embeddings(df)
    
    # Build index
    search.build_faiss_index()
    
    # Demo queries
    queries = [
        "unemployment and job growth",
        "housing prices and mortgages",
        "oil prices and energy",
        "consumer spending and retail"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        results = search.search(query, top_k=5)
        print(results.to_string(index=False))
    
    # Demo: Find similar series
    if len(df) > 0:
        example_series = df.iloc[0]['series_id']
        print(f"\n{'='*60}")
        print(f"Finding series similar to: {example_series}")
        print('='*60)
        
        similar = search.find_similar_series(example_series, top_k=5)
        print(similar.to_string(index=False))
    
    # Save embeddings
    save_dir = 'data/embeddings'
    search.save(save_dir)
    print(f"\n\nEmbeddings saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Vector search for FRED series')
    parser.add_argument('--data', type=str, default='data/fred_series_metadata.parquet',
                       help='Path to FRED series data')
    parser.add_argument('--create', action='store_true',
                       help='Create embeddings')
    parser.add_argument('--search', type=str,
                       help='Search query')
    parser.add_argument('--similar', type=str,
                       help='Find series similar to given series_id')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of results to return')
    parser.add_argument('--save', type=str, default='data/embeddings',
                       help='Directory to save embeddings')
    parser.add_argument('--load', type=str,
                       help='Directory to load embeddings from')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_search()
        return
    
    # Load data
    df = pd.read_parquet(args.data)
    search = FREDVectorSearch()
    
    if args.load:
        search.load(args.load)
    elif args.create:
        search.create_embeddings(df)
        search.build_faiss_index()
        if args.save:
            search.save(args.save)
    else:
        print("Specify --create to create embeddings or --load to load existing ones")
        return
    
    # Perform search
    if args.search:
        results = search.search(args.search, top_k=args.top_k)
        print(results.to_string(index=False))
    
    if args.similar:
        results = search.find_similar_series(args.similar, top_k=args.top_k)
        print(results.to_string(index=False))


if __name__ == '__main__':
    main()
