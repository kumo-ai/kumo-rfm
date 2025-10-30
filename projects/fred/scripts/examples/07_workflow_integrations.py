#!/usr/bin/env python3
"""
Workflow integration examples for FRED data.
Demonstrates integration with ElasticSearch, Pinecone, Weaviate, and Neo4j.
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json


# ============================================================================
# ElasticSearch Integration
# ============================================================================

class ElasticSearchIntegration:
    """ElasticSearch integration for FRED series search."""
    
    def __init__(self, host: str = 'localhost', port: int = 9200):
        try:
            from elasticsearch import Elasticsearch
            self.client = Elasticsearch([f"http://{host}:{port}"])
            self.index_name = "fred_series"
            print(f"Connected to ElasticSearch at {host}:{port}")
        except ImportError:
            print("elasticsearch-py not installed. Install: pip install elasticsearch")
            self.client = None
    
    def create_index(self):
        """Create index with appropriate mapping."""
        if not self.client:
            return
        
        mapping = {
            "mappings": {
                "properties": {
                    "series_id": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "english"},
                    "category": {"type": "keyword"},
                    "frequency": {"type": "keyword"},
                    "popularity": {"type": "integer"},
                    "notes": {"type": "text", "analyzer": "english"},
                    "notes_length": {"type": "integer"}
                }
            }
        }
        
        self.client.indices.create(index=self.index_name, body=mapping, ignore=400)
        print(f"Created index: {self.index_name}")
    
    def index_series(self, series_df: pd.DataFrame):
        """Bulk index series data."""
        if not self.client:
            return
        
        from elasticsearch.helpers import bulk
        
        # Prepare documents
        actions = []
        for _, row in series_df.iterrows():
            action = {
                "_index": self.index_name,
                "_id": row['series_id'],
                "_source": row.to_dict()
            }
            actions.append(action)
        
        # Bulk index
        success, failed = bulk(self.client, actions)
        print(f"Indexed {success} documents, {failed} failed")
    
    def search(self, query: str, size: int = 10) -> List[Dict]:
        """Full-text search across title and notes."""
        if not self.client:
            return []
        
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "notes", "category"],
                    "type": "best_fields"
                }
            },
            "size": size
        }
        
        results = self.client.search(index=self.index_name, body=search_body)
        
        return [
            {
                "series_id": hit["_source"]["series_id"],
                "title": hit["_source"]["title"],
                "category": hit["_source"]["category"],
                "score": hit["_score"]
            }
            for hit in results["hits"]["hits"]
        ]
    
    def aggregate_by_category(self) -> Dict:
        """Aggregate series counts by category."""
        if not self.client:
            return {}
        
        agg_body = {
            "size": 0,
            "aggs": {
                "categories": {
                    "terms": {"field": "category", "size": 20},
                    "aggs": {
                        "avg_popularity": {"avg": {"field": "popularity"}}
                    }
                }
            }
        }
        
        results = self.client.search(index=self.index_name, body=agg_body)
        return results["aggregations"]["categories"]


# ============================================================================
# Pinecone Vector DB Integration
# ============================================================================

class PineconeIntegration:
    """Pinecone vector database integration."""
    
    def __init__(self, api_key: str, environment: str = "us-west1-gcp"):
        try:
            import pinecone
            pinecone.init(api_key=api_key, environment=environment)
            self.index_name = "fred-series"
            self.pinecone = pinecone
            print(f"Connected to Pinecone in {environment}")
        except ImportError:
            print("pinecone-client not installed. Install: pip install pinecone-client")
            self.pinecone = None
    
    def create_index(self, dimension: int = 384):
        """Create Pinecone index."""
        if not self.pinecone:
            return
        
        if self.index_name not in self.pinecone.list_indexes():
            self.pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine"
            )
            print(f"Created Pinecone index: {self.index_name}")
    
    def upload_embeddings(self, embeddings, series_df: pd.DataFrame):
        """Upload embeddings to Pinecone."""
        if not self.pinecone:
            return
        
        index = self.pinecone.Index(self.index_name)
        
        # Prepare vectors
        vectors = []
        for i, (_, row) in enumerate(series_df.iterrows()):
            vectors.append((
                row['series_id'],
                embeddings[i].tolist(),
                {
                    "title": row['title'],
                    "category": row['category'],
                    "popularity": int(row['popularity'])
                }
            ))
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)
        
        print(f"Uploaded {len(vectors)} vectors to Pinecone")
    
    def search(self, query_embedding, top_k: int = 10):
        """Search for similar vectors."""
        if not self.pinecone:
            return []
        
        index = self.pinecone.Index(self.index_name)
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        return results['matches']


# ============================================================================
# Weaviate Vector DB Integration
# ============================================================================

class WeaviateIntegration:
    """Weaviate vector database integration."""
    
    def __init__(self, url: str = "http://localhost:8080"):
        try:
            import weaviate
            self.client = weaviate.Client(url)
            self.class_name = "FredSeries"
            print(f"Connected to Weaviate at {url}")
        except ImportError:
            print("weaviate-client not installed. Install: pip install weaviate-client")
            self.client = None
    
    def create_schema(self):
        """Create Weaviate schema."""
        if not self.client:
            return
        
        schema = {
            "class": self.class_name,
            "vectorizer": "none",  # We'll provide our own vectors
            "properties": [
                {"name": "series_id", "dataType": ["string"]},
                {"name": "title", "dataType": ["text"]},
                {"name": "category", "dataType": ["string"]},
                {"name": "frequency", "dataType": ["string"]},
                {"name": "popularity", "dataType": ["int"]},
                {"name": "notes", "dataType": ["text"]}
            ]
        }
        
        self.client.schema.create_class(schema)
        print(f"Created Weaviate class: {self.class_name}")
    
    def upload_data(self, series_df: pd.DataFrame, embeddings):
        """Upload data with embeddings to Weaviate."""
        if not self.client:
            return
        
        with self.client.batch as batch:
            for i, (_, row) in enumerate(series_df.iterrows()):
                properties = {
                    "series_id": row['series_id'],
                    "title": row['title'],
                    "category": row['category'],
                    "frequency": row.get('frequency', ''),
                    "popularity": int(row['popularity']),
                    "notes": row.get('notes', '')[:10000]  # Limit length
                }
                
                batch.add_data_object(
                    properties,
                    self.class_name,
                    vector=embeddings[i].tolist()
                )
        
        print(f"Uploaded {len(series_df)} objects to Weaviate")
    
    def semantic_search(self, query_embedding, limit: int = 10):
        """Perform semantic search."""
        if not self.client:
            return []
        
        result = (
            self.client.query
            .get(self.class_name, ["series_id", "title", "category", "popularity"])
            .with_near_vector({"vector": query_embedding.tolist()})
            .with_limit(limit)
            .do()
        )
        
        return result["data"]["Get"][self.class_name]


# ============================================================================
# Neo4j Graph Database Integration
# ============================================================================

class Neo4jIntegration:
    """Neo4j graph database integration for series relationships."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            print(f"Connected to Neo4j at {uri}")
        except ImportError:
            print("neo4j driver not installed. Install: pip install neo4j")
            self.driver = None
    
    def create_series_nodes(self, series_df: pd.DataFrame):
        """Create series nodes in Neo4j."""
        if not self.driver:
            return
        
        with self.driver.session() as session:
            for _, row in series_df.iterrows():
                session.run(
                    """
                    MERGE (s:Series {series_id: $series_id})
                    SET s.title = $title,
                        s.category = $category,
                        s.frequency = $frequency,
                        s.popularity = $popularity
                    """,
                    series_id=row['series_id'],
                    title=row['title'],
                    category=row['category'],
                    frequency=row.get('frequency', ''),
                    popularity=int(row['popularity'])
                )
        
        print(f"Created {len(series_df)} series nodes")
    
    def create_category_relationships(self):
        """Create relationships between series in the same category."""
        if not self.driver:
            return
        
        with self.driver.session() as session:
            session.run("""
                MATCH (s1:Series), (s2:Series)
                WHERE s1.category = s2.category AND s1.series_id < s2.series_id
                MERGE (s1)-[r:SAME_CATEGORY]->(s2)
                SET r.category = s1.category
            """)
        
        print("Created category relationships")
    
    def find_related_series(self, series_id: str, max_depth: int = 2):
        """Find related series using graph traversal."""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Series {series_id: $series_id})-[*1..%d]-(related:Series)
                RETURN DISTINCT related.series_id AS series_id,
                       related.title AS title,
                       related.category AS category
                LIMIT 20
                """ % max_depth,
                series_id=series_id
            )
            
            return [dict(record) for record in result]
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()


# ============================================================================
# Demonstration
# ============================================================================

def demo_integrations(data_path: str):
    """Demonstrate all workflow integrations."""
    
    print("="*60)
    print("WORKFLOW INTEGRATIONS DEMO")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} series\n")
    
    # Demo 1: ElasticSearch
    print("="*60)
    print("ElasticSearch Integration")
    print("="*60)
    print("\nExample code:")
    print("""
    es = ElasticSearchIntegration()
    es.create_index()
    es.index_series(df)
    results = es.search("unemployment rate")
    """)
    
    # Demo 2: Pinecone
    print("\n" + "="*60)
    print("Pinecone Vector DB Integration")
    print("="*60)
    print("\nExample code:")
    print("""
    pinecone = PineconeIntegration(api_key="your-api-key")
    pinecone.create_index(dimension=384)
    pinecone.upload_embeddings(embeddings, df)
    results = pinecone.search(query_embedding, top_k=10)
    """)
    
    # Demo 3: Weaviate
    print("\n" + "="*60)
    print("Weaviate Vector DB Integration")
    print("="*60)
    print("\nExample code:")
    print("""
    weaviate = WeaviateIntegration()
    weaviate.create_schema()
    weaviate.upload_data(df, embeddings)
    results = weaviate.semantic_search(query_embedding)
    """)
    
    # Demo 4: Neo4j
    print("\n" + "="*60)
    print("Neo4j Graph Database Integration")
    print("="*60)
    print("\nExample code:")
    print("""
    neo4j = Neo4jIntegration()
    neo4j.create_series_nodes(df)
    neo4j.create_category_relationships()
    related = neo4j.find_related_series("PAYEMS")
    neo4j.close()
    """)
    
    print("\n" + "="*60)
    print("Integration Summary")
    print("="*60)
    print("""
Available integrations:
1. ElasticSearch - Full-text search, aggregations
2. Pinecone - Cloud vector database, fast similarity search
3. Weaviate - Local/cloud vector DB, semantic search
4. Neo4j - Graph relationships, complex queries

Install required packages:
  pip install elasticsearch
  pip install pinecone-client
  pip install weaviate-client
  pip install neo4j
    """)


def main():
    parser = argparse.ArgumentParser(description='Workflow integration examples')
    parser.add_argument('--data', type=str, default='data/fred_series_metadata.parquet',
                       help='Path to FRED series data')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_integrations(args.data)
    else:
        print("Use --demo flag to see integration examples")


if __name__ == '__main__':
    main()
