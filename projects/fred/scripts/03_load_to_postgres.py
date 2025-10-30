#!/usr/bin/env python3
"""
Load parsed FRED series data into PostgreSQL.
Supports both initial load and incremental updates.
"""

import argparse
import pandas as pd
from pathlib import Path
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    print("Warning: psycopg2 not installed. Install with: pip install psycopg2-binary")


class PostgreSQLLoader:
    """Loader for FRED data into PostgreSQL."""
    
    def __init__(self, host: str = 'localhost', port: int = 5432, 
                 database: str = 'fred', user: str = None, password: str = None):
        """
        Initialize PostgreSQL connection.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username (or set POSTGRES_USER env var)
            password: Password (or set POSTGRES_PASSWORD env var)
        """
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 required. Install: pip install psycopg2-binary")
        
        self.host = host
        self.port = port
        self.database = database
        self.user = user or os.getenv('POSTGRES_USER', 'postgres')
        self.password = password or os.getenv('POSTGRES_PASSWORD')
        
        if not self.password:
            raise ValueError(
                "PostgreSQL password required. Set POSTGRES_PASSWORD env var or pass via --password"
            )
        
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.conn.autocommit = False
            print(f"Connected to PostgreSQL: {self.user}@{self.host}:{self.port}/{self.database}")
        except psycopg2.Error as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")
        
    def initialize_schema(self, schema_path: str = 'create_tables.sql'):
        """Initialize database schema from SQL file."""
        print(f"Initializing schema from {schema_path}...")
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema creation
        with self.conn.cursor() as cur:
            cur.execute(schema_sql)
        self.conn.commit()
    
    def load_series_metadata(self, data_path: str, format: str = 'parquet'):
        """Load series metadata from parsed data file."""
        print(f"Loading series metadata from {data_path}...")
        
        # Read data based on format
        if format == 'parquet':
            df = pd.read_parquet(data_path)
        elif format == 'csv':
            df = pd.read_csv(data_path)
        elif format == 'json':
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Prepare data for insertion
        columns = ['series_id', 'title', 'frequency', 'frequency_name', 'popularity',
                  'notes', 'source_file', 'category', 'has_notes', 'notes_length']
        
        # Convert 'popularity' and 'notes_length' to appropriate types to prevent NumericValueOutOfRange
        if 'popularity' in df.columns: # Guard against missing columns
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0).astype(int)
        if 'notes_length' in df.columns:
            df['notes_length'] = pd.to_numeric(df['notes_length'], errors='coerce').fillna(0).astype(int)
        if 'title' in df.columns:
            df['title'] = df['title'].fillna('') # Ensure title is not null
        
        # Deduplicate by series_id (keep first occurrence)
        df = df.drop_duplicates(subset=['series_id'], keep='first')
        
        # Convert DataFrame to list of tuples
        values = [tuple(row[col] if col in row.index else None for col in columns) 
                 for _, row in df.iterrows()]
        
        # Insert into database using ON CONFLICT for upsert
        insert_sql = sql.SQL("""
            INSERT INTO series_metadata 
            (series_id, title, frequency, frequency_name, popularity, 
             notes, source_file, category, has_notes, notes_length)
            VALUES %s
            ON CONFLICT (series_id) DO UPDATE SET
                title = EXCLUDED.title,
                frequency = EXCLUDED.frequency,
                frequency_name = EXCLUDED.frequency_name,
                popularity = EXCLUDED.popularity,
                notes = EXCLUDED.notes,
                source_file = EXCLUDED.source_file,
                category = EXCLUDED.category,
                has_notes = EXCLUDED.has_notes,
                notes_length = EXCLUDED.notes_length
        """)
        
        with self.conn.cursor() as cur:
            execute_values(cur, insert_sql.as_string(self.conn), values)
        self.conn.commit()
        
        # Get count
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM series_metadata")
            row_count = cur.fetchone()[0]
        
        print(f"Loaded {row_count} series into series_metadata table.")
        return row_count
    
    def query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        return pd.read_sql_query(query, self.conn)
    
    def get_summary_stats(self):
        """Get summary statistics about loaded data."""
        stats = {}
        
        # Total series count
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM series_metadata")
            stats['total_series'] = cur.fetchone()[0]
        
        # Series by category
        stats['by_category'] = self.query("""
            SELECT category, COUNT(*) as count, AVG(popularity) as avg_popularity
            FROM series_metadata
            GROUP BY category
            ORDER BY count DESC
        """)
        
        # Series by frequency
        stats['by_frequency'] = self.query("""
            SELECT frequency_name, COUNT(*) as count
            FROM series_metadata
            GROUP BY frequency_name
            ORDER BY count DESC
        """)
        
        # Top popular series
        stats['top_series'] = self.query("""
            SELECT series_id, title, category, popularity
            FROM series_metadata
            ORDER BY popularity DESC
            LIMIT 10
        """)
        
        return stats
    
    def export_for_kumo(self, output_path: str):
        """Export data in format suitable for Kumo RFM."""
        print(f"Exporting data for Kumo RFM to {output_path}...")
        
        # Export series metadata
        df = self.query("SELECT * FROM series_metadata")
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save as parquet (Kumo-friendly format)
        df.to_parquet(output_dir / 'series_metadata.parquet', index=False)
        
        # Also create a relationships file (example)
        relationships = self.query("""
            SELECT 
                m1.series_id as series_id_1,
                m2.series_id as series_id_2,
                'same_category' as relationship_type,
                0.8 as strength
            FROM series_metadata m1
            JOIN series_metadata m2 
                ON m1.category = m2.category 
                AND m1.series_id < m2.series_id
            LIMIT 1000
        """)
        
        relationships.to_parquet(output_dir / 'series_relationships.parquet', index=False)
        
        print(f"Exported {len(df)} series and {len(relationships)} relationships.")
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='Load FRED data into PostgreSQL')
    parser.add_argument('--host', type=str, default='localhost',
                       help='PostgreSQL host (default: localhost)')
    parser.add_argument('--port', type=int, default=5432,
                       help='PostgreSQL port (default: 5432)')
    parser.add_argument('--database', type=str, default='fred',
                       help='Database name (default: fred)')
    parser.add_argument('--user', type=str,
                       help='Database user (or set POSTGRES_USER env var)')
    parser.add_argument('--password', type=str,
                       help='Database password (or set POSTGRES_PASSWORD env var)')
    parser.add_argument('--schema', type=str, default='create_tables.sql',
                       help='SQL schema file')
    parser.add_argument('--data', type=str, default='data/fred_series_metadata.parquet',
                       help='Parsed data file to load')
    parser.add_argument('--format', type=str, default='parquet',
                       choices=['csv', 'parquet', 'json'],
                       help='Input data format')
    parser.add_argument('--init', action='store_true',
                       help='Initialize schema (creates tables)')
    parser.add_argument('--export-kumo', type=str,
                       help='Export data for Kumo RFM to specified directory')
    
    args = parser.parse_args()
    
    # Initialize loader
    try:
        loader = PostgreSQLLoader(
            host=args.host,
            port=args.port,
            database=args.database,
            user=args.user,
            password=args.password
        )
    except (ImportError, ValueError, ConnectionError) as e:
        print(f"Error: {e}")
        return
    
    # Initialize schema before loading data. The SQL script should use CREATE TABLE IF NOT EXISTS.
    loader.initialize_schema(args.schema)

    # Load data
    if Path(args.data).exists():
        loader.load_series_metadata(args.data, args.format)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("DATABASE SUMMARY")
        print("="*60)
        stats = loader.get_summary_stats()
        print(f"\nTotal series: {stats['total_series']}")
        print("\nSeries by category:")
        print(stats['by_category'].to_string(index=False))
        print("\nSeries by frequency:")
        print(stats['by_frequency'].to_string(index=False))
        print("\nTop 10 popular series:")
        print(stats['top_series'].to_string(index=False))
    else:
        print(f"Warning: Data file not found at {args.data}")
    
    # Export for Kumo if requested
    if args.export_kumo:
        loader.export_for_kumo(args.export_kumo)
    
    loader.close()
    print(f"\nData loaded to PostgreSQL: {args.user}@{args.host}:{args.port}/{args.database}")


if __name__ == '__main__':
    main()
