#!/usr/bin/env python3
"""
End-to-end pipeline orchestration for FRED data processing.
Combines parsing, DB loading, vector search, Kumo integration, and Monolith prep.
"""

import argparse
import sys
from pathlib import Path
import subprocess


class Pipeline:
    """Orchestrate the complete FRED data processing pipeline."""
    
    def __init__(self, config: dict):
        self.config = config
        self.txt_dir = Path(config.get('txt_dir', 'txt'))
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.db_path = config.get('db_path', 'fred.db')
        
    def run_step(self, step_name: str, command: list, required: bool = True):
        """Execute a pipeline step."""
        print("\n" + "="*70)
        print(f"STEP: {step_name}")
        print("="*70)
        print(f"Command: {' '.join(command)}\n")
        
        try:
            result = subprocess.run(
                command,
                check=required,
                capture_output=False,
                text=True
            )
            print(f"\n {step_name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n {step_name} failed with exit code {e.returncode}")
            if required:
                print("This is a required step. Pipeline aborted.")
                sys.exit(1)
            return False
    
    def step_1_parse(self):
        """Parse FRED txt files into structured data."""
        return self.run_step(
            "Parse TXT Files",
            [
                'python3', 'parse_fred_txt.py',
                '--input', str(self.txt_dir),
                '--output', str(self.data_dir),
                '--format', 'all'
            ]
        )
    
    def step_2_load_postgres(self):
        """Load data into PostgreSQL."""
        data_file = self.data_dir / 'fred_series_metadata.parquet'
        
        return self.run_step(
            "Load PostgreSQL",
            [
                'python3', '03_load_to_postgres.py',
                '--init',
                '--data', str(data_file),
                '--format', 'parquet'
            ]
        )
    
    def step_3_export_kumo(self):
        """Export data for Kumo RFM."""
        kumo_dir = self.data_dir / 'kumo'
        
        return self.run_step(
            "Export for Kumo",
            [
                'python3', '03_load_to_postgres.py',
                '--export-kumo', str(kumo_dir)
            ],
            required=False
        )
    
    def step_4_create_embeddings(self):
        """Create vector embeddings for semantic search."""
        data_file = self.data_dir / 'fred_series_metadata.parquet'
        embeddings_dir = self.data_dir / 'embeddings'
        
        return self.run_step(
            "Create Vector Embeddings",
            [
                'python3', 'vector_search.py',
                '--data', str(data_file),
                '--create',
                '--save', str(embeddings_dir)
            ],
            required=False
        )
    
    def step_5_vector_search_demo(self):
        """Run vector search demo."""
        return self.run_step(
            "Vector Search Demo",
            [
                'python3', 'vector_search.py',
                '--demo'
            ],
            required=False
        )
    
    def step_6_kumo_demo(self):
        """Run Kumo RFM demo."""
        data_file = self.data_dir / 'fred_series_metadata.parquet'
        
        return self.run_step(
            "Kumo RFM Demo",
            [
                'python3', 'kumo_rfm_integration.py',
                '--data', str(data_file),
                '--demo'
            ],
            required=False
        )
    
    def step_7_prepare_monolith(self):
        """Prepare features for ByteDance Monolith."""
        data_file = self.data_dir / 'fred_series_metadata.parquet'
        embeddings_file = self.data_dir / 'embeddings' / 'embeddings.npy'
        monolith_dir = self.data_dir / 'monolith'
        
        command = [
            'python3', 'prepare_monolith_features.py',
            '--data', str(data_file),
            '--output', str(monolith_dir),
            '--format', 'parquet',
            '--n-interactions', '1000'
        ]
        
        # Add embeddings if they exist
        if embeddings_file.exists():
            command.extend(['--embeddings', str(embeddings_file)])
        
        return self.run_step(
            "Prepare Monolith Features",
            command,
            required=False
        )
    
    def step_8_workflow_demo(self):
        """Run workflow integrations demo."""
        data_file = self.data_dir / 'fred_series_metadata.parquet'
        
        return self.run_step(
            "Workflow Integrations Demo",
            [
                'python3', 'workflow_integrations.py',
                '--data', str(data_file),
                '--demo'
            ],
            required=False
        )
    
    def run_full_pipeline(self):
        """Execute the complete pipeline."""
        print("="*70)
        print("FRED DATA PROCESSING PIPELINE")
        print("="*70)
        print(f"Configuration:")
        print(f"  - TXT directory: {self.txt_dir}")
        print(f"  - Data directory: {self.data_dir}")
        print(f"  - Database: {self.db_path}")
        print()
        
        # Required steps
        self.step_1_parse()
        self.step_2_load_postgres()
        
        # Optional steps (won't fail pipeline if they fail)
        self.step_3_export_kumo()
        self.step_4_create_embeddings()
        self.step_5_vector_search_demo()
        self.step_6_kumo_demo()
        self.step_7_prepare_monolith()
        self.step_8_workflow_demo()
        
        # Summary
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print("\nGenerated outputs:")
        
        outputs = [
            (self.data_dir / 'fred_series_metadata.parquet', "Parsed series metadata (Parquet)"),
            (self.data_dir / 'fred_series_metadata.csv', "Parsed series metadata (CSV)"),
            (self.data_dir / 'fred_series_metadata.json', "Parsed series metadata (JSON)"),
            (self.data_dir / 'kumo', "Kumo RFM exports"),
            (self.data_dir / 'embeddings', "Vector embeddings"),
            (self.data_dir / 'monolith', "Monolith features"),
        ]
        
        for path, description in outputs:
            if path.exists():
                print(f"   {description}: {path}")
            else:
                print(f"   {description}: {path} (not created)")
        
        print("\nNext steps:")
        print("  1. Query PostgreSQL: psql -d fred")
        print("  2. Search series: python3 04_vector_search.py --search 'your query'")
        print("  3. Use Kumo: python3 05_kumo_rfm_integration.py --demo")
        print("  4. Train Monolith: use data/monolith/ for model training")
        print("\nPostgreSQL connection info:")
        print("  Set POSTGRES_PASSWORD env var before running")


def main():
    parser = argparse.ArgumentParser(
        description='FRED data processing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with defaults
  python3 pipeline.py --full

  # Run specific steps
  python3 pipeline.py --steps parse postgres embeddings
  
  # Custom directories
  python3 pipeline.py --full --txt-dir my_txt --data-dir my_data
        """
    )
    
    parser.add_argument('--full', action='store_true',
                       help='Run full pipeline')
    parser.add_argument('--steps', nargs='+',
                       choices=['parse', 'postgres', 'kumo-export', 'embeddings', 
                               'vector-demo', 'kumo-demo', 'monolith', 'workflow-demo'],
                       help='Run specific steps only')
    parser.add_argument('--txt-dir', type=str, default='txt',
                       help='Directory containing txt files')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Output data directory')
    parser.add_argument('--db-path', type=str, default='fred.db',
                       help='DuckDB database path')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'txt_dir': args.txt_dir,
        'data_dir': args.data_dir,
        'db_path': args.db_path
    }
    
    pipeline = Pipeline(config)
    
    if args.full:
        pipeline.run_full_pipeline()
    elif args.steps:
        step_mapping = {
            'parse': pipeline.step_1_parse,
            'postgres': pipeline.step_2_load_postgres,
            'kumo-export': pipeline.step_3_export_kumo,
            'embeddings': pipeline.step_4_create_embeddings,
            'vector-demo': pipeline.step_5_vector_search_demo,
            'kumo-demo': pipeline.step_6_kumo_demo,
            'monolith': pipeline.step_7_prepare_monolith,
            'workflow-demo': pipeline.step_8_workflow_demo
        }
        
        for step_name in args.steps:
            step_mapping[step_name]()
    else:
        parser.print_help()
        print("\nPlease specify --full or --steps")


if __name__ == '__main__':
    main()
