#!/usr/bin/env python3
"""
Parse FRED series text files into structured data formats.
Extracts series metadata including ID, title, frequency, popularity, and notes.
"""

import re
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import json


class FREDSeriesParser:
    """Parser for FRED series text files."""
    
    def __init__(self):
        self.series_pattern = re.compile(r'^\d+\.\s+(\S+)')
        self.title_pattern = re.compile(r'^\s+Title:\s+(.+)')
        self.frequency_pattern = re.compile(r'^\s+Frequency:\s+(\S+)')
        self.popularity_pattern = re.compile(r'^\s+Popularity:\s+(\d+)')
        self.notes_pattern = re.compile(r'^\s+Notes:\s*')
        
    def parse_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Parse a single FRED series file."""
        series_list = []
        current_series = None
        capturing_notes = False
        notes_buffer = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Check for series ID
                series_match = self.series_pattern.match(line)
                if series_match:
                    # Save previous series if exists
                    if current_series:
                        current_series['notes'] = ' '.join(notes_buffer).strip()
                        series_list.append(current_series)
                    
                    # Start new series
                    current_series = {
                        'series_id': series_match.group(1),
                        'title': None,
                        'frequency': None,
                        'popularity': None,
                        'notes': '',
                        'source_file': filepath.stem
                    }
                    notes_buffer = []
                    capturing_notes = False
                    continue
                
                if not current_series:
                    continue
                
                # Extract title
                title_match = self.title_pattern.match(line)
                if title_match:
                    current_series['title'] = title_match.group(1).strip()
                    continue
                
                # Extract frequency
                freq_match = self.frequency_pattern.match(line)
                if freq_match:
                    current_series['frequency'] = freq_match.group(1).strip()
                    continue
                
                # Extract popularity
                pop_match = self.popularity_pattern.match(line)
                if pop_match:
                    current_series['popularity'] = int(pop_match.group(1))
                    continue
                
                # Start capturing notes
                notes_match = self.notes_pattern.match(line)
                if notes_match:
                    capturing_notes = True
                    continue
                
                # Capture notes content
                if capturing_notes and line.strip() and not line.startswith('---'):
                    notes_buffer.append(line.strip())
                
                # Stop capturing at separator
                if line.startswith('---'):
                    capturing_notes = False
        
        # Don't forget the last series
        if current_series:
            current_series['notes'] = ' '.join(notes_buffer).strip()
            series_list.append(current_series)
        
        return series_list
    
    def parse_directory(self, input_dir: Path) -> pd.DataFrame:
        """Parse all .txt files in a directory."""
        all_series = []
        
        txt_files = list(input_dir.glob('*.txt'))
        print(f"Found {len(txt_files)} text files to parse...")
        
        for filepath in txt_files:
            print(f"Parsing {filepath.name}...")
            series_data = self.parse_file(filepath)
            all_series.extend(series_data)
        
        df = pd.DataFrame(all_series)
        print(f"\nParsed {len(df)} total series from {len(txt_files)} files")
        
        return df
    
    def extract_category(self, source_file: str) -> str:
        """Extract category from source filename."""
        # Remove '_series' suffix and date suffix
        category = re.sub(r'_series(_\d{8})?$', '', source_file)
        return category.replace('_', ' ').title()


def main():
    parser = argparse.ArgumentParser(description='Parse FRED series text files')
    parser.add_argument('--input', type=str, default='txt',
                       help='Input directory containing .txt files')
    parser.add_argument('--output', type=str, default='data',
                       help='Output directory for parsed data')
    parser.add_argument('--format', type=str, default='all',
                       choices=['csv', 'parquet', 'json', 'all'],
                       help='Output format(s)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_path = Path(args.output)
    
    # Handle both directory and file paths for output
    if output_path.suffix in ['.parquet', '.csv', '.json']:
        output_dir = output_path.parent
    else:
        output_dir = output_path
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Parse files
    fred_parser = FREDSeriesParser()
    df = fred_parser.parse_directory(input_dir)
    
    # Add derived fields
    df['category'] = df['source_file'].apply(fred_parser.extract_category)
    df['has_notes'] = df['notes'].str.len() > 0
    df['notes_length'] = df['notes'].str.len()
    
    # Map frequency codes to full names
    freq_mapping = {
        'D': 'Daily',
        'W': 'Weekly',
        'M': 'Monthly',
        'Q': 'Quarterly',
        'A': 'Annual'
    }
    df['frequency_name'] = df['frequency'].map(freq_mapping).fillna(df['frequency'])
    
    # Output in requested format(s)
    if args.format in ['csv', 'all']:
        csv_path = output_dir / 'fred_series_metadata.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV to: {csv_path}")
    
    if args.format in ['parquet', 'all']:
        parquet_path = output_dir / 'fred_series_metadata.parquet'
        df.to_parquet(parquet_path, index=False)
        print(f"Saved Parquet to: {parquet_path}")
    
    if args.format in ['json', 'all']:
        json_path = output_dir / 'fred_series_metadata.json'
        df.to_json(json_path, orient='records', indent=2)
        print(f"Saved JSON to: {json_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total series: {len(df)}")
    print(f"Unique categories: {df['category'].nunique()}")
    print(f"Categories: {', '.join(df['category'].unique()[:5])}...")
    print(f"\nFrequency distribution:")
    print(df['frequency_name'].value_counts().to_string())
    print(f"\nPopularity statistics:")
    print(df['popularity'].describe().to_string())


if __name__ == '__main__':
    main()
