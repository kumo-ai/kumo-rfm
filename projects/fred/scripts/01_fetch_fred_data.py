#!/usr/bin/env python3
"""
FRED API Data Fetcher
Query and download economic series data from the Federal Reserve Economic Data (FRED) API.
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("requests library not found. Install with: pip install requests")


class FREDAPIClient:
    """Client for interacting with FRED API."""
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED API client.
        
        Args:
            api_key: FRED API key. If not provided, will check FRED_API_KEY env var.
                    Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required. Install: pip install requests")
        
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY env var or pass via --api-key.\n"
                "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        
        self.session = requests.Session()
    
    def _request(self, endpoint: str, params: Dict) -> Dict:
        """Make a request to the FRED API."""
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {}
    
    def search_series(self, search_text: str, limit: int = 1000, **kwargs) -> List[Dict]:
        """
        Search for FRED series by text.
        
        Args:
            search_text: Search query
            limit: Maximum number of results
            **kwargs: Additional FRED API parameters (order_by, sort_order, etc.)
        
        Returns:
            List of series dictionaries
        """
        params = {
            'search_text': search_text,
            'limit': limit,
            **kwargs
        }
        
        result = self._request('series/search', params)
        return result.get('seriess', [])
    
    def get_series_info(self, series_id: str) -> Dict:
        """Get detailed information about a series."""
        params = {'series_id': series_id}
        result = self._request('series', params)
        return result.get('seriess', [{}])[0] if result.get('seriess') else {}
    
    def get_series_observations(self, series_id: str, 
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> List[Dict]:
        """
        Get observation data for a series.
        
        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            List of observation dictionaries
        """
        params = {'series_id': series_id}
        if start_date:
            params['observation_start'] = start_date
        if end_date:
            params['observation_end'] = end_date
        
        result = self._request('series/observations', params)
        return result.get('observations', [])
    
    def get_series_categories(self, series_id: str) -> List[Dict]:
        """Get categories for a series."""
        params = {'series_id': series_id}
        result = self._request('series/categories', params)
        return result.get('categories', [])


class FREDDataFetcher:
    """High-level interface for fetching and saving FRED data."""
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = 'txt'):
        """
        Initialize data fetcher.
        
        Args:
            api_key: FRED API key
            output_dir: Directory to save output files
        """
        self.client = FREDAPIClient(api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def fetch_and_save(self, search_term: str, limit: int = 1000, 
                      delay: float = 0.5) -> str:
        """
        Search for series and save to text file.
        
        Args:
            search_term: Search query
            limit: Maximum number of series to fetch
            delay: Delay between requests (to respect API limits)
        
        Returns:
            Path to saved file
        """
        print(f"Querying: {search_term}")
        
        # Search for series
        series_list = self.client.search_series(search_term, limit=limit)
        
        if not series_list:
            print(f"  No results found for '{search_term}'")
            return None
        
        print(f"  Found {len(series_list)} series")
        
        # Create filename
        safe_name = search_term.replace(' ', '_').replace('&', 'and').lower()
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c in ('_', '-'))
        today = datetime.now().strftime("%Y%m%d")
        filename = f"{safe_name}_series_{today}.txt"
        filepath = self.output_dir / filename
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("FRED Series Query Results\n")
            f.write(f"Total Series Found: {len(series_list)}\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, series in enumerate(series_list, 1):
                f.write(f"{idx}. {series.get('id', 'N/A')}\n")
                f.write(f"   Title: {series.get('title', 'N/A')}\n")
                f.write(f"   Frequency: {series.get('frequency_short', 'N/A')}\n")
                f.write(f"   Popularity: {series.get('popularity', 'N/A')}\n")
                
                # Get notes if available
                notes = series.get('notes', '')
                if notes:
                    f.write(f"   Notes:\n   {notes}\n")
                
                f.write("\n" + "-" * 80 + "\n\n")
        
        print(f"  Saved to {filename}")
        
        # Respect API rate limits
        time.sleep(delay)
        
        return str(filepath)
    
    def fetch_bulk(self, search_terms: List[str], limit: int = 1000, 
                   delay: float = 0.5, continue_on_error: bool = True):
        """
        Fetch multiple search terms in bulk.
        
        Args:
            search_terms: List of search queries
            limit: Maximum series per query
            delay: Delay between requests
            continue_on_error: Continue if a query fails
        """
        print(f"\nFetching {len(search_terms)} search terms...")
        print("=" * 70)
        
        successful = 0
        failed = 0
        
        for term in search_terms:
            try:
                result = self.fetch_and_save(term, limit=limit, delay=delay)
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  Error: {e}")
                failed += 1
                if not continue_on_error:
                    raise
            print()  # Blank line for readability
        
        print("=" * 70)
        print(f"Fetch complete: {successful} successful, {failed} failed")
        print(f"Files saved to: {self.output_dir}")


# Predefined economic indicator categories
ECONOMIC_INDICATORS = {
    'core': [
        "GDP", "inflation", "deflation", "CPI", "PPI", "PCE",
        "unemployment", "employment", "labor force", "wages"
    ],
    'monetary': [
        "interest rate", "federal funds rate", "treasury yield",
        "money supply", "M1", "M2", "bank reserves", "credit",
        "mortgage rate", "prime rate"
    ],
    'trade': [
        "exports", "imports", "trade balance", "exchange rate",
        "current account", "trade deficit"
    ],
    'housing': [
        "housing starts", "building permits", "home sales",
        "home prices", "Case-Shiller", "rent", "mortgage"
    ],
    'consumer': [
        "consumer spending", "retail sales", "consumer confidence",
        "personal income", "disposable income", "savings rate"
    ],
    'business': [
        "industrial production", "capacity utilization", "manufacturing",
        "durable goods", "inventory", "productivity"
    ],
    'debt': [
        "household debt", "corporate debt", "government debt", "deficit"
    ],
    'markets': [
        "stock market", "S&P 500", "volatility", "corporate profits",
        "dividend yield"
    ],
    'commodities': [
        "energy prices", "oil prices", "commodity prices",
        "food prices", "gas prices"
    ]
}


def main():
    parser = argparse.ArgumentParser(
        description='Fetch FRED economic data series',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for specific term
  python3 fetch_fred_data.py --search "unemployment rate"
  
  # Search multiple terms
  python3 fetch_fred_data.py --search "GDP" "inflation" "CPI"
  
  # Fetch predefined category
  python3 fetch_fred_data.py --category core
  
  # Fetch all indicators
  python3 fetch_fred_data.py --all
  
  # Fetch with custom limit
  python3 fetch_fred_data.py --search "employment" --limit 20
  
  # List available categories
  python3 fetch_fred_data.py --list-categories

Get your free FRED API key at:
https://fred.stlouisfed.org/docs/api/api_key.html
        """
    )
    
    parser.add_argument('--api-key', type=str,
                       help='FRED API key (or set FRED_API_KEY env var)')
    parser.add_argument('--search', nargs='+', metavar='TERM',
                       help='Search term(s) to query')
    parser.add_argument('--from-file', type=str, metavar='FILE',
                       help='Read search terms from file (one per line)')
    parser.add_argument('--category', choices=list(ECONOMIC_INDICATORS.keys()),
                       help='Fetch predefined category of indicators')
    parser.add_argument('--all', action='store_true',
                       help='Fetch all predefined indicators')
    parser.add_argument('--list-categories', action='store_true',
                       help='List available indicator categories')
    parser.add_argument('--limit', type=int, default=1000,
                       help='Max results per query (default: 1000)')
    parser.add_argument('--output', type=str, default='txt',
                       help='Output directory (default: txt)')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between requests in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    # List categories
    if args.list_categories:
        print("\nAvailable indicator categories:")
        print("=" * 70)
        for category, terms in ECONOMIC_INDICATORS.items():
            print(f"\n{category.upper()} ({len(terms)} indicators):")
            for term in terms:
                print(f"  - {term}")
        print("\n" + "=" * 70)
        return
    
    # Check for API key
    api_key = args.api_key or os.getenv('FRED_API_KEY')
    if not api_key:
        print("Error: FRED API key required")
        print("\nSet via environment variable:")
        print("  export FRED_API_KEY='your-api-key-here'")
        print("\nOr pass as argument:")
        print("  python3 fetch_fred_data.py --api-key 'your-key' --search GDP")
        print("\nGet a free API key at:")
        print("  https://fred.stlouisfed.org/docs/api/api_key.html")
        return
    
    # Initialize fetcher
    fetcher = FREDDataFetcher(api_key=api_key, output_dir=args.output)
    
    # Determine what to fetch
    if args.all:
        # Fetch all indicators
        all_terms = []
        for terms in ECONOMIC_INDICATORS.values():
            all_terms.extend(terms)
        fetcher.fetch_bulk(all_terms, limit=args.limit, delay=args.delay)
    
    elif args.from_file:
        # Read terms from file
        try:
            with open(args.from_file, 'r') as f:
                terms = []
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        terms.append(line)
            
            if not terms:
                print(f"Error: No search terms found in {args.from_file}")
                return
            
            print(f"\nLoaded {len(terms)} search terms from {args.from_file}")
            fetcher.fetch_bulk(terms, limit=args.limit, delay=args.delay)
        except FileNotFoundError:
            print(f"Error: File not found: {args.from_file}")
            return
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    
    elif args.category:
        # Fetch specific category
        terms = ECONOMIC_INDICATORS[args.category]
        print(f"\nFetching {args.category.upper()} indicators...")
        fetcher.fetch_bulk(terms, limit=args.limit, delay=args.delay)
    
    elif args.search:
        # Fetch specific search terms
        fetcher.fetch_bulk(args.search, limit=args.limit, delay=args.delay)
    
    else:
        parser.print_help()
        print("\nPlease specify --search, --category, --from-file, or --all")


if __name__ == '__main__':
    main()
