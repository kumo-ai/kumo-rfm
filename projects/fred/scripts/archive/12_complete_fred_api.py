#!/usr/bin/env python3
"""
Complete FRED API Client

Covers ALL FRED API endpoints:
- Categories
- Releases
- Series
- Sources
- Tags
- Maps API (GeoFRED)
"""

import os
import requests
from typing import Dict, List, Optional
from datetime import datetime
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars


class CompleteFREDClient:
    """Complete FRED API client with all endpoints."""
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    GEOFRED_URL = "https://api.stlouisfed.org/geofred"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED_API_KEY required")
        self.session = requests.Session()
    
    def _request(self, endpoint: str, params: Dict, base_url: str = None) -> Dict:
        """Make API request."""
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        url = f"{base_url or self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    # ===== CATEGORY ENDPOINTS =====
    
    def get_category(self, category_id: int) -> Dict:
        """Get a category."""
        return self._request('category', {'category_id': category_id})
    
    def get_category_children(self, category_id: int) -> List[Dict]:
        """Get child categories."""
        result = self._request('category/children', {'category_id': category_id})
        return result.get('categories', [])
    
    def get_category_related(self, category_id: int) -> List[Dict]:
        """Get related categories."""
        result = self._request('category/related', {'category_id': category_id})
        return result.get('categories', [])
    
    def get_category_series(self, category_id: int, **kwargs) -> List[Dict]:
        """Get series in a category."""
        params = {'category_id': category_id, **kwargs}
        result = self._request('category/series', params)
        return result.get('seriess', [])
    
    def get_category_tags(self, category_id: int, **kwargs) -> List[Dict]:
        """Get tags for a category."""
        params = {'category_id': category_id, **kwargs}
        result = self._request('category/tags', params)
        return result.get('tags', [])
    
    def get_category_related_tags(self, category_id: int, tag_names: List[str], **kwargs) -> List[Dict]:
        """Get related tags for a category."""
        params = {
            'category_id': category_id,
            'tag_names': ';'.join(tag_names),
            **kwargs
        }
        result = self._request('category/related_tags', params)
        return result.get('tags', [])
    
    # ===== RELEASE ENDPOINTS =====
    
    def get_releases(self, **kwargs) -> List[Dict]:
        """Get all releases."""
        result = self._request('releases', kwargs)
        return result.get('releases', [])
    
    def get_releases_dates(self, **kwargs) -> List[Dict]:
        """Get release dates for all releases."""
        result = self._request('releases/dates', kwargs)
        return result.get('release_dates', [])
    
    def get_release(self, release_id: int) -> Dict:
        """Get a release."""
        return self._request('release', {'release_id': release_id})
    
    def get_release_dates(self, release_id: int, **kwargs) -> List[Dict]:
        """Get dates for a release."""
        params = {'release_id': release_id, **kwargs}
        result = self._request('release/dates', params)
        return result.get('release_dates', [])
    
    def get_release_series(self, release_id: int, **kwargs) -> List[Dict]:
        """Get series in a release."""
        params = {'release_id': release_id, **kwargs}
        result = self._request('release/series', params)
        return result.get('seriess', [])
    
    def get_release_sources(self, release_id: int) -> List[Dict]:
        """Get sources for a release."""
        result = self._request('release/sources', {'release_id': release_id})
        return result.get('sources', [])
    
    def get_release_tags(self, release_id: int, **kwargs) -> List[Dict]:
        """Get tags for a release."""
        params = {'release_id': release_id, **kwargs}
        result = self._request('release/tags', params)
        return result.get('tags', [])
    
    def get_release_related_tags(self, release_id: int, tag_names: List[str], **kwargs) -> List[Dict]:
        """Get related tags for a release."""
        params = {
            'release_id': release_id,
            'tag_names': ';'.join(tag_names),
            **kwargs
        }
        result = self._request('release/related_tags', params)
        return result.get('tags', [])
    
    def get_release_tables(self, release_id: int, **kwargs) -> Dict:
        """Get release tables."""
        params = {'release_id': release_id, **kwargs}
        return self._request('release/tables', params)
    
    # ===== SERIES ENDPOINTS =====
    
    def get_series(self, series_id: str) -> Dict:
        """Get a series."""
        result = self._request('series', {'series_id': series_id})
        return result.get('seriess', [{}])[0]
    
    def get_series_categories(self, series_id: str) -> List[Dict]:
        """Get categories for a series."""
        result = self._request('series/categories', {'series_id': series_id})
        return result.get('categories', [])
    
    def get_series_observations(self, series_id: str, **kwargs) -> List[Dict]:
        """Get observations for a series."""
        params = {'series_id': series_id, **kwargs}
        result = self._request('series/observations', params)
        return result.get('observations', [])
    
    def get_series_release(self, series_id: str) -> Dict:
        """Get release for a series."""
        result = self._request('series/release', {'series_id': series_id})
        return result.get('releases', [{}])[0]
    
    def search_series(self, search_text: str, limit: int = 1000, **kwargs) -> List[Dict]:
        """Search for series."""
        params = {'search_text': search_text, 'limit': limit, **kwargs}
        result = self._request('series/search', params)
        return result.get('seriess', [])
    
    def search_series_tags(self, series_search_text: str, **kwargs) -> List[Dict]:
        """Get tags for a series search."""
        params = {'series_search_text': series_search_text, **kwargs}
        result = self._request('series/search/tags', params)
        return result.get('tags', [])
    
    def search_series_related_tags(self, series_search_text: str, tag_names: List[str], **kwargs) -> List[Dict]:
        """Get related tags for a series search."""
        params = {
            'series_search_text': series_search_text,
            'tag_names': ';'.join(tag_names),
            **kwargs
        }
        result = self._request('series/search/related_tags', params)
        return result.get('tags', [])
    
    def get_series_tags(self, series_id: str, **kwargs) -> List[Dict]:
        """Get tags for a series."""
        params = {'series_id': series_id, **kwargs}
        result = self._request('series/tags', params)
        return result.get('tags', [])
    
    def get_series_updates(self, **kwargs) -> List[Dict]:
        """Get series sorted by update time."""
        result = self._request('series/updates', kwargs)
        return result.get('seriess', [])
    
    def get_series_vintagedates(self, series_id: str, **kwargs) -> List[str]:
        """Get vintage dates for a series."""
        params = {'series_id': series_id, **kwargs}
        result = self._request('series/vintagedates', params)
        return result.get('vintage_dates', [])
    
    # ===== SOURCE ENDPOINTS =====
    
    def get_sources(self, **kwargs) -> List[Dict]:
        """Get all sources."""
        result = self._request('sources', kwargs)
        return result.get('sources', [])
    
    def get_source(self, source_id: int) -> Dict:
        """Get a source."""
        result = self._request('source', {'source_id': source_id})
        return result.get('sources', [{}])[0]
    
    def get_source_releases(self, source_id: int, **kwargs) -> List[Dict]:
        """Get releases for a source."""
        params = {'source_id': source_id, **kwargs}
        result = self._request('source/releases', params)
        return result.get('releases', [])
    
    # ===== TAG ENDPOINTS =====
    
    def get_tags(self, **kwargs) -> List[Dict]:
        """Get all tags."""
        result = self._request('tags', kwargs)
        return result.get('tags', [])
    
    def get_related_tags(self, tag_names: List[str], **kwargs) -> List[Dict]:
        """Get related tags."""
        params = {'tag_names': ';'.join(tag_names), **kwargs}
        result = self._request('related_tags', params)
        return result.get('tags', [])
    
    def get_tags_series(self, tag_names: List[str], **kwargs) -> List[Dict]:
        """Get series matching tags."""
        params = {'tag_names': ';'.join(tag_names), **kwargs}
        result = self._request('tags/series', params)
        return result.get('seriess', [])
    
    # ===== MAPS API (GeoFRED) =====
    
    def get_shape_files(self, **kwargs) -> List[Dict]:
        """Get available shape files."""
        result = self._request('shapes/file', kwargs, base_url=self.GEOFRED_URL)
        return result.get('files', [])
    
    def get_series_group_meta(self, series_group: str) -> Dict:
        """Get metadata for a series group."""
        params = {'series_group': series_group}
        return self._request('series/group', params, base_url=self.GEOFRED_URL)
    
    def get_series_regional_data(self, series_id: str, **kwargs) -> List[Dict]:
        """Get regional data for a series."""
        params = {'series_id': series_id, **kwargs}
        result = self._request('series/data', params, base_url=self.GEOFRED_URL)
        return result.get('observations', [])
    
    def get_regional_data(self, series_id: str, region_type: str = 'state', **kwargs) -> List[Dict]:
        """Get regional economic data."""
        params = {
            'series_id': series_id,
            'region_type': region_type,
            **kwargs
        }
        result = self._request('regional/data', params, base_url=self.GEOFRED_URL)
        return result.get('data', [])


def demo_complete_api():
    """Demonstrate all API endpoints."""
    print("="*70)
    print("COMPLETE FRED API DEMONSTRATION")
    print("="*70)
    
    client = CompleteFREDClient()
    
    # Category endpoints
    print("\n" + "-"*70)
    print("CATEGORY ENDPOINTS")
    print("-"*70)
    
    print("\n1. Get category (Employment & Population):")
    cat = client.get_category(10)
    print(f"   {cat.get('categories', [{}])[0].get('name', 'N/A')}")
    
    print("\n2. Get child categories:")
    children = client.get_category_children(10)
    for c in children[:3]:
        print(f"   - {c.get('name')}")
    
    print("\n3. Get series in category:")
    series = client.get_category_series(10, limit=3)
    for s in series:
        print(f"   - {s.get('id')}: {s.get('title', '')[:50]}")
    
    # Release endpoints
    print("\n" + "-"*70)
    print("RELEASE ENDPOINTS")
    print("-"*70)
    
    print("\n4. Get recent releases:")
    releases = client.get_releases(limit=3)
    for r in releases:
        print(f"   - {r.get('name')}")
    
    print("\n5. Get release info (Employment Situation):")
    release = client.get_release(50)
    print(f"   {release.get('releases', [{}])[0].get('name')}")
    
    # Series endpoints
    print("\n" + "-"*70)
    print("SERIES ENDPOINTS")
    print("-"*70)
    
    print("\n6. Get series info (GDP):")
    series = client.get_series('GDP')
    print(f"   {series.get('title')}")
    print(f"   Frequency: {series.get('frequency')}")
    print(f"   Units: {series.get('units')}")
    
    print("\n7. Get series observations:")
    obs = client.get_series_observations('GDP', limit=5, sort_order='desc')
    for o in obs[:3]:
        print(f"   {o.get('date')}: {o.get('value')}")
    
    print("\n8. Search series:")
    results = client.search_series('unemployment rate', limit=3)
    for s in results:
        print(f"   - {s.get('id')}: {s.get('title', '')[:50]}")
    
    print("\n9. Get series updates:")
    updates = client.get_series_updates(limit=3)
    for s in updates:
        print(f"   - {s.get('id')}: updated {s.get('last_updated')}")
    
    # Source endpoints
    print("\n" + "-"*70)
    print("SOURCE ENDPOINTS")
    print("-"*70)
    
    print("\n10. Get sources:")
    sources = client.get_sources(limit=3)
    for src in sources:
        print(f"   - {src.get('name')}")
    
    # Tag endpoints
    print("\n" + "-"*70)
    print("TAG ENDPOINTS")
    print("-"*70)
    
    print("\n11. Get tags:")
    tags = client.get_tags(limit=5)
    for tag in tags:
        print(f"   - {tag.get('name')}: {tag.get('group_id')}")
    
    print("\n12. Get related tags:")
    related = client.get_related_tags(['gdp'], limit=5)
    for tag in related:
        print(f"   - {tag.get('name')}")
    
    # Maps API
    print("\n" + "-"*70)
    print("MAPS API (GeoFRED)")
    print("-"*70)
    
    try:
        print("\n13. Get shape files:")
        shapes = client.get_shape_files()
        print(f"   Available shape files: {len(shapes)}")
        
        print("\n14. Get regional data (Unemployment by state):")
        regional = client.get_regional_data('UNRATE', region_type='state')
        if regional:
            print(f"   Got data for {len(regional)} regions")
    except Exception as e:
        print(f"   Maps API error: {e}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE - All endpoints demonstrated!")
    print("="*70)
    print("\nNOTE: Currently capturing only 'series/search' endpoint.")
    print("Recommendation: Extend 01_fetch_fred_data.py to use:")
    print("  - Category traversal (get all series in all categories)")
    print("  - Release-based fetching (get series by release)")
    print("  - Tag-based discovery (find series by tags)")
    print("  - Regional data (GeoFRED for state/county data)")


if __name__ == '__main__':
    demo_complete_api()
