import os
import json
import requests
from typing import Optional, Dict, Any, List

class FREDClient:
    """A reusable client for the FRED API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.stlouisfed.org/fred"
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set FRED_API_KEY or pass api_key parameter.")
    
    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict:
        """Make a request to the FRED API"""
        p = dict(params or {})
        p.setdefault("file_type", "json")
        p["api_key"] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, params=p, timeout=timeout)
        response.raise_for_status()
        
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"Non-JSON response. Content-Type: {response.headers.get('Content-Type')}")
            print(response.text[:500])
            return None
    
    def get_series(self, series_id: str) -> Dict:
        """Get metadata for a specific series"""
        return self._request("series", {"series_id": series_id})
    
    def get_observations(self, series_id: str, 
                        observation_start: Optional[str] = None,
                        observation_end: Optional[str] = None,
                        **kwargs) -> Dict:
        """Get observations (data points) for a series"""
        params = {"series_id": series_id}
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        params.update(kwargs)
        return self._request("series/observations", params)
    
    def search_series(self, search_text: str, 
                     limit: int = 10,
                     order_by: str = "search_rank",
                     sort_order: str = "desc",
                     **kwargs) -> Dict:
        """Search for series by keyword"""
        params = {
            "search_text": search_text,
            "limit": limit,
            "order_by": order_by,
            "sort_order": sort_order
        }
        params.update(kwargs)
        return self._request("series/search", params)
    
    def get_categories(self, category_id: Optional[int] = None) -> Dict:
        """Get categories or a specific category"""
        endpoint = "category" if category_id else "category"
        params = {"category_id": category_id} if category_id else {}
        return self._request(endpoint, params)
    
    def get_releases(self, **kwargs) -> Dict:
        """Get all releases"""
        return self._request("releases", kwargs)
    
    def save_to_json(self, data: Any, filename: str):
        """Save data to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {filename}")
    
    @staticmethod
    def load_from_json(filename: str) -> Any:
        """Load data from a JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)
