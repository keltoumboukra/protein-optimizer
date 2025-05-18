import requests
from typing import List, Dict, Optional
import time
from datetime import datetime, timedelta
import logging
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProteinsAPIClient:
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/proteins/api"
        self.session = requests.Session()
        self.cache = {}
        self.last_request_time = None
        self.min_request_interval = 0.1  # 100ms between requests to respect rate limits
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a request to the API with rate limiting and error handling"""
        url = f"{self.base_url}/{endpoint}"
        
        # Rate limiting
        if self.last_request_time:
            time_since_last = time.time() - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
    
    def search_proteins(self, query: str, organism: Optional[str] = None) -> List[Dict]:
        """Search for proteins with optional organism filter
        
        Args:
            query: Search query (e.g., "insulin")
            organism: Optional organism filter (e.g., "9606" for Homo sapiens)
            
        Returns:
            List of protein entries matching the search criteria
        """
        # Construct the search query according to API documentation
        search_query = query
        if organism:
            # Use taxonomy ID for organism filtering
            search_query = f'name:{query} AND taxonomy:{organism}'
            
        params = {
            "query": search_query,
            "format": "json",
            "size": 10  # Limit results to 10 for testing
        }
            
        try:
            results = self._make_request("proteins/search", params=params)
            return results.get("results", [])
        except Exception as e:
            logger.error(f"Protein search failed: {str(e)}")
            return []
    
    def get_protein_details(self, accession: str) -> Dict:
        """Get detailed protein information
        
        Args:
            accession: Protein accession number
            
        Returns:
            Dictionary containing detailed protein information
        """
        try:
            return self._make_request(f"proteins/{accession}")
        except Exception as e:
            logger.error(f"Failed to get protein details for {accession}: {str(e)}")
            return {}
    
    def get_expression_data(self, accession: str) -> List[Dict]:
        """Get expression data for a protein
        
        Args:
            accession: Protein accession number
            
        Returns:
            List of expression data entries
        """
        try:
            return self._make_request(f"proteins/{accession}/expression")
        except Exception as e:
            logger.error(f"Failed to get expression data for {accession}: {str(e)}")
            return []
    
    def get_protein_features(self, accession: str) -> List[Dict]:
        """Get protein features and domains
        
        Args:
            accession: Protein accession number
            
        Returns:
            List of protein features
        """
        try:
            return self._make_request(f"proteins/{accession}/features")
        except Exception as e:
            logger.error(f"Failed to get protein features for {accession}: {str(e)}")
            return [] 