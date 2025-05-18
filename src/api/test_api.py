import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.proteins_api import ProteinsAPIClient
import json
from typing import Dict, List, Union
import time

def print_json(data: Union[Dict, List]) -> None:
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))

def test_api_functionality():
    client = ProteinsAPIClient()
    
    # Test 1: Search for proteins
    print("\n=== Testing Protein Search ===")
    # Use taxonomy ID 9606 for Homo sapiens
    search_results = client.search_proteins("insulin", organism="9606")
    print(f"Found {len(search_results)} results")
    if search_results:
        print("First result:")
        print_json(search_results[0])
        
        # Get the accession number from the first result
        accession = search_results[0].get("accession")
        if accession:
            print(f"\nAccession number: {accession}")
            
            # Test 2: Get protein details
            print(f"\n=== Testing Protein Details for {accession} ===")
            details = client.get_protein_details(accession)
            if details:
                print("Protein details retrieved successfully")
                print_json(details)
            else:
                print("Failed to retrieve protein details")
            
            # Test 3: Get expression data
            print(f"\n=== Testing Expression Data for {accession} ===")
            expression_data = client.get_expression_data(accession)
            if expression_data:
                print("Expression data retrieved successfully")
                print_json(expression_data)
            else:
                print("No expression data available")
            
            # Test 4: Get protein features
            print(f"\n=== Testing Protein Features for {accession} ===")
            features = client.get_protein_features(accession)
            if features:
                print("Protein features retrieved successfully")
                print_json(features)
            else:
                print("No protein features available")

if __name__ == "__main__":
    test_api_functionality() 