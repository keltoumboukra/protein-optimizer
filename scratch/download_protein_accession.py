import requests
import csv
import time

# Define the API endpoint and parameters
url = "https://rest.uniprot.org/uniprotkb/search"
params = {
    "query": 'organism_id:9606 AND reviewed:true',
    "format": "tsv",
    "fields": "accession,protein_name,organism_name",
    "size": 500  # UniProt max per request
}

all_rows = []
cursor = None
batch = 0

while True:
    if cursor:
        params['cursor'] = cursor
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        break
    lines = response.text.strip().split('\n')
    if batch == 0:
        all_rows.append(lines[0])  # header
    all_rows.extend(lines[1:])
    print(f"Downloaded batch {batch+1}, {len(lines)-1} proteins.")
    # Check for next cursor
    next_cursor = response.headers.get('X-Next-Page-Cursor')
    if not next_cursor or len(lines) <= 1:
        break
    cursor = next_cursor
    batch += 1
    time.sleep(0.5)  # be polite to UniProt

# Write the response content to a TSV file
with open("human_reviewed_proteins.tsv", "w", newline='', encoding='utf-8') as file:
    file.write('\n'.join(all_rows))
print(f"TSV file has been created successfully with {len(all_rows)-1} proteins.")
