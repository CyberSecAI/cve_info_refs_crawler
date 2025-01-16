import json
import csv

def extract_cves(json_content):
    # Parse the JSON content
    data = json.loads(json_content)
    
    # Check if it's a single CVE entry or multiple entries
    if isinstance(data, dict) and 'cve' in data:
        # Single CVE entry
        cves = [data['cve']['id']]
    elif isinstance(data, list):
        # Multiple CVE entries
        cves = [item['cve']['id'] for item in data if 'cve' in item]
    else:
        raise ValueError("Unexpected JSON structure")
    
    # Sort CVEs alphanumerically
    cves.sort()
    
    return cves

def save_to_csv(cves, output_file='./data_in/cves.csv'):
    # Write to CSV with header
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['CVE'])  # Header
        for cve in cves:
            writer.writerow([cve])

# Read the JSON file
try: 
    with open('data_in/nvd.jsonl', 'r') as f:
        json_content = f.read()
        
    # Extract and sort CVEs
    cves = extract_cves(json_content)
    
    # Save to CSV
    save_to_csv(cves)
    
    print(f"Successfully extracted {len(cves)} CVEs and saved to cves.csv")
    
except Exception as e:
    print(f"Error: {str(e)}")