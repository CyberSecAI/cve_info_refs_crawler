import csv
from urllib.parse import urlparse
from collections import Counter

def extract_domain(url):
    """Extract domain from URL."""
    try:
        # Remove 'https://' or 'http://' and get the domain
        domain = urlparse(url).netloc
        return domain if domain else None
    except:
        return None

def count_domains_from_csv(file_path):
    """Read CSV file and count domains from URLs."""
    domain_counts = Counter()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Ensure 'URL' column exists
            if 'URL' not in reader.fieldnames:
                raise ValueError("CSV file must contain a 'URL' column")
            
            # Process each row
            for row in reader:
                url = row['URL']
                domain = extract_domain(url)
                if domain:
                    domain_counts[domain] += 1
    
        # Sort domains by count (descending) and then alphabetically
        sorted_counts = sorted(domain_counts.items(), 
                             key=lambda x: (-x[1], x[0]))
        
        return sorted_counts
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return []
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return []

def write_results(counts, output_file='domain_counts.csv'):
    """Write domain counts to CSV file."""
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['domain', 'count'])
            writer.writerows(counts)
        print(f"Results written to {output_file}")
    except Exception as e:
        print(f"Error writing results: {str(e)}")

def main():
    input_file = './data_out/failed_urls.csv'  # Change this to your input file name
    output_file = './data_out/failed_domain_counts.csv'
    
    # Process the CSV and get domain counts
    domain_counts = count_domains_from_csv(input_file)
    
    if domain_counts:
        # Write results to CSV
        write_results(domain_counts, output_file)
        
        # Also print results to console
        print("\nDomain counts:")
        for domain, count in domain_counts:
            print(f"{domain},{count}")

if __name__ == "__main__":
    main()