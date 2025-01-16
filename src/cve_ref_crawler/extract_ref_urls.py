# src/cve_ref_crawler/extract_ref_urls.py

import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Iterator
from .utils.file_utils import ensure_directory
from .utils.logging_utils import setup_logging
from .utils.cve_filter import load_target_cves
from config import LOG_CONFIG

class CVEProcessor:
    def __init__(self, input_file: str, output_dir: str, target_cves: Optional[Set[str]] = None):
        """
        Initialize the CVE processor
        
        Args:
            input_file: Path to input JSON file
            output_dir: Path to output directory
            target_cves: Optional set of CVE IDs to process
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.target_cves = target_cves
        self.logger = setup_logging(
            log_dir=LOG_CONFIG["dir"],
            log_level=LOG_CONFIG["level"],
            module_name=__name__
        )
        
        if target_cves:
            self.logger.info(f"Initialized with {len(target_cves)} target CVEs")

    def should_process_cve_links(self, cve_id: str) -> bool:
        """Check if links.txt needs to be generated"""
        links_file = self.output_dir / cve_id / "links.txt"
        if links_file.exists():
            self.logger.info(f"Skipping {cve_id} - links.txt already exists")
            return False
        return True
    
    def process_cve(self, cve_id: str) -> bool:
        """Process all text files for a CVE and extract vulnerability information"""
        # Check if already processed
        if self.is_cve_extracted(cve_id):
            self.logger.info(f"Skipping {cve_id} - vulnerability info already extracted")
            return False
        
    def read_json_content(self) -> Iterator[Dict]:
        """
        Read and parse JSON content from file
        
        Yields:
            Dict: Each CVE entry from the JSON file
        """
        try:
            self.logger.info(f"Reading JSON file: {self.input_file}")
            
            if not self.input_file.exists():
                self.logger.error(f"Input file does not exist: {self.input_file}")
                return
                
            with open(self.input_file, 'r') as f:
                # Read first character to determine format
                first_char = f.read(1)
                f.seek(0)  # Reset file pointer
                
                if first_char == '[':
                    # File is a JSON array
                    self.logger.debug("Processing as JSON array")
                    data = json.load(f)
                    for entry in data:
                        if isinstance(entry, dict) and 'cve' in entry:
                            yield entry
                        else:
                            self.logger.debug(f"Skipping invalid entry: {str(entry)[:100]}...")
                else:
                    # File is JSONL format
                    self.logger.debug("Processing as JSONL")
                    for line_num, line in enumerate(f, 1):
                        try:
                            if not line.strip():
                                continue
                            
                            entry = json.loads(line.strip())
                            if isinstance(entry, dict) and 'cve' in entry:
                                yield entry
                            else:
                                self.logger.debug(f"Line {line_num}: Invalid entry format")
                                
                        except json.JSONDecodeError as e:
                            self.logger.error(f"JSON decode error at line {line_num}: {e}")
                            self.logger.debug(f"Problematic line: {line[:200]}...")
                            continue
                            
        except Exception as e:
            self.logger.error(f"Error reading JSON file: {e}", exc_info=True)
            
        
    def create_output_directories(self, cve_id: str) -> Path:
        """
        Create directory for CVE if it doesn't exist
        
        Args:
            cve_id: CVE identifier
            
        Returns:
            Path object for the created directory
        """
        cve_dir = self.output_dir / cve_id
        ensure_directory(cve_dir)
        self.logger.debug(f"Created directory: {cve_dir}")
        return cve_dir
        
    def extract_urls(self, references: List[Dict]) -> Set[str]:
        """
        Extract unique URLs from references
        
        Args:
            references: List of reference dictionaries
            
        Returns:
            Set of unique URLs
        """
        urls = {ref['url'] for ref in references if 'url' in ref}
        self.logger.debug(f"Extracted {len(urls)} unique URLs")
        return urls
        
    def save_urls(self, urls: Set[str], output_path: Path) -> None:
        """
        Save URLs to links.txt file
        
        Args:
            urls: Set of URLs to save
            output_path: Directory path to save links.txt
        """
        links_file = output_path / 'links.txt'
        with open(links_file, 'w') as f:
            for url in sorted(urls):
                f.write(f"{url}\n")
        self.logger.info(f"Saved {len(urls)} URLs to {links_file}")
    
    def should_process_cve(self, cve_id: str) -> bool:
        """
        Check if CVE should be processed based on target list
        
        Args:
            cve_id: CVE ID to check
            
        Returns:
            bool: True if CVE should be processed
        """
        if self.target_cves is None:
            return True
        return cve_id in self.target_cves
                
    def process_file(self) -> None:
        """Process as before but with better error handling and description storage"""
        ensure_directory(self.output_dir)
        
        self.logger.info(f"Starting to process file: {self.input_file}")
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        found_cves = set()
        descriptions = {}
        
        try:
            # Process each CVE entry
            for entry in self.read_json_content():
                try:
                    cve_data = entry.get('cve')
                    if not cve_data:
                        continue
                        
                    cve_id = cve_data.get('id')
                    if not cve_id:
                        continue
                    
                    if not self.should_process_cve_links(cve_id):
                        continue
                    
                    self.logger.debug(f"Found CVE ID: {cve_id}")
                    found_cves.add(cve_id)
                    
                    # Store CVE description
                    for desc in cve_data.get('descriptions', []):
                        if desc.get('lang') == 'en':
                            descriptions[cve_id] = desc.get('value')
                            break
                    
                    if not self.should_process_cve(cve_id):
                        skipped_count += 1
                        self.logger.debug(f"Skipping {cve_id} (not in target list)")
                        continue
                        
                    references = cve_data.get('references', [])
                    if not references:
                        self.logger.info(f"No references found for {cve_id}")
                        continue
                    
                    self.logger.info(f"Processing {cve_id}")
                    cve_dir = self.create_output_directories(cve_id)
                    
                    # Save description to CVE directory
                    if cve_id in descriptions:
                        try:
                            with open(cve_dir / 'description.txt', 'w', encoding='utf-8') as f:
                                f.write(descriptions[cve_id])
                        except Exception as e:
                            self.logger.error(f"Error saving description for {cve_id}: {e}")
                    
                    urls = self.extract_urls(references)
                    self.save_urls(urls, cve_dir)
                    self.logger.info(f"Successfully processed {cve_id} with {len(urls)} URLs")
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing entry: {e}")
                    error_count += 1
            
            # Save all descriptions to central file
            try:
                with open(self.output_dir / 'cve_descriptions.json', 'w', encoding='utf-8') as f:
                    json.dump(descriptions, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Saved {len(descriptions)} descriptions to central file")
            except Exception as e:
                self.logger.error(f"Error saving descriptions to central file: {e}")
            
            # Log statistics about found CVEs
            self.logger.info(f"Found {len(found_cves)} total CVEs in input file")
            if self.target_cves:
                found_targets = found_cves.intersection(self.target_cves)
                self.logger.info(f"Found {len(found_targets)} target CVEs out of {len(self.target_cves)} targets")
            
        except Exception as e:
            self.logger.error(f"Error in process_file: {e}", exc_info=True)
            
        self.logger.info(
            f"Processing completed. "
            f"Processed: {processed_count}, "
            f"Skipped: {skipped_count}, "
            f"Errors: {error_count}"
        )