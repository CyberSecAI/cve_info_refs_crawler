#secondary_processor.py

import os
from typing import Optional
import logging
from .utils.logging_utils import setup_logging
from config import LOG_CONFIG
import re
from pathlib import Path
from typing import Set
from .get_url_content import ContentCrawler
from .get_url_content import find_cve_specific_urls 

class SecondaryProcessor:
    """Process text files to find and fetch CVE-specific URLs"""
    
    def __init__(self, base_dir: str):
        """
        Initialize the secondary processor
        
        Args:
            base_dir: Base directory containing CVE directories
        """
        self.base_dir = Path(base_dir)
        self.logger = setup_logging(
            log_dir=LOG_CONFIG["dir"],
            log_level=LOG_CONFIG["level"],
            module_name=__name__
        )
        self.crawler = ContentCrawler(base_dir)

    def is_secondary_processing_completed(self, cve_id: str) -> bool:
        """
        Check if secondary processing has been completed for a CVE
        
        Args:
            cve_id: CVE ID to check
            
        Returns:
            bool: True if secondary processing is complete
        """
        secondary_links_file = self.base_dir / cve_id / "secondary_links_processed.txt"
        return secondary_links_file.exists()

    def save_secondary_links(self, cve_id: str, urls: Set[str]) -> None:
        """
        Save processed secondary links to file
        
        Args:
            cve_id: CVE ID being processed
            urls: Set of secondary URLs found and processed
        """
        if not urls:
            # Even if no URLs found, create the file to mark processing as complete
            secondary_links_file = self.base_dir / cve_id / "secondary_links_processed.txt"
            secondary_links_file.touch()
            self.logger.info(f"No secondary links found for {cve_id}, marked as processed")
            return

        try:
            secondary_links_file = self.base_dir / cve_id / "secondary_links_processed.txt"
            with open(secondary_links_file, 'w', encoding='utf-8') as f:
                for url in sorted(urls):
                    f.write(f"{url}\n")
            self.logger.info(f"Saved {len(urls)} secondary links for {cve_id}")
        except Exception as e:
            self.logger.error(f"Error saving secondary links for {cve_id}: {str(e)}")

    def process_cve_directory(self, cve_id: str) -> None:
        """
        Process text files in a CVE directory
        
        Args:
            cve_id: CVE ID to process
        """
        # Check if already processed
        if self.is_secondary_processing_completed(cve_id):
            self.logger.info(f"Skipping {cve_id} - secondary processing already completed")
            return

        cve_dir = self.base_dir / cve_id
        text_dir = cve_dir / "text"
        
        if not text_dir.exists():
            self.logger.warning(f"No text directory found for {cve_id}")
            return

        processed_urls = set()
        
        # Process each text file
        for text_file in text_dir.glob("*"):
            try:
                urls = find_cve_specific_urls(
                    text_file=text_file, 
                    target_cve=cve_id,
                    logger=self.logger
                )
                if urls:
                    self.logger.info(f"Found {len(urls)} CVE-specific URLs in {text_file}")
                    
                    for url in urls:
                        if not self.crawler.should_ignore_url(url):
                            self.logger.info(f"Processing secondary URL: {url}")
                            if self.crawler.process_url(url, cve_id):
                                processed_urls.add(url)
                            
            except UnicodeDecodeError as e:
                self.logger.error(f"Unicode decode error processing {text_file}: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error processing {text_file}: {str(e)}", exc_info=True)

        # Save secondary links and mark as processed 
        self.save_secondary_links(cve_id, processed_urls)