# src/cve_ref_crawler/get_url_content.py

import os
from pathlib import Path
import requests
from urllib.parse import urlparse
import hashlib
from datetime import datetime
from bs4 import BeautifulSoup
import PyPDF2
import io
import re
import pandas as pd
from typing import Optional, Dict, Tuple
from markitdown import MarkItDown
from markitdown._markitdown import FileConversionException
from tqdm import tqdm
from .utils.file_utils import ensure_directory
from .utils.logging_utils import setup_logging
from .handlers.googlesource import is_googlesource_url, handle_googlesource_url, parse_googlesource_response
from .handlers.cisa import is_cisa_url, handle_cisa_url, parse_cisa_response
from .handlers.youtube import is_youtube_url, handle_youtube_url  
from config import LOG_CONFIG, CRAWLER_SETTINGS, IGNORED_URLS, DEAD_DOMAINS_CSV
from .utils.domain_stats import DomainStatsCollector  
from .tracking import ProcessingTracker, URLStatus 
import urllib.robotparser
from urllib.parse import urlparse
from functools import lru_cache
import time

class RobotsParser:
    def __init__(self, user_agent):
        """Initialize robots parser with user agent"""
        self.user_agent = user_agent
        self.parsers = {}
        self.last_checked = {}
        self.cache_duration = CRAWLER_SETTINGS["robots_txt"]["cache_duration"]
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def get_robots_parser(self, url: str) -> urllib.robotparser.RobotFileParser:
        """Get or create robots parser for a given URL"""
        base_url = self.get_base_url(url)
        
        # Check if we need to refresh the cached parser
        now = time.time()
        if base_url in self.parsers:
            if now - self.last_checked[base_url] > self.cache_duration:
                del self.parsers[base_url]
                del self.last_checked[base_url]

        # Create new parser if needed
        if base_url not in self.parsers:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = f"{base_url}/robots.txt"
            rp.set_url(robots_url)
            
            try:
                # Fetch robots.txt with configured timeout
                response = self.session.get(
                    robots_url, 
                    timeout=CRAWLER_SETTINGS["robots_txt"]["fetch_timeout"]
                )
                if response.status_code == 200:
                    rp.parse(response.text.splitlines())
                else:
                    return None
                    
                self.parsers[base_url] = rp
                self.last_checked[base_url] = now
                
            except (requests.RequestException, Exception) as e:
                self.logger.warning(f"Error fetching robots.txt for {base_url}: {str(e)}")
                return None

        return self.parsers.get(base_url)

    def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        try:
            parser = self.get_robots_parser(url)
            if parser is None:
                return True
            
            # Add configured timeout for the actual robots.txt check
            with timeout(CRAWLER_SETTINGS["robots_txt"]["rule_check_timeout"]):
                return parser.can_fetch(self.user_agent, url)
                
        except Exception as e:
            self.logger.warning(f"Error checking robots.txt for {url}: {str(e)}")
            return True

# Add a context manager for timeout
from contextlib import contextmanager
import signal

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out")
    
    # Set the timeout handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm
        """Check if URL can be fetched according to robots.txt"""
        try:
            parser = self.get_robots_parser(url)
            if parser is None:  # No robots.txt available
                return True
            return parser.can_fetch(self.user_agent, url)
        except Exception as e:
            # On error, assume we can fetch
            return True
        
        
class ContentCrawler:
    def __init__(self, output_dir: str):
        """Initialize the content crawler"""
        self.output_dir = Path(output_dir)
        self.session = requests.Session()
        self.session.headers.update(CRAWLER_SETTINGS["headers"])
               
        # Initialize tracking
        self.tracker = ProcessingTracker(self.output_dir)
        
        # Initialize robots parser with our user agent
        self.robots = RobotsParser(
            CRAWLER_SETTINGS["headers"]["User-Agent"]
        )
        
        self.logger = setup_logging(
            log_dir=LOG_CONFIG["dir"],
            log_level=LOG_CONFIG["level"],
            module_name=__name__
        )
        self.md_converter = MarkItDown()
        
        # Load dead domains
        self.dead_domains = set()
        self._load_dead_domains()
        
        self.domain_stats = DomainStatsCollector(
            Path(output_dir),
            IGNORED_URLS,
            self.dead_domains
        )

    def should_process_raw_content(self, cve_id: str) -> bool:
        """Check if raw content needs to be downloaded"""
        raw_dir = self.output_dir / cve_id / "raw"
        if raw_dir.exists() and any(raw_dir.iterdir()):
            self.logger.info(f"Skipping {cve_id} - raw content already exists")
            return False
        return True

    def should_process_text_content(self, cve_id: str) -> bool:
        """Check if text conversion is needed"""
        text_dir = self.output_dir / cve_id / "text"
        if text_dir.exists() and any(text_dir.iterdir()):
            self.logger.info(f"Skipping {cve_id} - text content already exists")
            return False
        return True

 
    def _convert_content(self, filepath: Path) -> Tuple[Optional[str], str]:
        """
        Convert content using MarkItDown with fallback methods
        
        Args:
            filepath: Path to file to convert
            
        Returns:
            Tuple[Optional[str], str]: (converted content, method used)
        """
        # Try MarkItDown first
        try:
            result = self.md_converter.convert(str(filepath))
            return result.text_content, "markitdown"
        except FileConversionException as e:
            self.logger.warning(f"MarkItDown conversion failed: {str(e)}")
        except Exception as e:
            self.logger.warning(f"Unexpected error in MarkItDown conversion: {str(e)}")

        # Fallback for PDFs using PyPDF2
        if str(filepath).lower().endswith('.pdf'):
            try:
                self.logger.info("Attempting PDF conversion with PyPDF2")
                with open(filepath, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = []
                    for page in pdf_reader.pages:
                        try:
                            text.append(page.extract_text())
                        except Exception as e:
                            self.logger.warning(f"Error extracting page text: {str(e)}")
                            continue
                    if text:
                        return "\n".join(text), "pypdf2"
            except Exception as e:
                self.logger.warning(f"PyPDF2 conversion failed: {str(e)}")

        # If all methods fail
        self.logger.error(f"All conversion methods failed for {filepath}")
        return None, "none"

    def _load_dead_domains(self):
        """Load dead domains from CSV if it exists"""
        try:
            if Path(DEAD_DOMAINS_CSV).exists():
                df = pd.read_csv(DEAD_DOMAINS_CSV)
                if 'Domain' in df.columns:
                    self.dead_domains = set(df['Domain'].str.lower())
                    self.logger.info(f"Loaded {len(self.dead_domains)} dead domains from CSV")
            else:
                self.logger.debug("Dead domains CSV file not found")
        except Exception as e:
            self.logger.error(f"Error loading dead domains CSV: {str(e)}")
            

    def should_ignore_url(self, url: str) -> bool:
        """
        Check if URL should be ignored based on rules and robots.txt
        
        Args:
            url: URL to check
            
        Returns:
            bool: True if URL should be ignored
        """
        # First check existing ignore rules
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc.lower()
        
        # Remove 'www.' prefix if present for comparison
        if hostname.startswith('www.'):
            hostname = hostname[4:]
            
        # Check against IGNORED_URLS
        for ignored in IGNORED_URLS:
            ignored = ignored.lower()
            if ignored.startswith('www.'):
                ignored = ignored[4:]
            if ignored in hostname:
                self.logger.info(f"Ignoring URL {url} (matches ignore pattern: {ignored})")
                return True
                
        # Check against dead domains
        if hostname in self.dead_domains or f"www.{hostname}" in self.dead_domains:
            self.logger.info(f"Ignoring URL {url} (matches dead domain)")
            return True
                
        # By default, allow the URL
        return False

    def is_cve_processed(self, cve_id: str) -> bool:
        """
        Check if CVE has already been processed by checking text directory
        
        Args:
            cve_id: CVE ID to check
            
        Returns:
            bool: True if CVE has already been processed
        """
        raw_dir = self.output_dir / cve_id / "raw"
        text_dir = self.output_dir / cve_id / "text"
        
        # Consider processed if both raw and text directories exist and contain files
        if raw_dir.exists() and text_dir.exists():
            has_raw = any(raw_dir.iterdir())
            has_text = any(text_dir.iterdir())
            if has_raw and has_text:
                return True
                
        return False

    def _generate_filename(self, url: str, content_type: str) -> str:
        """Generate a unique filename based on URL and timestamp"""
        parsed_url = urlparse(url)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        base = f"{parsed_url.netloc}_{url_hash}_{timestamp}"
        if content_type == "pdf":
            return f"{base}.pdf"
        return f"{base}.html"

    def _save_raw_content(self, content: str | bytes, url: str, cve_id: str, content_type: str = "html") -> Optional[Path]:
        """Save the raw content to a file"""
        raw_dir = self.output_dir / cve_id / "raw"
        ensure_directory(raw_dir)
        
        if is_youtube_url(url):
            # For YouTube URLs, use .txt extension since we're saving transcript
            content_type = "txt"
        
        filename = self._generate_filename(url, content_type)
        filepath = raw_dir / filename
        
        mode = "wb" if isinstance(content, bytes) else "w"
        encoding = None if isinstance(content, bytes) else "utf-8"
        
        try:
            with open(filepath, mode, encoding=encoding) as f:
                f.write(content)
            self.logger.info(f"Raw content saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving raw content: {str(e)}")
            return None

    def _save_converted_content(self, content: str, url: str, cve_id: str) -> Optional[Path]:
        """Save the converted text content"""
        text_dir = self.output_dir / cve_id / "text"
        ensure_directory(text_dir)
        
        # For YouTube URLs, we want to save as .txt
        content_type = "txt" if is_youtube_url(url) else "txt"
        
        filename = self._generate_filename(url, content_type)
        filepath = text_dir / filename
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info(f"Converted content saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving converted content: {str(e)}")
            return None

         
    
    def _fetch_url(self, url: str) -> Optional[Dict]:
        """Fetch content from URL"""
        try:
            self.logger.debug(f"Starting URL fetch: {url}")
            
            # Handle YouTube URLs
            if is_youtube_url(url):
                self.logger.debug("Processing as YouTube URL")
                content = handle_youtube_url(url)
                if content:
                    self.logger.info(f"Successfully extracted transcript from YouTube video: {url}")
                    return {'content': content, 'type': 'txt'}  # Changed type to txt
                else:
                    self.logger.warning(f"No transcript available for YouTube video: {url}")
                    return None
            
            # Handle special cases for other sources
            if is_googlesource_url(url):
                self.logger.debug("Processing as googlesource URL")
                modified_url = handle_googlesource_url(url)
                if not modified_url:
                    self.logger.warning(f"Skipping unsupported googlesource URL: {url}")
                    return None
                    
                response = self.session.get(modified_url, timeout=CRAWLER_SETTINGS["timeout"])
                self.logger.debug(f"Got response from {modified_url}")
                response.raise_for_status()
                
                return {'content': response.text, 'type': 'html'}
                
            elif is_cisa_url(url):
                self.logger.debug("Processing as CISA URL")
                modified_url = handle_cisa_url(url)
                
                if modified_url != url:
                    self.logger.info(f"Redirecting to new CISA URL format: {modified_url}")
                
                response = self.session.get(modified_url, timeout=CRAWLER_SETTINGS["timeout"])
                self.logger.debug(f"Got response from {modified_url}")
                response.raise_for_status()
                
                return {'content': response.text, 'type': 'html'}
            
            # Normal URL handling
            self.logger.debug(f"Sending GET request to {url}")
            response = self.session.get(url, timeout=CRAWLER_SETTINGS["timeout"])
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            self.logger.debug(f"Got response with content-type: {content_type}")
            
            if 'application/pdf' in content_type:
                return {'content': response.content, 'type': 'pdf'}
            return {'content': response.text, 'type': 'html'}
                
        except requests.RequestException as e:
            self.logger.error(f"Request error fetching URL {url}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching URL {url}: {str(e)}", exc_info=True)
            return None
            

    def _save_converted_content(self, content: str, url: str, cve_id: str) -> Optional[Path]:
        """Save the converted text content"""
        text_dir = self.output_dir / cve_id / "text"
        ensure_directory(text_dir)
        
        filename = self._generate_filename(url, "txt")
        filepath = text_dir / filename
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info(f"Converted content saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving converted content: {str(e)}")
            return None

    def process_url(self, url: str, cve_id: str) -> bool:
        """Process a single URL: fetch, save raw content, and convert"""
        self.logger.info(f"Started processing URL: {url} for {cve_id}")
        
        if self.tracker.should_skip_url(url, cve_id):
            self.logger.info(f"Skipping previously failed URL: {url}")
            return False
        
        raw_filepath = None
        try:
            if self.should_ignore_url(url):
                self.domain_stats.add_url(url, ignored=True)
                self.logger.info(f"Skipping ignored URL: {url}")
                return True

            # Fetch content
            self.logger.debug(f"Fetching content from URL: {url}")
            result = self._fetch_url(url)
            if not result:
                self.tracker.add_failed_url(url, cve_id, "Failed to fetch content")
                self.domain_stats.add_url(url, success=False)
                return False

            # Check binary file size limit
            if isinstance(result['content'], bytes) and len(result['content']) > 1024 * 1024:  # 1MB
                self.logger.warning(f"Binary content exceeds 1MB limit: {url}")
                self.tracker.add_failed_url(url, cve_id, "Binary content too large")
                self.domain_stats.add_url(url, success=False)
                return False

            # Save raw content
            self.logger.debug(f"Saving raw content from {url}")
            raw_filepath = self._save_raw_content(
                content=result['content'],
                url=url,
                cve_id=cve_id,
                content_type=result['type']
            )
            
            if not raw_filepath:
                self.logger.error(f"Failed to save raw content from {url}")
                self.domain_stats.add_url(url, success=False)
                return False
            
            # Convert the content
            self.logger.debug(f"Converting content from {url}")
            converted_content, conversion_method = self._convert_content(raw_filepath)
            if not converted_content:
                self.logger.error(f"Failed to convert content from {url}")
                self._cleanup_failed_files(raw_filepath)
                self.domain_stats.add_url(url, success=False)
                return False
                
            # Save converted content
            self.logger.debug(f"Saving converted content from {url}")
            text_filepath = self._save_converted_content(converted_content, url, cve_id)
            success = text_filepath is not None
            
            if not success:
                self._cleanup_failed_files(raw_filepath)
                self.tracker.add_failed_url(url, cve_id, "Failed to save converted content")
                self.logger.error(f"Failed to save converted content from {url}")
            
            # Record final result
            self.domain_stats.add_url(url, success=success)
            return success
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            self.logger.error(f"Request error for URL {url}: {error_msg}")
            if raw_filepath:
                self._cleanup_failed_files(raw_filepath)
            self.domain_stats.add_url(url, success=False)
            self.tracker.add_failed_url(url, cve_id, error_msg)
            return False
        except TimeoutError:
            error_msg = "Request timed out"
            self.logger.error(f"Timeout processing URL {url}")
            if raw_filepath:
                self._cleanup_failed_files(raw_filepath)
            self.domain_stats.add_url(url, success=False)
            self.tracker.add_failed_url(url, cve_id, error_msg)
            return False
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"Error processing URL {url}: {error_msg}", exc_info=True)
            if raw_filepath:
                self._cleanup_failed_files(raw_filepath)
            self.domain_stats.add_url(url, success=False)
            self.tracker.add_failed_url(url, cve_id, error_msg)
            return False

    def _cleanup_failed_files(self, filepath: Path) -> None:
        """Delete files that failed to process successfully"""
        try:
            if filepath and filepath.exists():
                filepath.unlink()
                self.logger.info(f"Cleaned up failed file: {filepath}")
        except Exception as e:
            self.logger.error(f"Error cleaning up file {filepath}: {str(e)}")

    def process_cve_urls(self, cve_id: str) -> None:
        """Process URLs for a given CVE"""
        # Check if already processed
        if self.is_cve_processed(cve_id):
            self.logger.info(f"Skipping {cve_id} - raw and text already processed")
            return
                
        self.logger.info(f"Starting to process URLs for {cve_id}")
        
        # First pass: Process URLs from links.txt
        links_file = self.output_dir / cve_id / "links.txt"
        if not links_file.exists():
            self.logger.error(f"No links.txt found for {cve_id}")
            return
            
        try:
            with open(links_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            if not urls:
                self.logger.info(f"No URLs found in links.txt for {cve_id}")
                return
                
            self.logger.info(f"Found {len(urls)} URLs to process for {cve_id}")
            
            # Process each URL
            success_count = 0
            with tqdm(total=len(urls), desc=f"Processing {cve_id}", unit="url") as pbar:
                for url in urls:
                    if self.process_url(url, cve_id):
                        success_count += 1
                    pbar.update(1)
            
            self.logger.info(f"Completed processing {cve_id}: {success_count}/{len(urls)} URLs successful")
                
        except Exception as e:
            self.logger.error(f"Error processing URLs for {cve_id}: {str(e)}")

    def finish_processing(self):
        """Called after all processing is complete to generate final reports"""
        try:
            # Generate domain statistics report
            self.logger.info("Generating domain statistics report")
            self.domain_stats.generate_report()
            self.logger.info("Domain statistics report generated successfully")
            
            # Update status for all processed CVEs
            for cve_dir in self.output_dir.iterdir():
                if cve_dir.is_dir() and cve_dir.name.startswith("CVE-"):
                    self.tracker.update_cve_status(cve_dir.name)
                    
        except Exception as e:
            self.logger.error(f"Error in finish_processing: {e}", exc_info=True)
            

def find_cve_specific_urls(text_file: Path, target_cve: str, context_chars: int = 200, logger=None) -> set[str]:
    """
    Extract URLs that appear near mentions of the target CVE
    
    Args:
        text_file: Path to the text file to analyze
        target_cve: The specific CVE ID we're interested in
        context_chars: Number of characters to look before and after CVE mention
        logger: Logger instance to use for logging messages
        
    Returns:
        Set of URLs that appear within context_chars of the CVE
    """
    urls = set()
    
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find all mentions of the CVE
        cve_pattern = re.escape(target_cve)
        for match in re.finditer(cve_pattern, content):
            # Get text window around CVE mention
            start = max(0, match.start() - context_chars)
            end = min(len(content), match.end() + context_chars)
            window = content[start:end]
            
            # Find URLs in this window
            url_pattern = r'https://[^\s\(\)\[\]<>\"\']+' 
            for url_match in re.finditer(url_pattern, window):
                url = url_match.group(0).rstrip('.,;')
                urls.add(url)
                
        if urls and logger:
            logger.info(f"Found {len(urls)} URLs near {target_cve} in {text_file}")
        elif logger:
            logger.debug(f"No URLs found near {target_cve} in {text_file}")
            
        return urls
            
    except Exception as e:
        if logger:
            logger.error(f"Error extracting URLs for {text_file}: {str(e)}")
        return set()