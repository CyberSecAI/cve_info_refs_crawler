# src/cve_ref_crawler/tracking.py

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Set, Optional, List
from pathlib import Path
import csv
import json
import hashlib
from urllib.parse import urlparse
from .utils.logging_utils import setup_logging
from config import LOG_CONFIG
from threading import Lock

@dataclass
class URLStatus:
    url: str
    cve_id: str
    status: str  # 'success', 'failed', 'ignored', 'dead'
    reason: Optional[str] = None
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class ProcessingTracker:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.failed_urls_file = output_dir / "failed_urls.csv"
        self.cve_status_file = output_dir / "cve_status.json"
        self.failed_urls: Dict[str, URLStatus] = {}
        self._lock = Lock()
        self.logger = setup_logging(
            log_dir=LOG_CONFIG["dir"],
            log_level=LOG_CONFIG["level"],
            module_name=__name__
        )
        self._load_failed_urls()
        


    def should_skip_url(self, url: str, cve_id: str) -> bool:
        """Check if URL should be skipped based on previous failures"""
        key = f"{cve_id}:{url}"
        return key in self.failed_urls

    def add_failed_url(self, url: str, cve_id: str, reason: str) -> None:
        """Record a failed URL attempt"""
        with self._lock:
            key = f"{cve_id}:{url}"
            status = URLStatus(
                url=url,
                cve_id=cve_id,
                status='failed',
                reason=reason
            )
            self.failed_urls[key] = status
            self._append_failed_url(status)
        self.logger.info(f"Added failed URL for {cve_id}: {url} - {reason}")

    def _append_failed_url(self, status: URLStatus) -> None:
        """Append single failed URL to CSV file"""
        try:
            write_header = not self.failed_urls_file.exists()
            with open(self.failed_urls_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['CVE_ID', 'URL', 'Status', 'Failure_Reason', 'Timestamp'])
                writer.writerow([
                    status.cve_id,
                    status.url,
                    status.status,
                    status.reason,
                    status.timestamp
                ])
        except Exception as e:
            self.logger.error(f"Error appending failed URL: {e}")

    def _load_failed_urls(self) -> None:
        """Load previously failed URLs from CSV"""
        if self.failed_urls_file.exists():
            try:
                with open(self.failed_urls_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = f"{row['CVE_ID']}:{row['URL']}"
                        self.failed_urls[key] = URLStatus(
                            url=row['URL'],
                            cve_id=row['CVE_ID'],
                            status='failed',
                            reason=row['Failure_Reason'],
                            timestamp=row['Timestamp']
                        )
                self.logger.info(f"Loaded {len(self.failed_urls)} failed URLs from {self.failed_urls_file}")
            except Exception as e:
                self.logger.error(f"Error loading failed URLs: {e}")            
                
    def _save_failed_urls(self) -> None:
        """Save failed URLs to CSV file"""
        try:
            with self._lock:
                with open(self.failed_urls_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['CVE_ID', 'URL', 'Status', 'Failure_Reason', 'Timestamp'])
                    
                    failed_urls_snapshot = list(self.failed_urls.values())
                    for status in failed_urls_snapshot:
                        writer.writerow([
                            status.cve_id,
                            status.url,
                            status.status,
                            status.reason,
                            status.timestamp
                        ])
                self.logger.debug(f"Saved {len(self.failed_urls)} failed URLs")
        except Exception as e:
            self.logger.error(f"Error saving failed URLs: {e}")

    def update_cve_status(self, cve_id: str) -> None:
        """Update processing status for a CVE"""
        try:
            cve_dir = self.output_dir / cve_id
            status = {
                'cve_id': cve_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'phases': {
                    'primary': self._check_primary_phase(cve_dir),
                    'secondary': self._check_secondary_phase(cve_dir),
                    'extraction': self._check_extraction_phase(cve_dir)
                },
                'urls': self._get_url_statuses(cve_id)
            }
            
            self._save_cve_status(cve_id, status)
            self.logger.debug(f"Updated status for {cve_id}")
        except Exception as e:
            self.logger.error(f"Error updating CVE status for {cve_id}: {e}")

    def _check_primary_phase(self, cve_dir: Path) -> Dict:
        """Check status of primary phase"""
        links_file = cve_dir / "links.txt"
        return {
            'completed': links_file.exists(),
            'timestamp': datetime.fromtimestamp(links_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if links_file.exists() else None
        }

    def _check_secondary_phase(self, cve_dir: Path) -> Dict:
        """Check status of secondary phase"""
        secondary_file = cve_dir / "secondary_links_processed.txt"
        return {
            'completed': secondary_file.exists(),
            'timestamp': datetime.fromtimestamp(secondary_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if secondary_file.exists() else None
        }

    def _check_extraction_phase(self, cve_dir: Path) -> Dict:
        """Check status of extraction phase"""
        refined_file = cve_dir / "refined" / "refined.md"
        return {
            'completed': refined_file.exists(),
            'timestamp': datetime.fromtimestamp(refined_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if refined_file.exists() else None
        }

    def _get_url_statuses(self, cve_id: str) -> List[Dict]:
        """Get status of all URLs for a CVE"""
        url_statuses = []
        
        # Check links.txt for primary URLs
        links_file = self.output_dir / cve_id / "links.txt"
        if links_file.exists():
            try:
                with open(links_file, 'r') as f:
                    for url in f:
                        url = url.strip()
                        if not url:
                            continue
                        key = f"{cve_id}:{url}"
                        if key in self.failed_urls:
                            status = self.failed_urls[key]
                        else:
                            status = self._determine_url_status(url, cve_id)
                        url_statuses.append({
                            'url': url,
                            'status': status.status,
                            'reason': status.reason,
                            'timestamp': status.timestamp
                        })
            except Exception as e:
                self.logger.error(f"Error getting URL statuses for {cve_id}: {e}")
                    
        return url_statuses

    def _determine_url_status(self, url: str, cve_id: str) -> URLStatus:
        """Determine current status of a URL"""
        # Check if URL content exists in raw directory
        raw_dir = self.output_dir / cve_id / "raw"
        if raw_dir.exists():
            for file in raw_dir.iterdir():
                if self._url_matches_file(url, file.name):
                    return URLStatus(url=url, cve_id=cve_id, status='success')
        
        return URLStatus(url=url, cve_id=cve_id, status='unknown')

    def _url_matches_file(self, url: str, filename: str) -> bool:
        """Check if a URL corresponds to a file based on naming convention"""
        # Extract domain from URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Generate URL hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        # Check if both domain and hash are in filename
        return domain in filename and url_hash in filename

    def _save_cve_status(self, cve_id: str, status: Dict) -> None:
        """Save CVE status to JSON file"""
        try:
            status_dir = self.output_dir / "status"
            status_dir.mkdir(exist_ok=True)
            
            status_file = status_dir / f"{cve_id}_status.json"
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=2)
            self.logger.debug(f"Saved status for {cve_id} to {status_file}")
        except Exception as e:
            self.logger.error(f"Error saving CVE status for {cve_id}: {e}")