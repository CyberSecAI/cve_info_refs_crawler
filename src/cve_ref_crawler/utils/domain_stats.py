# src/cve_ref_crawler/utils/domain_stats.py

from collections import Counter
from urllib.parse import urlparse
from typing import Dict, Set, List, NamedTuple
from pathlib import Path
import json
import csv
from dataclasses import dataclass
from cve_ref_crawler.utils.logging_utils import setup_logging
from config import LOG_CONFIG, CRAWLER_SETTINGS, IGNORED_URLS, DEAD_DOMAINS_CSV

@dataclass
class DomainStatus:
    total_urls: int = 0
    successful: int = 0
    ignored: int = 0
    dead: int = 0
    failed: int = 0

class DomainStatsCollector:
    def __init__(self, output_dir: Path, ignored_urls: List[str], dead_domains: Set[str]):
        self.output_dir = output_dir
        self.ignored_urls = set(ignored_urls)
        self.dead_domains = dead_domains
        self.domain_stats: Dict[str, DomainStatus] = {}
        self.logger = setup_logging(LOG_CONFIG["dir"], LOG_CONFIG["level"], __name__)

    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain by removing www. prefix"""
        domain = domain.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain

    def add_url(self, url: str, success: bool = False, ignored: bool = False) -> None:
        """Record URL and its processing status"""
        try:
            domain = self._normalize_domain(urlparse(url).netloc)
            if not domain:
                return

            if domain not in self.domain_stats:
                self.domain_stats[domain] = DomainStatus()

            stats = self.domain_stats[domain]
            stats.total_urls += 1

            if ignored or any(self._normalize_domain(i) in domain for i in self.ignored_urls):
                stats.ignored += 1
            elif domain in self.dead_domains:
                stats.dead += 1
            elif success:
                stats.successful += 1
            else:
                stats.failed += 1

        except Exception as e:
            self.logger.error(f"Error processing URL {url}: {e}")

    def generate_report(self) -> None:
        """Generate domain statistics reports"""
        try:
            # Sort domains by total URLs
            sorted_stats = sorted(
                self.domain_stats.items(),
                key=lambda x: (x[1].total_urls, x[0]),
                reverse=True
            )

            # Save detailed CSV report
            csv_file = self.output_dir / 'domain_stats.csv'
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Domain',
                    'Total URLs',
                    'Successful',
                    'Ignored',
                    'Dead',
                    'Failed',
                    'Success Rate %'
                ])
                
                for domain, stats in sorted_stats:
                    success_rate = (stats.successful / stats.total_urls * 100) if stats.total_urls > 0 else 0
                    writer.writerow([
                        domain,
                        stats.total_urls,
                        stats.successful,
                        stats.ignored,
                        stats.dead,
                        stats.failed,
                        f"{success_rate:.1f}"
                    ])

            # Save JSON version for programmatic use
            json_file = self.output_dir / 'domain_stats.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {k: vars(v) for k, v in sorted_stats},
                    f,
                    indent=2
                )

            # Log summary statistics
            total_domains = len(self.domain_stats)
            total_urls = sum(s.total_urls for s in self.domain_stats.values())
            successful_urls = sum(s.successful for s in self.domain_stats.values())
            ignored_urls = sum(s.ignored for s in self.domain_stats.values())
            dead_urls = sum(s.dead for s in self.domain_stats.values())
            failed_urls = sum(s.failed for s in self.domain_stats.values())

            self.logger.info(f"Domain Statistics Summary:")
            self.logger.info(f"Total Domains: {total_domains}")
            self.logger.info(f"Total URLs: {total_urls}")
            self.logger.info(f"Successful: {successful_urls}")
            self.logger.info(f"Ignored: {ignored_urls}")
            self.logger.info(f"Dead: {dead_urls}")
            self.logger.info(f"Failed: {failed_urls}")

        except Exception as e:
            self.logger.error(f"Error generating domain statistics report: {e}")