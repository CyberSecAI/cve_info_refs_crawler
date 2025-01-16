#!/usr/bin/env python3

#Usage: python src/generate_report.py

import csv
from pathlib import Path
import json
from typing import Dict, List
import logging
from datetime import datetime

class CVEStatusReporter:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.data_dir / f"cve_status_report_{self.timestamp}.csv"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def count_urls_in_file(self, file_path: Path) -> int:
        """Count non-empty lines in a file"""
        try:
            if not file_path.exists():
                return 0
            with open(file_path, 'r') as f:
                return sum(1 for line in f if line.strip())
        except Exception as e:
            self.logger.error(f"Error counting URLs in {file_path}: {e}")
            return 0

    def get_cve_stats(self, cve_dir: Path) -> Dict:
        """Get statistics for a single CVE directory"""
        stats = {
            'cve_id': cve_dir.name,
            'primary_urls': 0,
            'secondary_urls': 0,
            'dead_links': 0,
            'ignored_links': 0,
            'failed_links': 0,
            'has_combined': False,
            'has_refined': False
        }

        # Count primary URLs from links.txt
        links_file = cve_dir / "links.txt"
        stats['primary_urls'] = self.count_urls_in_file(links_file)

        # Check for secondary URLs
        secondary_file = cve_dir / "secondary_links_processed.txt"
        stats['secondary_urls'] = self.count_urls_in_file(secondary_file)

        # Check for combined and refined content
        stats['has_combined'] = (cve_dir / "refined" / "combined.md").exists()
        stats['has_refined'] = (cve_dir / "refined" / "refined.md").exists()

        # Check status directory for detailed URL status
        status_file = self.data_dir / "status" / f"{cve_dir.name}_status.json"
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                    for url_info in status_data.get('urls', []):
                        status = url_info.get('status', '').lower()
                        if status == 'dead':
                            stats['dead_links'] += 1
                        elif status == 'ignored':
                            stats['ignored_links'] += 1
                        elif status == 'failed':
                            stats['failed_links'] += 1
            except Exception as e:
                self.logger.error(f"Error reading status file for {cve_dir.name}: {e}")

        return stats

    def generate_report(self):
        """Generate CSV report for all CVE directories"""
        self.logger.info("Starting CVE status report generation")
        
        cve_dirs = [d for d in self.data_dir.iterdir() 
                   if d.is_dir() and d.name.startswith("CVE-")]
        
        stats = []
        for cve_dir in cve_dirs:
            try:
                cve_stats = self.get_cve_stats(cve_dir)
                stats.append(cve_stats)
            except Exception as e:
                self.logger.error(f"Error processing {cve_dir.name}: {e}")

        # Write to CSV
        fieldnames = [
            'cve_id', 'primary_urls', 'secondary_urls', 'dead_links',
            'ignored_links', 'failed_links', 'has_combined', 'has_refined'
        ]

        with open(self.output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats)

        self.logger.info(f"Report generated: {self.output_file}")
        self.print_summary(stats)

    def print_summary(self, stats: List[Dict]):
        """Print overall summary of the report"""
        total_cves = len(stats)
        total_primary = sum(s['primary_urls'] for s in stats)
        total_secondary = sum(s['secondary_urls'] for s in stats)
        total_dead = sum(s['dead_links'] for s in stats)
        total_ignored = sum(s['ignored_links'] for s in stats)
        total_failed = sum(s['failed_links'] for s in stats)
        total_combined = sum(1 for s in stats if s['has_combined'])
        total_refined = sum(1 for s in stats if s['has_refined'])

        summary = f"""
Report Summary:
--------------
Total CVEs processed: {total_cves}
Total primary URLs: {total_primary}
Total secondary URLs: {total_secondary}
Total dead links: {total_dead}
Total ignored links: {total_ignored}
Total failed links: {total_failed}
CVEs with combined content: {total_combined}
CVEs with refined content: {total_refined}
        """
        print(summary)

if __name__ == "__main__":
    # Assuming script is run from project root
    data_dir = Path("data_out")
    reporter = CVEStatusReporter(data_dir)
    reporter.generate_report()