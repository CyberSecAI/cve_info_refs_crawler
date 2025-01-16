# src/cve_ref_crawler/handlers/cisa.py

import re
from urllib.parse import urlparse
from typing import Optional

def is_cisa_url(url: str) -> bool:
    """Check if URL is from CISA website"""
    parsed = urlparse(url)
    return parsed.netloc == 'www.cisa.gov'

def handle_cisa_url(url: str) -> Optional[str]:
    """
    Handle CISA URLs, converting old formats to new ones
    
    Args:
        url: Original URL
        
    Returns:
        Modified URL if needed, or original URL if no change needed
    """
    # Handle old ICS advisory URLs
    ics_pattern = r'www\.cisa\.gov/uscert/ics/advisories/(icsa-[\d\-]+)'
    match = re.search(ics_pattern, url)
    if match:
        advisory_id = match.group(1)
        return f"https://www.cisa.gov/news-events/ics-advisories/{advisory_id}"

    # Handle old US-CERT advisory URLs
    cert_pattern = r'www\.cisa\.gov/uscert/ncas/(alerts|analysis-reports)/([\w\-]+)'
    match = re.search(cert_pattern, url)
    if match:
        category, advisory_id = match.groups()
        return f"https://www.cisa.gov/news-events/{category}/{advisory_id}"
    
    return url

def parse_cisa_response(html_content: str) -> str:
    """
    Return raw HTML content without parsing
    
    Args:
        html_content: Raw HTML content from CISA webpage
        
    Returns:
        Raw HTML content as-is
    """
    return html_content