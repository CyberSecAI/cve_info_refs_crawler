# src/config.py

import logging
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_IN_DIR = BASE_DIR / "data_in"
DATA_OUT_DIR = BASE_DIR / "data_out"
LOG_DIR = BASE_DIR / "logs"

# External paths
EXTERNAL_CWE_TOP25_DIR = BASE_DIR.parent / "cwe_top25"

# Test mode flag
TEST = False  # Set to False for production mode
CISA_YT = False
CISA = False
TOP_25 = False

# File paths - conditional on TEST flag
if TEST:
    TARGET_CVES_CSV = "./data_in/top25-mitre-mapping-analysis-2023-public_10.csv"
elif CISA:
    TARGET_CVES_CSV = "./data_in/vulnrichment_all_cve_descriptions.csv"
elif CISA_YT:
    TARGET_CVES_CSV = "./test/youtube.csv"
elif TOP_25:
    TARGET_CVES_CSV = EXTERNAL_CWE_TOP25_DIR / "data_in" / "top25-mitre-mapping-analysis-2023-public.csv"
else:
    TARGET_CVES_CSV = "data_in/cves.csv"
    
NVD_JSONL_FILE = DATA_IN_DIR / "nvd.jsonl"
DEAD_DOMAINS_CSV = DATA_IN_DIR / "dead_domains.csv"  

# Logging configuration
LOG_CONFIG = {
    #"level": logging.INFO,
    "level": logging.DEBUG, 
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "dir": LOG_DIR
}

CRAWLER_SETTINGS = {
    "timeout": 30,  # General request timeout
    "retry_count": 3,
    "delay_between_requests": 1,  # seconds
    "robots_txt": {
        "fetch_timeout": 10,      # Timeout for fetching robots.txt
        "rule_check_timeout": 5,  # Timeout for checking rules in robots.txt
        "cache_duration": 3600,   # How long to cache robots.txt (seconds)
    },
    "url_process_timeout": 30,    # Overall timeout for processing a single URL
    "headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1"
    }
}


# URLs to ignore (will not be crawled)
# vuldb requires an API key - so ignored here
# https://hackerone.com/ requires account and human verification
IGNORED_URLS = [
    "www.cve.org",
    "cve.org",
    "cve.mitre.org",
    "cwe.mitre.org",
    "first.org",
    "nist.gov",    
    "www.securitytracker.com",
    "securitytracker.com",
    "exchange.xforce.ibmcloud.com",
    "vuldb.com",
    "hackerone.com"
]

# LLM extraction settings
LLM_SETTINGS = {
    "rate_limits": {
        "min_request_interval": 1.0,    # Minimum time between requests in seconds
        "requests_window": 60,          # Time window for request counting in seconds
        "max_requests_per_window": 60,  # Maximum requests per window
    },
    "backoff": {
        "initial": 1.0,                # Initial backoff time in seconds
        "max": 60.0,                   # Maximum backoff time in seconds
    },
    "retries": {
        "max_attempts": 3,             # Maximum number of retry attempts
        "timeout": 60,                 # Timeout per request in seconds
    },
    "model": {
        "name": "gemini-2.0-flash-exp",
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
}