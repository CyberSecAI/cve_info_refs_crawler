# src/cve_ref_crawler/utils/cve_filter.py

import pandas as pd
from pathlib import Path
from typing import Set
from .logging_utils import setup_logging
from config import LOG_CONFIG

def load_target_cves(csv_path: str | Path) -> Set[str]:
    """
    Load target CVE IDs from CSV file
    
    Args:
        csv_path: Path to CSV file containing CVE IDs
        
    Returns:
        Set of unique CVE IDs
    """
    logger = setup_logging(
        log_dir=LOG_CONFIG["dir"],
        log_level=LOG_CONFIG["level"],
        module_name=__name__
    )
    
    try:
        logger.info(f"Loading target CVEs from {csv_path}")
        df = pd.read_csv(csv_path)
        
        if 'CVE' not in df.columns:
            logger.error(f"No 'CVE' column found in {csv_path}")
            return set()
            
        # Extract unique CVE IDs
        cve_ids = set(df['CVE'].unique())
        logger.info(f"Found {len(cve_ids)} unique CVE IDs")
        
        return cve_ids
        
    except Exception as e:
        logger.error(f"Error loading CVE IDs from {csv_path}: {e}")
        return set()