from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import math
import re
from cve_ref_crawler import CVEProcessor, ContentCrawler
from cve_ref_crawler.secondary_processor import SecondaryProcessor
from cve_ref_crawler.utils.cve_filter import load_target_cves
from cve_ref_crawler.utils.logging_utils import setup_logging
from cve_ref_crawler.llm_extractor import VulnerabilityExtractor
from config import NVD_JSONL_FILE, DATA_OUT_DIR, LOG_CONFIG, TARGET_CVES_CSV, DEAD_DOMAINS_CSV

def split_into_batches(items: List[str], batch_size: int) -> List[List[str]]:
    """Split a list of items into batches of specified size."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def process_primary_batch(batch: List[str], crawler: ContentCrawler, logger) -> Tuple[int, int]:
    """Process a batch of CVEs for primary URLs."""
    skipped, processed = 0, 0
    for cve_id in batch:
        if crawler.is_cve_processed(cve_id):
            skipped += 1
            continue
        logger.info(f"Processing primary URLs for {cve_id}")
        crawler.process_cve_urls(cve_id)
        processed += 1
    return skipped, processed

def process_secondary_batch(batch: List[str], processor: SecondaryProcessor, logger) -> Tuple[int, int]:
    """Process a batch of CVEs for secondary URLs."""
    skipped, processed = 0, 0
    for cve_id in batch:
        if processor.is_secondary_processing_completed(cve_id):
            skipped += 1
            continue
        logger.info(f"Processing secondary URLs for {cve_id}")
        processor.process_cve_directory(cve_id)
        processed += 1
    return skipped, processed

def process_extraction_batch(batch: List[str], extractor: VulnerabilityExtractor, logger) -> Tuple[int, int]:
    """Process a batch of CVEs for vulnerability extraction."""
    skipped, processed = 0, 0
    for cve_id in batch:
        if extractor.is_cve_extracted(cve_id):
            skipped += 1
            continue
        if extractor.process_cve(cve_id):
            processed += 1
    return skipped, processed

def parallel_process_phase(cve_batches: List[List[str]], 
                         process_func, 
                         processor_instance,
                         max_workers: int,
                         phase_name: str,
                         logger) -> Tuple[int, int]:
    """
    Process batches of CVEs in parallel for a given phase.
    Returns tuple of (total_skipped, total_processed)
    """
    total_skipped, total_processed = 0, 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(process_func, batch, processor_instance, logger): batch 
            for batch in cve_batches
        }
        
        with tqdm(total=len(cve_batches), desc=f"Processing {phase_name}", unit="batch") as pbar:
            for future in as_completed(future_to_batch):
                try:
                    skipped, processed = future.result()
                    total_skipped += skipped
                    total_processed += processed
                    pbar.set_postfix({
                        "skipped": total_skipped,
                        "processed": total_processed
                    })
                    pbar.update(1)
                except Exception as e:
                    batch = future_to_batch[future]
                    logger.error(f"Error processing batch in {phase_name}: {str(e)}")
                    logger.error(f"Failed batch: {batch}")
    
    return total_skipped, total_processed

def process_extraction_sequential(cves: List[str], 
                               extractor: VulnerabilityExtractor, 
                               logger) -> Tuple[int, int]:
    """
    Process CVEs sequentially for vulnerability extraction.
    Returns tuple of (total_skipped, total_processed)
    """
    skipped, processed = 0, 0
    
    with tqdm(total=len(cves), desc="Processing vulnerability extraction", unit="CVE") as pbar:
        for cve_id in cves:
            try:
                if extractor.is_cve_extracted(cve_id):
                    skipped += 1
                    logger.info(f"Skipping {cve_id} - already processed")
                else:
                    logger.info(f"Processing vulnerability extraction for {cve_id}")
                    if extractor.process_cve(cve_id):
                        processed += 1
                    
                pbar.set_postfix({
                    "skipped": skipped,
                    "processed": processed
                })
                pbar.update(1)
                
            except Exception as e:
                logger.error(f"Error processing {cve_id}: {str(e)}")
                continue
                
    return skipped, processed


def main():
    logger = setup_logging(
        log_dir=LOG_CONFIG["dir"],
        log_level=LOG_CONFIG["level"],
        module_name="main"
    )
    logger.info("Starting parallel CVE reference processing")
    
    # Load target CVEs from CSV
    target_cves = load_target_cves(TARGET_CVES_CSV)
    
    if not target_cves:
        logger.error("No target CVEs loaded. Exiting.")
        return
    
    # Initialize processor
    processor = CVEProcessor(
        input_file=NVD_JSONL_FILE,
        output_dir=DATA_OUT_DIR,
        target_cves=target_cves
    )
    processor.process_file()
    
    # Get list of CVE directories to process
    cve_dirs = [d for d in DATA_OUT_DIR.iterdir() if d.is_dir()]
    valid_cves = [d.name for d in cve_dirs if d.name in target_cves]
    
    # Configuration for parallel processing
    batch_size = 20  # Adjust based on your needs
    max_workers = min(8, math.ceil(len(valid_cves) / batch_size))  # Adjust based on your system
    cve_batches = split_into_batches(valid_cves, batch_size)
    
    # Initialize processors
    crawler = ContentCrawler(output_dir=DATA_OUT_DIR)
    secondary_processor = SecondaryProcessor(DATA_OUT_DIR)
    vulnerability_extractor = VulnerabilityExtractor(output_dir=DATA_OUT_DIR)
    
    # Phase 1: Process primary URLs in parallel
    logger.info("Phase 1: Processing primary URLs")
    primary_skipped, primary_processed = parallel_process_phase(
        cve_batches,
        process_primary_batch,
        crawler,
        max_workers,
        "primary URLs",
        logger
    )
    
    # Phase 2: Process secondary URLs in parallel
    logger.info("Phase 2: Processing secondary URLs")
    secondary_skipped, secondary_processed = parallel_process_phase(
        cve_batches,
        process_secondary_batch,
        secondary_processor,
        max_workers,
        "secondary URLs",
        logger
    )
    
       
    # Phase 3: Sequential vulnerability extraction due to LLM rate limits
    logger.info("Phase 3: Extracting vulnerability information")
    extraction_skipped, extraction_processed = process_extraction_sequential(
        valid_cves,  # Process full list sequentially, not batches
        vulnerability_extractor,
        logger
    )
    
    crawler.finish_processing()
    
    logger.info(
        f"Completed parallel CVE reference processing:\n"
        f"Phase 1 (Primary URLs):\n"
        f"  - CVEs skipped (already processed): {primary_skipped}\n"
        f"  - CVEs processed: {primary_processed}\n"
        f"  - Total CVEs handled: {primary_skipped + primary_processed}\n"
        f"Phase 2 (Secondary URLs):\n"
        f"  - CVEs skipped (already processed): {secondary_skipped}\n"
        f"  - CVEs processed: {secondary_processed}\n"
        f"  - Total CVEs handled: {secondary_skipped + secondary_processed}\n"
        f"Phase 3 (Vulnerability Extraction):\n"
        f"  - CVEs skipped: {extraction_skipped}\n"
        f"  - CVEs processed: {extraction_processed}\n"
        f"  - Total CVEs handled: {extraction_skipped + extraction_processed}"
    )

if __name__ == "__main__":
    main()