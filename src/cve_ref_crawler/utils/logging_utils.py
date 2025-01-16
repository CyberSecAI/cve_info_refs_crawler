# src/cve_ref_crawler/utils/logging_utils.py

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logging(log_dir: Path = Path("logs"), 
                 log_level: int = logging.INFO,
                 module_name: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for a module
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default: INFO)
        module_name: Name of the module (default: None)
    
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #log_file = log_dir / f"crawler_{timestamp}.log"
    
    # Generate timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{module_name}_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Get logger
    logger_name = module_name if module_name else __name__
    logger = logging.getLogger(logger_name)
    
    # Set level and add handlers if they haven't been added
    logger.setLevel(log_level)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger