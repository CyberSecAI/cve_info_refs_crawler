from pathlib import Path

def ensure_directory(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary
    
    Args:
        path: Path to the directory
    """
    path.mkdir(parents=True, exist_ok=True)