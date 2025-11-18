import os
from pathlib import Path

def ensure_dir(path: Path):
    """Ensure directory exists before writing files."""
    os.makedirs(path, exist_ok=True)
    return path