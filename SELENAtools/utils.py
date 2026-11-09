"""
Utility functions for path handling, configuration loading, and file discovery.
"""

import json
import os
from pathlib import Path
from typing import List


def load_config(config_path: str) -> dict:
    """Load a JSON configuration file.

    Parameters
    ----------
    config_path : str
        Path to the JSON config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return json.load(f)


def find_dm3_files(folder: str, pattern: str = "60kX") -> List[str]:
    """Discover .dm3 TEM image files in a folder matching a name pattern.

    Parameters
    ----------
    folder : str
        Directory to search.
    pattern : str
        Substring that must appear in the filename (default: '60kX').

    Returns
    -------
    list of str
        Full paths to matching .dm3 files.
    """
    dm3_files = []
    for filename in os.listdir(folder):
        if pattern in filename and filename.lower().endswith(".dm3"):
            dm3_files.append(os.path.join(folder, filename))
    return sorted(dm3_files)


def ensure_analysis_dir(folder: str) -> str:
    """Ensure an 'analysis/' subdirectory exists inside the given folder.

    Parameters
    ----------
    folder : str
        Parent directory.

    Returns
    -------
    str
        Path to the 'analysis/' subdirectory.
    """
    analysis_dir = os.path.join(folder, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    return analysis_dir
