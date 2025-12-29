"""
Utility functions for the Real-Time Store Operations System
"""

from pathlib import Path
import platform
import os


def get_platform_info():
    """
    Get information about the current platform
    
    Returns:
        dict: Platform information including OS, architecture, etc.
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
    }


def ensure_path_exists(path):
    """
    Ensure that a path exists, creating directories if necessary
    
    Args:
        path (str or Path): Path to ensure exists
        
    Returns:
        Path: The path object that is guaranteed to exist
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_cross_platform_path(*path_parts):
    """
    Construct a path that works across platforms
    
    Args:
        *path_parts: Parts of the path to join
        
    Returns:
        Path: Cross-platform compatible path
    """
    return Path(*path_parts)


def normalize_path(path):
    """
    Normalize a path for the current platform
    
    Args:
        path (str or Path): Path to normalize
        
    Returns:
        Path: Normalized path
    """
    return Path(path).resolve()


# Additional utility functions can be added here