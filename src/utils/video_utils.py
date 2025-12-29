"""
Video utility functions for handling different video sources across platforms
"""

import cv2
import platform
from pathlib import Path


def get_video_backend():
    """
    Get the appropriate video backend based on the platform
    
    Returns:
        int: OpenCV video backend constant
    """
    system = platform.system().lower()
    
    if system == "windows":
        # On Windows, prefer DirectShow
        return cv2.CAP_DSHOW
    elif system == "linux":
        # On Linux, use FFMPEG
        return cv2.CAP_FFMPEG
    else:
        # For other platforms, use default
        return cv2.CAP_ANY


def create_video_capture(source):
    """
    Create a VideoCapture object with the appropriate backend for the platform
    
    Args:
        source: Video source - can be index, file path, or RTSP URL
        
    Returns:
        cv2.VideoCapture: Video capture object
    """
    # Determine the appropriate backend for the platform
    backend = get_video_backend()
    
    # Create the video capture object
    if isinstance(source, str) and source.startswith('rtsp://'):
        # For RTSP streams, we might need to specify the backend
        cap = cv2.VideoCapture(source, backend)
    elif isinstance(source, (int, str)):
        # For cameras or files, use the backend
        cap = cv2.VideoCapture(source, backend)
    else:
        raise ValueError(f"Invalid video source: {source}")
    
    # Set platform-specific properties if needed
    system = platform.system().lower()
    if system == "linux":
        # On Linux, sometimes we need to set buffer size to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    return cap


def validate_video_source(source):
    """
    Validate if a video source is accessible
    
    Args:
        source: Video source to validate
        
    Returns:
        bool: True if source is valid and accessible, False otherwise
    """
    try:
        cap = create_video_capture(source)
        is_valid = cap.isOpened()
        cap.release()
        return is_valid
    except Exception:
        return False


def get_video_info(cap):
    """
    Get information about a video capture object
    
    Args:
        cap: cv2.VideoCapture object
        
    Returns:
        dict: Video information including width, height, fps, etc.
    """
    if not cap.isOpened():
        raise ValueError("Video capture object is not opened")
    
    return {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'backend': int(cap.get(cv2.CAP_PROP_BACKEND)),
    }


def set_video_properties(cap, width=None, height=None, fps=None):
    """
    Set video properties for a capture object
    
    Args:
        cap: cv2.VideoCapture object
        width: Desired width
        height: Desired height
        fps: Desired FPS
        
    Returns:
        bool: True if properties were set successfully
    """
    success = True
    
    if width is not None:
        success &= cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    
    if height is not None:
        success &= cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if fps is not None:
        success &= cap.set(cv2.CAP_PROP_FPS, fps)
    
    return success