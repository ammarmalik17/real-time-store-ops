# Configuration for Real-Time Store Operations System
import os
from pathlib import Path

# Cross-platform path handling
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"

# Video processing settings
VIDEO_PROCESSING = {
    "default_camera_index": 0,
    "rtsp_buffer_size": 4,  # Number of frames to buffer for RTSP streams
    "frame_skip": 1,  # Process every Nth frame to improve performance
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "max_fps": 30,
}

# Computer vision settings
CV_SETTINGS = {
    "model_path": str(MODELS_DIR / "yolo11n.pt"),  # Default YOLO model
    "classes_to_detect": [0],  # 0 is 'person' class in COCO dataset
    "detection_timeout": 10,  # seconds
}

# Dashboard settings
DASHBOARD = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,
}

# Database settings
DATABASE = {
    "path": str(DATA_DIR / "store_ops.db"),
    "backup_interval": 3600,  # seconds
}

# Platform-specific settings
import platform
PLATFORM = platform.system().lower()

if PLATFORM == "windows":
    PLATFORM_SETTINGS = {
        "video_backend": "CAP_DSHOW",  # DirectShow for Windows
        "process_priority": "above_normal",
    }
elif PLATFORM == "linux":
    PLATFORM_SETTINGS = {
        "video_backend": "CAP_FFMPEG",  # FFmpeg for Linux
        "process_priority": "normal",
    }
else:
    PLATFORM_SETTINGS = {
        "video_backend": "CAP_ANY",
        "process_priority": "normal",
    }

# Logging settings
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(DATA_DIR / "app.log"),
}