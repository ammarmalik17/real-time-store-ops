"""
Script to download the YOLO model for the Real-Time Store Operations System
"""

from ultralytics import YOLO
import os
from pathlib import Path


def download_model(model_name="yolo11n.pt"):
    """
    Download the specified YOLO model
    
    Args:
        model_name (str): Name of the model to download
    """
    print(f"Downloading {model_name}...")
    
    try:
        # This will trigger the download of the model
        model = YOLO(model_name)
        print(f"Successfully downloaded {model_name}")
        
        # Get the path where the model is stored
        model_path = Path(model.ckpt_path if hasattr(model, 'ckpt_path') else model.overrides.get('model', model_name))
        print(f"Model stored at: {model_path}")
        
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download the model
    download_model()