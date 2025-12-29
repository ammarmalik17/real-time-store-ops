"""
Utility script to verify YOLO model availability for the Real-Time Store Operations System

Note: The ultralytics library automatically downloads models when first used,
so a separate download step is not required. This script simply verifies
that the model can be loaded.
"""

from ultralytics import YOLO
import os
from pathlib import Path


def verify_model(model_name="yolo11n.pt"):
    """
    Verify that the specified YOLO model can be loaded
    (this will trigger automatic download if needed)
    
    Args:
        model_name (str): Name of the model to verify
    """
    print(f"Verifying {model_name} (will auto-download if needed)...")
    
    try:
        # This will trigger the download of the model if not already present
        model = YOLO(model_name)
        print(f"Successfully loaded {model_name}")
        
        # Get the path where the model is stored
        model_path = Path(model.ckpt_path if hasattr(model, 'ckpt_path') else model.overrides.get('model', model_name))
        print(f"Model stored at: {model_path}")
        
        return model_path
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Verify the model (auto-download if needed)
    verify_model()