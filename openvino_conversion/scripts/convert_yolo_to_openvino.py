"""
Script to convert YOLO models to OpenVINO format for optimized inference
"""

import sys
import os
from pathlib import Path

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
import numpy as np


def convert_yolo_to_openvino(model_path="yolo11n.pt", output_dir="openvino_conversion/models"):
    """
    Convert a YOLO model to OpenVINO format
    
    Args:
        model_path (str): Path to the YOLO model to convert
        output_dir (str): Directory to save the converted model
    """
    print(f"Converting {model_path} to OpenVINO format...")
    
    try:
        # Load the YOLO model
        model = YOLO(model_path)
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export the model to OpenVINO format
        # This will create a directory with .xml and .bin files
        model.export(format='openvino', imgsz=[640, 640], int8=False)
        
        # The exported model will be saved in a subdirectory with the same name
        # For example, if the model is 'yolo11n.pt', it will be saved as 'yolo11n_openvino'
        model_name = Path(model_path).stem
        exported_dir = Path(f"{model_name}_openvino")
        
        # Move the exported model to our designated directory
        final_path = output_path / model_name
        if exported_dir.exists():
            if final_path.exists():
                import shutil
                shutil.rmtree(final_path)
            exported_dir.rename(final_path)
        
        print(f"Model successfully converted to OpenVINO format!")
        print(f"Converted model saved at: {final_path}")
        
        return final_path
        
    except Exception as e:
        print(f"Error converting model: {e}")
        raise


def convert_yolo_to_openvino_quantized(model_path="yolo11n.pt", output_dir="openvino_conversion/models"):
    """
    Convert a YOLO model to quantized OpenVINO format for better performance
    
    Args:
        model_path (str): Path to the YOLO model to convert
        output_dir (str): Directory to save the converted model
    """
    print(f"Converting {model_path} to quantized OpenVINO format (INT8)...")
    
    try:
        # Load the YOLO model
        model = YOLO(model_path)
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export the model to quantized OpenVINO format
        model.export(format='openvino', imgsz=[640, 640], int8=True)
        
        # The exported model will be saved in a subdirectory with the same name
        model_name = Path(model_path).stem
        exported_dir = Path(f"{model_name}_openvino")
        
        # Move the exported model to our designated directory with quantized suffix
        final_path = output_path / f"{model_name}_quantized"
        if exported_dir.exists():
            if final_path.exists():
                import shutil
                shutil.rmtree(final_path)
            exported_dir.rename(final_path)
        
        print(f"Model successfully converted to quantized OpenVINO format!")
        print(f"Quantized model saved at: {final_path}")
        
        return final_path
        
    except Exception as e:
        print(f"Error converting model to quantized format: {e}")
        raise


if __name__ == "__main__":
    print("OpenVINO Model Conversion Tool")
    print("="*40)
    
    # Convert to regular OpenVINO format
    convert_yolo_to_openvino()
    
    print()
    
    # Convert to quantized OpenVINO format
    convert_yolo_to_openvino_quantized()
    
    print()
    print("Conversion complete! You can now use the converted models for optimized inference.")