"""
Script to run inference using OpenVINO-converted YOLO models
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import OpenVINO runtime
try:
    from openvino.runtime import Core
    import openvino.properties.hint as hints
except ImportError:
    print("OpenVINO runtime not found. Please install it with: pip install openvino")
    exit(1)


class OpenVINOYOLO:
    """
    Class to run YOLO inference using OpenVINO-converted models
    """
    
    def __init__(self, model_path, optimization_mode="latency"):
        """
        Initialize the OpenVINO YOLO model
        
        Args:
            model_path (str or Path): Path to the OpenVINO model directory containing .xml and .bin files
            optimization_mode (str): "latency" or "throughput" optimization mode
        """
        model_path = Path(model_path)
        
        # Find the XML file (model description)
        xml_file = model_path / f"{model_path.name}.xml"
        if not xml_file.exists():
            # Try to find any XML file in the directory
            xml_files = list(model_path.glob("*.xml"))
            if xml_files:
                xml_file = xml_files[0]
            else:
                raise FileNotFoundError(f"No XML file found in {model_path}")
        
        # Load the OpenVINO model
        self.core = Core()
        self.model = self.core.read_model(xml_file)
        
        # Set optimization mode based on Ultralytics guide
        self.optimization_mode = optimization_mode
        if optimization_mode == "latency":
            # For low latency: single inference, optimize for fastest response time
            config = {hints.performance_mode: hints.PerformanceMode.LATENCY}
        elif optimization_mode == "throughput":
            # For high throughput: optimize for maximum number of inferences per second
            config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
        else:
            # Default to latency mode
            config = {hints.performance_mode: hints.PerformanceMode.LATENCY}
        
        # Compile the model with the optimization configuration
        self.compiled_model = self.core.compile_model(self.model, device_name="CPU", config=config)
        
        # Get input and output layers
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # Get input shape
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]
        
        print(f"Model loaded successfully from {xml_file}")
        print(f"Input shape: {self.input_shape}")
        print(f"Optimization mode: {optimization_mode}")
    
    def preprocess(self, image):
        """
        Preprocess the input image for the model
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Preprocessed image ready for inference
        """
        # Resize image to model input size
        resized = cv2.resize(image, (self.width, self.height))
        
        # Change data layout from HWC to CHW
        blob = np.transpose(resized, (2, 0, 1))
        
        # Add batch dimension
        blob = np.expand_dims(blob, axis=0)
        
        # Convert to FP32
        blob = blob.astype(np.float32)
        
        # Normalize to [0, 1] if needed (depends on model training)
        blob = blob / 255.0
        
        return blob
    
    def postprocess(self, output, original_shape, conf_threshold=0.5):
        """
        Postprocess the model output to get detections
        
        Args:
            output: Model output
            original_shape: Original image shape (height, width)
            conf_threshold: Confidence threshold for filtering detections
            
        Returns:
            List of detections in format [x1, y1, x2, y2, confidence, class_id]
        """
        # The output format may vary depending on the model
        # For YOLO models, the output typically has shape [batch, num_detections, 85] for YOLOv5/v8
        # or similar for other YOLO versions
        
        # Reshape output if needed
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        detections = []
        
        for detection in output:
            # Extract bounding box coordinates, confidence, and class info
            # Format may vary depending on the YOLO version
            if len(detection) >= 6:
                x_center, y_center, width, height, confidence = detection[0:5]
                class_id = int(detection[5])
                
                if confidence >= conf_threshold:
                    # Convert from center coordinates to corner coordinates
                    x1 = int((x_center - width / 2) * original_shape[1])
                    y1 = int((y_center - height / 2) * original_shape[0])
                    x2 = int((x_center + width / 2) * original_shape[1])
                    y2 = int((y_center + height / 2) * original_shape[0])
                    
                    # Clamp coordinates to image boundaries
                    x1 = max(0, min(original_shape[1], x1))
                    y1 = max(0, min(original_shape[0], y1))
                    x2 = max(0, min(original_shape[1], x2))
                    y2 = max(0, min(original_shape[0], y2))
                    
                    detections.append([x1, y1, x2, y2, confidence, class_id])
        
        return detections
    
    def infer(self, image, conf_threshold=0.5):
        """
        Run inference on an image
        
        Args:
            image: Input image (numpy array)
            conf_threshold: Confidence threshold for filtering detections
            
        Returns:
            List of detections
        """
        original_shape = image.shape[:2]  # (height, width)
        
        # Preprocess the image
        input_blob = self.preprocess(image)
        
        # Run inference
        results = self.compiled_model([input_blob])
        
        # Get output
        output = results[self.output_layer]
        
        # Postprocess the output
        detections = self.postprocess(output, original_shape, conf_threshold)
        
        return detections


def main():
    """
    Main function to demonstrate OpenVINO YOLO inference
    """
    print("OpenVINO YOLO Inference Demo")
    print("="*40)
    
    # Path to the converted model (default to the standard model)
    model_path = Path("openvino_conversion/models/yolo11n")
    
    # Optimization mode: "latency" for fastest response time, "throughput" for maximum FPS
    optimization_mode = "latency"  # Change to "throughput" for higher throughput
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please run the conversion script first: python openvino_conversion/scripts/convert_yolo_to_openvino.py")
        return
    
    # Initialize the OpenVINO YOLO model with optimization mode
    try:
        ov_yolo = OpenVINOYOLO(model_path, optimization_mode=optimization_mode)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Open video source (default to webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source. Trying with a video file...")
        # Try with a sample video file if available
        sample_video = Path("sample_video.mp4")
        if sample_video.exists():
            cap = cv2.VideoCapture(str(sample_video))
        else:
            print("No video source available")
            return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        detections = ov_yolo.infer(frame, conf_threshold=0.5)
        
        # Draw detections on the frame
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person: {confidence:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show the frame with detections
        cv2.imshow('OpenVINO YOLO Inference', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()