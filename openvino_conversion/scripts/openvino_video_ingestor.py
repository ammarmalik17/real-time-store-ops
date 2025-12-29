"""
OpenVINO-enhanced Video Ingestion and Person Detection Module

This module provides the same functionality as video_ingestor.py but using 
OpenVINO-optimized models for better performance on Intel hardware.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
from config.config import CV_SETTINGS, VIDEO_PROCESSING

# Try to import OpenVINO, with fallback to regular YOLO if not available
try:
    from openvino.runtime import Core
    import openvino.properties.hint as hints
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("OpenVINO not available, falling back to regular YOLO inference")
    from ultralytics import YOLO


class OpenVINOYOLO:
    """
    Class to run YOLO inference using OpenVINO-converted models
    """
    
    def __init__(self, model_path=None, optimization_mode="latency"):
        """
        Initialize the OpenVINO YOLO model
        
        Args:
            model_path (str or Path): Path to the OpenVINO model directory
            optimization_mode (str): "latency" or "throughput" optimization
        """
        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO is not available")
        
        if model_path is None:
            model_path = Path("openvino_conversion/models/yolo11n")
        
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
        
        # Set optimization mode
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
        
        print(f"OpenVINO Model loaded successfully from {xml_file}")
        print(f"Input shape: {self.input_shape}")
        print(f"Optimization mode: {optimization_mode}")
    
    def preprocess(self, image):
        """
        Preprocess the input image for the model
        """
        # Resize image to model input size
        resized = cv2.resize(image, (self.width, self.height))
        
        # Change data layout from HWC to CHW
        blob = np.transpose(resized, (2, 0, 1))
        
        # Add batch dimension
        blob = np.expand_dims(blob, axis=0)
        
        # Convert to FP32
        blob = blob.astype(np.float32)
        
        # Normalize to [0, 1]
        blob = blob / 255.0
        
        return blob
    
    def infer(self, image, conf_threshold=0.5):
        """
        Run inference on an image using OpenVINO
        
        Args:
            image: Input image (numpy array)
            conf_threshold: Confidence threshold for filtering detections
            
        Returns:
            List of detections in format [x1, y1, x2, y2, confidence, class_id]
        """
        if not OPENVINO_AVAILABLE:
            return []
        
        original_shape = image.shape[:2]  # (height, width)
        
        # Preprocess the image
        input_blob = self.preprocess(image)
        
        # Run inference
        results = self.compiled_model([input_blob])
        
        # Get output
        output = results[self.output_layer]
        
        # Process the output to extract detections
        # Note: This output format may need adjustment based on the actual model output
        detections = []
        
        # The output format depends on the model, for YOLO-like models:
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        # Process each detection
        for detection in output:
            if len(detection) >= 6:
                x_center, y_center, width, height, confidence = detection[0:5]
                class_id = int(detection[5])
                
                # Only include person class (class_id = 0) and with sufficient confidence
                if class_id == 0 and confidence >= conf_threshold:
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


class RegularYOLO:
    """
    Regular YOLO implementation as fallback when OpenVINO is not available
    """
    
    def __init__(self, model_path=None):
        """
        Initialize regular YOLO model
        
        Args:
            model_path: Path to YOLO model file. If None, uses default from config
        """
        if not model_path:
            model_path = CV_SETTINGS["model_path"]
        self.model = YOLO(model_path)
        self.classes = CV_SETTINGS["classes_to_detect"]
        
    def infer(self, image, conf_threshold=0.5):
        """
        Run inference on an image using regular YOLO
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold
            
        Returns:
            List of detections in format [x1, y1, x2, y2, confidence, class_id]
        """
        # Run inference
        results = self.model(
            image,
            classes=self.classes,  # Only detect persons
            conf=conf_threshold,
            iou=VIDEO_PROCESSING["iou_threshold"],
            verbose=False
        )
        
        # Extract detections from results
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) in self.classes:  # Only include person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    detections.append([x1, y1, x2, y2, conf, class_id])
        
        return detections


class OpenVINOVideoIngestor:
    """
    Video ingestion class that can use either OpenVINO-optimized models or regular YOLO
    """
    
    def __init__(self, source=0, use_openvino=True, optimization_mode="latency"):
        """
        Initialize video ingestor with option to use OpenVINO
        
        Args:
            source: Video source (same as before)
            use_openvino: Whether to use OpenVINO-optimized inference
            optimization_mode: "latency" or "throughput" optimization
        """
        self.source = source
        self.use_openvino = use_openvino and OPENVINO_AVAILABLE
        self.optimization_mode = optimization_mode
        self.cap = None
        self.is_rtsp = isinstance(source, str) and source.startswith('rtsp://')
        
        # Initialize the appropriate model
        if self.use_openvino:
            print(f"Using OpenVINO-optimized model with {optimization_mode} optimization")
            self.detector = OpenVINOYOLO(optimization_mode=optimization_mode)
        else:
            print("Using regular YOLO model")
            self.detector = RegularYOLO()
    
    def open(self):
        """Open the video source"""
        try:
            # Use cross-platform video capture utility from utils
            from src.utils.video_utils import create_video_capture
            self.cap = create_video_capture(self.source)
            
            if self.is_rtsp:
                # For RTSP streams, we may need to adjust buffering
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, VIDEO_PROCESSING["rtsp_buffer_size"])
            
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video source: {self.source}")
            
            return True
        except Exception as e:
            raise ValueError(f"Cannot open video source: {self.source}. Error: {e}")
    
    def read_frame(self):
        """Read a single frame from the video source"""
        if self.cap is None:
            raise RuntimeError("Video source not opened. Call open() first.")
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def detect_persons(self, frame):
        """
        Detect persons in a frame using the selected model (OpenVINO or regular)
        
        Args:
            frame: Input image/frame
            
        Returns:
            List of detection results
        """
        conf_threshold = VIDEO_PROCESSING["confidence_threshold"]
        return self.detector.infer(frame, conf_threshold)
    
    def annotate_frame(self, frame, detections):
        """
        Annotate frame with detection results
        
        Args:
            frame: Input frame to annotate
            detections: List of detections from infer method
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw bounding boxes for each detection
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f'Person: {confidence:.2f}',
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Add person count to frame
        person_count = len(detections)
        cv2.putText(
            annotated_frame,
            f'Persons: {person_count}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return annotated_frame
    
    def release(self):
        """Release the video source"""
        if self.cap:
            self.cap.release()
            self.cap = None


def main():
    """
    Main function to demonstrate OpenVINO-enhanced video ingestion and person detection
    """
    # Default to webcam (index 0) or can be changed to video file or RTSP URL
    video_source = 0  # Change this to test with different sources
    
    # Option to use OpenVINO (will fallback to regular YOLO if OpenVINO is not available)
    use_openvino = True
    
    # Optimization mode: "latency" for fastest response time, "throughput" for maximum FPS
    optimization_mode = "latency"  # Change to "throughput" for higher throughput
    
    try:
        # Initialize video ingestor with OpenVINO option and optimization mode
        ingestor = OpenVINOVideoIngestor(video_source, use_openvino=use_openvino, optimization_mode=optimization_mode)
        
        # Open video source
        ingestor.open()
        
        print(f"Successfully opened video source: {video_source}")
        print(f"Using OpenVINO: {ingestor.use_openvino}")
        print(f"Optimization mode: {optimization_mode}")
        print("Press 'q' to quit")
        
        while True:
            # Read frame
            frame = ingestor.read_frame()
            if frame is None:
                print("End of video stream or error reading frame")
                break
            
            # Detect persons
            detections = ingestor.detect_persons(frame)
            
            # Annotate frame with results
            annotated_frame = ingestor.annotate_frame(frame, detections)
            
            # Display the frame
            cv2.imshow('OpenVINO Person Detection', annotated_frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    finally:
        # Clean up
        cv2.destroyAllWindows()
        if 'ingestor' in locals():
            ingestor.release()


if __name__ == "__main__":
    main()