"""
Async OpenVINO YOLO implementation using the Async API for throughput optimization
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import time
from config.config import CV_SETTINGS, VIDEO_PROCESSING

# Try to import OpenVINO
try:
    from openvino.runtime import Core
    import openvino.properties.hint as hints
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("OpenVINO not available")
    exit(1)


class AsyncOpenVINOYOLO:
    """
    Async OpenVINO YOLO implementation using the Async API for throughput optimization
    """
    
    def __init__(self, model_path=None, optimization_mode="throughput", num_requests=4):
        """
        Initialize the Async OpenVINO YOLO model
        
        Args:
            model_path: Path to OpenVINO model
            optimization_mode: "latency" or "throughput"
            num_requests: Number of concurrent inference requests for async processing
        """
        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO is not available")
        
        if model_path is None:
            model_path = Path("openvino_conversion/models/yolo11n")
        
        model_path = Path(model_path)
        
        # Find the XML file (model description)
        xml_file = model_path / f"{model_path.name}.xml"
        if not xml_file.exists():
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
            # For latency optimization
            config = {
                hints.performance_mode: hints.PerformanceMode.LATENCY,
                hints.num_requests: 1
            }
        elif optimization_mode == "throughput":
            # For throughput optimization with multiple requests
            config = {
                hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
                hints.num_requests: num_requests  # Use multiple inference requests
            }
        else:
            # Default to latency mode
            config = {
                hints.performance_mode: hints.PerformanceMode.LATENCY,
                hints.num_requests: 1
            }
        
        # Compile the model with the optimization configuration
        self.compiled_model = self.core.compile_model(self.model, device_name="CPU", config=config)
        
        # Get input and output layers
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # Get input shape
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]
        
        # Initialize async inference requests
        self.num_requests = config.get(hints.num_requests, 1) if optimization_mode == "throughput" else 1
        self.infer_requests = []
        
        for i in range(self.num_requests):
            self.infer_requests.append(self.compiled_model.create_infer_request())
        
        # Track busy requests
        self.busy_requests = [False] * self.num_requests
        
        print(f"Async OpenVINO Model loaded successfully from {xml_file}")
        print(f"Input shape: {self.input_shape}")
        print(f"Optimization mode: {optimization_mode}")
        print(f"Number of async requests: {self.num_requests}")
    
    def preprocess(self, image):
        """Preprocess the input image for the model"""
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
    
    def async_infer(self, image, callback=None, user_data=None, conf_threshold=0.5):
        """
        Start async inference on an image
        
        Args:
            image: Input image (numpy array)
            callback: Optional callback function to be called when inference completes
            user_data: Optional user data to pass to the callback
            conf_threshold: Confidence threshold for filtering detections
            
        Returns:
            Request ID if a request is available, None otherwise
        """
        # Find an available inference request
        request_id = -1
        for i in range(self.num_requests):
            if not self.busy_requests[i]:
                request_id = i
                break
        
        if request_id == -1:
            # No available requests
            return None
        
        # Mark the request as busy
        self.busy_requests[request_id] = True
        
        # Preprocess the image
        input_blob = self.preprocess(image)
        
        # Set input tensor
        self.infer_requests[request_id].set_tensor(self.input_layer, input_blob)
        
        # Set callback if provided
        if callback:
            self.infer_requests[request_id].set_callback(callback, user_data)
        
        # Start async inference
        self.infer_requests[request_id].start_async()
        
        return request_id
    
    def get_result(self, request_id, conf_threshold=0.5):
        """
        Get the result of an async inference request
        
        Args:
            request_id: ID of the inference request
            conf_threshold: Confidence threshold for filtering detections
            
        Returns:
            List of detections in format [x1, y1, x2, y2, confidence, class_id]
        """
        if request_id < 0 or request_id >= self.num_requests:
            raise ValueError(f"Invalid request_id: {request_id}")
        
        # Wait for the inference to complete
        self.infer_requests[request_id].wait()
        
        # Get output
        output = self.infer_requests[request_id].get_tensor(self.output_layer).data
        
        # Mark the request as available
        self.busy_requests[request_id] = False
        
        # Process the output to extract detections
        detections = []
        
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
                    original_shape = (self.height, self.width)  # Using input shape as reference
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
    
    def is_ready(self, request_id):
        """
        Check if an inference request is ready (completed)
        
        Args:
            request_id: ID of the inference request
            
        Returns:
            True if the request is ready, False otherwise
        """
        if request_id < 0 or request_id >= self.num_requests:
            return False
        
        # Check if the request is still busy
        return not self.busy_requests[request_id]


def main():
    """
    Main function to demonstrate async OpenVINO YOLO inference
    """
    print("Async OpenVINO YOLO Inference Demo")
    print("="*50)
    
    # Path to the converted model
    model_path = Path("openvino_conversion/models/yolo11n")
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please run the conversion script first: python openvino_conversion/scripts/convert_yolo_to_openvino.py")
        return
    
    # Initialize the Async OpenVINO YOLO model with throughput optimization
    try:
        async_yolo = AsyncOpenVINOYOLO(model_path, optimization_mode="throughput", num_requests=4)
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
    print("Async inference is running with throughput optimization...")
    
    # Track async requests
    active_requests = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Submit a new inference request if there's an available slot
        request_id = async_yolo.async_infer(frame, conf_threshold=0.5)
        
        if request_id is not None:
            # Add to active requests
            active_requests.append((frame_count, request_id, frame.copy()))
            print(f"Submitted frame {frame_count} for async inference (request_id: {request_id})")
        
        # Check for completed requests and display results
        completed_requests = []
        for frame_num, req_id, original_frame in active_requests:
            if not async_yolo.busy_requests[req_id]:  # Request is complete
                # Get the detections
                detections = async_yolo.get_result(req_id, conf_threshold=0.5)
                
                # Draw detections on the original frame
                for detection in detections:
                    x1, y1, x2, y2, confidence, class_id = detection
                    cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(original_frame, f'Person: {confidence:.2f}', (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show the frame with detections
                cv2.imshow('Async OpenVINO YOLO Inference', original_frame)
                
                # Mark this request as completed
                completed_requests.append((frame_num, req_id))
        
        # Remove completed requests from the active list
        for completed in completed_requests:
            active_requests = [req for req in active_requests if req[1] != completed[1]]
        
        frame_count += 1
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Wait for all remaining requests to complete
    print("Waiting for remaining async requests to complete...")
    for frame_num, req_id, original_frame in active_requests:
        detections = async_yolo.get_result(req_id, conf_threshold=0.5)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print("Async inference demo completed.")


if __name__ == "__main__":
    main()