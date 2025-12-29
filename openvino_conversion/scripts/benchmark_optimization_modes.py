"""
Script to demonstrate OpenVINO optimization modes for YOLO models
based on the Ultralytics guide for latency vs throughput optimization
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


class OpenVINOYOLOOptimized:
    """
    OpenVINO YOLO implementation with optimization modes based on Ultralytics guide
    """
    
    def __init__(self, model_path=None, optimization_mode="latency"):
        """
        Initialize OpenVINO YOLO with specified optimization mode
        
        Args:
            model_path: Path to OpenVINO model
            optimization_mode: "latency" or "throughput"
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
        
        # Configure based on optimization mode as per Ultralytics guide
        self.optimization_mode = optimization_mode
        if optimization_mode == "latency":
            # For latency optimization: single inference, fastest response time
            config = {
                hints.performance_mode: hints.PerformanceMode.LATENCY,
                # For latency, we want to limit the number of streams to 1
                hints.num_requests: 1
            }
            print("Configured for LATENCY optimization: single inference, fastest response time")
        elif optimization_mode == "throughput":
            # For throughput optimization: maximum number of inferences per second
            config = {
                hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
                # For throughput, OpenVINO will optimize internally
            }
            print("Configured for THROUGHPUT optimization: maximum inferences per second")
        else:
            # Default to latency mode
            config = {
                hints.performance_mode: hints.PerformanceMode.LATENCY
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
        
        print(f"OpenVINO Model loaded successfully from {xml_file}")
        print(f"Input shape: {self.input_shape}")
        print(f"Optimization mode: {optimization_mode}")
    
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
    
    def infer(self, image, conf_threshold=0.5):
        """Run inference on an image"""
        original_shape = image.shape[:2]  # (height, width)
        
        # Preprocess the image
        input_blob = self.preprocess(image)
        
        # Run inference
        results = self.compiled_model([input_blob])
        
        # Get output
        output = results[self.output_layer]
        
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


def benchmark_model(model, test_image, num_iterations=10):
    """
    Benchmark the model performance
    
    Args:
        model: OpenVINO YOLO model instance
        test_image: Test image for inference
        num_iterations: Number of inference iterations for benchmarking
    """
    print(f"\nBenchmarking {num_iterations} inferences...")
    
    # Warm up
    for _ in range(3):
        _ = model.infer(test_image)
    
    # Measure inference time
    start_time = time.time()
    for i in range(num_iterations):
        detections = model.infer(test_image)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = 1.0 / avg_time
    
    print(f"Total time for {num_iterations} inferences: {total_time:.4f}s")
    print(f"Average inference time: {avg_time:.4f}s ({avg_time*1000:.2f}ms)")
    print(f"Average FPS: {fps:.2f}")
    
    return avg_time, fps


def main():
    """
    Main function to demonstrate both optimization modes
    """
    print("OpenVINO Optimization Modes Demo")
    print("="*50)
    
    # Path to the converted model
    model_path = Path("openvino_conversion/models/yolo11n")
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please run the conversion script first: python openvino_conversion/scripts/convert_yolo_to_openvino.py")
        return
    
    # Create a test image (we'll use a blank image for the benchmark)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test latency optimization
    print("\n1. Testing LATENCY optimization mode...")
    latency_model = OpenVINOYOLOOptimized(model_path, optimization_mode="latency")
    latency_time, latency_fps = benchmark_model(latency_model, test_image)
    
    print("\n" + "-"*50)
    
    # Test throughput optimization
    print("\n2. Testing THROUGHPUT optimization mode...")
    throughput_model = OpenVINOYOLOOptimized(model_path, optimization_mode="throughput")
    throughput_time, throughput_fps = benchmark_model(throughput_model, test_image)
    
    print("\n" + "="*50)
    print("OPTIMIZATION MODE COMPARISON")
    print("="*50)
    print(f"Latency Mode   - Avg. time: {latency_time:.4f}s ({latency_time*1000:.2f}ms), FPS: {latency_fps:.2f}")
    print(f"Throughput Mode - Avg. time: {throughput_time:.4f}s ({throughput_time*1000:.2f}ms), FPS: {throughput_fps:.2f}")
    
    if latency_fps > throughput_fps:
        print(f"\nFor this model/input, LATENCY mode is faster by {((latency_fps-throughput_fps)/throughput_fps)*100:.1f}%")
        print("Use LATENCY mode for real-time applications requiring fastest response times")
    else:
        print(f"\nFor this model/input, THROUGHPUT mode is faster by {((throughput_fps-latency_fps)/latency_fps)*100:.1f}%")
        print("Use THROUGHPUT mode for applications processing many requests simultaneously")
    
    print("\nSUMMARY OF OPTIMIZATION MODES:")
    print("Latency Mode:")
    print("  - Optimized for fastest single-inference response time")
    print("  - Best for real-time applications with immediate response needs")
    print("  - Limits to single inference per device")
    
    print("\nThroughput Mode:")
    print("  - Optimized for maximum inferences per second")
    print("  - Best for processing multiple requests simultaneously")
    print("  - Better for batch processing scenarios")


if __name__ == "__main__":
    main()