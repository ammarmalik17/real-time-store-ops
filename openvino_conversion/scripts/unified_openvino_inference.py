"""
Unified OpenVINO Inference Script for YOLO models
Combines basic inference, video ingestion, async processing, and benchmarking
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import time
import argparse
from config.config import CV_SETTINGS, VIDEO_PROCESSING

# Try to import OpenVINO
try:
    from openvino.runtime import Core
    import openvino.properties.hint as hints
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("OpenVINO not available")


def get_optimal_num_requests(compiled_model):
    """Get the optimal number of inference requests for the compiled model"""
    try:
        return compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    except:
        return 4


class OpenVINOYOLO:
    """Standard OpenVINO YOLO implementation with optimization modes"""
    
    def __init__(self, model_path=None, optimization_mode="latency"):
        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO is not available")
        
        if model_path is None:
            model_path = Path("openvino_conversion/models/yolo11n")
        
        model_path = Path(model_path)
        
        # Find the XML file
        xml_file = model_path / f"{model_path.name}.xml"
        if not xml_file.exists():
            xml_files = list(model_path.glob("*.xml"))
            if xml_files:
                xml_file = xml_files[0]
            else:
                raise FileNotFoundError(f"No XML file found in {model_path}")
        
        # Load and compile the model with optimization
        self.core = Core()
        self.model = self.core.read_model(xml_file)
        
        self.optimization_mode = optimization_mode
        if optimization_mode == "latency":
            config = {hints.performance_mode: hints.PerformanceMode.LATENCY}
        elif optimization_mode == "throughput":
            config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
        else:
            config = {hints.performance_mode: hints.PerformanceMode.LATENCY}
        
        self.compiled_model = self.core.compile_model(self.model, device_name="CPU", config=config)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]
        
        print(f"Model loaded: {xml_file}, Mode: {optimization_mode}")
    
    def preprocess(self, image):
        resized = cv2.resize(image, (self.width, self.height))
        blob = np.transpose(resized, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0
        return blob
    
    def infer(self, image, conf_threshold=0.5):
        input_blob = self.preprocess(image)
        results = self.compiled_model([input_blob])
        output = results[self.output_layer]
        
        detections = []
        if len(output.shape) == 3:
            output = output[0]
        
        for detection in output:
            if len(detection) >= 6:
                x_center, y_center, width, height, confidence = detection[0:5]
                class_id = int(detection[5])
                
                if class_id == 0 and confidence >= conf_threshold:  # Person class
                    x1 = int((x_center - width / 2) * image.shape[1])
                    y1 = int((y_center - height / 2) * image.shape[0])
                    x2 = int((x_center + width / 2) * image.shape[1])
                    y2 = int((y_center + height / 2) * image.shape[0])
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                    
                    detections.append([x1, y1, x2, y2, confidence, class_id])
        
        return detections


class AsyncOpenVINOYOLO:
    """Async OpenVINO YOLO implementation using the Async API"""
    
    def __init__(self, model_path=None, optimization_mode="throughput", num_requests=None):
        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO is not available")
        
        if model_path is None:
            model_path = Path("openvino_conversion/models/yolo11n")
        
        model_path = Path(model_path)
        
        # Find the XML file
        xml_file = model_path / f"{model_path.name}.xml"
        if not xml_file.exists():
            xml_files = list(model_path.glob("*.xml"))
            if xml_files:
                xml_file = xml_files[0]
            else:
                raise FileNotFoundError(f"No XML file found in {model_path}")
        
        # Load and compile the model
        self.core = Core()
        self.model = self.core.read_model(xml_file)
        
        # Set configuration based on optimization mode
        self.optimization_mode = optimization_mode
        if optimization_mode == "latency":
            config = {
                hints.performance_mode: hints.PerformanceMode.LATENCY,
                hints.num_requests: 1
            }
        elif optimization_mode == "throughput":
            # Determine optimal number of requests if not specified
            temp_compiled = self.core.compile_model(self.model, device_name="CPU")
            optimal_requests = get_optimal_num_requests(temp_compiled)
            temp_compiled = None  # Free memory
            
            actual_num_requests = num_requests or optimal_requests
            config = {
                hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
                hints.num_requests: actual_num_requests
            }
        else:
            config = {
                hints.performance_mode: hints.PerformanceMode.LATENCY,
                hints.num_requests: 1
            }
        
        self.compiled_model = self.core.compile_model(self.model, device_name="CPU", config=config)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]
        
        # Initialize async requests
        self.num_requests = config.get(hints.num_requests, 1)
        self.infer_requests = []
        for i in range(self.num_requests):
            self.infer_requests.append(self.compiled_model.create_infer_request())
        
        # Track busy requests
        self.busy_requests = [False] * self.num_requests
        
        print(f"Async model loaded: {xml_file}, Mode: {optimization_mode}, Requests: {self.num_requests}")
    
    def preprocess(self, image):
        resized = cv2.resize(image, (self.width, self.height))
        blob = np.transpose(resized, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0).astype(np.float32) / 255.0
        return blob
    
    def async_infer(self, image, conf_threshold=0.5):
        # Find an available inference request
        request_id = -1
        for i in range(self.num_requests):
            if not self.busy_requests[i]:
                request_id = i
                break
        
        if request_id == -1:
            return None  # No available requests
        
        self.busy_requests[request_id] = True
        input_blob = self.preprocess(image)
        self.infer_requests[request_id].set_tensor(self.input_layer, input_blob)
        self.infer_requests[request_id].start_async()
        
        return request_id
    
    def get_result(self, request_id, conf_threshold=0.5):
        if request_id < 0 or request_id >= self.num_requests:
            raise ValueError(f"Invalid request_id: {request_id}")
        
        self.infer_requests[request_id].wait()
        output = self.infer_requests[request_id].get_tensor(self.output_layer).data
        
        self.busy_requests[request_id] = False
        
        # Process detections
        detections = []
        if len(output.shape) == 3:
            output = output[0]
        
        for detection in output:
            if len(detection) >= 6:
                x_center, y_center, width, height, confidence = detection[0:5]
                class_id = int(detection[5])
                
                if class_id == 0 and confidence >= conf_threshold:
                    x1 = int((x_center - width / 2) * self.width)
                    y1 = int((y_center - height / 2) * self.height)
                    x2 = int((x_center + width / 2) * self.width)
                    y2 = int((y_center + height / 2) * self.height)
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(self.width, x2), min(self.height, y2)
                    
                    detections.append([x1, y1, x2, y2, confidence, class_id])
        
        return detections


def benchmark_model(model, test_images, num_iterations=20):
    """Benchmark a model with a set of test images"""
    print(f"Benchmarking {num_iterations} inferences...")
    
    # Warm up
    for i in range(min(3, len(test_images))):
        _ = model.infer(test_images[i % len(test_images)])
    
    # Measure inference time
    start_time = time.time()
    total_detections = 0
    
    for i in range(num_iterations):
        img_idx = i % len(test_images)
        detections = model.infer(test_images[img_idx])
        total_detections += len(detections)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = num_iterations / total_time
    
    print(f"Total time: {total_time:.4f}s for {num_iterations} inferences")
    print(f"Average time: {avg_time:.4f}s ({avg_time*1000:.2f}ms)")
    print(f"Average FPS: {fps:.2f}")
    print(f"Total detections: {total_detections}")
    
    return avg_time, fps, total_detections


def benchmark_async_model(model, test_images, num_iterations=20):
    """Benchmark an async model with a set of test images"""
    print(f"Async benchmarking {num_iterations} inferences...")
    
    # Warm up
    for i in range(min(3, len(test_images))):
        req_id = model.async_infer(test_images[i % len(test_images)])
        if req_id is not None:
            model.get_result(req_id)
    
    # Measure inference time
    start_time = time.time()
    submitted_count = 0
    completed_count = 0
    total_detections = 0
    
    # Submit initial batch of requests
    active_requests = []
    for i in range(min(model.num_requests, num_iterations)):
        req_id = model.async_infer(test_images[i % len(test_images)])
        if req_id is not None:
            active_requests.append(req_id)
            submitted_count += 1
    
    # Process requests until all are completed
    while completed_count < num_iterations:
        # Check for completed requests and get results
        completed_requests = []
        for req_id in active_requests:
            if not model.busy_requests[req_id]:  # Request is complete
                detections = model.get_result(req_id)
                total_detections += len(detections)
                completed_requests.append(req_id)
                completed_count += 1
        
        # Remove completed requests from active list
        for req_id in completed_requests:
            active_requests.remove(req_id)
        
        # Submit new requests if needed
        if submitted_count < num_iterations and len(active_requests) < model.num_requests:
            img_idx = submitted_count % len(test_images)
            req_id = model.async_infer(test_images[img_idx])
            if req_id is not None:
                active_requests.append(req_id)
                submitted_count += 1
        
        # Small sleep to prevent busy waiting
        time.sleep(0.001)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = num_iterations / total_time
    
    print(f"Total time: {total_time:.4f}s for {num_iterations} inferences")
    print(f"Average FPS: {fps:.2f}")
    print(f"Total detections: {total_detections}")
    
    return total_time, fps, total_detections


def run_basic_inference(model_path, optimization_mode):
    """Run basic inference on a camera feed"""
    model = OpenVINOYOLO(model_path, optimization_mode=optimization_mode)
    
    # Open video source
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print(f"Running basic inference in {optimization_mode} mode")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        detections = model.infer(frame)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person: {confidence:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add FPS counter
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        cv2.putText(frame, f'FPS: {fps:.2f} ({optimization_mode})', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(f'OpenVINO YOLO - {optimization_mode}', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def run_async_inference(model_path, optimization_mode):
    """Run async inference on a camera feed"""
    model = AsyncOpenVINOYOLO(model_path, optimization_mode=optimization_mode)
    
    # Open video source
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print(f"Running async inference in {optimization_mode} mode")
    print("Press 'q' to quit")
    
    # Track async requests
    active_requests = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Submit a new inference request if there's an available slot
        request_id = model.async_infer(frame, conf_threshold=0.5)
        
        if request_id is not None:
            # Add to active requests
            active_requests.append((frame_count, request_id, frame.copy()))
        
        # Check for completed requests and display results
        completed_requests = []
        for frame_num, req_id, original_frame in active_requests:
            if not model.busy_requests[req_id]:  # Request is complete
                # Get the detections
                detections = model.get_result(req_id, conf_threshold=0.5)
                
                # Draw detections on the original frame
                for detection in detections:
                    x1, y1, x2, y2, confidence, class_id = detection
                    cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(original_frame, f'Person: {confidence:.2f}', (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show the frame with detections
                cv2.imshow('Async OpenVINO YOLO', original_frame)
                
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
        detections = model.get_result(req_id, conf_threshold=0.5)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def run_benchmark(model_path, optimization_mode, iterations):
    """Run benchmark for a specific optimization mode"""
    print(f"\nBenchmarking {optimization_mode} optimization mode...")
    
    # Create test images for benchmarking
    test_images = []
    for i in range(5):  # Create 5 different test images
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_images.append(img)
    
    if optimization_mode == "async":
        model = AsyncOpenVINOYOLO(model_path, optimization_mode="throughput")
        async_time, async_fps, async_detections = benchmark_async_model(
            model, test_images, iterations)
        print(f"Async Throughput - FPS: {async_fps:.2f}, Total time: {async_time:.4f}s")
    else:
        model = OpenVINOYOLO(model_path, optimization_mode=optimization_mode)
        avg_time, fps, total_detections = benchmark_model(
            model, test_images, iterations)
        print(f"{optimization_mode.capitalize()} Mode - FPS: {fps:.2f}, Time: {avg_time:.4f}s per inference")


def main():
    parser = argparse.ArgumentParser(description="Unified OpenVINO Inference Script for YOLO models")
    parser.add_argument("--mode", choices=["basic", "async", "benchmark"], 
                        default="basic", help="Inference mode to run")
    parser.add_argument("--model-path", default="openvino_conversion/models/yolo11n",
                        help="Path to the OpenVINO model")
    parser.add_argument("--optimization-mode", choices=["latency", "throughput", "async"], 
                        default="latency", help="Optimization mode")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of iterations for benchmarking")
    parser.add_argument("--benchmark-all", action="store_true",
                        help="Run benchmark for all optimization modes")
    
    args = parser.parse_args()
    
    print("Unified OpenVINO Inference Script for YOLO Models")
    print("="*60)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please run the conversion script first: python openvino_conversion/scripts/convert_yolo_to_openvino.py")
        return
    
    if args.benchmark_all:
        # Run benchmark for all modes
        modes = ["latency", "throughput", "async"]
        for mode in modes:
            run_benchmark(model_path, mode, args.iterations)
        
        print("\n" + "="*70)
        print("OPTIMIZATION MODE COMPARISON")
        print("="*70)
        print("Latency Mode: Best for single requests requiring fastest response time")
        print("Throughput Mode: Best for processing many requests efficiently")
        print("Async Mode: Best for maximum throughput with concurrent processing")
    elif args.mode == "benchmark":
        run_benchmark(model_path, args.optimization_mode, args.iterations)
    elif args.mode == "basic":
        run_basic_inference(model_path, args.optimization_mode)
    elif args.mode == "async":
        run_async_inference(model_path, args.optimization_mode)


if __name__ == "__main__":
    main()