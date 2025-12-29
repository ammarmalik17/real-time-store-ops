"""
Basic Video Ingestion and Person Detection Module

This module provides the foundational functionality for:
- Reading video from files or RTSP streams
- Performing person detection using YOLOv8
- Displaying results with bounding boxes
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys
from config.config import CV_SETTINGS, VIDEO_PROCESSING
from src.utils.video_utils import create_video_capture, get_video_info


class VideoIngestor:
    """
    Handles video ingestion from various sources (files, RTSP streams, cameras)
    """
    
    def __init__(self, source=0):
        """
        Initialize video ingestor
        
        Args:
            source: Video source - can be:
                   - Integer (0, 1, etc.) for camera index
                   - String path to video file
                   - RTSP URL string
        """
        self.source = source
        self.cap = None
        self.is_rtsp = isinstance(source, str) and source.startswith('rtsp://')
        
    def open(self):
        """Open the video source"""
        try:
            # Use cross-platform video capture utility
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
    
    def release(self):
        """Release the video source"""
        if self.cap:
            self.cap.release()
            self.cap = None


class PersonDetector:
    """
    Performs person detection using YOLO model
    """
    
    def __init__(self, model_path=None):
        """
        Initialize person detector
        
        Args:
            model_path: Path to YOLO model file. If None, uses default from config
        """
        model_path = model_path or CV_SETTINGS["model_path"]
        self.model = YOLO(model_path)
        self.classes = CV_SETTINGS["classes_to_detect"]
        
    def detect_persons(self, frame):
        """
        Detect persons in a frame
        
        Args:
            frame: Input image/frame
            
        Returns:
            List of detection results
        """
        # Run inference
        results = self.model(
            frame,
            classes=self.classes,  # Only detect persons
            conf=VIDEO_PROCESSING["confidence_threshold"],
            iou=VIDEO_PROCESSING["iou_threshold"],
            verbose=False
        )
        
        return results
    
    def annotate_frame(self, frame, results):
        """
        Annotate frame with detection results
        
        Args:
            frame: Input frame to annotate
            results: YOLO results object
            
        Returns:
            Annotated frame
        """
        # Use the built-in plot method to draw bounding boxes
        annotated_frame = results[0].plot()
        
        # Add person count to frame
        person_count = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(
            annotated_frame,
            f'Persons: {person_count}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Add FPS if available in results
        if 'speed' in results[0].speed:
            fps = 1000 / results[0].speed['inference'] if results[0].speed['inference'] > 0 else 0
            cv2.putText(
                annotated_frame,
                f'FPS: {fps:.1f}',
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        return annotated_frame


def main():
    """
    Main function to demonstrate basic video ingestion and person detection
    """
    # Default to webcam (index 0) or can be changed to video file or RTSP URL
    video_source = 0  # Change this to test with different sources
    
    # You can also test with:
    # video_source = "path/to/video.mp4"  # for video file
    # video_source = "rtsp://username:password@ip_address:port/path"  # for RTSP stream
    
    try:
        # Initialize video ingestor and person detector
        ingestor = VideoIngestor(video_source)
        detector = PersonDetector()
        
        # Open video source
        ingestor.open()
        
        print(f"Successfully opened video source: {video_source}")
        print("Press 'q' to quit")
        
        while True:
            # Read frame
            frame = ingestor.read_frame()
            if frame is None:
                print("End of video stream or error reading frame")
                break
            
            # Detect persons
            results = detector.detect_persons(frame)
            
            # Annotate frame with results
            annotated_frame = detector.annotate_frame(frame, results)
            
            # Display the frame
            cv2.imshow('Person Detection', annotated_frame)
            
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