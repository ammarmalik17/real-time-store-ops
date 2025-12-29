"""
Main entry point for the Real-Time Store Operations System

Note: The first time the YOLO model is used, it will be automatically downloaded.
This may take a few minutes depending on your internet connection.
"""

def main():
    """
    Main function to start the real-time store operations system
    """
    print("Starting Real-Time Store Operations System...")
    
    # For now, just run the basic video ingestion demo
    from src.cv_engine.video_ingestor import main as video_main
    video_main()


if __name__ == "__main__":
    main()