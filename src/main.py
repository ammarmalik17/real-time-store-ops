"""
Main entry point for the Real-Time Store Operations System
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