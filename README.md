# Real-Time Store Operations System

This project implements an edge-based computer vision system that converts existing retail camera feeds into actionable staffing and operations intelligence.

## Project Structure

```
real-time-store-ops/
├── src/                    # Source code
│   ├── cv_engine/          # Computer vision engine
│   ├── dashboard/          # Web dashboard
│   ├── ai_engine/          # AI staffing recommendations
│   └── utils/              # Utility functions
├── config/                 # Configuration files
├── data/                   # Data files and database
├── models/                 # ML models
├── requirements.txt        # Python dependencies
├── GOALS.md               # Project goals and specifications
└── README.md              # This file
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. The YOLO model will be automatically downloaded when first used by the application. No separate download step is required.

## Running the Basic Video Ingestion

To test the basic video ingestion and person detection:

```bash
python src/cv_engine/video_ingestor.py
```

This will open your default camera and perform real-time person detection.

## Cross-Platform Compatibility

This project is designed to work on both Windows and Linux platforms. The configuration system automatically adapts to the platform it's running on.