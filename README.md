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

## OpenVINO Integration

This project includes support for Intel OpenVINO toolkit for optimized inference on Intel hardware. The OpenVINO integration provides:

- Up to 3x performance improvement for YOLO models on Intel CPUs
- INT8 and FP16 quantization for faster inference with minimal accuracy loss
- Optimized CPU utilization leveraging Intel CPU features

To use OpenVINO optimization:

1. Install OpenVINO requirements: `pip install -r openvino_conversion/requirements-openvino.txt`
2. Convert the YOLO model: `python openvino_conversion/scripts/convert_yolo_to_openvino.py`
3. Run inference with the converted model: `python openvino_conversion/scripts/unified_openvino_inference.py`

The unified script supports multiple modes including basic inference, async processing, and benchmarking with different optimization modes.