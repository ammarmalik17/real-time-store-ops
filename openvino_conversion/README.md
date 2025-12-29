# OpenVINO Model Conversion

This directory contains scripts and resources for converting YOLO models to OpenVINO format for optimized inference on Intel hardware.

## Directory Structure

```
openvino_conversion/
├── scripts/                    # Conversion and inference scripts
│   ├── convert_yolo_to_openvino.py      # Script to convert YOLO models to OpenVINO format
│   └── run_openvino_inference.py        # Script to run inference with OpenVINO models
├── models/                    # Converted OpenVINO models
├── requirements-openvino.txt  # OpenVINO-specific requirements
└── README.md                  # This file
```

## Setup

1. Install OpenVINO requirements:
   ```bash
   pip install -r openvino_conversion/requirements-openvino.txt
   ```

2. Make sure you have the base requirements installed:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Convert YOLO Model to OpenVINO

To convert a YOLO model to OpenVINO format:

```bash
python openvino_conversion/scripts/convert_yolo_to_openvino.py
```

This will:
- Convert the YOLO model to standard OpenVINO format
- Create a quantized INT8 version for better performance
- Save both models in the `openvino_conversion/models/` directory

### Run Inference with OpenVINO Model

To run inference using the converted OpenVINO model:

```bash
python openvino_conversion/scripts/run_openvino_inference.py
```

This will load the converted model and run real-time inference on your default camera.

## Benefits of Using OpenVINO

- **Performance**: Up to 3x speedup for YOLO models on Intel CPUs
- **Quantization**: INT8 and FP16 quantization for smaller models and faster inference
- **Optimization**: Leverages Intel CPU features like AVX2 and AVX-512
- **Compatibility**: Works with your existing Intel Core i7-8565U processor

For more information about OpenVINO, visit the [official documentation](https://docs.openvino.ai/).