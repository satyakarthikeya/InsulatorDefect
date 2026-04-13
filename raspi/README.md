# Raspberry Pi Deployment — YOLO11n-Ghost-Hybrid-P3P4-Medium

Lightweight insulator damage detection model (868K params, 3.7 MB ONNX)
optimized for edge deployment on Raspberry Pi.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run benchmark (ONNX mode — recommended)
python inference_benchmark.py

# 3. View results
ls results/   # annotated images with detections + timing
```

## Package Contents

```
raspi/
├── models/
│   ├── best_exp012.onnx      # 3.7 MB — ONNX (Pi-optimized)
│   └── best_exp012.pt        # 2.0 MB — PyTorch (optional)
├── test_images/               # 10 test images + YOLO labels
├── results/                   # (created on first run) annotated outputs
├── inference_benchmark.py     # Benchmark script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Model Details

| Property       | Value                            |
|----------------|----------------------------------|
| Architecture   | YOLO11n-Ghost-Hybrid-P3P4-Medium |
| Parameters     | 867,664 (868K)                   |
| GFLOPs         | 5.7                              |
| Input size     | 768×768                          |
| Classes        | 2 (Damaged_1, insulator)         |
| Val mAP50      | 96.46%                           |
| ONNX size      | 3.7 MB                           |
| ONNX opset     | 17                               |

## Usage Options

```bash
# Default: ONNX on all test images, 10 runs/image
python inference_benchmark.py

# PyTorch mode (requires ultralytics)
python inference_benchmark.py --mode pytorch

# Custom images folder
python inference_benchmark.py --images /path/to/my/images

# Adjust thresholds
python inference_benchmark.py --conf 0.3 --iou 0.65

# More benchmark runs for stable timing
python inference_benchmark.py --warmup 5 --runs 30
```

## Output

The benchmark prints:
1. **Per-image table**: inference time (ms), full pipeline time, FPS, detection counts
2. **Timing summary**: mean, median, min, max, throughput FPS
3. **Ground truth comparison**: predicted vs actual detections per image
4. **Model card**: architecture, size, accuracy summary

Annotated images with bounding boxes are saved to `results/`.

## Raspberry Pi Setup

```bash
# Install system dependencies (Pi OS)
sudo apt update && sudo apt install -y python3-pip python3-opencv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Run
python inference_benchmark.py
```

### Expected Performance (approximate)

| Device            | Inference (ms) | FPS  |
|-------------------|----------------|------|
| Raspberry Pi 4B   | ~800-1200      | ~1   |
| Raspberry Pi 5    | ~400-600       | ~2   |
| Jetson Orin Nano  | ~15-30         | ~40+ |
| x86 CPU (laptop)  | ~50-150        | ~10  |

## Classes

- **Damaged_1** (class 0): Damaged insulators — drawn in **red**
- **insulator** (class 1): Normal insulators — drawn in **green**
