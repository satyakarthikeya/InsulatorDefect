# InsulatorDefect: Ultra-Lightweight Damage Detection for Power Transmission Insulators

A production-ready training, evaluation, and edge-deployment framework for detecting insulator damage on power transmission lines using an ultra-lightweight YOLO model.

**Model:** YOLO11n-Ghost-Hybrid-P3P4-Medium (867K parameters)

**Objective:** Detect rare insulator defects (Damaged_1) with 96%+ mAP50 on constrained edge hardware (Raspberry Pi / NVIDIA Jetson).

## Key Results

Best verified single-model metrics (no ensemble):

| Metric | Value |
|---|---:|
| Parameters | 867,664 (~868K) |
| GFLOPs | 5.7 |
| Input resolution | 768x768 |
| mAP50 | 96.10% |
| mAP50-95 | 66.79% |
| Precision | 93.65% |
| Recall | 94.18% |

Class AP50:
- Damaged_1: 95.05%
- insulator: 97.15%

## Repository Layout

- `configs/` - experiment configs
- `docs/` - architecture notes, experiment journey, benchmark reports
- `experiments/` - training runs, checkpoints, CSV logs, soups/SWA outputs
- `models/` - custom architecture YAML definitions
- `raspi/` - Raspberry Pi benchmarking package and inference script
- `scripts/` - training and evaluation scripts
- `VOC/` - dataset config and VOC-format data

## Environment Setup

Recommended:
- Python 3.10+
- Linux with NVIDIA GPU for training (project was developed on RTX 3050 4GB)

Install core dependencies:

```bash
pip install --upgrade pip
pip install ultralytics onnxruntime opencv-python numpy pandas matplotlib
```

For Raspberry Pi benchmarking package:

```bash
pip install -r raspi/requirements.txt
```

## Dataset Setup

The project expects VOC data referenced by `VOC/voc.yaml`.

Current classes:
- `Damaged_1`
- `insulator`

Expected structure (from `VOC/voc.yaml`):

```text
VOC/
  images/
    train/
    val/
  labels/
    train/
    val/
  voc.yaml
```

Important:
- `VOC/voc.yaml` currently contains an absolute `path` entry.
- Update that path on your machine if needed.

## Quick Start

Validate a strong checkpoint at 768:

```bash
yolo detect val \
  model=experiments/exp_012_head_finetune_768/weights/best.pt \
  data=VOC/voc.yaml \
  imgsz=768
```

Run prediction on validation images:

```bash
yolo detect predict \
  model=experiments/exp_012_head_finetune_768/weights/best.pt \
  source=VOC/images/val \
  imgsz=768 \
  conf=0.25 \
  save=True
```

## Training Pipeline

### 1) Baseline custom student (Ghost-Hybrid)

```bash
python scripts/train_exp002.py
```

Output:
- `experiments/exp_002_ghost_hybrid_medium/`

### 2) Train teacher model

```bash
python scripts/train_teacher.py --epochs 100 --batch 4 --device 0
```

Output:
- `experiments/exp_004_teacher_yolo11m/`

### 3) Knowledge Distillation (teacher -> student)

```bash
python scripts/train_kd.py \
  --teacher experiments/exp_004_teacher_yolo11m/weights/best.pt \
  --kd-alpha 0.5 \
  --kd-temperature 4.0 \
  --batch 2
```

Output:
- `experiments/exp_005_kd_student3/`

### 4) Fine-tune KD model at 768

```bash
python scripts/train_finetune_768.py
```

Output:
- `experiments/exp_009_finetune_768/`

### 5) Optional continuation scripts

You can continue exploration using available scripts such as:
- `scripts/train_finetune_768_v2.py`
- `scripts/train_finetune_896.py`
- `scripts/train_swa.py`
- `scripts/run_soup.py`
- `scripts/run_tta_v2.py`

## Export for Edge Deployment

Export ONNX for Raspberry Pi:

```bash
yolo export \
  model=experiments/exp_012_head_finetune_768/weights/best.pt \
  format=onnx \
  imgsz=768 \
  opset=17
```

Export TensorRT engine for Jetson:

```bash
yolo export \
  model=experiments/exp_012_head_finetune_768/weights/best.pt \
  format=engine \
  imgsz=768 \
  half=True
```

Export TFLite INT8:

```bash
yolo export \
  model=experiments/exp_012_head_finetune_768/weights/best.pt \
  format=tflite \
  imgsz=768 \
  int8=True
```

## Raspberry Pi Benchmark

Use the packaged benchmark workflow:

```bash
cd raspi
pip install -r requirements.txt
python inference_benchmark.py
```

Also see:
- `raspi/README.md`

## Experiment Notes and Reports

For detailed experiment history and analysis:
- `BENCHMARK.md`
- `docs/ARCHITECTURE.md`
- `docs/EXPERIMENTS.md`
- `docs/EXP_009_012_DETAILS.md`
- `docs/EXPERIMENT_JOURNEY.md`
- `docs/EXPERIMENT_JOURNEY_V2.md`

## Team

- P P Satya Karthikeya
- B Karthikeya
- M Karthik Reddy
- P Rohit

## License

No license file is currently present in the repository.
If you plan to distribute this project, add a LICENSE file explicitly.
