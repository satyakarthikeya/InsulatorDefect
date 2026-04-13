"""
Baseline: YOLO11n Standard Model Training
==========================================
Train the standard YOLO11n model (~2.56M parameters) on the VOC
insulator defect dataset as a baseline comparison.

This establishes what the off-the-shelf YOLO11n (pretrained on COCO)
can achieve on our 2-class defect detection task, before any custom
architecture changes or knowledge distillation.

Architecture: Standard YOLO11n (~2.56M params) — pretrained on COCO
Target: Baseline mAP50 for comparison against Ghost-Hybrid (868K)
Hardware: RTX 3050 (4GB VRAM)

Usage:
    python scripts/train_baseline_yolo11n.py
    python scripts/train_baseline_yolo11n.py --epochs 300 --batch 8
"""

import os
import sys
import argparse
from pathlib import Path

# ── Prevent CUDA memory fragmentation on 4GB VRAM ──
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Project root setup ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline: Train standard YOLO11n on VOC insulator dataset"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n.pt",
        help="Pretrained YOLO11n weights (auto-downloads if not found)",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (8 for YOLO11n on 4GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Baseline: Standard YOLO11n (~2.56M params)")
    print("  Pretrained on COCO → Fine-tuned on VOC Insulator")
    print("=" * 60)

    data_yaml = "VOC/voc.yaml"

    # Verify dataset exists
    assert Path(data_yaml).exists(), f"Data YAML not found: {data_yaml}"

    # ── Load pretrained YOLO11n ──
    # This is the standard YOLO11n with ~2.56M parameters
    model = YOLO(args.weights, task="detect")

    # Print model info
    print(f"\n{'─' * 40}")
    print("Model Summary:")
    model.info(verbose=True)
    print(f"{'─' * 40}\n")

    # ── Train ──
    # Same hyperparameters as exp_002 for fair comparison
    results = model.train(
        data=data_yaml,

        # Training duration
        epochs=args.epochs,
        patience=50,

        # Batch & image size
        batch=args.batch,
        imgsz=args.imgsz,

        # Optimizer (same as exp_002 for fair comparison)
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,

        # Warmup
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Augmentation (same as exp_002)
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        translate=0.2,
        scale=0.9,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,
        copy_paste=0.0,

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Performance - OPTIMIZED for 4GB VRAM + 16GB RAM
        cache="ram",
        device=args.device,
        workers=4,
        amp=True,

        # Logging
        project="experiments",
        name="baseline_yolo11n",
        exist_ok=True,
        verbose=True,
        plots=True,
    )

    # ── Results ──
    print("\n" + "=" * 60)
    print("  Baseline Training Complete!")
    print("=" * 60)
    print(f"  Model: Standard YOLO11n (~2.56M parameters)")
    print(f"  Best weights: experiments/baseline_yolo11n/weights/best.pt")
    print(f"  Results CSV:  experiments/baseline_yolo11n/results.csv")
    print(f"\n  Compare against Ghost-Hybrid (868K params):")
    print(f"    - exp_002: 94.21% mAP50 (from scratch @640)")
    print(f"    - exp_012: 96.46% mAP50 (best, head-only @768)")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
