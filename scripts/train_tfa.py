"""
TFA Fine-Tuning: Two-Stage Fine-Tuning Approach
=================================================
Stage 2: Freeze backbone (layers 0-9), fine-tune head (layers 10-23)
Uses best weights from exp_002 as starting point.

This addresses class imbalance (1:3.5 Damaged_1 vs insulator) by:
  - Freezing backbone → preserves learned feature extraction
  - Training head only → re-learns balanced class boundaries
  - Lower LR (10x) → gentle updates prevent destroying features
  - Reduced augmentation → prevents overfitting on rare class

Target: ~90% mAP50 (+9% over base)

Usage:
    python scripts/train_tfa.py
    python scripts/train_tfa.py --weights path/to/custom/best.pt
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="TFA Few-Shot Fine-Tuning")
    parser.add_argument(
        "--weights",
        type=str,
        default="experiments/exp_002_ghost_hybrid_medium/weights/best.pt",
        help="Path to exp_002 best weights (default: experiments/exp_002_ghost_hybrid_medium/weights/best.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of fine-tuning epochs (default: 50)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8, optimized for 4GB VRAM)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use (default: 0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  TFA: Two-Stage Fine-Tuning Approach")
    print("  Stage 2 - Freeze Backbone, Fine-Tune Head")
    print("=" * 60)

    # ── Paths ──
    weights_path = Path(args.weights)
    data_yaml = "VOC/voc.yaml"

    # Verify files exist
    assert weights_path.exists(), (
        f"Base model weights not found: {weights_path}\n"
        f"  → Run exp_002 training first: python scripts/train_exp002.py"
    )
    assert Path(data_yaml).exists(), f"Data YAML not found: {data_yaml}"

    print(f"\n  Base weights:  {weights_path}")
    print(f"  Dataset:       {data_yaml}")
    print(f"  Freeze layers: 0-9 (backbone)")
    print(f"  Train layers:  10-23 (head + detection)")
    print(f"  Learning rate: 0.0001 (10x lower than base)")
    print(f"  Augmentation:  Reduced (few-shot optimized)")

    # ── Load pre-trained model ──
    model = YOLO(str(weights_path), task="detect")

    # Print model info
    print(f"\n{'─' * 40}")
    print("Model Summary (from exp_002 best weights):")
    model.info(verbose=True)
    print(f"{'─' * 40}\n")

    # ── Timestamp for unique experiment name ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_tfa_{timestamp}"

    # ── TFA Fine-Tuning ──
    # Optimized for RTX 3050 (4GB VRAM) + 16GB RAM
    results = model.train(
        data=data_yaml,

        # ── CRITICAL: Freeze backbone ──
        freeze=10,  # Freeze layers 0-9 (backbone)

        # Training duration (shorter than base)
        epochs=args.epochs,
        patience=20,

        # Batch & image size - OPTIMIZED for 4GB VRAM
        batch=args.batch,   # Default 8, safe for 4GB VRAM
        imgsz=640,

        # Optimizer - LOW learning rate!
        optimizer="AdamW",
        lr0=0.0001,         # 10x lower than base (was 0.01)
        lrf=0.1,            # Final LR = 1e-5
        momentum=0.937,
        weight_decay=0.0005,

        # Short warmup
        warmup_epochs=2.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,

        # ── REDUCED AUGMENTATION (Key for Few-Shot!) ──
        # Heavy augmentation hurts with limited rare-class samples
        mosaic=0.5,          # REDUCED from 1.0
        mixup=0.0,           # DISABLED (was 0.1)
        copy_paste=0.0,      # DISABLED

        # Geometric - reduced
        degrees=5.0,         # REDUCED from 10.0
        translate=0.1,       # REDUCED from 0.2
        scale=0.3,           # REDUCED from 0.9
        shear=1.0,           # REDUCED from 2.0
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,

        # Color augmentation - reduced
        hsv_h=0.01,          # REDUCED from 0.015
        hsv_s=0.5,           # REDUCED from 0.7
        hsv_v=0.3,           # REDUCED from 0.4

        # Erasing - reduced
        erasing=0.2,         # REDUCED from 0.4

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Performance - OPTIMIZED for 4GB VRAM + 16GB RAM
        cache="ram",         # Cache in RAM (not VRAM)
        device=args.device,
        workers=4,           # Reduced workers to save RAM
        amp=True,            # Mixed precision - saves ~40% VRAM

        # Logging
        project="experiments",
        name=exp_name,
        exist_ok=False,
        verbose=True,
        plots=True,
    )

    # ── Results ──
    best_weights = f"experiments/{exp_name}/weights/best.pt"

    print("\n" + "=" * 60)
    print("  TFA Fine-Tuning Complete!")
    print("=" * 60)
    print(f"  Best weights: {best_weights}")
    print(f"  Results CSV:  experiments/{exp_name}/results.csv")

    # ── Validation at optimal resolution ──
    print(f"\n{'─' * 40}")
    print("  Running validation at optimal resolution (704×704)...")
    print(f"{'─' * 40}")

    val_model = YOLO(best_weights)
    val_results = val_model.val(
        data=data_yaml,
        imgsz=704,       # Optimal inference resolution
        batch=8,
        device=args.device,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("  Optimal Inference Validation (704×704)")
    print("=" * 60)
    print(f"  mAP50:    {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")
    print("=" * 60)
    print(f"\n  Final model ready for deployment!")
    print(f"  Export command:")
    print(f"    yolo export model={best_weights} format=onnx imgsz=704 simplify=True")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
