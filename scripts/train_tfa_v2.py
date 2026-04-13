"""
TFA v2: Enhanced Few-Shot Fine-Tuning for Damaged_1 Class
==========================================================
Targeted improvements for rare class (Damaged_1) accuracy:

Strategy Changes from TFA v1:
  1. Higher classification loss weight (cls=2.0) → penalizes rare class errors more
  2. Copy-paste augmentation (0.3) → synthetically adds more Damaged_1 instances
  3. Vertical flip enabled (flipud=0.5) → doubles orientation diversity for insulators
  4. Slightly higher LR (0.0003) → stronger head re-learning
  5. Longer patience (30) → gives more time to find optimal weights
  6. Box loss increased (10.0) → better localization of small defects

Usage:
    python scripts/train_tfa_v2.py
    python scripts/train_tfa_v2.py --weights path/to/best.pt
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="TFA v2 - Enhanced Rare Class Fine-Tuning")
    parser.add_argument(
        "--weights",
        type=str,
        default="experiments/exp_002_ghost_hybrid_medium3/weights/best.pt",
        help="Path to base model best weights",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs (default: 80)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--device", type=str, default="0", help="Device")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  TFA v2: Enhanced Rare Class Fine-Tuning")
    print("  Target: Boost Damaged_1 mAP50 > 93%")
    print("=" * 60)

    weights_path = Path(args.weights)
    data_yaml = "VOC/voc.yaml"

    assert weights_path.exists(), f"Weights not found: {weights_path}\n  Run exp_002 first!"
    assert Path(data_yaml).exists(), f"Data YAML not found: {data_yaml}"

    print(f"\n  Strategy Changes from TFA v1:")
    print(f"  ├── cls loss:    0.5 → 2.0  (4x, penalize rare class errors)")
    print(f"  ├── box loss:    7.5 → 10.0 (better defect localization)")
    print(f"  ├── copy_paste:  0.0 → 0.3  (synthetic rare class augment)")
    print(f"  ├── flipud:      0.0 → 0.5  (orientation diversity)")
    print(f"  ├── lr0:         1e-4 → 3e-4 (stronger head re-learning)")
    print(f"  └── epochs:      50 → 80    (more time to converge)")

    model = YOLO(str(weights_path), task="detect")

    print(f"\n{'─' * 40}")
    model.info(verbose=True)
    print(f"{'─' * 40}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_tfa_v2_{timestamp}"

    # ══════════════════════════════════════════
    # TFA v2 TRAINING - Rare Class Focused
    # ══════════════════════════════════════════
    results = model.train(
        data=data_yaml,

        # Freeze backbone (same as TFA v1)
        freeze=10,

        # Training duration - longer for better convergence
        epochs=args.epochs,
        patience=30,         # Increased from 20

        # Batch & image size
        batch=args.batch,
        imgsz=640,

        # Optimizer - slightly higher LR for stronger head tuning
        optimizer="AdamW",
        lr0=0.0003,          # 3x higher than TFA v1 (was 0.0001)
        lrf=0.05,            # Lower final LR ratio
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,   # Slightly longer warmup
        warmup_momentum=0.8,
        warmup_bias_lr=0.01,

        # ── KEY CHANGE: Augmentation for rare class ──
        mosaic=0.6,          # Slightly higher than v1 (was 0.5)
        mixup=0.05,          # Light mixup (was 0.0)
        copy_paste=0.3,      # ENABLED - pastes rare class instances!
        
        # Geometric
        degrees=5.0,
        translate=0.15,      # Slightly more than v1
        scale=0.4,           # Slightly more than v1
        shear=1.0,
        perspective=0.0,
        flipud=0.5,          # ENABLED - vertical flip for insulators!
        fliplr=0.5,

        # Color - slightly more aggressive
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.35,
        erasing=0.3,

        # ── KEY CHANGE: Loss weights for rare class ──
        box=10.0,            # INCREASED from 7.5 (better localization)
        cls=2.0,             # 4x INCREASE from 0.5 (rare class focus!)
        dfl=1.5,

        # Performance
        cache="ram",
        device=args.device,
        workers=4,
        amp=True,

        # Logging
        project="experiments",
        name=exp_name,
        exist_ok=False,
        verbose=True,
        plots=True,
    )

    # ── Validation at multiple resolutions ──
    best_weights = f"experiments/{exp_name}/weights/best.pt"
    val_model = YOLO(best_weights)

    print("\n" + "=" * 60)
    print("  Multi-Scale Validation")
    print("=" * 60)

    for imgsz in [640, 704]:
        print(f"\n{'─' * 40}")
        print(f"  Validation @ {imgsz}×{imgsz}")
        print(f"{'─' * 40}")
        val_results = val_model.val(
            data=data_yaml,
            imgsz=imgsz,
            batch=8,
            device=args.device,
            verbose=True,
        )
        print(f"  mAP50:    {val_results.box.map50:.4f}")
        print(f"  mAP50-95: {val_results.box.map:.4f}")

    print("\n" + "=" * 60)
    print(f"  TFA v2 Complete! Best weights: {best_weights}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
