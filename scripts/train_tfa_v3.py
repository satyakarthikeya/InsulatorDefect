"""
TFA v3: Conservative Fine-Tuning + High Resolution
=====================================================
Strategy: The base exp_002 model is already strong (94.2% mAP50).
The key to pushing past 95% is:

1. Train at 704×704 (higher res → better small defect detection)
2. VERY gentle fine-tuning (lr=5e-5, minimal aug changes)
3. Close mosaic early (epoch 10) for cleaner final training
4. Keep backbone frozen, but be very conservative with head

Key lesson from TFA v1/v2:
  - v1: Too low LR (0.0001) → barely improved (94.1%)
  - v2: Too aggressive cls loss (2.0) → HURT performance (93.7%)
  - v3: Sweet spot - moderate LR, no cls weight change, higher resolution

Usage:
    python scripts/train_tfa_v3.py
    python scripts/train_tfa_v3.py --weights path/to/best.pt
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
    parser = argparse.ArgumentParser(description="TFA v3 - Conservative High-Res Fine-Tuning")
    parser.add_argument(
        "--weights",
        type=str,
        default="experiments/exp_002_ghost_hybrid_medium3/weights/best.pt",
        help="Path to base model best weights",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (4 for 704 on 4GB VRAM)")
    parser.add_argument("--device", type=str, default="0", help="Device")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  TFA v3: Conservative + High Resolution (704×704)")
    print("  Target: mAP50 > 95%")
    print("=" * 60)

    weights_path = Path(args.weights)
    data_yaml = "VOC/voc.yaml"

    assert weights_path.exists(), f"Weights not found: {weights_path}"
    assert Path(data_yaml).exists(), f"Data YAML not found: {data_yaml}"

    print(f"\n  Key changes from v1/v2:")
    print(f"  ├── imgsz:       640 → 704  (higher res for small defects)")
    print(f"  ├── lr0:         5e-5       (very gentle)")
    print(f"  ├── cls loss:    0.5        (NO change - v2's 2.0 was too aggressive)")
    print(f"  ├── box loss:    7.5        (NO change)")
    print(f"  ├── close_mosaic: 10        (cleaner training in final epochs)")
    print(f"  ├── copy_paste:  0.0        (disabled - was hurting in v2)")
    print(f"  └── batch:       4          (fits 704×704 in 4GB VRAM)")

    model = YOLO(str(weights_path), task="detect")

    print(f"\n{'─' * 40}")
    model.info(verbose=True)
    print(f"{'─' * 40}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_tfa_v3_{timestamp}"

    # ══════════════════════════════════════════
    # TFA v3: CONSERVATIVE + HIGH RESOLUTION
    # ══════════════════════════════════════════
    results = model.train(
        data=data_yaml,

        # Freeze backbone
        freeze=10,

        # Training duration
        epochs=args.epochs,
        patience=25,

        # HIGH RESOLUTION - key for small defect detection
        batch=args.batch,    # 4 for 704×704 on 4GB VRAM
        imgsz=704,           # HIGHER than base training (was 640)

        # Optimizer - VERY gentle
        optimizer="AdamW",
        lr0=0.00005,         # 5e-5 (very conservative)
        lrf=0.1,             # Final LR = 5e-6
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.005,

        # Augmentation - KEEP CLOSE TO BASE (v2 showed aggressive changes hurt!)
        mosaic=0.8,          # Slightly reduced from 1.0
        close_mosaic=10,     # Disable mosaic last 10 epochs for cleaner training
        mixup=0.05,          # Very light mixup
        copy_paste=0.0,      # DISABLED (hurt in v2)

        # Geometric - almost same as base
        degrees=8.0,         # Slightly reduced from 10.0
        translate=0.2,       # Same as base
        scale=0.7,           # Slightly reduced from 0.9
        shear=2.0,           # Same as base
        perspective=0.0,
        flipud=0.0,          # Disabled (v2 showed flipud didn't help)
        fliplr=0.5,          # Same as base

        # Color augmentation - same as base
        hsv_h=0.015,         # Same as base
        hsv_s=0.7,           # Same as base
        hsv_v=0.4,           # Same as base
        erasing=0.4,         # Same as base

        # Loss weights - NO CHANGES (v2 showed cls=2.0 was destructive)
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Performance - optimized for 704×704 on 4GB VRAM
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

    # ── Multi-scale Validation ──
    best_weights = f"experiments/{exp_name}/weights/best.pt"
    val_model = YOLO(best_weights)

    print("\n" + "=" * 60)
    print("  Multi-Scale Validation")
    print("=" * 60)

    for imgsz in [640, 704, 768]:
        print(f"\n{'─' * 40}")
        print(f"  Validation @ {imgsz}×{imgsz}")
        print(f"{'─' * 40}")
        val_results = val_model.val(
            data=data_yaml,
            imgsz=imgsz,
            batch=4,
            device=args.device,
            verbose=True,
        )
        print(f"  mAP50:    {val_results.box.map50:.4f}")
        print(f"  mAP50-95: {val_results.box.map:.4f}")

    print("\n" + "=" * 60)
    print(f"  TFA v3 Complete! Best weights: {best_weights}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
