"""
Phase 1 v2: Train Teacher Model (YOLO11s) for Knowledge Distillation
======================================================================
Retry teacher training with YOLO11s instead of YOLO11m.

Why YOLO11m failed (88% mAP50 vs student's 94%):
  - 20M params is over-parameterized for ~500 images / 2 classes
  - High LR (0.01) caused oscillating val loss (83-88% mAP50 swings)
  - Model never properly converged

Fixes in this version:
  - YOLO11s (~9.4M params) — better capacity match for small dataset
  - Lower LR (0.002) — less aggressive, more stable convergence
  - Longer training (150 epochs, patience=40) — give it time to settle
  - Stronger weight_decay (0.001) — regularize against overfitting
  - Freeze backbone first 10 epochs, then unfreeze — transfer learning

Architecture: YOLO11s (~9.4M params) — pretrained on COCO
Target: >95% mAP50 (must beat student's 94.17%)
Hardware: RTX 3050 (4GB VRAM) — batch=4

Usage:
    python scripts/train_teacher_v2.py
    python scripts/train_teacher_v2.py --epochs 150 --batch 4
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
        description="Phase 1 v2: Train YOLO11s Teacher for Knowledge Distillation"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11s.pt",
        help="Pretrained YOLO11s weights (auto-downloads if not found)",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs (more than v1)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Phase 1 v2: Train Teacher Model (YOLO11s)")
    print("  Knowledge Distillation Pipeline — Retry")
    print("=" * 60)

    data_yaml = "VOC/voc.yaml"
    assert Path(data_yaml).exists(), f"Data YAML not found: {data_yaml}"

    print(f"\n  Why YOLO11m failed (v1):")
    print(f"  ├── 20M params overfits on ~500 images / 2 classes")
    print(f"  ├── LR=0.01 too aggressive → val mAP oscillated 83-88%")
    print(f"  └── Best was 88.25% — WORSE than student's 94.17%")
    print(f"\n  v2 Fixes:")
    print(f"  ├── YOLO11s (~9.4M params) — better capacity match")
    print(f"  ├── LR=0.002 — less aggressive")
    print(f"  ├── weight_decay=0.001 — stronger regularization")
    print(f"  ├── 150 epochs, patience=40 — longer to converge")
    print(f"  └── freeze=10 for first pass → transfer learning")

    print(f"\n  Training Configuration:")
    print(f"  ├── Teacher:   {args.weights}")
    print(f"  ├── Dataset:   {data_yaml}")
    print(f"  ├── Epochs:    {args.epochs}")
    print(f"  ├── Batch:     {args.batch}")
    print(f"  ├── Image sz:  640")
    print(f"  ├── Device:    cuda:{args.device}")
    print(f"  └── Save to:   experiments/exp_004_teacher_yolo11s/")

    # ── Load pretrained YOLO11s ──
    model = YOLO(args.weights, task="detect")

    print(f"\n{'─' * 40}")
    print("Teacher Model Summary:")
    model.info(verbose=True)
    print(f"{'─' * 40}\n")

    # ══════════════════════════════════════════════════════════
    #  Train Teacher — YOLO11s on VOC (2 classes)
    #  Key changes from v1: lower LR, stronger regularization,
    #  freeze backbone, more epochs
    # ══════════════════════════════════════════════════════════
    results = model.train(
        data=data_yaml,

        # Training duration — longer to let it stabilize
        epochs=args.epochs,
        patience=40,

        # Batch & image size
        batch=args.batch,
        imgsz=640,

        # Freeze backbone — transfer learning from COCO
        # YOLO11s backbone layers 0-9
        freeze=10,

        # Optimizer — MUCH lower LR than v1
        optimizer="AdamW",
        lr0=0.002,              # v1 was 0.01 — way too aggressive
        lrf=0.05,               # Final LR = 0.0001
        momentum=0.937,
        weight_decay=0.001,     # Stronger regularization (v1 was 0.0005)
        cos_lr=True,

        # Warmup — longer warmup for stability
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,

        # Augmentation — standard, not aggressive
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,         # Disabled per project rules
        degrees=10.0,
        translate=0.2,
        scale=0.9,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,             # Disabled per project rules
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,

        # Loss weights — same as student for consistency
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Performance — optimized for RTX 3050
        cache="ram",
        device=args.device,
        workers=4,
        amp=True,

        # Logging
        project="experiments",
        name="exp_004_teacher_yolo11s",
        exist_ok=False,
        verbose=True,
        plots=True,
    )

    # ── Post-Training Validation ──
    best_weights = "experiments/exp_004_teacher_yolo11s/weights/best.pt"

    print("\n" + "=" * 60)
    print("  Teacher v2 Training Complete!")
    print("=" * 60)

    if Path(best_weights).exists():
        print(f"\n  Validating teacher model...")
        val_model = YOLO(best_weights)

        print(f"\n{'─' * 40}")
        print(f"  Validation @ 640×640")
        print(f"{'─' * 40}")
        val_results = val_model.val(
            data=data_yaml,
            imgsz=640,
            batch=4,
            device=args.device,
            verbose=True,
        )
        teacher_map50 = val_results.box.map50
        teacher_map = val_results.box.map
        print(f"  Teacher mAP50:    {teacher_map50:.4f}")
        print(f"  Teacher mAP50-95: {teacher_map:.4f}")
        print(f"\n  Student baseline: 0.9417 mAP50")
        if teacher_map50 > 0.9417:
            print(f"  ✓ Teacher BEATS student! KD should help.")
            print(f"    Proceed with: python scripts/train_kd.py --teacher {best_weights}")
        else:
            print(f"  ✗ Teacher still below student ({teacher_map50:.4f} < 0.9417)")
            print(f"    Consider: self-distillation or unfrozen high-res training instead")
    else:
        print(f"  WARNING: Best weights not found at {best_weights}")

    print("\n" + "=" * 60)
    print(f"  Best weights: {best_weights}")
    print(f"  Results CSV:  experiments/exp_004_teacher_yolo11s/results.csv")
    print(f"\n  Next step (if teacher > 94.17% mAP50):")
    print(f"    python scripts/train_kd.py --teacher {best_weights}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
