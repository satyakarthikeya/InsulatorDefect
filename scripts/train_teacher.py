"""
Phase 1: Train Teacher Model (YOLO11m) for Knowledge Distillation
===================================================================
Train a YOLO11m (Medium) model on the VOC dataset to serve as the
"Oracle" teacher for our lightweight 867K-parameter student model.

The teacher's soft-label distributions will be used in Phase 2
(scripts/train_kd.py) to distill knowledge into the student.

Architecture: YOLO11m (~20M params) — pretrained on COCO
Target: ~97-98% mAP50 on our 2-class VOC dataset
Hardware: RTX 3050 (4GB VRAM) — single model, batch=8 is safe

Usage:
    python scripts/train_teacher.py
    python scripts/train_teacher.py --epochs 100 --batch 8
    python scripts/train_teacher.py --device 0
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
        description="Phase 1: Train YOLO11m Teacher for Knowledge Distillation"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11m.pt",
        help="Pretrained YOLO11m weights (auto-downloads if not found)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (4 for YOLO11m on 4GB VRAM)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Phase 1: Train Teacher Model (YOLO11m)")
    print("  Knowledge Distillation Pipeline")
    print("=" * 60)

    data_yaml = "VOC/voc.yaml"

    # ── Verify dataset exists ──
    assert Path(data_yaml).exists(), f"Data YAML not found: {data_yaml}"

    print(f"\n  Training Configuration:")
    print(f"  ├── Teacher:   {args.weights}")
    print(f"  ├── Dataset:   {data_yaml}")
    print(f"  ├── Epochs:    {args.epochs}")
    print(f"  ├── Batch:     {args.batch}  (YOLO11m needs batch=4 on 4GB VRAM)")
    print(f"  ├── Image sz:  640")
    print(f"  ├── Device:    cuda:{args.device}")
    print(f"  └── Save to:   experiments/exp_004_teacher_yolo11m/")

    # ── Load pretrained YOLO11m (auto-downloads from Ultralytics hub) ──
    model = YOLO(args.weights, task="detect")

    print(f"\n{'─' * 40}")
    print("Teacher Model Summary:")
    model.info(verbose=True)
    print(f"{'─' * 40}\n")

    # ══════════════════════════════════════════════════════════
    #  Train Teacher — YOLO11m on VOC (2 classes)
    #  Single model in VRAM → batch=8 is safe on RTX 3050
    # ══════════════════════════════════════════════════════════
    results = model.train(
        data=data_yaml,

        # Training duration
        epochs=args.epochs,
        patience=30,

        # Batch & image size — YOLO11m needs batch=4 on 4GB VRAM
        batch=args.batch,
        imgsz=640,

        # Optimizer
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,

        # Warmup
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Augmentation — standard YOLO augmentation
        # No heavy rare-class aug (per agent.md rules)
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

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Performance — optimized for RTX 3050 (4GB VRAM) + 16GB RAM
        cache="ram",            # Cache images in RAM (~2GB)
        device=args.device,
        workers=4,              # Prevent RAM thrashing
        amp=True,               # Mixed precision — saves ~40% VRAM

        # Logging
        project="experiments",
        name="exp_004_teacher_yolo11m",
        exist_ok=False,
        verbose=True,
        plots=True,
    )

    # ── Post-Training Validation ──
    best_weights = "experiments/exp_004_teacher_yolo11m/weights/best.pt"

    print("\n" + "=" * 60)
    print("  Teacher Training Complete!")
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
        print(f"  Teacher mAP50:    {val_results.box.map50:.4f}")
        print(f"  Teacher mAP50-95: {val_results.box.map:.4f}")
    else:
        print(f"  WARNING: Best weights not found at {best_weights}")

    print("\n" + "=" * 60)
    print(f"  Best weights: {best_weights}")
    print(f"  Results CSV:  experiments/exp_004_teacher_yolo11m/results.csv")
    print(f"\n  Next step: Run Knowledge Distillation with:")
    print(f"    python scripts/train_kd.py")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
