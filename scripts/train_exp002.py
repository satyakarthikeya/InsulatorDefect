"""
Experiment 002: Ghost-Hybrid-P3P4 Medium - Base Training
=========================================================
Train the 867K parameter lightweight YOLO model from scratch.
This is Stage 1 of the Two-Stage Fine-Tuning Approach (TFA).

Architecture: GhostConv backbone + DWConv head + P3/P4 detection
Target: ~81% mAP50 (base for TFA fine-tuning)

Usage:
    python scripts/train_exp002.py
"""

import os
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def main():
    print("=" * 60)
    print("  Experiment 002: Ghost-Hybrid-P3P4 Medium")
    print("  Base Training (Stage 1 of TFA)")
    print("=" * 60)

    # ── Model Architecture ──
    model_yaml = "models/yolo11n-ghost-hybrid-p3p4-medium.yaml"
    data_yaml = "VOC/voc.yaml"

    # Verify files exist
    assert Path(model_yaml).exists(), f"Model YAML not found: {model_yaml}"
    assert Path(data_yaml).exists(), f"Data YAML not found: {data_yaml}"

    # ── Load model from YAML (train from scratch) ──
    model = YOLO(model_yaml, task="detect")

    # Print model info
    print(f"\n{'─' * 40}")
    print("Model Summary:")
    model.info(verbose=True)
    print(f"{'─' * 40}\n")

    # ── Train ──
    # Optimized for RTX 3050 (4GB VRAM) + 16GB RAM
    results = model.train(
        data=data_yaml,

        # Training duration
        epochs=300,
        patience=50,

        # Batch & image size - OPTIMIZED for 4GB VRAM
        batch=8,            # Fixed batch size (safe for 4GB VRAM)
        imgsz=640,

        # Optimizer
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,

        # Warmup
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Augmentation (standard YOLO augmentation)
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
        cache="ram",        # Cache images in RAM (not VRAM) - uses ~2GB RAM
        device=0,
        workers=4,          # Reduced workers to save RAM
        amp=True,           # Mixed precision - reduces VRAM ~40%

        # Logging
        project="experiments",
        name="exp_002_ghost_hybrid_medium",
        exist_ok=False,
        verbose=True,
        plots=True,
    )

    # ── Results ──
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best weights: experiments/exp_002_ghost_hybrid_medium/weights/best.pt")
    print(f"  Results CSV:  experiments/exp_002_ghost_hybrid_medium/results.csv")
    print(f"\n  Next step: Run TFA fine-tuning with:")
    print(f"    python scripts/train_tfa.py")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
