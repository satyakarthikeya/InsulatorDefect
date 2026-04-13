"""
Experiment 014 — Stage 2: Head-Only Fine-Tune from KD@768 Weights
==================================================================
After Stage 1 (KD at 768px), the backbone has learned teacher's knowledge
at high resolution. Now freeze the backbone and polish ONLY the detection
head with a higher learning rate.

This mirrors exp_012's approach (which gave the +0.35% push to 96.46%)
but starts from the stronger KD@768 base.

Pipeline:
  exp_014_kd_768 (Stage 1) ──→ freeze backbone ──→ head fine-tune ──→ exp_014_stage2

Expected: If Stage 1 produces ~96.2%+, this head polish could push to ~96.5-97%.

Hardware: RTX 3050 4GB VRAM — batch=2, single model (no teacher needed)

Usage:
    python scripts/train_exp014_stage2_head.py
    python scripts/train_exp014_stage2_head.py --weights experiments/exp_014_kd_768/weights/best.pt
    python scripts/train_exp014_stage2_head.py --lr0 0.003 --epochs 20
"""

import os
import sys
import argparse
from pathlib import Path

# ── Prevent CUDA memory fragmentation ──
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Project root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="EXP 014 Stage 2: Head-Only Fine-Tune from KD@768"
    )
    parser.add_argument(
        "--weights", type=str,
        default="experiments/exp_014_kd_768/weights/best.pt",
        help="Best weights from Stage 1 (KD@768)",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Fine-tuning epochs")
    parser.add_argument("--batch", type=int, default=2, help="Batch size (2 for 768 on 4GB)")
    parser.add_argument("--lr0", type=float, default=0.005, help="Initial LR (higher for head-only)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    return parser.parse_args()


def freeze_backbone(model):
    """Freeze all backbone layers, keep detection head trainable.

    The Ghost-Hybrid-P3P4-Medium architecture:
      - Layers 0-9: Backbone (GhostConv, C3Ghost, SPPF)
      - Layers 10+: Detection head (FPN-PAN, Detect)

    We freeze layers 0-9 so only the head adapts.
    """
    frozen_count = 0
    trainable_count = 0

    for name, param in model.model.named_parameters():
        # Extract layer index from parameter name (e.g., "model.0.conv.weight" → 0)
        parts = name.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            layer_idx = int(parts[1])
            if layer_idx <= 9:
                param.requires_grad = False
                frozen_count += 1
            else:
                param.requires_grad = True
                trainable_count += 1
        else:
            # Non-indexed params (rare) — keep trainable
            param.requires_grad = True
            trainable_count += 1

    frozen_params = sum(p.numel() for p in model.model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    print(f"\n  Backbone frozen:")
    print(f"  ├── Frozen params:    {frozen_params:,} ({frozen_count} tensors)")
    print(f"  ├── Trainable params: {trainable_params:,} ({trainable_count} tensors)")
    print(f"  └── % Trainable:      {trainable_params/(frozen_params+trainable_params)*100:.1f}%")


def main():
    args = parse_args()

    print("=" * 60)
    print("  EXP 014 — Stage 2: Head-Only Fine-Tune @ 768px")
    print("  Freeze backbone, polish detection head")
    print("=" * 60)

    weights_path = Path(args.weights)
    data_yaml = "VOC/voc.yaml"

    assert weights_path.exists(), f"Stage 1 weights not found: {weights_path}"
    assert Path(data_yaml).exists(), f"Data YAML not found: {data_yaml}"

    print(f"\n  Configuration:")
    print(f"  ├── Weights:    {args.weights}")
    print(f"  ├── Resolution: 768×768")
    print(f"  ├── Epochs:     {args.epochs}")
    print(f"  ├── Batch:      {args.batch}")
    print(f"  ├── LR:         {args.lr0} (head-only, higher is OK)")
    print(f"  └── Save to:    experiments/exp_014_stage2_head/")

    # Load model
    model = YOLO(str(weights_path))

    # Freeze backbone
    freeze_backbone(model)

    # ══════════════════════════════════════════════════════
    #  Head-Only Fine-Tune
    # ══════════════════════════════════════════════════════
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=768,
        batch=args.batch,
        device=args.device,
        workers=4,
        cache="ram",
        amp=True,

        # Optimizer — SGD with higher LR (only head is learning)
        optimizer="SGD",
        lr0=args.lr0,
        lrf=0.1,            # End at 10% of lr0
        cos_lr=True,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1,

        # Augmentation — minimal (just polishing the head)
        mosaic=0.5,
        close_mosaic=5,
        mixup=0.0,
        copy_paste=0.0,
        degrees=3.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,

        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Save
        save_period=5,
        patience=12,
        project="experiments",
        name="exp_014_stage2_head",
        exist_ok=True,
        verbose=True,
        plots=True,

        # Freeze backbone layers 0-9
        freeze=10,
    )

    # ══════════════════════════════════════════════════════
    #  Post-Training Validation
    # ══════════════════════════════════════════════════════
    best_weights = "experiments/exp_014_stage2_head/weights/best.pt"

    print("\n" + "=" * 60)
    print("  Stage 2 Complete! Running final validation...")
    print("=" * 60)

    if Path(best_weights).exists():
        val_model = YOLO(best_weights)

        # Standard validation
        print(f"\n{'─' * 40}")
        print(f"  Final Validation @ 768×768")
        print(f"{'─' * 40}")
        results = val_model.val(
            data=data_yaml, imgsz=768, batch=2, device=args.device, verbose=True,
        )
        map50 = results.box.map50
        map50_95 = results.box.map
        print(f"  mAP50:    {map50:.4f}")
        print(f"  mAP50-95: {map50_95:.4f}")
        if hasattr(results.box, "ap50") and len(results.box.ap50) >= 2:
            print(f"  ├── Damaged_1: {results.box.ap50[0]:.4f}")
            print(f"  └── insulator: {results.box.ap50[1]:.4f}")

        # TTA validation
        print(f"\n{'─' * 40}")
        print(f"  Final Validation @ 768×768 (TTA)")
        print(f"{'─' * 40}")
        val_model.model.stride = torch.tensor([8., 16., 32.])  # TTA stride fix
        results_tta = val_model.val(
            data=data_yaml, imgsz=768, batch=1, device=args.device,
            augment=True, verbose=True,
        )
        tta_map50 = results_tta.box.map50
        print(f"  TTA mAP50:    {tta_map50:.4f}")
        print(f"  TTA mAP50-95: {results_tta.box.map:.4f}")

        # Final comparison
        print(f"\n{'═' * 60}")
        print(f"  FINAL RESULTS — EXP 014 (Full Pipeline):")
        print(f"  ├── Previous best (exp_012):      96.46% mAP50")
        print(f"  ├── Previous best + TTA:           96.21% mAP50 (validated)")
        print(f"  ├── EXP 014 Stage 2 (no TTA):     {map50*100:.2f}% mAP50")
        print(f"  └── EXP 014 Stage 2 + TTA:        {tta_map50*100:.2f}% mAP50")
        print(f"{'═' * 60}")

        if map50 > 0.9646:
            print(f"\n  NEW ALL-TIME RECORD! {map50*100:.2f}% > 96.46%")
        elif tta_map50 > 0.9621:
            print(f"\n  NEW TTA RECORD! {tta_map50*100:.2f}% > 96.21%")
        else:
            print(f"\n  Saturated. Try different hyperparams:")
            print(f"  - Stage 1: --kd-alpha 0.2 --kd-temperature 2.0")
            print(f"  - Stage 2: --lr0 0.003 --epochs 30")
    else:
        print(f"  WARNING: Best weights not found at {best_weights}")

    print(f"\n  Best weights: {best_weights}")
    print(f"  Results CSV:  experiments/exp_014_stage2_head/results.csv")


if __name__ == "__main__":
    main()
