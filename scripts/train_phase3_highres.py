"""
Phase 3: High-Resolution Fine-Tuning at 704
=============================================
Fully unfrozen fine-tune at imgsz=704 to capture fine-grained spatial
features for Damaged_1 class improvement.

Strategy:
  - Start from best KD student weights (94.2% mAP50)
  - Train at 704 resolution (fully unfrozen backbone + neck + head)
  - Boosted cls loss weight (1.0 → 2x default) for classification focus
  - Proven augmentation recipe from exp_002 (which achieved 94.17%)
  - Lower learning rate for stable fine-tuning
  - cos_lr for smooth convergence

Previous attempts:
  - Frozen backbone at 704: FAILED (93.11% — spatial mismatch)
  - This run: UNFROZEN at 704 to let backbone adapt to new resolution

Hardware: RTX 3050 4GB, batch=4 @ 704, amp=True
Expected VRAM: ~3.2GB (704² is 1.21x area of 640², batch halved 8→4)

Usage:
    python scripts/train_phase3_highres.py
    python scripts/train_phase3_highres.py --imgsz 704 --cls 1.0
    python scripts/train_phase3_highres.py --imgsz 768 --batch 2
"""

import os
import sys
import argparse
from pathlib import Path

# ── Project root setup ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3: High-Resolution Fine-Tuning at 704"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="experiments/exp_005_kd_student3/weights/best.pt",
        help="Starting weights (KD student or exp_002 baseline)",
    )
    parser.add_argument("--imgsz", type=int, default=704, help="Training image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (4 for 704, 2 for 768)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--cls", type=float, default=1.0, help="Classification loss weight (default 0.5, boosted to 1.0)")
    parser.add_argument("--lr0", type=float, default=0.005, help="Initial LR (lower for fine-tuning)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify weights exist
    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        # Fallback to original exp_002 weights
        fallback = "experiments/exp_002_ghost_hybrid_medium3/weights/best.pt"
        if Path(fallback).exists():
            print(f"  Using fallback: {fallback}")
            args.model = fallback
        else:
            sys.exit(1)

    print("=" * 65)
    print("  Phase 3: High-Resolution Fine-Tuning")
    print("  Unfrozen 704 — Boosted Classification")
    print("=" * 65)
    print(f"\n  Configuration:")
    print(f"  ├── Model:     {args.model}")
    print(f"  ├── Image sz:  {args.imgsz}")
    print(f"  ├── Batch:     {args.batch}")
    print(f"  ├── Epochs:    {args.epochs}")
    print(f"  ├── Patience:  {args.patience}")
    print(f"  ├── CLS weight:{args.cls} (default=0.5)")
    print(f"  ├── LR0:       {args.lr0}")
    print(f"  ├── Device:    cuda:{args.device}")
    print(f"  └── Strategy:  Fully unfrozen, cos_lr, higher cls emphasis")
    print()

    model = YOLO(args.model, task="detect")

    results = model.train(
        data="VOC/voc.yaml",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        amp=True,
        cache="ram",
        workers=4,

        # ── Fine-tuning learning rate ──
        lr0=args.lr0,
        lrf=0.05,           # Decay to 5% of lr0
        cos_lr=True,         # Cosine annealing for smooth convergence
        warmup_epochs=5,     # Longer warmup for resolution change
        warmup_bias_lr=0.01,
        optimizer="AdamW",
        weight_decay=0.0005,

        # ── Loss weights — boosted classification ──
        cls=args.cls,        # 2x default (0.5→1.0) — prioritize classification
        box=7.5,             # Keep standard box loss
        dfl=1.5,             # Keep standard DFL loss

        # ── Augmentation — proven recipe from exp_002 ──
        mosaic=1.0,          # Full mosaic
        mixup=0.1,           # Mild mixup
        scale=0.9,           # Aggressive multi-scale
        degrees=10.0,        # Rotation for damage at various angles
        translate=0.2,       # Translation
        shear=2.0,           # Mild shear
        hsv_h=0.015,
        hsv_s=0.7,           # Strong saturation jitter
        hsv_v=0.4,           # Value jitter
        erasing=0.4,         # Random erasing for robustness
        fliplr=0.5,          # Horizontal flip
        flipud=0.0,          # NO vertical flip (per agent.md)
        copy_paste=0.0,      # NO copy_paste (per agent.md)
        perspective=0.0,
        close_mosaic=10,     # Disable mosaic for last 10 epochs

        # ── Training control ──
        patience=args.patience,
        pretrained=True,
        deterministic=True,
        seed=0,
        plots=True,

        # ── Save settings ──
        project="experiments",
        name=f"exp_006_highres_{args.imgsz}",
        exist_ok=False,
    )

    # ── Post-training: TTA validation at training resolution ──
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        print("\n" + "=" * 65)
        print("  Post-Training: TTA Validation")
        print("=" * 65)

        best_model = YOLO(str(best_weights), task="detect")

        for augment in [False, True]:
            tag = "TTA" if augment else "Standard"
            print(f"\n  Validating: {tag} @ {args.imgsz}")
            r = best_model.val(
                data="VOC/voc.yaml",
                imgsz=args.imgsz,
                batch=4,
                device=args.device,
                augment=augment,
                plots=False,
                name=f"val_{tag.lower()}_{args.imgsz}",
            )

            class_ap50 = r.box.ap50
            print(f"  {tag}: mAP50={r.box.map50:.4f} | "
                  f"Damaged_1={class_ap50[0]:.4f} | "
                  f"insulator={class_ap50[1]:.4f}")

        print("\n" + "=" * 65)
        print(f"  Best weights: {best_weights}")
        print("=" * 65)
    else:
        print(f"\n  WARNING: Best weights not found at {best_weights}")


if __name__ == "__main__":
    main()
