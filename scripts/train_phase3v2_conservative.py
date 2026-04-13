"""
Phase 3v2: Conservative High-Resolution Fine-Tuning at 704
============================================================
Two-stage fine-tune: freeze backbone → unfreeze at 704.

Previous attempt (Phase 3v1) FAILED because lr0=0.005 was too aggressive,
causing wild mAP oscillation (85%→93%). This version uses:
  - 10x lower LR (0.0005)
  - Two-stage: frozen backbone first → then unfreeze
  - Exact proven augmentation from exp_002 (94.17%)
  - Start from ORIGINAL baseline weights (not KD)

Strategy:
  Stage 1: Freeze backbone (layers 0-9), train neck+head at 704 for 30 ep
  Stage 2: Unfreeze all, continue with 5x lower LR for 70 more epochs

Hardware: RTX 3050 4GB, batch=4 @ 704, amp=True

Usage:
    python scripts/train_phase3v2_conservative.py
    python scripts/train_phase3v2_conservative.py --stage 1
    python scripts/train_phase3v2_conservative.py --stage 2 --resume
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3v2: Conservative High-Res 704 Fine-Tuning"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="experiments/exp_002_ghost_hybrid_medium3/weights/best.pt",
        help="Starting weights (original 94.17%% baseline)",
    )
    parser.add_argument("--imgsz", type=int, default=704)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--stage", type=int, default=0,
        help="1=frozen backbone only, 2=unfrozen only, 0=both stages sequentially"
    )
    parser.add_argument("--resume", action="store_true", help="Resume stage 2 from stage 1 best.pt")
    return parser.parse_args()


def run_stage1(model_path, imgsz, batch, device):
    """Stage 1: Frozen backbone, adapt neck+head to 704 resolution."""
    print("\n" + "=" * 65)
    print("  Stage 1: Frozen Backbone — Adapt Neck+Head to 704")
    print("  Freeze layers 0-9 (backbone), train 10-23 (neck+head)")
    print("  LR=0.001, 40 epochs, patience=25")
    print("=" * 65 + "\n")

    model = YOLO(model_path, task="detect")

    results = model.train(
        data="VOC/voc.yaml",
        epochs=40,
        batch=batch,
        imgsz=imgsz,
        device=device,
        amp=True,
        cache="ram",
        workers=4,

        # ── Frozen backbone ──
        freeze=10,  # Freeze layers 0-9 (backbone)

        # ── Conservative LR ──
        lr0=0.001,
        lrf=0.1,
        cos_lr=True,
        warmup_epochs=3,
        warmup_bias_lr=0.01,
        optimizer="AdamW",
        weight_decay=0.0005,

        # ── Standard loss weights ──
        cls=0.5,
        box=7.5,
        dfl=1.5,

        # ── Exact exp_002 augmentation recipe ──
        mosaic=1.0,
        mixup=0.1,
        scale=0.9,
        degrees=10.0,
        translate=0.2,
        shear=2.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,
        fliplr=0.5,
        flipud=0.0,
        copy_paste=0.0,
        perspective=0.0,
        close_mosaic=10,

        # ── Training control ──
        patience=25,
        pretrained=True,
        deterministic=True,
        seed=0,
        plots=True,

        # ── Save ──
        project="experiments",
        name="exp_007_highres704_s1_frozen",
        exist_ok=True,
    )

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n  Stage 1 best weights: {best_path}")
    return str(best_path)


def run_stage2(model_path, imgsz, batch, device):
    """Stage 2: Unfreeze everything, fine-tune with very low LR."""
    print("\n" + "=" * 65)
    print("  Stage 2: Fully Unfrozen — Full Model Adaptation at 704")
    print(f"  Starting from: {model_path}")
    print("  LR=0.0002, 80 epochs, patience=30")
    print("=" * 65 + "\n")

    model = YOLO(model_path, task="detect")

    results = model.train(
        data="VOC/voc.yaml",
        epochs=80,
        batch=batch,
        imgsz=imgsz,
        device=device,
        amp=True,
        cache="ram",
        workers=4,

        # ── Fully unfrozen ──
        # No freeze parameter = all layers trainable

        # ── Very conservative LR for full fine-tune ──
        lr0=0.0002,
        lrf=0.05,
        cos_lr=True,
        warmup_epochs=5,
        warmup_bias_lr=0.005,
        optimizer="AdamW",
        weight_decay=0.0005,

        # ── Standard loss weights ──
        cls=0.5,
        box=7.5,
        dfl=1.5,

        # ── Exact exp_002 augmentation recipe ──
        mosaic=1.0,
        mixup=0.1,
        scale=0.9,
        degrees=10.0,
        translate=0.2,
        shear=2.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,
        fliplr=0.5,
        flipud=0.0,
        copy_paste=0.0,
        perspective=0.0,
        close_mosaic=10,

        # ── Training control ──
        patience=30,
        pretrained=True,
        deterministic=True,
        seed=0,
        plots=True,

        # ── Save ──
        project="experiments",
        name="exp_007_highres704_s2_unfrozen",
        exist_ok=True,
    )

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    return str(best_path), results.save_dir


def validate_final(best_path, imgsz, device):
    """Run per-class validation."""
    print("\n" + "=" * 65)
    print("  Final Validation — Per-Class Breakdown")
    print("=" * 65)

    model = YOLO(best_path, task="detect")

    for sz in [imgsz, 640]:
        r = model.val(
            data="VOC/voc.yaml", imgsz=sz, batch=4,
            device=device, plots=False, verbose=False
        )
        ap = r.box.ap50
        p = r.box.p
        rec = r.box.r
        print(f"  @{sz}: mAP50={r.box.map50:.4f} | "
              f"Damaged_1={ap[0]:.4f}(P={p[0]:.3f},R={rec[0]:.3f}) | "
              f"insulator={ap[1]:.4f}(P={p[1]:.3f},R={rec[1]:.3f})")

    print(f"\n  Best weights: {best_path}")
    print("=" * 65)


def main():
    args = parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: Weights not found: {args.model}")
        sys.exit(1)

    print("=" * 65)
    print("  Phase 3v2: Conservative High-Resolution Fine-Tuning")
    print(f"  Model:  {args.model}")
    print(f"  ImgSz:  {args.imgsz}")
    print(f"  Strategy: Frozen backbone → Unfreeze → Low LR")
    print("=" * 65)

    if args.stage == 1:
        run_stage1(args.model, args.imgsz, args.batch, args.device)
    elif args.stage == 2:
        s1_best = "experiments/exp_007_highres704_s1_frozen/weights/best.pt"
        if args.resume and Path(s1_best).exists():
            model_path = s1_best
        else:
            model_path = args.model
        best_path, _ = run_stage2(model_path, args.imgsz, args.batch, args.device)
        validate_final(best_path, args.imgsz, args.device)
    else:
        # Full pipeline: Stage 1 → Stage 2
        s1_best = run_stage1(args.model, args.imgsz, args.batch, args.device)
        if Path(s1_best).exists():
            best_path, _ = run_stage2(s1_best, args.imgsz, args.batch, args.device)
            validate_final(best_path, args.imgsz, args.device)
        else:
            print("ERROR: Stage 1 failed to produce best.pt")
            sys.exit(1)


if __name__ == "__main__":
    main()
