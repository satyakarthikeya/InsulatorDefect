"""
TTA (Test-Time Augmentation) Validation
========================================
Multi-scale + flip inference for free mAP boost.
Compares standard vs TTA at multiple resolutions.

Usage:
    python scripts/validate_tta.py
    python scripts/validate_tta.py --model experiments/exp_005_kd_student3/weights/best.pt
    python scripts/validate_tta.py --model best.pt --sizes 640 704
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def validate(model_path, imgsz, augment=False, name_suffix=""):
    """Run validation and return metrics."""
    model = YOLO(model_path, task="detect")
    tag = "TTA" if augment else "STD"
    name = f"val_{tag}_{imgsz}{name_suffix}"

    results = model.val(
        data="VOC/voc.yaml",
        imgsz=imgsz,
        batch=4,
        device="0",
        split="val",
        augment=augment,
        save_json=False,
        plots=False,
        name=name,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="TTA Validation — free mAP boost")
    parser.add_argument(
        "--model",
        type=str,
        default="experiments/exp_005_kd_student3/weights/best.pt",
        help="Model weights to validate",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[640, 704],
        help="Image sizes to test",
    )
    args = parser.parse_args()

    model_name = Path(args.model).parent.parent.name
    print("=" * 65)
    print("  TTA Validation — Multi-Scale Augmented Inference")
    print(f"  Model: {args.model}")
    print(f"  Sizes: {args.sizes}")
    print("=" * 65)

    results_table = []

    for imgsz in args.sizes:
        for augment in [False, True]:
            tag = "TTA" if augment else "Standard"
            print(f"\n{'─' * 50}")
            print(f"  Validating: {tag} @ {imgsz}")
            print(f"{'─' * 50}")

            r = validate(args.model, imgsz, augment=augment)

            # Extract per-class metrics
            map50 = r.box.map50
            map50_95 = r.box.map
            class_map50 = r.box.ap50  # per-class AP50 array

            results_table.append({
                "mode": tag,
                "imgsz": imgsz,
                "mAP50": map50,
                "mAP50-95": map50_95,
                "Damaged_1": class_map50[0] if len(class_map50) > 0 else 0,
                "insulator": class_map50[1] if len(class_map50) > 1 else 0,
            })

    # ── Summary Table ──
    print("\n" + "=" * 75)
    print("  RESULTS SUMMARY")
    print("=" * 75)
    print(f"  {'Mode':<10} {'ImgSz':<7} {'mAP50':>8} {'mAP50-95':>10} {'Damaged_1':>11} {'insulator':>11}")
    print(f"  {'─' * 10} {'─' * 7} {'─' * 8} {'─' * 10} {'─' * 11} {'─' * 11}")

    for r in results_table:
        print(
            f"  {r['mode']:<10} {r['imgsz']:<7} "
            f"{r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} "
            f"{r['Damaged_1']:>11.4f} {r['insulator']:>11.4f}"
        )

    # ── Best result ──
    best = max(results_table, key=lambda x: x["mAP50"])
    print(f"\n  🏆 Best: {best['mode']} @ {best['imgsz']} → mAP50={best['mAP50']:.4f}")
    print(f"     Damaged_1={best['Damaged_1']:.4f}, insulator={best['insulator']:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
