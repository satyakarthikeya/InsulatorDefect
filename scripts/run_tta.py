"""
Custom Multi-Scale TTA for P3P4 architecture.
Bypasses Ultralytics' built-in augment=True which crashes on 2-scale models.

Runs inference at multiple scales + horizontal flip, merges with class-aware NMS,
then computes mAP50 against ground truth labels.
"""
import torch
import numpy as np
import cv2
import os
import yaml
import torchvision
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent.parent
DATA_YAML = str(PROJECT / "VOC" / "voc.yaml")

# ─── Configuration ──────────────────────────────────────
TTA_CONFIGS = {
    "3scale_flip": {
        "scales": [576, 640, 704],
        "flip": True,
        "nms_iou": 0.6,
    },
    "5scale_flip": {
        "scales": [512, 576, 640, 704, 768],
        "flip": True,
        "nms_iou": 0.6,
    },
    "3scale_noflip": {
        "scales": [576, 640, 704],
        "flip": False,
        "nms_iou": 0.6,
    },
}


def load_val_data(data_yaml):
    """Load validation image paths and ground truth labels."""
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    
    val_img_dir = cfg.get("val", "")
    if not os.path.isabs(val_img_dir):
        val_img_dir = str(Path(data_yaml).parent / val_img_dir)
    
    nc = cfg.get("nc", 2)
    names = cfg.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    
    # Get image files
    img_files = sorted([
        os.path.join(val_img_dir, f) 
        for f in os.listdir(val_img_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    # Get label directory
    label_dir = val_img_dir.replace("/images/", "/labels/")
    
    # Load ground truths
    gts = {}  # img_idx -> list of (cls, x1, y1, x2, y2) in pixel coords
    for idx, img_path in enumerate(img_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        label_file = os.path.join(label_dir, Path(img_path).stem + ".txt")
        boxes = []
        if os.path.exists(label_file):
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = (cx - bw/2) * w
                        y1 = (cy - bh/2) * h
                        x2 = (cx + bw/2) * w
                        y2 = (cy + bh/2) * h
                        boxes.append((cls, x1, y1, x2, y2))
        gts[idx] = boxes
    
    return img_files, gts, nc, names


def run_tta_predictions(model, img_files, scales, do_flip, conf=0.001, per_scale_nms=0.7):
    """
    Run TTA: predict at each scale (and optionally flip), collect all boxes.
    Returns dict: img_idx -> np.array of (x1, y1, x2, y2, conf, cls) in original coords.
    """
    all_preds = defaultdict(list)  # img_idx -> list of [x1,y1,x2,y2,conf,cls]
    
    total_passes = len(scales) * (2 if do_flip else 1)
    print(f"  Running {total_passes} forward passes per image ({len(img_files)} images)...")
    
    for scale_idx, scale in enumerate(scales):
        flip_options = [False, True] if do_flip else [False]
        
        for flip in flip_options:
            tag = f"scale={scale}" + (" flip" if flip else "")
            
            for img_idx, img_path in enumerate(img_files):
                img = cv2.imread(img_path)
                if img is None:
                    continue
                h0, w0 = img.shape[:2]
                
                source = cv2.flip(img, 1) if flip else img
                
                results = model.predict(
                    source,
                    imgsz=scale,
                    conf=conf,
                    iou=per_scale_nms,
                    device="0",
                    verbose=False,
                    augment=False,
                )
                
                if results and len(results[0].boxes):
                    xyxy = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    clss = results[0].boxes.cls.cpu().numpy()
                    
                    if flip:
                        xyxy_orig = xyxy.copy()
                        xyxy_orig[:, 0] = w0 - xyxy[:, 2]
                        xyxy_orig[:, 2] = w0 - xyxy[:, 0]
                        xyxy = xyxy_orig
                    
                    for i in range(len(xyxy)):
                        all_preds[img_idx].append([
                            xyxy[i, 0], xyxy[i, 1], xyxy[i, 2], xyxy[i, 3],
                            confs[i], clss[i]
                        ])
            
            print(f"    Done: {tag}")
    
    return all_preds


def merge_predictions_nms(all_preds, nc, nms_iou=0.6):
    """Apply class-aware NMS to merge multi-scale predictions per image."""
    merged = {}
    
    for img_idx, boxes_list in all_preds.items():
        if not boxes_list:
            merged[img_idx] = np.empty((0, 6))
            continue
        
        boxes_arr = np.array(boxes_list)  # (N, 6): x1,y1,x2,y2,conf,cls
        
        keep_all = []
        for c in range(nc):
            mask = boxes_arr[:, 5] == c
            if mask.sum() == 0:
                continue
            c_boxes = torch.from_numpy(boxes_arr[mask, :4]).float()
            c_scores = torch.from_numpy(boxes_arr[mask, 4]).float()
            c_indices = np.where(mask)[0]
            
            keep = torchvision.ops.nms(c_boxes, c_scores, nms_iou)
            keep_all.extend(c_indices[keep.numpy()].tolist())
        
        merged[img_idx] = boxes_arr[keep_all] if keep_all else np.empty((0, 6))
    
    return merged


def compute_ap(recall, precision):
    """Compute AP using 101-point interpolation (COCO style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    recall_pts = np.linspace(0, 1, 101)
    prec_interp = np.zeros_like(recall_pts)
    for i, r in enumerate(recall_pts):
        mask = mrec >= r
        if mask.any():
            prec_interp[i] = mpre[mask].max()
    
    return prec_interp.mean()


def compute_iou(b1, b2):
    """IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


def compute_map50(merged_preds, gts, nc, names):
    """Compute mAP@50 from merged predictions and ground truths."""
    per_class = {}
    
    for c in range(nc):
        # Gather all preds and gts for this class
        preds_c = []  # (img_idx, conf, x1, y1, x2, y2)
        n_gt = 0
        gt_matched = {}  # (img_idx, gt_local_idx) -> bool
        gt_by_img = defaultdict(list)
        
        for img_idx, gt_boxes in gts.items():
            local_idx = 0
            for cls, x1, y1, x2, y2 in gt_boxes:
                if cls == c:
                    gt_by_img[img_idx].append((local_idx, x1, y1, x2, y2))
                    gt_matched[(img_idx, local_idx)] = False
                    local_idx += 1
                    n_gt += 1
        
        for img_idx, boxes in merged_preds.items():
            if boxes.shape[0] == 0:
                continue
            for box in boxes:
                if int(box[5]) == c:
                    preds_c.append((img_idx, box[4], box[0], box[1], box[2], box[3]))
        
        if n_gt == 0:
            per_class[c] = 0.0
            continue
        
        # Sort by confidence
        preds_c.sort(key=lambda x: x[1], reverse=True)
        
        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))
        
        for pi, (img_idx, conf, px1, py1, px2, py2) in enumerate(preds_c):
            if img_idx not in gt_by_img:
                fp[pi] = 1
                continue
            
            best_iou, best_gi = 0, -1
            for gi, gx1, gy1, gx2, gy2 in gt_by_img[img_idx]:
                iou = compute_iou([px1, py1, px2, py2], [gx1, gy1, gx2, gy2])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            
            if best_iou >= 0.5 and not gt_matched.get((img_idx, best_gi), True):
                tp[pi] = 1
                gt_matched[(img_idx, best_gi)] = True
            else:
                fp[pi] = 1
        
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt
        precision = tp_cum / (tp_cum + fp_cum)
        
        ap = compute_ap(recall, precision)
        per_class[c] = ap
        
        # Also get P/R at best F1
        if len(recall) > 0:
            f1 = 2 * precision * recall / (precision + recall + 1e-16)
            best_f1_idx = np.argmax(f1)
            p_at_best = precision[best_f1_idx]
            r_at_best = recall[best_f1_idx]
        else:
            p_at_best = r_at_best = 0
        
        cname = names.get(c, f"class_{c}")
        print(f"  {cname}: AP50={ap:.4f}  P={p_at_best:.3f} R={r_at_best:.3f} (nGT={n_gt}, nPred={len(preds_c)})")
    
    mAP50 = np.mean(list(per_class.values()))
    return mAP50, per_class


def main():
    print("=" * 70)
    print("CUSTOM MULTI-SCALE TTA EVALUATION")
    print("=" * 70)
    
    # Load data
    img_files, gts, nc, names = load_val_data(DATA_YAML)
    print(f"Loaded {len(img_files)} val images, {nc} classes: {names}")
    n_gt_total = sum(len(v) for v in gts.values())
    print(f"Total GT boxes: {n_gt_total}")
    
    # Models to test
    models_to_test = {
        "Baseline (exp_002)": str(PROJECT / "experiments/exp_002_ghost_hybrid_medium3/weights/best.pt"),
        "Soup 3way": str(PROJECT / "experiments/soup_3way.pt"),
        "Teacher (YOLO11s)": str(PROJECT / "experiments/exp_004_teacher_yolo11s/weights/best.pt"),
    }
    
    results_summary = []
    
    for model_name, model_path in models_to_test.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")
        
        model = YOLO(model_path)
        
        for tta_name, tta_cfg in TTA_CONFIGS.items():
            print(f"\n--- TTA: {tta_name} ---")
            
            # Run predictions
            raw_preds = run_tta_predictions(
                model, img_files,
                scales=tta_cfg["scales"],
                do_flip=tta_cfg["flip"],
                conf=0.001,
            )
            
            # Merge with NMS
            merged = merge_predictions_nms(raw_preds, nc, nms_iou=tta_cfg["nms_iou"])
            total_merged = sum(len(v) for v in merged.values())
            print(f"  Merged predictions: {total_merged}")
            
            # Compute mAP
            mAP50, per_class_ap = compute_map50(merged, gts, nc, names)
            print(f"  >>> mAP50 = {mAP50:.4f}")
            
            results_summary.append({
                "model": model_name,
                "tta": tta_name,
                "mAP50": mAP50,
                "per_class": per_class_ap,
            })
        
        del model
        torch.cuda.empty_cache()
    
    # Final summary
    print("\n\n" + "=" * 70)
    print("FINAL TTA RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'TTA Config':<20} {'mAP50':>8} {'D1 AP50':>8} {'ins AP50':>8}")
    print("-" * 70)
    for r in results_summary:
        d1 = r['per_class'].get(0, 0)  # class 0 = Damaged_1
        ins = r['per_class'].get(1, 0)  # class 1 = insulator
        print(f"{r['model']:<25} {r['tta']:<20} {r['mAP50']:>8.4f} {d1:>8.4f} {ins:>8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
