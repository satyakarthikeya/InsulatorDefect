"""
Model Soup + Custom Multi-Scale TTA for pushing mAP50 beyond 94.17% baseline.

Strategy 1: Model Soup — Average weights from exp_002 (94.21%) and exp_005_kd (94.53%)
Strategy 2: Custom TTA — Multi-scale + flip inference with NMS merging
Strategy 3: Combined Soup + TTA

Designed for P3P4-only architecture (2 detection scales) that crashes with
Ultralytics' built-in augment=True.
"""

import torch
import numpy as np
import copy
import os
import sys
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression
import torchvision

# ─── Configuration ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = str(PROJECT_ROOT / "VOC" / "voc.yaml")

# Weight paths
WEIGHTS = {
    "exp_002_baseline": str(PROJECT_ROOT / "experiments/exp_002_ghost_hybrid_medium3/weights/best.pt"),
    "exp_005_kd": str(PROJECT_ROOT / "experiments/exp_005_kd_student3/weights/best.pt"),
    "exp_tfa": str(PROJECT_ROOT / "experiments/exp_tfa_20260217_182417/weights/best.pt"),
}

SOUP_OUTPUT = str(PROJECT_ROOT / "experiments/model_soup_best.pt")

# TTA settings
TTA_SCALES = [576, 640, 704]  # Multiple resolutions
TTA_FLIPS = [False, True]     # Original + horizontal flip
TTA_CONF = 0.001              # Low conf for mAP computation
TTA_IOU = 0.6                 # NMS IoU threshold for merging


def model_soup(weight_paths: list, alphas: list = None) -> dict:
    """
    Average model weights from multiple checkpoints (Model Soup).
    
    Args:
        weight_paths: List of .pt file paths
        alphas: Optional weights for each model (must sum to 1.0)
    
    Returns:
        Averaged state_dict
    """
    if alphas is None:
        alphas = [1.0 / len(weight_paths)] * len(weight_paths)
    
    assert len(alphas) == len(weight_paths)
    assert abs(sum(alphas) - 1.0) < 1e-6, f"Alphas must sum to 1.0, got {sum(alphas)}"
    
    # Load all checkpoints
    checkpoints = []
    for p in weight_paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        checkpoints.append(ckpt)
        print(f"  Loaded: {Path(p).parent.parent.name} — keys: {list(ckpt.keys())[:5]}")
    
    # Average the model state_dict
    avg_ckpt = copy.deepcopy(checkpoints[0])
    model_key = "model"
    
    # Get state dict from the model
    sd0 = checkpoints[0][model_key].state_dict() if hasattr(checkpoints[0][model_key], 'state_dict') else checkpoints[0][model_key]
    
    avg_sd = {}
    for key in sd0:
        tensors = []
        for ckpt in checkpoints:
            sd = ckpt[model_key].state_dict() if hasattr(ckpt[model_key], 'state_dict') else ckpt[model_key]
            tensors.append(sd[key].float())
        
        # Weighted average
        avg_sd[key] = sum(a * t for a, t in zip(alphas, tensors))
    
    # Load averaged weights back into the model
    if hasattr(avg_ckpt[model_key], 'load_state_dict'):
        avg_ckpt[model_key].load_state_dict(avg_sd)
    else:
        avg_ckpt[model_key] = avg_sd
    
    return avg_ckpt


def validate_model(model_path: str, imgsz: int = 640, label: str = ""):
    """Validate a model and return metrics."""
    print(f"\n{'='*60}")
    print(f"Validating: {label} @ imgsz={imgsz}")
    print(f"{'='*60}")
    
    model = YOLO(model_path)
    results = model.val(
        data=DATA_YAML,
        imgsz=imgsz,
        batch=8 if imgsz <= 640 else 4,
        device="0",
        workers=4,
        conf=0.001,
        iou=0.7,
        plots=False,
        save_json=False,
        verbose=True,
    )
    
    mAP50 = results.results_dict.get("metrics/mAP50(B)", 0)
    mAP5095 = results.results_dict.get("metrics/mAP50-95(B)", 0)
    
    # Per-class
    names = results.names
    ap50_per_class = results.box.ap50  # shape: (num_classes,)
    p_per_class = results.box.p       # precision per class
    r_per_class = results.box.r       # recall per class
    
    print(f"\n  mAP50={mAP50:.4f}  mAP50-95={mAP5095:.4f}")
    for i, name in names.items():
        print(f"  {name}: mAP50={ap50_per_class[i]:.4f}  P={p_per_class[i]:.3f}  R={r_per_class[i]:.3f}")
    
    return mAP50, mAP5095, ap50_per_class, p_per_class, r_per_class


def custom_tta_validate(model_path: str, data_yaml: str, 
                         scales: list = None, flips: list = None,
                         conf_thres: float = 0.001, iou_thres: float = 0.6,
                         label: str = ""):
    """
    Custom Test-Time Augmentation for P3P4 models.
    
    Runs inference at multiple scales and with horizontal flip,
    then merges predictions using NMS.
    """
    if scales is None:
        scales = TTA_SCALES
    if flips is None:
        flips = TTA_FLIPS
    
    print(f"\n{'='*60}")
    print(f"Custom TTA: {label}")
    print(f"  Scales: {scales}")
    print(f"  Flips: {flips}")
    print(f"  Total forward passes per image: {len(scales) * len(flips)}")
    print(f"{'='*60}")
    
    from ultralytics import YOLO
    from ultralytics.data.build import build_dataloader
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    import yaml
    
    model = YOLO(model_path)
    
    # We'll validate at each scale+flip separately, then do WBF/NMS merge
    # For mAP computation, we need to use the ultralytics validator framework
    # Simplest approach: run val at each scale, track raw predictions, merge
    
    # Actually, the simplest reliable approach for mAP computation:
    # Run prediction on each val image at each scale+flip, merge with NMS,
    # then compute mAP against ground truth.
    
    # But for quick testing, let's use an approximation:
    # Run val at each scale and pick best per-image predictions.
    # Better approach: use the model in predict mode and compute mAP manually.
    
    # Let's use the Ultralytics validator but intercept predictions.
    # Actually the most practical approach: run val at multiple scales and 
    # see if any single scale beats baseline.
    
    print("\n--- Per-scale validation ---")
    best_map50 = 0
    best_scale = 0
    results_by_scale = {}
    
    for scale in scales:
        batch = 8 if scale <= 640 else 4
        results = model.val(
            data=data_yaml,
            imgsz=scale,
            batch=batch,
            device="0",
            workers=4,
            conf=0.001,
            iou=0.7,
            plots=False,
            verbose=False,
        )
        m50 = results.results_dict.get("metrics/mAP50(B)", 0)
        m5095 = results.results_dict.get("metrics/mAP50-95(B)", 0)
        ap50 = results.box.ap50
        names = results.names
        
        print(f"  Scale {scale}: mAP50={m50:.4f}  mAP50-95={m5095:.4f}  "
              f"D1={ap50[1]:.4f}  ins={ap50[0]:.4f}")
        
        results_by_scale[scale] = results
        if m50 > best_map50:
            best_map50 = m50
            best_scale = scale
    
    print(f"\n  Best single scale: {best_scale} with mAP50={best_map50:.4f}")
    
    # Now do actual multi-scale TTA with NMS merging
    print("\n--- Multi-scale TTA with NMS merge ---")
    tta_map50 = run_actual_tta(model, data_yaml, scales, flips, conf_thres, iou_thres)
    
    return best_map50, tta_map50


def run_actual_tta(model, data_yaml: str, scales: list, flips: list,
                   conf_thres: float, iou_thres: float):
    """
    Run actual TTA: predict at multiple scales/flips, merge with NMS, compute mAP.
    """
    import yaml
    from pathlib import Path
    from ultralytics.utils.metrics import DetMetrics, box_iou
    import cv2
    
    # Load dataset info
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)
    
    val_path = data_cfg.get("val", "")
    if not os.path.isabs(val_path):
        val_path = str(Path(data_yaml).parent / val_path)
    
    nc = data_cfg.get("nc", 2)
    names = data_cfg.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    
    # Get validation images
    if os.path.isdir(val_path):
        img_dir = val_path
        img_files = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
    else:
        with open(val_path) as f:
            img_files = [line.strip() for line in f if line.strip()]
    
    print(f"  Val images: {len(img_files)}")
    
    # Collect all predictions and ground truths
    all_preds = []  # List of (img_idx, class, conf, x1, y1, x2, y2) in original coords
    all_gts = []    # List of (img_idx, class, x1, y1, x2, y2)
    
    # Get label directory 
    # Ultralytics expects labels in ../labels/ relative to images
    label_dir = str(Path(val_path).parent / "labels" / Path(val_path).name)
    if not os.path.isdir(label_dir):
        # Try standard ultralytics path
        label_dir = val_path.replace("/images/", "/labels/")
    
    for img_idx, img_path in enumerate(img_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        
        # Collect predictions from all scale+flip combinations
        all_boxes = []  # (x1, y1, x2, y2, conf, cls) in original image coords
        
        for scale in scales:
            for do_flip in flips:
                # Prepare source: flip image if needed
                if do_flip:
                    img_flipped = cv2.flip(img, 1)  # Horizontal flip
                    source = img_flipped
                else:
                    source = img
                
                # Predict
                results = model.predict(
                    source,
                    imgsz=scale,
                    conf=conf_thres,
                    iou=0.7,  # Per-scale NMS
                    device="0",
                    verbose=False,
                    augment=False,  # We do our own augmentation
                )
                
                if results and len(results[0].boxes):
                    boxes = results[0].boxes
                    xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
                    confs = boxes.conf.cpu().numpy()  # (N,)
                    clss = boxes.cls.cpu().numpy()    # (N,)
                    
                    if do_flip:
                        # Mirror x coordinates back to original image space
                        xyxy_orig = xyxy.copy()
                        xyxy_orig[:, 0] = w0 - xyxy[:, 2]  # x1 = w - x2
                        xyxy_orig[:, 2] = w0 - xyxy[:, 0]  # x2 = w - x1
                        xyxy = xyxy_orig
                    
                    for i in range(len(xyxy)):
                        all_boxes.append([
                            xyxy[i, 0], xyxy[i, 1], xyxy[i, 2], xyxy[i, 3],
                            confs[i], clss[i]
                        ])
        
        if all_boxes:
            all_boxes = np.array(all_boxes)
            
            # Apply NMS across all scale+flip predictions
            boxes_tensor = torch.from_numpy(all_boxes[:, :4]).float()
            scores_tensor = torch.from_numpy(all_boxes[:, 4]).float()
            classes_tensor = torch.from_numpy(all_boxes[:, 5]).long()
            
            # Class-aware NMS
            keep_indices = []
            for c in range(nc):
                mask = classes_tensor == c
                if mask.sum() == 0:
                    continue
                c_boxes = boxes_tensor[mask]
                c_scores = scores_tensor[mask]
                c_indices = torch.where(mask)[0]
                
                keep = torchvision.ops.nms(c_boxes, c_scores, iou_thres)
                keep_indices.extend(c_indices[keep].tolist())
            
            for idx in keep_indices:
                x1, y1, x2, y2, conf, cls = all_boxes[idx]
                all_preds.append([img_idx, int(cls), conf, x1, y1, x2, y2])
        
        # Load ground truth labels
        label_file = os.path.join(label_dir, Path(img_path).stem + ".txt")
        if os.path.exists(label_file):
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        # Convert YOLO format to xyxy
                        x1 = (cx - bw/2) * w0
                        y1 = (cy - bh/2) * h0
                        x2 = (cx + bw/2) * w0
                        y2 = (cy + bh/2) * h0
                        all_gts.append([img_idx, cls, x1, y1, x2, y2])
        
        if (img_idx + 1) % 20 == 0:
            print(f"    Processed {img_idx + 1}/{len(img_files)} images...")
    
    print(f"  Total predictions after NMS: {len(all_preds)}")
    print(f"  Total ground truths: {len(all_gts)}")
    
    # Compute mAP
    mAP50, per_class_ap = compute_map(all_preds, all_gts, nc, iou_threshold=0.5)
    
    print(f"\n  TTA mAP50 = {mAP50:.4f}")
    for c in range(nc):
        cname = names.get(c, f"class_{c}")
        print(f"  {cname}: AP50 = {per_class_ap[c]:.4f}")
    
    return mAP50


def compute_map(predictions, ground_truths, nc, iou_threshold=0.5):
    """
    Compute mAP at a given IoU threshold.
    
    predictions: list of [img_idx, cls, conf, x1, y1, x2, y2]
    ground_truths: list of [img_idx, cls, x1, y1, x2, y2]
    """
    per_class_ap = []
    
    for c in range(nc):
        # Get predictions and GTs for this class
        c_preds = [p for p in predictions if p[1] == c]
        c_gts = [g for g in ground_truths if g[1] == c]
        
        n_gt = len(c_gts)
        if n_gt == 0:
            per_class_ap.append(0.0)
            continue
        
        # Sort predictions by confidence (descending)
        c_preds.sort(key=lambda x: x[2], reverse=True)
        
        # Track which GTs have been matched
        gt_matched = {}  # (img_idx, gt_idx) -> bool
        for i, g in enumerate(c_gts):
            gt_matched[(g[0], i)] = False
        
        # Build GT lookup by image
        gt_by_img = {}
        for i, g in enumerate(c_gts):
            img_idx = g[0]
            if img_idx not in gt_by_img:
                gt_by_img[img_idx] = []
            gt_by_img[img_idx].append((i, g))
        
        tp = np.zeros(len(c_preds))
        fp = np.zeros(len(c_preds))
        
        for pred_idx, pred in enumerate(c_preds):
            img_idx = pred[0]
            pred_box = pred[3:7]
            
            if img_idx not in gt_by_img:
                fp[pred_idx] = 1
                continue
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_local_idx, gt in gt_by_img[img_idx]:
                gt_box = gt[2:6]
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_local_idx
            
            if best_iou >= iou_threshold and not gt_matched[(img_idx, best_gt_idx)]:
                tp[pred_idx] = 1
                gt_matched[(img_idx, best_gt_idx)] = True
            else:
                fp[pred_idx] = 1
        
        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / n_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # AP using all-point interpolation (PASCAL VOC style)
        ap = compute_ap_interpolated(recall, precision)
        per_class_ap.append(ap)
    
    mAP = np.mean(per_class_ap) if per_class_ap else 0.0
    return mAP, per_class_ap


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def compute_ap_interpolated(recall, precision):
    """Compute AP using 101-point interpolation (COCO style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    # 101-point interpolation
    recall_interp = np.linspace(0, 1, 101)
    precision_interp = np.zeros_like(recall_interp)
    for i, r in enumerate(recall_interp):
        # Find precision at recall >= r
        mask = mrec >= r
        if mask.any():
            precision_interp[i] = mpre[mask].max()
    
    ap = precision_interp.mean()
    return ap


def main():
    print("=" * 70)
    print("MODEL SOUP + CUSTOM TTA PIPELINE")
    print("=" * 70)
    
    # ─── Step 1: Validate Baseline ────────────────────────────────────
    print("\n\n🔹 STEP 1: Baseline Validation")
    baseline_map, _, _, _, _ = validate_model(
        WEIGHTS["exp_002_baseline"], imgsz=640, label="exp_002 Baseline @ 640"
    )
    
    # ─── Step 2: Validate KD Model ────────────────────────────────────
    print("\n\n🔹 STEP 2: KD Model Validation")
    kd_map, _, _, _, _ = validate_model(
        WEIGHTS["exp_005_kd"], imgsz=640, label="exp_005 KD @ 640"
    )
    
    # ─── Step 3: Model Soup (Equal Average) ───────────────────────────
    print("\n\n🔹 STEP 3: Model Soup (exp_002 + exp_005_kd)")
    print("  Averaging weights with alpha=[0.5, 0.5]...")
    
    soup_ckpt = model_soup(
        [WEIGHTS["exp_002_baseline"], WEIGHTS["exp_005_kd"]],
        alphas=[0.5, 0.5]
    )
    torch.save(soup_ckpt, SOUP_OUTPUT)
    print(f"  Saved soup model to: {SOUP_OUTPUT}")
    
    soup_map, _, _, _, _ = validate_model(
        SOUP_OUTPUT, imgsz=640, label="Model Soup (0.5/0.5) @ 640"
    )
    
    # ─── Step 3b: Soup with different alpha ───────────────────────────
    # If KD model is slightly better, try weighting it more
    if kd_map > baseline_map:
        print("\n\n🔹 STEP 3b: Model Soup (exp_002:0.4 + exp_005_kd:0.6)")
        soup_ckpt2 = model_soup(
            [WEIGHTS["exp_002_baseline"], WEIGHTS["exp_005_kd"]],
            alphas=[0.4, 0.6]
        )
        soup_output2 = str(PROJECT_ROOT / "experiments/model_soup_kd_heavy.pt")
        torch.save(soup_ckpt2, soup_output2)
        
        soup_map2, _, _, _, _ = validate_model(
            soup_output2, imgsz=640, label="Model Soup (0.4/0.6) @ 640"
        )
    
    # ─── Step 3c: 3-way soup including TFA ────────────────────────────
    if os.path.exists(WEIGHTS["exp_tfa"]):
        print("\n\n🔹 STEP 3c: 3-Way Model Soup (exp_002 + exp_005_kd + exp_tfa)")
        soup_ckpt3 = model_soup(
            [WEIGHTS["exp_002_baseline"], WEIGHTS["exp_005_kd"], WEIGHTS["exp_tfa"]],
            alphas=[0.4, 0.4, 0.2]
        )
        soup_output3 = str(PROJECT_ROOT / "experiments/model_soup_3way.pt")
        torch.save(soup_ckpt3, soup_output3)
        
        soup_map3, _, _, _, _ = validate_model(
            soup_output3, imgsz=640, label="3-Way Soup (0.4/0.4/0.2) @ 640"
        )
    
    # ─── Step 4: Custom TTA on best model ─────────────────────────────
    # Use whichever model scored highest so far
    best_weights = WEIGHTS["exp_002_baseline"]
    best_label = "baseline"
    best_score = baseline_map
    
    if kd_map > best_score:
        best_weights = WEIGHTS["exp_005_kd"]
        best_label = "kd"
        best_score = kd_map
    if soup_map > best_score:
        best_weights = SOUP_OUTPUT
        best_label = "soup"
        best_score = soup_map
    
    print(f"\n\n🔹 STEP 4: Custom TTA on best model ({best_label}, mAP50={best_score:.4f})")
    best_single, tta_map = custom_tta_validate(
        best_weights, DATA_YAML,
        scales=[576, 640, 704],
        flips=[False, True],
        conf_thres=0.001,
        iou_thres=0.6,
        label=f"TTA on {best_label}"
    )
    
    # ─── Final Summary ────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Baseline (exp_002 @ 640):    mAP50 = {baseline_map:.4f}")
    print(f"  KD Model (exp_005 @ 640):    mAP50 = {kd_map:.4f}")
    print(f"  Model Soup (0.5/0.5 @ 640):  mAP50 = {soup_map:.4f}")
    print(f"  Best single model:           {best_label} = {best_score:.4f}")
    print(f"  Custom TTA (best single):    mAP50 = {best_single:.4f}")
    print(f"  Custom TTA (merged):         mAP50 = {tta_map:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
