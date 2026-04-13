"""
Multi-Model Ensemble Validation
================================
Runs multiple models on val set, merges their predictions via WBF (Weighted Box Fusion)
or NMS, then evaluates using Ultralytics' own validation pipeline.

Strategy: Hook into the validator to replace single-model predictions with ensemble predictions.
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def weighted_box_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0):
    """
    Simplified Weighted Box Fusion implementation.
    
    Args:
        boxes_list: list of np arrays of shape (N, 4) in xyxy format, normalized [0,1]
        scores_list: list of np arrays of shape (N,)
        labels_list: list of np arrays of shape (N,) 
        weights: list of model weights
        iou_thr: IoU threshold for merging
        skip_box_thr: minimum score threshold
    
    Returns:
        boxes, scores, labels (merged)
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)
    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()
    
    # Collect all boxes with model index
    all_boxes = []
    for model_idx, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        for i in range(len(boxes)):
            if scores[i] > skip_box_thr:
                all_boxes.append({
                    'box': boxes[i],
                    'score': scores[i] * weights[model_idx],
                    'label': int(labels[i]),
                    'model_idx': model_idx,
                })
    
    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    
    # Group by label
    unique_labels = set(b['label'] for b in all_boxes)
    final_boxes, final_scores, final_labels = [], [], []
    
    for label in unique_labels:
        label_boxes = [b for b in all_boxes if b['label'] == label]
        # Sort by score descending
        label_boxes.sort(key=lambda x: x['score'], reverse=True)
        
        clusters = []
        used = [False] * len(label_boxes)
        
        for i in range(len(label_boxes)):
            if used[i]:
                continue
            cluster = [label_boxes[i]]
            used[i] = True
            
            for j in range(i + 1, len(label_boxes)):
                if used[j]:
                    continue
                iou = compute_iou(label_boxes[i]['box'], label_boxes[j]['box'])
                if iou > iou_thr:
                    cluster.append(label_boxes[j])
                    used[j] = True
            
            # Fuse cluster: weighted average of boxes, max score
            if len(cluster) == 1:
                fused_box = cluster[0]['box']
                fused_score = cluster[0]['score']
            else:
                total_w = sum(c['score'] for c in cluster)
                fused_box = np.zeros(4)
                for c in cluster:
                    fused_box += c['box'] * c['score']
                fused_box /= total_w
                # Score boost: average * (N_models / total_models) factor
                fused_score = sum(c['score'] for c in cluster) / len(weights)
                # Boost if multiple models agree
                n_models = len(set(c['model_idx'] for c in cluster))
                fused_score *= (1.0 + 0.1 * (n_models - 1))
            
            final_boxes.append(fused_box)
            final_scores.append(fused_score)
            final_labels.append(label)
    
    if not final_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    
    return np.array(final_boxes), np.array(final_scores), np.array(final_labels, dtype=int)


def compute_iou(box1, box2):
    """Compute IoU between two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def soft_nms_merge(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.5):
    """Simple NMS-based merge: concat all, then NMS per class."""
    if weights is None:
        weights = [1.0] * len(boxes_list)
    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()
    
    all_boxes, all_scores, all_labels = [], [], []
    for i, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        all_boxes.append(boxes)
        all_scores.append(scores * weights[i])
        all_labels.append(labels)
    
    if not all_boxes or all(len(b) == 0 for b in all_boxes):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    
    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # NMS per class
    final_boxes, final_scores, final_labels = [], [], []
    for cls in np.unique(all_labels):
        mask = all_labels == cls
        cls_boxes = torch.tensor(all_boxes[mask], dtype=torch.float32)
        cls_scores = torch.tensor(all_scores[mask], dtype=torch.float32)
        
        if len(cls_boxes) == 0:
            continue
            
        keep = torch.ops.torchvision.nms(cls_boxes, cls_scores, iou_thr)
        final_boxes.append(cls_boxes[keep].numpy())
        final_scores.append(cls_scores[keep].numpy())
        final_labels.append(np.full(len(keep), cls, dtype=int))
    
    if not final_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    
    return np.concatenate(final_boxes), np.concatenate(final_scores), np.concatenate(final_labels)


def run_model_predict(model_path, imgsz, augment=False, conf=0.001, iou=0.7):
    """Run a model on val set and collect all predictions."""
    model = YOLO(model_path)
    model.model.stride = torch.tensor([8., 16., 32.])
    
    # Use val() with save_json to get structured predictions
    # Instead, use predict on each image
    import yaml
    with open("VOC/voc.yaml") as f:
        data_cfg = yaml.safe_load(f)
    
    val_dir = Path(data_cfg.get("path", "VOC")) / "images" / "val"
    img_files = sorted([f for f in val_dir.iterdir() if f.suffix in ('.jpg', '.jpeg', '.png')])
    
    all_predictions = {}  # filename -> (boxes_xyxy, scores, classes)
    
    results = model.predict(
        source=str(val_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device="0",
        augment=augment,
        verbose=False,
        batch=4,
    )
    
    for r in results:
        fname = Path(r.path).name
        if len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
        else:
            boxes = np.zeros((0, 4))
            scores = np.zeros(0)
            classes = np.zeros(0, dtype=int)
        all_predictions[fname] = (boxes, scores, classes)
    
    return all_predictions


def ensemble_and_save(models_config, fusion_method="wbf", iou_thr=0.55):
    """
    Run ensemble: collect predictions from all models, fuse, save as temp labels.
    Then run Ultralytics val() using those saved predictions.
    """
    print(f"\n{'='*60}")
    print(f"Ensemble: {fusion_method.upper()}, IoU threshold: {iou_thr}")
    print(f"{'='*60}")
    
    # Collect predictions from each model
    all_model_preds = []
    model_weights = []
    
    for cfg in models_config:
        print(f"\nRunning: {cfg['name']} @ imgsz={cfg['imgsz']}, augment={cfg.get('augment', False)}")
        preds = run_model_predict(
            cfg['path'], 
            cfg['imgsz'], 
            augment=cfg.get('augment', False),
            conf=0.001,
            iou=0.7,
        )
        all_model_preds.append(preds)
        model_weights.append(cfg.get('weight', 1.0))
        print(f"  Got predictions for {len(preds)} images")
    
    # Get all filenames
    all_fnames = set()
    for preds in all_model_preds:
        all_fnames.update(preds.keys())
    all_fnames = sorted(all_fnames)
    
    print(f"\nFusing predictions for {len(all_fnames)} images...")
    
    # Fuse per image
    fused_predictions = {}
    for fname in all_fnames:
        boxes_list, scores_list, labels_list = [], [], []
        
        for preds in all_model_preds:
            if fname in preds:
                boxes, scores, labels = preds[fname]
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)
            else:
                boxes_list.append(np.zeros((0, 4)))
                scores_list.append(np.zeros(0))
                labels_list.append(np.zeros(0, dtype=int))
        
        if fusion_method == "wbf":
            fused_b, fused_s, fused_l = weighted_box_fusion(
                boxes_list, scores_list, labels_list, 
                weights=model_weights, iou_thr=iou_thr
            )
        else:
            fused_b, fused_s, fused_l = soft_nms_merge(
                boxes_list, scores_list, labels_list,
                weights=model_weights, iou_thr=iou_thr
            )
        
        fused_predictions[fname] = (fused_b, fused_s, fused_l)
    
    return fused_predictions


def evaluate_with_ultralytics(fused_predictions):
    """
    Evaluate fused predictions using Ultralytics' metric pipeline.
    Uses the same matching and AP computation as model.val().
    """
    import yaml
    from ultralytics.utils.metrics import ap_per_class, box_iou
    
    with open("VOC/voc.yaml") as f:
        data_cfg = yaml.safe_load(f)
    
    data_path = Path(data_cfg.get("path", "VOC"))
    label_dir = data_path / "labels" / "val"
    img_dir = data_path / "images" / "val"
    
    # Collect stats: for each image, match predictions to GT
    iouv = torch.linspace(0.5, 0.95, 10)  # IoU thresholds from 0.5 to 0.95
    stats = []
    
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix in ('.jpg', '.jpeg', '.png')])
    
    for img_file in img_files:
        fname = img_file.name
        label_file = label_dir / (img_file.stem + ".txt")
        
        # Load GT
        import cv2
        img = cv2.imread(str(img_file))
        h, w = img.shape[:2]
        
        gt_boxes = []
        gt_classes = []
        if label_file.exists():
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
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_classes.append(cls)
        
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4))
        gt_classes = torch.tensor(gt_classes, dtype=torch.long) if gt_classes else torch.zeros(0, dtype=torch.long)
        
        # Get predictions
        if fname in fused_predictions:
            pred_boxes, pred_scores, pred_labels = fused_predictions[fname]
            if len(pred_boxes) > 0:
                pred_boxes_t = torch.tensor(pred_boxes, dtype=torch.float32)
                pred_scores_t = torch.tensor(pred_scores, dtype=torch.float32)
                pred_labels_t = torch.tensor(pred_labels, dtype=torch.long)
            else:
                pred_boxes_t = torch.zeros((0, 4))
                pred_scores_t = torch.zeros(0)
                pred_labels_t = torch.zeros(0, dtype=torch.long)
        else:
            pred_boxes_t = torch.zeros((0, 4))
            pred_scores_t = torch.zeros(0)
            pred_labels_t = torch.zeros(0, dtype=torch.long)
        
        # Match predictions to GT (same as Ultralytics _process_batch)
        nl = len(gt_classes)
        npred = len(pred_labels_t)
        correct = torch.zeros(npred, len(iouv), dtype=torch.bool)
        
        if npred > 0 and nl > 0:
            iou = box_iou(gt_boxes, pred_boxes_t)  # (nl, npred)
            correct_class = gt_classes[:, None] == pred_labels_t[None, :]  # (nl, npred)
            
            for i, iou_thresh in enumerate(iouv):
                # Find matches
                matches = (iou >= iou_thresh) & correct_class
                if matches.any():
                    # Get best matches
                    matches_np = matches.nonzero(as_tuple=False)
                    if matches_np.shape[0]:
                        iou_vals = iou[matches_np[:, 0], matches_np[:, 1]]
                        # Sort by IoU descending
                        order = iou_vals.argsort(descending=True)
                        matches_np = matches_np[order]
                        
                        # Unique predictions (each pred matched once)
                        _, pred_unique = torch.unique(matches_np[:, 1], return_inverse=True)
                        _, first_idx = torch.unique(pred_unique, return_inverse=True)
                        # Actually: keep first occurrence of each unique pred
                        seen_pred = set()
                        seen_gt = set()
                        for m in matches_np:
                            gt_idx, pred_idx = m[0].item(), m[1].item()
                            if pred_idx not in seen_pred and gt_idx not in seen_gt:
                                correct[pred_idx, i] = True
                                seen_pred.add(pred_idx)
                                seen_gt.add(gt_idx)
        
        stats.append((correct, pred_scores_t, pred_labels_t, gt_classes))
    
    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() if isinstance(x[0], torch.Tensor) else x for x in zip(*stats)]
    tp, conf, pred_cls, target_cls = stats
    
    results = ap_per_class(tp, conf, pred_cls, target_cls, names={0: "Damaged_1", 1: "insulator"})
    
    # ap_per_class returns: tp, fp, p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, px, prec_values
    tp_out, fp_out, p, r, f1, ap, unique_classes = results[:7]
    
    # ap shape: (nc, 10) - per class, per IoU threshold
    ap50 = ap[:, 0]  # AP at IoU=0.5
    map50 = ap50.mean()
    
    names = {0: "Damaged_1", 1: "insulator"}
    print(f"\n{'Class':<15} {'AP50':>8} {'P':>8} {'R':>8}")
    print("-" * 45)
    for i, cls_id in enumerate(unique_classes):
        cls_name = names.get(int(cls_id), f"cls{int(cls_id)}")
        print(f"{cls_name:<15} {ap50[i]*100:>7.2f}% {p[i]*100:>7.2f}% {r[i]*100:>7.2f}%")
    print("-" * 45)
    print(f"{'mAP50':<15} {map50*100:>7.2f}%")
    
    return map50, ap50, p, r


if __name__ == "__main__":
    # Define model configs for ensemble
    configs_to_test = [
        # Config 1: KD + Baseline ensemble with TTA
        {
            "name": "KD+Baseline TTA@768",
            "models": [
                {"name": "KD", "path": "experiments/exp_005_kd_student3/weights/best.pt", 
                 "imgsz": 768, "augment": True, "weight": 1.2},
                {"name": "Baseline", "path": "experiments/exp_002_ghost_hybrid_medium3/weights/best.pt",
                 "imgsz": 768, "augment": True, "weight": 1.0},
            ],
            "fusion": "wbf",
        },
        # Config 2: KD + Baseline + Soup3way
        {
            "name": "KD+Baseline+Soup TTA@768",
            "models": [
                {"name": "KD", "path": "experiments/exp_005_kd_student3/weights/best.pt",
                 "imgsz": 768, "augment": True, "weight": 1.2},
                {"name": "Baseline", "path": "experiments/exp_002_ghost_hybrid_medium3/weights/best.pt",
                 "imgsz": 768, "augment": True, "weight": 1.0},
                {"name": "Soup3way", "path": "experiments/soup_3way.pt",
                 "imgsz": 768, "augment": True, "weight": 0.8},
            ],
            "fusion": "wbf",
        },
        # Config 3: KD multi-scale (no TTA, just different resolutions)
        {
            "name": "KD multi-res",
            "models": [
                {"name": "KD@640", "path": "experiments/exp_005_kd_student3/weights/best.pt",
                 "imgsz": 640, "augment": False, "weight": 1.0},
                {"name": "KD@768", "path": "experiments/exp_005_kd_student3/weights/best.pt",
                 "imgsz": 768, "augment": False, "weight": 1.2},
                {"name": "KD@832", "path": "experiments/exp_005_kd_student3/weights/best.pt",
                 "imgsz": 832, "augment": False, "weight": 0.8},
            ],
            "fusion": "wbf",
        },
        # Config 4: KD + Baseline NMS ensemble  
        {
            "name": "KD+Baseline NMS@768",
            "models": [
                {"name": "KD", "path": "experiments/exp_005_kd_student3/weights/best.pt",
                 "imgsz": 768, "augment": True, "weight": 1.2},
                {"name": "Baseline", "path": "experiments/exp_002_ghost_hybrid_medium3/weights/best.pt",
                 "imgsz": 768, "augment": True, "weight": 1.0},
            ],
            "fusion": "nms",
        },
    ]
    
    print("=" * 60)
    print("  MULTI-MODEL ENSEMBLE EVALUATION")
    print("=" * 60)
    
    # But first, the custom metrics may not match ultralytics.
    # Let's calibrate: run single model and compare
    print("\n--- CALIBRATION: Single model through ensemble pipeline ---")
    single_config = [
        {"name": "KD_only", "path": "experiments/exp_005_kd_student3/weights/best.pt",
         "imgsz": 640, "augment": False, "weight": 1.0}
    ]
    fused = ensemble_and_save(single_config, fusion_method="nms")
    cal_map50, _, _, _ = evaluate_with_ultralytics(fused)
    print(f"\nCalibration: ensemble pipeline mAP50 = {cal_map50*100:.2f}%")
    print(f"Expected from model.val(): 94.12%")
    
    if abs(cal_map50 * 100 - 94.12) > 2.0:
        print("\n⚠ WARNING: Calibration gap > 2%. Results may not be directly comparable.")
        print("But relative comparisons between ensemble configs should still be valid.")
    
    # Run all configs
    for cfg in configs_to_test:
        print(f"\n\n{'#'*60}")
        print(f"# {cfg['name']}")
        print(f"{'#'*60}")
        fused = ensemble_and_save(cfg['models'], fusion_method=cfg['fusion'])
        map50, ap50, p, r = evaluate_with_ultralytics(fused)
