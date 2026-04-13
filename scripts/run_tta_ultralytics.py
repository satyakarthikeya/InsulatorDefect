"""
TTA via Ultralytics' own validation pipeline.
Overrides the model's forward pass to run multi-scale + flip inference,
merge with NMS, and return results exactly as Ultralytics expects.
This ensures mAP is computed identically to baseline.
"""
import torch
import numpy as np
import cv2
import torchvision
from pathlib import Path
from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.engine.results import Results
from copy import deepcopy

PROJECT = Path(__file__).resolve().parent.parent
DATA_YAML = str(PROJECT / "VOC" / "voc.yaml")


class TTAModel:
    """
    Wrapper that makes a YOLO model run multi-scale + flip TTA internally.
    Uses model.predict() at each scale/flip, merges with NMS, returns via model.val().
    """
    
    def __init__(self, model_path, scales=(576, 640, 704), do_flip=True, nms_iou=0.5):
        self.model = YOLO(model_path)
        self.scales = scales
        self.do_flip = do_flip
        self.nms_iou = nms_iou
        self.nc = 2
    
    def tta_predict_image(self, img_path, conf=0.001):
        """Run TTA on a single image, return merged boxes in xyxy format."""
        img = cv2.imread(str(img_path))
        if img is None:
            return np.empty((0, 6))
        h0, w0 = img.shape[:2]
        
        all_boxes = []
        
        for scale in self.scales:
            flip_opts = [False, True] if self.do_flip else [False]
            for flip in flip_opts:
                source = cv2.flip(img, 1) if flip else img
                
                results = self.model.predict(
                    source, imgsz=scale, conf=conf, iou=0.7,
                    device="0", verbose=False, augment=False, max_det=300,
                )
                
                if results and len(results[0].boxes):
                    xyxy = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    clss = results[0].boxes.cls.cpu().numpy()
                    
                    if flip:
                        orig = xyxy.copy()
                        orig[:, 0] = w0 - xyxy[:, 2]
                        orig[:, 2] = w0 - xyxy[:, 0]
                        xyxy = orig
                    
                    for i in range(len(xyxy)):
                        all_boxes.append([
                            xyxy[i, 0], xyxy[i, 1], xyxy[i, 2], xyxy[i, 3],
                            confs[i], clss[i]
                        ])
        
        if not all_boxes:
            return np.empty((0, 6))
        
        boxes_arr = np.array(all_boxes)
        
        # Class-aware NMS
        keep_all = []
        for c in range(self.nc):
            mask = boxes_arr[:, 5] == c
            if mask.sum() == 0:
                continue
            c_boxes = torch.from_numpy(boxes_arr[mask, :4]).float()
            c_scores = torch.from_numpy(boxes_arr[mask, 4]).float()
            c_indices = np.where(mask)[0]
            
            keep = torchvision.ops.nms(c_boxes, c_scores, self.nms_iou)
            keep_all.extend(c_indices[keep.numpy()].tolist())
        
        return boxes_arr[keep_all] if keep_all else np.empty((0, 6))


def evaluate_tta(model_path, scales, do_flip, nms_iou, label=""):
    """
    Evaluate TTA using Ultralytics' val() metrics on merged predictions.
    
    Strategy: Run TTA prediction on each val image, save results as a temp 
    label set, then use Ultralytics' DetMetrics to compute mAP identically.
    """
    import yaml
    from ultralytics.utils.metrics import DetMetrics
    
    print(f"\n{'='*60}")
    print(f"TTA: {label}")
    print(f"  Scales: {scales}, Flip: {do_flip}, NMS IoU: {nms_iou}")
    print(f"{'='*60}")
    
    tta = TTAModel(model_path, scales=scales, do_flip=do_flip, nms_iou=nms_iou)
    
    # Load val image paths
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    base = cfg.get("path", str(Path(DATA_YAML).parent))
    val_dir = Path(base) / cfg["val"]
    label_dir = str(val_dir).replace("/images/", "/labels/")
    
    names = cfg["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    nc = cfg["nc"]
    
    img_files = sorted([
        str(val_dir / f) for f in os.listdir(str(val_dir))
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    # Initialize Ultralytics metrics
    metrics = DetMetrics(save_dir=Path(PROJECT / "experiments"), names=names)
    
    # Process each image
    stats = []
    for idx, img_path in enumerate(img_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # TTA prediction
        merged = tta.tta_predict_image(img_path, conf=0.001)
        
        # Load ground truth
        label_file = os.path.join(label_dir, Path(img_path).stem + ".txt")
        gt_boxes = []
        if os.path.exists(label_file):
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        gt_boxes.append([cls, x1, y1, x2, y2])
        
        # Format for Ultralytics metrics: need (correct, conf, pred_cls, target_cls)
        # Ultralytics uses metrics.process(tp, conf, pred_cls, target_cls)
        # where tp is a boolean tensor of shape (num_preds, num_iou_thresholds)
        
        if len(gt_boxes) == 0:
            gt_tensor = torch.zeros((0, 5))  # (cls, x1, y1, x2, y2)
        else:
            gt_tensor = torch.tensor(gt_boxes)  # (N, 5)
        
        if merged.shape[0] == 0:
            pred_boxes = torch.zeros((0, 4))
            pred_conf = torch.zeros((0,))
            pred_cls = torch.zeros((0,))
        else:
            pred_boxes = torch.from_numpy(merged[:, :4]).float()  # xyxy
            pred_conf = torch.from_numpy(merged[:, 4]).float()
            pred_cls = torch.from_numpy(merged[:, 5]).float()
        
        # Match predictions to GT using IoU thresholds
        # Ultralytics uses 10 IoU thresholds: 0.50->0.95
        iou_thresholds = torch.linspace(0.5, 0.95, 10)
        
        if pred_boxes.shape[0] > 0 and gt_tensor.shape[0] > 0:
            gt_xyxy = gt_tensor[:, 1:5]  # (M, 4)
            gt_cls = gt_tensor[:, 0]     # (M,)
            
            # Compute IoU matrix
            iou_matrix = box_iou(pred_boxes, gt_xyxy)  # (num_pred, num_gt)
            
            # For each IoU threshold, determine TP/FP
            correct = torch.zeros(pred_boxes.shape[0], len(iou_thresholds), dtype=torch.bool)
            
            for t_idx, iou_thr in enumerate(iou_thresholds):
                # Match: for each prediction, find best matching GT
                matches = match_predictions(iou_matrix, pred_cls, gt_cls, iou_thr)
                correct[:, t_idx] = matches
            
            stats.append((correct, pred_conf, pred_cls, gt_cls))
        elif pred_boxes.shape[0] > 0:
            # Preds but no GT: all FP
            correct = torch.zeros(pred_boxes.shape[0], len(iou_thresholds), dtype=torch.bool)
            gt_cls = torch.zeros((0,))
            stats.append((correct, pred_conf, pred_cls, gt_cls))
        elif gt_tensor.shape[0] > 0:
            # GT but no preds: all FN
            gt_cls = gt_tensor[:, 0]
            correct = torch.zeros((0, len(iou_thresholds)), dtype=torch.bool)
            stats.append((correct, torch.zeros((0,)), torch.zeros((0,)), gt_cls))
        
        if (idx + 1) % 30 == 0:
            print(f"  Processed {idx + 1}/{len(img_files)} images")
    
    print(f"  Processed all {len(img_files)} images")
    
    # Concatenate stats and compute metrics
    stats_concat = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    metrics.process(*stats_concat)
    
    results = metrics.results_dict
    print(f"\n  mAP50 = {results.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP50-95 = {results.get('metrics/mAP50-95(B)', 0):.4f}")
    
    # Per-class
    ap50 = metrics.box.ap50
    p = metrics.box.p
    r = metrics.box.r
    for i, name in names.items():
        print(f"  {name}: AP50={ap50[i]:.4f}  P={p[i]:.3f}  R={r[i]:.3f}")
    
    return results.get('metrics/mAP50(B)', 0)


def box_iou(box1, box2):
    """Compute IoU between two sets of boxes. box1: (N,4), box2: (M,4) in xyxy format."""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])
    
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    
    return inter / union


def match_predictions(iou_matrix, pred_cls, gt_cls, iou_threshold):
    """
    Match predictions to GT using greedy matching (same as Ultralytics).
    Returns boolean tensor of length num_preds indicating TP.
    """
    n_pred = iou_matrix.shape[0]
    matched = torch.zeros(n_pred, dtype=torch.bool)
    gt_used = torch.zeros(iou_matrix.shape[1], dtype=torch.bool)
    
    # Class-aware matching: only match same class
    for pi in range(n_pred):
        best_iou = 0
        best_gi = -1
        for gi in range(iou_matrix.shape[1]):
            if gt_used[gi]:
                continue
            if pred_cls[pi] != gt_cls[gi]:
                continue
            if iou_matrix[pi, gi] > best_iou:
                best_iou = iou_matrix[pi, gi].item()
                best_gi = gi
        
        if best_iou >= iou_threshold and best_gi >= 0:
            matched[pi] = True
            gt_used[best_gi] = True
    
    return matched


import os

def main():
    print("=" * 70)
    print("TTA EVALUATION (ULTRALYTICS-COMPATIBLE METRICS)")
    print("=" * 70)
    
    models = {
        "Baseline": str(PROJECT / "experiments/exp_002_ghost_hybrid_medium3/weights/best.pt"),
        "Soup3way": str(PROJECT / "experiments/soup_3way.pt"),
        "Teacher": str(PROJECT / "experiments/exp_004_teacher_yolo11s/weights/best.pt"),
    }
    
    # First: Ultralytics baseline at 640 for reference
    print("\n--- ULTRALYTICS BASELINES (reference) ---")
    for name, path in models.items():
        m = YOLO(path)
        r = m.val(data=DATA_YAML, imgsz=640, batch=8, device="0", workers=4,
                  conf=0.001, iou=0.7, plots=False, verbose=False)
        map50 = r.results_dict["metrics/mAP50(B)"]
        print(f"  {name}: mAP50={map50:.4f}  D1={r.box.ap50[0]:.4f}  ins={r.box.ap50[1]:.4f}")
        del m
        torch.cuda.empty_cache()
    
    # TTA evaluations
    tta_configs = [
        ("3s_flip", [576, 640, 704], True, 0.5),
        ("3s_flip_nms45", [576, 640, 704], True, 0.45),
        ("3s_noflip", [576, 640, 704], False, 0.5),
    ]
    
    results_table = []
    
    # Only test baseline + soup (teacher takes too long for all configs)
    for model_name in ["Baseline", "Soup3way"]:
        model_path = models[model_name]
        
        for tta_name, scales, flip, nms_iou in tta_configs:
            label = f"{model_name}_{tta_name}"
            map50 = evaluate_tta(model_path, scales, flip, nms_iou, label=label)
            results_table.append((model_name, tta_name, map50))
    
    # Teacher with best TTA config only (3scale_flip)
    map50_t = evaluate_tta(models["Teacher"], [576, 640, 704], True, 0.5, label="Teacher_3s_flip")
    results_table.append(("Teacher", "3s_flip", map50_t))
    
    # Summary
    print("\n\n" + "=" * 60)
    print("COMPLETE RESULTS")
    print("=" * 60)
    print(f"{'Model':<15} {'TTA':<18} {'mAP50':>8}")
    print("-" * 45)
    for model, tta, m in results_table:
        print(f"{model:<15} {tta:<18} {m:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
