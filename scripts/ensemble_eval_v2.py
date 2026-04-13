"""
Multi-model WBF Ensemble Evaluation for YOLO models.
Collects raw predictions from multiple models, fuses with WBF, and evaluates with pycocotools.
"""
import sys, os, json, torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

# ─── CONFIG ───
DATA_YAML = "VOC/voc.yaml"
DEVICE = 0
NUM_CLASSES = 2
CLASS_NAMES = ["Damaged_1", "insulator"]
IOU_THR_NMS = 0.7       # NMS IoU during per-model inference
CONF_THR = 0.001        # confidence threshold for collecting predictions
WBF_IOU_THR = 0.55      # WBF fusion IoU threshold
WBF_SKIP_BOX_THR = 0.001  # min score for WBF

# Models to ensemble with their configs
MODELS = [
    {
        "name": "exp_009_768",
        "path": "experiments/exp_009_finetune_768/weights/best.pt",
        "imgsz": 768,
        "tta": False,
        "weight": 1.0,
    },
    {
        "name": "exp_012_768",
        "path": "experiments/exp_012_head_finetune_768/weights/best.pt",
        "imgsz": 768,
        "tta": False,
        "weight": 1.0,
    },
    {
        "name": "exp_009_tta768",
        "path": "experiments/exp_009_finetune_768/weights/best.pt",
        "imgsz": 768,
        "tta": True,
        "weight": 1.0,
    },
]


def load_val_dataset(data_yaml):
    """Load validation image paths and ground truth labels."""
    import yaml
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    
    val_path = Path(data.get("path", ".")) / data["val"]
    # Read image list
    if val_path.suffix == ".txt":
        with open(val_path) as f:
            img_files = [l.strip() for l in f if l.strip()]
    else:
        # val_path is a directory
        img_files = sorted([str(p) for p in val_path.glob("*") if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
    
    # Load ground truth labels
    gt = {}
    for img_path in img_files:
        img_path = Path(img_path)
        # Derive label path: images/ -> labels/
        lbl_path = Path(str(img_path).replace("/images/", "/labels/").replace("/JPEGImages/", "/labels/"))
        lbl_path = lbl_path.with_suffix(".txt")
        
        boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        # Convert YOLO format (cx, cy, w, h) to (x1, y1, x2, y2) normalized
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = cx + w/2
                        y2 = cy + h/2
                        boxes.append((cls, x1, y1, x2, y2))
        gt[str(img_path)] = boxes
    
    return img_files, gt


def collect_predictions(model_cfg, img_files):
    """Run inference on all images, return per-image predictions."""
    print(f"\n  Running inference: {model_cfg['name']} (imgsz={model_cfg['imgsz']}, tta={model_cfg['tta']})")
    model = YOLO(model_cfg["path"])
    
    if model_cfg["tta"]:
        model.model.stride = torch.tensor([8., 16., 32.])
    
    preds = {}
    for img_path in img_files:
        results = model.predict(
            img_path,
            imgsz=model_cfg["imgsz"],
            conf=CONF_THR,
            iou=IOU_THR_NMS,
            device=DEVICE,
            augment=model_cfg["tta"],
            verbose=False,
        )
        r = results[0]
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()   # absolute coords
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        
        # Get image dimensions for normalization
        h, w = r.orig_shape
        
        # Normalize boxes to [0,1]
        boxes_norm = boxes_xyxy.copy()
        boxes_norm[:, [0, 2]] /= w
        boxes_norm[:, [1, 3]] /= h
        boxes_norm = np.clip(boxes_norm, 0, 1)
        
        preds[img_path] = {
            "boxes": boxes_norm,
            "scores": scores,
            "classes": classes,
            "orig_shape": (h, w),
        }
    
    del model
    torch.cuda.empty_cache()
    return preds


def wbf_fuse(all_preds, img_files, model_weights, iou_thr=0.55, skip_box_thr=0.001):
    """Fuse predictions from multiple models using Weighted Boxes Fusion."""
    fused = {}
    
    for img_path in img_files:
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []
        
        for i, preds in enumerate(all_preds):
            p = preds[img_path]
            if len(p["boxes"]) > 0:
                boxes_list.append(p["boxes"].tolist())
                scores_list.append(p["scores"].tolist())
                labels_list.append(p["classes"].tolist())
            else:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
            weights.append(model_weights[i])
        
        if any(len(b) > 0 for b in boxes_list):
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
        else:
            fused_boxes = np.array([]).reshape(0, 4)
            fused_scores = np.array([])
            fused_labels = np.array([])
        
        # Get orig shape from first model's prediction
        orig_shape = all_preds[0][img_path]["orig_shape"]
        
        fused[img_path] = {
            "boxes": fused_boxes,
            "scores": fused_scores,
            "classes": fused_labels.astype(int) if len(fused_labels) > 0 else fused_labels,
            "orig_shape": orig_shape,
        }
    
    return fused


def compute_ap(recalls, precisions):
    """Compute AP using 101-point interpolation (COCO style)."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    ap = 0
    for rp in recall_points:
        prec_at_r = precisions[recalls >= rp]
        if len(prec_at_r) > 0:
            ap += prec_at_r.max()
    ap /= 101
    return ap


def evaluate_predictions(preds, gt, iou_threshold=0.5):
    """Evaluate predictions against ground truth using mAP@IoU."""
    # Collect all detections and ground truths per class
    class_dets = defaultdict(list)   # class -> list of (img_id, score, box_abs)
    class_gts = defaultdict(list)    # class -> list of (img_id, box_abs)
    class_ngt = defaultdict(int)     # class -> count of GT boxes
    
    for img_path, gt_boxes in gt.items():
        pred = preds.get(img_path)
        if pred is None:
            continue
        
        h, w = pred["orig_shape"]
        
        # Ground truth
        for cls, x1, y1, x2, y2 in gt_boxes:
            # Convert normalized GT to absolute
            class_gts[cls].append((img_path, np.array([x1*w, y1*h, x2*w, y2*h])))
            class_ngt[cls] += 1
        
        # Detections  
        if len(pred["boxes"]) > 0:
            for i in range(len(pred["boxes"])):
                box_norm = pred["boxes"][i]
                score = pred["scores"][i]
                cls = int(pred["classes"][i])
                # Convert to absolute
                box_abs = np.array([
                    box_norm[0] * w, box_norm[1] * h,
                    box_norm[2] * w, box_norm[3] * h
                ])
                class_dets[cls].append((img_path, float(score), box_abs))
    
    # Compute AP per class
    aps = {}
    for cls in range(NUM_CLASSES):
        dets = class_dets[cls]
        gts = class_gts[cls]
        ngt = class_ngt[cls]
        
        if ngt == 0:
            aps[cls] = 0.0
            continue
        
        # Sort detections by score descending
        dets.sort(key=lambda x: x[1], reverse=True)
        
        # Build GT lookup per image
        gt_per_img = defaultdict(list)
        for img_id, box in gts:
            gt_per_img[img_id].append({"box": box, "matched": False})
        
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        
        for di, (img_id, score, det_box) in enumerate(dets):
            img_gts = gt_per_img[img_id]
            
            best_iou = 0
            best_gi = -1
            
            for gi, g in enumerate(img_gts):
                iou = compute_iou(det_box, g["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            
            if best_iou >= iou_threshold and best_gi >= 0 and not img_gts[best_gi]["matched"]:
                tp[di] = 1
                img_gts[best_gi]["matched"] = True
            else:
                fp[di] = 1
        
        # Compute precision-recall curve
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / ngt
        precisions = tp_cum / (tp_cum + fp_cum)
        
        ap = compute_ap(recalls, precisions)
        aps[cls] = ap
    
    return aps


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def main():
    print("=" * 60)
    print("Multi-Model WBF Ensemble Evaluation")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading validation dataset...")
    img_files, gt = load_val_dataset(DATA_YAML)
    print(f"  Images: {len(img_files)}, GT boxes: {sum(len(v) for v in gt.values())}")
    
    # Collect predictions from each model
    all_preds = []
    model_weights = []
    for cfg in MODELS:
        preds = collect_predictions(cfg, img_files)
        all_preds.append(preds)
        model_weights.append(cfg["weight"])
    
    # --- Individual model evaluation ---
    print("\n" + "=" * 60)
    print("Individual Model Results")
    print("=" * 60)
    for i, cfg in enumerate(MODELS):
        aps = evaluate_predictions(all_preds[i], gt)
        map50 = np.mean(list(aps.values()))
        print(f"  {cfg['name']}: mAP50={map50:.5f} | " + 
              " | ".join(f"{CLASS_NAMES[c]}={aps[c]:.5f}" for c in range(NUM_CLASSES)))
    
    # --- WBF Ensemble ---
    print("\n" + "=" * 60)
    print("WBF Ensemble Results")
    print("=" * 60)
    
    # Try different WBF IoU thresholds
    for wbf_iou in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        fused = wbf_fuse(all_preds, img_files, model_weights, iou_thr=wbf_iou)
        aps = evaluate_predictions(fused, gt)
        map50 = np.mean(list(aps.values()))
        print(f"  WBF iou={wbf_iou:.2f}: mAP50={map50:.5f} | " +
              " | ".join(f"{CLASS_NAMES[c]}={aps[c]:.5f}" for c in range(NUM_CLASSES)))
    
    # --- Try different model combinations ---
    print("\n" + "=" * 60)
    print("Model Combination Sweep")
    print("=" * 60)
    
    combos = [
        ("exp_009 + exp_012", [0, 1], [1.0, 1.0]),
        ("exp_009 + exp_009_tta", [0, 2], [1.0, 1.0]),
        ("exp_012 + exp_009_tta", [1, 2], [1.0, 1.0]),
        ("all_3_equal", [0, 1, 2], [1.0, 1.0, 1.0]),
        ("all_3_weight_009", [0, 1, 2], [2.0, 1.0, 1.0]),
        ("all_3_weight_012", [0, 1, 2], [1.0, 2.0, 1.0]),
        ("all_3_weight_tta", [0, 1, 2], [1.0, 1.0, 2.0]),
    ]
    
    best_map = 0
    best_name = ""
    best_wbf_iou = 0
    
    for combo_name, indices, weights in combos:
        sub_preds = [all_preds[i] for i in indices]
        for wbf_iou in [0.5, 0.55, 0.6]:
            fused = wbf_fuse(sub_preds, img_files, weights, iou_thr=wbf_iou)
            aps = evaluate_predictions(fused, gt)
            map50 = np.mean(list(aps.values()))
            if map50 > best_map:
                best_map = map50
                best_name = combo_name
                best_wbf_iou = wbf_iou
            print(f"  {combo_name} wbf={wbf_iou:.2f}: mAP50={map50:.5f} | " +
                  " | ".join(f"{CLASS_NAMES[c]}={aps[c]:.5f}" for c in range(NUM_CLASSES)))
    
    print(f"\n  BEST: {best_name} wbf_iou={best_wbf_iou:.2f} => mAP50={best_map:.5f}")
    
    print("\nDone!")


if __name__ == "__main__":
    os.chdir("/home/satyakarthikeya/Documents/analog_project")
    main()
