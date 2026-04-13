"""
Custom TTA with proper COCO-style mAP evaluation using pycocotools.
Bypasses Ultralytics' built-in augment=True which crashes on P3P4 models.

Uses Ultralytics' own val() for single-scale baselines,
and pycocotools for TTA-merged multi-scale evaluation.
"""
import torch
import numpy as np
import cv2
import os
import json
import yaml
import torchvision
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT = Path(__file__).resolve().parent.parent
DATA_YAML = str(PROJECT / "VOC" / "voc.yaml")
CACHE_DIR = str(PROJECT / "experiments" / "tta_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def load_dataset():
    """Load val images and GT, return in both raw and COCO format."""
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    
    base = cfg.get("path", str(Path(DATA_YAML).parent))
    val_rel = cfg["val"]
    val_dir = os.path.join(base, val_rel)
    label_dir = val_dir.replace("/images/", "/labels/")
    nc = cfg["nc"]
    names = cfg["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    
    img_files = sorted([
        os.path.join(val_dir, f) for f in os.listdir(val_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    # Build COCO-format GT
    coco_images = []
    coco_annotations = []
    ann_id = 1
    
    for img_id, img_path in enumerate(img_files, start=1):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        coco_images.append({
            "id": img_id,
            "file_name": os.path.basename(img_path),
            "width": w,
            "height": h,
        })
        
        label_file = os.path.join(label_dir, Path(img_path).stem + ".txt")
        if os.path.exists(label_file):
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        box_w = bw * w
                        box_h = bh * h
                        coco_annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls,
                            "bbox": [x1, y1, box_w, box_h],  # COCO format: x,y,w,h
                            "area": box_w * box_h,
                            "iscrowd": 0,
                        })
                        ann_id += 1
    
    categories = [{"id": i, "name": n} for i, n in names.items()]
    
    coco_gt = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    
    gt_path = os.path.join(CACHE_DIR, "gt_coco.json")
    with open(gt_path, "w") as f:
        json.dump(coco_gt, f)
    
    return img_files, gt_path, nc, names


def run_tta_and_save(model, img_files, scales, do_flip, nms_iou=0.5, conf=0.001):
    """
    Run TTA at multiple scales + optional flip.
    Returns COCO-format predictions list.
    """
    all_preds = defaultdict(list)
    
    total = len(scales) * (2 if do_flip else 1) * len(img_files)
    done = 0
    
    for scale in scales:
        flip_opts = [False, True] if do_flip else [False]
        for flip in flip_opts:
            for img_idx, img_path in enumerate(img_files):
                img = cv2.imread(img_path)
                h0, w0 = img.shape[:2]
                source = cv2.flip(img, 1) if flip else img
                
                results = model.predict(
                    source, imgsz=scale, conf=conf, iou=0.7,
                    device="0", verbose=False, augment=False,
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
                        all_preds[img_idx].append([
                            xyxy[i, 0], xyxy[i, 1], xyxy[i, 2], xyxy[i, 3],
                            confs[i], int(clss[i])
                        ])
                
                done += 1
            
            tag = f"scale={scale}" + (" flip" if flip else "")
            print(f"  {tag} done ({done}/{total})")
    
    # Merge with class-aware NMS per image
    nc = 2
    coco_preds = []
    
    for img_idx in range(len(img_files)):
        boxes_list = all_preds.get(img_idx, [])
        if not boxes_list:
            continue
        
        img_id = img_idx + 1  # COCO img_id is 1-indexed
        boxes_arr = np.array(boxes_list)
        
        for c in range(nc):
            mask = boxes_arr[:, 5] == c
            if mask.sum() == 0:
                continue
            
            c_boxes = torch.from_numpy(boxes_arr[mask, :4]).float()
            c_scores = torch.from_numpy(boxes_arr[mask, 4]).float()
            
            keep = torchvision.ops.nms(c_boxes, c_scores, nms_iou)
            
            for k in keep:
                idx = k.item()
                x1, y1, x2, y2 = c_boxes[idx].tolist()
                score = c_scores[idx].item()
                coco_preds.append({
                    "image_id": img_id,
                    "category_id": c,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO: x,y,w,h
                    "score": score,
                })
    
    return coco_preds


def eval_coco(gt_path, preds, label=""):
    """Evaluate predictions using pycocotools."""
    coco_gt = COCO(gt_path)
    
    if not preds:
        print(f"  {label}: No predictions!")
        return 0.0, {}
    
    pred_path = os.path.join(CACHE_DIR, "pred_coco.json")
    with open(pred_path, "w") as f:
        json.dump(preds, f)
    
    coco_dt = coco_gt.loadRes(pred_path)
    
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.array([0.5])  # Only IoU=0.5
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Per-category AP
    per_class = {}
    cats = coco_gt.getCatIds()
    for cat_id in cats:
        coco_eval_c = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval_c.params.iouThrs = np.array([0.5])
        coco_eval_c.params.catIds = [cat_id]
        coco_eval_c.evaluate()
        coco_eval_c.accumulate()
        
        # Get AP at IoU=0.5 (index 0)
        ap = coco_eval_c.stats[0]  # AP @ IoU=0.50
        cat_name = coco_gt.loadCats(cat_id)[0]["name"]
        per_class[cat_name] = ap
        print(f"  {cat_name}: AP50 = {ap:.4f}")
    
    mAP50 = coco_eval.stats[0]
    print(f"  mAP50 = {mAP50:.4f}")
    
    return mAP50, per_class


def main():
    print("=" * 70)
    print("TTA EVALUATION WITH PYCOCOTOOLS (COCO-STYLE AP)")
    print("=" * 70)
    
    img_files, gt_path, nc, names = load_dataset()
    print(f"Val: {len(img_files)} images, {nc} classes")
    
    # Models to evaluate
    models_cfg = {
        "Baseline": str(PROJECT / "experiments/exp_002_ghost_hybrid_medium3/weights/best.pt"),
        "Soup3way": str(PROJECT / "experiments/soup_3way.pt"),
        "Teacher": str(PROJECT / "experiments/exp_004_teacher_yolo11s/weights/best.pt"),
    }
    
    # TTA configurations 
    tta_cfgs = {
        "single_640": {"scales": [640], "flip": False, "nms_iou": 0.6},
        "3scale": {"scales": [576, 640, 704], "flip": False, "nms_iou": 0.5},
        "3scale_flip": {"scales": [576, 640, 704], "flip": True, "nms_iou": 0.5},
        "5scale_flip": {"scales": [512, 576, 640, 704, 768], "flip": True, "nms_iou": 0.5},
    }
    
    results = []
    
    for model_name, model_path in models_cfg.items():
        model = YOLO(model_path)
        
        for tta_name, cfg in tta_cfgs.items():
            print(f"\n{'='*60}")
            print(f"{model_name} — {tta_name}")
            print(f"{'='*60}")
            
            preds = run_tta_and_save(
                model, img_files,
                scales=cfg["scales"],
                do_flip=cfg["flip"],
                nms_iou=cfg["nms_iou"],
            )
            print(f"  Total merged predictions: {len(preds)}")
            
            mAP50, per_class = eval_coco(gt_path, preds, f"{model_name}_{tta_name}")
            
            results.append({
                "model": model_name,
                "tta": tta_name,
                "mAP50": mAP50,
                "per_class": per_class,
            })
        
        del model
        torch.cuda.empty_cache()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY (COCO AP@IoU=0.50)")
    print("=" * 80)
    print(f"{'Model':<15} {'TTA':<18} {'mAP50':>8} {'Damaged_1':>10} {'insulator':>10}")
    print("-" * 80)
    for r in results:
        d1 = r["per_class"].get("Damaged_1", 0)
        ins = r["per_class"].get("insulator", 0)
        print(f"{r['model']:<15} {r['tta']:<18} {r['mAP50']:>8.4f} {d1:>10.4f} {ins:>10.4f}")
    print("=" * 80)
    
    # Also show Ultralytics baseline for reference
    print("\n--- Ultralytics reference (single-scale) ---")
    for model_name, model_path in models_cfg.items():
        m = YOLO(model_path)
        rv = m.val(data=DATA_YAML, imgsz=640, batch=8, device="0", workers=4,
                   conf=0.001, iou=0.7, plots=False, verbose=False)
        map50 = rv.results_dict["metrics/mAP50(B)"]
        print(f"  {model_name}: mAP50={map50:.4f}  D1={rv.box.ap50[0]:.4f}  ins={rv.box.ap50[1]:.4f}")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
