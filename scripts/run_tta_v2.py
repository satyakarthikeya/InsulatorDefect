"""
TTA: Uses Ultralytics' ap_per_class for identical mAP computation.
Collects per-image TTA predictions, matches against GT, feeds to ap_per_class.
"""
import torch
import numpy as np
import cv2
import os
import yaml
import torchvision
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, DetMetrics
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent.parent
DATA_YAML = str(PROJECT / "VOC" / "voc.yaml")


def load_dataset():
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    base = cfg.get("path", str(Path(DATA_YAML).parent))
    val_dir = os.path.join(base, cfg["val"])
    label_dir = val_dir.replace("/images/", "/labels/")
    nc = cfg["nc"]
    names = cfg["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    
    img_files = sorted([
        os.path.join(val_dir, f) for f in os.listdir(val_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    gts = {}  # img_idx -> list of (cls, x1, y1, x2, y2)
    for idx, ip in enumerate(img_files):
        img = cv2.imread(ip)
        h, w = img.shape[:2]
        lf = os.path.join(label_dir, Path(ip).stem + ".txt")
        boxes = []
        if os.path.exists(lf):
            with open(lf) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        boxes.append((cls,
                                      (cx - bw/2)*w, (cy - bh/2)*h,
                                      (cx + bw/2)*w, (cy + bh/2)*h))
        gts[idx] = boxes
    
    return img_files, gts, nc, names


def tta_predict(model, img_path, scales, do_flip, nms_iou=0.5, conf=0.001):
    """Multi-scale + flip TTA for one image. Returns (N, 6): x1,y1,x2,y2,conf,cls."""
    img = cv2.imread(str(img_path))
    if img is None:
        return np.empty((0, 6))
    h0, w0 = img.shape[:2]
    
    all_boxes = []
    for scale in scales:
        flip_opts = [False, True] if do_flip else [False]
        for flip in flip_opts:
            source = cv2.flip(img, 1) if flip else img
            results = model.predict(
                source, imgsz=scale, conf=conf, iou=0.7,
                device="0", verbose=False, augment=False, max_det=300,
            )
            if results and len(results[0].boxes):
                xyxy = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                if flip:
                    o = xyxy.copy()
                    o[:, 0] = w0 - xyxy[:, 2]
                    o[:, 2] = w0 - xyxy[:, 0]
                    xyxy = o
                for i in range(len(xyxy)):
                    all_boxes.append([xyxy[i,0], xyxy[i,1], xyxy[i,2], xyxy[i,3], confs[i], clss[i]])
    
    if not all_boxes:
        return np.empty((0, 6))
    
    arr = np.array(all_boxes)
    
    # Class-aware NMS
    keep = []
    for c in range(2):
        mask = arr[:, 5] == c
        if mask.sum() == 0:
            continue
        c_boxes = torch.from_numpy(arr[mask, :4]).float()
        c_scores = torch.from_numpy(arr[mask, 4]).float()
        c_idx = np.where(mask)[0]
        k = torchvision.ops.nms(c_boxes, c_scores, nms_iou)
        keep.extend(c_idx[k.numpy()].tolist())
    
    return arr[keep] if keep else np.empty((0, 6))


def box_iou_matrix(b1, b2):
    """IoU between two box sets. b1: (N,4), b2: (M,4) xyxy. Returns (N,M)."""
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    ix1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    iy1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    ix2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    iy2 = torch.min(b1[:, None, 3], b2[None, :, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    return inter / (a1[:, None] + a2[None, :] - inter)


def evaluate_tta(model_path, scales, do_flip, nms_iou, label):
    """Run TTA and compute Ultralytics-identical mAP via ap_per_class."""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"  Scales={scales}, Flip={do_flip}, NMS={nms_iou}")
    print(f"{'='*60}")
    
    img_files, gts, nc, names = load_dataset()
    model = YOLO(model_path)
    
    # 10 IoU thresholds: 0.50 → 0.95 (matches Ultralytics)
    iou_thresholds = torch.linspace(0.5, 0.95, 10)
    
    all_tp = []       # per-image TP arrays
    all_conf = []     # per-image confidence arrays
    all_pred_cls = [] # per-image predicted class arrays
    all_target_cls = []  # per-image target class arrays
    
    for idx, img_path in enumerate(img_files):
        # TTA prediction
        merged = tta_predict(model, img_path, scales, do_flip, nms_iou)
        
        # Ground truth
        gt_list = gts.get(idx, [])
        
        if merged.shape[0] == 0 and len(gt_list) == 0:
            continue
        
        if merged.shape[0] > 0:
            pred_boxes = torch.from_numpy(merged[:, :4]).float()
            pred_conf = torch.from_numpy(merged[:, 4]).float()
            pred_cls = torch.from_numpy(merged[:, 5]).float()
        else:
            pred_boxes = torch.zeros((0, 4))
            pred_conf = torch.zeros((0,))
            pred_cls = torch.zeros((0,))
        
        if len(gt_list) > 0:
            gt_tensor = torch.tensor(gt_list)  # (M, 5): cls, x1, y1, x2, y2
            gt_boxes = gt_tensor[:, 1:5]
            gt_cls = gt_tensor[:, 0]
        else:
            gt_boxes = torch.zeros((0, 4))
            gt_cls = torch.zeros((0,))
        
        # Match predictions to GT (Ultralytics style: greedy by confidence)
        n_pred = pred_boxes.shape[0]
        tp = torch.zeros(n_pred, len(iou_thresholds), dtype=torch.bool)
        
        if n_pred > 0 and gt_boxes.shape[0] > 0:
            iou_mat = box_iou_matrix(pred_boxes, gt_boxes)  # (n_pred, n_gt)
            
            # Sort predictions by confidence (descending)
            sort_idx = pred_conf.argsort(descending=True)
            
            for t_idx, iou_thr in enumerate(iou_thresholds):
                gt_used = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
                for pi in sort_idx:
                    pc = int(pred_cls[pi].item())
                    best_iou, best_gi = 0, -1
                    for gi in range(gt_boxes.shape[0]):
                        if gt_used[gi]:
                            continue
                        if int(gt_cls[gi].item()) != pc:
                            continue
                        iou_val = iou_mat[pi, gi].item()
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_gi = gi
                    if best_iou >= iou_thr and best_gi >= 0:
                        tp[pi, t_idx] = True
                        gt_used[best_gi] = True
        
        all_tp.append(tp)
        all_conf.append(pred_conf)
        all_pred_cls.append(pred_cls)
        all_target_cls.append(gt_cls)
        
        if (idx + 1) % 30 == 0:
            print(f"  {idx+1}/{len(img_files)} images...")
    
    print(f"  All {len(img_files)} images done.")
    
    # Concatenate all stats
    tp_all = torch.cat(all_tp, 0).cpu().numpy()
    conf_all = torch.cat(all_conf, 0).cpu().numpy()
    pred_cls_all = torch.cat(all_pred_cls, 0).cpu().numpy()
    target_cls_all = torch.cat(all_target_cls, 0).cpu().numpy()
    
    print(f"  Total preds: {tp_all.shape[0]}, Total GT: {len(target_cls_all)}")
    
    # Use Ultralytics' ap_per_class — THIS is the exact same function used by model.val()
    results = ap_per_class(
        tp_all, conf_all, pred_cls_all, target_cls_all,
        plot=False, names=names, prefix="TTA",
    )
    # ap_per_class returns: (tp, fp, p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, px, prec_values)
    # We need indices 2,3,4,5 = p, r, f1, ap
    # Actually in newer ultralytics it returns a different structure
    # Let's check what we get
    
    if isinstance(results, tuple):
        if len(results) >= 6:
            tp_r, fp_r, p_r, r_r, f1_r, ap_r = results[:6]
            # ap_r shape: (nc, n_iou) — AP at each IoU threshold for each class
            ap50 = ap_r[:, 0]  # AP at IoU=0.50 (first threshold)
            mAP50 = ap50.mean()
            
            print(f"\n  mAP50 = {mAP50:.4f}")
            for i, n in names.items():
                if i < len(ap50):
                    p_val = p_r[i] if i < len(p_r) else 0
                    r_val = r_r[i] if i < len(r_r) else 0
                    print(f"  {n}: AP50={ap50[i]:.4f}  P={p_val:.3f}  R={r_val:.3f}")
            
            del model
            torch.cuda.empty_cache()
            return mAP50, {names[i]: ap50[i] for i in range(len(ap50))}
    
    print(f"  Unexpected results format: {type(results)}, len={len(results) if isinstance(results, tuple) else 'N/A'}")
    del model
    torch.cuda.empty_cache()
    return 0, {}


def main():
    print("=" * 70)
    print("TTA EVALUATION (ULTRALYTICS ap_per_class)")
    print("=" * 70)
    
    img_files, gts, nc, names = load_dataset()
    print(f"Dataset: {len(img_files)} val images, {nc} classes")
    gt_counts = defaultdict(int)
    for boxes in gts.values():
        for cls, *_ in boxes:
            gt_counts[names[cls]] += 1
    print(f"GT: {dict(gt_counts)}")
    
    models = {
        "Baseline": str(PROJECT / "experiments/exp_002_ghost_hybrid_medium3/weights/best.pt"),
        "Soup3way": str(PROJECT / "experiments/soup_3way.pt"),
        "Teacher": str(PROJECT / "experiments/exp_004_teacher_yolo11s/weights/best.pt"),
    }
    
    # Reference: Ultralytics baselines
    print("\n--- Ultralytics reference ---")
    for nm, mp in models.items():
        m = YOLO(mp)
        r = m.val(data=DATA_YAML, imgsz=640, batch=8, device="0", workers=4,
                  conf=0.001, iou=0.7, plots=False, verbose=False)
        print(f"  {nm}: mAP50={r.results_dict['metrics/mAP50(B)']:.4f}  "
              f"D1={r.box.ap50[0]:.4f}  ins={r.box.ap50[1]:.4f}")
        del m
        torch.cuda.empty_cache()
    
    # First: validate single-scale matches ultralytics
    print("\n--- Single-scale sanity check ---")
    m50, pc = evaluate_tta(models["Baseline"], [640], False, 0.6, "Baseline single@640")
    print(f"  -> Should match ~0.9417")
    
    # TTA configurations
    configs = [
        ("3s_flip", [576, 640, 704], True, 0.5),
        ("3s_flip_nms45", [576, 640, 704], True, 0.45),
        ("3s_flip_nms55", [576, 640, 704], True, 0.55),
        ("3s_noflip", [576, 640, 704], False, 0.5),
    ]
    
    results_table = []
    
    for model_name in ["Baseline", "Soup3way"]:
        for cfg_name, scales, flip, nms in configs:
            m50, pc = evaluate_tta(models[model_name], scales, flip, nms, f"{model_name}_{cfg_name}")
            results_table.append((model_name, cfg_name, m50, pc))
    
    # Teacher with best config only
    m50, pc = evaluate_tta(models["Teacher"], [576, 640, 704], True, 0.5, "Teacher_3s_flip")
    results_table.append(("Teacher", "3s_flip", m50, pc))
    
    # Summary
    print("\n\n" + "=" * 75)
    print("COMPLETE RESULTS")
    print("=" * 75)
    print(f"{'Model':<12} {'TTA':<16} {'mAP50':>7} {'D1':>7} {'ins':>7}")
    print("-" * 55)
    for model, tta, m, pc in results_table:
        d1 = pc.get("Damaged_1", 0)
        ins = pc.get("insulator", 0)
        print(f"{model:<12} {tta:<16} {m:>7.4f} {d1:>7.4f} {ins:>7.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
