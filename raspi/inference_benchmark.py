"""
Raspberry Pi Inference Benchmark — YOLO11n-Ghost-Hybrid-P3P4-Medium
====================================================================
Benchmarks inference speed and detection accuracy on test images.
Designed to run on Raspberry Pi (ARM CPU) with ONNX Runtime.

Supports two modes:
  1. ONNX Runtime (default, recommended for Pi)
  2. Ultralytics PyTorch (optional, for comparison)

Usage:
    python inference_benchmark.py                          # ONNX mode (default)
    python inference_benchmark.py --mode pytorch           # PyTorch mode
    python inference_benchmark.py --images path/to/images  # Custom image folder
    python inference_benchmark.py --warmup 5 --runs 20     # Custom benchmark params

Output:
    - Per-image inference times
    - Detection results with bounding boxes
    - Summary statistics (mean, median, min, max, FPS)
    - Saved annotated images in results/ folder
"""

import os
import sys
import time
import argparse
import platform
from pathlib import Path

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════

CLASS_NAMES = ["Damaged_1", "insulator"]
CLASS_COLORS = [(0, 0, 255), (0, 255, 0)]   # Red for damage, Green for insulator
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
INPUT_SIZE = 768


# ══════════════════════════════════════════════════════════════
#  ONNX Runtime Inference Engine
# ══════════════════════════════════════════════════════════════

class ONNXInferenceEngine:
    """Lightweight ONNX Runtime engine for Raspberry Pi deployment."""

    def __init__(self, model_path: str):
        import onnxruntime as ort

        # Use CPU execution provider (Pi doesn't have CUDA)
        providers = ["CPUExecutionProvider"]

        # Session options optimized for Pi
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4     # Pi 4 has 4 cores
        sess_opts.inter_op_num_threads = 1

        self.session = ort.InferenceSession(model_path, sess_opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 768, 768]
        self.output_names = [o.name for o in self.session.get_outputs()]

        print(f"  ONNX model loaded: {model_path}")
        print(f"  Input: {self.input_name} {self.input_shape}")
        print(f"  Output: {self.output_names}")
        print(f"  Provider: {self.session.get_providers()}")

    def preprocess(self, image: np.ndarray) -> tuple:
        """Letterbox resize + normalize. Returns (blob, scale_info)."""
        h, w = image.shape[:2]
        scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Letterbox pad to INPUT_SIZE × INPUT_SIZE
        pad_w = (INPUT_SIZE - new_w) // 2
        pad_h = (INPUT_SIZE - new_h) // 2
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # HWC → CHW, BGR → RGB, normalize to [0, 1], add batch dim
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        return blob, (scale, pad_w, pad_h, w, h)

    def postprocess(self, output: np.ndarray, scale_info: tuple) -> list:
        """Parse YOLO output → list of (x1, y1, x2, y2, conf, cls_id)."""
        scale, pad_w, pad_h, orig_w, orig_h = scale_info

        # Output shape: (1, num_classes + 4, num_boxes) → transpose to (num_boxes, num_classes + 4)
        predictions = output[0].T  # (num_boxes, 4 + num_classes)

        # Filter by confidence
        class_scores = predictions[:, 4:]           # (num_boxes, num_classes)
        max_scores = np.max(class_scores, axis=1)   # (num_boxes,)
        mask = max_scores > CONF_THRESHOLD
        predictions = predictions[mask]
        max_scores = max_scores[mask]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        if len(predictions) == 0:
            return []

        # Convert cx, cy, w, h → x1, y1, x2, y2
        boxes = predictions[:, :4].copy()
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # Remove letterbox padding and rescale to original image
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        # Clip to image bounds
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # NMS
        detections = []
        for cls_id in range(len(CLASS_NAMES)):
            cls_mask = class_ids == cls_id
            if not np.any(cls_mask):
                continue
            cls_boxes = np.stack([x1[cls_mask], y1[cls_mask],
                                  x2[cls_mask], y2[cls_mask]], axis=1)
            cls_scores = max_scores[cls_mask]

            indices = self._nms(cls_boxes, cls_scores, IOU_THRESHOLD)
            for idx in indices:
                detections.append((
                    float(cls_boxes[idx, 0]), float(cls_boxes[idx, 1]),
                    float(cls_boxes[idx, 2]), float(cls_boxes[idx, 3]),
                    float(cls_scores[idx]),   int(cls_id)
                ))

        return detections

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list:
        """Pure NumPy Non-Maximum Suppression."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            remaining = np.where(iou <= iou_thresh)[0]
            order = order[remaining + 1]

        return keep

    def infer(self, image: np.ndarray) -> tuple:
        """Run full pipeline: preprocess → inference → postprocess.
        Returns (detections, inference_time_ms).
        inference_time_ms only measures the ONNX session.run() call.
        """
        blob, scale_info = self.preprocess(image)

        # --- Measure ONLY model inference time ---
        t_start = time.perf_counter()
        output = self.session.run(self.output_names, {self.input_name: blob})[0]
        t_end = time.perf_counter()
        infer_ms = (t_end - t_start) * 1000

        detections = self.postprocess(output, scale_info)
        return detections, infer_ms

    def infer_full_pipeline(self, image: np.ndarray) -> tuple:
        """Run full pipeline and measure end-to-end time.
        Returns (detections, inference_ms, total_pipeline_ms).
        """
        t_total_start = time.perf_counter()

        blob, scale_info = self.preprocess(image)

        t_infer_start = time.perf_counter()
        output = self.session.run(self.output_names, {self.input_name: blob})[0]
        t_infer_end = time.perf_counter()

        detections = self.postprocess(output, scale_info)

        t_total_end = time.perf_counter()

        infer_ms = (t_infer_end - t_infer_start) * 1000
        total_ms = (t_total_end - t_total_start) * 1000

        return detections, infer_ms, total_ms


# ══════════════════════════════════════════════════════════════
#  Ultralytics PyTorch Inference Engine (Optional)
# ══════════════════════════════════════════════════════════════

class PyTorchInferenceEngine:
    """Ultralytics-based PyTorch inference for comparison."""

    def __init__(self, model_path: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        print(f"  PyTorch model loaded: {model_path}")

    def infer(self, image: np.ndarray) -> tuple:
        t_start = time.perf_counter()
        results = self.model.predict(
            image, imgsz=INPUT_SIZE, conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD, verbose=False
        )
        t_end = time.perf_counter()
        infer_ms = (t_end - t_start) * 1000

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append((x1, y1, x2, y2, conf, cls_id))

        return detections, infer_ms

    def infer_full_pipeline(self, image: np.ndarray) -> tuple:
        detections, infer_ms = self.infer(image)
        return detections, infer_ms, infer_ms


# ══════════════════════════════════════════════════════════════
#  Visualization
# ══════════════════════════════════════════════════════════════

def draw_detections(image: np.ndarray, detections: list, infer_ms: float) -> np.ndarray:
    """Draw bounding boxes, labels, and inference time on image."""
    annotated = image.copy()

    for (x1, y1, x2, y2, conf, cls_id) in detections:
        color = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else (255, 255, 255)
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

        # Box
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (int(x1), int(y1) - th - 8),
                       (int(x1) + tw + 4, int(y1)), color, -1)
        cv2.putText(annotated, label, (int(x1) + 2, int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Inference time overlay
    time_label = f"Inference: {infer_ms:.1f}ms ({1000/infer_ms:.1f} FPS)"
    cv2.putText(annotated, time_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return annotated


# ══════════════════════════════════════════════════════════════
#  Ground Truth Loader
# ══════════════════════════════════════════════════════════════

def load_ground_truth(label_path: str, img_w: int, img_h: int) -> list:
    """Load YOLO format labels → list of (x1, y1, x2, y2, cls_id)."""
    if not os.path.exists(label_path):
        return []

    gt_boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            gt_boxes.append((x1, y1, x2, y2, cls_id))

    return gt_boxes


# ══════════════════════════════════════════════════════════════
#  Main Benchmark
# ══════════════════════════════════════════════════════════════

def run_benchmark(args):
    global CONF_THRESHOLD, IOU_THRESHOLD
    CONF_THRESHOLD = args.conf
    IOU_THRESHOLD = args.iou

    print("=" * 70)
    print("  YOLO11n-Ghost-Hybrid-P3P4-Medium — Inference Benchmark")
    print("=" * 70)

    # System info
    print(f"\n  Platform:    {platform.platform()}")
    print(f"  Machine:     {platform.machine()}")
    print(f"  Processor:   {platform.processor() or 'N/A'}")
    print(f"  Python:      {platform.python_version()}")
    print(f"  Mode:        {args.mode.upper()}")
    print(f"  Model size:  868K parameters | 5.7 GFLOPs")
    print(f"  Input:       {INPUT_SIZE}×{INPUT_SIZE}")
    print(f"  Conf:        {CONF_THRESHOLD} | IoU: {IOU_THRESHOLD}")
    print()

    # Load engine
    print("─" * 70)
    print("  Loading model...")
    if args.mode == "onnx":
        model_path = os.path.join(args.model_dir, "best_exp012.onnx")
        engine = ONNXInferenceEngine(model_path)
    else:
        model_path = os.path.join(args.model_dir, "best_exp012.pt")
        engine = PyTorchInferenceEngine(model_path)
    print("─" * 70)

    # Discover test images
    img_dir = args.images
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    if not img_files:
        print(f"  ERROR: No images found in {img_dir}")
        sys.exit(1)

    print(f"\n  Found {len(img_files)} test images in {img_dir}")

    # Results directory
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # ── Warmup runs ──
    print(f"\n  Warming up ({args.warmup} runs)...")
    warmup_img = cv2.imread(os.path.join(img_dir, img_files[0]))
    for _ in range(args.warmup):
        engine.infer(warmup_img)
    print("  Warmup complete.\n")

    # ── Benchmark ──
    print("=" * 70)
    print(f"  {'Image':<40} {'Infer(ms)':>10} {'Total(ms)':>10} {'FPS':>8} {'Detections':>12}")
    print("─" * 70)

    all_infer_times = []
    all_total_times = []
    all_detections = []
    total_d1 = 0
    total_ins = 0

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"  SKIP: Cannot read {img_file}")
            continue

        # Run multiple times and take median for stable measurement
        infer_times = []
        total_times = []
        last_dets = None

        for _ in range(args.runs):
            dets, infer_ms, total_ms = engine.infer_full_pipeline(image)
            infer_times.append(infer_ms)
            total_times.append(total_ms)
            last_dets = dets

        median_infer = sorted(infer_times)[len(infer_times) // 2]
        median_total = sorted(total_times)[len(total_times) // 2]
        fps = 1000.0 / median_total if median_total > 0 else 0

        # Count detections by class
        n_d1 = sum(1 for d in last_dets if d[5] == 0)
        n_ins = sum(1 for d in last_dets if d[5] == 1)
        total_d1 += n_d1
        total_ins += n_ins

        det_str = f"D1={n_d1} ins={n_ins}"
        print(f"  {img_file:<40} {median_infer:>8.1f}ms {median_total:>8.1f}ms {fps:>7.1f} {det_str:>12}")

        all_infer_times.append(median_infer)
        all_total_times.append(median_total)
        all_detections.append((img_file, last_dets))

        # Save annotated image
        annotated = draw_detections(image, last_dets, median_infer)
        out_path = os.path.join(results_dir, f"det_{img_file}")
        cv2.imwrite(out_path, annotated)

    # ── Summary Statistics ──
    print("─" * 70)
    print()

    if all_infer_times:
        infer_arr = np.array(all_infer_times)
        total_arr = np.array(all_total_times)

        print("=" * 70)
        print("  INFERENCE TIME SUMMARY")
        print("=" * 70)
        print(f"  ┌──────────────────────────────────────────────────────┐")
        print(f"  │  Model Inference Only (ONNX session.run)             │")
        print(f"  │    Mean:     {np.mean(infer_arr):>8.1f} ms                          │")
        print(f"  │    Median:   {np.median(infer_arr):>8.1f} ms                          │")
        print(f"  │    Min:      {np.min(infer_arr):>8.1f} ms                          │")
        print(f"  │    Max:      {np.max(infer_arr):>8.1f} ms                          │")
        print(f"  │    Std:      {np.std(infer_arr):>8.1f} ms                          │")
        print(f"  ├──────────────────────────────────────────────────────┤")
        print(f"  │  Full Pipeline (preprocess + infer + postprocess)    │")
        print(f"  │    Mean:     {np.mean(total_arr):>8.1f} ms                          │")
        print(f"  │    Median:   {np.median(total_arr):>8.1f} ms                          │")
        print(f"  │    Min:      {np.min(total_arr):>8.1f} ms                          │")
        print(f"  │    Max:      {np.max(total_arr):>8.1f} ms                          │")
        print(f"  ├──────────────────────────────────────────────────────┤")
        avg_fps = 1000.0 / np.mean(total_arr)
        print(f"  │  Throughput: {avg_fps:>8.1f} FPS (full pipeline)          │")
        print(f"  └──────────────────────────────────────────────────────┘")

        print()
        print("=" * 70)
        print("  DETECTION SUMMARY")
        print("=" * 70)
        print(f"  Images processed:     {len(all_infer_times)}")
        print(f"  Total Damaged_1:      {total_d1} detections")
        print(f"  Total insulator:      {total_ins} detections")
        print(f"  Annotated images:     {results_dir}/")
        print()

        # Load and compare with ground truth if labels exist
        print("=" * 70)
        print("  GROUND TRUTH COMPARISON")
        print("=" * 70)
        print(f"  {'Image':<40} {'GT D1':>6} {'Pred D1':>8} {'GT ins':>7} {'Pred ins':>9}")
        print("  " + "─" * 68)

        gt_total_d1 = 0
        gt_total_ins = 0

        for img_file, dets in all_detections:
            img_path = os.path.join(img_dir, img_file)
            image = cv2.imread(img_path)
            h, w = image.shape[:2]

            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(img_dir, label_file)
            gt_boxes = load_ground_truth(label_path, w, h)

            gt_d1 = sum(1 for g in gt_boxes if g[4] == 0)
            gt_ins = sum(1 for g in gt_boxes if g[4] == 1)
            pred_d1 = sum(1 for d in dets if d[5] == 0)
            pred_ins = sum(1 for d in dets if d[5] == 1)

            gt_total_d1 += gt_d1
            gt_total_ins += gt_ins

            match_d1 = "✓" if pred_d1 >= gt_d1 else "✗"
            match_ins = "✓" if pred_ins >= gt_ins else "✗"

            print(f"  {img_file:<40} {gt_d1:>5}  {pred_d1:>5} {match_d1}  {gt_ins:>5}  {pred_ins:>5} {match_ins}")

        print("  " + "─" * 68)
        print(f"  {'TOTAL':<40} {gt_total_d1:>5}  {total_d1:>5}    {gt_total_ins:>5}  {total_ins:>5}")

        print()
        print("=" * 70)
        print("  MODEL CARD")
        print("=" * 70)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Architecture:   YOLO11n-Ghost-Hybrid-P3P4-Medium")
        print(f"  Parameters:     867,664 (868K)")
        print(f"  GFLOPs:         5.7")
        print(f"  Model size:     {model_size_mb:.1f} MB ({args.mode.upper()})")
        print(f"  Input size:     {INPUT_SIZE}×{INPUT_SIZE}")
        print(f"  Classes:        {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES)})")
        print(f"  Val mAP50:      96.46% (exp_012)")
        print(f"  Val mAP50-95:   66.87%")
        print(f"  Avg inference:  {np.mean(infer_arr):.1f}ms on {platform.machine()}")
        print(f"  Format:         {args.mode.upper()}")
        print("=" * 70)
        print()
        print(f"  Annotated images saved to: {results_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Inference Benchmark for Raspberry Pi"
    )
    parser.add_argument(
        "--mode", choices=["onnx", "pytorch"], default="onnx",
        help="Inference backend: 'onnx' (default, Pi-optimized) or 'pytorch'"
    )
    parser.add_argument(
        "--images", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images"),
        help="Path to test images folder (default: test_images/)"
    )
    parser.add_argument(
        "--model-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
        help="Path to models folder (default: models/)"
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup runs before benchmark (default: 3)"
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of runs per image for stable timing (default: 10)"
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESHOLD,
        help=f"Confidence threshold (default: {CONF_THRESHOLD})"
    )
    parser.add_argument(
        "--iou", type=float, default=IOU_THRESHOLD,
        help=f"IoU threshold for NMS (default: {IOU_THRESHOLD})"
    )

    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
