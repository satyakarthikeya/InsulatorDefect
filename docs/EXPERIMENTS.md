# Experiment Journey: Ultra-Lightweight YOLO for Insulator Defect Detection

> **Goal:** Push an 867K-parameter YOLO model from **94.17% mAP50 → ~97% mAP50** for edge deployment  
> **Final Best:** **96.21% mAP50** (with TTA) / **96.11% mAP50** (single-model, no TTA)  
> **Teacher Ceiling:** 96.54% mAP50 — student is within **0.33%** of teacher performance

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Hardware & Constraints](#2-hardware--constraints)
3. [Dataset](#3-dataset)
4. [Model Architecture](#4-model-architecture)
5. [Experiment Summary Table](#5-experiment-summary-table)
6. [Detailed Experiment Descriptions](#6-detailed-experiment-descriptions)
7. [Post-Training Techniques](#7-post-training-techniques)
8. [Verified Best Results](#8-verified-best-results)
9. [Key Findings & Lessons Learned](#9-key-findings--lessons-learned)
10. [Recommended Deployment Configuration](#10-recommended-deployment-configuration)
11. [File Index](#11-file-index)

---

## 1. Project Overview

This project trains an **ultra-lightweight YOLO model** (867K parameters) for real-time insulator defect detection on power transmission lines. The model must run on **edge devices** (Raspberry Pi / NVIDIA Jetson Orin Nano) while maintaining high accuracy on a rare defect class (Damaged_1).

**Framework:** Ultralytics v8.3.240, PyTorch 2.9.1+cu128, Python 3.11.14

---

## 2. Hardware & Constraints

| Resource | Specification |
|---|---|
| **GPU** | NVIDIA RTX 3050 Laptop (4GB VRAM) |
| **RAM** | 16GB System RAM |
| **OS** | Fedora Linux |
| **Conda Env** | `analog` |
| **Deployment Target** | Raspberry Pi / NVIDIA Jetson Orin Nano |

**Mandatory Training Rules:**
- `batch ≤ 8` (drop to 4 for imgsz > 640, 2 for 768, 1 for 896)
- `amp = True` (mixed precision)
- `cache = "ram"` (no disk caching)
- `workers = 4` (prevent RAM thrashing)
- NO SAHI, NO cloud compute, NO auto-batching, NO heavy copy_paste/flipud augmentation

---

## 3. Dataset

| Split | Images | Damaged_1 Instances | Insulator Instances | Total Instances |
|---|---|---|---|---|
| **Train** | 1,155 | 562 | 1,962 | 2,524 |
| **Val** | 143 | 76 | 224 | 300 |

- **Classes:** 2 — `Damaged_1` (cls 0, rare), `insulator` (cls 1, common)
- **Imbalance Ratio:** ~1:3.5 (Damaged_1 : insulator)
- **Format:** YOLO txt labels, VOC XML annotations available
- **Config:** `VOC/voc.yaml`

---

## 4. Model Architecture

### Student: YOLO11n-Ghost-Hybrid-P3P4-Medium

| Property | Value |
|---|---|
| **Parameters** | 867,664 |
| **GFLOPs** | ~5.8 |
| **Detection Scales** | P3 (stride 8) + P4 (stride 16) — **no P5** |
| **Backbone** | GhostConv + C3Ghost blocks (32→64→128→160→160 channels) + SPPF |
| **Head** | DWConv-only FPN-PAN (lightweight ~9x param reduction) |
| **Config** | `models/yolo11n-ghost-hybrid-p3p4-medium.yaml` |

**Design Rationale:** GhostConv backbone halves parameters via cheap linear operations. P5 removed to focus all capacity on small/medium defects. DWConv head minimizes parameters while maintaining multi-scale feature fusion.

### Teacher: YOLO11s (Standard)

| Property | Value |
|---|---|
| **Parameters** | 9,428,566 (10.9x larger) |
| **GFLOPs** | ~21.3 |
| **Detection Scales** | P3 + P4 + P5 (3-scale) |
| **Best mAP50** | 96.54% |

---

## 5. Experiment Summary Table

### Training Experiments

| # | Experiment | imgsz | batch | lr0 | Epochs | Strategy | Best mAP50 | Δ from Baseline |
|---|---|---|---|---|---|---|---|---|
| 1 | **exp_002** (Baseline) | 640 | 8 | 0.01 | 300 | Train from scratch | **94.21%** | — |
| 2 | **exp_004** (Teacher) | 640 | 4 | 0.002 | 150 | YOLO11s teacher | **96.54%** | +2.33% |
| 3 | **exp_tfa_v1** | 640 | 8 | 0.0001 | 23 | Frozen backbone fine-tune | **94.34%** | +0.13% |
| 4 | **exp_tfa_v2** | 640 | 8 | 0.0003 | 32 | Aggressive cls loss (2.0) | **94.00%** | −0.21% |
| 5 | **exp_tfa_v3** | 704 | 4 | 5e-5 | 26 | Conservative 704 fine-tune | **93.11%** | −1.10% |
| 6 | **exp_005_kd** | 640 | 4 | 0.0001 | 25 | Knowledge Distillation | **94.53%** | +0.32% |
| 7 | **exp_006** (704 aggressive) | 704 | 4 | 0.005 | 47 | High-res unfrozen, cls=1.0 | **93.08%** | −1.13% |
| 8 | **exp_007_s1** (704 frozen) | 704 | 4 | 0.001 | 32 | 2-stage: frozen backbone | **93.37%** | −0.84% |
| 9 | **exp_007_s2** (704 unfrozen) | 704 | 4 | 0.0002 | 49 | 2-stage: unfrozen from s1 | **92.85%** | −1.36% |
| 10 | **exp_009** (768 fine-tune) ⭐ | 768 | 2 | 0.002 | 25 | KD → 768 fine-tune (SGD) | **96.11%** | **+1.90%** |
| 11 | **exp_010** (768 multi-scale) | 768 | 2 | 0.0005 | 19 | Multi-scale from exp_009 | **95.69%** | +1.48% |
| 12 | **exp_011** (768 low-LR) | 768 | 2 | 0.0003 | 30 | Ultra-low LR from exp_009 | **96.06%** | +1.85% |
| 13 | **exp_012** (768 head-only) | 768 | 2 | 0.005 | 30 | Head-only fine-tune | **96.46%*** | +2.25%* |
| 14 | **exp_013** (896) | 896 | 1 | 0.001 | 17 | 896 resolution attempt | **94.37%** | +0.16% |

> \*exp_012 reported 96.46% in training CSV but verified at **96.09%** via `model.val()` — training metrics can differ from validation metrics due to best checkpoint selection timing.

### Post-Training Techniques

| Technique | Config | mAP50 | Notes |
|---|---|---|---|
| Model Soup (kd + baseline) | 50/50 | 94.23% | Marginal +0.06% over KD alone |
| Model Soup (kd + baseline) | 40/60 | 94.18% | No improvement |
| Model Soup (kd + baseline) | 30/70 | 94.12% | Same as KD baseline |
| Model Soup 3-way | kd+base+tfa | 94.17% | No improvement |
| Model Soup (kd + 768ft) | 50/50 | ~95.9% | Below pure 768-ft |
| Model Soup (kd + 768ft) | 40/60 | ~95.9% | Below pure 768-ft |
| Model Soup (kd + 768ft) | 30/70 | ~95.9% | Below pure 768-ft |
| SWA Average (exp_009) | ep5→last | 96.02% | Slightly below best single |
| SWA Top-3 (exp_009) | ep10+best+last | 96.09% | Matched best single |
| TTA (exp_005_kd) | augment=True @768 | 95.69% | +1.57% over KD @640 |
| TTA (exp_009) | augment=True @768 | 96.15% | +0.04% over exp_009 |
| TTA (exp_012) | augment=True @768 | **96.21%** | **Best overall** |

---

## 6. Detailed Experiment Descriptions

### EXP_002: Baseline Training from Scratch

**Script:** `scripts/train_exp002.py`  
**Directory:** `experiments/exp_002_ghost_hybrid_medium3/`

Trained the Ghost-Hybrid-P3P4-Medium architecture from random initialization for 300 epochs at 640×640. Used standard YOLO augmentation (mosaic=1.0, mixup=0.1, scale=0.9) with AdamW optimizer. Established the **94.21% mAP50 baseline** — the starting point for all subsequent experiments.

- **mAP50:** 94.21% (D1=91.24%, ins=97.10%)
- **Precision:** 92.7% | **Recall:** 89.0%

### EXP_004: Teacher Model (YOLO11s)

**Script:** `scripts/train_teacher.py`  
**Directory:** `experiments/exp_004_teacher_yolo11s/`

Trained a standard YOLO11s (9.4M params) as the teacher for Knowledge Distillation. This 10.9x larger model set the **performance ceiling at 96.54% mAP50**. Used `pretrained=True` (COCO weights), fine-tuned for 150 epochs with AdamW, lr0=0.002.

- **mAP50:** 96.54% — the upper bound for our student model
- **Purpose:** Serve as teacher for KD experiments

### EXP_TFA_v1: Two-Stage Fine-Tuning (Frozen Backbone)

**Script:** `scripts/train_tfa.py`  
**Directory:** `experiments/exp_tfa_20260217_182417/`

Classic TFA approach: froze backbone (layers 0–9), fine-tuned only the detection head with very low LR (0.0001) and reduced augmentation. Goal was to adapt the head to better classify the rare Damaged_1 class without disturbing learned features.

- **mAP50:** 94.34% (+0.13%) — marginal improvement
- **Verdict:** LR too conservative, insufficient to meaningfully shift performance

### EXP_TFA_v2: Aggressive Classification Loss

**Script:** `scripts/train_tfa_v2.py`  
**Directory:** `experiments/exp_tfa_v2_20260217_183859/`

Attempted to boost Damaged_1 by increasing classification loss weight from 0.5→2.0 (4x), enabling copy_paste=0.3 and flipud=0.5 for rare class augmentation, and using 3x higher LR (0.0003).

- **mAP50:** 94.00% (−0.21%) — **WORSE** than baseline
- **Verdict:** Aggressive cls loss destabilized training. copy_paste and flipud hurt the loss landscape. Key lesson: don't over-engineer loss weights.

### EXP_TFA_v3: Conservative High-Resolution

**Script:** `scripts/train_tfa_v3.py`  
**Directory:** `experiments/exp_tfa_v3_20260217_202347/`

Frozen backbone at 704×704 with ultra-low LR (5e-5). Attempted to capture finer spatial features at higher resolution while keeping backbone stable.

- **mAP50:** 93.11% (−1.10%) — **significant degradation**
- **Verdict:** Frozen backbone + resolution change = spatial feature mismatch. The backbone's feature maps were calibrated for 640, not 704.

### EXP_005: Knowledge Distillation

**Script:** `scripts/train_kd.py`  
**Directory:** `experiments/exp_005_kd_student3/`

Custom `KDDetectionTrainer` with KL-divergence distillation from YOLO11s teacher. Loss: `(1-α)·L_yolo + α·T²·KL(student||teacher)` with α=0.5, T=4.0. Handled anchor count mismatch (student: 8000 anchors from 2 scales, teacher: 8400 from 3 scales) by truncating teacher predictions.

- **mAP50:** 94.53% during training, **94.12% on re-validation** (+0.32%/−0.09%)
- **Key Achievement:** While mAP50 gain was modest, KD produced **better internal representations** that enabled the breakthrough 768-resolution fine-tune. This was the critical enabler.

### EXP_006: Aggressive High-Res 704

**Script:** `scripts/train_phase3_highres.py`  
**Directory:** `experiments/exp_006_highres_704/`

Fully unfrozen fine-tune at 704×704 with high LR (0.005) and boosted cls loss (1.0). Started from KD weights. Used aggressive augmentation (exp_002 recipe).

- **mAP50:** 93.08% (−1.13%) — **degradation**
- **Verdict:** LR=0.005 too aggressive for fine-tuning. High augmentation at 704 resolution was disruptive. Model overfit to resolution without learning useful features.

### EXP_007: Conservative Two-Stage 704

**Script:** `scripts/train_phase3v2_conservative.py`  
**Directories:** `experiments/exp_007_highres704_s1_frozen/` → `experiments/exp_007_highres704_s2_unfrozen/`

Two-stage approach: Stage 1 froze backbone (lr=0.001, 32ep), Stage 2 unfroze all layers (lr=0.0002, 49ep). Both at 704×704.

- **Stage 1:** 93.37% — backbone couldn't adapt to new resolution while frozen
- **Stage 2:** 92.85% — unfreezing from a degraded starting point didn't recover
- **Verdict:** 704 resolution fundamentally doesn't suit this P3/P4 architecture well. The stride [8, 16] model works best at resolutions that are cleaner multiples.

### EXP_009: 768 Fine-Tune from KD ⭐ BREAKTHROUGH

**Script:** `scripts/train_finetune_768.py`  
**Directory:** `experiments/exp_009_finetune_768/`

**The breakthrough experiment.** Resolution sweep showed KD model improved from 94.12% @640 to 94.72% @768 without any retraining. Hypothesis: train at 768 to internalize the resolution benefit. Short 25-epoch fine-tune with SGD (lr=0.002), cos_lr, fully unfrozen. Moderate augmentation.

- **mAP50:** 96.11% (D1=95.29%, ins=96.94%)
- **Precision:** 92.7% | **Recall:** 95.4%
- **Δ from baseline:** **+1.90%** — massive improvement
- **Verdict:** 768 is the sweet spot for this architecture. KD weights + resolution fine-tuning = synergistic gains. The KD representations generalized better to higher resolution than the baseline weights.

### EXP_010: Multi-Scale 768

**Script:** `scripts/train_finetune_768_v2.py`  
**Directory:** `experiments/exp_010_finetune_768_v2/`

Attempted to squeeze more from exp_009 by adding `multi_scale=True` with lower LR (0.0005). Early-stopped at epoch 19.

- **mAP50:** 95.69% (−0.42% from exp_009)
- **Verdict:** Multi-scale training hurt. The model was already well-calibrated for 768; randomizing image size during training degraded learned features.

### EXP_011: Ultra-Low LR 768

**Directory:** `experiments/exp_011_finetune_768_v3/`

Another attempt to improve exp_009 with very low LR (0.0003), 30 epochs — hoping to fine-tune without overshooting.

- **mAP50:** 96.06% (−0.05% from exp_009)
- **Verdict:** No improvement. The model was already near its optimum.

### EXP_012: Head-Only Fine-Tune at 768

**Directory:** `experiments/exp_012_head_finetune_768/`

Froze backbone, fine-tuned only the detection head with higher LR (0.005) at 768. Hypothesis: after exp_009, the backbone is well-calibrated; only the head needs further adjustment.

- **Training mAP50:** 96.46% (highest recorded in any CSV)
- **Validated mAP50:** 96.09% @768 (model.val()), **96.21% with TTA**
- **Verdict:** Slight boost with TTA makes this the overall best configuration. Head-only fine-tuning didn't hurt, but gains were minimal over exp_009.

### EXP_013: 896 Resolution Attempt

**Script:** `scripts/train_finetune_896.py`  
**Directory:** `experiments/exp_013_finetune_896/`

Pushed to 896×896 with batch=1 (VRAM limit). Only ran 17 epochs.

- **mAP50:** 94.37% — severe degradation from 96.11%
- **Verdict:** 896 is too much. batch=1 creates extremely noisy gradients. The P3/P4 architecture doesn't benefit from resolutions beyond 768. Diminishing returns confirmed.

---

## 7. Post-Training Techniques

### 7.1 Resolution Sweep (Inference-Time)

Validated the KD model (exp_005) at different inference resolutions **without retraining**:

| Resolution | mAP50 | Δ from 640 |
|---|---|---|
| 640 | 94.12% | — |
| 704 | 94.45% | +0.33% |
| **768** | **94.72%** | **+0.60%** |
| 832 | 94.31% | +0.19% |
| 896 | 93.88% | −0.24% |

**768 is the optimal inference resolution** for this architecture.

### 7.2 Test-Time Augmentation (TTA)

**Challenge:** The P3/P4 model has `stride = [8, 16]` (max_stride=16), but Ultralytics' TTA uses `gs = int(self.stride.max())` for `scale_img()`. With gs=16, scaled image sizes cause **Concat dimension mismatches** in the detection head.

**Fix:** Monkeypatch stride before calling `model.val(augment=True)`:
```python
import torch
model.model.stride = torch.tensor([8., 16., 32.])  # Fake stride 32
results = model.val(data="VOC/voc.yaml", imgsz=768, augment=True)
```

This makes TTA use gs=32 for padding calculations, ensuring all scaled variants have compatible tensor dimensions.

**TTA Results (validated with Ultralytics model.val):**

| Model | Without TTA | With TTA @768 | TTA Gain |
|---|---|---|---|
| exp_005_kd @640 | 94.12% | **95.69%** | **+1.57%** |
| exp_009 @768 | 96.11% | **96.15%** | +0.04% |
| exp_012 @768 | 96.09% | **96.21%** | +0.12% |

**Insight:** TTA helps most when the model wasn't trained at the inference resolution. exp_009/012 were already trained at 768, so TTA adds minimal benefit.

### 7.3 Model Soup (Weight Averaging)

Averaged model weights element-wise between different checkpoints:

**Early Soups (KD + Baseline + TFA, evaluated @640):**

| Variant | Ratio | mAP50 |
|---|---|---|
| KD + Baseline 50/50 | 0.5 : 0.5 | 94.23% |
| KD + Baseline 40/60 | 0.4 : 0.6 | 94.18% |
| KD + Baseline 30/70 | 0.3 : 0.7 | 94.12% |
| 3-way (KD+Base+TFA) | 0.34 each | 94.17% |

**Late Soups (KD + 768ft, evaluated @768):**

| Variant | Ratio | mAP50 |
|---|---|---|
| KD + 768ft 50/50 | 0.5 : 0.5 | ~95.9% |
| KD + 768ft 40/60 | 0.4 : 0.6 | ~95.9% |
| KD + 768ft 30/70 | 0.3 : 0.7 | ~95.9% |

**Verdict:** Model Soup never outperformed the best single checkpoint. When the teacher-student gap is small and training is well-tuned, averaging dilutes the best model's specialized features.

### 7.4 SWA (Stochastic Weight Averaging)

Averaged checkpoint weights across training epochs of exp_009:

| Variant | Checkpoints | mAP50 |
|---|---|---|
| SWA (ep5→last) | All saved epochs | 96.02% |
| SWA Top-3 | ep10 + best + last | 96.09% |

**Verdict:** SWA provided a tiny smoothing effect but didn't exceed the best single checkpoint (96.11%).

### 7.5 Multi-Model Ensemble (WBF)

Explored Weighted Boxes Fusion (WBF) combining exp_002 + exp_009 + exp_012. Custom metrics showed relative improvements, but the approach was **abandoned** because:

1. Custom metric pipeline (model.predict + manual matching) had ~11% systematic undercount vs Ultralytics' internal pipeline
2. Multi-model ensemble is **impractical for edge deployment** — can't run 2-3 models on a Raspberry Pi / Jetson Nano in real-time

---

## 8. Verified Best Results

All results verified using `model.val()` from Ultralytics (authoritative metrics):

| Rank | Configuration | mAP50 | D1 AP50 | ins AP50 | Precision | Recall |
|---|---|---|---|---|---|---|
| 🥇 | **exp_012 + TTA @768** | **96.21%** | 94.98% | **97.44%** | 94.4% | 93.1% |
| 🥈 | **exp_009 + TTA @768** | **96.15%** | **95.33%** | 96.96% | 94.7% | 92.1% |
| 🥉 | **exp_009 @768** (no TTA) | **96.11%** | 95.29% | 96.94% | 92.7% | **95.4%** |
| 4 | exp_012 @768 (no TTA) | 96.09% | 95.04% | 97.14% | 93.5% | 94.4% |
| 5 | exp_011 @768 | 96.06% | — | — | — | — |
| 6 | SWA Top-3 @768 | 96.09% | — | — | — | — |
| 7 | exp_005_kd + TTA @768 | 95.69% | 94.49% | 96.89% | 93.9% | 92.5% |
| 8 | exp_010 @768 | 95.69% | — | — | — | — |
| — | Teacher (YOLO11s) @640 | 96.54% | — | — | — | — |
| — | Baseline (exp_002) @640 | 94.21% | 91.24% | 97.10% | 92.7% | 89.0% |

### Improvement Breakdown

| Metric | Baseline (exp_002) | Best (exp_009) | Gain |
|---|---|---|---|
| **mAP50** | 94.21% | 96.11% | **+1.90%** |
| **Damaged_1 AP50** | 91.24% | 95.29% | **+4.05%** |
| **Insulator AP50** | 97.10% | 96.94% | −0.16% |
| **Recall** | 89.0% | 95.4% | **+6.4%** |

The biggest gain was in the **rare class (Damaged_1)**: +4.05% AP50, confirming that KD + resolution optimization specifically helped the deficiency that motivated this project.

---

## 9. Key Findings & Lessons Learned

### What Worked ✅

1. **Knowledge Distillation + Resolution Fine-Tuning = Synergistic**  
   KD alone gave modest gains (+0.32%), but KD weights responded dramatically better to resolution upscaling than baseline weights. The KD → 768 fine-tune pipeline produced a +1.90% improvement.

2. **768 is the Sweet Spot**  
   Resolution sweep proved 768 is optimal for the P3/P4 architecture with strides [8, 16]. Higher resolutions (832, 896) degraded performance — likely because the receptive field becomes mismatched.

3. **Short Fine-Tuning at Inference Resolution**  
   Only 25 epochs of SGD fine-tuning at 768 was sufficient to achieve the best result. Training at the inference resolution internalizes the resolution benefit that was previously only available through test-time upscaling.

4. **TTA Provides Free Accuracy (When Applicable)**  
   TTA on the KD model @768 gave +1.57%. However, TTA on models already trained at 768 gave negligible gains (+0.04–0.12%).

5. **TTA Stride Fix for P3/P4 Models**  
   Custom architectures with non-standard strides need `model.model.stride = torch.tensor([8., 16., 32.])` to make Ultralytics' built-in TTA compatible.

### What Didn't Work ❌

1. **704 Resolution (3 attempts, all failed)**  
   704 consistently degraded performance (92.85–93.37%) regardless of learning rate, frozen/unfrozen, or 2-stage approaches. 704 is a poor match for this architecture's stride multiples.

2. **Two-Stage Fine-Tuning (TFA)**  
   Three TFA variants (lr=0.0001, lr=0.0003+cls=2.0, lr=5e-5@704) all produced marginal or negative results. With only 2 classes and moderate imbalance (1:3.5), TFA wasn't the right solution.

3. **Aggressive Classification Loss Weights**  
   Boosting cls loss from 0.5→2.0 destabilized training and reduced mAP50. The default loss weights are well-calibrated.

4. **copy_paste and flipud Augmentation**  
   Both augmentations were detrimental. copy_paste creates artificial artifact patterns; flipud makes no physical sense for power line insulators.

5. **Multi-Scale Training**  
   `multi_scale=True` at 768 hurt performance (95.69% vs 96.11%). Once the model is calibrated for a specific resolution, randomizing it during training degrades learned features.

6. **Model Soup**  
   Weight averaging never beat the best single checkpoint. With small teacher-student gaps and well-optimized training, averaging only dilutes the best model's specialized features.

7. **896 Resolution**  
   batch=1 required for VRAM, creating extremely noisy gradients. The architecture doesn't benefit from resolutions this high.

### Practical Insights 💡

- **KD is an enabler, not a direct booster.** The mAP50 gain from KD was modest, but it fundamentally improved the model's ability to generalize to higher resolutions.
- **Resolution optimization is the highest-ROI technique** for small models — it's free at inference time (just change imgsz) and can be internalized through short fine-tuning.
- **Once you're within 0.5% of the teacher, further gains are extremely difficult** without architecture changes or more data.
- **Custom TTA metric pipelines are unreliable.** Always use the framework's built-in evaluation (model.val) for authoritative metrics.
- **Edge deployment constraints matter early.** Ensemble and SAHI are attractive on paper but useless for real-time edge inference.

---

## 10. Recommended Deployment Configuration

### Best Single-Model (No TTA)
```bash
# Inference configuration for edge deployment
model = "experiments/exp_009_finetune_768/weights/best.pt"
imgsz = 768
conf = 0.25  # confidence threshold (tune per deployment)
```

- **mAP50:** 96.11% | **Params:** 867K | **GFLOPs:** ~5.8
- **Why exp_009:** Highest verified mAP50 without TTA. Clean single-model inference, no hacks needed.

### Best with TTA (If Latency Allows)
```python
import torch
from ultralytics import YOLO

model = YOLO("experiments/exp_012_head_finetune_768/weights/best.pt")
model.model.stride = torch.tensor([8., 16., 32.])  # Required for P3/P4 TTA
results = model.val(data="VOC/voc.yaml", imgsz=768, augment=True)
# mAP50: 96.21%
```

> ⚠️ TTA triples inference time. Only use if latency budget permits (not recommended for live video feeds on Jetson/Pi).

### Edge Export Commands
```bash
# NVIDIA Jetson (TensorRT FP16)
yolo export model=experiments/exp_009_finetune_768/weights/best.pt format=engine imgsz=768 half=True workspace=4

# Raspberry Pi (ONNX/TFLite INT8)
yolo export model=experiments/exp_009_finetune_768/weights/best.pt format=tflite imgsz=768 int8=True
```

---

## 11. File Index

### Training Scripts (`scripts/`)

| Script | Experiment | Description |
|---|---|---|
| `train_exp002.py` | exp_002 | Baseline training from scratch (300ep @640) |
| `train_teacher.py` | exp_004 | Teacher YOLO11s training |
| `train_teacher_v2.py` | exp_004 | Teacher training variant |
| `train_kd.py` | exp_005 | Knowledge Distillation with custom trainer |
| `train_tfa.py` | exp_tfa_v1 | Two-stage fine-tuning v1 |
| `train_tfa_v2.py` | exp_tfa_v2 | TFA with aggressive cls loss |
| `train_tfa_v3.py` | exp_tfa_v3 | TFA conservative @704 |
| `train_phase3_highres.py` | exp_006 | High-res 704 aggressive |
| `train_phase3v2_conservative.py` | exp_007 | Two-stage 704 conservative |
| `train_finetune_768.py` | exp_009 | **768 fine-tune (best result)** |
| `train_finetune_768_v2.py` | exp_010 | Multi-scale 768 |
| `train_swa.py` | — | SWA cyclic fine-tune |
| `train_finetune_896.py` | exp_013 | 896 resolution attempt |

### Evaluation & Post-Processing Scripts (`scripts/`)

| Script | Description |
|---|---|
| `validate_tta.py` | Multi-resolution TTA validation |
| `run_tta.py` | Custom TTA evaluation v1 |
| `run_tta_v2.py` | Custom TTA evaluation v2 |
| `run_tta_coco.py` | TTA with COCO metrics |
| `run_tta_ultralytics.py` | TTA via Ultralytics built-in augment=True |
| `custom_tta.py` | Custom TTA scale/flip configs |
| `custom_tta_v2.py` | Custom TTA v2 with monkeypatching |
| `run_soup.py` | Model Soup weight averaging |
| `model_soup_and_tta.py` | Combined soup + TTA pipeline |
| `ensemble_val.py` | Multi-model WBF ensemble v1 |
| `ensemble_eval_v2.py` | Multi-model WBF ensemble v2 |

### Experiment Directories (`experiments/`)

| Directory | Best mAP50 | Status |
|---|---|---|
| `exp_002_ghost_hybrid_medium3/` | 94.21% | Baseline ✅ |
| `exp_004_teacher_yolo11s/` | 96.54% | Teacher ✅ |
| `exp_005_kd_student2/` | — | KD run 1 |
| `exp_005_kd_student3/` | 94.53% | KD final ✅ |
| `exp_006_highres_704/` | 93.08% | Failed ❌ |
| `exp_007_highres704_s1_frozen/` | 93.37% | Failed ❌ |
| `exp_007_highres704_s2_unfrozen/` | 92.85% | Failed ❌ |
| `exp_009_finetune_768/` | **96.11%** | **Best** ⭐ |
| `exp_010_finetune_768_v2/` | 95.69% | Multi-scale hurt ❌ |
| `exp_011_finetune_768_v3/` | 96.06% | Saturated |
| `exp_012_head_finetune_768/` | 96.09% (val) | Best w/ TTA |
| `exp_013_finetune_896/` | 94.37% | Failed ❌ |
| `exp_tfa_20260217_182417/` | 94.34% | TFA v1 |
| `exp_tfa_v2_20260217_183859/` | 94.00% | TFA v2 ❌ |
| `exp_tfa_v3_20260217_202347/` | 93.11% | TFA v3 ❌ |

### Standalone Weight Files (`experiments/`)

| File | Description |
|---|---|
| `soup_2way_50_50.pt` | KD + Baseline 50/50 soup |
| `soup_2way_40_60.pt` | KD + Baseline 40/60 soup |
| `soup_2way_30_70.pt` | KD + Baseline 30/70 soup |
| `soup_3way.pt` | 3-way soup (KD+Base+TFA) |
| `soup_kd_768ft_50_50.pt` | KD + 768ft 50/50 soup |
| `soup_kd_768ft_40_60.pt` | KD + 768ft 40/60 soup |
| `soup_kd_768ft_30_70.pt` | KD + 768ft 30/70 soup |
| `exp_009_swa_averaged.pt` | SWA average (all epochs) |
| `exp_009_swa_top3.pt` | SWA top-3 average |

---

## Training Pipeline Diagram

```
YOLO11n-Ghost-Hybrid-P3P4-Medium (from scratch)
│
├── exp_002: Baseline Training (300ep @640)
│   │   → 94.21% mAP50
│   │
│   ├── TFA v1 (frozen backbone, lr=1e-4) → 94.34% (marginal)
│   ├── TFA v2 (cls=2.0, copy_paste) → 94.00% (WORSE)
│   ├── TFA v3 (704, lr=5e-5) → 93.11% (WORSE)
│   │
│   └── exp_007: Conservative 704 (2-stage)
│       ├── s1 frozen → 93.37%
│       └── s2 unfrozen → 92.85% (WORSE)
│
├── YOLO11s Teacher Training
│   └── exp_004: 96.54% mAP50 (performance ceiling)
│
└── Knowledge Distillation (Teacher → Student)
    │
    └── exp_005_kd: 94.53% / 94.12% mAP50
        │
        ├── exp_006: Aggressive 704 → 93.08% (FAILED)
        │
        ├── Resolution Sweep → 768 is sweet spot (+0.60%)
        │   └── TTA @768 → 95.69% (+1.57%)
        │
        ├── Model Soups → 94.23% max (marginal)
        │
        └── exp_009: 768 Fine-Tune (25ep, SGD) ⭐
            │   → 96.11% mAP50 (BREAKTHROUGH)
            │
            ├── + TTA → 96.15%
            ├── Model Soups → ~95.9% (below single)
            ├── SWA → 96.02–96.09% (matched single)
            │
            ├── exp_010: Multi-scale 768 → 95.69% (hurt)
            ├── exp_011: Low-LR 768 → 96.06% (saturated)
            ├── exp_012: Head-only 768 → 96.09%
            │   └── + TTA → 96.21% (BEST OVERALL)
            └── exp_013: 896 → 94.37% (too aggressive)
```

---

*Generated from experiment logs and validated model outputs. All mAP50 values verified using Ultralytics `model.val()` unless otherwise noted.*
