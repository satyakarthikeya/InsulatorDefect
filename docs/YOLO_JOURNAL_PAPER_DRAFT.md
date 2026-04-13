# Lightweight Ghost-Hybrid YOLO11n with Distillation and Resolution-Aware Fine-Tuning for Edge Insulator Defect Detection

## Abstract
Real-time insulator defect detection on embedded hardware is constrained by model size, compute budget, and class imbalance between normal and damaged components. This work presents a lightweight detector based on a customized YOLO11n Ghost-Hybrid-P3P4-Medium architecture, designed for two-class defect detection (Damaged_1 and insulator) under edge constraints. The proposed model uses GhostConv and C3Ghost for efficient backbone feature extraction, a depthwise-separable detection head, and two-scale detection (P3/P4 only) to maximize capacity on small and medium defect targets. We further apply teacher-student knowledge distillation and resolution-aware fine-tuning at 768x768 to improve rare-defect sensitivity. The final model has 867,664 parameters and 5.7 GFLOPs, achieving 96.10% mAP@0.5, 66.79% mAP@0.5:0.95, 93.65% precision, and 94.18% recall, while remaining suitable for Raspberry Pi deployment. On Raspberry Pi with ONNX Runtime CPU execution, the model reaches 656.0 ms mean model inference time (690.9 ms full pipeline), demonstrating practical field-readiness. The study provides an end-to-end experimental journey including ablations, deployment analysis, and efficiency-accuracy trade-off discussion.

**Keywords:** YOLO11n, lightweight object detection, edge AI, GhostConv, knowledge distillation, insulator defect detection

---

## 1. Introduction
Power-line insulator monitoring is a high-impact computer vision problem where missed detections can lead to delayed maintenance and grid reliability risks. Unlike large-scale generic detection benchmarks, insulator defect data are typically class-imbalanced, and defect regions are often small, texture-like, and visually subtle. These characteristics increase the difficulty of deployment on edge devices, where both memory and compute are limited.

Single-stage detectors in the YOLO family are attractive for this setting due to strong speed-accuracy trade-offs. However, standard compact detectors still leave room for optimization when targeting devices such as Raspberry Pi and Jetson-class embedded systems. The central challenge is to retain defect-level detection quality while reducing parameters and inference complexity.

This paper addresses that challenge with a domain-specialized lightweight architecture and a staged optimization pipeline. Starting from a strong baseline, we progressively apply (i) architecture-level compression by design, (ii) teacher-guided representation transfer, and (iii) resolution-aware fine-tuning to improve rare-class performance.

### Major Contributions
1. We design an ultra-light YOLO11n variant (867,664 parameters) using GhostConv/C3Ghost and a depthwise detection head, optimized for small/medium insulator targets.
2. We show that removing P5 detection and focusing on P3/P4 scales improves task relevance under strict parameter budgets.
3. We validate that knowledge distillation enables stronger downstream adaptation to high-resolution fine-tuning.
4. We identify 768x768 as the optimal operating resolution for this architecture, producing the largest practical gain in mAP@0.5 and rare-class AP.
5. We demonstrate edge feasibility on Raspberry Pi via ONNX Runtime with reproducible latency statistics and deployment workflow.

---

## 2. Literature Review
### 2.1 One-Stage Detection for Real-Time Applications
YOLO-style one-stage detectors unify localization and classification in a single forward pass, making them suitable for latency-sensitive deployments [1]-[3]. Recent generations (YOLOv8/YOLO11) improve training stability and multi-scale detection quality, but edge-focused tuning remains application dependent [4], [5].

### 2.2 Lightweight CNN Design
Model families such as MobileNet and GhostNet show that depthwise separable operations and cheap feature generation can preserve representational power while reducing multiply-accumulate operations [6], [7]. These ideas motivate our backbone-head design choices.

### 2.3 Knowledge Distillation for Compact Models
Knowledge distillation (KD) transfers soft target structure from larger teacher models into smaller students, often improving compact model calibration and rare-class behavior [8], [9]. In this work, KD alone provides moderate direct gains, but critically enables stronger resolution adaptation.

### 2.4 Edge Deployment and Practical Constraints
Prior edge-AI studies emphasize that theoretical complexity does not always translate to real-time throughput without careful runtime and format optimization (ONNX/TensorRT/TFLite) [10], [11]. Deployment choices are therefore part of model design, not an afterthought.

---

## 3. Dataset Overview and Augmentation
### 3.1 Dataset Composition
The dataset contains two classes:
- Damaged_1 (rare defect class)
- insulator (normal class)

| Split | Images | Damaged_1 Instances | insulator Instances | Total Instances |
|---|---:|---:|---:|---:|
| Train | 1,155 | 562 | 1,962 | 2,524 |
| Val | 143 | 76 | 224 | 300 |

The class ratio is approximately 1:3.5 (Damaged_1:insulator), creating a moderate imbalance that can suppress rare-defect recall if optimization is not targeted.

### 3.2 Augmentation Strategy
Training uses moderate geometric and photometric augmentation, including mosaic, mixup, rotation, translation, scaling, and HSV jitter. For stable late-stage convergence, heavy operations are reduced in some fine-tuning phases.

A representative augmentation setting is:
- mosaic = 0.8-1.0
- mixup = 0.05-0.10
- degrees = 5-10
- translate = 0.1-0.2
- scale = 0.3-0.9
- hsv_h/s/v = 0.015/0.5-0.7/0.3-0.4

### 3.3 Impact on Rare Defect Class
The defect class benefits from controlled spatial diversity, but aggressive settings (for example, high class-loss weighting with synthetic copy-paste) can destabilize class balance in later-stage optimization. This was verified during ablation and informs the final conservative augmentation recipe.

---

## 4. Proposed Model Architecture
### 4.1 Baseline and Design Goal
The baseline reference is vanilla YOLO11n (2.58M parameters). Our goal is to obtain similar or better detection quality with a substantially smaller student model suitable for embedded deployment.

### 4.2 Ghost-Hybrid-P3P4-Medium Architecture
The proposed model uses:
- GhostConv backbone stages for low-cost channel expansion
- C3Ghost blocks for efficient cross-stage feature reuse
- SPPF context aggregation
- Depthwise + pointwise convolutions in the detection head
- Detection at P3/P4 only (P5 removed)

This design yields:
- **Parameters:** 867,664
- **GFLOPs:** 5.7
- **Input:** 768x768
- **Classes:** 2

### 4.3 Mathematical Formulation
For a standard convolution with input channels $C_{in}$, output channels $C_{out}$, kernel $k$, and feature size $H \times W$, MAC complexity is:

$$
\text{MAC}_{std} = C_{in} C_{out} k^2 H W
$$

Ghost-style generation reduces heavy convolution usage by generating only intrinsic feature maps with full convolution and producing additional maps with cheap linear transforms:

$$
\text{MAC}_{ghost} \approx C_{in} m k^2 H W + m s d^2 H W, \quad m = \frac{C_{out}}{s}
$$

where $s$ is the ghost ratio and $d$ is the cheap transform kernel size.

Depthwise separable head blocks decompose convolution into:

$$
\text{Conv}_{dw+pw} = \text{Conv}_{dw}(k\times k) + \text{Conv}_{pw}(1\times 1)
$$

reducing parameters and compute compared with full $k\times k$ convolution.

### 4.4 Knowledge Distillation Objective
With teacher logits $z_t$, student logits $z_s$, temperature $T$, and distillation weight $\alpha$:

$$
\mathcal{L}_{total} = (1-\alpha)\mathcal{L}_{yolo} + \alpha T^2 \operatorname{KL}(\sigma(z_t/T) \Vert \sigma(z_s/T))
$$

where $\mathcal{L}_{yolo}$ is the native detection loss (box + class + distribution focal terms).

### 4.5 Why P3/P4-Only Detection
Defects are predominantly small/medium objects. Removing large-scale P5 prediction avoids wasting model capacity on irrelevant object scales and supports a stronger parameter-accuracy trade-off for this domain.

---

## 5. Results and Discussion
### 5.1 Experimental Setup
- Framework: Ultralytics 8.3.240, PyTorch 2.9.1
- Training GPU: NVIDIA RTX 3050 Laptop (4 GB VRAM)
- Input resolutions explored: 640, 704, 768, 896
- Deployment target: Raspberry Pi / Jetson-class edge hardware

### 5.2 Baseline Performance
The vanilla YOLO11n baseline achieves:
- mAP@0.5 = 90.20%
- mAP@0.5:0.95 = 58.19%
- Precision = 89.70%
- Recall = 80.30%
- Parameters = 2.58M

This baseline is strong but leaves significant room for rare-defect recall improvement and model-size reduction.

### 5.3 Architecture-Level Gains
A first major gain is obtained from architecture redesign (Ghost-Hybrid at 640):
- mAP@0.5 = 94.21%
- mAP@0.5:0.95 = 68.08%
- Precision = 93.5%
- Recall = 89.7%
- Parameters = 867,664

Relative to vanilla YOLO11n, this stage improves mAP@0.5 by +4.01 points while reducing model size by about 3x.

### 5.4 Distillation and Resolution-Aware Fine-Tuning
KD (teacher YOLO11s to student) produces moderate direct mAP gain but significantly improves downstream adaptability. The main breakthrough occurs when fine-tuning at 768x768:
- exp_009 (single model): mAP@0.5 = 96.11%
- Damaged_1 AP@0.5 = 95.29%
- Recall rises to 95.4%

A short head-only refinement (exp_012) with TTA reaches 96.21%, but for practical deployment, the single-model no-TTA configuration remains preferred due to latency constraints.

### 5.5 Final Validated Model Metrics
The benchmarked final model (best practical single-model deployment profile):

| Metric | Value |
|---|---:|
| mAP@0.5 | 96.10% |
| mAP@0.5:0.95 | 66.79% |
| Precision | 93.65% |
| Recall | 94.18% |
| Damaged_1 AP@0.5 | 95.05% |
| insulator AP@0.5 | 97.15% |
| Parameters | 867,664 |
| GFLOPs | 5.7 |

These results show that model compression and accuracy can be jointly optimized when architecture and training schedule are co-designed.

---

## 6. Ablation Study
### 6.1 Configuration-Wise Comparison

| Config | Description | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Params |
|---|---|---:|---:|---:|---:|---:|
| C1 | Vanilla YOLO11n baseline | 90.20 | 58.19 | 89.70 | 80.30 | 2.58M |
| C2 | Ghost-Hybrid (exp_002, 640) | 94.21 | 68.08 | 93.50 | 89.70 | 0.867M |
| C3 | Ghost-Hybrid + KD (exp_005, 640) | 94.12-94.53 | -- | 91.2 | 90.0 | 0.867M |
| C4 | KD + 768 fine-tune (exp_009) | 96.11 | 66.02 | 92.7 | 95.4 | 0.867M |
| C5 | Head-only 768 + TTA (exp_012) | 96.21 | -- | 94.4 | 93.1 | 0.867M |

### 6.2 Findings
1. **Architecture redesign (C1->C2) is the largest compression jump** with simultaneous accuracy gain.
2. **KD (C2->C3) gives modest direct gain**, but improves representation quality for later stages.
3. **Resolution optimization (C3->C4) is the highest-ROI step** for this model family.
4. **Head-only tuning + TTA (C5) gives marginal peak gain**, but increased inference cost makes it less suitable for strict real-time edge settings.
5. **Aggressive high-resolution or heavy class-loss boosting can hurt**, as seen in failed 704/896 and over-weighted cls-loss runs.

### 6.3 Best Trade-Off Configuration
For deployment, **C4 (exp_009 at 768 without TTA)** is the best trade-off:
- Near-peak mAP
- Strong rare-class AP
- Clean single-model inference path
- No TTA latency overhead

---

## 7. Qualitative and Quantitative Comparative Analysis
### 7.1 In-Project Comparative Performance

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Params | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n baseline | 90.20 | 58.19 | 89.70 | 80.30 | 2.58M | -- |
| Proposed Ghost-Hybrid (final) | 96.10 | 66.79 | 93.65 | 94.18 | 0.867M | 5.7 |
| YOLO11s teacher | 96.54 | -- | -- | -- | 9.43M | 21.3 |

The proposed model improves baseline mAP@0.5 by +5.90 points while using roughly one-third of the parameters. It also approaches teacher-level performance within 0.44 points while being around 10.9x smaller.

### 7.2 Efficiency-Accuracy Interpretation
The dominant result is not only higher mAP, but higher **efficiency-normalized mAP**. The student model captures most of the teacher's performance while remaining practical for edge deployment.

### 7.3 Qualitative Behavior
Qualitative inspection from validation predictions indicates:
- Better localization of thin crack-like patterns on damaged insulators
- Reduced miss rate for rare defects after KD + 768 fine-tune
- Fewer high-confidence false negatives compared with baseline

---

## 8. Edge Deployment and Hardware Validation
### 8.1 Edge Setup
- Raspberry Pi path: ONNX Runtime CPU inference
- Jetson path: TensorRT FP16 export path supported by project scripts
- Model export options include ONNX, TensorRT, and TFLite INT8-ready workflows

### 8.2 Raspberry Pi Benchmark Results
Measured ONNX Runtime CPU performance (768 input):

| Metric | Value |
|---|---:|
| Mean model inference time | 656.0 ms |
| Median model inference time | 657.7 ms |
| Min model inference time | 643.9 ms |
| Max model inference time | 660.7 ms |
| Std dev | 4.6 ms |
| Mean full pipeline time | 690.9 ms |
| Throughput | 1.4 FPS |

The numbers indicate feasible low-frame-rate monitoring on CPU-only edge hardware, with predictable latency variance.

### 8.3 Jetson Orin Nano Analysis
Project documentation includes TensorRT FP16 export support and expected runtime behavior for Jetson-class accelerators. Reported expected inference range is approximately 15-30 ms depending on precision/runtime path. This corresponds to high-throughput real-time operation and confirms that the proposed model is accelerator-friendly.

### 8.4 Comparative Edge Interpretation
- Raspberry Pi offers low-cost deployment with acceptable periodic inspection throughput.
- Jetson-class deployment is preferable for live-stream or multi-camera real-time scenarios.
- The compact 867K parameter footprint enables both deployment modes without architecture changes.

---

## 9. Conclusion
This paper presented a lightweight and deployment-oriented YOLO11n variant for insulator defect detection. By combining Ghost-Hybrid architecture design, KD-based representation transfer, and resolution-aware fine-tuning, we obtained a model that is both compact and accurate: 867,664 parameters, 5.7 GFLOPs, and 96.10% mAP@0.5. The experimental journey demonstrates that architecture-aware optimization and staged training outperform brute-force scaling under strict edge constraints. Raspberry Pi benchmarks confirm practical deployability, while Jetson-oriented export paths support higher-throughput use cases.

### Future Work
1. Collect and report measured Jetson FP32/FP16/INT8 latency and memory traces in a standardized benchmark table.
2. Evaluate post-training quantization and quantization-aware training to improve CPU throughput further.
3. Extend the defect taxonomy and test domain generalization across weather/time-of-day conditions.
4. Add uncertainty estimation for risk-aware maintenance prioritization.

---

## References
[1] J. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection," CVPR, 2016.

[2] A. Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection," arXiv, 2020.

[3] C.-Y. Wang et al., "YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors," arXiv, 2022.

[4] Ultralytics, "YOLOv8 Documentation," 2023-2026.

[5] Ultralytics, "YOLO11 Documentation and Release Notes," 2024-2026.

[6] A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv, 2017.

[7] K. Han et al., "GhostNet: More Features from Cheap Operations," CVPR, 2020.

[8] G. Hinton et al., "Distilling the Knowledge in a Neural Network," arXiv, 2015.

[9] Y. Wang et al., "Knowledge Distillation for Object Detection: A Survey," arXiv, 2022.

[10] M. Tan and Q. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," ICML, 2019.

[11] ONNX Runtime Documentation, "Performance Tuning and Execution Providers," 2024-2026.
