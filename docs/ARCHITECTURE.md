# Architecture: YOLO11n-Ghost-Hybrid-P3P4-Medium

**Team:** P P Satya Karthikeya · B Karthikeya · M Karthik Reddy · P Rohit

> Custom ultra-lightweight object detection architecture for insulator defect detection.
> **867K parameters · 5.7 GFLOPs · 768×768 input · 2-class detection**

---

## Overview

The model is a heavily customised YOLO11n variant built around three core ideas:

| Design Choice | What it replaces | Why |
|---|---|---|
| **GhostConv backbone** | Standard Conv backbone | ~50% fewer backbone parameters |
| **DWConv-only head** | Standard Conv head | ~9× parameter reduction in head |
| **P3 + P4 detection only** (no P5) | 3-scale detection (P3/P4/P5) | All capacity focused on small/medium defects |

---

## Full Architecture Diagram

```
INPUT IMAGE  768 × 768 × 3
      │
      ▼
╔══════════════════════════════════════════════════════════╗
║                      BACKBONE                            ║
║         GhostConv + C3Ghost (Feature Extraction)         ║
╚══════════════════════════════════════════════════════════╝
      │
      ▼
  [0] Conv(32, 3×3, s=2)           → 384×384×32     Stem — standard conv for initial edge extraction
      │
      ▼
  [1] GhostConv(64, 3×3, s=2)      → 192×192×64     P2 — first ghost stage, halves spatial size
      │
      ▼
  [2] C3Ghost(64) ×2               → 192×192×64     Feature extraction block at P2 scale
      │
      ▼
  [3] GhostConv(128, 3×3, s=2)     → 96×96×128      P3/8 — stride-8 features (small objects) ─────────────────┐
      │                                                                                                         │
      ▼                                                                                                         │
  [4] C3Ghost(128) ×3              → 96×96×128      ★ P3 feature map (saved for FPN)  ──────────────────────┐  │
      │                                                                                                      │  │
      ▼                                                                                                      │  │
  [5] GhostConv(160, 3×3, s=2)     → 48×48×160      P4/16 — stride-16 features (medium objects) ───────┐   │  │
      │                                                                                                   │   │  │
      ▼                                                                                                   │   │  │
  [6] C3Ghost(160) ×3              → 48×48×160      ★ P4 feature map (saved for FPN)  ─────────────┐   │   │  │
      │                                                                                              │   │   │  │
      ▼                                                                                              │   │   │  │
  [7] GhostConv(160, 3×3, s=2)     → 24×24×160      P5/32 — context/global features               │   │   │  │
      │                                                                                              │   │   │  │
      ▼                                                                                              │   │   │  │
  [8] C3Ghost(160) ×2              → 24×24×160      Deeper context extraction                      │   │   │  │
      │                                                                                              │   │   │  │
      ▼                                                                                              │   │   │  │
  [9] SPPF(160, k=5)               → 24×24×160      Multi-scale pooling (3× pool at 5,9,13)       │   │   │  │
      │                                                                                              │   │   │  │
╔═════════════════════════════════════════════════════════╗                                         │   │   │  │
║                       HEAD                              ║                                         │   │   │  │
║     DWConv FPN — Top-down + Bottom-up feature fusion    ║                                         │   │   │  │
╚═════════════════════════════════════════════════════════╝                                         │   │   │  │
      │                                                                                              │   │   │  │
      ▼  ── TOP-DOWN PATH (P5 → P4 → P3) ──                                                         │   │   │  │
 [10] Upsample(×2)                 → 48×48×160      Upsample P5 context up to P4 scale             │   │   │  │
      │                                                                                              │   │   │  │
 [11] Concat([10, 6])              → 48×48×320      Merge upsampled P5 with P4 backbone ←──────────┘   │   │  │
      │                                                                                                  │   │  │
 [12] DWConv(80, 3×3)             → 48×48×80       Depthwise: channel compress + fuse                  │   │  │
      │                                                                                                  │   │  │
 [13] Conv(80, 1×1)               → 48×48×80       Pointwise: mix channels                             │   │  │
      │                                                                                          ┌───────┘   │  │
      ▼                                                                                          │           │  │
 [14] Upsample(×2)                 → 96×96×80       Upsample P4 head up to P3 scale             │           │  │
      │                                                                                          │           │  │
 [15] Concat([14, 4])              → 96×96×208      Merge with P3 backbone ←─────────────────────────────────┘  │
      │                                                                                                           │
 [16] DWConv(64, 3×3)             → 96×96×64       Depthwise: compress to P3 channels                          │
      │                                                                                                           │
 [17] Conv(64, 1×1)               → 96×96×64       ★ P3 Detection Features (small objects) ←────────────────────┘
      │
      │   ── BOTTOM-UP PATH (P3 → P4) ──
      ▼
 [18] DWConv(64, 3×3, s=2)        → 48×48×64       Downsample P3 back to P4 scale
      │
 [19] Conv(80, 1×1)               → 48×48×80       Channel expand to match P4 head width
      │
 [20] Concat([19, 13])            → 48×48×160      Re-merge with P4 top-down features ←── [13]
      │
 [21] DWConv(80, 3×3)             → 48×48×80       Depthwise fusion
      │
 [22] Conv(80, 1×1)               → 48×48×80       ★ P4 Detection Features (medium objects)
      │
╔══════════════════════════════════╗
║         DETECTION HEAD           ║
║   Detect([17, 22], nc=2)         ║
╚══════════════════════════════════╝
      │              │
      ▼              ▼
  P3 (96×96)    P4 (48×48)
  Small objs    Medium objs
  Damaged_1     insulators
```

---

## Block-by-Block Explanation

### Backbone Blocks

#### `Conv` — Standard Convolution (Layer 0 only)
```
Input → Conv2d(k, s) → BatchNorm → SiLU activation
```
Used only at the stem (first layer) where standard convolution is needed to capture raw image edges and textures before entering the ghost pipeline.

---

#### `GhostConv` — Ghost Convolution (Layers 1, 3, 5, 7)
```
Input ──► Primary Conv (half channels) ──► Primary features
                                               │
                          Cheap linear ops ────┘
                          (depthwise, 1×1)
                               │
                          Ghost features
                               │
                          Concat ──► Full output
```
**Why:** A standard conv generating N feature maps is replaced by a conv generating N/2 maps + N/2 "ghost" copies via cheap depthwise operations. This achieves ~50% compute/parameter savings with minimal accuracy loss.

- Used at every stride: P2 (×2 downsample), P3 (×4), P4 (×8), P5 (×16)
- Channel progression: 64 → 128 → 160 → 160

---

#### `C3Ghost` — Cross-Stage Partial with Ghost Bottleneck (Layers 2, 4, 6, 8)
```
Input ──► Split
            │                    │
            ▼                    ▼
      GhostBottleneck ×N    Identity skip
            │                    │
            └──────── Concat ────┘
                          │
                       Conv(1×1)
                          │
                       Output
```
**Why:** Cross-stage partial (CSP) design passes part of the input directly to the output (skip path), so only half the channels go through the expensive bottleneck. Combined with Ghost bottlenecks inside, this gives excellent gradient flow and parameter efficiency.

- Repeats: ×2 at P2, ×3 at P3, ×3 at P4, ×2 at P5
- More repeats at P3/P4 = more capacity where detection happens

---

#### `SPPF` — Spatial Pyramid Pooling Fast (Layer 9)
```
Input ──► Conv(1×1)
              │
              ▼
         MaxPool(5×5) ──► MaxPool(5×5) ──► MaxPool(5×5)
              │                │                │
              └──── Concat all 4 (orig + 3 pools) ────┘
                                  │
                              Conv(1×1)
                                  │
                               Output
```
**Why:** Applies max-pooling at 3 effective receptive field sizes (5, 9, 13) in sequence using a single small kernel. This captures multi-scale context at the top of the backbone so the head can reason about both local texture and global structure simultaneously.

---

### Head Blocks

#### `nn.Upsample` — Nearest-Neighbour Upsampling (Layers 10, 14)
```
48×48 ──► 96×96   (nearest neighbour, ×2)
```
**Why:** Simple, fast, parameter-free upsampling. Nearest-neighbour avoids checkerboard artefacts and is faster than bilinear for real-time inference on edge hardware.

---

#### `Concat` — Feature Concatenation (Layers 11, 15, 20)
```
[Upsampled higher-level features]
         +
[Same-scale backbone features]
         ↓
  Concatenated along channel dim
```
**Why:** This is the FPN (Feature Pyramid Network) skip connection. It merges semantic context (from deeper layers) with spatial detail (from shallower layers), giving the detector both "what" and "where" information at each scale.

---

#### `DWConv` — Depthwise Separable Convolution (Layers 12, 16, 18, 21)
```
Input ──► Depthwise Conv (k×k, per-channel) ──► Output
```
Combined with pointwise `Conv(1×1)` next:
```
DWConv(k×k) → Conv(1×1)  ≈ standard Conv(k×k) at ~9× fewer params
```
**Why:** The entire detection head uses DWConv instead of standard convolutions. Standard 3×3 conv on 320 channels = 320×320×9 = 921,600 weights. Depthwise + pointwise = 320×9 + 320×80 = 28,480 weights — a **~32× reduction** in that single layer. This is what makes the head ultra-lightweight.

---

#### `Conv(1×1)` — Pointwise Convolution (Layers 13, 17, 19, 22)
```
Input (C_in channels) ──► 1×1 Conv ──► Output (C_out channels)
```
**Why:** After each DWConv (which can't mix channels), a pointwise conv mixes information across channels. Together DWConv + Conv(1×1) = full depthwise-separable convolution, the building block of MobileNet.

---

#### `Detect` — Detection Head (Layer 23)
```
P3 features (96×96×64) ─┐
                          ├──► Per-scale anchor-free predictions
P4 features (48×48×80) ─┘       │
                                  ▼
                          For each cell: [x, y, w, h, conf, cls×2]
                          Using DFL (Distribution Focal Loss) regression
```
**Why — P3+P4 only (no P5):** Insulators and their defects are small-to-medium objects. P5 (stride-32) has a receptive field tuned for large objects (people, cars, etc.) — removing it frees capacity and removes noise for this specific task. All 867K parameters are focused on the relevant scales.

---

## Why No P5 Detection?

```
Scale    Stride   RF       Best for              This dataset?
──────   ──────   ──────   ─────────────────     ─────────────
P3        8       ~50px    Small objects         ✅ Damaged_1 cracks
P4       16       ~100px   Medium objects        ✅ Insulators
P5       32       ~200px   Large objects (car)   ❌ Not needed → REMOVED
```

Removing P5:
- Eliminates ~15% of head parameters
- Removes false positives from large-scale anchors on tiny targets
- Lets the model specialise entirely on small/medium detection

---

## Parameter Budget Breakdown

| Component | Approx Params | % of Total |
|---|---|---|
| Stem Conv (L0) | ~2.8K | 0.3% |
| GhostConv stages (L1,3,5,7) | ~180K | 20.8% |
| C3Ghost blocks (L2,4,6,8) | ~450K | 51.9% |
| SPPF (L9) | ~41K | 4.7% |
| Head DWConv+Conv (L12-22) | ~190K | 21.9% |
| Detect layer | ~3K | 0.3% |
| **Total** | **~867K** | **100%** |

The C3Ghost blocks consume the majority — this is intentional. More parameters in multi-scale feature extraction (backbone) than in the detection head is the right tradeoff for accuracy.

---

## Data Flow Summary

```
768×768 image
    │
    ▼ Backbone (stride ladder: 2→4→8→16→32)
    │
    ├─── P3 @ stride-8  (96×96) ─────────────────────────────┐
    │                                                         │
    ├─── P4 @ stride-16 (48×48) ────────────────────────┐    │
    │                                                    │    │
    └─── P5+SPPF @ stride-32 (24×24, context only)      │    │
              │                                          │    │
              ▼ Head (FPN top-down + bottom-up)          │    │
              │                                          │    │
         Upsample P5 ──── merge with P4 ────────────────┘    │
                              │                               │
                         Upsample P4 ──── merge with P3 ──────┘
                                              │
                                      Downsample back ──── merge with P4
                                              │
                                    ┌─────────┴──────────┐
                                    ▼                     ▼
                              P3 detect              P4 detect
                            (96×96 grid)           (48×48 grid)
                         Small Damaged_1           Insulators
```

---

*Architecture file: [`models/yolo11n-ghost-hybrid-p3p4-medium.yaml`](../models/yolo11n-ghost-hybrid-p3p4-medium.yaml)*
*Validated accuracy: 96.10% mAP50 on VOC-format insulator dataset (143 val images, 300 instances)*
