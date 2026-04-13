# Plan: Push Beyond 94% mAP50 with Advanced Techniques

**TL;DR:** A 5-phase, ~4 day plan to systematically squeeze more accuracy from the 867K Ghost-Hybrid model. We start with zero-effort quick wins (TTA, cosine LR), then add a lightweight attention mechanism (CBAM — built-in, ~480 extra params), retrain with multi-scale + progressive resolution, and finally apply inference-time ensemble (WBF). Each phase builds on the previous, with clear go/no-go checkpoints. Expected cumulative gain: **+1.5–3.5% mAP50** (target: ~96-97%).

---

## Phase 0: Baseline Measurement (30 min)

1. Run `model.val()` on `exp_002_ghost_hybrid_medium3/weights/best.pt` to get **per-class mAP50 breakdown** (Damaged_1 vs insulator) — this is currently unknown and critical for directing all effort
2. Run `model.val(augment=True)` on the same weights to measure free **TTA gain** at 640×640
3. Record baseline numbers — this is the scoreboard everything else is measured against

---

## Phase 1: Zero-Code Quick Wins — Retrain with Cosine LR + Multi-Scale (Day 1)

4. Create `scripts/train_exp004_cosine.py` — retrain from scratch with the same base config as exp_002_medium3, but add:
   - `cos_lr=True` (cosine annealing instead of linear decay)
   - `multi_scale=0.25` (random resolution 480–800 per batch, makes model resolution-robust)
   - `cutmix=0.2` (built-in CutMix augmentation, never tried before)
   - Keep everything else identical: batch=8, cache=ram, 300 epochs, etc.
5. Train for 300 epochs — expected gain: **+0.3–0.8% mAP50** from cosine LR + multi-scale synergy
6. **Checkpoint:** If mAP50 ≤ 94.21%, skip Phase 2 architecture changes and go straight to Phase 3 (inference tricks still help)

---

## Phase 2: Architecture Enhancement — CBAM Attention (Day 2)

7. Create a new model YAML `models/yolo11n-ghost-hybrid-p3p4-medium-cbam.yaml` — copy the existing architecture and insert CBAM attention at two strategic locations:
   - After SPPF (layer 9) — `CBAM[160, 7]` (+~480 params) — helps backbone focus on defect-relevant features before entering the head
   - After the P3 detection branch merge (layer 17) — `CBAM[64, 7]` (+~192 params) — sharpens small defect features at the finest detection scale
   - Total added: **~672 parameters** (867K → ~868K, well under 1.2M budget)
   - Update all layer index references in the head accordingly
8. Create `scripts/train_exp005_cbam.py` — train the CBAM model from scratch with the best config from Phase 1 (cosine LR + multi-scale)
9. Train for 300 epochs — expected gain from CBAM: **+0.2–0.5% mAP50** (attention helps the rare Damaged_1 class disproportionately)

---

## Phase 3: Progressive Resolution Fine-Tuning (Day 3)

10. Take the best weights from Phase 1 or Phase 2 (whichever is better)
11. Create `scripts/train_exp006_hires.py` — **unfrozen** fine-tuning at 704×704:
    - `freeze=None` (entire model trainable — critical lesson from TFA v3 failure)
    - `imgsz=704`, `batch=4` (fits 4GB VRAM)
    - `lr0=0.001`, `cos_lr=True` (gentle but unfrozen)
    - `epochs=80`, `patience=30`
    - Standard augmentation (no aggressive rare-class tricks)
    - The key difference from TFA v3: backbone is **unfrozen**, so features can adapt to 704 spatial layout
12. Expected gain: **+0.3–1.0% mAP50** — higher resolution helps detect small defects in Damaged_1 class

---

## Phase 4: Inference-Time Ensemble — WBF (Day 3–4)

13. Install `ensemble-boxes` library
14. Create `scripts/inference_wbf.py` — multi-scale Weighted Box Fusion:
    - Run the best model at 3 scales: 576, 640, 704 (+ optionally 768)
    - Also run with horizontal flip at each scale (6 total predictions)
    - Fuse all predictions using WBF with `iou_thr=0.55`
    - This is **inference-only** — no retraining, purely additive
15. Expected gain: **+0.5–1.5% mAP50** on top of the best single-model result

---

## Phase 5: Model Soup — Checkpoint Averaging (Day 4)

16. Create `scripts/model_soup.py` — average weights from:
    - Top 3-5 checkpoints from the best experiment (pseudo-SWA)
    - Or: average weights across Phase 1 + Phase 2 best models (cross-experiment soup)
17. Validate the averaged model — expected gain: **+0.1–0.3% mAP50** (smooths the loss landscape)
18. Apply WBF inference ensemble (Phase 4) on top of the souped model for maximum accuracy

---

## Verification Checkpoints

| Checkpoint | Command | Expected Result |
|------------|---------|-----------------|
| Phase 0 | `yolo detect val model=exp_002_medium3/weights/best.pt data=VOC/voc.yaml` | Per-class baseline |
| Phase 0 TTA | `yolo detect val model=...best.pt data=VOC/voc.yaml augment=True` | +0.5-1% mAP50 |
| Phase 1 | Check `experiments/exp_004_*/results.csv` — best mAP50 | ≥ 94.5% mAP50 |
| Phase 2 | Check `experiments/exp_005_*/results.csv` — best mAP50 | ≥ 94.5% mAP50 |
| Phase 3 | Check `experiments/exp_006_*/results.csv` — best mAP50 | ≥ 95% mAP50 |
| Phase 4 | Run `python scripts/inference_wbf.py` — compute mAP from fused preds | ≥ 96% mAP50 |
| Phase 5 | Run `python scripts/model_soup.py` + WBF | ≥ 96.5% mAP50 |

---

## Design Decisions

- **CBAM over SE/ECA/SimAM:** CBAM is fully built-in to Ultralytics (no custom module registration needed), combines both channel + spatial attention, and adds negligible params (~672)
- **Unfrozen progressive resolution over frozen TFA:** TFA v3 proved frozen backbone + resolution mismatch is destructive; unfreezing is mandatory for resolution changes
- **WBF over built-in TTA:** WBF gives finer control over scale weights and fusion IoU threshold; built-in `augment=True` TTA is a good quick test but WBF with custom scales performs better
- **No knowledge distillation:** Too complex for 4-day timeline and 4GB VRAM (running teacher + student simultaneously); save for a future round
- **No aggressive class rebalancing:** Experiments proved cls=2.0 / copy_paste / flipud all hurt this well-trained model; focus on resolution, attention, and ensemble instead
- **cosine LR + multi_scale as default going forward:** These are proven techniques with zero downside; should be standard for all future experiments
