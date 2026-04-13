# Ablation Study: Main Model (`exp_012_head_finetune_768`)

## Scope

This ablation analyzes how each major design/training decision contributed to the final main model:

- Final checkpoint: `experiments/exp_012_head_finetune_768/weights/best.pt`
- Architecture family: `YOLO11n-Ghost-Hybrid-P3P4-Medium`

## Method

- Metrics were taken from each run's `results.csv` in `experiments/*/results.csv`.
- For every run, the best (max) epoch value was used for:
  - `metrics/mAP50(B)`
  - `metrics/mAP50-95(B)`
  - `metrics/precision(B)`
  - `metrics/recall(B)`
- These are peak per-epoch validation metrics from training logs.

## Cumulative Ablation (Main Path)

| Step | Change Introduced | Run | mAP50 | Delta vs Previous | mAP50-95 | Precision | Recall |
|---|---|---|---:|---:|---:|---:|---:|
| A0 | Vanilla baseline (YOLO11n) | `baseline_yolo11n` | 90.20 | - | 59.31 | 91.28 | 85.14 |
| A1 | Custom Ghost-Hybrid-P3P4 architecture | `exp_002_ghost_hybrid_medium3` | 94.21 | +4.01 | 68.08 | 94.26 | 92.45 |
| A2 | Knowledge Distillation | `exp_005_kd_student3` | 94.53 | +0.31 | 68.70 | 93.46 | 92.88 |
| A3 | 768 fine-tune (SGD, unfrozen) | `exp_009_finetune_768` | 96.11 | +1.59 | 66.02 | 93.80 | 95.36 |
| A4 | Head-only fine-tune at 768 (`freeze=10`) | `exp_012_head_finetune_768` | 96.46 | +0.35 | 66.87 | 95.08 | 95.16 |

### Net Gain to Main Model

- A0 -> A4: `+6.26` mAP50 points (90.20 -> 96.46)
- Largest single gain came from architecture redesign (A1), followed by 768 fine-tuning (A3)

## Negative/Control Ablations

These runs were useful controls to isolate what did not help.

| Control | Setting | Run | mAP50 | Delta vs Reference |
|---|---|---|---:|---:|
| C1 | 704 aggressive high-res (`lr0=0.005`, unfrozen) | `exp_006_highres_704` | 93.08 | -1.44 vs A2 |
| C2 | 704 frozen stage (`freeze=10`) | `exp_007_highres704_s1_frozen` | 93.37 | -1.16 vs A2 |
| C3 | 704 stage-2 unfrozen continuation | `exp_007_highres704_s2_unfrozen` | 92.85 | -1.68 vs A2 |
| C4 | 768 multi-scale continuation | `exp_010_finetune_768_v2` | 95.69 | -0.42 vs A3 |
| C5 | 768 ultra-low LR continuation | `exp_011_finetune_768_v3` | 96.06 | -0.05 vs A3 |
| C6 | 896 resolution attempt | `exp_013_finetune_896` | 94.37 | -1.74 vs A3 |

## Main Findings

1. Architecture change delivered the biggest foundational jump.
   - Moving from vanilla baseline to Ghost-Hybrid-P3P4 gave `+4.01` mAP50.

2. KD gave a modest direct gain but enabled stronger high-resolution transfer.
   - Direct KD gain was `+0.31`, but it provided the starting point for the successful 768 stage.

3. Resolution sweet spot was 768, not 704 or 896.
   - 768 fine-tune gave `+1.59` over KD baseline.
   - 704 and 896 variants underperformed significantly.

4. Head-only final tuning provided the last stable push.
   - `freeze=10` at 768 gave a final `+0.35` mAP50.

5. Trade-off note:
   - Relative to A2, A3 increased mAP50/recall strongly, while mAP50-95 dipped.
   - A4 recovered part of mAP50-95 while retaining top mAP50.

## Teacher Reference

Teacher run (`exp_004_teacher_yolo11s`) peak mAP50: `96.54`.

Main model (`exp_012_head_finetune_768`) peak mAP50: `96.46`.

Gap to teacher (training-log peak basis): `0.08` mAP50 points.

## Reproduce This Table (quick)

```bash
python - <<'PY'
import csv
from pathlib import Path

exp_dirs = [
    'baseline_yolo11n',
    'exp_002_ghost_hybrid_medium3',
    'exp_005_kd_student3',
    'exp_009_finetune_768',
    'exp_012_head_finetune_768',
]

for d in exp_dirs:
    p = Path('experiments') / d / 'results.csv'
    rows = list(csv.DictReader(p.open(newline='')))
    km = {k.strip(): k for k in rows[0]}
    def best(k):
        return max(float(r[km[k]]) for r in rows if r[km[k]].strip())
    print(
        d,
        f"mAP50={best('metrics/mAP50(B)'):.6f}",
        f"mAP50-95={best('metrics/mAP50-95(B)'):.6f}",
        f"P={best('metrics/precision(B)'):.6f}",
        f"R={best('metrics/recall(B)'):.6f}",
    )
PY
```
