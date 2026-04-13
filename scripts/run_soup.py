"""Model Soup: average weights from multiple experiments and validate."""
import torch, copy, sys
from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent
w1 = str(PROJECT / 'experiments/exp_002_ghost_hybrid_medium3/weights/best.pt')
w2 = str(PROJECT / 'experiments/exp_005_kd_student3/weights/best.pt')
w3 = str(PROJECT / 'experiments/exp_tfa_20260217_182417/weights/best.pt')
DATA = str(PROJECT / 'VOC/voc.yaml')

# Load checkpoints
print("Loading checkpoints...")
ckpt1 = torch.load(w1, map_location='cpu', weights_only=False)
ckpt2 = torch.load(w2, map_location='cpu', weights_only=False)
ckpt3 = torch.load(w3, map_location='cpu', weights_only=False)

def make_soup(ckpts, alphas, output_path):
    avg = copy.deepcopy(ckpts[0])
    sd0 = ckpts[0]['model'].state_dict()
    avg_sd = {}
    for key in sd0:
        tensors = [c['model'].state_dict()[key].float() for c in ckpts]
        avg_sd[key] = sum(a * t for a, t in zip(alphas, tensors))
    avg['model'].load_state_dict(avg_sd)
    torch.save(avg, output_path)
    return output_path

# Create soups
print("Creating soups...")
soups = {
    'Soup 50/50': make_soup([ckpt1, ckpt2], [0.5, 0.5], str(PROJECT / 'experiments/soup_2way_50_50.pt')),
    'Soup 40/60': make_soup([ckpt1, ckpt2], [0.4, 0.6], str(PROJECT / 'experiments/soup_2way_40_60.pt')),
    'Soup 30/70': make_soup([ckpt1, ckpt2], [0.3, 0.7], str(PROJECT / 'experiments/soup_2way_30_70.pt')),
    'Soup 3way': make_soup([ckpt1, ckpt2, ckpt3], [0.4, 0.4, 0.2], str(PROJECT / 'experiments/soup_3way.pt')),
}

# All models to test
models = {
    'Baseline exp_002': w1,
    'KD exp_005': w2,
    'TFA exp_tfa': w3,
}
models.update(soups)

# Validate
print()
hdr = f"{'Model':<25} {'mAP50':>8} {'D1_AP50':>8} {'ins_AP50':>8} {'D1_R':>6} {'D1_P':>6}"
print('=' * len(hdr))
print(hdr)
print('=' * len(hdr))

for name, wpath in models.items():
    m = YOLO(wpath)
    r = m.val(data=DATA, imgsz=640, batch=8, device='0', workers=4,
              conf=0.001, iou=0.7, plots=False, verbose=False)
    map50 = r.results_dict['metrics/mAP50(B)']
    ap50 = r.box.ap50
    p = r.box.p
    rec = r.box.r
    print(f'{name:<25} {map50:>8.4f} {ap50[1]:>8.4f} {ap50[0]:>8.4f} {rec[1]:>6.3f} {p[1]:>6.3f}')
    del m
    torch.cuda.empty_cache()

print('=' * len(hdr))
