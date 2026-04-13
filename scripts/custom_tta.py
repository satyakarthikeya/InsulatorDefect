"""Custom TTA with different scale/flip combinations.
Monkeypatches _predict_augment to test various configurations.
"""

import torch
import types
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, BaseModel
from ultralytics.utils.torch_utils import scale_img


def make_predict_augment(scales, flips):
    """Create a custom _predict_augment with given scales and flips."""
    def _predict_augment(self, x):
        img_size = x.shape[-2:]
        y = []
        for si, fi in zip(scales, flips):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=32)
            yi = BaseModel.predict(self, xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, -1), None
    return _predict_augment


configs = {
    # Default TTA
    "default_3scale": {
        "scales": [1, 0.83, 0.67],
        "flips": [None, 3, None],
    },
    # Just lr-flip (no rescaling)
    "flip_only": {
        "scales": [1, 1],
        "flips": [None, 3],
    },
    # 2 scales (drop smallest)
    "2scale": {
        "scales": [1, 0.83],
        "flips": [None, 3],
    },
    # 2 scales + both flipped
    "2scale_4aug": {
        "scales": [1, 1, 0.83, 0.83],
        "flips": [None, 3, None, 3],
    },
    # Mild scales
    "mild_3scale": {
        "scales": [1, 0.92, 0.83],
        "flips": [None, 3, None],
    },
    # Upscale from 640
    "upscale_640": {
        "scales": [1, 1.2, 1.2],
        "flips": [None, None, 3],
        "imgsz": 640,
    },
    # Just flip at 640
    "flip_640": {
        "scales": [1, 1],
        "flips": [None, 3],
        "imgsz": 640,
    },
}


if __name__ == "__main__":
    model_path = "experiments/exp_005_kd_student3/weights/best.pt"
    
    print("=" * 70)
    print("Custom TTA Experiments - KD model")
    print("=" * 70)
    
    # First, single-scale baselines
    model = YOLO(model_path)
    for sz in [640, 768]:
        metrics = model.val(data="VOC/voc.yaml", imgsz=sz, batch=4, device="0", workers=4)
        d1 = metrics.box.all_ap[0, 0] * 100
        ins = metrics.box.all_ap[1, 0] * 100
        print(f"BASELINE @{sz}: mAP50={metrics.box.map50*100:.2f}%, D1={d1:.2f}%, ins={ins:.2f}%")
    
    print()
    
    # Custom TTA configs
    for name, cfg in configs.items():
        model = YOLO(model_path)
        model.model.stride = torch.tensor([8., 16., 32.])
        
        # Monkeypatch
        custom_fn = make_predict_augment(cfg["scales"], cfg["flips"])
        model.model._predict_augment = types.MethodType(custom_fn, model.model)
        
        imgsz = cfg.get("imgsz", 768)
        
        metrics = model.val(
            data="VOC/voc.yaml", imgsz=imgsz, batch=4,
            device="0", augment=True, workers=4
        )
        d1 = metrics.box.all_ap[0, 0] * 100
        ins = metrics.box.all_ap[1, 0] * 100
        scales_str = str(cfg["scales"])
        flips_str = str(cfg["flips"])
        print(f"{name} @{imgsz} s={scales_str} f={flips_str}: "
              f"mAP50={metrics.box.map50*100:.2f}%, D1={d1:.2f}%, ins={ins:.2f}%")
    
    print()
    print("Done!")
