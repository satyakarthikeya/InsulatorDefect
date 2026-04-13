"""Advanced TTA: more scales, ensembling, and NMS tuning."""

import torch
import types
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, BaseModel
from ultralytics.utils.torch_utils import scale_img


def make_predict_augment(scales, flips):
    def _predict_augment(self, x):
        img_size = x.shape[-2:]
        y = []
        for si, fi in zip(scales, flips):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=32)
            yi = BaseModel.predict(self, xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, -1), None
    return _predict_augment


configs = {
    # 4-scale TTA
    "4scale": {
        "scales": [1, 0.92, 0.83, 0.67],
        "flips": [None, None, 3, None],
        "imgsz": 768,
    },
    # 5-scale with all flips
    "5scale_flips": {
        "scales": [1, 1, 0.83, 0.83, 0.67],
        "flips": [None, 3, None, 3, None],
        "imgsz": 768,
    },
    # Aggressive multi-scale
    "6scale": {
        "scales": [1, 1, 0.92, 0.83, 0.83, 0.67],
        "flips": [None, 3, None, None, 3, 3],
        "imgsz": 768,
    },
    # Higher base with default TTA
    "default_896": {
        "scales": [1, 0.83, 0.67],
        "flips": [None, 3, None],
        "imgsz": 896,
    },
    # default @768 with ud flip too
    "3scale_ud": {
        "scales": [1, 0.83, 0.67],
        "flips": [None, 2, None],
        "imgsz": 768,
    },
    # 3scale all flipped
    "3scale_all_flip": {
        "scales": [1, 1, 0.83, 0.83, 0.67, 0.67],
        "flips": [None, 3, None, 3, None, 3],
        "imgsz": 768,
    },
}


if __name__ == "__main__":
    model_path = "experiments/exp_005_kd_student3/weights/best.pt"
    
    print("=" * 70)
    print("Advanced TTA - KD model (exp_005_kd)")
    print("Reference: default_3scale @768 = 95.69% (D1=94.49, ins=96.89)")
    print("=" * 70)
    
    for name, cfg in configs.items():
        model = YOLO(model_path)
        model.model.stride = torch.tensor([8., 16., 32.])
        
        custom_fn = make_predict_augment(cfg["scales"], cfg["flips"])
        model.model._predict_augment = types.MethodType(custom_fn, model.model)
        
        imgsz = cfg["imgsz"]
        
        metrics = model.val(
            data="VOC/voc.yaml", imgsz=imgsz, batch=4 if imgsz <= 768 else 2,
            device="0", augment=True, workers=4
        )
        d1 = metrics.box.all_ap[0, 0] * 100
        ins = metrics.box.all_ap[1, 0] * 100
        print(f"{name} @{imgsz}: mAP50={metrics.box.map50*100:.2f}%, "
              f"D1={d1:.2f}%, ins={ins:.2f}%  "
              f"[s={cfg['scales']}, f={cfg['flips']}]")
    
    print("\nDone!")
