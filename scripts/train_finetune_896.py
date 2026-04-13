"""
Fine-tune exp_009 (768-trained) at 896 resolution to improve Damaged_1 recall.
batch=1 due to VRAM constraints.
"""
from ultralytics import YOLO
import os

os.chdir("/home/satyakarthikeya/Documents/analog_project")

model = YOLO("experiments/exp_009_finetune_768/weights/best.pt")

results = model.train(
    data="VOC/voc.yaml",
    epochs=20,
    imgsz=896,
    batch=1,
    device=0,
    workers=4,
    amp=True,
    cache="ram",
    
    # Optimizer
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.1,
    cos_lr=True,
    weight_decay=0.0005,
    warmup_epochs=2,
    
    # Augmentation - moderate
    mosaic=0.7,
    mixup=0.05,
    copy_paste=0.1,   # copy-paste aug to boost D1 representation
    close_mosaic=5,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=5.0,
    translate=0.1,
    scale=0.3,
    fliplr=0.5,
    
    # Project
    project="experiments",
    name="exp_013_finetune_896",
    exist_ok=True,
    
    # Eval
    val=True,
    plots=True,
    patience=10,
    save_period=5,
    
    # No multi_scale (it hurts)
    multi_scale=False,
)

print("\n=== TRAINING COMPLETE ===")
print(f"Best mAP50: {results.box.map50:.5f}")
