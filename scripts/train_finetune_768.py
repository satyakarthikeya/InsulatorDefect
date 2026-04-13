"""
Fine-tune KD model at 768 resolution.
Since inference at 768 gives big D1 boost (91.16 → 93.26%), 
training at 768 should help the model learn better at that scale.
Short fine-tune: 20 epochs from KD checkpoint.
"""

from ultralytics import YOLO

def main():
    model = YOLO("experiments/exp_005_kd_student3/weights/best.pt")
    
    print("=" * 60)
    print("  Fine-tune KD model at 768 resolution")
    print("  Base: exp_005_kd (94.12% @640, 94.72% @768)")
    print("=" * 60)
    
    model.train(
        data="VOC/voc.yaml",
        epochs=25,
        imgsz=768,
        batch=2,          # 4GB VRAM limit at 768
        device="0",
        workers=4,
        cache="ram",
        amp=True,
        
        # Fine-tune LR: very conservative
        lr0=0.002,
        lrf=0.1,         # End at 10% = 0.0002
        cos_lr=True,
        warmup_epochs=1,
        
        # Optimizer
        optimizer="SGD",
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation: moderate (don't want to diverge too much)
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.05,
        close_mosaic=5,   # Turn off mosaic last 5 epochs
        
        # Save
        save_period=5,
        
        project="experiments",
        name="exp_009_finetune_768",
        exist_ok=True,
        
        verbose=True,
        patience=15,
    )

if __name__ == "__main__":
    main()
