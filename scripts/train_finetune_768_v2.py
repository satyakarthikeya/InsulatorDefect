"""
Second-stage fine-tune at 768: Ultra-low LR from best 768-ft weights.
Targeted to squeeze out the last ~1% to reach 97%.
Also tries multi-scale training to improve robustness.
"""

from ultralytics import YOLO

def main():
    model = YOLO("experiments/exp_009_finetune_768/weights/best.pt")
    
    print("=" * 60)
    print("  Stage 2 Fine-tune: Very low LR @ 768")
    print("  Base: exp_009 best (96.11% @768)")
    print("=" * 60)
    
    model.train(
        data="VOC/voc.yaml",
        epochs=20,
        imgsz=768,
        batch=2,
        device="0",
        workers=4,
        cache="ram",
        amp=True,
        
        # Ultra-conservative LR
        lr0=0.0005,       # Very low - gentle optimization
        lrf=0.2,          # End at 0.0001
        cos_lr=True,
        warmup_epochs=0,   # No warmup from already-trained model
        
        # Optimizer
        optimizer="SGD",
        momentum=0.937,
        weight_decay=0.0003,   # Slightly less regularization
        
        # Mild augmentation — don't shake the model too much
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.2,
        degrees=3.0,
        translate=0.08,
        scale=0.2,
        fliplr=0.5,
        mosaic=0.5,       # Less mosaic 
        mixup=0.0,        # No mixup
        close_mosaic=5,   # Turn off mosaic last 5 epochs
        
        # Multi-scale training: randomly vary imgsz +-50%
        # This helps with scale robustness
        multi_scale=True,
        
        # Save
        save_period=2,
        
        project="experiments",
        name="exp_010_finetune_768_v2",
        exist_ok=True,
        
        verbose=True,
        patience=12,
    )

if __name__ == "__main__":
    main()
