"""
SWA Fine-tune from KD weights.
Stochastic Weight Averaging: cyclic LR, collect snapshots, average weights.
Uses Ultralytics training with cosine LR and manual SWA averaging.
"""

import torch
import copy
import os
from pathlib import Path
from ultralytics import YOLO


def swa_finetune():
    """SWA fine-tune strategy:
    1. Short cyclic training from KD checkpoint with high LR
    2. Collect weight snapshots every N epochs
    3. Average all snapshots
    4. Update BN statistics
    5. Evaluate with and without TTA
    """
    
    base_weights = "experiments/exp_005_kd_student3/weights/best.pt"
    save_dir = Path("experiments/exp_008_swa")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # Stage 1: Short cyclic fine-tune with cosine LR restart
    # Using moderate LR to explore around the minimum
    # ============================================================
    print("=" * 70)
    print("Stage 1: SWA Cyclic Fine-tune from KD weights")
    print("=" * 70)
    
    model = YOLO(base_weights)
    
    # Train with cosine LR for 30 epochs with high initial LR
    # cos_lr gives a natural cyclic decay
    results = model.train(
        data="VOC/voc.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device="0",
        workers=4,
        cache="ram",
        amp=True,
        
        # LR: Start moderate, cosine decay
        lr0=0.005,      # Higher than typical fine-tune 
        lrf=0.2,        # End at 20% of lr0 = 0.001
        cos_lr=True,
        warmup_epochs=0, # No warmup - jump right in
        
        # Optimizer
        optimizer="SGD",
        momentum=0.9,
        weight_decay=0.0005,
        
        # Augmentation: moderate
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.05,
        
        # Save every epoch for SWA averaging
        save_period=1,
        
        project="experiments",
        name="exp_008_swa_cyclic",
        exist_ok=True,
        
        verbose=True,
    )
    
    print("\nStage 1 complete!")
    return "experiments/exp_008_swa_cyclic"


def collect_and_average_swa(train_dir, start_epoch=10, end_epoch=30):
    """Average model weights from epoch range (SWA)."""
    
    weights_dir = Path(train_dir) / "weights"
    save_path = Path("experiments") / "exp_008_swa_averaged.pt"
    
    print("=" * 70)
    print(f"Stage 2: SWA Weight Averaging (epochs {start_epoch}-{end_epoch})")
    print("=" * 70)
    
    # Load base model structure
    base_model = YOLO(str(weights_dir / "best.pt"))
    avg_state = None
    count = 0
    
    for epoch in range(start_epoch, end_epoch + 1):
        ckpt_path = weights_dir / f"epoch{epoch}.pt"
        if not ckpt_path.exists():
            print(f"  Skipping epoch {epoch} (not found)")
            continue
        
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model_state = ckpt["model"].float().state_dict()
        
        if avg_state is None:
            avg_state = {k: v.clone() for k, v in model_state.items()}
        else:
            for k in avg_state:
                avg_state[k] += model_state[k]
        count += 1
        print(f"  Loaded epoch {epoch}")
    
    if count == 0:
        print("ERROR: No checkpoints found!")
        return None
    
    # Average
    for k in avg_state:
        avg_state[k] /= count
    
    print(f"\nAveraged {count} checkpoints")
    
    # Save averaged model using the base checkpoint structure
    ckpt = torch.load(str(weights_dir / "best.pt"), map_location="cpu", weights_only=False)
    
    # Load averaged weights into model  
    model_instance = ckpt["model"].float()
    model_instance.load_state_dict(avg_state)
    ckpt["model"] = model_instance
    
    torch.save(ckpt, str(save_path))
    print(f"Saved SWA averaged model to {save_path}")
    
    return str(save_path)


def evaluate_swa(swa_path):
    """Evaluate SWA model with and without TTA."""
    print("=" * 70)
    print("Stage 3: Evaluation")
    print("=" * 70)
    
    model = YOLO(swa_path)
    
    # Single-scale
    for sz in [640, 768]:
        metrics = model.val(data="VOC/voc.yaml", imgsz=sz, batch=4, device="0", workers=4)
        d1 = metrics.box.all_ap[0, 0] * 100
        ins = metrics.box.all_ap[1, 0] * 100
        print(f"SWA @{sz}: mAP50={metrics.box.map50*100:.2f}%, D1={d1:.2f}%, ins={ins:.2f}%")
    
    # TTA
    model.model.stride = torch.tensor([8., 16., 32.])
    metrics = model.val(
        data="VOC/voc.yaml", imgsz=768, batch=4,
        device="0", augment=True, workers=4
    )
    d1 = metrics.box.all_ap[0, 0] * 100
    ins = metrics.box.all_ap[1, 0] * 100
    print(f"SWA TTA@768: mAP50={metrics.box.map50*100:.2f}%, D1={d1:.2f}%, ins={ins:.2f}%")


if __name__ == "__main__":
    # Stage 1: Train
    train_dir = swa_finetune()
    
    # Stage 2: Average weights 
    swa_path = collect_and_average_swa(train_dir, start_epoch=10, end_epoch=30)
    
    # Stage 3: Evaluate
    if swa_path:
        evaluate_swa(swa_path)
    
    print("\n" + "=" * 70)
    print("SWA Pipeline Complete!")
    print("=" * 70)
