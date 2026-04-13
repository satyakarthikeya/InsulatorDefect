 Task: Implement Knowledge Distillation (Teacher-Student Pipeline)
====================================================================

🎯 Objective
------------

Write the Python scripts required to perform Knowledge Distillation (KD). We will transfer knowledge from a heavy, highly accurate "Teacher" model (YOLO11m) to our current 867K-parameter "Student" model (YOLO11n-Ghost-Hybrid-P3P4-Medium).

Our current Student model is stuck at **94.17% mAP50**. The goal of this KD pipeline is to push it to **~97% mAP50** by learning from the Teacher's soft-label distributions, specifically to improve the rare Damaged\_1 class.

⚠️ CRITICAL HARDWARE CONSTRAINTS (RTX 3050 - 4GB VRAM)
------------------------------------------------------

You must engineer these scripts to survive on exactly 4GB of VRAM and 16GB of System RAM.During KD, **both the Teacher and Student models are loaded into VRAM simultaneously.**Any model.train() calls in your code **MUST** include:

1.  batch=4 (or 2 if 4 fails). Never use 8 or -1 for KD.
    
2.  amp=True (Mandatory mixed precision).
    
3.  cache="ram" (Prevents disk IO bottlenecks).
    
4.  workers=4 (Prevents CPU/RAM thrashing).
    
5.  device="0"
    

🛠️ Implementation Phases
-------------------------

### Phase 1: Train the Teacher Model (train\_teacher.py)

Write a script to train the yolo11m.pt (Medium) model on VOC/voc.yaml.

*   **Goal:** Create our "Oracle" model.
    
*   **Config:** Train for 100 epochs, imgsz=640, batch=8 (since only one model is in memory here).
    
*   **Save Location:** experiments/exp\_004\_teacher\_yolo11m/
    

### Phase 2: The Distillation Script (train\_kd.py)

Write the main distillation script. Do **NOT** write a custom PyTorch training loop from scratch. Use the native Ultralytics KD support.

*   **Student:** Load our best base weights: experiments/exp\_002\_ghost\_hybrid\_medium3/weights/best.pt.
    
*   **Teacher:** Load the best weights from Phase 1.
    
*   **Ultralytics API:** Pass the teacher model to the student's train method._(Example: student\_model.train(data="VOC/voc.yaml", teacher=teacher\_model.model, ...))_
    
*   **Hyperparameters for KD:**
    
    *   lr0=0.0001 (Use a low learning rate so the student doesn't unlearn its base weights).
        
    *   cos\_lr=True
        
    *   Disable heavy augmentations (mosaic=0.5, mixup=0.0, copy\_paste=0.0) so the student focuses on the Teacher's soft labels, not noisy data.
        
*   **Save Location:** experiments/exp\_005\_kd\_student/
    

### Phase 3: Validation Script Updates

Update or create a validation script that runs the new KD-trained student model at imgsz=704 (Unfrozen High-Resolution testing) to verify if the Damaged\_1 class improved over the 91.24% baseline.

📋 Instructions for the AI Agent
--------------------------------

1.  Review this document carefully.
    
2.  Acknowledge the 4GB VRAM constraints.
    
3.  Write scripts/train\_teacher.py (Phase 1).
    
4.  Write scripts/train\_kd.py (Phase 2).
    
5.  Ensure all file paths match the provided structure.