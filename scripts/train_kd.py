"""
Phase 2: Knowledge Distillation — Teacher-Student Pipeline
===========================================================
Transfer knowledge from a heavy YOLO11m teacher to our lightweight
867K-parameter Ghost-Hybrid student model.

Strategy:
  - Subclass Ultralytics DetectionTrainer to inject KD loss
  - Teacher runs in eval() + FP16 + no_grad (minimal VRAM overhead)
  - KL-divergence on classification logits (temperature-scaled)
  - Box/DFL losses remain unchanged from standard YOLO loss
  - Both models fit in 4GB VRAM at batch=4 with AMP

Student baseline: 94.17% mAP50 (91.24% on Damaged_1)
Target:           ~97% mAP50 (improved Damaged_1 detection)

Usage:
    python scripts/train_kd.py
    python scripts/train_kd.py --teacher experiments/exp_004_teacher_yolo11m/weights/best.pt
    python scripts/train_kd.py --kd-alpha 0.5 --kd-temperature 4.0
    python scripts/train_kd.py --batch 2  # fallback if batch=4 OOMs
"""

import os
import sys
import argparse
from pathlib import Path
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Project root setup ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import autocast, unwrap_model


# ══════════════════════════════════════════════════════════════
#  Knowledge Distillation Trainer
# ══════════════════════════════════════════════════════════════

class KDDetectionTrainer(DetectionTrainer):
    """Custom DetectionTrainer that adds Knowledge Distillation loss.

    Injects a KL-divergence distillation loss on classification logits
    between a frozen teacher and the trainable student. The teacher
    runs in eval + FP16 + no_grad to minimize VRAM usage.

    The total loss becomes:
        L_total = (1 - alpha) * L_yolo + alpha * T^2 * KL(student || teacher)

    where L_yolo is the standard YOLO detection loss (box + cls + dfl),
    and T is the temperature for softening logit distributions.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None,
                 teacher_weights=None, kd_alpha=0.5, kd_temperature=4.0):
        """Initialize KD trainer with teacher model path and KD hyperparams.

        Args:
            cfg: Default configuration.
            overrides: Configuration overrides.
            _callbacks: Callback functions.
            teacher_weights: Path to teacher model best.pt.
            kd_alpha: Weight for distillation loss (0-1). Higher = more teacher influence.
            kd_temperature: Temperature for softening logit distributions.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.teacher_weights = teacher_weights
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        self.teacher_model = None

    def _setup_train(self):
        """Extend parent setup to load and prepare the teacher model."""
        super()._setup_train()

        # ── Load teacher model ──
        LOGGER.info(f"\n{'═' * 60}")
        LOGGER.info(f"  Loading Teacher Model for Knowledge Distillation")
        LOGGER.info(f"  Teacher weights: {self.teacher_weights}")
        LOGGER.info(f"  KD Alpha: {self.kd_alpha} | Temperature: {self.kd_temperature}")
        LOGGER.info(f"{'═' * 60}\n")

        teacher_yolo = YOLO(self.teacher_weights, task="detect")
        self.teacher_model = teacher_yolo.model

        # Move to same device, set eval mode, freeze, convert to FP16
        self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.teacher_model.half()  # FP16 to save VRAM

        LOGGER.info(f"  Teacher loaded: {sum(p.numel() for p in self.teacher_model.parameters()):,} params (FP16, frozen)")
        LOGGER.info(f"  Student params:  {sum(p.numel() for p in self.model.parameters()):,}")

        # Store loss names including KD loss for logging
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "kd_loss"

    def get_validator(self):
        """Return a DetectionValidator — use standard 3-component loss for validation.

        Validation uses the standard YOLO loss (box, cls, dfl) — no KD loss.
        We only add kd_loss to loss_names for training logging.
        """
        from ultralytics.models import yolo
        # Temporarily set 3-component loss names for validator initialization
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        validator = yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
        # Restore 4-component loss names for training logging
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "kd_loss"
        return validator

    def validate(self):
        """Override validate to temporarily use 3-component loss.

        The validator initializes self.loss from trainer.loss_items shape (line 152
        in validator.py), and model.loss() returns 3 components. We must ensure
        trainer.loss_items has 3 components during validation.
        """
        # Save 4-component state
        saved_loss_names = self.loss_names
        saved_loss_items = self.loss_items

        # Switch to 3-component for validation
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        if self.loss_items is not None and len(self.loss_items) == 4:
            self.loss_items = self.loss_items[:3]  # strip kd_loss component

        metrics, fitness = super().validate()

        # Restore 4-component state for training
        self.loss_names = saved_loss_names
        self.loss_items = saved_loss_items
        return metrics, fitness

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Return a loss dict with labeled training loss items (4 components)."""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return formatted training progress string with KD loss column."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    @staticmethod
    def _extract_cls_logits(feats, nc, reg_max):
        """Extract classification logits from detection feature maps.

        Args:
            feats: List of feature maps from Detect head [P3, P4, ...].
                   Each has shape (B, reg_max*4 + nc, H_i, W_i).
            nc: Number of classes.
            reg_max: Regression max value.

        Returns:
            cls_logits: (B, total_anchors, nc) — raw classification logits.
        """
        no = reg_max * 4 + nc
        batch_size = feats[0].shape[0]

        # Concatenate all scale levels: (B, no, total_anchors)
        all_feats = torch.cat(
            [xi.view(batch_size, no, -1) for xi in feats], dim=2
        )
        # Split box and cls: cls_logits shape (B, nc, total_anchors)
        _, cls_logits = all_feats.split((reg_max * 4, nc), dim=1)
        # Permute to (B, total_anchors, nc)
        return cls_logits.permute(0, 2, 1).contiguous()

    def _compute_kd_loss(self, student_feats, images):
        """Compute KL-divergence distillation loss on classification logits.

        Args:
            student_feats: Student's raw detection feature maps (list of tensors).
            images: Input images tensor (B, C, H, W) for teacher forward pass.

        Returns:
            kd_loss: Scalar KL-divergence distillation loss.
        """
        student_model = unwrap_model(self.model)
        nc = student_model.model[-1].nc
        student_reg_max = student_model.model[-1].reg_max

        # ── Teacher forward pass (FP16, no grad) ──
        # Call the DetectionModel directly (NOT .model which is raw nn.Sequential)
        # In eval mode, DetectionModel.forward(tensor) → predict() → (inference_out, raw_feats)
        with torch.no_grad():
            teacher_out = self.teacher_model(images.half())
            # eval mode returns (y, x) where x is list of raw feature maps
            if isinstance(teacher_out, tuple):
                teacher_feats = teacher_out[1] if len(teacher_out) > 1 else teacher_out[0]
            elif isinstance(teacher_out, dict):
                teacher_feats = teacher_out.get("one2many", list(teacher_out.values())[0])
            else:
                teacher_feats = teacher_out

        teacher_nc = self.teacher_model.model[-1].nc
        teacher_reg_max = self.teacher_model.model[-1].reg_max

        # ── Extract classification logits ──
        student_cls = self._extract_cls_logits(student_feats, nc, student_reg_max)
        teacher_cls = self._extract_cls_logits(teacher_feats, teacher_nc, teacher_reg_max).float()

        # Handle anchor count mismatch (different stride/resolution between models)
        # YOLO11s has 3 detection scales (8400 anchors), student has 2 (8000)
        min_anchors = min(student_cls.shape[1], teacher_cls.shape[1])
        if student_cls.shape[1] != teacher_cls.shape[1]:
            if not hasattr(self, '_anchor_mismatch_warned'):
                LOGGER.warning(
                    f"Anchor mismatch: student={student_cls.shape[1]}, "
                    f"teacher={teacher_cls.shape[1]}. Truncating to {min_anchors}. "
                    f"(This warning will only appear once.)"
                )
                self._anchor_mismatch_warned = True
            student_cls = student_cls[:, :min_anchors, :]
            teacher_cls = teacher_cls[:, :min_anchors, :]

        T = self.kd_temperature

        # ── KL Divergence with temperature scaling ──
        # Soften both distributions with temperature T
        student_log_probs = F.log_softmax(student_cls / T, dim=-1)
        teacher_probs = F.softmax(teacher_cls / T, dim=-1)

        # KL(teacher || student) — averaged over batch, anchors AND classes
        # NOTE: reduction="batchmean" divides by batch_size only (NOT anchors),
        # giving loss ~20,000 instead of ~1.0. Use "none" + .mean() instead.
        kd_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).mean()

        # Scale by T^2 (standard KD scaling to match gradient magnitudes)
        kd_loss = kd_loss * (T ** 2)

        return kd_loss

    def _do_train(self):
        """Override _do_train to inject KD loss into the training loop.

        This is a targeted override of the forward + loss computation section
        from BaseTrainer._do_train(). We keep all other infrastructure
        (warmup, scheduling, logging, checkpointing, EMA) intact.
        """
        import math
        import time
        import warnings
        import numpy as np
        from torch import distributed as dist
        from ultralytics.utils import TQDM

        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {self.save_dir}\n"
            f"Starting KD training for {self.epochs} epochs...\n"
            f"KD Alpha={self.kd_alpha}, Temperature={self.kd_temperature}"
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # ══════════════════════════════════════════════
                #  FORWARD + KD LOSS (this is where we differ)
                # ══════════════════════════════════════════════
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    images = batch["img"]

                    # Student forward — get raw feature maps (training mode)
                    student_preds = self.model(images)
                    # student_preds is a list of feature maps from Detect head

                    # Standard YOLO loss (box + cls + dfl)
                    std_loss, self.loss_items = unwrap_model(self.model).loss(batch, student_preds)

                    # KD loss — KL divergence on classification logits
                    # Get the raw feature maps for logit extraction
                    if isinstance(student_preds, tuple):
                        student_feats = student_preds[1] if len(student_preds) > 1 else student_preds[0]
                    elif isinstance(student_preds, dict):
                        student_feats = student_preds.get("one2many", student_preds)
                    else:
                        student_feats = student_preds

                    kd_loss = self._compute_kd_loss(student_feats, images)

                    # ── Combine losses ──
                    # L_total = (1 - alpha) * L_yolo + alpha * L_kd
                    combined_loss = (1 - self.kd_alpha) * std_loss.sum() + self.kd_alpha * kd_loss

                    self.loss = combined_loss
                    if RANK != -1:
                        self.loss *= self.world_size

                    # Append KD loss to loss_items for logging (4 components)
                    kd_loss_item = kd_loss.detach().unsqueeze(0)
                    self.loss_items = torch.cat([self.loss_items, kd_loss_item])

                    self.tloss = (
                        self.loss_items if self.tloss is None
                        else (self.tloss * i + self.loss_items) / (i + 1)
                    )

                # Backward
                self.scaler.scale(self.loss).backward()
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)
                self.metrics, self.fitness = self.validate()

            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)

            # Early Stopping
            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        from ultralytics.utils.torch_utils import unset_deterministic
        unset_deterministic()
        self.run_callbacks("teardown")


# ══════════════════════════════════════════════════════════════
#  Main Script
# ══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: Knowledge Distillation — Teacher-Student Pipeline"
    )
    parser.add_argument(
        "--student",
        type=str,
        default="experiments/exp_002_ghost_hybrid_medium3/weights/best.pt",
        help="Path to student model weights (our 867K base model)",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="experiments/exp_004_teacher_yolo11s/weights/best.pt",
        help="Path to teacher model weights (Phase 1 v2 output — YOLO11s)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="KD training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (4 for dual-model on 4GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (640 for KD, 704 would OOM)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument("--kd-alpha", type=float, default=0.5, help="KD loss weight (0-1)")
    parser.add_argument("--kd-temperature", type=float, default=4.0, help="Temperature for soft labels")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Phase 2: Knowledge Distillation")
    print("  Teacher-Student Pipeline")
    print("=" * 60)

    # ── Verify paths ──
    student_path = Path(args.student)
    teacher_path = Path(args.teacher)
    data_yaml = "VOC/voc.yaml"

    assert student_path.exists(), f"Student weights not found: {student_path}"
    assert teacher_path.exists(), f"Teacher weights not found: {teacher_path}"
    assert Path(data_yaml).exists(), f"Data YAML not found: {data_yaml}"

    print(f"\n  KD Configuration:")
    print(f"  ├── Student:     {args.student}")
    print(f"  ├── Teacher:     {args.teacher}")
    print(f"  ├── Dataset:     {data_yaml}")
    print(f"  ├── Epochs:      {args.epochs}")
    print(f"  ├── Batch:       {args.batch}")
    print(f"  ├── Image sz:    {args.imgsz}")
    print(f"  ├── KD Alpha:    {args.kd_alpha}")
    print(f"  ├── Temperature: {args.kd_temperature}")
    print(f"  ├── Device:      cuda:{args.device}")
    print(f"  └── Save to:     experiments/exp_005_kd_student/")

    # ══════════════════════════════════════════════════════
    #  Configure and launch KD Trainer
    # ══════════════════════════════════════════════════════
    overrides = dict(
        # Model — student weights
        model=str(student_path),
        data=data_yaml,

        # Training duration
        epochs=args.epochs,
        patience=20,

        # Batch & image size — CRITICAL for 4GB VRAM with 2 models
        batch=args.batch,
        imgsz=args.imgsz,

        # Optimizer — low LR to preserve student's learned features
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,

        # Warmup
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.005,

        # Augmentation — REDUCED so student focuses on teacher's soft labels
        mosaic=0.5,              # Reduced from 1.0
        close_mosaic=10,         # Disable mosaic last 10 epochs
        mixup=0.0,              # Disabled for KD
        copy_paste=0.0,         # Disabled per project rules

        # Geometric — minimal
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,

        # Color augmentation — light
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        erasing=0.2,

        # Loss weights — unchanged from base
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Performance — optimized for dual-model on 4GB VRAM
        cache="ram",
        device=args.device,
        workers=4,
        amp=True,

        # Logging
        project="experiments",
        name="exp_005_kd_student",
        exist_ok=False,
        verbose=True,
        plots=True,
    )

    # ── Launch KD Trainer ──
    trainer = KDDetectionTrainer(
        overrides=overrides,
        teacher_weights=str(teacher_path),
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
    )
    trainer.train()

    # ══════════════════════════════════════════════════════
    #  Post-Training Multi-Scale Validation
    # ══════════════════════════════════════════════════════
    best_weights = "experiments/exp_005_kd_student/weights/best.pt"

    print("\n" + "=" * 60)
    print("  KD Training Complete!")
    print("  Running Multi-Scale Validation...")
    print("=" * 60)

    if Path(best_weights).exists():
        val_model = YOLO(best_weights)

        for imgsz in [640, 704]:
            print(f"\n{'─' * 40}")
            print(f"  Validation @ {imgsz}×{imgsz}")
            print(f"{'─' * 40}")
            val_results = val_model.val(
                data=data_yaml,
                imgsz=imgsz,
                batch=args.batch,
                device=args.device,
                verbose=True,
            )
            print(f"  mAP50:    {val_results.box.map50:.4f}")
            print(f"  mAP50-95: {val_results.box.map:.4f}")

            # Per-class breakdown
            if hasattr(val_results.box, "ap50"):
                ap50 = val_results.box.ap50
                if len(ap50) >= 2:
                    print(f"  ├── Damaged_1 mAP50:  {ap50[0]:.4f}  (baseline: 0.9124)")
                    print(f"  └── insulator mAP50:  {ap50[1]:.4f}  (baseline: 0.9710)")
    else:
        print(f"  WARNING: Best weights not found at {best_weights}")

    print("\n" + "=" * 60)
    print(f"  Best weights: {best_weights}")
    print(f"  Results CSV:  experiments/exp_005_kd_student/results.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
