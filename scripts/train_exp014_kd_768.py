"""
Experiment 014 — Stage 1: Knowledge Distillation at 768px Resolution
=====================================================================
The KEY untapped idea: original KD (exp_005) happened at 640px, then we
separately fine-tuned to 768px (exp_009). But the teacher's soft labels
at 640px miss fine spatial details.

By running KD at 768px:
  - Teacher (YOLO11s) sees 44% more pixels → richer soft labels
  - Damaged_1 cracks/burns occupy more pixels → better supervision signal
  - Student learns high-res KD representations from the start (no mismatch)
  - We skip the separate resolution-adaptation step

Pipeline:
  exp_009 (768-calibrated backbone, 96.11%) ──→ KD@768 ──→ exp_014_stage1
  Then run train_exp014_stage2_head.py for final head polish.

Expected outcome: push beyond 96.46% by combining:
  1. Resolution advantage (768px proven to be the sweet spot)
  2. Teacher's soft labels (richer at higher resolution)
  3. Better starting point (exp_009 already calibrated for 768)

Hardware: RTX 3050 4GB VRAM — batch=1 for dual-model at 768
  - Teacher: YOLO11s in FP16, frozen, no_grad (~19MB overhead)
  - Student: 867K params, AMP
  - batch=1 is safe; batch=2 may work (try with --batch 2)

Usage:
    python scripts/train_exp014_kd_768.py
    python scripts/train_exp014_kd_768.py --batch 2
    python scripts/train_exp014_kd_768.py --kd-alpha 0.3 --kd-temperature 3.0
"""

import os
import sys
import argparse
from pathlib import Path
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Prevent CUDA memory fragmentation on 4GB VRAM ──
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Project root setup ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import autocast, unwrap_model


# ══════════════════════════════════════════════════════════════
#  Knowledge Distillation Trainer (768px variant)
# ══════════════════════════════════════════════════════════════

class KDDetectionTrainer768(DetectionTrainer):
    """KD trainer optimized for 768px resolution distillation.

    Same KL-divergence approach as the original KD trainer, but tuned for
    768px resolution. Key differences:
      - Handles larger anchor counts at 768 (student: 11520, teacher: 12096)
      - Lower alpha default (0.3) since student starts from a strong 768 base
      - Label smoothing on teacher soft targets for regularization

    Loss: L_total = (1 - alpha) * L_yolo + alpha * T^2 * KL(student || teacher)
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None,
                 teacher_weights=None, kd_alpha=0.3, kd_temperature=3.0,
                 label_smoothing=0.01):
        super().__init__(cfg, overrides, _callbacks)
        self.teacher_weights = teacher_weights
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        self.label_smoothing = label_smoothing
        self.teacher_model = None

    def _setup_train(self):
        """Load and prepare teacher model alongside student."""
        super()._setup_train()

        LOGGER.info(f"\n{'═' * 60}")
        LOGGER.info(f"  EXP 014 — Knowledge Distillation @ 768px")
        LOGGER.info(f"  Teacher: {self.teacher_weights}")
        LOGGER.info(f"  Alpha: {self.kd_alpha} | Temp: {self.kd_temperature} | LabelSmooth: {self.label_smoothing}")
        LOGGER.info(f"{'═' * 60}\n")

        teacher_yolo = YOLO(self.teacher_weights, task="detect")
        self.teacher_model = teacher_yolo.model

        # Move to same device, eval mode, frozen, FP16
        self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.teacher_model.half()  # FP16 to save VRAM

        t_params = sum(p.numel() for p in self.teacher_model.parameters())
        s_params = sum(p.numel() for p in self.model.parameters())
        LOGGER.info(f"  Teacher: {t_params:,} params (FP16, frozen)")
        LOGGER.info(f"  Student: {s_params:,} params")

        # 4-component loss names for training (add kd_loss)
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "kd_loss"

    def get_validator(self):
        """Standard 3-component validator (no KD loss during validation)."""
        from ultralytics.models import yolo
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        validator = yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "kd_loss"
        return validator

    def validate(self):
        """Temporarily use 3-component loss for validation compatibility."""
        saved_loss_names = self.loss_names
        saved_loss_items = self.loss_items
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        if self.loss_items is not None and len(self.loss_items) == 4:
            self.loss_items = self.loss_items[:3]

        metrics, fitness = super().validate()

        self.loss_names = saved_loss_names
        self.loss_items = saved_loss_items
        return metrics, fitness

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Return loss dict with 4 components (including kd_loss)."""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Formatted training progress with KD loss column."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch", "GPU_mem", *self.loss_names, "Instances", "Size",
        )

    @staticmethod
    def _extract_cls_logits(feats, nc, reg_max):
        """Extract classification logits from raw detection feature maps.

        Args:
            feats: List of feature maps [P3, P4, ...], each (B, reg_max*4+nc, H, W)
            nc: Number of classes
            reg_max: Regression max value

        Returns:
            cls_logits: (B, total_anchors, nc)
        """
        no = reg_max * 4 + nc
        batch_size = feats[0].shape[0]
        all_feats = torch.cat([xi.view(batch_size, no, -1) for xi in feats], dim=2)
        _, cls_logits = all_feats.split((reg_max * 4, nc), dim=1)
        return cls_logits.permute(0, 2, 1).contiguous()

    def _compute_kd_loss(self, student_feats, images):
        """Compute KL-divergence distillation loss with optional label smoothing.

        At 768px:
          - Student anchors: P3(96×96) + P4(48×48) = 11520
          - Teacher anchors: P3(96×96) + P4(48×48) + P5(24×24) = 12096
          - Truncate teacher to 11520 (drop P5 predictions)
        """
        student_model = unwrap_model(self.model)
        nc = student_model.model[-1].nc
        student_reg_max = student_model.model[-1].reg_max

        # Teacher forward (FP16, no grad)
        with torch.no_grad():
            teacher_out = self.teacher_model(images.half())
            if isinstance(teacher_out, tuple):
                teacher_feats = teacher_out[1] if len(teacher_out) > 1 else teacher_out[0]
            elif isinstance(teacher_out, dict):
                teacher_feats = teacher_out.get("one2many", list(teacher_out.values())[0])
            else:
                teacher_feats = teacher_out

        teacher_nc = self.teacher_model.model[-1].nc
        teacher_reg_max = self.teacher_model.model[-1].reg_max

        # Extract classification logits
        student_cls = self._extract_cls_logits(student_feats, nc, student_reg_max)
        teacher_cls = self._extract_cls_logits(teacher_feats, teacher_nc, teacher_reg_max).float()

        # Handle anchor mismatch (truncate teacher's P5 predictions)
        min_anchors = min(student_cls.shape[1], teacher_cls.shape[1])
        if student_cls.shape[1] != teacher_cls.shape[1]:
            if not hasattr(self, '_anchor_mismatch_warned'):
                LOGGER.warning(
                    f"Anchor mismatch @768: student={student_cls.shape[1]}, "
                    f"teacher={teacher_cls.shape[1]}. Truncating to {min_anchors}."
                )
                self._anchor_mismatch_warned = True
            student_cls = student_cls[:, :min_anchors, :]
            teacher_cls = teacher_cls[:, :min_anchors, :]

        T = self.kd_temperature

        # Teacher soft targets with optional label smoothing
        teacher_probs = F.softmax(teacher_cls / T, dim=-1)
        if self.label_smoothing > 0:
            n_classes = teacher_probs.shape[-1]
            teacher_probs = (1 - self.label_smoothing) * teacher_probs + \
                            self.label_smoothing / n_classes

        # Student log-probabilities
        student_log_probs = F.log_softmax(student_cls / T, dim=-1)

        # KL divergence (mean over all dimensions)
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none").mean()

        # Standard T^2 scaling
        kd_loss = kd_loss * (T ** 2)

        return kd_loss

    def _do_train(self):
        """Training loop with KD loss injection (same structure as original KD trainer)."""
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
            f"Starting KD@768 training for {self.epochs} epochs...\n"
            f"KD Alpha={self.kd_alpha}, Temperature={self.kd_temperature}, LabelSmoothing={self.label_smoothing}"
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
                ni = i + nb * epoch

                # Warmup
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # ══ FORWARD + KD LOSS ══
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    images = batch["img"]

                    # Student forward → raw feature maps
                    student_preds = self.model(images)

                    # Standard YOLO loss
                    std_loss, self.loss_items = unwrap_model(self.model).loss(batch, student_preds)

                    # Extract feature maps for KD
                    if isinstance(student_preds, tuple):
                        student_feats = student_preds[1] if len(student_preds) > 1 else student_preds[0]
                    elif isinstance(student_preds, dict):
                        student_feats = student_preds.get("one2many", student_preds)
                    else:
                        student_feats = student_preds

                    # KD loss
                    kd_loss = self._compute_kd_loss(student_feats, images)

                    # Combined: L = (1 - alpha) * L_yolo + alpha * L_kd
                    combined_loss = (1 - self.kd_alpha) * std_loss.sum() + self.kd_alpha * kd_loss

                    self.loss = combined_loss
                    if RANK != -1:
                        self.loss *= self.world_size

                    # Append KD loss for logging
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

            # Early stopping
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
#  Main
# ══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="EXP 014 Stage 1: Knowledge Distillation at 768px"
    )
    parser.add_argument(
        "--student", type=str,
        default="experiments/exp_009_finetune_768/weights/best.pt",
        help="Student starting weights (768-calibrated from exp_009)",
    )
    parser.add_argument(
        "--teacher", type=str,
        default="experiments/exp_004_teacher_yolo11s/weights/best.pt",
        help="Teacher weights (YOLO11s, 96.54%%)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="KD training epochs (short fine-tune)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (1 safe, try 2)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument("--kd-alpha", type=float, default=0.3, help="KD loss weight (lower=more YOLO loss)")
    parser.add_argument("--kd-temperature", type=float, default=3.0, help="Softmax temperature")
    parser.add_argument("--label-smoothing", type=float, default=0.01, help="Label smoothing on teacher targets")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  EXP 014 — Stage 1: Knowledge Distillation @ 768px")
    print("  Hypothesis: KD at 768 > KD at 640 + fine-tune to 768")
    print("=" * 60)

    student_path = Path(args.student)
    teacher_path = Path(args.teacher)
    data_yaml = "VOC/voc.yaml"

    assert student_path.exists(), f"Student weights not found: {student_path}"
    assert teacher_path.exists(), f"Teacher weights not found: {teacher_path}"
    assert Path(data_yaml).exists(), f"Data YAML not found: {data_yaml}"

    print(f"\n  Configuration:")
    print(f"  ├── Student:        {args.student}")
    print(f"  ├── Teacher:        {args.teacher}")
    print(f"  ├── Resolution:     768×768")
    print(f"  ├── Epochs:         {args.epochs}")
    print(f"  ├── Batch:          {args.batch}")
    print(f"  ├── KD Alpha:       {args.kd_alpha}")
    print(f"  ├── Temperature:    {args.kd_temperature}")
    print(f"  ├── LabelSmoothing: {args.label_smoothing}")
    print(f"  └── Save to:        experiments/exp_014_kd_768/")

    # ══════════════════════════════════════════════════════
    #  Launch KD Trainer at 768px
    # ══════════════════════════════════════════════════════
    overrides = dict(
        model=str(student_path),
        data=data_yaml,

        # Duration — short fine-tune (student already strong at 768)
        epochs=args.epochs,
        patience=10,

        # Resolution — THE key change
        imgsz=768,
        batch=args.batch,

        # Optimizer — SGD (proven more stable for 768 fine-tuning in exp_009)
        optimizer="SGD",
        lr0=0.001,           # Conservative (student already at 96.11%)
        lrf=0.1,             # Decay to 0.0001
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,

        # Warmup — short (already trained model)
        warmup_epochs=1.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.005,

        # Augmentation — light (preserve strong features, focus on teacher signal)
        mosaic=0.5,           # Reduced (let teacher's signal dominate)
        close_mosaic=5,       # Disable mosaic last 5 epochs
        mixup=0.0,           # Disabled
        copy_paste=0.0,       # Disabled (hurt in previous experiments)

        # Geometric — minimal
        degrees=3.0,
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,

        # Color — light
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,

        # Loss weights — standard
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Hardware — optimized for dual-model at 768 on 4GB VRAM
        cache="ram",
        device=args.device,
        workers=4,
        amp=True,

        # Save
        save_period=5,
        project="experiments",
        name="exp_014_kd_768",
        exist_ok=True,
        verbose=True,
        plots=True,
    )

    trainer = KDDetectionTrainer768(
        overrides=overrides,
        teacher_weights=str(teacher_path),
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
        label_smoothing=args.label_smoothing,
    )
    trainer.train()

    # ══════════════════════════════════════════════════════
    #  Post-Training Validation
    # ══════════════════════════════════════════════════════
    best_weights = "experiments/exp_014_kd_768/weights/best.pt"

    print("\n" + "=" * 60)
    print("  Stage 1 Complete! Running validation...")
    print("=" * 60)

    if Path(best_weights).exists():
        val_model = YOLO(best_weights)

        # Validate at 768 (training resolution)
        print(f"\n{'─' * 40}")
        print(f"  Validation @ 768×768 (no TTA)")
        print(f"{'─' * 40}")
        results_768 = val_model.val(
            data=data_yaml, imgsz=768, batch=2, device=args.device, verbose=True,
        )
        map50 = results_768.box.map50
        map50_95 = results_768.box.map
        print(f"  mAP50:    {map50:.4f}")
        print(f"  mAP50-95: {map50_95:.4f}")
        if hasattr(results_768.box, "ap50") and len(results_768.box.ap50) >= 2:
            print(f"  ├── Damaged_1: {results_768.box.ap50[0]:.4f}  (prev best: 0.9504)")
            print(f"  └── insulator: {results_768.box.ap50[1]:.4f}  (prev best: 0.9714)")

        # Validate with TTA
        print(f"\n{'─' * 40}")
        print(f"  Validation @ 768×768 (with TTA)")
        print(f"{'─' * 40}")
        import torch as _torch
        val_model.model.stride = _torch.tensor([8., 16., 32.])  # TTA stride fix
        results_tta = val_model.val(
            data=data_yaml, imgsz=768, batch=1, device=args.device,
            augment=True, verbose=True,
        )
        print(f"  TTA mAP50:    {results_tta.box.map50:.4f}")
        print(f"  TTA mAP50-95: {results_tta.box.map:.4f}")

        # Comparison
        print(f"\n{'═' * 60}")
        print(f"  COMPARISON vs PREVIOUS BEST:")
        print(f"  ├── exp_009 (KD@640→768ft):  96.11% mAP50")
        print(f"  ├── exp_012 (head-only):     96.46% mAP50")
        print(f"  ├── exp_014 Stage 1 (this):  {map50*100:.2f}% mAP50")
        print(f"  └── exp_014 + TTA:           {results_tta.box.map50*100:.2f}% mAP50")
        print(f"{'═' * 60}")

        if map50 > 0.9646:
            print(f"\n  NEW RECORD! {map50*100:.2f}% > 96.46%")
            print(f"  Run train_exp014_stage2_head.py for final polish!")
        elif map50 > 0.9611:
            print(f"\n  Improved over exp_009! {map50*100:.2f}% > 96.11%")
            print(f"  Run train_exp014_stage2_head.py — head fine-tune may push past 96.46%")
        else:
            print(f"\n  No improvement over exp_009 ({map50*100:.2f}% vs 96.11%)")
            print(f"  Try: --kd-alpha 0.2 or --kd-temperature 2.0")
    else:
        print(f"  WARNING: Best weights not found at {best_weights}")

    print(f"\n  Best weights: {best_weights}")
    print(f"  Results CSV:  experiments/exp_014_kd_768/results.csv")
    print(f"  Next step:    python scripts/train_exp014_stage2_head.py")


if __name__ == "__main__":
    main()
