"""
RayNet v5 Training Script (Triple-M1 architecture, parallel MTL).

Training strategy — one stage, all losses active from epoch 1:

  The v5 Triple-M1 graph fully isolates branches at the stem with
  `s1.detach()` on pose and gaze, so there is no need for sequential
  freeze phases. Landmark gradients steer the shared stem; pose and
  gaze each train their own s2+s3 encoder in parallel; AERI segments
  the iris and eyeball on the gaze feature pyramid. Curriculum over
  epochs is a loss-weight ramp only — nothing is ever frozen.

  Phase 1 (epochs 1-8):   warmup — all losses active, moderate weights
  Phase 2 (epochs 9-16):  main  — full weights + multi-view consistency
  Phase 3 (epochs 17-25): fine-tune — reduced LR, gaze emphasis

  Gradient clipping (max_norm) varies by phase:
    Phase 1: max_norm=5.0 (aggressive, allows large multi-task gradients)
    Phase 2+: max_norm=2.0 (conservative, prevents gaze/pose interference)

Usage:
    # Single GPU
    python -m RayNet.train --profile t4 --mds_streaming \
        --mds_train s3://gazegene/train --mds_val s3://gazegene/val ...

    # Kaggle 2× Tesla T4 (single node, multi-GPU)
    accelerate launch --multi_gpu --num_processes 2 \
        -m RayNet.train --profile kaggle_t4x2 --mds_streaming ...

    # Two machines on the same network (see RayNet/hardware_profiles.py)
    accelerate launch --multi_gpu --num_machines 2 --num_processes 2 \
        --machine_rank <0|1> --main_process_ip $MAIN_IP --main_process_port 29500 \
        -m RayNet.train --profile multi_node_t4 --mds_streaming ...
"""

import argparse
import os
import csv
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.amp import GradScaler, autocast
import numpy as np
from datetime import datetime

from RayNet.dataset import create_dataloaders
from RayNet.hardware_profiles import (
    HARDWARE_PROFILES,
    AMP_DTYPE_MAP,
    apply_hardware_profile,
    setup_hardware,
    build_accelerator,
)
from RayNet.losses import total_loss
from RayNet.multiview_loss import multiview_consistency_loss
from RayNet.raynet_v5 import create_raynet_v5

log = logging.getLogger(__name__)


# ============== Staged Training Configuration ==============
#
# -----------------------------------------------------------------------------
# IMPORTANT — meaning of the `multiview` flag in phase configs below:
#
#   `multiview` does NOT control whether CrossViewAttention runs. It ONLY
#   gates the auxiliary `multiview_consistency_loss` (which adds
#   `lam_reproj` gaze-consistency and `lam_mask` shape terms).
#
#   CrossViewAttention itself is *always active during training* when
#   --mv_groups > 1, regardless of the phase's `multiview` setting. This is
#   because the MDS dataloader is constructed once before the phase loop,
#   producing 9-grouped multi-view batches, and `active_n_views` is
#   hardcoded to 9 at train.py:active_n_views (only suppressed by the
#   --no_multiview CLI ablation flag).
#
#   Validation uses n_views=1, so CrossViewAttention, camera embeddings,
#   and the PoseEncoder features ALL bypass during validation. This creates
#   a deliberate train/val asymmetry: training sees fused 9-view geometry,
#   val sees single-view — val metrics are therefore a strict lower bound
#   on what the model is actually capable of at inference with multi-view.
#
#   If you truly want a "no multi-view fusion" phase, pass --no_multiview
#   on the CLI (not the `multiview: False` config flag).
# -----------------------------------------------------------------------------

PHASE_CONFIG = {
    # ---- Phase 1: warmup — all losses active, moderate weights ----
    # Landmark owns the stem and converges fastest; pose + gaze train
    # through their detached s1 clone so the noisy early gaze gradient
    # can't corrupt the landmark-owned low-level encoder. AERI segments
    # supervise the gaze branch from epoch 1 so the learned eyeball
    # attention map is meaningful by the time gaze loss matters.
    1: {
        'epochs': (1, 8),
        'lam_lm': 1.0,
        'lam_gaze': 0.5,
        'lam_eyeball': 0.3,
        'lam_pupil': 0.3,
        'lam_geom_angular': 0.1,
        'lam_ray': 0.0,
        'lam_reproj': 0.0,
        'lam_mask': 0.0,
        'lam_pose': 0.5,
        'lam_trans': 0.5,
        'lam_iris_seg': 0.5,
        'lam_eyeball_seg': 0.5,
        'lr': 5e-4,
        'sigma': 2.0,
        'multiview': False,
        'description': 'V5-P1: warmup (lm + pose + gaze + AERI, moderate)',
    },
    # ---- Phase 2: main — full weights + multi-view consistency ----
    2: {
        'epochs': (9, 16),
        'lam_lm': 1.0,
        'lam_gaze': 1.0,
        'lam_gaze_sv': 0.3,   # single-view pathway supervision (pre-CrossViewAttn)
        'lam_eyeball': 0.5,
        'lam_pupil': 0.5,
        'lam_geom_angular': 0.2,
        'lam_ray': 0.2,
        'lam_reproj': 0.1,
        'lam_mask': 0.05,
        'lam_pose': 1.0,
        'lam_trans': 1.0,
        'lam_iris_seg': 0.5,
        'lam_eyeball_seg': 0.5,
        'lr': 3e-4,
        'sigma': 1.5,
        'multiview': True,
        'description': 'V5-P2: full losses + multi-view consistency',
    },
    # ---- Phase 3: fine-tune — reduced LR, gaze emphasis ----
    # lam_reproj=0: gaze_consist hit its kappa-angle floor (~0.174) in P2 and
    # is now generating wrong gradients that bias predictions toward the
    # kappa-averaged mean direction. Disabling it removes that interference.
    3: {
        'epochs': (17, 25),
        'lam_lm': 0.5,
        'lam_gaze': 1.0,
        'lam_gaze_sv': 0.5,   # higher SV weight in fine-tune to close train/val gap
        'lam_eyeball': 0.5,
        'lam_pupil': 0.5,
        'lam_geom_angular': 0.3,
        'lam_ray': 0.3,
        'lam_reproj': 0.0,    # disabled: gaze_consist floor causes wrong gradients
        'lam_mask': 0.05,
        'lam_pose': 0.5,
        'lam_trans': 0.5,
        'lam_iris_seg': 0.3,
        'lam_eyeball_seg': 0.3,
        'lr': 1e-4,
        'sigma': 1.0,
        'multiview': True,
        'description': 'V5-P3: fine-tune (lower LR, gaze emphasis, no gaze_consist)',
    },
}


def get_phase(epoch):
    """Return the training phase for a given epoch."""
    for phase, cfg in PHASE_CONFIG.items():
        start, end = cfg['epochs']
        if start <= epoch <= end:
            return phase
    # Fallback: return the last defined phase (not hardcoded 3)
    return max(PHASE_CONFIG.keys())


def get_phase_config(epoch):
    """Get loss weights and hyperparams for the current epoch (returns a copy)."""
    phase = get_phase(epoch)
    return dict(PHASE_CONFIG[phase])


def _unwrap_raynet(model):
    """Unwrap DDP / torch.compile to reach the underlying RayNetV5."""
    m = model
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def _filter_compatible_state(src_sd, target_sd):
    """
    Keep only (key, tensor) pairs from src_sd that have matching shape
    in target_sd. Returns (filtered_sd, dropped_keys). Used to bridge
    cross-stage forks where the model architecture changed between
    runs (e.g. old Stage 1 checkpoint into new eye-crop Stage 2 code).
    """
    filtered = {}
    dropped = []
    for k, v in src_sd.items():
        tgt = target_sd.get(k)
        if tgt is not None and hasattr(tgt, 'shape') and tgt.shape == v.shape:
            filtered[k] = v
        else:
            dropped.append(k)
    return filtered, dropped


def _optimizer_state_compatible(saved_state, new_optimizer):
    """
    Check whether a saved optimizer state_dict can be loaded into
    `new_optimizer`. PyTorch validates that every param_group has the
    same number of parameters as the saved group; if the model changed
    shape (e.g. new branches added) that assertion fails.

    Returns True iff groups match in count AND in per-group param count.
    """
    saved_groups = saved_state.get('param_groups', [])
    new_groups = new_optimizer.state_dict()['param_groups']
    if len(saved_groups) != len(new_groups):
        return False
    for sg, ng in zip(saved_groups, new_groups):
        if len(sg.get('params', [])) != len(ng.get('params', [])):
            return False
    return True


# ============== EMA helpers ==============

def _ema_copy_buffers(src_module, ema_module):
    """
    Copy non-parameter buffers (BN running_mean/running_var, etc.) from
    the live model into the EMA model.

    AveragedModel only averages parameters by default. If we left the
    EMA model's BN buffers at their init-time values, validation through
    the EMA would use stale BN statistics and silently degrade. The
    standard fix (timm's ModelEmaV2, MoCo, etc.) is to mirror buffers
    directly rather than EMA them — BN stats are already a running
    average inside the live model, so smoothing them again is double
    momentum and slows BN tracking unnecessarily.
    """
    src_buffers = dict(src_module.named_buffers())
    for name, buf in ema_module.named_buffers():
        # AveragedModel wraps the model as `self.module.<name>`; strip
        # the leading "module." prefix when looking up in src.
        key = name[len("module."):] if name.startswith("module.") else name
        if key in src_buffers:
            buf.data.copy_(src_buffers[key].data)


# ============== Training Loop ==============

def train_one_epoch(model, train_loader, optimizer, device, epoch, cfg,
                    scaler=None, grad_accum_steps=1, amp_enabled=False,
                    amp_dtype=torch.float16, batch_csv_writer=None,
                    n_views=1, ema_model=None):
    """Run one training epoch with AMP and gradient accumulation support."""
    model.train()
    use_multiview = cfg.get('multiview', False)
    if not amp_enabled:
        amp_dtype = torch.float32

    running_losses = {
        'total': 0.0, 'landmark': 0.0, 'angular': 0.0, 'angular_deg': 0.0,
        'reproj': 0.0, 'mask': 0.0, 'ray_target': 0.0, 'pose': 0.0,
        'translation': 0.0,
        # V5-specific
        'eyeball_center': 0.0, 'pupil_center': 0.0, 'geom_angular': 0.0,
        'iris_seg': 0.0, 'eyeball_seg': 0.0,
    }
    n_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        # We don't process the samples that are skipped when samples_per_subject is used

        # To prevent accumulation on missing values we use this flag to not process the filtered samples
        has_accumulated = False
        if not batch or len(batch) == 0:
            continue

        # UINT 8 to tensor of floats normalized from 0.0 to 1.0
        images = batch['image'].to(device, non_blocking=True).float().div_(255.0)
        gt_landmarks = batch['landmark_coords'].to(device, non_blocking=True)
        gt_optical_axis = batch['optical_axis'].to(device, non_blocking=True)

        # Camera extrinsics for geometry-conditioned attention
        R_cam = batch['R_cam'].to(device, non_blocking=True)
        T_cam = batch['T_cam'].to(device, non_blocking=True)

        # GT head pose for auxiliary pose loss (not passed to model)
        gt_head_R = batch.get('head_R')
        if gt_head_R is not None:
            gt_head_R = gt_head_R.to(device, non_blocking=True)

        gt_head_t = batch.get('head_t')
        if gt_head_t is not None:
            gt_head_t = gt_head_t.to(device, non_blocking=True)

        # Intrinsic-Delta face bbox GT for the MAGE BoxEncoder (v5 only).
        # Shape (B, 3) = [x_p ∈ [-1,1], y_p ∈ [-1,1], L_x > 0].
        face_bbox_gt = batch.get('face_bbox_gt')
        if face_bbox_gt is not None:
            face_bbox_gt = face_bbox_gt.to(device, non_blocking=True)

        # AERI ground-truth masks for segmentation supervision (uint8
        # {0, 255} @ 56x56).
        gt_iris_mask = batch.get('iris_mask')
        if gt_iris_mask is not None:
            gt_iris_mask = gt_iris_mask.to(device, non_blocking=True)
        gt_eyeball_mask = batch.get('eyeball_mask')
        if gt_eyeball_mask is not None:
            gt_eyeball_mask = gt_eyeball_mask.to(device, non_blocking=True)

        mv_components = None
        # Forward pass with AMP autocast
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            predictions = model(images, n_views=n_views,
                                R_cam=R_cam, T_cam=T_cam,
                                face_bbox=face_bbox_gt)

            pred_hm = predictions['landmark_heatmaps']
            pred_coords = predictions['landmark_coords']
            pred_gaze = predictions['gaze_vector']

            feat_H, feat_W = pred_hm.shape[2], pred_hm.shape[3]

            gt_eyeball = batch['eyeball_center_3d'].to(device, non_blocking=True)
            gt_pupil = batch.get('pupil_center_3d')
            if gt_pupil is not None:
                gt_pupil = gt_pupil.to(device, non_blocking=True)

            loss, components = total_loss(
                pred_hm, pred_coords, pred_gaze,
                gt_landmarks, gt_optical_axis,
                feat_H, feat_W,
                lam_lm=cfg['lam_lm'],
                lam_gaze=cfg['lam_gaze'],
                sigma=cfg['sigma'],
                lam_eyeball=cfg.get('lam_eyeball', 0.0),
                pred_eyeball=predictions.get('eyeball_center'),
                gt_eyeball=gt_eyeball,
                lam_pupil=cfg.get('lam_pupil', 0.0),
                pred_pupil=predictions.get('pupil_center'),
                gt_pupil=gt_pupil,
                lam_geom_angular=cfg.get('lam_geom_angular', 0.0),
                lam_ray=cfg.get('lam_ray', 0.0),
                eyeball_center=gt_eyeball,
                gaze_target=batch['gaze_target'].to(device, non_blocking=True),
                gaze_depth=batch['gaze_depth'].to(device, non_blocking=True),
                lam_pose=cfg.get('lam_pose', 0.0),
                pred_pose_6d=predictions.get('pred_pose_6d'),
                gt_head_R=gt_head_R,
                lam_trans=cfg.get('lam_trans', 0.0),
                pred_pose_t=predictions.get('pred_pose_t'),
                gt_head_t=gt_head_t,
                lam_iris_seg=cfg.get('lam_iris_seg', 0.0),
                pred_iris_mask_logits=predictions.get('iris_mask_logits'),
                gt_iris_mask=gt_iris_mask,
                lam_eyeball_seg=cfg.get('lam_eyeball_seg', 0.0),
                pred_eyeball_mask_logits=predictions.get('eyeball_mask_logits'),
                gt_eyeball_mask=gt_eyeball_mask,
            )

            # Single-view pathway supervision: direct gaze L1 on pre-CrossViewAttn
            # features so the GeometricGazeHead is also optimized for val
            # (which bypasses CrossViewAttention entirely).
            lam_gaze_sv = cfg.get('lam_gaze_sv', 0.0)
            if lam_gaze_sv > 0:
                gaze_sv = predictions.get('gaze_vector_sv')
                if gaze_sv is not None:
                    from RayNet.losses import gaze_loss as _gaze_loss
                    sv_loss = _gaze_loss(gaze_sv, gt_optical_axis)
                    if torch.isfinite(sv_loss):
                        loss = loss + lam_gaze_sv * sv_loss

            # Multi-view consistency loss (Phase 2+)
            # Ray-based: works with unit gaze vectors, numerically stable.
            if use_multiview:
                mv_loss, mv_components = multiview_consistency_loss(
                    pred_gaze,
                    pred_coords,
                    R_cam,
                    lam_gaze_consist=cfg['lam_reproj'],
                    lam_shape=cfg['lam_mask'],
                )
                if torch.isfinite(mv_loss):
                    mv_weight = min(1.0, epoch / 10.0)  # smooth ramp
                    loss = loss + mv_weight * mv_loss

            # Scale loss by accumulation steps
            loss = loss / grad_accum_steps

        # Backward pass with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # using this flag only when a batch is valid to process
        has_accumulated = True

        # max_norm for multi-task learning scenario like ours requires a value>1
        # therefore we set max_norm=5.0 for more aggressive training strategy
        current_phase_num = get_phase(epoch)
        if current_phase_num >= 2:
            max_norm = 2.0
        else:
            max_norm = 5.0
        # Optimizer step at accumulation boundary
        if (step + 1) % grad_accum_steps == 0:
            if has_accumulated:
                did_step = False
                if scaler is not None:
                    # FP16 path: scaler handles inf/nan detection internally
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    did_step = True
                else:
                    # FP32 / BF16 path: manually check grad norm and skip step
                    # if non-finite (BF16 has no scaler to catch overflow/nan)
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=max_norm)
                    if torch.isfinite(total_norm):
                        optimizer.step()
                        did_step = True
                    else:
                        log.warning("Epoch %d batch %d: non-finite grad norm "
                                    "(%s), skipping optimizer step",
                                    epoch, step + 1, total_norm.item())
                if did_step and ema_model is not None:
                    raynet = _unwrap_raynet(model)
                    ema_model.update_parameters(raynet)
                    _ema_copy_buffers(raynet, ema_model)
                optimizer.zero_grad()
                has_accumulated = False
            else:
                optimizer.zero_grad()

        # Accumulate metrics (use unscaled loss); skip nan to prevent poisoning epoch avg
        batch_total = components['total_loss'].item()
        if batch_total != batch_total:  # nan check
            log.warning("Epoch %d batch %d: nan loss detected, skipping metrics", epoch, step + 1)
            continue
        running_losses['total'] += batch_total
        running_losses['landmark'] += components['landmark_loss'].item()
        running_losses['angular'] += components['angular_loss'].item()
        running_losses['angular_deg'] += components['angular_loss_deg'].item()
        if mv_components is not None:
            running_losses['reproj'] += mv_components['gaze_consist_loss'].item()
            running_losses['mask'] += mv_components['shape_loss'].item()
        if 'ray_target_loss' in components:
            running_losses['ray_target'] += components['ray_target_loss'].item()
        if 'pose_loss' in components:
            running_losses['pose'] += components['pose_loss'].item()
        if 'translation_loss' in components:
            running_losses['translation'] += components['translation_loss'].item()
        if 'eyeball_center_loss' in components:
            running_losses['eyeball_center'] += components['eyeball_center_loss'].item()
        if 'pupil_center_loss' in components:
            running_losses['pupil_center'] += components['pupil_center_loss'].item()
        if 'geometric_angular_loss' in components:
            running_losses['geom_angular'] += components['geometric_angular_loss'].item()
        if 'iris_seg_loss' in components:
            running_losses['iris_seg'] += components['iris_seg_loss'].item()
        if 'eyeball_seg_loss' in components:
            running_losses['eyeball_seg'] += components['eyeball_seg_loss'].item()
        n_batches += 1

        # Per-batch CSV logging (high granularity)
        if batch_csv_writer is not None:
            ray_val = components.get('ray_target_loss',
                                     torch.tensor(0.0)).item()
            pose_val = components.get('pose_loss',
                                      torch.tensor(0.0)).item()
            trans_val = components.get('translation_loss',
                                       torch.tensor(0.0)).item()
            iris_seg_val = components.get('iris_seg_loss',
                                          torch.tensor(0.0)).item()
            eye_seg_val = components.get('eyeball_seg_loss',
                                         torch.tensor(0.0)).item()
            batch_csv_writer.writerow([
                epoch, step + 1,
                f"{components['total_loss'].item():.6f}",
                f"{components['landmark_loss'].item():.6f}",
                f"{components['angular_loss_deg'].item():.4f}",
                f"{running_losses['reproj'] / max(n_batches, 1):.6f}",
                f"{running_losses['mask'] / max(n_batches, 1):.6f}",
                f"{ray_val:.6f}",
                f"{pose_val:.6f}",
                f"{trans_val:.6f}",
                f"{iris_seg_val:.6f}",
                f"{eye_seg_val:.6f}",
                f"{optimizer.param_groups[0]['lr']:.8f}",
            ])

        # Progress logging
        total_batches = len(train_loader)
        if step == 0 or (step + 1) % max(1, total_batches // 10) == 0 or step + 1 == total_batches:
            avg_loss = running_losses['total'] / max(n_batches, 1)
            print(f"  [Epoch {epoch}] batch {step+1}/{total_batches} "
                  f"loss={avg_loss:.4f}", flush=True)

    # Flush any remaining gradients from incomplete accumulation
    if n_batches % grad_accum_steps != 0:
        if has_accumulated:
            did_step = False
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                scaler.step(optimizer)
                scaler.update()
                did_step = True
            else:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_norm)
                if torch.isfinite(total_norm):
                    optimizer.step()
                    did_step = True
            if did_step and ema_model is not None:
                raynet = _unwrap_raynet(model)
                ema_model.update_parameters(raynet)
                _ema_copy_buffers(raynet, ema_model)
            optimizer.zero_grad()

    for k in running_losses:
        running_losses[k] /= max(n_batches, 1)

    return running_losses


@torch.no_grad()
def validate(model, val_loader, device, epoch, cfg, amp_enabled=False,
             amp_dtype=torch.float16, n_views=1):
    """Run validation."""
    model.eval()
    if not amp_enabled:
        amp_dtype = torch.float32

    running_losses = {
        'total': 0.0, 'landmark': 0.0, 'angular': 0.0, 'angular_deg': 0.0,
        'landmark_px': 0.0, 'pose': 0.0, 'ray': 0.0, 'translation': 0.0,
        'iris_seg': 0.0, 'eyeball_seg': 0.0,
    }
    n_batches = 0

    for batch in val_loader:

        #When using samples_per_subject in MDS shard loading we'll skip empty batches
        if not batch:
            continue

        images = batch['image'].to(device, non_blocking=True).float().div_(255.0)
        gt_landmarks = batch['landmark_coords'].to(device, non_blocking=True)
        gt_optical_axis = batch['optical_axis'].to(device, non_blocking=True)
        gt_landmarks_px = batch['landmark_coords_px'].to(device, non_blocking=True)

        R_cam = batch['R_cam'].to(device, non_blocking=True)
        T_cam = batch['T_cam'].to(device, non_blocking=True)

        face_bbox_gt = batch.get('face_bbox_gt')
        if face_bbox_gt is not None:
            face_bbox_gt = face_bbox_gt.to(device, non_blocking=True)

        gt_iris_mask = batch.get('iris_mask')
        if gt_iris_mask is not None:
            gt_iris_mask = gt_iris_mask.to(device, non_blocking=True)
        gt_eyeball_mask = batch.get('eyeball_mask')
        if gt_eyeball_mask is not None:
            gt_eyeball_mask = gt_eyeball_mask.to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            predictions = model(images, n_views=n_views,
                                R_cam=R_cam, T_cam=T_cam,
                                face_bbox=face_bbox_gt)

            pred_hm = predictions['landmark_heatmaps']
            pred_coords = predictions['landmark_coords']
            pred_gaze = predictions['gaze_vector']

            feat_H, feat_W = pred_hm.shape[2], pred_hm.shape[3]

            loss, components = total_loss(
                pred_hm, pred_coords, pred_gaze,
                gt_landmarks, gt_optical_axis,
                feat_H, feat_W,
                lam_lm=cfg['lam_lm'],
                lam_gaze=cfg['lam_gaze'],
                sigma=cfg['sigma'],
                lam_iris_seg=cfg.get('lam_iris_seg', 0.0),
                pred_iris_mask_logits=predictions.get('iris_mask_logits'),
                gt_iris_mask=gt_iris_mask,
                lam_eyeball_seg=cfg.get('lam_eyeball_seg', 0.0),
                pred_eyeball_mask_logits=predictions.get('eyeball_mask_logits'),
                gt_eyeball_mask=gt_eyeball_mask,
            )

        img_size = images.shape[-1]
        feat_size = feat_H
        pred_coords_px = pred_coords * (img_size / feat_size)
        lm_px_error = torch.mean(torch.norm(pred_coords_px - gt_landmarks_px, dim=-1))

        batch_total = components['total_loss'].item()
        if batch_total != batch_total:  # nan check
            continue
        running_losses['total'] += batch_total
        running_losses['landmark'] += components['landmark_loss'].item()
        running_losses['angular'] += components['angular_loss'].item()
        running_losses['angular_deg'] += components['angular_loss_deg'].item()
        running_losses['landmark_px'] += lm_px_error.item()
        pose_val = components.get('pose_loss')
        if pose_val is not None:
            running_losses['pose'] += pose_val.item()
        ray_val = components.get('ray_target_loss')
        if ray_val is not None:
            running_losses['ray'] += ray_val.item()
        trans_val = components.get('translation_loss')
        if trans_val is not None:
            running_losses['translation'] += trans_val.item()
        iris_seg_val = components.get('iris_seg_loss')
        if iris_seg_val is not None:
            running_losses['iris_seg'] += iris_seg_val.item()
        eye_seg_val = components.get('eyeball_seg_loss')
        if eye_seg_val is not None:
            running_losses['eyeball_seg'] += eye_seg_val.item()
        n_batches += 1

    for k in running_losses:
        running_losses[k] /= max(n_batches, 1)

    return running_losses


# ============== Main Training ==============

def _build_run_config(args, hw):
    """Collect all training configuration into a single dict for metadata."""
    return {
        'profile': args.profile,
        'epochs': args.epochs,
        'eye': args.eye,
        'hardware': hw,
        'phase_config': {
            str(k): {kk: vv for kk, vv in v.items() if kk != 'description'}
            for k, v in PHASE_CONFIG.items()
        },
        'data_dir': getattr(args, 'data_dir', None),
        'core_backbone_weight_path': args.core_backbone_weight_path,
        'started_at': datetime.now().isoformat(),
    }


def _init_checkpoint_manager(args):
    """Create a CheckpointManager if MinIO checkpointing is enabled."""
    if not args.ckpt_bucket:
        return None
    from RayNet.streaming.checkpoint import CheckpointManager
    return CheckpointManager(
        bucket=args.ckpt_bucket,
        prefix=args.ckpt_prefix,
        run_id=args.run_id,
        endpoint=args.minio_endpoint,
        local_cache=os.path.join(args.output_dir, 'ckpt_cache'),
        save_local_copy=True,
    )


def train(args):
    """Main training function."""
    # Auto-cap --epochs to the last epoch defined in PHASE_CONFIG
    stage_max_epoch = max(cfg['epochs'][1] for cfg in PHASE_CONFIG.values())
    if args.epochs > stage_max_epoch:
        print(f"[auto-cap] --epochs={args.epochs} exceeds the phase "
              f"schedule max epoch ({stage_max_epoch}). Capping to "
              f"{stage_max_epoch}.")
        args.epochs = stage_max_epoch

    # HuggingFace Accelerate — handles DDP wrapping, rank-aware dataloaders,
    # and multi-node orchestration. AMP (autocast + GradScaler) is still
    # managed manually below; Accelerator is configured with
    # mixed_precision='no' so it does not interfere.
    accelerator = build_accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"Device: {device}")
        print(f"Distributed: num_processes={accelerator.num_processes}, "
              f"process_index={accelerator.process_index}")

    # Apply hardware profile
    hw = apply_hardware_profile(args)
    if is_main:
        setup_hardware(hw, device)
    else:
        # Non-main processes still need TF32 flags on their own CUDA context,
        # but skip the GPU-info print.
        if hw['tf32'] and device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')

    # Resolve AMP dtype from profile string
    amp_dtype = AMP_DTYPE_MAP.get(hw['amp_dtype'], torch.float16)

    if is_main:
        print(f"Profile: {args.profile}")
        print(f"  Batch size (per-process): {hw['batch_size']} "
              f"(effective/proc: {hw['batch_size'] * hw['grad_accum_steps']}, "
              f"global: {hw['batch_size'] * hw['grad_accum_steps'] * accelerator.num_processes})")
        print(f"  AMP: {hw['amp']} ({hw['amp_dtype']})")
        print(f"  Gradient accumulation: {hw['grad_accum_steps']} steps")
        print(f"  torch.compile: {hw['compile_model']}")

    # In distributed mode, each rank would otherwise auto-generate its own
    # timestamp inside CheckpointManager and the paths would diverge.
    # Generate on rank 0 and broadcast so every rank writes to the same
    # checkpoint directory.
    if (accelerator.num_processes > 1 and args.ckpt_bucket
            and not args.run_id):
        from accelerate.utils import broadcast_object_list
        run_id_holder = [
            datetime.now().strftime('run_%Y%m%d_%H%M%S') if is_main else None
        ]
        broadcast_object_list(run_id_holder, from_process=0)
        args.run_id = run_id_holder[0]

    # Checkpoint manager (MinIO) — created on every rank so that resume
    # loads identical optimizer state across ranks. Save/upload calls are
    # gated on is_main_process further down.
    ckpt_mgr = _init_checkpoint_manager(args)
    if ckpt_mgr is not None and is_main:
        print(f"  MinIO checkpoints: s3://{args.ckpt_bucket}/{args.ckpt_prefix}/{ckpt_mgr.run_id}/")

    # Create output directory — main process only. All log and local
    # checkpoint writes are gated on is_main, so ranks without an
    # output_dir simply skip those writes.
    if is_main:
        if ckpt_mgr is not None:
            output_dir = os.path.join(args.output_dir, ckpt_mgr.run_id)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    # Create model
    model = create_raynet_v5(
        backbone_weight_path=args.core_backbone_weight_path,
        n_landmarks=14,
    )

    if hw['compile_model'] and hasattr(torch, 'compile'):
        if is_main:
            print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # AMP scaler — ONLY for float16. BF16 has FP32 range, so loss scaling is
    # unnecessary and actively harmful (the scaler misdetects inf gradients
    # and silently skips every optimizer step, freezing training at init).
    use_scaler = hw['amp'] and hw.get('amp_dtype', 'float16') == 'float16'
    scaler = GradScaler('cuda', enabled=True) if use_scaler else None

    # Wrap model in DDP (no-op on single process). accelerator.prepare
    # also broadcasts rank-0 weights to all ranks, so after this call all
    # ranks start from identical parameters.
    model = accelerator.prepare(model)

    # EMA shadow model for validation. Tracks a moving average of the
    # live model's parameters; we run validation through it so the
    # reported metric reflects a smoothed weight trajectory rather than
    # the noisy step-by-step weights. Created from the unwrapped model
    # so the EMA copy carries no DDP/torch.compile machinery — it's a
    # plain RayNetV5 used only for forward passes during eval. BN
    # buffers are mirrored from the live model after every optimizer
    # step (see _ema_copy_buffers); only parameters are EMA'd.
    ema_model = None
    ema_restored_from_ckpt = False
    if args.ema_decay > 0:
        ema_model = AveragedModel(
            accelerator.unwrap_model(model),
            multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay),
        )
        ema_model.to(device)
        ema_model.eval()
        if is_main:
            print(f"EMA enabled (decay={args.ema_decay})")

    # --- Data loading ---
    # Multi-view is always active. Train + val both use 9-grouped batches so
    # CrossViewAttention, cam_embed, and PoseEncoder fusion are exercised at
    # validation (without this, val gaze metrics are an uninformative
    # single-view lower bound).
    #
    # For MDS streaming: StreamingDataset shards by RANK/WORLD_SIZE env
    # vars that `accelerate launch` sets, so we DON'T wrap it.
    # For local loaders: accelerator.prepare injects a DistributedSampler.
    if args.mds_streaming:
        train_loader_mv, val_loader = _create_mds_mv_loader(args, hw)
    else:
        _, train_loader_mv, val_loader = _create_local_loaders(args, hw)
        train_loader_mv, val_loader = accelerator.prepare(
            train_loader_mv, val_loader)

    # Record config in checkpoint metadata (main only — no rank divergence).
    if is_main:
        run_config = _build_run_config(args, hw)
        if ckpt_mgr is not None:
            ckpt_mgr.set_config(run_config)

    # CSV loggers — main process only. Non-main ranks keep these as None
    # and all write sites check `is not None` before writing.
    csv_path = batch_csv_path = None
    csv_file = batch_csv_file = None
    csv_writer = batch_csv_writer = None
    if is_main:
        csv_path = os.path.join(output_dir, 'training_log.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'epoch', 'phase', 'lr',
            'train_total', 'train_landmark', 'train_angular_deg',
            'train_reproj', 'train_mask', 'train_ray_target', 'train_pose',
            'train_translation', 'train_iris_seg', 'train_eyeball_seg',
            'val_total', 'val_landmark', 'val_angular_deg', 'val_landmark_px',
            'val_iris_seg', 'val_eyeball_seg',
        ])

        batch_csv_path = os.path.join(output_dir, 'batch_log.csv')
        batch_csv_file = open(batch_csv_path, 'w', newline='')
        batch_csv_writer = csv.writer(batch_csv_file)
        batch_csv_writer.writerow([
            'epoch', 'batch', 'loss', 'landmark', 'angular_deg',
            'gaze_consist', 'shape', 'ray_target', 'pose', 'translation',
            'iris_seg', 'eyeball_seg', 'lr',
        ])

    # Training loop with phase transitions
    best_val_loss = float('inf')
    current_phase = 0
    start_epoch = 1

    # --- Resume from checkpoint ---
    # We need a dummy optimizer/scheduler for resume, so we create them
    # for the first phase first, then resume overwrites.
    optimizer = None
    scheduler = None

    if args.resume and ckpt_mgr is not None:
        if is_main:
            print(f"Resuming from run {ckpt_mgr.run_id} ...")
        # We need optimizer & scheduler to exist before resume_state.
        # Create them for the starting phase; resume will overwrite state.
        # Each rank runs this independently so all ranks end up with the
        # same optimizer/scheduler state (they read the same checkpoint).
        resume_phase_cfg = get_phase_config(1)
        optimizer = optim.AdamW(model.parameters(), lr=resume_phase_cfg['lr'],
                                betas=(0.5, 0.95), weight_decay=1e-4)
        phase_start, phase_end = resume_phase_cfg['epochs']
        scheduler = CosineAnnealingLR(optimizer, T_max=phase_end - phase_start + 1)

        resume_file = getattr(args, 'resume_from', None)
        # resume_state loads into the unwrapped model so DDP wrapping
        # doesn't interfere with state_dict key names.
        unwrapped = accelerator.unwrap_model(model)
        # Serialize the MinIO download: rank 0 downloads the checkpoint
        # to the local cache first, then non-main ranks cache-hit. Doing
        # this concurrently races on MinIO's .part.minio temp file.
        with accelerator.main_process_first():
            start_epoch, resume_ckpt = ckpt_mgr.resume_state(
                unwrapped, optimizer, scheduler=scheduler, scaler=scaler,
                map_location=device, filename=resume_file,
            )
        current_phase = get_phase(resume_ckpt['epoch'])
        best_val_loss = resume_ckpt.get('val_metrics', {}).get('total', best_val_loss)
        # Restore EMA shadow weights so resume picks up the smoothed
        # trajectory rather than re-EMAing from current weights (which
        # would discard ~all previous decay history and bias val toward
        # noisy weights for a few hundred steps).
        if ema_model is not None and 'ema_state_dict' in resume_ckpt:
            ema_model.load_state_dict(resume_ckpt['ema_state_dict'])
            ema_restored_from_ckpt = True
            if is_main:
                print("  Restored EMA shadow weights from checkpoint")
        if is_main:
            print(f"  Resumed at epoch {start_epoch} (phase {current_phase}), "
                  f"best_val_loss={best_val_loss:.4f}")

        # Guard: empty training range. Happens when resuming a completed run
        # without extending --epochs. Fail loudly instead of silently exiting.
        if start_epoch > args.epochs:
            raise RuntimeError(
                f"Cannot resume: checkpoint is already at epoch "
                f"{resume_ckpt['epoch']}, so training would start at epoch "
                f"{start_epoch}, but --epochs={args.epochs}. Increase --epochs "
                f"to a value > {resume_ckpt['epoch']}, or use --warmstart_from "
                f"{ckpt_mgr.run_id} to start a new stage from these weights."
            )

    # --- Fork from a different run (new run_id, FULL training state) ---
    # Like --resume, but writes under a fresh run_id so the source run is
    # never overwritten. Loads model + optimizer + scheduler + scaler +
    # epoch counter + best_val_loss, then continues training under
    # ckpt_mgr.run_id (auto-generated unless --run_id was passed).
    # Use this to extend training past the original epoch budget or to
    # branch off for hyperparameter variations while keeping the baseline
    # intact.
    if args.fork_from and ckpt_mgr is not None:
        if is_main:
            print(f"Forking from run {args.fork_from} "
                  f"({args.fork_checkpoint}) into new run {ckpt_mgr.run_id} ...")
        # Serialize MinIO download across ranks — see note at the resume
        # site above for the race condition this avoids.
        with accelerator.main_process_first():
            fork_state = ckpt_mgr.load_from_run(
                source_run_id=args.fork_from,
                filename=args.fork_checkpoint,
                map_location=device,
            )

        if 'optimizer_state_dict' not in fork_state:
            raise RuntimeError(
                f"--fork_from: checkpoint '{args.fork_checkpoint}' from run "
                f"{args.fork_from} has no optimizer_state_dict. Pick a "
                f"checkpoint that carries full state (e.g. latest.pt or a "
                f"periodic checkpoint_epoch*.pt). If you only want model "
                f"weights, use --warmstart_from instead."
            )

        target = accelerator.unwrap_model(model)
        # Drop any tensors whose shape changed between the source run
        # and the current architecture (e.g. an old Stage 1 checkpoint
        # carries a 2-input FusionBlock — the new 3-input GazeFusionBlock
        # has a larger first linear layer). strict=False alone doesn't
        # handle shape mismatches; we filter by shape here so the fork
        # survives architecture migrations.
        filtered_sd, shape_dropped = _filter_compatible_state(
            fork_state['model_state_dict'], target.state_dict())
        missing, unexpected = target.load_state_dict(
            filtered_sd, strict=False)
        if is_main:
            if missing:
                print(f"  [fork] missing keys: {len(missing)} "
                      f"(first: {missing[:3]})")
            if unexpected:
                print(f"  [fork] unexpected keys: {len(unexpected)} "
                      f"(first: {unexpected[:3]})")
            if shape_dropped:
                print(f"  [fork] shape-mismatch keys dropped: "
                      f"{len(shape_dropped)} (first: {shape_dropped[:3]})")

        src_epoch = fork_state['epoch']

        if args.fork_reset_epoch:
            # Explicit reset: restart from epoch 1 under the current
            # phase schedule. Optimizer m/v state is preserved (the
            # whole point of fork vs warmstart); scheduler is rebuilt
            # fresh by the phase-transition block on the first loop
            # iteration (current_phase=0 forces that transition to fire).
            start_epoch = 1
            current_phase = 0
            phase1_cfg = get_phase_config(1)
            optimizer = optim.AdamW(model.parameters(),
                                    lr=phase1_cfg['lr'],
                                    betas=(0.5, 0.95), weight_decay=1e-4)
            # The param count may not match after an architecture change.
            # PyTorch's optimizer.load_state_dict asserts per-group param
            # count equality, so we check first and fall back to a fresh
            # optimizer when the shapes diverge.
            optimizer_preserved = _optimizer_state_compatible(
                fork_state['optimizer_state_dict'], optimizer)
            if optimizer_preserved:
                optimizer.load_state_dict(
                    fork_state['optimizer_state_dict'])
            # Scheduler intentionally left None; rebuilt in-loop.
            scheduler = None
            if scaler is not None and 'scaler_state_dict' in fork_state:
                scaler.load_state_dict(fork_state['scaler_state_dict'])
            # Don't inherit src best_val_loss — it was measured under a
            # different loss composition and isn't comparable.
            if is_main:
                opt_status = ("preserved"
                              if optimizer_preserved
                              else "RESET (param count mismatch — "
                                   "architecture changed)")
                print(f"  Epoch counter reset to 1 (--fork_reset_epoch). "
                      f"Optimizer momentum: {opt_status}; scheduler "
                      f"and best_val_loss rebuilt for phase 1.")
        else:
            # Continuation fork: resume from src_epoch under the same
            # phase map, restoring full state.
            start_epoch = src_epoch + 1
            current_phase = get_phase(src_epoch)
            fork_phase_cfg = get_phase_config(src_epoch)
            phase_start, phase_end = fork_phase_cfg['epochs']

            optimizer = optim.AdamW(model.parameters(),
                                    lr=fork_phase_cfg['lr'],
                                    betas=(0.5, 0.95), weight_decay=1e-4)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=phase_end - phase_start + 1)

            # Same guard as the cross-stage path — if the model changed
            # shape the saved optimizer state can't load.
            if _optimizer_state_compatible(
                    fork_state['optimizer_state_dict'], optimizer):
                optimizer.load_state_dict(
                    fork_state['optimizer_state_dict'])
            elif is_main:
                print("  [fork] optimizer param count mismatch — "
                      "starting with fresh AdamW state.")
            if 'scheduler_state_dict' in fork_state:
                scheduler.load_state_dict(fork_state['scheduler_state_dict'])
            if scaler is not None and 'scaler_state_dict' in fork_state:
                scaler.load_state_dict(fork_state['scaler_state_dict'])

            best_val_loss = fork_state.get(
                'val_metrics', {}).get('total', best_val_loss)
            # Same-stage fork: architecture is unchanged so the EMA state
            # is meaningful — load it. Cross-stage fork (above) leaves
            # EMA fresh because the architecture may have changed and
            # any saved EMA tensors that survive the shape filter would
            # be of a different epoch's task distribution.
            if ema_model is not None and 'ema_state_dict' in fork_state:
                ema_sd = fork_state['ema_state_dict']
                ema_target_sd = ema_model.state_dict()
                ema_filtered, ema_dropped = _filter_compatible_state(
                    ema_sd, ema_target_sd)
                ema_model.load_state_dict(ema_filtered, strict=False)
                ema_restored_from_ckpt = True
                if is_main:
                    print(f"  [fork] EMA state loaded "
                          f"({len(ema_filtered)} kept, "
                          f"{len(ema_dropped)} dropped)")
            if is_main:
                print(f"  Fork: loaded full state from epoch "
                      f"{src_epoch} (phase {current_phase}). New run "
                      f"continues at epoch {start_epoch}, "
                      f"best_val_loss={best_val_loss:.4f}")

            if start_epoch > args.epochs:
                raise RuntimeError(
                    f"Cannot fork: source checkpoint is at epoch "
                    f"{src_epoch}, so training would start at epoch "
                    f"{start_epoch}, but --epochs={args.epochs}. Increase "
                    f"--epochs to extend training past the fork point."
                )

    # --- Warmstart from a different run (e.g. Stage 1 -> Stage 2) ---
    # Loads ONLY model weights. Optimizer, scheduler, epoch counter, and
    # scaler stay fresh. A new run_id has already been generated by the
    # checkpoint manager (since --run_id is forbidden with --warmstart_from).
    if args.warmstart_from and ckpt_mgr is not None:
        if is_main:
            print(f"Warmstarting from run {args.warmstart_from} "
                  f"({args.warmstart_checkpoint}) into new run {ckpt_mgr.run_id} ...")
        # Serialize MinIO download across ranks — see note at the resume
        # site above for the race condition this avoids.
        with accelerator.main_process_first():
            ws_state = ckpt_mgr.load_from_run(
                source_run_id=args.warmstart_from,
                filename=args.warmstart_checkpoint,
                map_location=device,
            )
        target = accelerator.unwrap_model(model)
        # Filter by shape to survive architecture migrations (see fork block).
        filtered_sd, shape_dropped = _filter_compatible_state(
            ws_state['model_state_dict'], target.state_dict())
        missing, unexpected = target.load_state_dict(
            filtered_sd, strict=False)
        if is_main:
            if missing:
                print(f"  [warmstart] missing keys: {len(missing)} "
                      f"(first: {missing[:3]})")
            if unexpected:
                print(f"  [warmstart] unexpected keys: {len(unexpected)} "
                      f"(first: {unexpected[:3]})")
            if shape_dropped:
                print(f"  [warmstart] shape-mismatch keys dropped: "
                      f"{len(shape_dropped)} (first: {shape_dropped[:3]})")
            src_epoch = ws_state.get('epoch', '?')
            print(f"  Loaded weights from epoch {src_epoch}. "
                  f"Starting fresh optimizer at epoch 1.")
        # start_epoch stays at 1, optimizer/scheduler stay None — they will
        # be created in the phase-transition block below exactly like a
        # from-scratch run.

    # Seed the EMA shadow from the live model whenever no EMA was
    # restored from a checkpoint. EMA was created right after
    # accelerator.prepare, BEFORE warmstart/fork loaded the actual
    # starting weights — leaving it at random init would mean the
    # first ~1/(1-decay) ≈ 1000 updates produce a validation model
    # that's a mix of random init and trained weights. Copying the
    # current live state into the EMA shadow makes it identical to
    # the live model at step 0 and decays correctly from there.
    if ema_model is not None and not ema_restored_from_ckpt:
        live_sd = _unwrap_raynet(model).state_dict()
        ema_inner_sd = ema_model.module.state_dict()
        for k in ema_inner_sd:
            if k in live_sd and ema_inner_sd[k].shape == live_sd[k].shape:
                ema_inner_sd[k].data.copy_(live_sd[k].data)
        if is_main:
            print("  EMA shadow seeded from current live weights")

    for epoch in range(start_epoch, args.epochs + 1):
        phase = get_phase(epoch)
        cfg = get_phase_config(epoch)

        # Ablation overrides
        if args.no_multiview:
            cfg['multiview'] = False
            cfg['lam_reproj'] = 0.0
            cfg['lam_mask'] = 0.0
        if args.gaze_only:
            cfg['lam_lm'] = 0.0

        # Phase transition: update LR schedule, keep optimizer state
        if phase != current_phase:
            current_phase = phase
            if is_main:
                print(f"\n{'='*60}")
                print(f"Phase {phase}: {cfg['description']}")
                print(f"  lr={cfg['lr']}, lam_lm={cfg['lam_lm']}, "
                      f"lam_gaze={cfg['lam_gaze']}, lam_ray={cfg.get('lam_ray', 0)}, "
                      f"lam_pose={cfg.get('lam_pose', 0)}, lam_trans={cfg.get('lam_trans', 0)}, "
                      f"sigma={cfg['sigma']}")
                if cfg.get('multiview'):
                    print(f"  Multi-view: lam_gaze_consist={cfg['lam_reproj']}, "
                          f"lam_shape={cfg['lam_mask']}")
                print(f"{'='*60}")

            phase_start, phase_end = cfg['epochs']

            if optimizer is None:
                # First phase of the stage: create fresh optimizer
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=cfg['lr'],
                    betas=(0.5, 0.95),
                    weight_decay=1e-4,
                )
                if is_main:
                    print(f"  Created new AdamW optimizer (lr={cfg['lr']})")
            else:
                # Subsequent phases: carry over optimizer state (momentum +
                # adaptive second-moment estimates), only update the LR.
                # Recreating AdamW here would zero out m and v, causing a
                # destructive transient: raw-gradient steps at peak LR with
                # no per-parameter scaling — the "phase shock" that wastes
                # ~4 epochs recovering each time.
                for pg in optimizer.param_groups:
                    pg['lr'] = cfg['lr']
                    pg['initial_lr'] = cfg['lr']  # CosineAnnealingLR reads this
                if is_main:
                    print(f"  Carried over optimizer state, updated LR → {cfg['lr']}")

            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=phase_end - phase_start + 1,
            )

        active_train_loader = train_loader_mv
        active_val_loader = val_loader

        # n_views for CrossViewAttention during TRAINING.
        # NOTE: this is independent of cfg['multiview']. CrossViewAttention
        # runs whenever mv_groups > 1 (which produces 9-grouped batches).
        # cfg['multiview'] only gates the auxiliary multiview_consistency_loss
        # inside train_one_epoch — see header comment on PHASE_CONFIG above.
        # Pass --no_multiview on the CLI to truly ablate cross-view fusion.
        active_n_views = 1 if args.no_multiview else 9

        # Train
        train_losses = train_one_epoch(
            model, active_train_loader, optimizer, device, epoch, cfg,
            scaler=scaler,
            grad_accum_steps=hw['grad_accum_steps'],
            amp_enabled=hw['amp'],
            amp_dtype=amp_dtype,
            batch_csv_writer=batch_csv_writer,
            n_views=active_n_views,
            ema_model=ema_model,
        )
        if batch_csv_file is not None:
            batch_csv_file.flush()

        # Validate with the SAME multi-view topology used in training.
        # Previously hardcoded to n_views=1, which bypassed CrossViewAttention,
        # camera embeddings, and the PoseEncoder fusion path — producing
        # uninformative val gaze metrics. The val loader now delivers
        # 9-grouped batches (see _create_mds_mv_loader), so the reshape
        # inside CrossViewAttention is well-defined.
        # Validate through the EMA shadow when enabled. EMA tracks the
        # smoothed weight trajectory so val metrics are not contaminated
        # by single-batch weight oscillation. Falls back to the live
        # model when --ema_decay 0 was passed.
        val_model = ema_model if ema_model is not None else model
        val_losses = validate(
            val_model, active_val_loader, device, epoch, cfg,
            amp_enabled=hw['amp'],
            amp_dtype=amp_dtype,
            n_views=active_n_views,
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # All ranks finish the epoch before the main process saves / uploads.
        accelerator.wait_for_everyone()

        if is_main:
            mv_str = ""
            if cfg.get('multiview'):
                mv_str = (f" gaze_mv={train_losses['reproj']:.4f}"
                          f" shape={train_losses['mask']:.4f}")
            if train_losses.get('ray_target', 0) > 0:
                mv_str += f" ray={train_losses['ray_target']:.4f}"
            if train_losses.get('pose', 0) > 0:
                mv_str += f" pose={train_losses['pose']:.4f}"
            if train_losses.get('translation', 0) > 0:
                mv_str += f" trans={train_losses['translation']:.4f}"
            print(f"Epoch {epoch:3d} | Phase {phase} | lr {current_lr:.2e} | "
                  f"Train: loss={train_losses['total']:.4f} lm={train_losses['landmark']:.4f} "
                  f"ang={train_losses['angular_deg']:.2f}deg{mv_str} | "
                  f"Val: loss={val_losses['total']:.4f} lm_px={val_losses['landmark_px']:.2f}px "
                  f"ang={val_losses['angular_deg']:.2f}deg")

            if hw['amp'] and device.type == 'cuda':
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f"  GPU memory peak (rank 0): {mem_gb:.1f} GB")

            csv_writer.writerow([
                epoch, phase, f"{current_lr:.2e}",
                f"{train_losses['total']:.6f}",
                f"{train_losses['landmark']:.6f}",
                f"{train_losses['angular_deg']:.4f}",
                f"{train_losses.get('reproj', 0):.6f}",
                f"{train_losses.get('mask', 0):.6f}",
                f"{train_losses.get('ray_target', 0):.6f}",
                f"{train_losses.get('pose', 0):.6f}",
                f"{train_losses.get('translation', 0):.6f}",
                f"{train_losses.get('iris_seg', 0):.6f}",
                f"{train_losses.get('eyeball_seg', 0):.6f}",
                f"{val_losses['total']:.6f}",
                f"{val_losses['landmark']:.6f}",
                f"{val_losses['angular_deg']:.4f}",
                f"{val_losses['landmark_px']:.4f}",
                f"{val_losses.get('iris_seg', 0):.6f}",
                f"{val_losses.get('eyeball_seg', 0):.6f}",
            ])
            csv_file.flush()

            # Unwrap DDP (and torch.compile) before handing the raw module to
            # the checkpoint code; its state_dict is what's serialised.
            save_model = accelerator.unwrap_model(model)

            # Build the extras blob once per epoch — picked up by both
            # save_best and the periodic save below. EMA state ships in
            # `ema_state_dict` so resume can restore the smoothed weights;
            # without this, resume would EMA-from-init and lose the
            # smoothing benefit for the rest of training.
            ckpt_extras = {'profile': args.profile}
            if ema_model is not None:
                ckpt_extras['ema_state_dict'] = ema_model.state_dict()

            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']

                if ckpt_mgr is not None:
                    ckpt_mgr.save_best(
                        epoch=epoch, model=save_model, val_loss=best_val_loss,
                        val_metrics=val_losses, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler,
                        extra=ckpt_extras,
                    )
                else:
                    save_dict = {
                        'epoch': epoch,
                        'phase': phase,
                        'model_state_dict': save_model.state_dict(),
                        'val_loss': best_val_loss,
                        'val_landmark_px': val_losses['landmark_px'],
                        'val_angular_deg': val_losses['angular_deg'],
                        'profile': args.profile,
                    }
                    if ema_model is not None:
                        save_dict['ema_state_dict'] = ema_model.state_dict()
                    torch.save(save_dict, os.path.join(output_dir, 'best_model.pt'))
                print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

            # Periodic + latest checkpoint
            if ckpt_mgr is not None:
                # Always save latest (for resume)
                ckpt_mgr.save(
                    epoch=epoch, model=save_model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, phase=phase,
                    train_metrics=train_losses, val_metrics=val_losses,
                    tag='latest', extra=ckpt_extras,
                )
                # Periodic named checkpoint
                if epoch % args.ckpt_every == 0:
                    ckpt_mgr.save(
                        epoch=epoch, model=save_model, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, phase=phase,
                        train_metrics=train_losses, val_metrics=val_losses,
                        extra=ckpt_extras,
                    )
                # Upload logs after each epoch (survives interruption/crash)
                try:
                    batch_csv_file.flush()
                    csv_file.flush()
                    for local_path in (batch_csv_path, csv_path):
                        log_key = f"{args.ckpt_prefix}/{ckpt_mgr.run_id}/{os.path.basename(local_path)}"
                        ckpt_mgr._client.fput_object(args.ckpt_bucket, log_key, local_path)
                except Exception as e:
                    log.warning("MinIO log upload failed: %s", e)
            else:
                if epoch % args.ckpt_every == 0:
                    save_dict = {
                        'epoch': epoch,
                        'phase': phase,
                        'model_state_dict': save_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    if scaler is not None:
                        save_dict['scaler_state_dict'] = scaler.state_dict()
                    torch.save(save_dict, os.path.join(output_dir, f'checkpoint_epoch{epoch}.pt'))

        # Broadcast best_val_loss from main so every rank stays in sync on
        # the "was this a new best?" question. For DDP this doesn't affect
        # training state (only main saves), but keeping the value consistent
        # makes the resume path simpler.
        accelerator.wait_for_everyone()

    if is_main:
        csv_file.close()
        batch_csv_file.close()

        # Upload training logs to MinIO
        if ckpt_mgr is not None:
            for log_file in [csv_path, batch_csv_path]:
                if os.path.exists(log_file):
                    log_key = f"{args.ckpt_prefix}/{ckpt_mgr.run_id}/{os.path.basename(log_file)}"
                    try:
                        ckpt_mgr._client.fput_object(args.ckpt_bucket, log_key, log_file)
                        print(f"  Uploaded {os.path.basename(log_file)} to s3://{args.ckpt_bucket}/{log_key}")
                    except Exception as e:
                        print(f"  Warning: failed to upload {os.path.basename(log_file)}: {e}")

        print(f"\nTraining complete. Results saved to {output_dir}")
        print(f"Best val loss: {best_val_loss:.4f}")
        if ckpt_mgr is not None:
            print(f"Checkpoints: s3://{args.ckpt_bucket}/{args.ckpt_prefix}/{ckpt_mgr.run_id}/")


# ============== Data Loader Helpers ==============

def _create_local_loaders(args, hw):
    """Create local disk-based dataloaders."""
    train_subjects = list(range(1, 47))
    val_subjects = list(range(47, 57))

    common_kwargs = dict(
        num_workers=hw['num_workers'],
        samples_per_subject=args.samples_per_subject,
        eye=args.eye,
    )

    train_loader_standard, val_loader = create_dataloaders(
        base_dir=args.data_dir,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        batch_size=hw['batch_size'],
        ensure_multiview=False,
        **common_kwargs,
    )

    train_loader_mv, _ = create_dataloaders(
        base_dir=args.data_dir,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        batch_size=hw['mv_groups'],
        ensure_multiview=True,
        **common_kwargs,
    )

    return train_loader_standard, train_loader_mv, val_loader


def _create_mds_mv_loader(args, hw):
    """Create multi-view MDS loaders (both train and val).

    Returns (train_loader_mv, val_loader_mv). The MV val loader delivers
    batches of mv_groups * 9 samples in (subject, frame, cam) order so
    that CrossViewAttention, cam_embed, and PoseEncoder features are
    exercised during validation — matching the training forward pass.
    Without this, validation bypasses all multi-view fusion (n_views=1)
    and val gaze metrics are a pessimistic lower bound that never moves.
    """
    from RayNet.streaming.dataset import create_multiview_streaming_dataloaders

    from torchvision import transforms as T

    # Normalize to ImageNet stats (applied to BOTH train and val so that
    # the shared stem's BN running_mean/var — calibrated on ColorJitter'd
    # training images — match the activation scale seen at val time.
    # Without this, the jitter ±40% brightness/contrast widens the training
    # pixel variance relative to clean val images; the stem's BN running_var
    # grows to absorb that extra variance, then attenuates clean val
    # activations by the corresponding factor — degrading both landmark and
    # gaze metrics progressively with each epoch.
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    train_transform = T.Compose([
        T.ColorJitter(brightness=0.4, contrast=0.4,
                      saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        normalize,
    ])
    val_transform = normalize

    print("Creating multi-view MDS streaming loaders (train + val)...")
    train_loader_mv, val_loader_mv = create_multiview_streaming_dataloaders(
        remote_train=args.mds_train,
        remote_val=args.mds_val,
        local_cache=os.path.join(args.output_dir, 'mds_cache_mv'),
        mv_groups=hw['mv_groups'],
        num_workers=hw['num_workers'],
        transform=train_transform,
        val_transform=val_transform,
        pin_memory=hw['pin_memory'],
        prefetch_factor=hw['prefetch_factor'],
        persistent_workers=hw['persistent_workers'],
        samples_per_subject=args.samples_per_subject,
    )
    return train_loader_mv, val_loader_mv


# ============== CLI ==============

def parse_args():
    parser = argparse.ArgumentParser(description='RayNet v5 Training')

    # Data
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to GazeGene dataset (local mode)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--samples_per_subject', type=int, default=200)
    parser.add_argument('--eye', type=str, default='L', choices=['L', 'R'])

    # MDS streaming (MosaicML + MinIO)
    parser.add_argument('--mds_streaming', action='store_true',
                        help='Stream training data from MDS shards on MinIO/S3')
    parser.add_argument('--mds_train', type=str, default=None,
                        help='MDS shard URL for training (e.g. s3://gazegene/train)')
    parser.add_argument('--mds_val', type=str, default=None,
                        help='MDS shard URL for validation (e.g. s3://gazegene/val)')

    # Model — v5 uses RepNeXt-M1 for all branches (shared stem + 3 branches)
    parser.add_argument('--core_backbone_weight_path', type=str, default=None,
                        help='Path to pretrained RepNeXt-M1 weights '
                             '(loaded into all 4 M1 instances: shared + landmark + gaze + pose)')

    # Hardware profile
    parser.add_argument('--profile', type=str, default='default',
                        choices=list(HARDWARE_PROFILES.keys()),
                        help=f'Hardware profile ({", ".join(HARDWARE_PROFILES.keys())})')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile even if profile enables it')
    parser.add_argument('--no_multiview', action='store_true',
                        help='Disable multi-view losses and CrossViewAttention '
                             '(n_views=1 throughout, for ablation)')
    parser.add_argument('--gaze_only', action='store_true',
                        help='Disable landmark loss (lam_lm=0) for gaze-only '
                             'training, matching GazeGene paper baseline')

    # Overrides (None = use profile default)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--mv_groups', type=int, default=None,
                        help='Multi-view batch groups (actual batch = mv_groups * 9)')
    parser.add_argument('--grad_accum_steps', type=int, default=None,
                        help='Gradient accumulation steps')

    # MinIO checkpoint storage
    parser.add_argument('--ckpt_bucket', type=str, default=None,
                        help='MinIO bucket for checkpoints (enables MinIO storage)')
    parser.add_argument('--ckpt_prefix', type=str, default='checkpoints',
                        help='Key prefix under the bucket (default: checkpoints)')
    parser.add_argument('--minio_endpoint', type=str, default=None,
                        help='MinIO endpoint URL (default: S3_ENDPOINT_URL env var)')
    parser.add_argument('--ckpt_every', type=int, default=5,
                        help='Save a named checkpoint every N epochs (default: 5)')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Run ID for checkpoint grouping (auto-generated if omitted)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint of --run_id '
                             '(same run, continues epoch counter)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from a specific checkpoint file '
                             '(e.g. checkpoint_epoch5.pt, best_model.pt)')
    parser.add_argument('--warmstart_from', type=str, default=None,
                        help='Warmstart from a DIFFERENT run. Loads ONLY model '
                             'weights — optimizer/scheduler/epoch are reset. '
                             'A new run_id is auto-generated.')
    parser.add_argument('--warmstart_checkpoint', type=str, default='best_model.pt',
                        help='Checkpoint file to pull from --warmstart_from run '
                             '(default: best_model.pt)')
    parser.add_argument('--fork_from', type=str, default=None,
                        help='Fork from a previous run: load FULL training '
                             'state (model + optimizer + scheduler + scaler '
                             '+ epoch + best_val_loss) from the source run, '
                             'but write new checkpoints under a NEW run_id '
                             'so the source run is not overwritten. Use to '
                             'extend training past the original epoch budget '
                             'or branch off for hyperparameter variations '
                             'without corrupting the baseline. A new run_id '
                             'is auto-generated unless --run_id is passed.')
    parser.add_argument('--fork_checkpoint', type=str, default='latest.pt',
                        help='Checkpoint file to pull from --fork_from run '
                             '(default: latest.pt — has full optimizer/'
                             'scheduler state). best_model.pt also works if '
                             'it was saved with optimizer state.')
    parser.add_argument('--fork_reset_epoch', action='store_true',
                        help='Force fork to reset epoch counter to 1 under '
                             'the current phase schedule. Optimizer momentum '
                             'and scaler state are still preserved.')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay for validation weights (default 0.999). '
                             'Set 0 to disable EMA. The EMA model tracks a '
                             'moving average of the trainable parameters and '
                             'is what validation runs through; the live model '
                             'is what gets gradient updates.')
    args = parser.parse_args()

    if not args.mds_streaming and args.data_dir is None:
        parser.error("--data_dir is required when not using --mds_streaming")

    if args.mds_streaming and (not args.mds_train or not args.mds_val):
        parser.error("--mds_streaming requires both --mds_train and --mds_val")

    # --resume_from implies --resume
    if args.resume_from:
        args.resume = True

    if args.resume and not args.run_id:
        parser.error("--resume requires --run_id to identify which run to resume")
    if args.resume and not args.ckpt_bucket:
        parser.error("--resume requires --ckpt_bucket")
    if args.warmstart_from and args.resume:
        parser.error("--warmstart_from and --resume are mutually exclusive. "
                     "Use --resume to continue an interrupted run; use "
                     "--warmstart_from to start a new run with weights "
                     "from another run.")
    if args.warmstart_from and not args.ckpt_bucket:
        parser.error("--warmstart_from requires --ckpt_bucket")
    if args.warmstart_from and args.run_id:
        parser.error("--warmstart_from creates a new run — do not pass --run_id "
                     "(a fresh timestamped run_id will be generated).")
    if args.fork_from:
        if args.resume or args.warmstart_from:
            parser.error("--fork_from is mutually exclusive with --resume "
                         "and --warmstart_from. Use --resume to continue the "
                         "same run in place; --warmstart_from to start fresh "
                         "with weights only; --fork_from to branch the "
                         "full training state into a new run_id.")
        if not args.ckpt_bucket:
            parser.error("--fork_from requires --ckpt_bucket")
        if args.run_id and args.run_id == args.fork_from:
            parser.error("--run_id cannot equal --fork_from — that would "
                         "overwrite the source run. Pick a different "
                         "--run_id or omit it to auto-generate.")

    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)
