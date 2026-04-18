"""
RayNet v5 Training Script (Triple-M1 architecture).

Staged training strategy:

  Stage 1 — Landmark + Pose baseline (no gaze, no bridges):
    Phase 1 (epochs 1-8):   λ_lm=1.0, λ_pose=0.5, λ_trans=0.5
    Phase 2 (epochs 9-15):  λ_lm=1.0, λ_pose=1.0, λ_trans=1.0

  Stage 2 — Add gaze with GazeGene 3D eyeball losses (bridges zero-init):
    Phase 1 (epochs 1-5):   λ_gaze=0.1, λ_eyeball=0.1, λ_pupil=0.1
    Phase 2 (epochs 6-15):  λ_gaze=0.5, λ_eyeball=0.3, λ_pupil=0.3, λ_ray=0.1
    Phase 3 (epochs 16-25): λ_gaze=1.0, λ_eyeball=0.5, λ_pupil=0.5, λ_geom=0.2, λ_ray=0.3

  Stage 3 — Full pipeline with bridges + MAGE box encoder:
    Phase 1 (epochs 1-5):   Bridge warmup, geometric angular active
    Phase 2 (epochs 6-15):  Full losses + multi-view consistency
    Phase 3 (epochs 16-25): Gaze-focused fine-tuning

  Gradient clipping (max_norm) varies by phase:
    Phase 1: max_norm=5.0 (aggressive, allows large multi-task gradients)
    Phase 2+: max_norm=2.0 (conservative, prevents gaze/pose interference)

Usage:
    # Single GPU
    python -m RayNet.train --stage 1 --profile t4 --mds_streaming \
        --mds_train s3://gazegene/train --mds_val s3://gazegene/val ...

    # Kaggle 2× Tesla T4 (single node, multi-GPU)
    accelerate launch --multi_gpu --num_processes 2 \
        -m RayNet.train --stage 1 --profile kaggle_t4x2 --mds_streaming ...

    # Two machines on the same network (see RayNet/hardware_profiles.py)
    accelerate launch --multi_gpu --num_machines 2 --num_processes 2 \
        --machine_rank <0|1> --main_process_ip $MAIN_IP --main_process_port 29500 \
        -m RayNet.train --stage 1 --profile multi_node_t4 --mds_streaming ...
"""

import argparse
import os
import csv
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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

STAGE_CONFIGS = {
    # ---- Stage 1: Landmark + Pose baseline (no gaze) ----
    # Purpose: Validate shared stem + branch encoders learn useful features.
    # Expect: Landmark px error < 4px by epoch 8, pose geodesic < 10° by epoch 12.
    1: {
        1: {
            'epochs': (1, 8),
            'lam_lm': 1.0,
            'lam_gaze': 0.0,
            'lam_eyeball': 0.0,
            'lam_pupil': 0.0,
            'lam_geom_angular': 0.0,
            'lam_ray': 0.0,
            'lam_reproj': 0.0,
            'lam_mask': 0.0,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 1e-3,
            'sigma': 2.0,
            'multiview': False,
            'use_landmark_bridge': False,
            'use_pose_bridge': False,
            'description': 'V5-S1P1: Landmark + pose warmup (shared stem + branch encoders)',
        },
        2: {
            'epochs': (9, 15),
            'lam_lm': 1.0,
            'lam_gaze': 0.0,
            'lam_eyeball': 0.0,
            'lam_pupil': 0.0,
            'lam_geom_angular': 0.0,
            'lam_ray': 0.0,
            'lam_reproj': 0.0,
            'lam_mask': 0.0,
            'lam_pose': 1.0,
            'lam_trans': 1.0,
            'lr': 3e-4,
            'sigma': 1.5,
            'multiview': False,
            'use_landmark_bridge': False,
            'use_pose_bridge': False,
            'description': 'V5-S1P2: Landmark refinement + pose emphasis',
        },
    },

    # ---- Stage 2: Add gaze + GazeGene 3D eyeball losses (bridges zero-init) ----
    # Purpose: Test gaze learning with explicit 3D eyeball structure.
    # Bridges are present but zero-init — they start as identity (skip).
    # Expect: Angular error improving steadily; eyeball/pupil L1 converging.
    2: {
        1: {
            'epochs': (1, 5),
            'lam_lm': 1.0,
            'lam_gaze': 0.1,
            'lam_eyeball': 0.1,
            'lam_pupil': 0.1,
            'lam_geom_angular': 0.0,  # too early — geometry not converged
            'lam_ray': 0.0,
            'lam_reproj': 0.0,
            'lam_mask': 0.0,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 3e-4,
            'sigma': 1.5,
            'multiview': False,
            'use_landmark_bridge': False,
            'use_pose_bridge': False,
            'description': 'V5-S2P1: Gaze warmup + 3D eyeball structure learning',
        },
        2: {
            'epochs': (6, 15),
            'lam_lm': 1.0,
            'lam_gaze': 0.5,
            'lam_eyeball': 0.3,
            'lam_pupil': 0.3,
            'lam_geom_angular': 0.0,  # still off — let eyeball/pupil stabilize
            'lam_ray': 0.1,
            'lam_reproj': 0.05,
            'lam_mask': 0.02,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 3e-4,
            'sigma': 1.3,
            'multiview': True,
            'use_landmark_bridge': False,
            'use_pose_bridge': False,
            'description': 'V5-S2P2: Balanced gaze + eyeball/pupil + ray + MV',
        },
        3: {
            'epochs': (16, 25),
            'lam_lm': 1.0,
            'lam_gaze': 1.0,
            'lam_eyeball': 0.5,
            'lam_pupil': 0.5,
            'lam_geom_angular': 0.2,  # NOW turn on — geometry should be close
            'lam_ray': 0.3,
            'lam_reproj': 0.1,
            'lam_mask': 0.05,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 1e-4,
            'sigma': 1.0,
            'multiview': True,
            'use_landmark_bridge': False,
            'use_pose_bridge': False,
            'description': 'V5-S2P3: Gaze fine-tuning + geometric angular loss',
        },
    },

    # ---- Stage 3: Full pipeline WITH bridges + MAGE box encoder ----
    # Purpose: Activate inter-branch bridges and BoxEncoder fusion.
    # Bridges are zero-init from checkpoint (never trained in S1/S2) so they
    # start as identity and learn gradually.
    3: {
        1: {
            'epochs': (1, 5),
            'lam_lm': 1.0,
            'lam_gaze': 0.3,
            'lam_eyeball': 0.3,
            'lam_pupil': 0.3,
            'lam_geom_angular': 0.1,
            'lam_ray': 0.0,
            'lam_reproj': 0.0,
            'lam_mask': 0.0,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 3e-4,
            'sigma': 2.0,
            'multiview': False,
            'use_landmark_bridge': True,
            'use_pose_bridge': True,
            'description': 'V5-S3P1: Bridge warmup (zero-init, gentle LR)',
        },
        2: {
            'epochs': (6, 15),
            'lam_lm': 1.0,
            'lam_gaze': 0.5,
            'lam_eyeball': 0.5,
            'lam_pupil': 0.5,
            'lam_geom_angular': 0.2,
            'lam_ray': 0.1,
            'lam_reproj': 0.05,
            'lam_mask': 0.02,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 3e-4,
            'sigma': 1.5,
            'multiview': True,
            'use_landmark_bridge': True,
            'use_pose_bridge': True,
            'description': 'V5-S3P2: Full losses + bridges + multi-view',
        },
        3: {
            'epochs': (16, 25),
            'lam_lm': 0.5,
            'lam_gaze': 1.0,
            'lam_eyeball': 0.5,
            'lam_pupil': 0.5,
            'lam_geom_angular': 0.3,
            'lam_ray': 0.3,
            'lam_reproj': 0.1,
            'lam_mask': 0.05,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 1e-4,
            'sigma': 1.0,
            'multiview': True,
            'use_landmark_bridge': True,
            'use_pose_bridge': True,
            'description': 'V5-S3P3: Gaze-focused fine-tuning (full bridges)',
        },
    },
}


# Default: Stage 3 (full pipeline) for backward compatibility
PHASE_CONFIG = STAGE_CONFIGS[3]


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


# ============== Training Loop ==============

def train_one_epoch(model, train_loader, optimizer, device, epoch, cfg,
                    scaler=None, grad_accum_steps=1, amp_enabled=False,
                    amp_dtype=torch.float16, batch_csv_writer=None,
                    n_views=1):
    """Run one training epoch with AMP and gradient accumulation support."""
    model.train()
    use_multiview = cfg.get('multiview', False)
    use_landmark_bridge = cfg.get('use_landmark_bridge', True)
    use_pose_bridge = cfg.get('use_pose_bridge', True)
    if not amp_enabled:
        amp_dtype = torch.float32

    running_losses = {
        'total': 0.0, 'landmark': 0.0, 'angular': 0.0, 'angular_deg': 0.0,
        'reproj': 0.0, 'mask': 0.0, 'ray_target': 0.0, 'pose': 0.0,
        'translation': 0.0,
        # V5-specific
        'eyeball_center': 0.0, 'pupil_center': 0.0, 'geom_angular': 0.0,
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

        mv_components = None
        # Forward pass with AMP autocast
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            predictions = model(images, n_views=n_views,
                                R_cam=R_cam, T_cam=T_cam,
                                face_bbox=face_bbox_gt,
                                use_landmark_bridge=use_landmark_bridge,
                                use_pose_bridge=use_pose_bridge)

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
            )

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
                if scaler is not None:
                    # FP16 path: scaler handles inf/nan detection internally
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # FP32 / BF16 path: manually check grad norm and skip step
                    # if non-finite (BF16 has no scaler to catch overflow/nan)
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=max_norm)
                    if torch.isfinite(total_norm):
                        optimizer.step()
                    else:
                        log.warning("Epoch %d batch %d: non-finite grad norm "
                                    "(%s), skipping optimizer step",
                                    epoch, step + 1, total_norm.item())
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
        n_batches += 1

        # Per-batch CSV logging (high granularity)
        if batch_csv_writer is not None:
            ray_val = components.get('ray_target_loss',
                                     torch.tensor(0.0)).item()
            pose_val = components.get('pose_loss',
                                      torch.tensor(0.0)).item()
            trans_val = components.get('translation_loss',
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
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_norm)
                if torch.isfinite(total_norm):
                    optimizer.step()
            optimizer.zero_grad()

    for k in running_losses:
        running_losses[k] /= max(n_batches, 1)

    return running_losses


@torch.no_grad()
def validate(model, val_loader, device, epoch, cfg, amp_enabled=False,
             amp_dtype=torch.float16, n_views=1):
    """Run validation."""
    model.eval()
    use_landmark_bridge = cfg.get('use_landmark_bridge', True)
    use_pose_bridge = cfg.get('use_pose_bridge', True)
    if not amp_enabled:
        amp_dtype = torch.float32

    running_losses = {
        'total': 0.0, 'landmark': 0.0, 'angular': 0.0, 'angular_deg': 0.0,
        'landmark_px': 0.0, 'pose': 0.0, 'ray': 0.0, 'translation': 0.0,
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

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            predictions = model(images, n_views=n_views,
                                R_cam=R_cam, T_cam=T_cam,
                                face_bbox=face_bbox_gt,
                                use_landmark_bridge=use_landmark_bridge,
                                use_pose_bridge=use_pose_bridge)

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
        n_batches += 1

    for k in running_losses:
        running_losses[k] /= max(n_batches, 1)

    return running_losses


# ============== Main Training ==============

def _build_run_config(args, hw):
    """Collect all training configuration into a single dict for metadata."""
    return {
        'stage': args.stage,
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
    # Select stage config — updates module-level PHASE_CONFIG for get_phase/get_phase_config
    global PHASE_CONFIG
    PHASE_CONFIG = STAGE_CONFIGS[args.stage]

    # Auto-cap --epochs to the last epoch defined in this stage's phase config
    stage_max_epoch = max(cfg['epochs'][1] for cfg in PHASE_CONFIG.values())
    if args.epochs > stage_max_epoch:
        print(f"[auto-cap] --epochs={args.epochs} exceeds stage {args.stage} "
              f"max epoch ({stage_max_epoch}). Capping to {stage_max_epoch}.")
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
        print(f"Training stage: {args.stage}")
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
            'epoch', 'stage', 'phase', 'lr',
            'train_total', 'train_landmark', 'train_angular_deg',
            'train_reproj', 'train_mask', 'train_ray_target', 'train_pose',
            'train_translation',
            'val_total', 'val_landmark', 'val_angular_deg', 'val_landmark_px',
        ])

        batch_csv_path = os.path.join(output_dir, 'batch_log.csv')
        batch_csv_file = open(batch_csv_path, 'w', newline='')
        batch_csv_writer = csv.writer(batch_csv_file)
        batch_csv_writer.writerow([
            'epoch', 'batch', 'loss', 'landmark', 'angular_deg',
            'gaze_consist', 'shape', 'ray_target', 'pose', 'translation', 'lr',
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
        missing, unexpected = target.load_state_dict(
            fork_state['model_state_dict'], strict=False)
        if is_main:
            if missing:
                print(f"  [fork] missing keys: {len(missing)} "
                      f"(first: {missing[:3]})")
            if unexpected:
                print(f"  [fork] unexpected keys: {len(unexpected)} "
                      f"(first: {unexpected[:3]})")

        src_epoch = fork_state['epoch']
        start_epoch = src_epoch + 1
        current_phase = get_phase(src_epoch)
        fork_phase_cfg = get_phase_config(src_epoch)
        phase_start, phase_end = fork_phase_cfg['epochs']

        # Build optimizer/scheduler matching the saved phase so state_dict
        # shapes line up, then overwrite with the checkpoint's state.
        optimizer = optim.AdamW(model.parameters(), lr=fork_phase_cfg['lr'],
                                betas=(0.5, 0.95), weight_decay=1e-4)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=phase_end - phase_start + 1)

        optimizer.load_state_dict(fork_state['optimizer_state_dict'])
        if 'scheduler_state_dict' in fork_state:
            scheduler.load_state_dict(fork_state['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in fork_state:
            scaler.load_state_dict(fork_state['scaler_state_dict'])

        best_val_loss = fork_state.get(
            'val_metrics', {}).get('total', best_val_loss)
        if is_main:
            print(f"  Loaded full state from epoch {src_epoch} "
                  f"(phase {current_phase}). New run continues at epoch "
                  f"{start_epoch}, best_val_loss={best_val_loss:.4f}")

        if start_epoch > args.epochs:
            raise RuntimeError(
                f"Cannot fork: source checkpoint is at epoch {src_epoch}, "
                f"so training would start at epoch {start_epoch}, but "
                f"--epochs={args.epochs}. Increase --epochs to extend "
                f"training past the fork point."
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
        missing, unexpected = target.load_state_dict(
            ws_state['model_state_dict'], strict=False)
        if is_main:
            if missing:
                print(f"  [warmstart] missing keys: {len(missing)} "
                      f"(first: {missing[:3]})")
            if unexpected:
                print(f"  [warmstart] unexpected keys: {len(unexpected)} "
                      f"(first: {unexpected[:3]})")
            src_stage = ws_state.get('config', {}).get('stage', '?')
            src_epoch = ws_state.get('epoch', '?')
            print(f"  Loaded weights from stage {src_stage} epoch {src_epoch}. "
                  f"Starting fresh optimizer at epoch 1 of stage {args.stage}.")

        # Translation loss formulation changed (tanh/exp → direct linear
        # meters). The old pose_head rows 6:9 were trained for the prior
        # interpretation and would output garbage under the new loss.
        # Zero-init those rows while keeping rotation rows 0:6 (which
        # converged cleanly to ~0.015 rad geodesic). With zeroed
        # weight+bias, initial translation prediction is (0,0,0) m and
        # gradient flow resumes normally from the first batch.
        if (args.reset_pose_translation and hasattr(target, 'pose_encoder')
                and hasattr(target.pose_encoder, 'pose_head')):
            with torch.no_grad():
                target.pose_encoder.pose_head.weight[6:].zero_()
                target.pose_encoder.pose_head.bias[6:].zero_()
            if is_main:
                print("  [warmstart] Reset pose_head translation rows (6:9) "
                      "to zero — translation loss reformulated to direct "
                      "cm→m SmoothL1. Rotation rows (0:6) preserved.")
        # start_epoch stays at 1, optimizer/scheduler stay None — they will
        # be created in the phase-transition block below exactly like a
        # from-scratch run.

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
        # inside train_one_epoch — see header comment on STAGE_CONFIGS above.
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
        )
        if batch_csv_file is not None:
            batch_csv_file.flush()

        # Validate with the SAME multi-view topology used in training.
        # Previously hardcoded to n_views=1, which bypassed CrossViewAttention,
        # camera embeddings, and the PoseEncoder fusion path — producing
        # uninformative val gaze metrics. The val loader now delivers
        # 9-grouped batches (see _create_mds_mv_loader), so the reshape
        # inside CrossViewAttention is well-defined.
        val_losses = validate(
            model, active_val_loader, device, epoch, cfg,
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
                epoch, args.stage, phase, f"{current_lr:.2e}",
                f"{train_losses['total']:.6f}",
                f"{train_losses['landmark']:.6f}",
                f"{train_losses['angular_deg']:.4f}",
                f"{train_losses.get('reproj', 0):.6f}",
                f"{train_losses.get('mask', 0):.6f}",
                f"{train_losses.get('ray_target', 0):.6f}",
                f"{train_losses.get('pose', 0):.6f}",
                f"{train_losses.get('translation', 0):.6f}",
                f"{val_losses['total']:.6f}",
                f"{val_losses['landmark']:.6f}",
                f"{val_losses['angular_deg']:.4f}",
                f"{val_losses['landmark_px']:.4f}",
            ])
            csv_file.flush()

            # Unwrap DDP (and torch.compile) before handing the raw module to
            # the checkpoint code; its state_dict is what's serialised.
            save_model = accelerator.unwrap_model(model)

            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']

                if ckpt_mgr is not None:
                    ckpt_mgr.save_best(
                        epoch=epoch, model=save_model, val_loss=best_val_loss,
                        val_metrics=val_losses, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler,
                        extra={'profile': args.profile},
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
                    torch.save(save_dict, os.path.join(output_dir, 'best_model.pt'))
                print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

            # Periodic + latest checkpoint
            if ckpt_mgr is not None:
                # Always save latest (for resume)
                ckpt_mgr.save(
                    epoch=epoch, model=save_model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, phase=phase,
                    train_metrics=train_losses, val_metrics=val_losses,
                    tag='latest',
                )
                # Periodic named checkpoint
                if epoch % args.ckpt_every == 0:
                    ckpt_mgr.save(
                        epoch=epoch, model=save_model, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, phase=phase,
                        train_metrics=train_losses, val_metrics=val_losses,
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

    # Same augmentation as single-view loader (train only)
    from torchvision import transforms as T
    train_transform = T.Compose([
        T.ColorJitter(brightness=0.4, contrast=0.4,
                      saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ])

    print("Creating multi-view MDS streaming loaders (train + val)...")
    train_loader_mv, val_loader_mv = create_multiview_streaming_dataloaders(
        remote_train=args.mds_train,
        remote_val=args.mds_val,
        local_cache=os.path.join(args.output_dir, 'mds_cache_mv'),
        mv_groups=hw['mv_groups'],
        num_workers=hw['num_workers'],
        transform=train_transform,
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
    parser.add_argument('--stage', type=int, default=3, choices=[1, 2, 3],
                        help='Training stage: 1=landmark+pose baseline, '
                             '2=add gaze (no bridge), 3=full pipeline (with bridge)')
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
                             '(same run, same stage, continues epoch counter)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from a specific checkpoint file '
                             '(e.g. checkpoint_epoch5.pt, best_model.pt)')
    parser.add_argument('--warmstart_from', type=str, default=None,
                        help='Warmstart from a DIFFERENT run (e.g. start Stage 2 '
                             'with weights from a completed Stage 1 run). Loads '
                             'ONLY model weights — optimizer/scheduler/epoch are '
                             'reset. A new run_id is auto-generated.')
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
    parser.add_argument('--reset_pose_translation', action='store_true',
                        help='After warmstart, zero-init pose_head rows 6:9 '
                             '(translation). Use when the translation loss '
                             'formulation changed between the source run and '
                             'the current code (e.g. tanh/exp → direct '
                             'cm→m SmoothL1). Rotation rows 0:6 are preserved.')
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
                     "--warmstart_from to start a new stage with weights "
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
                         "same run in place; --warmstart_from to start a new "
                         "stage with weights only; --fork_from to branch the "
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
