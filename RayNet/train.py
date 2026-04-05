"""
RayNet v2 Training Script.

Progressive 3-phase training schedule:
  Phase 1 (epochs 1-5):   landmark head only         λ_lm=1.0, λ_gaze=0.0
  Phase 2 (epochs 6-15):  introduce gaze gently      λ_lm=1.0, λ_gaze=0.3
  Phase 3 (epochs 16-30): balance and fine-tune       λ_lm=0.5, λ_gaze=0.5

Supports hardware profiles (default, t4, l4, a10g, v100, a100, h100) with AMP,
gradient accumulation, torch.compile, and MDS/WebDataset streaming.

Usage:
    # Local dataset, default hardware
    python -m RayNet.train --data_dir /path/to/gazegene --backbone repnext_m3

    # MDS streaming from MinIO with checkpoints, on A100
    python -m RayNet.train --profile a100 --mds_streaming \
        --mds_train s3://gazegene/train --mds_val s3://gazegene/val \
        --ckpt_bucket raynet-checkpoints --minio_endpoint http://IP:9000

    # Resume interrupted run
    python -m RayNet.train --profile a100 --mds_streaming \
        --mds_train s3://gazegene/train --mds_val s3://gazegene/val \
        --ckpt_bucket raynet-checkpoints --run_id run_20260405_003135 --resume
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

from RayNet.raynet import create_raynet
from RayNet.dataset import create_dataloaders, GazeGeneDataset
from RayNet.losses import total_loss, angular_loss, landmark_loss
from RayNet.multiview_loss import multiview_consistency_loss, sanity_check_roundtrip

log = logging.getLogger(__name__)


# ============== Hardware Profiles ==============

HARDWARE_PROFILES = {
    # ---- CPU / low-end GPU (testing, debugging) ----
    'default': {
        'batch_size': 512,
        'mv_groups': 2,
        'num_workers': 4,
        'pin_memory': True,
        'amp': False,
        'amp_dtype': 'float32',
        'grad_accum_steps': 1,
        'compile_model': False,
        'tf32': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
    },
    # ---- NVIDIA T4  (16 GB, Colab free / GCP n1-standard) ----
    # FP16 is critical — T4 has weak FP32 but decent FP16 (65 TFLOPS).
    # 16 GB VRAM is tight: keep batch ≤ 512 to leave room for activations.
    't4': {
        'batch_size': 512,
        'mv_groups': 4,             # 36 samples per multi-view batch
        'num_workers': 2,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'float16',
        'grad_accum_steps': 2,      # effective batch = 1024
        'compile_model': False,     # T4 doesn't benefit much from compile
        'tf32': False,              # T4 doesn't support TF32
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
    # ---- NVIDIA L4  (24 GB, GCP g2-standard) ----
    # Ada Lovelace arch: good FP16/BF16 (121 TFLOPS FP16),
    # 24 GB allows larger batches than T4.
    'l4': {
        'batch_size': 1024,
        'mv_groups': 8,             # 72 samples per multi-view batch
        'num_workers': 4,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'float16',
        'grad_accum_steps': 2,      # effective batch = 2048
        'compile_model': True,      # Ada supports compile well
        'tf32': True,               # Ada supports TF32
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
    # ---- NVIDIA A10G  (24 GB, AWS g5) ----
    # Ampere arch, similar to L4 in VRAM but different compute profile.
    # Slightly weaker FP16 than L4 (125 vs 121 TFLOPS) but same VRAM.
    'a10g': {
        'batch_size': 1024,
        'mv_groups': 8,             # 72 samples per multi-view batch
        'num_workers': 4,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'float16',
        'grad_accum_steps': 2,      # effective batch = 2048
        'compile_model': True,
        'tf32': True,
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
    # ---- NVIDIA V100  (16 GB / 32 GB, GCP / AWS p3) ----
    # Volta: no TF32, no torch.compile benefit. Good FP16 via Tensor Cores.
    'v100': {
        'batch_size': 512,
        'mv_groups': 4,             # 36 samples per multi-view batch
        'num_workers': 4,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'float16',
        'grad_accum_steps': 2,      # effective batch = 1024
        'compile_model': False,     # Volta doesn't benefit from compile
        'tf32': False,              # Volta doesn't support TF32
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
    # ---- NVIDIA A100  (40 GB / 80 GB, GCP a2, Colab Pro+) ----
    # Ampere flagship: TF32, BF16, huge memory bandwidth (2 TB/s).
    'a100': {
        'batch_size': 2048,
        'mv_groups': 16,            # 144 samples per multi-view batch
        'num_workers': 8,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'float16',
        'grad_accum_steps': 2,      # effective batch = 4096
        'compile_model': True,
        'tf32': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
    },
    # ---- NVIDIA H100  (80 GB, GCP a3, Lambda Labs) ----
    # Hopper: FP8 support, Transformer Engine, 3.4 TB/s bandwidth.
    # BF16 preferred (less overflow risk than FP16 at similar speed).
    'h100': {
        'batch_size': 4096,
        'mv_groups': 32,            # 288 samples per multi-view batch
        'num_workers': 8,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'bfloat16',
        'grad_accum_steps': 2,      # effective batch = 8192
        'compile_model': True,
        'tf32': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
    },
}


# ============== Training Phase Configuration ==============

PHASE_CONFIG = {
    1: {
        'epochs': (1, 5),
        'lam_lm': 1.0,
        'lam_gaze': 0.0,
        'lam_reproj': 0.0,
        'lam_mask': 0.0,
        'lr': 1e-3,
        'sigma': 2.0,
        'multiview': False,
        'description': 'Landmark warmup (gaze frozen)',
    },
    2: {
        'epochs': (6, 15),
        'lam_lm': 1.0,
        'lam_gaze': 0.3,
        'lam_reproj': 0.1,
        'lam_mask': 0.05,
        'lr': 5e-4,
        'sigma': 1.5,
        'multiview': True,
        'description': 'Introduce gaze + multi-view consistency',
    },
    3: {
        'epochs': (16, 30),
        'lam_lm': 0.5,
        'lam_gaze': 0.5,
        'lam_reproj': 0.2,
        'lam_mask': 0.1,
        'lr': 1e-4,
        'sigma': 1.0,
        'multiview': True,
        'description': 'Balanced fine-tuning with full multi-view',
    },
}


def get_phase(epoch):
    """Return the training phase for a given epoch."""
    for phase, cfg in PHASE_CONFIG.items():
        start, end = cfg['epochs']
        if start <= epoch <= end:
            return phase
    return 3


def get_phase_config(epoch):
    """Get loss weights and hyperparams for the current epoch."""
    phase = get_phase(epoch)
    return PHASE_CONFIG[phase]


AMP_DTYPE_MAP = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
}


def apply_hardware_profile(args):
    """Apply hardware profile settings, allowing CLI overrides."""
    hw = HARDWARE_PROFILES[args.profile].copy()

    # CLI flags override profile defaults
    if args.batch_size is not None:
        hw['batch_size'] = args.batch_size
    if args.mv_groups is not None:
        hw['mv_groups'] = args.mv_groups
    if args.num_workers is not None:
        hw['num_workers'] = args.num_workers
    if args.grad_accum_steps is not None:
        hw['grad_accum_steps'] = args.grad_accum_steps
    if args.no_compile:
        hw['compile_model'] = False

    return hw


def setup_hardware(hw, device):
    """Configure hardware-specific optimizations."""
    if hw['tf32'] and device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        print("  TF32 enabled for matmul and cuDNN")

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")


# ============== Training Loop ==============

def train_one_epoch(model, train_loader, optimizer, device, epoch, cfg,
                    scaler=None, grad_accum_steps=1, amp_enabled=False,
                    amp_dtype=torch.float16):
    """Run one training epoch with AMP and gradient accumulation support."""
    model.train()
    use_multiview = cfg.get('multiview', False) and cfg.get('lam_reproj', 0) > 0
    if not amp_enabled:
        amp_dtype = torch.float32

    running_losses = {
        'total': 0.0, 'landmark': 0.0, 'angular': 0.0, 'angular_deg': 0.0,
        'reproj': 0.0, 'mask': 0.0,
    }
    n_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        images = batch['image'].to(device, non_blocking=True)
        gt_landmarks = batch['landmark_coords'].to(device, non_blocking=True)
        gt_optical_axis = batch['optical_axis'].to(device, non_blocking=True)

        # Forward pass with AMP autocast
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            predictions = model(images)

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

            # Multi-view consistency loss (Phase 2+)
            # Ray-based: works with unit gaze vectors, numerically stable.
            if use_multiview:
                mv_loss, mv_components = multiview_consistency_loss(
                    pred_gaze,
                    pred_coords,
                    batch['R_norm'].to(device, non_blocking=True),
                    lam_gaze_consist=cfg['lam_reproj'],
                    lam_shape=cfg['lam_mask'],
                )
                if torch.isfinite(mv_loss):
                    loss = loss + mv_loss
                running_losses['reproj'] += mv_components['gaze_consist_loss'].item()
                running_losses['mask'] += mv_components['shape_loss'].item()

            # Scale loss by accumulation steps
            loss = loss / grad_accum_steps

        # Backward pass with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step at accumulation boundary
        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        # Accumulate metrics (use unscaled loss)
        running_losses['total'] += components['total_loss'].item()
        running_losses['landmark'] += components['landmark_loss'].item()
        running_losses['angular'] += components['angular_loss'].item()
        running_losses['angular_deg'] += components['angular_loss_deg'].item()
        n_batches += 1

        # Progress logging
        total_batches = len(train_loader)
        if step == 0 or (step + 1) % max(1, total_batches // 10) == 0 or step + 1 == total_batches:
            avg_loss = running_losses['total'] / n_batches
            print(f"  [Epoch {epoch}] batch {step+1}/{total_batches} "
                  f"loss={avg_loss:.4f}", flush=True)

    # Flush any remaining gradients from incomplete accumulation
    if n_batches % grad_accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()

    for k in running_losses:
        running_losses[k] /= max(n_batches, 1)

    return running_losses


@torch.no_grad()
def validate(model, val_loader, device, epoch, cfg, amp_enabled=False,
             amp_dtype=torch.float16):
    """Run validation."""
    model.eval()
    if not amp_enabled:
        amp_dtype = torch.float32

    running_losses = {
        'total': 0.0, 'landmark': 0.0, 'angular': 0.0, 'angular_deg': 0.0,
        'landmark_px': 0.0,
    }
    n_batches = 0

    for batch in val_loader:
        images = batch['image'].to(device, non_blocking=True)
        gt_landmarks = batch['landmark_coords'].to(device, non_blocking=True)
        gt_optical_axis = batch['optical_axis'].to(device, non_blocking=True)
        gt_landmarks_px = batch['landmark_coords_px'].to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            predictions = model(images)

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

        running_losses['total'] += components['total_loss'].item()
        running_losses['landmark'] += components['landmark_loss'].item()
        running_losses['angular'] += components['angular_loss'].item()
        running_losses['angular_deg'] += components['angular_loss_deg'].item()
        running_losses['landmark_px'] += lm_px_error.item()
        n_batches += 1

    for k in running_losses:
        running_losses[k] /= max(n_batches, 1)

    return running_losses


# ============== Main Training ==============

def _build_run_config(args, hw):
    """Collect all training configuration into a single dict for metadata."""
    return {
        'profile': args.profile,
        'backbone': args.backbone,
        'epochs': args.epochs,
        'eye': args.eye,
        'hardware': hw,
        'phase_config': {
            str(k): {kk: vv for kk, vv in v.items() if kk != 'description'}
            for k, v in PHASE_CONFIG.items()
        },
        'streaming': args.streaming,
        'dataset_url': getattr(args, 'dataset_url', None),
        'data_dir': getattr(args, 'data_dir', None),
        'weight_path': args.weight_path,
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Apply hardware profile
    hw = apply_hardware_profile(args)
    setup_hardware(hw, device)

    # Resolve AMP dtype from profile string
    amp_dtype = AMP_DTYPE_MAP.get(hw['amp_dtype'], torch.float16)

    print(f"Profile: {args.profile}")
    print(f"  Batch size: {hw['batch_size']} (effective: "
          f"{hw['batch_size'] * hw['grad_accum_steps']})")
    print(f"  AMP: {hw['amp']} ({hw['amp_dtype']})")
    print(f"  Gradient accumulation: {hw['grad_accum_steps']} steps")
    print(f"  torch.compile: {hw['compile_model']}")

    # Checkpoint manager (MinIO)
    ckpt_mgr = _init_checkpoint_manager(args)
    if ckpt_mgr is not None:
        print(f"  MinIO checkpoints: s3://{args.ckpt_bucket}/{args.ckpt_prefix}/{ckpt_mgr.run_id}/")

    # Create output directory
    if ckpt_mgr is not None:
        output_dir = os.path.join(args.output_dir, ckpt_mgr.run_id)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Create model
    model = create_raynet(
        backbone_name=args.backbone,
        weight_path=args.weight_path,
        n_landmarks=14,
    )

    if hw['compile_model'] and hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # AMP scaler
    scaler = GradScaler('cuda', enabled=hw['amp']) if hw['amp'] else None

    # --- Data loading ---
    # Three modes: local disk, MDS streaming (MosaicML + MinIO), WebDataset streaming
    if args.mds_streaming:
        train_loader_standard, val_loader = _create_mds_loaders(args, hw)
        train_loader_mv = None  # created lazily in Phase 2
        streaming_mode = False  # MDS loaders are persistent, not recreated per-phase
    elif args.streaming:
        _create_streaming_loaders(args, hw)
        train_loader_standard = None
        train_loader_mv = None
        val_loader = None
        streaming_mode = True
    else:
        streaming_mode = False
        train_loader_standard, train_loader_mv, val_loader = _create_local_loaders(
            args, hw)

    # Record config in checkpoint metadata
    run_config = _build_run_config(args, hw)
    if ckpt_mgr is not None:
        ckpt_mgr.set_config(run_config)

    # CSV logger
    csv_path = os.path.join(output_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'epoch', 'phase', 'lr',
        'train_total', 'train_landmark', 'train_angular_deg',
        'train_reproj', 'train_mask',
        'val_total', 'val_landmark', 'val_angular_deg', 'val_landmark_px',
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
        print(f"Resuming from run {ckpt_mgr.run_id} ...")
        # We need optimizer & scheduler to exist before resume_state.
        # Create them for the starting phase; resume will overwrite state.
        resume_phase_cfg = get_phase_config(1)
        optimizer = optim.AdamW(model.parameters(), lr=resume_phase_cfg['lr'],
                                betas=(0.5, 0.95), weight_decay=1e-4)
        phase_start, phase_end = resume_phase_cfg['epochs']
        scheduler = CosineAnnealingLR(optimizer, T_max=phase_end - phase_start + 1)

        resume_file = getattr(args, 'resume_from', None)
        start_epoch, resume_ckpt = ckpt_mgr.resume_state(
            model, optimizer, scheduler=scheduler, scaler=scaler,
            map_location=device, filename=resume_file,
        )
        current_phase = get_phase(resume_ckpt['epoch'])
        best_val_loss = resume_ckpt.get('val_metrics', {}).get('total', best_val_loss)
        print(f"  Resumed at epoch {start_epoch} (phase {current_phase}), "
              f"best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        phase = get_phase(epoch)
        cfg = get_phase_config(epoch)

        # Phase transition: update optimizer and scheduler
        if phase != current_phase:
            current_phase = phase
            print(f"\n{'='*60}")
            print(f"Phase {phase}: {cfg['description']}")
            print(f"  lr={cfg['lr']}, lam_lm={cfg['lam_lm']}, "
                  f"lam_gaze={cfg['lam_gaze']}, sigma={cfg['sigma']}")
            if cfg.get('multiview'):
                print(f"  Multi-view: lam_gaze_consist={cfg['lam_reproj']}, "
                      f"lam_shape={cfg['lam_mask']}")
            print(f"{'='*60}")

            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg['lr'],
                betas=(0.5, 0.95),
                weight_decay=1e-4,
            )
            phase_start, phase_end = cfg['epochs']
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=phase_end - phase_start + 1,
            )

        # Select dataloader based on phase
        if streaming_mode:
            active_train_loader, active_val_loader = _get_streaming_loaders(
                args, hw, cfg)
        else:
            if cfg.get('multiview'):
                # Lazy-create multi-view loader on first use (MDS or local)
                if train_loader_mv is None and args.mds_streaming:
                    train_loader_mv = _create_mds_mv_loader(args, hw)
                if train_loader_mv is not None:
                    active_train_loader = train_loader_mv
                else:
                    active_train_loader = train_loader_standard
            else:
                active_train_loader = train_loader_standard
            active_val_loader = val_loader

        # Train
        train_losses = train_one_epoch(
            model, active_train_loader, optimizer, device, epoch, cfg,
            scaler=scaler,
            grad_accum_steps=hw['grad_accum_steps'],
            amp_enabled=hw['amp'],
            amp_dtype=amp_dtype,
        )

        # Validate
        val_losses = validate(
            model, active_val_loader, device, epoch, cfg,
            amp_enabled=hw['amp'],
            amp_dtype=amp_dtype,
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        mv_str = ""
        if cfg.get('multiview'):
            mv_str = (f" gaze_mv={train_losses['reproj']:.4f}"
                      f" shape={train_losses['mask']:.4f}")
        print(f"Epoch {epoch:3d} | Phase {phase} | lr {current_lr:.2e} | "
              f"Train: loss={train_losses['total']:.4f} lm={train_losses['landmark']:.4f} "
              f"ang={train_losses['angular_deg']:.2f}deg{mv_str} | "
              f"Val: loss={val_losses['total']:.4f} lm_px={val_losses['landmark_px']:.2f}px "
              f"ang={val_losses['angular_deg']:.2f}deg")

        if hw['amp'] and device.type == 'cuda':
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"  GPU memory peak: {mem_gb:.1f} GB")

        csv_writer.writerow([
            epoch, phase, f"{current_lr:.2e}",
            f"{train_losses['total']:.6f}",
            f"{train_losses['landmark']:.6f}",
            f"{train_losses['angular_deg']:.4f}",
            f"{train_losses.get('reproj', 0):.6f}",
            f"{train_losses.get('mask', 0):.6f}",
            f"{val_losses['total']:.6f}",
            f"{val_losses['landmark']:.6f}",
            f"{val_losses['angular_deg']:.4f}",
            f"{val_losses['landmark_px']:.4f}",
        ])
        csv_file.flush()

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']

            if ckpt_mgr is not None:
                ckpt_mgr.save_best(
                    epoch=epoch, model=model, val_loss=best_val_loss,
                    val_metrics=val_losses, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler,
                    extra={'profile': args.profile},
                )
            else:
                save_dict = {
                    'epoch': epoch,
                    'phase': phase,
                    'model_state_dict': (model._orig_mod.state_dict()
                                         if hasattr(model, '_orig_mod')
                                         else model.state_dict()),
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
                epoch=epoch, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, phase=phase,
                train_metrics=train_losses, val_metrics=val_losses,
                tag='latest',
            )
            # Periodic named checkpoint
            if epoch % args.ckpt_every == 0:
                ckpt_mgr.save(
                    epoch=epoch, model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, phase=phase,
                    train_metrics=train_losses, val_metrics=val_losses,
                )
        else:
            if epoch % args.ckpt_every == 0:
                save_dict = {
                    'epoch': epoch,
                    'phase': phase,
                    'model_state_dict': (model._orig_mod.state_dict()
                                         if hasattr(model, '_orig_mod')
                                         else model.state_dict()),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if scaler is not None:
                    save_dict['scaler_state_dict'] = scaler.state_dict()
                torch.save(save_dict, os.path.join(output_dir, f'checkpoint_epoch{epoch}.pt'))

    csv_file.close()
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

    # Sanity check
    print("Running normalization roundtrip sanity check...")
    sanity_check_roundtrip(train_loader_mv.dataset, n_samples=50)

    return train_loader_standard, train_loader_mv, val_loader


def _create_mds_loaders(args, hw):
    """Create MosaicML MDS streaming dataloaders from MinIO / S3."""
    from RayNet.streaming import create_streaming_dataloaders
    from RayNet.streaming.minio_utils import configure_minio_env

    # Configure MinIO env vars if endpoint is provided
    if args.minio_endpoint:
        configure_minio_env(
            args.minio_endpoint,
            os.environ.get('AWS_ACCESS_KEY_ID', 'minioadmin'),
            os.environ.get('AWS_SECRET_ACCESS_KEY', 'minioadmin'),
        )

    print(f"MDS streaming: train={args.mds_train}, val={args.mds_val}")

    train_loader, val_loader = create_streaming_dataloaders(
        remote_train=args.mds_train,
        remote_val=args.mds_val,
        local_cache=os.path.join(args.output_dir, 'mds_cache'),
        batch_size=hw['batch_size'],
        num_workers=hw['num_workers'],
        pin_memory=hw['pin_memory'],
        prefetch_factor=hw['prefetch_factor'],
        persistent_workers=hw['persistent_workers'],
        download_timeout=120,
    )

    print(f"  Train dataset: {len(train_loader.dataset)} samples, "
          f"{len(train_loader)} batches")
    print(f"  Val dataset:   {len(val_loader.dataset)} samples, "
          f"{len(val_loader)} batches")

    return train_loader, val_loader


def _create_mds_mv_loader(args, hw):
    """Create multi-view MDS loader lazily (only called when Phase 2 starts)."""
    from RayNet.streaming.dataset import create_multiview_streaming_dataloaders

    print("Creating multi-view MDS streaming loader...")
    train_loader_mv, _ = create_multiview_streaming_dataloaders(
        remote_train=args.mds_train,
        remote_val=args.mds_val,
        local_cache=os.path.join(args.output_dir, 'mds_cache_mv'),
        mv_groups=hw['mv_groups'],
        num_workers=hw['num_workers'],
        pin_memory=hw['pin_memory'],
        prefetch_factor=hw['prefetch_factor'],
        persistent_workers=hw['persistent_workers'],
    )
    return train_loader_mv


def _create_streaming_loaders(args, hw):
    """Validate streaming args at startup (WebDataset mode)."""
    if not args.dataset_url:
        raise ValueError("--dataset_url required when --streaming is set")
    print(f"WebDataset streaming mode: {args.dataset_url}")


def _get_streaming_loaders(args, hw, cfg):
    """Create streaming dataloaders for the current phase."""
    from RayNet.webdataset_utils import (
        create_streaming_dataloader,
        create_multiview_streaming_dataloader,
    )

    if cfg.get('multiview'):
        train_loader = create_multiview_streaming_dataloader(
            urls=args.dataset_url,
            mv_groups=hw['mv_groups'],
            num_workers=hw['num_workers'],
            shuffle=True,
        )
    else:
        train_loader = create_streaming_dataloader(
            urls=args.dataset_url,
            batch_size=hw['batch_size'],
            num_workers=hw['num_workers'],
            shuffle=True,
        )

    val_url = args.val_dataset_url or args.dataset_url
    val_loader = create_streaming_dataloader(
        urls=val_url,
        batch_size=hw['batch_size'],
        num_workers=hw['num_workers'],
        shuffle=False,
    )

    return train_loader, val_loader


# ============== CLI ==============

def parse_args():
    parser = argparse.ArgumentParser(description='RayNet v2 Training')

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

    # WebDataset streaming (legacy)
    parser.add_argument('--streaming', action='store_true',
                        help='Use WebDataset streaming instead of local disk')
    parser.add_argument('--dataset_url', type=str, default=None,
                        help='WebDataset shard URL pattern for training')
    parser.add_argument('--val_dataset_url', type=str, default=None,
                        help='WebDataset shard URL pattern for validation')

    # Model
    parser.add_argument('--backbone', type=str, default='repnext_m3',
                        choices=['repnext_m0', 'repnext_m1', 'repnext_m2',
                                 'repnext_m3', 'repnext_m4', 'repnext_m5'])
    parser.add_argument('--weight_path', type=str, default=None,
                        help='Path to pretrained backbone weights')

    # Hardware profile
    parser.add_argument('--profile', type=str, default='default',
                        choices=list(HARDWARE_PROFILES.keys()),
                        help=f'Hardware profile ({", ".join(HARDWARE_PROFILES.keys())})')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile even if profile enables it')

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
                        help='Resume training from the latest checkpoint of --run_id')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from a specific checkpoint file '
                             '(e.g. checkpoint_epoch5.pt, best_model.pt)')

    args = parser.parse_args()

    if not args.streaming and not args.mds_streaming and args.data_dir is None:
        parser.error("--data_dir is required when not using --streaming or --mds_streaming")

    if args.mds_streaming and (not args.mds_train or not args.mds_val):
        parser.error("--mds_streaming requires both --mds_train and --mds_val")

    # --resume_from implies --resume
    if args.resume_from:
        args.resume = True

    if args.resume and not args.run_id:
        parser.error("--resume requires --run_id to identify which run to resume")
    if args.resume and not args.ckpt_bucket:
        parser.error("--resume requires --ckpt_bucket")

    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)
