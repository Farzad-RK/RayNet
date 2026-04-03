"""
RayNet v2 Training Script.

Progressive 3-phase training schedule:
  Phase 1 (epochs 1-5):   landmark head only         λ_lm=1.0, λ_gaze=0.0
  Phase 2 (epochs 6-15):  introduce gaze gently      λ_lm=1.0, λ_gaze=0.3
  Phase 3 (epochs 16-30): balance and fine-tune       λ_lm=0.5, λ_gaze=0.5

Supports hardware profiles (default, a100) with AMP, gradient accumulation,
torch.compile, and WebDataset streaming.

Usage:
    # Local dataset, default hardware
    python train.py --data_dir /path/to/gazegene --backbone repnext_m3

    # A100 optimized
    python train.py --data_dir /path/to/gazegene --profile a100

    # Streaming from HF Hub on A100
    python train.py --profile a100 --streaming \
        --dataset_url "pipe:curl -sL https://huggingface.co/.../train/gazegene-train-{000000..000099}.tar"
"""

import argparse
import os
import csv
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from datetime import datetime

from RayNet.raynet import create_raynet
from RayNet.dataset import create_dataloaders, GazeGeneDataset
from RayNet.losses import total_loss, angular_loss, landmark_loss
from RayNet.multiview_loss import multiview_consistency_loss, sanity_check_roundtrip


# ============== Hardware Profiles ==============

HARDWARE_PROFILES = {
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
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")


# ============== Training Loop ==============

def train_one_epoch(model, train_loader, optimizer, device, epoch, cfg,
                    scaler=None, grad_accum_steps=1, amp_enabled=False):
    """Run one training epoch with AMP and gradient accumulation support."""
    model.train()
    use_multiview = cfg.get('multiview', False) and cfg.get('lam_reproj', 0) > 0
    amp_dtype = torch.float16 if amp_enabled else torch.float32

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
            if use_multiview:
                mv_loss, mv_components = multiview_consistency_loss(
                    pred_coords,
                    batch_meta={
                        'K': batch['K'].to(device, non_blocking=True),
                        'R_cam': batch['R_cam'].to(device, non_blocking=True),
                        'T_cam': batch['T_cam'].to(device, non_blocking=True),
                        'M_norm_inv': batch['M_norm_inv'].to(device, non_blocking=True),
                        'eyeball_center_3d': batch['eyeball_center_3d'].to(device, non_blocking=True),
                    },
                    lam_reproj=cfg['lam_reproj'],
                    lam_mask=cfg['lam_mask'],
                    img_size=images.shape[-1],
                    feat_size=feat_H,
                )
                loss = loss + mv_loss
                running_losses['reproj'] += mv_components['reproj_loss'].item()
                running_losses['mask'] += mv_components['mask_loss'].item()

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
def validate(model, val_loader, device, epoch, cfg, amp_enabled=False):
    """Run validation."""
    model.eval()
    amp_dtype = torch.float16 if amp_enabled else torch.float32

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

def train(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Apply hardware profile
    hw = apply_hardware_profile(args)
    setup_hardware(hw, device)

    print(f"Profile: {args.profile}")
    print(f"  Batch size: {hw['batch_size']} (effective: "
          f"{hw['batch_size'] * hw['grad_accum_steps']})")
    print(f"  AMP: {hw['amp']} ({hw['amp_dtype']})")
    print(f"  Gradient accumulation: {hw['grad_accum_steps']} steps")
    print(f"  torch.compile: {hw['compile_model']}")

    # Create output directory
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
    scaler = GradScaler(enabled=hw['amp']) if hw['amp'] else None

    # --- Data loading ---
    if args.streaming:
        _create_streaming_loaders(args, hw)
        # Streaming mode: loaders created per-phase in the training loop
        train_loader_standard = None
        train_loader_mv = None
        val_loader = None
        streaming_mode = True
    else:
        streaming_mode = False
        train_loader_standard, train_loader_mv, val_loader = _create_local_loaders(
            args, hw)

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

    for epoch in range(1, args.epochs + 1):
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
                print(f"  Multi-view: lam_reproj={cfg['lam_reproj']}, "
                      f"lam_mask={cfg['lam_mask']}")
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
            if cfg.get('multiview') and train_loader_mv is not None:
                active_train_loader = train_loader_mv
            else:
                active_train_loader = train_loader_standard
            active_val_loader = val_loader

        # Train
        train_losses = train_one_epoch(
            model, active_train_loader, optimizer, device, epoch, cfg,
            scaler=scaler,
            grad_accum_steps=hw['grad_accum_steps'],
            amp_enabled=hw['amp'],
        )

        # Validate
        val_losses = validate(
            model, active_val_loader, device, epoch, cfg,
            amp_enabled=hw['amp'],
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        mv_str = ""
        if cfg.get('multiview'):
            mv_str = (f" reproj={train_losses['reproj']:.4f}"
                      f" mask={train_losses['mask']:.4f}")
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

        # Periodic checkpoint
        if epoch % 5 == 0:
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


def _create_streaming_loaders(args, hw):
    """Validate streaming args at startup."""
    if not args.dataset_url:
        raise ValueError("--dataset_url required when --streaming is set")
    print(f"Streaming mode: {args.dataset_url}")


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

    # Streaming
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
                        help='Hardware profile (default, a100)')
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

    args = parser.parse_args()

    if not args.streaming and args.data_dir is None:
        parser.error("--data_dir is required when not using --streaming")

    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)
