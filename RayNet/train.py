"""
RayNet v2 Training Script.

Progressive 3-phase training schedule:
  Phase 1 (epochs 1-5):   landmark head only         λ_lm=1.0, λ_gaze=0.0
  Phase 2 (epochs 6-15):  introduce gaze gently      λ_lm=1.0, λ_gaze=0.3
  Phase 3 (epochs 16-30): balance and fine-tune       λ_lm=0.5, λ_gaze=0.5

Usage:
    python train.py --data_dir /path/to/gazegene --backbone repnext_m3
"""

import argparse
import os
import csv
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from datetime import datetime

from RayNet.raynet import create_raynet
from RayNet.dataset import create_dataloaders, GazeGeneDataset
from RayNet.losses import total_loss, angular_loss, landmark_loss
from RayNet.multiview_loss import multiview_consistency_loss, sanity_check_roundtrip


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
    return 3  # default to phase 3 if beyond defined epochs


def get_phase_config(epoch):
    """Get loss weights and hyperparams for the current epoch."""
    phase = get_phase(epoch)
    return PHASE_CONFIG[phase]


# ============== Training Loop ==============

def train_one_epoch(model, train_loader, optimizer, device, epoch, cfg):
    """Run one training epoch."""
    model.train()
    use_multiview = cfg.get('multiview', False) and cfg.get('lam_reproj', 0) > 0
    running_losses = {
        'total': 0.0, 'landmark': 0.0, 'angular': 0.0, 'angular_deg': 0.0,
        'reproj': 0.0, 'mask': 0.0,
    }
    n_batches = 0

    for batch in train_loader:
        images = batch['image'].to(device)
        gt_landmarks = batch['landmark_coords'].to(device)
        gt_optical_axis = batch['optical_axis'].to(device)

        # Forward pass
        predictions = model(images)

        pred_hm = predictions['landmark_heatmaps']
        pred_coords = predictions['landmark_coords']
        pred_gaze = predictions['gaze_vector']

        # Feature map spatial dimensions
        feat_H, feat_W = pred_hm.shape[2], pred_hm.shape[3]

        # Compute single-view loss with phase-specific weights
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
                    'K': batch['K'].to(device),
                    'R_cam': batch['R_cam'].to(device),
                    'T_cam': batch['T_cam'].to(device),
                    'M_norm_inv': batch['M_norm_inv'].to(device),
                    'eyeball_center_3d': batch['eyeball_center_3d'].to(device),
                },
                lam_reproj=cfg['lam_reproj'],
                lam_mask=cfg['lam_mask'],
                img_size=images.shape[-1],
                feat_size=feat_H,
            )
            loss = loss + mv_loss
            running_losses['reproj'] += mv_components['reproj_loss'].item()
            running_losses['mask'] += mv_components['mask_loss'].item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate metrics
        running_losses['total'] += components['total_loss'].item()
        running_losses['landmark'] += components['landmark_loss'].item()
        running_losses['angular'] += components['angular_loss'].item()
        running_losses['angular_deg'] += components['angular_loss_deg'].item()
        n_batches += 1

    # Average
    for k in running_losses:
        running_losses[k] /= max(n_batches, 1)

    return running_losses


@torch.no_grad()
def validate(model, val_loader, device, epoch, cfg):
    """Run validation."""
    model.eval()
    running_losses = {
        'total': 0.0, 'landmark': 0.0, 'angular': 0.0, 'angular_deg': 0.0,
        'landmark_px': 0.0,
    }
    n_batches = 0

    for batch in val_loader:
        images = batch['image'].to(device)
        gt_landmarks = batch['landmark_coords'].to(device)
        gt_optical_axis = batch['optical_axis'].to(device)
        gt_landmarks_px = batch['landmark_coords_px'].to(device)

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

        # Also compute pixel-space landmark error for reporting
        # Scale predicted coords back to pixel space
        img_size = images.shape[-1]  # 224
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

    # Create dataloaders: standard for Phase 1, multi-view for Phases 2-3
    train_subjects = list(range(1, 47))
    val_subjects = list(range(47, 57))

    train_loader_standard, val_loader = create_dataloaders(
        base_dir=args.data_dir,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_per_subject=args.samples_per_subject,
        eye=args.eye,
        ensure_multiview=False,
    )

    train_loader_mv, val_loader_mv = create_dataloaders(
        base_dir=args.data_dir,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        batch_size=args.mv_groups,
        num_workers=args.num_workers,
        samples_per_subject=args.samples_per_subject,
        eye=args.eye,
        ensure_multiview=True,
    )

    # Run normalization sanity check on the multi-view dataset
    print("Running normalization roundtrip sanity check...")
    sanity_check_roundtrip(train_loader_mv.dataset, n_samples=50)

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
        if cfg.get('multiview') and train_loader_mv is not None:
            active_train_loader = train_loader_mv
        else:
            active_train_loader = train_loader_standard

        # Train
        train_losses = train_one_epoch(
            model, active_train_loader, optimizer, device, epoch, cfg)

        # Validate (always single-view for consistent metrics)
        val_losses = validate(model, val_loader, device, epoch, cfg)

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
            torch.save({
                'epoch': epoch,
                'phase': phase,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'val_landmark_px': val_losses['landmark_px'],
                'val_angular_deg': val_losses['angular_deg'],
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'phase': phase,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(output_dir, f'checkpoint_epoch{epoch}.pt'))

    csv_file.close()
    print(f"\nTraining complete. Results saved to {output_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description='RayNet v2 Training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to GazeGene dataset')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--backbone', type=str, default='repnext_m3',
                        choices=['repnext_m0', 'repnext_m1', 'repnext_m2',
                                 'repnext_m3', 'repnext_m4', 'repnext_m5'])
    parser.add_argument('--weight_path', type=str, default=None,
                        help='Path to pretrained backbone weights')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--samples_per_subject', type=int, default=200)
    parser.add_argument('--eye', type=str, default='L', choices=['L', 'R'])
    parser.add_argument('--mv_groups', type=int, default=2,
                        help='Multi-view batch size (number of subject-frame groups; '
                             'actual batch = mv_groups * 9 cameras)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
