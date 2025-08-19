import argparse
import os
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from raynet import create_raynet_model, RayNetLoss, compute_enhanced_metrics

ENHANCED_MODEL = True

from dataset import GazeGeneDataset, EnhancedMultiViewBatchSampler
from head_pose.loss import multiview_headpose_losses

from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced RayNet Training with Iris Mesh Regression")
    parser.add_argument('--base_dir', type=str, required=True, help="Root of GazeGene dataset")
    parser.add_argument('--backbone_name', type=str, default="repnext_m3")
    parser.add_argument('--weight_path', type=str, default="./repnext_m3_pretrained.pt")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--samples_per_subject', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    parser.add_argument('--log_csv', type=str, default="train_log.csv")
    parser.add_argument('--checkpoint_freq', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', type=str, default="train", choices=["train", "test"],
                        help="Use official GazeGene train/test split")

    # New arguments for enhanced training
    parser.add_argument('--include_2d_landmarks', action="store_true", default=True,
                        help="Include 2D landmark supervision")
    parser.add_argument('--progressive_training', action="store_true", default=False,
                        help="Use progressive training strategy")
    parser.add_argument('--image_size', type=int, nargs=2, default=[448, 448],
                        help="Input image size (H W)")

    # Loss weights
    parser.add_argument('--head_pose_weight', type=float, default=1.0)
    parser.add_argument('--iris_mesh_weight', type=float, default=1.0)
    parser.add_argument('--reconstruction_3d_weight', type=float, default=1.0)
    parser.add_argument('--reconstruction_2d_weight', type=float, default=0.5)
    parser.add_argument('--projection_consistency_weight', type=float, default=0.3)
    parser.add_argument('--spherical_weight', type=float, default=0.1)
    parser.add_argument('--circular_weight', type=float, default=0.1)
    parser.add_argument('--smoothing_weight', type=float, default=0.05)
    parser.add_argument('--edge_weight', type=float, default=0.05)
    parser.add_argument('--geometric_weight', type=float, default=0.1)
    parser.add_argument('--depth_consistency_weight', type=float, default=0.05)

    return parser.parse_args()


class ProgressiveTrainingScheduler:
    """
    Manages progressive training phases for iris mesh regression.
    """

    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.phase_transitions = [
            int(0.3 * total_epochs),  # Phase 1: 30% - 3D only
            int(0.6 * total_epochs),  # Phase 2: 60% - Add 2D supervision
            int(0.8 * total_epochs),  # Phase 3: 80% - Add projection consistency
        ]

    def get_phase_weights(self, epoch):
        """Get loss weights for current training phase."""
        if epoch < self.phase_transitions[0]:
            # Phase 1: 3D reconstruction + geometric constraints only
            return {
                'reconstruction_3d_weight': 1.0,
                'reconstruction_2d_weight': 0.0,
                'projection_consistency_weight': 0.0,
                'spherical_weight': 0.1,
                'circular_weight': 0.1,
                'smoothing_weight': 0.05,
                'edge_weight': 0.05,
                'geometric_weight': 0.1,
                'depth_consistency_weight': 0.05
            }
        elif epoch < self.phase_transitions[1]:
            # Phase 2: Add 2D supervision
            return {
                'reconstruction_3d_weight': 1.0,
                'reconstruction_2d_weight': 0.3,
                'projection_consistency_weight': 0.0,
                'spherical_weight': 0.1,
                'circular_weight': 0.1,
                'smoothing_weight': 0.05,
                'edge_weight': 0.05,
                'geometric_weight': 0.1,
                'depth_consistency_weight': 0.05
            }
        elif epoch < self.phase_transitions[2]:
            # Phase 3: Add projection consistency
            return {
                'reconstruction_3d_weight': 1.0,
                'reconstruction_2d_weight': 0.5,
                'projection_consistency_weight': 0.2,
                'spherical_weight': 0.1,
                'circular_weight': 0.1,
                'smoothing_weight': 0.05,
                'edge_weight': 0.05,
                'geometric_weight': 0.1,
                'depth_consistency_weight': 0.05
            }
        else:
            # Phase 4: Full training
            return {
                'reconstruction_3d_weight': 1.0,
                'reconstruction_2d_weight': 0.5,
                'projection_consistency_weight': 0.3,
                'spherical_weight': 0.1,
                'circular_weight': 0.1,
                'smoothing_weight': 0.05,
                'edge_weight': 0.05,
                'geometric_weight': 0.1,
                'depth_consistency_weight': 0.05
            }


class RunningNormalizer:
    """Simple online normalizer for loss values."""

    def __init__(self, init_min=1e8, init_max=-1e8):
        self.min = init_min
        self.max = init_max

    def update(self, val):
        v = float(val)
        if v < self.min:
            self.min = v
        if v > self.max:
            self.max = v

    def normalize(self, val):
        if self.max - self.min < 1e-8:
            return 0.0
        return (float(val) - self.min) / (self.max - self.min)


def create_datasets_and_loaders(args):
    """Create enhanced datasets and loaders."""
    if args.split == "train":
        # Use string format like original dataset expects
        train_subjects = [f"subject{i}" for i in range(1, 47)]  # Subjects 1-46
        val_subjects = [f"subject{i}" for i in range(47, 57)]  # Subjects 47-56
    else:
        # For testing, use the official test split
        train_subjects = [f"subject{i}" for i in range(47, 57)]
        val_subjects = None

    print(f"Loading training subjects: {train_subjects[:5]}...{train_subjects[-5:]}")

    # Training dataset
    train_dataset = GazeGeneDataset(
        base_dir=args.base_dir,
        subject_ids=train_subjects,
        samples_per_subject=args.samples_per_subject,
        include_2d_landmarks=args.include_2d_landmarks,
        include_camera_params=True,
        balance_attributes=['ethicity']
    )

    train_sampler = EnhancedMultiViewBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        balance_attributes=['ethicity'],
        ensure_multiview=True,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Validation dataset (if available)
    val_loader = None
    if val_subjects is not None:
        print(f"Loading validation subjects: {val_subjects[:3]}...{val_subjects[-3:]}")
        val_dataset = GazeGeneDataset(
            base_dir=args.base_dir,
            subject_ids=val_subjects,
            samples_per_subject=min(100, args.samples_per_subject) if args.samples_per_subject else 100,
            include_2d_landmarks=args.include_2d_landmarks,
            include_camera_params=True,
            balance_attributes=None
        )

        val_sampler = EnhancedMultiViewBatchSampler(
            val_dataset,
            batch_size=args.batch_size,
            ensure_multiview=True,
            shuffle=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )

    print(f"Dataset statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    if val_loader:
        print(f"  Validation samples: {len(val_loader.dataset)}")

    # Debug: Print some dataset info
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Image shape: {sample['img'].shape}")
        if 'mesh' in sample:
            print(f"  3D mesh keys: {list(sample['mesh'].keys())}")
            print(f"  Iris mesh 3D shape: {sample['mesh']['iris_mesh_3D'].shape}")
    else:
        print("  WARNING: No training samples found!")
        print(f"  Base directory exists: {os.path.exists(args.base_dir)}")
        if os.path.exists(args.base_dir):
            subdirs = [d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))]
            print(f"  Subdirectories found: {subdirs[:10]}")

    return train_loader, val_loader


def get_loss_config(args, epoch=None, progressive_scheduler=None):
    """Get loss configuration based on arguments and training phase."""
    if progressive_scheduler and epoch is not None:
        return progressive_scheduler.get_phase_weights(epoch)
    else:
        return {
            'reconstruction_3d_weight': args.reconstruction_3d_weight,
            'reconstruction_2d_weight': args.reconstruction_2d_weight,
            'projection_consistency_weight': args.projection_consistency_weight,
            'spherical_weight': args.spherical_weight,
            'circular_weight': args.circular_weight,
            'smoothing_weight': args.smoothing_weight,
            'edge_weight': args.edge_weight,
            'geometric_weight': args.geometric_weight,
            'depth_consistency_weight': args.depth_consistency_weight
        }


def train_step(model, batch, loss_fn, optimizer, args, device):
    """Single training step."""
    model.train()
    optimizer.zero_grad()

    # Extract data from batch
    images = batch['img'].to(device)  # [B*9, 3, H, W]

    # Get batch size (accounting for 9 views per sample)
    total_batch_size = images.shape[0]
    B = total_batch_size // 9

    # Reshape for multi-view processing
    images = images.view(B * 9, *images.shape[1:])

    # Forward pass - handle both enhanced and basic models
    if ENHANCED_MODEL and hasattr(model, 'iris_mesh_regression'):
        # Enhanced model with iris mesh regression
        intrinsics = batch['intrinsic'].to(device)  # [B*9, 3, 3]
        intrinsics = intrinsics.view(B * 9, *intrinsics.shape[1:])

        predictions = model(
            images,
            intrinsic_matrix=intrinsics,
            image_size=tuple(args.image_size)
        )

        # Reshape predictions for multi-view loss computation
        for key in predictions:
            if torch.is_tensor(predictions[key]):
                if key == "head_pose_6d":
                    predictions[key] = predictions[key].view(B, 9, -1)
                elif "iris_mesh" in key:
                    predictions[key] = predictions[key].view(B, 9, *predictions[key].shape[1:])
                elif key == "pupil_centers_3d":
                    predictions[key] = predictions[key].view(B, 9, *predictions[key].shape[1:])

        # Prepare targets (ensure consistent shapes)
        targets = {}
        for key in ['mesh', 'head_pose']:
            if key in batch:
                targets[key] = {}
                for subkey in batch[key]:
                    if torch.is_tensor(batch[key][subkey]):
                        # Move to device and reshape
                        target_tensor = batch[key][subkey].to(device)
                        targets[key][subkey] = target_tensor.view(B, 9, *target_tensor.shape[1:])
                    else:
                        targets[key][subkey] = batch[key][subkey]

        # Add image size to targets for 2D normalization
        targets['image_size'] = tuple(args.image_size)

    else:
        # Basic model - original RayNet
        predictions = model(images)

        # Prepare targets for basic loss - MOVE TO DEVICE
        targets = {
            'head_pose': {
                'R': batch['head_pose']['R'].to(device)
            }
        }

    # Compute loss
    total_loss, individual_losses = loss_fn(predictions, targets)

    # Backward pass
    total_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # Compute metrics
    with torch.no_grad():
        if ENHANCED_MODEL:
            metrics = compute_enhanced_metrics(predictions, targets)
        else:
            # Basic metrics - just head pose error
            metrics = {}
            if "head_pose_6d" in predictions:
                from utils import ortho6d_to_rotmat
                pred_rotmat = ortho6d_to_rotmat(predictions["head_pose_6d"].view(-1, 6))
                gt_rotmat = targets["head_pose"]["R"].view(-1, 3, 3)

                trace = torch.sum(pred_rotmat * gt_rotmat, dim=(1, 2))
                cos_angle = (trace - 1) / 2
                cos_angle = torch.clamp(cos_angle, -1, 1)
                angle_error = torch.acos(cos_angle) * 180 / 3.14159
                metrics['head_pose_error_deg'] = torch.mean(angle_error)

    return {
        'total_loss': total_loss.item(),
        'losses': {k: v.item() if torch.is_tensor(v) else v for k, v in individual_losses.items()},
        'metrics': {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
    }


def validate_step(model, val_loader, loss_fn, args, device):
    """Validation step."""
    model.eval()
    val_losses = defaultdict(list)
    val_metrics = defaultdict(list)

    with torch.no_grad():
        for batch in val_loader:
            # Same processing as training step
            images = batch['img'].to(device)
            intrinsics = batch['intrinsic'].to(device)

            total_batch_size = images.shape[0]
            B = total_batch_size // 9

            images = images.view(B * 9, *images.shape[1:])
            intrinsics = intrinsics.view(B * 9, *intrinsics.shape[1:])

            predictions = model(
                images,
                intrinsic_matrix=intrinsics,
                image_size=tuple(args.image_size)
            )

            # Reshape predictions
            for key in predictions:
                if torch.is_tensor(predictions[key]):
                    if key == "head_pose_6d":
                        predictions[key] = predictions[key].view(B, 9, -1)
                    elif "iris_mesh" in key:
                        predictions[key] = predictions[key].view(B, 9, *predictions[key].shape[1:])
                    elif key == "pupil_centers_3d":
                        predictions[key] = predictions[key].view(B, 9, *predictions[key].shape[1:])

            # Prepare targets
            targets = batch.copy()
            for key in ['mesh', 'head_pose']:
                if key in targets:
                    for subkey in targets[key]:
                        if torch.is_tensor(targets[key][subkey]):
                            targets[key][subkey] = targets[key][subkey].view(B, 9, *targets[key][subkey].shape[1:])

            targets['image_size'] = tuple(args.image_size)

            if 'mesh' in targets and 'iris_mesh_3D' in targets['mesh']:
                targets['mesh']['iris_mesh_3D'] = targets['mesh']['iris_mesh_3D'].to(device)

            def move_all_to_device(obj, device):
                if isinstance(obj, torch.Tensor):
                    return obj.to(device)
                elif isinstance(obj, dict):
                    return {k: move_all_to_device(v, device) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [move_all_to_device(item, device) for item in obj]
                else:
                    return obj

            targets = move_all_to_device(targets, device)

            # Compute loss and metrics
            total_loss, individual_losses = loss_fn(predictions, targets)
            metrics = compute_enhanced_metrics(predictions, targets)

            # Accumulate results
            val_losses['total'].append(total_loss.item())
            for k, v in individual_losses.items():
                val_losses[k].append(v.item() if torch.is_tensor(v) else v)
            for k, v in metrics.items():
                val_metrics[k].append(v.item() if torch.is_tensor(v) else v)

    # Average results
    avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
    avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}

    return avg_losses, avg_metrics


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets and loaders
    print("Creating datasets...")
    train_loader, val_loader = create_datasets_and_loaders(args)
    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    print("Creating model...")
    if ENHANCED_MODEL:
        model = create_raynet_model(args.backbone_name, args.weight_path)
    else:
        print("Using fallback basic RayNet model")
        # Fallback to basic model creation
        backbone_channels_dict = {
            'repnext_m0': [40, 80, 160, 320],
            'repnext_m1': [48, 96, 192, 384],
            'repnext_m2': [56, 112, 224, 448],
            'repnext_m3': [64, 128, 256, 512],
            'repnext_m4': [64, 128, 256, 512],
            'repnext_m5': [80, 160, 320, 640],
        }
        backbone = load_pretrained_repnext(args.backbone_name, args.weight_path)
        in_channels_list = backbone_channels_dict[args.backbone_name]
        model = RayNet(backbone, in_channels_list, panet_out_channels=256).to(device)

    # Progressive training scheduler
    progressive_scheduler = None
    if args.progressive_training:
        progressive_scheduler = ProgressiveTrainingScheduler(args.epochs)
        print("Using progressive training strategy")

    # Create loss function
    if ENHANCED_MODEL:
        iris_loss_config = get_loss_config(args, epoch=0, progressive_scheduler=progressive_scheduler)
        loss_fn = RayNetLoss(
            head_pose_weight=args.head_pose_weight,
            iris_mesh_weight=args.iris_mesh_weight,
            iris_loss_config=iris_loss_config
        )
    else:
        # Fallback: use only head pose loss
        def basic_loss_fn(predictions, targets):
            # Simple head pose loss only
            if "head_pose_6d" in predictions and "head_pose" in targets:
                pred_6d = predictions["head_pose_6d"]  # [B*9, 6]
                gt_rotmat = targets["head_pose"]["R"]  # [B*9, 3, 3]

                # Reshape for multi-view processing
                B = pred_6d.shape[0] // 9
                pred_6d_reshaped = pred_6d.view(B, 9, 6)
                gt_rotmat_reshaped = gt_rotmat.view(B, 9, 3, 3)

                head_pose_losses = multiview_headpose_losses(pred_6d_reshaped, gt_rotmat_reshaped)
                total_loss = head_pose_losses['accuracy'] + 0.1 * head_pose_losses['consistency']
                return total_loss, {'total': total_loss, **head_pose_losses}
            else:
                return torch.tensor(0.0, device=device), {'total': torch.tensor(0.0, device=device)}

        loss_fn = basic_loss_fn

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Setup logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logfile = open(args.log_csv, "w", newline="")
    fieldnames = [
        "epoch", "step", "batch_size", "lr", "phase",
        "total_loss", "head_pose_accuracy", "head_pose_consistency",
        "iris_reconstruction_3d", "iris_reconstruction_2d", "iris_projection_consistency",
        "iris_spherical", "iris_circular", "iris_smoothing", "iris_edge_consistency",
        "iris_mesh_3d_l2_error", "iris_mesh_2d_pixel_error", "projection_consistency_error",
        "val_total_loss", "val_iris_mesh_3d_l2_error"
    ]
    csv_writer = csv.DictWriter(logfile, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Resume from checkpoint if available
    start_epoch = 0
    checkpoint_files = sorted([f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pth')])
    if checkpoint_files:
        last_ckpt = os.path.join(args.checkpoint_dir, checkpoint_files[-1])
        print(f"Resuming from checkpoint: {last_ckpt}")
        checkpoint = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    # Training loop
    print("Starting training...")
    print(f"Total epochs: {args.epochs}")
    print(f"Steps per epoch: {len(train_loader)}")

    if len(train_loader) == 0:
        print("ERROR: No training data available! Check your dataset path and subject IDs.")
        return

    for epoch in range(start_epoch, args.epochs):
        # Update loss weights for progressive training
        if progressive_scheduler:
            iris_loss_config = get_loss_config(args, epoch, progressive_scheduler)
            loss_fn.iris_loss_config = iris_loss_config
            phase = min(3, epoch // (args.epochs // 4))
            print(f"Epoch {epoch + 1}/{args.epochs} - Training phase: {phase + 1}/4")
        else:
            phase = 0
            print(f"Epoch {epoch + 1}/{args.epochs}")

        # Training
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)

        num_batches = 0
        for step, batch in enumerate(train_loader):
            try:
                result = train_step(model, batch, loss_fn, optimizer, args, device)

                # Accumulate results
                for k, v in result['losses'].items():
                    epoch_losses[k].append(v)
                for k, v in result['metrics'].items():
                    epoch_metrics[k].append(v)

                num_batches += 1

                # Log progress
                if (step + 1) % 10 == 0:
                    print(f"  Step {step + 1}/{len(train_loader)} | "
                          f"Loss: {result['total_loss']:.4f} | "
                          f"3D Error: {result['metrics'].get('iris_mesh_3d_l2_error', 0):.4f}")

            except Exception as e:
                print(f"Error in training step {step}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if num_batches == 0:
            print(f"WARNING: No successful training steps in epoch {epoch}")
            continue

        print(f"Completed {num_batches} training steps")

        # Validation
        val_losses, val_metrics = {}, {}
        if val_loader:
            print("Running validation...")
            val_losses, val_metrics = validate_step(model, val_loader, loss_fn, args, device)
            print(f"Validation - Total Loss: {val_losses['total']:.4f} | "
                  f"3D Error: {val_metrics.get('iris_mesh_3d_l2_error', 0):.4f}")

        # Update learning rate (after optimizer.step())
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        # Log to CSV
        log_dict = {
            "epoch": epoch + 1,
            "step": len(train_loader),
            "batch_size": args.batch_size,
            "lr": optimizer.param_groups[0]['lr'],
            "phase": phase,
            "total_loss": np.mean(epoch_losses['total']),
            "head_pose_accuracy": np.mean(epoch_losses.get('head_pose_accuracy', [0])),
            "head_pose_consistency": np.mean(epoch_losses.get('head_pose_consistency', [0])),
            "iris_reconstruction_3d": np.mean(epoch_losses.get('iris_reconstruction_3d', [0])),
            "iris_reconstruction_2d": np.mean(epoch_losses.get('iris_reconstruction_2d', [0])),
            "iris_projection_consistency": np.mean(epoch_losses.get('iris_projection_consistency', [0])),
            "iris_spherical": np.mean(epoch_losses.get('iris_spherical', [0])),
            "iris_circular": np.mean(epoch_losses.get('iris_circular', [0])),
            "iris_smoothing": np.mean(epoch_losses.get('iris_smoothing', [0])),
            "iris_edge_consistency": np.mean(epoch_losses.get('iris_edge_consistency', [0])),
            "iris_mesh_3d_l2_error": np.mean(epoch_metrics.get('iris_mesh_3d_l2_error', [0])),
            "iris_mesh_2d_pixel_error": np.mean(epoch_metrics.get('iris_mesh_2d_pixel_error', [0])),
            "projection_consistency_error": np.mean(epoch_metrics.get('projection_consistency_error', [0])),
            "val_total_loss": val_losses.get('total', 0),
            "val_iris_mesh_3d_l2_error": val_metrics.get('iris_mesh_3d_l2_error', 0),
        }
        csv_writer.writerow(log_dict)
        logfile.flush()

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"enhanced_raynet_epoch{epoch + 1}.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'args': vars(args),
                'iris_loss_config': iris_loss_config
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    logfile.close()
    print("Training complete!")


if __name__ == "__main__":
    main()