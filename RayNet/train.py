import argparse
import os
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from EyeFLAME.loss import EyeFLAMELoss
from dataset import GazeGeneDataset, EnhancedMultiViewBatchSampler, convert_dataset_to_model_format
from raynet import create_raynet_model

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
    parser.add_argument('--include_2d_landmarks', action="store_true", default=True,
                        help="Include 2D landmark supervision")
    parser.add_argument('--image_size', type=int, nargs=2, default=[448, 448],
                        help="Input image size (H W)")
    return parser.parse_args()


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
        pin_memory=True,
        collate_fn=gazegene_collate_fn  # Use custom collate function
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
            pin_memory=True,
            collate_fn=gazegene_collate_fn  # Use custom collate function
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


def train_step(model, batch, loss_fn, optimizer, args, device):
    """Single training step."""
    model.train()
    optimizer.zero_grad()
    # Convert batch to model format
    model_inputs = convert_dataset_to_model_format(batch, device)
    # Move to device
    images = model_inputs['images']
    gazegene_subject_params = {k: v.to(device) for k, v in model_inputs['gazegene_subject_params'].items()}
    camera_params = {k: v.to(device) for k, v in model_inputs['camera_params'].items()} if model_inputs[
        'camera_params'] else None
    ground_truth = {k: v.to(device) for k, v in model_inputs['ground_truth'].items()}

    subject_attrs_list = batch.get('subject_attributes', [])

    # # Extract data from batch
    # images = batch['img'].to(device)  # [B*9, 3, H, W]
    #
    # # Get batch size (accounting for 9 views per sample)
    # total_batch_size = images.shape[0]
    # B = total_batch_size // 9
    #
    # # Reshape for multi-view processing
    # images = images.view(B * 9, *images.shape[1:])

    # Subject specific parameters
    """
        subject_params =
        {
            'ID': int,                     # Subject ID from 1 to 56
            'gender': str,                 # ['F', 'M'] refers to female and male
            'ethicity': str,               # ['B', 'Y', 'W'] refers to Black, Yellow and White
            'eyecenter_L': np.array(3,)    # Left eyeball center coordinates under HCS
            'eyecenter_R': np.array(3,)    # Right eyeball center coordinates under HCS
            'eyeball_radius': float,       # Eyeball radius
            'iris_radius': float,          # Iris radius
            'cornea_radius': float,        # Cornea radius
            'cornea2center': float,        # Distance from cornea center to eyeball center
            'UVRadius': float,             # Normalized relative pupil size
            'L_kappa': np.array(3,),       # Euler angles of left eye kappa
            'R_kappa': np.array(3,)        # Euler angles of right eye kappa
        }
    """
    # subject_params = batch['subject_attributes'].to(device)
    # subject_params= subject_params.view(B * 9, *subject_params.shape[1:])
    #
    # # Camera intrinsics parameters 3x3 matrix for each view
    # camera_params = batch['intrinsic'].to(device)  # [B*9, 3, 3]
    # camera_params = camera_params.view(B * 9, *camera_params.shape[1:])

    predictions = model(
        images,
        subject_params=gazegene_subject_params,
        camera_params=camera_params)

    # Compute loss
    total_loss, individual_losses = loss_fn(predictions, ground_truth, gazegene_subject_params, camera_params)

    # Backward pass
    total_loss.backward()

    optimizer.step()

    # Compute metrics
    with torch.no_grad():
        metrics = compute_metrics(predictions, ground_truth)

    return {
        'total_loss': total_loss.item(),
        'losses': {k: v.item() if torch.is_tensor(v) else v for k, v in individual_losses.items()},
        'metrics': {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
    }


def validation_step(model, val_loader, loss_fn, args, device):
    """Validation step."""
    model.eval()
    val_losses = defaultdict(list)
    val_metrics = defaultdict(list)

    with torch.no_grad():
        for batch in val_loader:
            # Convert batch to model format (same as train_step)
            model_inputs = convert_dataset_to_model_format(batch, device)

            # Move to device (same as train_step)
            images = model_inputs['images'].to(device)
            gazegene_subject_params = {k: v.to(device) for k, v in model_inputs['gazegene_subject_params'].items()}
            camera_params = {k: v.to(device) for k, v in model_inputs['camera_params'].items()} if model_inputs[
                'camera_params'] else None
            ground_truth = {k: v.to(device) for k, v in model_inputs['ground_truth'].items()}

            # Model prediction (same as train_step)
            predictions = model(
                images,
                subject_params=gazegene_subject_params,
                camera_params=camera_params
            )

            # Compute loss (same as train_step)
            total_loss, individual_losses = loss_fn(predictions, ground_truth, gazegene_subject_params, camera_params)

            # Compute metrics (same as train_step)
            metrics = compute_metrics(predictions, ground_truth)

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


def compute_metrics(predictions, ground_truth):
    """
    Compute evaluation metrics

    Args:
        predictions: Model predictions
        ground_truth: Ground truth data

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # 3D reconstruction errors (in cm)
    if 'eyeball_centers' in predictions:
        eyeball_error = torch.mean(torch.norm(
            predictions['eyeball_centers'] - ground_truth['eyeball_center_3D'], dim=-1
        ))
        metrics['eyeball_error_cm'] = eyeball_error.item()

    if 'pupil_centers' in predictions:
        pupil_error = torch.mean(torch.norm(
            predictions['pupil_centers'] - ground_truth['pupil_center_3D'], dim=-1
        ))
        metrics['pupil_error_cm'] = pupil_error.item()

    if 'iris_landmarks_100' in predictions:
        iris_error = torch.mean(torch.norm(
            predictions['iris_landmarks_100'] - ground_truth['iris_mesh_3D'], dim=-1
        ))
        metrics['iris_error_cm'] = iris_error.item()

    # Angular errors (in degrees)
    if 'head_gaze_direction' in predictions:
        gaze_cosine_sim = torch.sum(
            predictions['head_gaze_direction'] * ground_truth['gaze_C'], dim=-1
        )
        gaze_angle_error = torch.acos(torch.clamp(gaze_cosine_sim, -1 + 1e-7, 1 - 1e-7))
        metrics['gaze_angle_error_deg'] = torch.mean(gaze_angle_error).item() * 180 / np.pi

    if 'optical_axes' in predictions:
        # Left eye optical axis error
        left_optical_cosine = torch.sum(
            predictions['optical_axes'][:, 0] * ground_truth['optic_axis_L'], dim=-1
        )
        left_optical_error = torch.acos(torch.clamp(left_optical_cosine, -1 + 1e-7, 1 - 1e-7))

        # Right eye optical axis error
        right_optical_cosine = torch.sum(
            predictions['optical_axes'][:, 1] * ground_truth['optic_axis_R'], dim=-1
        )
        right_optical_error = torch.acos(torch.clamp(right_optical_cosine, -1 + 1e-7, 1 - 1e-7))

        metrics['optical_axis_error_deg'] = (torch.mean(left_optical_error) + torch.mean(
            right_optical_error)).item() * 90 / np.pi

    # 2D projection errors (in pixels, if available)
    if 'projections_2d' in predictions and 'iris_mesh_2D' in ground_truth:
        iris_2d_error = torch.mean(torch.norm(
            predictions['projections_2d']['iris_landmarks_2d'] - ground_truth['iris_mesh_2D'], dim=-1
        ))
        metrics['iris_2d_error_px'] = iris_2d_error.item()

    return metrics


def gazegene_collate_fn(batch):
    """
    Custom collate function for GazeGene dataset that handles:
    - Multi-view samples (9 cameras per subject)
    - Complex nested dictionaries
    - Mixed data types (tensors, scalars, None values)

    Args:
        batch: List of samples from dataset.__getitem__()

    Returns:
        Collated batch dictionary
    """
    if len(batch) == 0:
        return {}

    # Handle case where some samples might be None or invalid
    valid_batch = [item for item in batch if item is not None]
    if len(valid_batch) == 0:
        return {}

    # Initialize collated batch
    collated = {}

    # Collate simple tensor fields
    tensor_fields = ['img', 'intrinsic']
    for field in tensor_fields:
        if field in valid_batch[0]:
            try:
                collated[field] = torch.stack([item[field] for item in valid_batch])
            except Exception as e:
                print(f"Error collating {field}: {e}")
                # If stacking fails, keep as list
                collated[field] = [item[field] for item in valid_batch]

    # Collate simple scalar fields
    scalar_fields = ['subject', 'camera', 'frame_idx']
    for field in scalar_fields:
        if field in valid_batch[0]:
            collated[field] = [item[field] for item in valid_batch]

    # Collate nested dictionary fields
    nested_dict_fields = ['mesh', 'gaze', 'head_pose']
    for field in nested_dict_fields:
        if field in valid_batch[0] and valid_batch[0][field] is not None:
            collated[field] = {}

            # Get all keys from the nested dictionary
            all_keys = set()
            for item in valid_batch:
                if item[field] is not None:
                    all_keys.update(item[field].keys())

            # Collate each key in the nested dictionary
            for key in all_keys:
                try:
                    values = []
                    for item in valid_batch:
                        if item[field] is not None and key in item[field]:
                            values.append(item[field][key])
                        else:
                            # Handle missing values by using None or zero tensor
                            values.append(None)

                    # Filter out None values
                    non_none_values = [v for v in values if v is not None]
                    if non_none_values:
                        if torch.is_tensor(non_none_values[0]):
                            # Try to stack tensors
                            try:
                                collated[field][key] = torch.stack(non_none_values)
                            except Exception as e:
                                print(f"Error stacking {field}.{key}: {e}")
                                collated[field][key] = non_none_values
                        else:
                            # Keep as list for non-tensor values
                            collated[field][key] = non_none_values
                    else:
                        collated[field][key] = None

                except Exception as e:
                    print(f"Error collating {field}.{key}: {e}")
                    collated[field][key] = [item[field][key] if item[field] else None for item in valid_batch]

    # Handle optional nested fields that might be None
    optional_nested_fields = ['mesh_2d', 'camera_params']
    for field in optional_nested_fields:
        if field in valid_batch[0]:
            # Check if any item has non-None values for this field
            has_valid_data = any(item[field] is not None for item in valid_batch)

            if has_valid_data:
                collated[field] = {}

                # Get all possible keys
                all_keys = set()
                for item in valid_batch:
                    if item[field] is not None:
                        all_keys.update(item[field].keys())

                # Collate each key
                for key in all_keys:
                    values = []
                    for item in valid_batch:
                        if item[field] is not None and key in item[field]:
                            values.append(item[field][key])
                        else:
                            values.append(None)

                    # Filter and collate non-None values
                    non_none_values = [v for v in values if v is not None]
                    if non_none_values:
                        if torch.is_tensor(non_none_values[0]):
                            try:
                                collated[field][key] = torch.stack(non_none_values)
                            except:
                                collated[field][key] = non_none_values
                        else:
                            collated[field][key] = non_none_values
                    else:
                        collated[field][key] = None
            else:
                collated[field] = None

    # Handle subject_attributes specially (list of dictionaries)
    if 'subject_attributes' in valid_batch[0]:
        collated['subject_attributes'] = [item['subject_attributes'] for item in valid_batch]

    return collated


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
    model = create_raynet_model(args.backbone_name, args.weight_path)

    loss_fn = EyeFLAMELoss(
        use_uncertainty_weighting=True,
        use_scale_weighting=True
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Setup logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logfile = open(args.log_csv, "w", newline="")
    fieldnames = ['epoch', 'train_loss', 'val_loss', 'lr',
                  'eyeball_error_cm', 'pupil_error_cm', 'iris_error_cm',
                  'gaze_angle_error_deg', 'optical_axis_error_deg']
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
        print(f"Epoch {epoch + 1}/{args.epochs}")
        # Training
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)

        num_batches = 0
        for step, batch in enumerate(train_loader):
            try:
                train_results = train_step(model, batch, loss_fn, optimizer, args, device)

                # Accumulate results
                for k, v in train_results['losses'].items():
                    epoch_losses[k].append(v)
                for k, v in train_results['metrics'].items():
                    epoch_metrics[k].append(v)

                num_batches += 1

                # Log progress
                if (step + 1) % 10 == 0:
                    print(f"  Step {step + 1}/{len(train_loader)} | "
                          f"Total loss: {train_results['total_loss']} | "
                          f"Losses: {train_results['losses']} | "
                          f"Metrics Error: {train_results['metrics']}")
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
            val_losses, val_metrics = validation_step(model, val_loader, loss_fn, args, device)
            print(f"Validation - Total Loss: {val_losses['total']:.4f} | "
                  f"3D Error: {val_metrics.get('iris_mesh_3d_l2_error', 0):.4f}")

        # Update learning rate (after optimizer.step())
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        # Log to CSV
        csv_row = [
            epoch,
            epoch_losses.get('total', 0),
            val_losses.get('total', 0),
            optimizer.param_groups[0]['lr'],
            # Training metrics
            epoch_metrics.get('eyeball_error_cm', 0),
            epoch_metrics.get('pupil_error_cm', 0),
            epoch_metrics.get('iris_error_cm', 0),
            epoch_metrics.get('gaze_angle_error_deg', 0),
            epoch_metrics.get('optical_axis_error_deg', 0),
            # Validation metrics
            val_metrics.get('eyeball_error_cm', 0),
            val_metrics.get('pupil_error_cm', 0),
            val_metrics.get('iris_error_cm', 0),
            val_metrics.get('gaze_angle_error_deg', 0),
            val_metrics.get('optical_axis_error_deg', 0)
        ]
        csv_writer.writerow(csv_row)
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
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    logfile.close()
    print("Training complete!")


if __name__ == "__main__":
    main()