import argparse
import os
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from datetime import datetime

# Import the new model and loss
from EyeFLAME.loss import WeakPerspectiveLoss
from dataset import GazeGeneDataset, EnhancedMultiViewBatchSampler, gazegene_collate_fn
from raynet import create_raynet_model_with_depth_aware


class DynamicTrainingStrategy:
    """
    Dynamic training strategy with automatic phase transitions
    based on loss thresholds and convergence metrics
    """

    def __init__(self):
        self.phase = 1  # Start with phase 1
        self.phase_epochs = 0  # Epochs in current phase
        self.best_2d_loss = float('inf')
        self.best_3d_loss = float('inf')
        self.loss_history = []

        # Phase transition thresholds
        self.thresholds = {
            'phase1_to_2': {
                '2d_loss': 100.0,  # pixels - when 2D projection is reasonably good
                'min_epochs': 3,  # Minimum epochs in phase 1
                'convergence_rate': 0.05  # Loss should be decreasing
            },
            'phase2_to_3': {
                '2d_loss': 50.0,  # pixels - when 2D projection is quite good
                '3d_loss': 50.0,  # cm - when 3D is starting to converge
                'min_epochs': 5,  # Minimum epochs in phase 2
                'convergence_rate': 0.02
            }
        }

        # Loss weights for each phase
        self.phase_weights = {
            1: {'2d': 1.0, '3d': 0.01, 'angular': 1.0, 'reg': 0.01},
            2: {'2d': 1.0, '3d': 0.1, 'angular': 1.0, 'reg': 0.01},
            3: {'2d': 0.5, '3d': 0.5, 'angular': 1.0, 'reg': 0.005}
        }

    def should_transition(self, metrics):
        """Check if we should transition to next phase"""
        self.phase_epochs += 1

        # Extract key metrics
        loss_2d = metrics.get('loss_2d_avg', float('inf'))
        loss_3d = metrics.get('loss_3d_avg', float('inf'))

        # Update best losses
        self.best_2d_loss = min(self.best_2d_loss, loss_2d)
        self.best_3d_loss = min(self.best_3d_loss, loss_3d)

        # Check phase 1 -> 2 transition
        if self.phase == 1:
            thresh = self.thresholds['phase1_to_2']
            if (loss_2d < thresh['2d_loss'] and
                    self.phase_epochs >= thresh['min_epochs']):

                # Check convergence rate
                if len(self.loss_history) > 2:
                    recent_improvement = (self.loss_history[-3] - loss_2d) / (self.loss_history[-3] + 1e-8)
                    if recent_improvement > thresh['convergence_rate']:
                        print(f"\n=== PHASE TRANSITION 1->2 ===")
                        print(f"  2D loss: {loss_2d:.2f} < {thresh['2d_loss']}")
                        print(f"  Convergence rate: {recent_improvement:.3f}")
                        self.phase = 2
                        self.phase_epochs = 0
                        return True

        # Check phase 2 -> 3 transition
        elif self.phase == 2:
            thresh = self.thresholds['phase2_to_3']
            if (loss_2d < thresh['2d_loss'] and
                    loss_3d < thresh['3d_loss'] and
                    self.phase_epochs >= thresh['min_epochs']):
                print(f"\n=== PHASE TRANSITION 2->3 ===")
                print(f"  2D loss: {loss_2d:.2f} < {thresh['2d_loss']}")
                print(f"  3D loss: {loss_3d:.2f} < {thresh['3d_loss']}")
                self.phase = 3
                self.phase_epochs = 0
                return True

        self.loss_history.append(loss_2d)
        if len(self.loss_history) > 10:
            self.loss_history.pop(0)

        return False

    def get_current_weights(self):
        """Get current phase weights"""
        return self.phase_weights[self.phase]

    def get_status(self):
        """Get training strategy status"""
        return {
            'phase': self.phase,
            'phase_epochs': self.phase_epochs,
            'weights': self.phase_weights[self.phase],
            'best_2d_loss': self.best_2d_loss,
            'best_3d_loss': self.best_3d_loss
        }


def parse_args():
    parser = argparse.ArgumentParser(description="EyeFLAME Training with Depth-Aware Model")
    parser.add_argument('--base_dir', type=str, required=True, help="Root of GazeGene dataset")
    parser.add_argument('--backbone_name', type=str, default="repnext_m3")
    parser.add_argument('--weight_path', type=str, default="./repnext_m3_pretrained.pt")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--samples_per_subject', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints_depth_aware")
    parser.add_argument('--log_csv', type=str, default="train_log_depth_aware.csv")
    parser.add_argument('--checkpoint_freq', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--validate_freq', type=int, default=1, help="Validation frequency (epochs)")
    parser.add_argument('--log_freq', type=int, default=50, help="Logging frequency (steps)")
    parser.add_argument('--debug', action='store_true', help="Debug mode with verbose output")
    return parser.parse_args()


def create_datasets_and_loaders(args):
    """Create datasets ensuring 2D annotations are included"""

    # Use official train/val split
    train_subjects = [f"subject{i}" for i in range(1, 47)]  # Subjects 1-46
    val_subjects = [f"subject{i}" for i in range(47, 57)]  # Subjects 47-56

    print(f"Loading training subjects: {len(train_subjects)} subjects")
    print(f"Loading validation subjects: {len(val_subjects)} subjects")

    # Training dataset - MUST include 2D landmarks
    train_dataset = GazeGeneDataset(
        base_dir=args.base_dir,
        subject_ids=train_subjects,
        samples_per_subject=args.samples_per_subject,  # Use all samples
        include_2d_landmarks=True,  # CRITICAL for depth-aware model
        include_camera_params=True,
        balance_attributes=['ethicity']
    )

    # Multi-view batch sampler
    train_sampler = EnhancedMultiViewBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        balance_attributes=['ethicity'],
        ensure_multiview=True,  # Ensure 9 views per sample
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=gazegene_collate_fn
    )

    # Validation dataset
    val_dataset = GazeGeneDataset(
        base_dir=args.base_dir,
        subject_ids=val_subjects,
        samples_per_subject=100,  # Fewer samples for faster validation
        include_2d_landmarks=True,
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
        collate_fn=gazegene_collate_fn
    )

    print(f"Dataset statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Verify 2D annotations are present
    sample = train_dataset[0]
    if 'mesh_2d' not in sample or sample['mesh_2d'] is None:
        raise ValueError("CRITICAL: Dataset does not contain 2D annotations! Cannot train depth-aware model.")

    return train_loader, val_loader


def fix_parameter_expansion(val, batch_size, param_name):
    """
    Properly expand parameters to batch size
    """
    if not isinstance(val, torch.Tensor):
        if isinstance(val, (int, float)):
            val = torch.tensor([val], dtype=torch.float32)
        else:
            val = torch.tensor(val, dtype=torch.float32)

    # Special handling for kappa (2D -> 3D)
    if param_name in ['L_kappa', 'R_kappa']:
        if val.numel() == 2:  # If it's 2D kappa
            val = torch.cat([val.reshape(-1), torch.zeros(1)])  # Add zero roll
        elif val.shape[-1] == 2:  # [1, 2] or [2]
            val = torch.cat([val.reshape(-1)[:2], torch.zeros(1)])

    # Ensure at least 1D
    if val.dim() == 0:
        val = val.unsqueeze(0)

    # Flatten to 1D if needed for expansion
    if val.dim() > 1:
        # Keep the last dimension, flatten the rest
        original_shape = val.shape
        if len(original_shape) > 1:
            val = val.reshape(-1, original_shape[-1])
            if val.shape[0] == 1:
                val = val.squeeze(0)  # Now it's 1D

    # Now expand to batch size
    if val.dim() == 1:
        if val.shape[0] == 1:
            val = val.expand(batch_size)
        elif val.shape[0] == 2 and param_name not in ['eyecenter_L', 'eyecenter_R']:
            # For 2-element params that aren't eye centers
            val = val.unsqueeze(0).expand(batch_size, -1)
        elif val.shape[0] == 3:
            # For 3-element params (like kappa, eye centers)
            val = val.unsqueeze(0).expand(batch_size, -1)
        else:
            # General case
            val = val.unsqueeze(0).expand(batch_size, -1)

    return val


def convert_batch_for_depth_aware_kappa_2d(batch, device):
    """
    Convert batch with CORRECT 2D kappa handling
    Note: In in the original GazeGene document mistakenly states kappa is 3D, but it's actually 2D!
    """
    batch_size = batch['img'].shape[0]

    # Images
    images = batch['img'].to(device)

    # Extract subject parameters
    gazegene_subject_params = {}
    subject_attrs_list = batch.get('subject_attributes', [])

    # Get reference attributes
    reference_attrs = None
    for attrs in subject_attrs_list:
        if attrs is not None and isinstance(attrs, dict):
            reference_attrs = attrs
            break

    if reference_attrs:
        for param_name in ['eyecenter_L', 'eyecenter_R', 'eyeball_radius',
                           'iris_radius', 'cornea_radius', 'cornea2center',
                           'UVRadius', 'L_kappa', 'R_kappa']:

            if param_name in reference_attrs:
                val = reference_attrs[param_name]

                # Convert to tensor
                if isinstance(val, (int, float)):
                    tensor_val = torch.tensor([val], dtype=torch.float32)
                else:
                    tensor_val = torch.tensor(val, dtype=torch.float32)

                # DO NOT expand kappa to 3D - keep it as 2D!
                # Kappa is [horizontal, vertical] only

                # Ensure proper shape
                if tensor_val.dim() == 0:
                    tensor_val = tensor_val.unsqueeze(0)

                # Expand to batch size
                if tensor_val.shape[0] == 1:
                    if tensor_val.dim() == 1:
                        tensor_val = tensor_val.unsqueeze(0).expand(batch_size, -1)
                    else:
                        tensor_val = tensor_val.expand(batch_size, *tensor_val.shape[1:])

                gazegene_subject_params[param_name] = tensor_val.to(device)

    # Camera parameters
    camera_params = {
        'intrinsic_matrix': batch['intrinsic'].to(device)
    }

    # Ground truth
    ground_truth = {}

    # 3D annotations
    if 'mesh' in batch:
        mesh = batch['mesh']
        ground_truth['eyeball_center_3D'] = mesh['eyeball_center_3D'].to(device)
        ground_truth['pupil_center_3D'] = mesh['pupil_center_3D'].to(device)

        # Reshape iris 3D: [B, 2, 100, 3] -> [B, 200, 3]
        iris_3d = mesh['iris_mesh_3D'].to(device)
        if iris_3d.dim() == 4 and iris_3d.shape[1] == 2:
            ground_truth['iris_mesh_3D'] = iris_3d.reshape(batch_size, -1, 3)
        else:
            ground_truth['iris_mesh_3D'] = iris_3d

    # 2D annotations
    if 'mesh_2d' in batch and batch['mesh_2d'] is not None:
        mesh_2d = batch['mesh_2d']

        if 'eyeball_center_2D' in mesh_2d and mesh_2d['eyeball_center_2D'] is not None:
            ground_truth['eyeball_center_2D'] = mesh_2d['eyeball_center_2D'].to(device)

        if 'pupil_center_2D' in mesh_2d and mesh_2d['pupil_center_2D'] is not None:
            ground_truth['pupil_center_2D'] = mesh_2d['pupil_center_2D'].to(device)

        if 'iris_mesh_2D' in mesh_2d and mesh_2d['iris_mesh_2D'] is not None:
            iris_2d = mesh_2d['iris_mesh_2D'].to(device)
            if iris_2d.dim() == 4 and iris_2d.shape[1] == 2:
                ground_truth['iris_mesh_2D'] = iris_2d.reshape(batch_size, -1, 2)
            else:
                ground_truth['iris_mesh_2D'] = iris_2d

    # Gaze annotations
    if 'gaze' in batch:
        gaze = batch['gaze']
        ground_truth['gaze_C'] = gaze['gaze_vector_C'].to(device)
        ground_truth['optic_axis_L'] = gaze['optic_axis_L'].to(device)
        ground_truth['optic_axis_R'] = gaze['optic_axis_R'].to(device)

        if 'visual_axis_L' in gaze:
            ground_truth['visual_axis_L'] = gaze['visual_axis_L'].to(device)
        if 'visual_axis_R' in gaze:
            ground_truth['visual_axis_R'] = gaze['visual_axis_R'].to(device)

    return {
        'images': images,
        'gazegene_subject_params': gazegene_subject_params,
        'camera_params': camera_params,
        'ground_truth': ground_truth
    }


def train_step(model, batch, loss_fn, optimizer, strategy, args, device):
    """
        Single training step with dynamic loss weighting and gradient clipping
    """
    model.train()
    optimizer.zero_grad()

    # Convert batch with fixed parameter expansion
    model_inputs = convert_batch_for_depth_aware_kappa_2d(batch, device)

    # Verify 2D annotations exist
    if 'iris_mesh_2D' not in model_inputs['ground_truth']:
        raise ValueError("2D annotations are required for depth-aware training.")

    # Forward pass through FULL RayNet model (not just EyeFLAME)
    predictions = model(
        model_inputs['images'],  # Raw images go to RayNet
        subject_params=model_inputs['gazegene_subject_params'],
        camera_params=model_inputs['camera_params']
    )

    # Compute losses
    total_loss, individual_losses = loss_fn(
        predictions,
        model_inputs['ground_truth'],
        model_inputs['gazegene_subject_params'],
        model_inputs['camera_params']
    )

    # Apply dynamic weighting based on strategy phase
    weights = strategy.get_current_weights()
    weighted_loss = 0
    weighted_losses = {}

    for loss_name, loss_val in individual_losses.items():
        if '2d' in loss_name:
            weight = weights['2d']
        elif '3d' in loss_name:
            weight = weights['3d']
        elif 'gaze' in loss_name or 'optical' in loss_name:
            weight = weights['angular']
        else:  # Regularization
            weight = weights['reg']

        weighted_losses[loss_name] = weight * loss_val
        weighted_loss += weighted_losses[loss_name]

    # Backward pass with gradient clipping
    weighted_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
    optimizer.step()

    # Compute metrics
    with torch.no_grad():
        metrics = compute_metrics(predictions, model_inputs['ground_truth'])

    # Aggregate losses for strategy
    loss_2d_avg = np.mean([v.item() for k, v in individual_losses.items() if '2d' in k and v > 0])
    loss_3d_avg = np.mean([v.item() for k, v in individual_losses.items() if '3d' in k and v > 0])

    return {
        'total_loss': weighted_loss.item(),
        'losses': {k: v.item() if torch.is_tensor(v) else v for k, v in weighted_losses.items()},
        'raw_losses': {k: v.item() if torch.is_tensor(v) else v for k, v in individual_losses.items()},
        'metrics': metrics,
        'loss_2d_avg': loss_2d_avg if not np.isnan(loss_2d_avg) else 0,
        'loss_3d_avg': loss_3d_avg if not np.isnan(loss_3d_avg) else 0,
        'strategy_phase': strategy.phase
    }


def compute_metrics(predictions, ground_truth):
    """
    Compute evaluation metrics with error handling
    """
    metrics = {}

    # 3D metrics (in cm)
    if 'eyeball_centers' in predictions and 'eyeball_center_3D' in ground_truth:
        if predictions['eyeball_centers'].shape == ground_truth['eyeball_center_3D'].shape:
            error = torch.norm(predictions['eyeball_centers'] - ground_truth['eyeball_center_3D'], dim=-1)
            metrics['eyeball_error_cm'] = torch.mean(error).item()

    if 'pupil_centers' in predictions and 'pupil_center_3D' in ground_truth:
        if predictions['pupil_centers'].shape == ground_truth['pupil_center_3D'].shape:
            error = torch.norm(predictions['pupil_centers'] - ground_truth['pupil_center_3D'], dim=-1)
            metrics['pupil_error_cm'] = torch.mean(error).item()

    if 'iris_landmarks_100' in predictions and 'iris_mesh_3D' in ground_truth:
        if predictions['iris_landmarks_100'].shape == ground_truth['iris_mesh_3D'].shape:
            error = torch.norm(predictions['iris_landmarks_100'] - ground_truth['iris_mesh_3D'], dim=-1)
            metrics['iris_error_cm'] = torch.mean(error).item()

    # Angular metrics (in degrees)
    if 'head_gaze_direction' in predictions and 'gaze_C' in ground_truth:
        if predictions['head_gaze_direction'].shape == ground_truth['gaze_C'].shape:
            cosine_sim = torch.sum(predictions['head_gaze_direction'] * ground_truth['gaze_C'], dim=-1)
            cosine_sim = torch.clamp(cosine_sim, -1 + 1e-7, 1 - 1e-7)
            angle_error = torch.acos(cosine_sim)
            metrics['gaze_angle_error_deg'] = torch.mean(angle_error).item() * 180 / np.pi

    # 2D metrics (in pixels)
    if 'projections_2d' in predictions and predictions['projections_2d'] is not None:
        if 'iris_landmarks_2d' in predictions['projections_2d'] and 'iris_mesh_2D' in ground_truth:
            pred_2d = predictions['projections_2d']['iris_landmarks_2d']
            gt_2d = ground_truth['iris_mesh_2D']
            if pred_2d.shape == gt_2d.shape:
                error = torch.norm(pred_2d - gt_2d, dim=-1)
                metrics['iris_2d_error_px'] = torch.mean(error).item()

    # Weak perspective metrics
    if 'weak_perspective' in predictions:
        wp = predictions['weak_perspective']
        metrics['depth_scale'] = wp['scale'].mean().item()
        metrics['translation_magnitude'] = torch.norm(wp['translation_2d'], dim=-1).mean().item()

    # Add defaults for missing metrics
    default_metrics = {
        'eyeball_error_cm': 0,
        'pupil_error_cm': 0,
        'iris_error_cm': 0,
        'gaze_angle_error_deg': 0,
        'iris_2d_error_px': 0,
        'depth_scale': 1.0,
        'translation_magnitude': 0
    }

    for key, default_val in default_metrics.items():
        if key not in metrics:
            metrics[key] = default_val

    return metrics


def validation_step(model, val_loader, loss_fn, strategy, args, device):
    """Validation step"""

    model.eval()
    val_losses = defaultdict(list)
    val_metrics = defaultdict(list)
    val_raw_losses = defaultdict(list)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Convert batch
            model_inputs = convert_batch_for_depth_aware_kappa_2d(batch, device)
            # Verify 2D annotations exist
            if 'iris_mesh_2D' not in model_inputs['ground_truth']:
                print("Warning: Missing 2D annotations in batch!")
                # Could skip this batch or use dummy 2D loss

            # Forward pass
            predictions = model(
                model_inputs['images'],
                subject_params=model_inputs['gazegene_subject_params'],
                camera_params=model_inputs['camera_params']
            )

            # Compute losses
            total_loss, individual_losses = loss_fn(
                predictions,
                model_inputs['ground_truth'],
                model_inputs['gazegene_subject_params'],
                model_inputs['camera_params']
            )

            # Apply strategy weights
            weights = strategy.get_current_weights()
            weighted_loss = 0

            for loss_name, loss_val in individual_losses.items():
                if '2d' in loss_name:
                    weight = weights['2d']
                elif '3d' in loss_name:
                    weight = weights['3d']
                elif 'gaze' in loss_name or 'optical' in loss_name:
                    weight = weights['angular']
                else:
                    weight = weights['reg']

                weighted_loss += weight * loss_val
                val_losses[loss_name].append(loss_val.item())
                val_raw_losses[f"raw_{loss_name}"].append(loss_val.item())

            val_losses['total'].append(weighted_loss.item())

            # Compute metrics
            metrics = compute_metrics(predictions, model_inputs['ground_truth'])
            for k, v in metrics.items():
                val_metrics[k].append(v)

    # Average results
    avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
    avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
    avg_raw_losses = {k: np.mean(v) for k, v in val_raw_losses.items()}

    # Compute average 2D and 3D losses for strategy
    loss_2d_avg = np.mean([v for k, v in avg_raw_losses.items() if '2d' in k])
    loss_3d_avg = np.mean([v for k, v in avg_raw_losses.items() if '3d' in k])

    return {
        'avg_losses': avg_losses,
        'avg_metrics': avg_metrics,
        'avg_raw_losses': avg_raw_losses,
        'loss_2d_avg': loss_2d_avg,
        'loss_3d_avg': loss_3d_avg
    }


def save_checkpoint(model, optimizer, scheduler, epoch, strategy, metrics, checkpoint_path):
    """Save training checkpoint"""

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'strategy': {
            'phase': strategy.phase,
            'phase_epochs': strategy.phase_epochs,
            'best_2d_loss': strategy.best_2d_loss,
            'best_3d_loss': strategy.best_3d_loss
        },
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, strategy=None):
    """Load training checkpoint"""

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if strategy and 'strategy' in checkpoint:
        strategy.phase = checkpoint['strategy']['phase']
        strategy.phase_epochs = checkpoint['strategy']['phase_epochs']
        strategy.best_2d_loss = checkpoint['strategy']['best_2d_loss']
        strategy.best_3d_loss = checkpoint['strategy']['best_3d_loss']

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Strategy phase: {checkpoint['strategy']['phase']}")

    return checkpoint['epoch']


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize CSV logger
    csv_file = open(args.log_csv, 'w', newline='')
    csv_fields = ['epoch', 'phase', 'train_loss', 'val_loss', 'lr',
                  'loss_2d_avg', 'loss_3d_avg',
                  'eyeball_error_cm', 'pupil_error_cm', 'iris_error_cm',
                  'gaze_angle_error_deg', 'optical_axis_error_deg', 'iris_2d_error_px']
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Create datasets
    print("Creating datasets...")
    train_loader, val_loader = create_datasets_and_loaders(args)

    # Create model
    print("Creating depth-aware model...")
    model = create_raynet_model_with_depth_aware(
        backbone_name=args.backbone_name,
        weight_path=args.weight_path,
    )
    model = model.to(device)

    # Create loss function
    loss_fn = WeakPerspectiveLoss().to(device)

    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Initialize dynamic strategy
    strategy = DynamicTrainingStrategy()

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, strategy)

    # Training loop
    print("\n=== Starting Training ===")
    print(f"Total epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Steps per epoch: {len(train_loader)}")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.epochs} - Phase {strategy.phase}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Current weights: {strategy.get_current_weights()}")

        # Training
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)

        for step, batch in enumerate(train_loader):
            try:

                train_results = train_step(
                    model, batch, loss_fn, optimizer, strategy, args, device
                )
                # Accumulate results
                epoch_losses['total'].append(train_results['total_loss'])
                epoch_losses['2d_avg'].append(train_results['loss_2d_avg'])
                epoch_losses['3d_avg'].append(train_results['loss_3d_avg'])

                for k, v in train_results['metrics'].items():
                    epoch_metrics[k].append(v)

                # Logging
                if (step + 1) % args.log_freq == 0:
                    print(f"  Step {step + 1}/{len(train_loader)}")
                    print(f"    Total loss: {train_results['total_loss']:.4f}")
                    print(f"    2D loss avg: {train_results['loss_2d_avg']:.2f} px")
                    print(f"    3D loss avg: {train_results['loss_3d_avg']:.2f} cm")

                    if args.debug:
                        print(f"    Raw losses: {train_results['raw_losses']}")
                        print(f"    Metrics: {train_results['metrics']}")

            except Exception as e:
                print(f"Error in training step {step}: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                continue

        # Compute epoch averages
        avg_train_loss = np.mean(epoch_losses['total'])
        avg_2d_loss = np.mean(epoch_losses['2d_avg'])
        avg_3d_loss = np.mean(epoch_losses['3d_avg'])

        print(f"\nTraining Summary:")
        print(f"  Average loss: {avg_train_loss:.4f}")
        print(f"  Average 2D loss: {avg_2d_loss:.2f} px")
        print(f"  Average 3D loss: {avg_3d_loss:.2f} cm")

        # Validation
        val_results = None
        if (epoch + 1) % args.validate_freq == 0:
            print("\nRunning validation...")
            val_results = validation_step(
                model, val_loader, loss_fn, strategy, args, device
            )

            print(f"Validation Summary:")
            print(f"  Average loss: {val_results['avg_losses']['total']:.4f}")
            print(f"  Average 2D loss: {val_results['loss_2d_avg']:.2f} px")
            print(f"  Average 3D loss: {val_results['loss_3d_avg']:.2f} cm")
            print(f"  Metrics: {val_results['avg_metrics']}")

            # Check for best model
            if val_results['avg_losses']['total'] < best_val_loss:
                best_val_loss = val_results['avg_losses']['total']
                best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                save_checkpoint(
                    model, optimizer, scheduler, epoch, strategy,
                    val_results['avg_metrics'], best_checkpoint_path
                )
                print(f"  New best model saved!")

        # Check phase transition
        transition_metrics = {
            'loss_2d_avg': avg_2d_loss,
            'loss_3d_avg': avg_3d_loss
        }

        if strategy.should_transition(transition_metrics):
            print(f"\n{'*' * 50}")
            print(f"* TRANSITIONING TO PHASE {strategy.phase} *")
            print(f"{'*' * 50}\n")

        # Update learning rate
        scheduler.step()

        # Log to CSV
        csv_row = {
            'epoch': epoch + 1,
            'phase': strategy.phase,
            'train_loss': avg_train_loss,
            'val_loss': val_results['avg_losses']['total'] if val_results else 0,
            'lr': optimizer.param_groups[0]['lr'],
            'loss_2d_avg': avg_2d_loss,
            'loss_3d_avg': avg_3d_loss
        }

        # Add metrics if validation was performed
        if val_results:
            for metric_name in ['eyeball_error_cm', 'pupil_error_cm', 'iris_error_cm',
                                'gaze_angle_error_deg', 'optical_axis_error_deg', 'iris_2d_error_px']:
                csv_row[metric_name] = val_results['avg_metrics'].get(metric_name, 0)

        csv_writer.writerow(csv_row)
        csv_file.flush()

        # Save checkpoint periodically
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"checkpoint_epoch{epoch + 1}_phase{strategy.phase}.pth"
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch, strategy,
                val_results['avg_metrics'] if val_results else epoch_metrics,
                checkpoint_path
            )

    # Save final model
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, strategy,
        val_results['avg_metrics'] if val_results else epoch_metrics,
        final_checkpoint_path
    )

    csv_file.close()
    print("\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final strategy phase: {strategy.phase}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Training log saved to: {args.log_csv}")


if __name__ == "__main__":
    main()