import argparse
import os
import csv
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Import the eye-focused RayNet and losses
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from raynet import RayNet  # Eye-focused version
from dataset import GazeGeneDataset, MultiViewBatchSampler
from eye_losses import CombinedRayNetLoss

import matplotlib

# Set backend before importing pyplot
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Eye-Focused RayNet Training")
    parser.add_argument('--base_dir', type=str, required=True, help="Root of GazeGene dataset")
    parser.add_argument('--backbone_name', type=str, default="repnext_m3")
    parser.add_argument('--weight_path', type=str, default="./repnext_m3_pretrained.pt")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--samples_per_subject', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints_eye")
    parser.add_argument('--log_csv', type=str, default="train_eye_log.csv")
    parser.add_argument('--checkpoint_freq', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--plot_live', action="store_true")
    parser.add_argument('--visualize_freq', type=int, default=100, help="Frequency of 3D visualization")
    return parser.parse_args()


def get_backbone(backbone_name, weight_path, device):
    from backbone.repnext_utils import load_pretrained_repnext
    model = load_pretrained_repnext(backbone_name, weight_path)
    return model.to(device)


def visualize_eye_reconstruction(model_output, gt_data, save_path=None):
    """
    Visualize 3D eye reconstruction including eyeball, iris, and gaze rays
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(20, 5))

    # Take first sample from batch
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x[0].detach().cpu().numpy()
        return x

    # Plot 1: Left eye reconstruction
    ax1 = fig.add_subplot(141, projection='3d')

    # Eyeball mesh (simplified visualization)
    if 'eyeball_left' in model_output:
        eyeball = model_output['eyeball_left']
        center = to_numpy(eyeball['center'])
        radius = to_numpy(eyeball['radius'])

        # Draw sphere for eyeball
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        ax1.plot_surface(x, y, z, alpha=0.3, color='pink')

    # Iris landmarks
    if 'iris_landmarks' in model_output:
        iris_left = to_numpy(model_output['iris_landmarks']['left'])
        ax1.scatter(iris_left[:, 0], iris_left[:, 1], iris_left[:, 2],
                    c='blue', s=10, label='Predicted Iris')

    # Ground truth iris
    if 'mesh' in gt_data:
        gt_iris = to_numpy(gt_data['mesh']['iris_mesh_3D'])
        if len(gt_iris.shape) > 2:
            gt_iris_left = gt_iris[0] if gt_iris.shape[0] == 2 else gt_iris[:, 0]
            ax1.scatter(gt_iris_left[:, 0], gt_iris_left[:, 1], gt_iris_left[:, 2],
                        c='green', s=5, alpha=0.5, label='GT Iris')

    # Pupil center
    if 'pupil_centers' in model_output:
        pupil = to_numpy(model_output['pupil_centers']['left'])
        ax1.scatter(pupil[0], pupil[1], pupil[2], c='red', s=50, marker='*', label='Pupil')

    # Gaze ray
    if 'eyeball_center_left' in model_output and 'visual_axis_left' in model_output:
        origin = to_numpy(model_output['eyeball_center_left'])
        direction = to_numpy(model_output['visual_axis_left'])
        ax1.quiver(origin[0], origin[1], origin[2],
                   direction[0] * 20, direction[1] * 20, direction[2] * 20,
                   color='red', arrow_length_ratio=0.1, linewidth=2, label='Visual Axis')

    ax1.set_title('Left Eye')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.legend()
    ax1.set_box_aspect([1, 1, 1])

    # Plot 2: Right eye reconstruction
    ax2 = fig.add_subplot(142, projection='3d')

    # Similar visualization for right eye
    if 'eyeball_right' in model_output:
        eyeball = model_output['eyeball_right']
        center = to_numpy(eyeball['center'])
        radius = to_numpy(eyeball['radius'])

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        ax2.plot_surface(x, y, z, alpha=0.3, color='pink')

    if 'iris_landmarks' in model_output:
        iris_right = to_numpy(model_output['iris_landmarks']['right'])
        ax2.scatter(iris_right[:, 0], iris_right[:, 1], iris_right[:, 2],
                    c='blue', s=10, label='Predicted Iris')

    ax2.set_title('Right Eye')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    ax2.set_box_aspect([1, 1, 1])

    # Plot 3: Combined gaze visualization
    ax3 = fig.add_subplot(143, projection='3d')

    # Both eyes and combined gaze
    if 'ray_origin' in model_output and 'ray_direction' in model_output:
        origin = to_numpy(model_output['ray_origin'])
        direction = to_numpy(model_output['ray_direction'])

        # Draw ray
        t = np.linspace(0, 500, 100)
        ray_points = origin[:, np.newaxis] + direction[:, np.newaxis] * t
        ax3.plot(ray_points[0], ray_points[1], ray_points[2], 'r-', linewidth=2, label='Gaze Ray')

        # Mark origin
        ax3.scatter(origin[0], origin[1], origin[2], c='red', s=100, marker='o')

        # Gaze point
        if 'gaze_point_3d' in model_output:
            gaze_pt = to_numpy(model_output['gaze_point_3d'])
            ax3.scatter(gaze_pt[0], gaze_pt[1], gaze_pt[2], c='green', s=100, marker='*', label='Gaze Point')

    ax3.set_title('Combined Gaze Ray')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Z (mm)')
    ax3.legend()

    # Plot 4: Angular error visualization
    ax4 = fig.add_subplot(144)

    # Show angular errors if available
    if hasattr(ax4, 'text_data'):
        text_data = ax4.text_data
    else:
        text_data = []

    ax4.axis('off')
    y_pos = 0.9
    for line in text_data:
        ax4.text(0.1, y_pos, line, fontsize=10, transform=ax4.transAxes)
        y_pos -= 0.05

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def main():
    args = parse_args()

    # Set up device and seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset split
    if args.split == "train":
        subject_ids = [f"subject{i}" for i in range(1, 46)]
    else:
        subject_ids = [f"subject{i}" for i in range(46, 56)]

    # Create dataset and dataloader
    dataset = GazeGeneDataset(
        args.base_dir,
        subject_ids=subject_ids,
        samples_per_subject=args.samples_per_subject,
        transform=None,
        balance_attributes=['ethicity']
    )

    batch_sampler = MultiViewBatchSampler(
        dataset,
        batch_size=args.batch_size,
        balance_attributes=['ethicity'],
        shuffle=True
    )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model setup
    backbone_channels_dict = {
        'repnext_m0': [40, 80, 160, 320],
        'repnext_m1': [48, 96, 192, 384],
        'repnext_m2': [56, 112, 224, 448],
        'repnext_m3': [64, 128, 256, 512],
        'repnext_m4': [64, 128, 256, 512],
        'repnext_m5': [80, 160, 320, 640],
    }

    in_channels_list = backbone_channels_dict[args.backbone_name]
    backbone = get_backbone(args.backbone_name, args.weight_path, device)

    model = RayNet(
        backbone=backbone,
        in_channels_list=in_channels_list,
        n_iris_landmarks=100,
        panet_out_channels=256
    ).to(device)

    # Loss and optimizer
    criterion = CombinedRayNetLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Check for existing checkpoints to resume training
    start_epoch = 0
    checkpoint_files = []
    if os.path.exists(args.checkpoint_dir):
        checkpoint_files = sorted([f for f in os.listdir(args.checkpoint_dir)
                                   if f.endswith('.pth') and 'raynet_eye_epoch' in f])

    if checkpoint_files:
        last_ckpt = os.path.join(args.checkpoint_dir, checkpoint_files[-1])
        print(f"Resuming from checkpoint: {last_ckpt}")
        try:
            checkpoint = torch.load(last_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting from scratch...")

    # Logging setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logfile = open(args.log_csv, "w", newline="")
    csv_writer = csv.DictWriter(logfile, fieldnames=[
        "epoch", "step", "batch_size",
        "eyeball_loss", "iris_loss", "gaze_loss",
        "rotation_loss", "multiview_loss", "total_loss",
        "gaze_angular_error", "lr"
    ])
    csv_writer.writeheader()

    # Live plotting setup
    if args.plot_live:
        plt.ion()
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        loss_history = defaultdict(list)

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for step, batch in enumerate(loader):
            global_step += 1

            # Prepare batch
            images = batch['img'].to(device)
            B = images.shape[0] // 9  # Multi-view batch
            images = images.view(B * 9, images.shape[1], images.shape[2], images.shape[3])

            # Forward pass
            outputs = model(images)

            # Prepare ground truth data for loss computation
            # Move all data to device and reshape for multi-view
            gt_data = {
                'mesh': {
                    'eyeball_center_3D': batch['mesh']['eyeball_center_3D'].to(device),
                    'iris_mesh_3D': batch['mesh']['iris_mesh_3D'].to(device),
                    'pupil_center_3D': batch['mesh']['pupil_center_3D'].to(device)
                },
                'gaze': {
                    'gaze_C': batch['gaze']['gaze_C'].to(device),
                    'optic_axis_L': batch['gaze']['optic_axis_L'].to(device),
                    'optic_axis_R': batch['gaze']['optic_axis_R'].to(device),
                    'visual_axis_L': batch['gaze']['visual_axis_L'].to(device),
                    'visual_axis_R': batch['gaze']['visual_axis_R'].to(device),
                    'gaze_depth': batch['gaze']['gaze_depth'].to(device)
                },
                'gaze_point': batch['gaze_point'].to(device),
                'head_pose': {
                    'R': batch['head_pose']['R'].to(device),
                    't': batch['head_pose']['t'].to(device)
                }
            }

            # Compute loss
            losses = criterion(outputs, gt_data, is_multiview=True)
            total_loss = losses['total']

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update epoch losses
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    epoch_losses[k] += v.item()
            n_batches += 1

            # Logging
            log_dict = {
                "epoch": epoch,
                "step": step,
                "batch_size": args.batch_size,
                "eyeball_loss": losses.get('eyeball_total', 0),
                "iris_loss": losses.get('iris_total', 0),
                "gaze_loss": losses.get('gaze_total', 0),
                "rotation_loss": losses.get('rotation_total', 0),
                "multiview_loss": losses.get('multiview_total', 0),
                "total_loss": total_loss.item(),
                "gaze_angular_error": losses.get('gaze_mean_angular_error', 0),
                "lr": optimizer.param_groups[0]['lr']
            }
            csv_writer.writerow(log_dict)
            logfile.flush()

            # Live plotting
            if args.plot_live and step % 10 == 0:
                for key in ['eyeball_loss', 'iris_loss', 'gaze_loss', 'total_loss']:
                    if key in log_dict:
                        loss_history[key].append(log_dict[key])

                # Update plots
                for idx, (ax, key) in enumerate(zip(axes.flat, loss_history.keys())):
                    ax.clear()
                    ax.plot(loss_history[key])
                    ax.set_title(key)
                    ax.set_xlabel('Step')
                    ax.set_ylabel('Loss')
                    ax.grid(True, alpha=0.3)

                plt.suptitle(f'Epoch {epoch}, Step {step}')
                plt.tight_layout()
                plt.pause(0.01)

            # 3D Visualization
            if args.visualize_freq > 0 and global_step % args.visualize_freq == 0:
                with torch.no_grad():
                    vis_path = os.path.join(args.checkpoint_dir, f'vis_epoch{epoch}_step{step}.png')
                    visualize_eye_reconstruction(outputs, batch, save_path=vis_path)
                    print(f"Saved visualization to {vis_path}")

            # Print progress
            if step % 10 == 0:
                gaze_error = losses.get('gaze_mean_angular_error', 0)
                if isinstance(gaze_error, torch.Tensor):
                    gaze_error = gaze_error.item()

                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(loader)}] "
                      f"Loss: {total_loss.item():.4f} "
                      f"Gaze Error: {gaze_error:.2f}° "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Epoch summary
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Total Loss: {avg_losses.get('total', 0):.4f}")
        print(f"  Average Gaze Angular Error: {avg_losses.get('gaze_mean_angular_error', 0):.2f}°")

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"raynet_eye_epoch{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'losses': avg_losses,
                'args': vars(args)
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    # Cleanup
    logfile.close()
    if args.plot_live:
        plt.ioff()
        plt.show()

    print("Training complete!")


def evaluate_model(model, dataloader, device):
    """
    Evaluate model performance
    """
    model.eval()
    criterion = CombinedRayNetLoss()

    total_losses = defaultdict(float)
    angular_errors = []
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['img'].to(device)
            B = images.shape[0] // 9
            images = images.view(B * 9, images.shape[1], images.shape[2], images.shape[3])

            outputs = model(images)
            losses = criterion(outputs, batch, is_multiview=True)

            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    total_losses[k] += v.item()

            # Collect angular errors
            if 'gaze_mean_angular_error' in losses:
                angular_errors.append(losses['gaze_mean_angular_error'].item())

            n_batches += 1

    # Compute averages
    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    mean_angular_error = np.mean(angular_errors) if angular_errors else 0
    std_angular_error = np.std(angular_errors) if angular_errors else 0

    print("\nEvaluation Results:")
    print(f"  Average Total Loss: {avg_losses.get('total', 0):.4f}")
    print(f"  Mean Gaze Angular Error: {mean_angular_error:.2f}° ± {std_angular_error:.2f}°")
    print(f"  Eyeball Loss: {avg_losses.get('eyeball_total', 0):.4f}")
    print(f"  Iris Loss: {avg_losses.get('iris_total', 0):.4f}")
    print(f"  Gaze Loss: {avg_losses.get('gaze_total', 0):.4f}")

    return avg_losses, mean_angular_error


if __name__ == "__main__":
    main()