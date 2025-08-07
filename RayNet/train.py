# train.py -- with GradNorm loss balancing

import argparse
import os
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from raynet import RayNet
from dataset import GazeGeneDataset, MultiViewBatchSampler

from head_pose.loss import multiview_headpose_losses
from gaze_vector.loss import multiview_gaze_vector_geodesic_losses
from gaze_point.loss import multiview_gaze_point_losses
from gaze_depth.loss import multiview_gaze_depth_losses
from pupil_center.loss import multiview_pupil_center_losses
from ray_consistency_loss import multiview_ray_consistency_loss

import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="RayNet GradNorm Multitask Training")
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
    parser.add_argument('--plot_live', action="store_true", help="Show live loss plot during training (Jupyter/Colab)")
    parser.add_argument('--gradnorm_alpha', type=float, default=1.5)
    return parser.parse_args()


def get_backbone(backbone_name, weight_path, device):
    from backbone.repnext_utils import load_pretrained_repnext
    model = load_pretrained_repnext(backbone_name, weight_path)
    return model.to(device)


# Simple online normalizer
class RunningNormalizer:
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
        # If no spread, return 0.0
        if self.max - self.min < 1e-8:
            return 0.0
        return (float(val) - self.min) / (self.max - self.min)


def main():
    args = parse_args()

    if args.split == "train":
        subject_ids = [f"subject{i}" for i in range(1, 46)]
    else:
        subject_ids = [f"subject{i}" for i in range(46, 56)]

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Dataset and loader
    dataset = GazeGeneDataset(
        args.base_dir,
        subject_ids=subject_ids,
        samples_per_subject=args.samples_per_subject,
        transform=None,
        balance_attributes=['ethicity']
    )

    batch_sampler = MultiViewBatchSampler(dataset, batch_size=args.batch_size, balance_attributes=['ethicity'],
                                          shuffle=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

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
    model = RayNet(backbone, in_channels_list, panet_out_channels=256).to(device)

    NUM_TASKS = 5
    # Step 1: Learnable task weights
    task_weights = torch.nn.Parameter(torch.ones(NUM_TASKS, device=device), requires_grad=True)
    optimizer = optim.Adam(list(model.parameters()) + [task_weights], lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logfile = open(args.log_csv, "w", newline="")
    csv_writer = csv.DictWriter(logfile, fieldnames=[
        "epoch", "step", "batch_size",
        "head_pose_acc", "head_pose_cons",
        "gaze_vector_acc", "gaze_vector_cons",
        "gaze_point_acc", "gaze_point_cons",
        "gaze_depth_acc", "gaze_depth_cons",
        "pupil_center_acc", "pupil_center_cons",
        "ray_consistency",
        "norm_head_pose", "norm_gaze_vector", "norm_gaze_point",
        "norm_gaze_depth", "norm_pupil_center", "norm_ray_consistency"
    ])
    csv_writer.writeheader()

    # For normalization
    normalizers = {
        "head_pose": RunningNormalizer(),
        "gaze_vector": RunningNormalizer(),
        "gaze_point": RunningNormalizer(),
        "gaze_depth": RunningNormalizer(),
        "pupil_center": RunningNormalizer(),
        "ray_consistency": RunningNormalizer(),
    }

    # For GradNorm bookkeeping
    initial_losses = None
    alpha = args.gradnorm_alpha

    # For live plotting
    if args.plot_live:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 7))
        loss_hist = defaultdict(list)

    start_epoch = 0
    checkpoint_files = sorted([f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pth')])
    if checkpoint_files:
        last_ckpt = os.path.join(args.checkpoint_dir, checkpoint_files[-1])
        print(f"Resuming from checkpoint: {last_ckpt}")
        checkpoint = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    # -- Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for step, batch in enumerate(loader):
            images = batch['img'].to(device)
            B = images.shape[0] // 9
            images = images.view(B, 9, images.shape[1], images.shape[2], images.shape[3])
            images = images.reshape(B * 9, images.shape[2], images.shape[3], images.shape[4]).to(device)

            outputs = model(images)
            head_pose_gt = batch['head_pose']['R'].to(device)
            gaze_vector_gt = batch['gaze']['gaze_C'].to(device)
            gaze_point_gt = batch['gaze_point'].to(device)
            gaze_depth_gt = batch['gaze']['gaze_depth'].to(device)
            pupil_center_gt = batch['mesh']['pupil_center_3D'].to(device)

            B = head_pose_gt.shape[0] // 9
            N = 9

            head_pose_pred = outputs["head_pose_6d"].view(B, N, 6)
            gaze_vector_pred = outputs["gaze_vector_6d"].view(B, N, 6)
            gaze_point_pred = outputs["gaze_point_3d"].view(B, N, 3)
            gaze_depth_pred = outputs["gaze_depth"].view(B, N)
            pupil_center_pred = outputs["pupil_center_3d"].view(B, N, 2, 3)
            gaze_point_from_ray = outputs["gaze_point_from_ray"].view(B, N, 3)
            origin = outputs["origin"].view(B, N, 3)
            direction = outputs["direction"].view(B, N, 3)

            # Compute all losses
            head_pose_losses = multiview_headpose_losses(head_pose_pred, head_pose_gt.view(B, N, 3, 3))
            gaze_vector_losses = multiview_gaze_vector_geodesic_losses(gaze_vector_pred, gaze_vector_gt.view(B, N, 3))
            gaze_point_losses = multiview_gaze_point_losses(gaze_point_pred, gaze_point_gt.view(B, N, 3))
            gaze_depth_losses = multiview_gaze_depth_losses(gaze_depth_pred, gaze_depth_gt.view(B, N))
            pupil_center_losses = multiview_pupil_center_losses(pupil_center_pred, pupil_center_gt.view(B, N, 2, 3))
            ray_consist_loss = multiview_ray_consistency_loss(
                origins=origin,  # [B, N, 3]
                directions=direction,  # [B, N, 3]
                gaze_depths=gaze_depth_pred,  # [B, N]
                gaze_points_pred=gaze_point_pred  # [B, N, 3]
            )["total"]

            # Normalize (online, running min/max)
            for key, loss_val in [
                ("head_pose", head_pose_losses['accuracy'] + head_pose_losses['consistency']),
                ("gaze_vector", gaze_vector_losses['accuracy'] + gaze_vector_losses['consistency']),
                ("gaze_point", gaze_point_losses['accuracy'] + gaze_point_losses['consistency']),
                ("gaze_depth", gaze_depth_losses['accuracy'] + gaze_depth_losses['consistency']),
                ("pupil_center", pupil_center_losses['accuracy'] + pupil_center_losses['consistency']),
                # ("ray_consistency", ray_consist_loss),
            ]:
                normalizers[key].update(loss_val.item())

            norm_head_pose = normalizers["head_pose"].normalize(
                head_pose_losses['accuracy'] + head_pose_losses['consistency'])
            norm_gaze_vector = normalizers["gaze_vector"].normalize(
                gaze_vector_losses['accuracy'] + gaze_vector_losses['consistency'])
            norm_gaze_point = normalizers["gaze_point"].normalize(
                gaze_point_losses['accuracy'] + gaze_point_losses['consistency'])
            norm_gaze_depth = normalizers["gaze_depth"].normalize(
                gaze_depth_losses['accuracy'] + gaze_depth_losses['consistency'])
            norm_pupil_center = normalizers["pupil_center"].normalize(
                pupil_center_losses['accuracy'] + pupil_center_losses['consistency'])
            # norm_ray_consist = normalizers["ray_consistency"].normalize(ray_consist_loss)

            per_task_losses = torch.stack([
                (head_pose_losses['accuracy'] + head_pose_losses['consistency']) * norm_head_pose,
                (gaze_vector_losses['accuracy'] + gaze_vector_losses['consistency']) * norm_gaze_vector,
                (gaze_point_losses['accuracy'] + gaze_point_losses['consistency']) * norm_gaze_point,
                (gaze_depth_losses['accuracy'] + gaze_depth_losses['consistency']) * norm_gaze_depth,
                (pupil_center_losses['accuracy'] + pupil_center_losses['consistency']) * norm_pupil_center,
                # ray_consist_loss * norm_ray_consist,
            ])

            # ----------- GRADNORM BLOCK -------------
            # Compute the weighted sum of losses
            # After optimizer.zero_grad()
            weighted_losses = task_weights * per_task_losses
            L = weighted_losses.sum()
            optimizer.zero_grad()
            L.backward(retain_graph=True)

            # Pick a shared parameter to compute gradients with respect to
            # (for GradNorm, pick e.g. the last layer's weights)
            shared_param = list(model.fusion.parameters())[0]

            # Compute the gradient norm of each (task_weights[i] * per_task_losses[i])
            G_norm = []
            for i in range(NUM_TASKS):
                optimizer.zero_grad()
                # Create graph so task_weights get their gradients
                g = torch.autograd.grad(task_weights[i] * per_task_losses[i], shared_param, retain_graph=True,
                                        create_graph=True)[0]
                G_norm.append(g.norm())
            G_norm = torch.stack(G_norm)

            # Store the initial losses
            if initial_losses is None:
                initial_losses = per_task_losses.detach().clone()

            loss_ratios = per_task_losses.detach() / (initial_losses + 1e-8)
            avg_loss_ratio = loss_ratios.mean()
            target = G_norm.mean() * (loss_ratios / avg_loss_ratio) ** alpha

            gradnorm_loss = nn.L1Loss()(G_norm, target.detach())

            # This backward will propagate into task_weights!
            gradnorm_loss.backward()
            optimizer.step()

            # Optional: Normalize task weights (so they sum to NUM_TASKS)
            with torch.no_grad():
                coef = NUM_TASKS / (task_weights.sum() + 1e-8)
                task_weights.mul_(coef)

            # -------------- LOGGING AND PLOTTING ------------------
            log_dict = {
                "epoch": epoch,
                "step": step,
                "batch_size": args.batch_size,
                "head_pose_acc": float(head_pose_losses['accuracy']),
                "head_pose_cons": float(head_pose_losses['consistency']),
                "gaze_vector_acc": float(gaze_vector_losses['accuracy']),
                "gaze_vector_cons": float(gaze_vector_losses['consistency']),
                "gaze_point_acc": float(gaze_point_losses['accuracy']),
                "gaze_point_cons": float(gaze_point_losses['consistency']),
                "gaze_depth_acc": float(gaze_depth_losses['accuracy']),
                "gaze_depth_cons": float(gaze_depth_losses['consistency']),
                "pupil_center_acc": float(pupil_center_losses['accuracy']),
                "pupil_center_cons": float(pupil_center_losses['consistency']),
                "ray_consistency": float(ray_consist_loss),
                "norm_head_pose": norm_head_pose,
                "norm_gaze_vector": norm_gaze_vector,
                "norm_gaze_point": norm_gaze_point,
                "norm_gaze_depth": norm_gaze_depth,
                "norm_pupil_center": norm_pupil_center,
                # "norm_ray_consistency": norm_ray_consist,
            }
            csv_writer.writerow(log_dict)
            logfile.flush()

            # Live plot
            if args.plot_live:
                for key in ["head_pose_acc", "gaze_vector_acc", "gaze_point_acc", "gaze_depth_acc", "pupil_center_acc",
                            "ray_consistency"]:
                    loss_hist[key].append(log_dict[key])
                ax.clear()
                for key in ["head_pose_acc", "gaze_vector_acc", "gaze_point_acc", "gaze_depth_acc", "pupil_center_acc",
                            "ray_consistency"]:
                    ax.plot(loss_hist[key], label=key)
                ax.legend()
                ax.set_title(f"Epoch {epoch} Step {step} | Batch {args.batch_size}")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss")
                plt.pause(0.01)

            if (step + 1) % 10 == 0:
                print(
                    f"Epoch {epoch} | Step {step + 1}/{len(loader)} | Batch: {args.batch_size} | "
                    f"HP: {log_dict['head_pose_acc']:.4f} | GV: {log_dict['gaze_vector_acc']:.4f} | "
                    f"GP: {log_dict['gaze_point_acc']:.2f} | GD: {log_dict['gaze_depth_acc']:.2f} | "
                    f"PC: {log_dict['pupil_center_acc']:.2f} | RayCons: {log_dict['ray_consistency']:.2f} |"
                )

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"raynet_epoch{epoch + 1}.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': vars(args)
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    logfile.close()
    print("Training complete.")

    if args.plot_live:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
