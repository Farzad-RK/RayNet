# train.py

import argparse
import os
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from raynet import RayNet
from dataset import GazeGeneDataset, MultiViewBatchSampler

from head_pose.loss import multiview_headpose_losses
from gaze_vector.loss import multiview_gaze_vector_geodesic_losses
from gaze_point.loss import multiview_gaze_point_losses
from pupil_center.loss import multiview_pupil_center_losses
from sixdrepnet.utils import compute_rotation_matrix_from_ortho6d
from pcgrad import PCGrad
# Optional: from utils import ... (for seeding, reproducibility, etc.)

def parse_args():
    parser = argparse.ArgumentParser(description="RayNet PCGrad Multitask Training")
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


    # Add any other hyperparams here
    return parser.parse_args()

def get_backbone(backbone_name, weight_path, device):
    from backbone.repnext_utils import load_pretrained_repnext
    model = load_pretrained_repnext(backbone_name, weight_path)
    return model.to(device)

def main():
    args = parse_args()

    if args.split == "train":
        subject_ids = [f"subject{i}" for i in range(1,46)]
    else:
        subject_ids = [f"subject{i}" for i in range(46, 56)]

    torch.manual_seed(args.seed)

    # -- Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Dataset and loader
    dataset = GazeGeneDataset(
        args.base_dir,
        subject_ids=subject_ids,
        samples_per_subject=args.samples_per_subject,       # Only 50 random frames per subject
        transform=None,               # or your torchvision transforms
        balance_attributes=['ethicity']  # or other attribute(s) from subject_label.pkl
    )

    batch_sampler = MultiViewBatchSampler(dataset, balance_attributes=['ethicity'], shuffle=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

    # -- Backbone, RayNet, etc.
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


    # -- Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # -- Logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logfile = open(args.log_csv, "w", newline="")
    csv_writer = csv.DictWriter(logfile, fieldnames=[
        "epoch", "step", "head_pose_acc", "head_pose_cons",
        "gaze_vector_acc", "gaze_vector_cons",
        "gaze_point_acc", "gaze_point_cons",
        "pupil_center_acc", "pupil_center_cons",
    ])
    csv_writer.writeheader()


    start_epoch = 0
    # --- Resume logic (if needed)
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
            images = batch['img'].to(device)  # [B*9, 3, H, W] or [B, 9, 3, H, W] depending on loader
            # -- Repack for [B, 9, ...]
            B = images.shape[0] // 9
            images = images.view(B, 9, images.shape[1], images.shape[2], images.shape[3])
            images = images.reshape(B*9, images.shape[2], images.shape[3], images.shape[4])
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            # You may need to unpack outputs as [B, 9, ...] for each
            # Here: assuming each output is [B*9, ...]
            head_pose_gt = batch['head_pose']['R'].to(device)  # [B*9, 3, 3]
            gaze_vector_gt = batch['gaze']['gaze_C'].to(device)  # [B*9, 3]
            gaze_point_gt = batch['gaze_point'].to(device)       # [B*9, 3]
            pupil_center_gt = batch['mesh']['pupil_center_3D'].to(device) # [B*9, 2, 3]

            # Reshape for multi-view [B, 9, ...]
            B = head_pose_gt.shape[0] // 9
            N = 9
            # Head pose GT is [B*9, 3, 3], reshape to [B, N, 3, 3]
            head_pose_pred = outputs["head_pose_6d"].view(B, N, 6)
            head_pose_pred = compute_rotation_matrix_from_ortho6d(head_pose_pred.view(-1, 6)).view(B, N, 3, 3)
            # Gaze vector GT is [B*9, 3], reshape to [B, N, 3]
            gaze_vector_pred = outputs["gaze_vector_6d"].view(B, N, 6)
            # Gaze point GT is [B*9, 3], reshape to [B, N, 3]
            gaze_point_pred = outputs["gaze_point_3d"].view(B, N, 3)
            pupil_center_pred = outputs["pupil_center_3d"].view(B, N, 2, 3)

            # Losses: multi-view for each head
            head_pose_losses = multiview_headpose_losses(head_pose_pred, head_pose_gt.view(B, N, 3, 3))
            gaze_vector_losses = multiview_gaze_vector_geodesic_losses(gaze_vector_pred, gaze_vector_gt.view(B, N, 3))
            gaze_point_losses = multiview_gaze_point_losses(gaze_point_pred, gaze_point_gt.view(B, N, 3))
            pupil_center_losses = multiview_pupil_center_losses(pupil_center_pred, pupil_center_gt.view(B, N, 2, 3))

            per_task_total = [
                head_pose_losses['accuracy'] + head_pose_losses['consistency'],
                gaze_vector_losses['accuracy'] + gaze_vector_losses['consistency'],
                gaze_point_losses['accuracy'] + gaze_point_losses['consistency'],
                pupil_center_losses['accuracy'] + pupil_center_losses['consistency'],
            ]
            task_names = ["head_pose", "gaze_vector", "gaze_point", "pupil_center"]

            # Get shared parameters (all except head-private)
            shared_params = []
            for n, p in model.named_parameters():
                if any(h in n for h in [
                    "head_pose_regression", "gaze_vector_regression", "gaze_point_regression", "pupil_center_regression"
                ]):
                    continue
                shared_params.append(p)

            # Instantiate PCGrad once, after you build your optimizer:
            pc_optimizer = PCGrad(optimizer)

            # … inside each training step …
            # per_task_total is your list: [head_loss, gaze_vec_loss, gaze_point_loss, pupil_center_loss]
            shared_params = [p for n, p in model.named_parameters() if p.requires_grad]

            # 1) De‐conflict & accumulate grads:
            pc_optimizer.pc_backward(per_task_total, shared_params, retain_graph=True)

            # 2) Take the optimizer step:
            pc_optimizer.step()

            # Logging step/epoch results
            log_dict = {
                "epoch": epoch,
                "step": step,
                "head_pose_acc": float(head_pose_losses['accuracy']),
                "head_pose_cons": float(head_pose_losses['consistency']),
                "gaze_vector_acc": float(gaze_vector_losses['accuracy']),
                "gaze_vector_cons": float(gaze_vector_losses['consistency']),
                "gaze_point_acc": float(gaze_point_losses['accuracy']),
                "gaze_point_cons": float(gaze_point_losses['consistency']),
                "pupil_center_acc": float(pupil_center_losses['accuracy']),
                "pupil_center_cons": float(pupil_center_losses['consistency']),
            }
            csv_writer.writerow(log_dict)
            logfile.flush()

            if (step+1) % 10 == 0:
                print(
                    f"Epoch {epoch} Step {step}: "
                    f"HP: {log_dict['head_pose_acc']:.4f}/{log_dict['head_pose_cons']:.4f} | "
                    f"GV: {log_dict['gaze_vector_acc']:.4f}/{log_dict['gaze_vector_cons']:.4f} | "
                    f"GP: {log_dict['gaze_point_acc']:.4f}/{log_dict['gaze_point_cons']:.4f} | "
                    f"PC: {log_dict['pupil_center_acc']:.4f}/{log_dict['pupil_center_cons']:.4f} | "
                )

        # -- Save checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"raynet_epoch{epoch+1}.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': vars(args)
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    logfile.close()
    print("Training complete.")

if __name__ == "__main__":
    main()
