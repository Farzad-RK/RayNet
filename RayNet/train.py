# train.py
import argparse
import os
import csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from raynet import RayNet
from dataset import GazeGeneDataset, MultiViewBatchSampler

# multiview losses you already use
from head_pose.loss import multiview_headpose_losses
from gaze_vector.loss import multiview_gaze_vector_geodesic_losses
from gaze_point.loss import multiview_gaze_point_losses
from gaze_depth.loss import multiview_gaze_depth_losses
from pupil_center.loss import multiview_pupil_center_losses
from ray_consistency_loss import multiview_ray_consistency_loss

import matplotlib.pyplot as plt


# -----------------------------
# helpers
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser("RayNet + MediaPipe-iris depth")
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--backbone_name", type=str, default="repnext_m3")
    ap.add_argument("--weight_path", type=str, default="./repnext_m3_pretrained.pt")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    ap.add_argument("--log_csv", type=str, default="train_log.csv")
    ap.add_argument("--checkpoint_freq", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--samples_per_subject", type=int, default=None)
    ap.add_argument("--plot_live", action="store_true")
    ap.add_argument("--gradnorm_alpha", type=float, default=1.5)

    # Optional fallbacks if dataset does not provide intrinsics/identity
    ap.add_argument("--fx_pixels_fallback", type=float, default=1200.0,
                    help="Fallback focal length in pixels if K not in dataset")
    ap.add_argument("--iris_diam_cm_fallback", type=float, default=1.17,
                    help="Fallback iris diameter (cm) if identity missing")
    ap.add_argument("--eyeball_radius_cm_fallback", type=float, default=1.2,
                    help="Fallback eyeball radius (cm) if identity missing")
    return ap.parse_args()


def get_backbone(backbone_name, weight_path, device):
    from backbone.repnext_utils import load_pretrained_repnext
    return load_pretrained_repnext(backbone_name, weight_path).to(device)


class RunningNormalizer:
    def __init__(self):
        self.min = float("inf"); self.max = float("-inf")
    def update(self, v):
        v = float(v); self.min = min(self.min, v); self.max = max(self.max, v)
    def __call__(self, v):
        if self.max <= self.min + 1e-8: return 0.0
        return (float(v) - self.min) / (self.max - self.min)


def ray_to_point_l1(origins, directions, points, eps=1e-6):
    """
    Perpendicular L1 distance from points to rays (in cm).
    origins    : [B,V,3]
    directions : [B,V,3] (unit)
    points     : [B,V,3]
    """
    v = points - origins
    cross = torch.linalg.vector_norm(torch.cross(v, directions, dim=-1), dim=-1)  # [B,V]
    return cross.mean()


# -----------------------------
# main
# -----------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset / loader
    if args.split == "train":
        subject_ids = [f"subject{i}" for i in range(1, 46)]
    else:
        subject_ids = [f"subject{i}" for i in range(46, 56)]

    dataset = GazeGeneDataset(
        args.base_dir,
        subject_ids=subject_ids,
        samples_per_subject=args.samples_per_subject,
        transform=None,
        balance_attributes=["ethicity"],
    )
    batch_sampler = MultiViewBatchSampler(dataset, batch_size=args.batch_size,
                                          balance_attributes=["ethicity"], shuffle=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                        num_workers=args.num_workers, pin_memory=True)

    # model
    backbone_channels = {
        "repnext_m0": [40, 80, 160, 320],
        "repnext_m1": [48, 96, 192, 384],
        "repnext_m2": [56, 112, 224, 448],
        "repnext_m3": [64, 128, 256, 512],
        "repnext_m4": [64, 128, 256, 512],
        "repnext_m5": [80, 160, 320, 640],
    }
    backbone = get_backbone(args.backbone_name, args.weight_path, device)
    model = RayNet(backbone=backbone, in_channels_list=backbone_channels[args.backbone_name],
                   n_iris_points=100).to(device)

    # optimizers / GradNorm
    NUM_TASKS = 5  # HP, GV, GP, GD, PC
    task_weights = nn.Parameter(torch.ones(NUM_TASKS, device=device))
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_weights = optim.Adam([task_weights], lr=args.lr)

    # logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logf = open(args.log_csv, "w", newline="")
    csv_writer = csv.DictWriter(
        logf,
        fieldnames=[
            "epoch","step","batch",
            "hp_acc","hp_cons",
            "gv_acc","gv_cons",
            "gp_acc","gp_cons",
            "gd_acc","gd_cons",
            "pc_acc","pc_cons",
            "ray_cons","rayp_cons",
            "w_hp","w_gv","w_gp","w_gd","w_pc",
        ],
    )
    csv_writer.writeheader()

    # live plot (optional)
    if args.plot_live:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 6))
        curves = defaultdict(list)

    # GradNorm control
    init_losses = None
    alpha = args.gradnorm_alpha
    normalizer = {k: RunningNormalizer() for k in ["hp","gv","gp","gd","pc"]}

    # resume
    start_epoch = 0
    ckpts = sorted([f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pth")])
    if ckpts:
        last = os.path.join(args.checkpoint_dir, ckpts[-1])
        print(f"Resuming from {last}")
        state = torch.load(last, map_location=device)
        model.load_state_dict(state["model"])
        optimizer_model.load_state_dict(state["optimizer_model"])
        optimizer_weights.load_state_dict(state["optimizer_weights"])
        start_epoch = int(state["epoch"]) + 1

    # training
    V = 9  # multi-view cameras per sample
    CM2M = 0.01

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for step, batch in enumerate(loader):
            imgs = batch["img"].to(device)              # [B*V, 3, H, W]
            B = imgs.shape[0] // V
            _, _, H, W = imgs.shape

            # intrinsics / identity
            if "K" in batch:
                # batch["K"]: likely [B*V,3,3] or [B,V,3,3] depending on your dataset.py
                K = batch["K"]
                if K.dim() == 4:   # [B,V,3,3] -> [B*V,3,3]
                    K = K.to(device).view(B*V, 3, 3)
                else:
                    K = K.to(device)
                fx = K[:, 0, 0]  # [B*V]
            else:
                fx = torch.full((B*V,), args.fx_pixels_fallback, device=device)

            identity = batch.get("identity", {})
            if "iris_diameter_cm" in identity:
                iris_diam = identity["iris_diameter_cm"]
                if iris_diam.dim() == 3:  # [B,V,2] -> average per view to [B*V] or keep per-eye if needed
                    iris_diam = iris_diam.mean(dim=2)  # [B,V]
                if iris_diam.dim() == 2:  # [B,V] -> [B*V]
                    iris_diam = iris_diam.to(device).reshape(B*V)
                else:
                    iris_diam = iris_diam.to(device).reshape(B*V)
            else:
                iris_diam = torch.full((B*V,), args.iris_diam_cm_fallback, device=device)

            # optional per-eye eyeball radius
            if "eyeball_radius_cm" in identity:
                er = identity["eyeball_radius_cm"].to(device)
                if er.dim() == 3:  # [B,V,2] -> [B*V,2]
                    er = er.view(B*V, 2)
            else:
                er = torch.full((B*V, 2), args.eyeball_radius_cm_fallback, device=device)

            Kdict = {"fx": fx}  # [B*V]
            identity_dict = {
                "iris_diam_cm": iris_diam,              # [B*V]
                "eyeball_radius_cm": er,                # [B*V,2]
            }

            # forward
            out = model(imgs, K=Kdict, identity=identity_dict)

            # reshape predictions to [B, V, ...]
            hp_pred   = out["head_pose_6d"].view(B, V, 6)
            gv_pred   = out["gaze_vector_6d"].view(B, V, 6)
            gp_pred   = out["gaze_point_3d"].view(B, V, 3)   # cm
            gd_pred   = out["gaze_depth"].view(B, V)         # cm
            pc_pred   = out["pupil_center_3d"].view(B, V, 2, 3)
            origins    = out["origin"].view(B, V, 3)
            directions = out["direction"].view(B, V, 3)

            # ground truth
            hp_gt = batch["head_pose"]["R"].to(device).view(B, V, 3, 3)
            gv_gt = batch["gaze"]["gaze_C"].to(device).view(B, V, 3)
            gp_gt = batch["gaze_point"].to(device).view(B, V, 3)      # cm
            gd_gt = batch["gaze"]["gaze_depth"].to(device).view(B, V) # cm
            pc_gt = batch["mesh"]["pupil_center_3D"].to(device).view(B, V, 2, 3)

            # convert GP/GD to meters for numerically stable regression
            gp_pred_m = gp_pred * CM2M
            gp_gt_m   = gp_gt   * CM2M
            gd_pred_m = gd_pred * CM2M
            gd_gt_m   = gd_gt   * CM2M

            # task losses
            hp_loss = multiview_headpose_losses(hp_pred, hp_gt)
            gv_loss = multiview_gaze_vector_geodesic_losses(gv_pred, gv_gt)
            gp_loss = multiview_gaze_point_losses(gp_pred_m, gp_gt_m)
            gd_loss = multiview_gaze_depth_losses(gd_pred_m, gd_gt_m)   # analytic depth supervised
            pc_loss = multiview_pupil_center_losses(pc_pred, pc_gt)
            ray_loss = multiview_ray_consistency_loss(
                origins=origins, directions=directions, gaze_depths=gd_pred, gaze_points_pred=gp_pred
            )["total"]
            rayp_loss = ray_to_point_l1(origins, directions, gp_gt)     # cm

            # running normalizers
            for k, L in [
                ("hp", hp_loss["accuracy"] + hp_loss["consistency"]),
                ("gv", gv_loss["accuracy"] + gv_loss["consistency"]),
                ("gp", gp_loss["accuracy"] + gp_loss["consistency"]),
                ("gd", gd_loss["accuracy"] + gd_loss["consistency"]),
                ("pc", pc_loss["accuracy"] + pc_loss["consistency"]),
            ]:
                normalizer[k].update(L.item())

            n_hp = normalizer["hp"](hp_loss["accuracy"] + hp_loss["consistency"])
            n_gv = normalizer["gv"](gv_loss["accuracy"] + gv_loss["consistency"])
            n_gp = normalizer["gp"](gp_loss["accuracy"] + gp_loss["consistency"])
            n_gd = normalizer["gd"](gd_loss["accuracy"] + gd_loss["consistency"])
            n_pc = normalizer["pc"](pc_loss["accuracy"] + pc_loss["consistency"])

            per_task_losses = torch.stack([
                (hp_loss["accuracy"] + hp_loss["consistency"]) * n_hp,
                (gv_loss["accuracy"] + gv_loss["consistency"]) * n_gv,
                (gp_loss["accuracy"] + gp_loss["consistency"]) * n_gp,
                (gd_loss["accuracy"] + gd_loss["consistency"]) * n_gd,
                (pc_loss["accuracy"] + pc_loss["consistency"]) * n_pc,
            ])

            # GradNorm base norms w.r.t a shared param (CoordNeck)
            shared_param = list(model.fusion.parameters())[0]
            base_norms = []
            for i in range(NUM_TASKS):
                g = torch.autograd.grad(per_task_losses[i], shared_param, retain_graph=True)[0]
                base_norms.append(g.norm().detach())
            base_norms = torch.stack(base_norms)

            # phase 1: update model
            optimizer_model.zero_grad()
            model_loss = (task_weights.detach() * per_task_losses).sum() + 0.1 * ray_loss + 0.1 * rayp_loss
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer_model.step()

            # GradNorm initialization
            if init_losses is None:
                init_losses = per_task_losses.detach().clone()

            # phase 2: update task weights
            loss_ratios = per_task_losses.detach() / (init_losses + 1e-8)
            avg_ratio = loss_ratios.mean()
            G = task_weights * base_norms
            target_G = G.mean().detach() * (loss_ratios / (avg_ratio + 1e-8)) ** args.gradnorm_alpha

            optimizer_weights.zero_grad()
            gradnorm_loss = F.l1_loss(G, target_G)
            gradnorm_loss.backward()
            optimizer_weights.step()

            with torch.no_grad():
                task_weights.mul_(NUM_TASKS / (task_weights.sum() + 1e-8))

            # logging
            row = {
                "epoch": int(epoch), "step": int(step), "batch": int(args.batch_size),
                "hp_acc": float(hp_loss["accuracy"]), "hp_cons": float(hp_loss["consistency"]),
                "gv_acc": float(gv_loss["accuracy"]), "gv_cons": float(gv_loss["consistency"]),
                "gp_acc": float(gp_loss["accuracy"]), "gp_cons": float(gp_loss["consistency"]),
                "gd_acc": float(gd_loss["accuracy"]), "gd_cons": float(gd_loss["consistency"]),
                "pc_acc": float(pc_loss["accuracy"]), "pc_cons": float(pc_loss["consistency"]),
                "ray_cons": float(ray_loss), "rayp_cons": float(rayp_loss),
                "w_hp": float(task_weights[0].item()),
                "w_gv": float(task_weights[1].item()),
                "w_gp": float(task_weights[2].item()),
                "w_gd": float(task_weights[3].item()),
                "w_pc": float(task_weights[4].item()),
            }
            csv_writer.writerow(row); logf.flush()

            if args.plot_live:
                for k in ["hp_acc","gv_acc","gp_acc","gd_acc","pc_acc","ray_cons","rayp_cons"]:
                    curves[k].append(row[k])
                ax.clear()
                for k, v in curves.items():
                    ax.plot(np.asarray(v, dtype=float), label=k)
                ax.legend(); ax.set_title(f"Epoch {epoch} Step {step}"); plt.pause(0.01)

            if (step + 1) % 10 == 0:
                print(
                    f"Epoch {epoch} | Step {step+1}/{len(loader)} | "
                    f"HP {row['hp_acc']:.3f} | GV {row['gv_acc']:.3f} | "
                    f"GP {row['gp_acc']:.3f} | GD {row['gd_acc']:.3f} | "
                    f"PC {row['pc_acc']:.3f} | Ray {row['ray_cons']:.3f} | RayP {row['rayp_cons']:.3f}"
                )

        # checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            ckpt = os.path.join(args.checkpoint_dir, f"raynet_epoch{epoch+1}.pth")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer_model": optimizer_model.state_dict(),
                    "optimizer_weights": optimizer_weights.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                }, ckpt
            )
            print(f"Saved checkpoint: {ckpt}")

    logf.close()
    if args.plot_live:
        plt.ioff(); plt.show()


if __name__ == "__main__":
    main()
