# train.py
import argparse
import os
import csv
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from raynet import RayNet, GeomBounds  # geometric model
from dataset import GazeGeneDataset, MultiViewBatchSampler, multiview_collate


# -------------------------
# Small math helpers
# -------------------------

def unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize vectors on the last dim."""
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def angle_between(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    u, v: [...,3] (not necessarily unit)
    returns angle in radians [...,]
    """
    u = unit(u); v = unit(v)
    cos = (u * v).sum(dim=-1)
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    return torch.arccos(cos)


def rotation_geodesic(R_pred: torch.Tensor, R_gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Geodesic distance on SO(3) between predicted and GT rotations (radians).
    R_pred, R_gt: [..., 3, 3]
    """
    M = torch.matmul(R_gt.transpose(-1, -2), R_pred)  # [...,3,3]
    trace = M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2]
    cos = (trace - 1.0) * 0.5
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    return torch.arccos(cos)


# -------------------------
# Simpler dynamic weighting (uncertainty weighting)
# Kendall & Gal (2018): sum( 0.5 * exp(-s_i) * L_i + 0.5 * s_i )
# -------------------------

class UncertaintyWeighting(nn.Module):
    def __init__(self, task_names: List[str]):
        super().__init__()
        self.log_vars = nn.ParameterDict({name: nn.Parameter(torch.zeros(1)) for name in task_names})

    def forward(self, losses: Dict[str, torch.Tensor]):
        """
        losses: dict name -> scalar tensor (mean reduced)
        returns: total loss scalar, dicts for logging
        """
        total = 0.0
        weighted = {}
        sigmas = {}
        for name, L in losses.items():
            if L is None:
                continue
            log_var = self.log_vars[name]
            weight = torch.exp(-log_var)
            term = 0.5 * weight * L + 0.5 * log_var
            total = total + term
            weighted[name] = term.detach()
            sigmas[name] = torch.exp(0.5 * log_var).detach()
        return total, weighted, sigmas


# -------------------------
# Arg parsing
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RayNet geometric training with uncertainty-weighted multi-task loss")
    parser.add_argument('--base_dir', type=str, required=True, help="Root of GazeGene dataset")
    parser.add_argument('--backbone_name', type=str, default="repnext_m3")
    parser.add_argument('--weight_path', type=str, default="./repnext_m3_pretrained.pt")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--samples_per_subject', type=int, default=None)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    parser.add_argument('--log_csv', type=str, default="train_log.csv")
    parser.add_argument('--checkpoint_freq', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', type=str, default="train", choices=["train", "test"],
                        help="Use official GazeGene train/test split: subjects 1..45 vs 46..55")
    parser.add_argument('--include_2d', action='store_true', help="Use 2D reprojection supervision if available")
    return parser.parse_args()


# -------------------------
# Backbone loader
# -------------------------

def get_backbone(backbone_name, weight_path, device):
    from backbone.repnext_utils import load_pretrained_repnext
    model = load_pretrained_repnext(backbone_name, weight_path)
    return model.to(device)


# -------------------------
# Training
# -------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Subject split per GazeGene paper
    if args.split == "train":
        subject_ids = [f"subject{i}" for i in range(1, 46)]
    else:
        subject_ids = [f"subject{i}" for i in range(46, 56)]

    # Dataset / Loader (multi-view sampler + collate => [B,9,...])
    dataset = GazeGeneDataset(
        base_dir=args.base_dir,
        subject_ids=subject_ids,
        samples_per_subject=args.samples_per_subject,
        transform=None,
        balance_attributes=['ethicity'],
        include_2d=args.include_2d
    )
    batch_sampler = MultiViewBatchSampler(dataset, batch_size=args.batch_size,
                                          balance_attributes=['ethicity'], shuffle=True)
    loader = DataLoader(dataset,
                        batch_sampler=batch_sampler,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        collate_fn=multiview_collate)

    # Backbone channels (for multi-scale token concat in RayNet)
    backbone_channels_dict = {
        'repnext_m0': [40, 80, 160, 320],
        'repnext_m1': [48, 96, 192, 384],
        'repnext_m2': [56, 112, 224, 448],
        'repnext_m3': [64, 128, 256, 512],
        'repnext_m4': [64, 128, 256, 512],
        'repnext_m5': [80, 160, 320, 640],
    }
    in_channels_list = backbone_channels_dict[args.backbone_name]

    # Model
    backbone = get_backbone(args.backbone_name, args.weight_path, device)
    model = RayNet(backbone, in_channels_list=in_channels_list, bounds=GeomBounds()).to(device)

    # Tasks for uncertainty weighting (include all; if a loss is missing, it’s skipped)
    task_names = [
        "head_rot", "head_trans",
        "eye_center",
        "optic_axis", "visual_axis",
        "r_eye", "r_iris", "r_cornea", "d_cornea",
        "iris3d", "iris2d",
        "pupil_center"
    ]
    weight_layer = UncertaintyWeighting(task_names).to(device)

    # Optimizer (model + learned log-variances)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(weight_layer.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # CSV logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logfile = open(args.log_csv, "w", newline="")
    csv_fields = [
        "epoch", "step", "batch_size", "loss_total",
        # raw losses
        "head_rot", "head_trans", "eye_center", "optic_axis", "visual_axis",
        "r_eye", "r_iris", "r_cornea", "d_cornea", "iris3d", "iris2d", "pupil_center",
        # weighted contributions
        "w_head_rot", "w_head_trans", "w_eye_center", "w_optic_axis", "w_visual_axis",
        "w_r_eye", "w_r_iris", "w_r_cornea", "w_d_cornea", "w_iris3d", "w_iris2d", "w_pupil_center",
        # learned sigmas
        "sigma_head_rot", "sigma_head_trans", "sigma_eye_center", "sigma_optic_axis", "sigma_visual_axis",
        "sigma_r_eye", "sigma_r_iris", "sigma_r_cornea", "sigma_d_cornea", "sigma_iris3d", "sigma_iris2d", "sigma_pupil_center",
        # interpretable metrics
        "deg_head_rot", "deg_optic_axis", "deg_visual_axis",
        "rmse_t_head", "rmse_center",
        "err_r_eye", "err_r_iris", "err_r_cornea", "err_d_cornea"
    ]
    csv_writer = csv.DictWriter(logfile, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Resume
    start_epoch = 0
    ckpts = [f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pth")]
    if ckpts:
        ckpts.sort()
        last_ckpt = os.path.join(args.checkpoint_dir, ckpts[-1])
        print(f"Resuming from checkpoint: {last_ckpt}")
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state['model'])
        # Backward compat: optimizer key may vary
        opt_key = 'optimizer' if 'optimizer' in state else ('optimizer_model' if 'optimizer_model' in state else None)
        if opt_key is not None:
            optimizer.load_state_dict(state[opt_key])
        if 'weights' in state:
            weight_layer.load_state_dict(state['weights'])
        start_epoch = state['epoch'] + 1

    # -------------------- Training loop --------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for step, batch in enumerate(loader):
            # ---- Move inputs to device ----
            imgs = batch["img"].to(device)              # [B,9,3,H,W]
            K = batch["K"].to(device)                   # [B,9,3,3]
            R_head_gt = batch["R_head"].to(device)      # [B,9,3,3]
            t_head_gt = batch["t_head"].to(device)      # [B,9,3]
            iris3d_gt = batch["iris3d"].to(device)      # [B,9,2,100,3]
            pupil3d_gt = batch["pupil3d"].to(device)    # [B,9,2,3]
            eyecenter3d_gt = batch["eyecenter3d"].to(device)  # [B,9,2,3]
            optic_axis_gt = batch["optic_axis"].to(device)     # [B,9,2,3]
            visual_axis_gt = batch["visual_axis"].to(device)   # [B,9,2,3]

            beta = {k: v.to(device) for k, v in batch["beta"].items()}  # subject-level priors (cm)

            # Optional 2D labels
            has_2d = args.include_2d and ("iris2d" in batch)
            if has_2d:
                iris2d_gt = batch["iris2d"].to(device)        # [B,9,2,100,2]

            # ---- Forward ----
            out = model(imgs)
            R_head_pred = out["per_view"]["R_head"]   # [B,9,3,3]
            t_head_pred = out["per_view"]["t_head"]   # [B,9,3]

            c_eye_h = out["frame"]["c_eye"]           # [B,2,3]
            R_eye = out["frame"]["R_eye"]             # [B,2,3,3]
            kappa = out["frame"]["kappa"]             # [B,2,3]
            r_eye_pred = out["frame"]["r_eye"]        # [B,1]
            r_iris_pred = out["frame"]["r_iris"]      # [B,1]
            r_cornea_pred = out["frame"]["r_cornea"]  # [B,1]
            d_cornea_pred = out["frame"]["d_cornea"]  # [B,1]
            iris_pts_h = out["frame"]["iris_pts_h"]   # [B,2,N,3]
            iris_params = out["frame"]["iris_params"]
            z_plane = iris_params["z_plane"]          # [B,2,1]

            Bsz, V = imgs.shape[:2]
            N = iris_pts_h.shape[2]  # usually 100

            # ---- Transforms / projections ----
            # 3D iris points: HCS -> CCS per view
            iris_pts_c = RayNet.hcs_to_ccs_points(iris_pts_h, R_head_pred, t_head_pred)  # [B,9,2,N,3]

            # 2D projection (pixels)
            if has_2d:
                iris_uv = RayNet.project_points(iris_pts_c, K)                           # [B,9,2,N,2]

            # Eyeball centers (HCS -> CCS)
            c_eye_h_exp = c_eye_h.unsqueeze(1).expand(-1, V, -1, -1)                     # [B,9,2,3]
            c_eye_c = torch.matmul(R_head_pred.unsqueeze(2), c_eye_h_exp.unsqueeze(-1)).squeeze(-1) + \
                      t_head_pred.unsqueeze(2)                                            # [B,9,2,3]

            # Predicted axes in CCS
            a_optic_pred, a_visual_pred = RayNet.compose_axes(R_head_pred, R_eye, kappa)  # [B,9,2,3] each

            # Pupil center prediction (eye-local [0,0,z_plane], into HCS, then CCS)
            pupil_local = torch.zeros_like(c_eye_h)                                       # [B,2,3]
            pupil_local[..., 2:3] = z_plane                                               # [B,2,1] on z
            pupil_h = torch.matmul(R_eye, pupil_local.unsqueeze(-1)).squeeze(-1) + c_eye_h  # [B,2,3]
            pupil_c = torch.matmul(R_head_pred.unsqueeze(2),
                                   pupil_h.unsqueeze(1).expand(-1, V, -1, -1).unsqueeze(-1)
                                   ).squeeze(-1) + t_head_pred.unsqueeze(2)              # [B,9,2,3]

            # ---- Losses (all in cm/rad) ----
            losses: Dict[str, torch.Tensor] = {}

            # Head pose
            losses["head_rot"] = rotation_geodesic(R_head_pred, R_head_gt).mean()
            losses["head_trans"] = F.mse_loss(t_head_pred, t_head_gt)  # cm^2

            # Eye center in CCS
            losses["eye_center"] = F.mse_loss(c_eye_c, eyecenter3d_gt)  # cm^2

            # Axes (use 1 - cosine similarity)
            optic_cos = (unit(a_optic_pred) * unit(optic_axis_gt)).sum(dim=-1)  # [B,9,2]
            visual_cos = (unit(a_visual_pred) * unit(visual_axis_gt)).sum(dim=-1)
            losses["optic_axis"] = (1.0 - optic_cos).mean()
            losses["visual_axis"] = (1.0 - visual_cos).mean()

            # Radii / cornea distances (subject-level priors; cm^2)
            losses["r_eye"] = F.mse_loss(r_eye_pred, beta["r_eye"])
            losses["r_iris"] = F.mse_loss(r_iris_pred, beta["r_iris"])
            losses["r_cornea"] = F.mse_loss(r_cornea_pred, beta["r_cornea"])
            losses["d_cornea"] = F.mse_loss(d_cornea_pred, beta["d_cornea"])

            # Iris mesh 3D in CCS
            losses["iris3d"] = F.mse_loss(iris_pts_c, iris3d_gt)  # cm^2

            # Optional 2D reprojection
            if has_2d:
                losses["iris2d"] = F.mse_loss(iris_uv, iris2d_gt)  # pixels^2
            else:
                losses["iris2d"] = None

            # Pupil center
            losses["pupil_center"] = F.mse_loss(pupil_c, pupil3d_gt)  # cm^2

            # ---- Uncertainty weighting ----
            total_loss, weighted, sigmas = weight_layer(losses)

            # ---- Optimize ----
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # ---- Metrics (human-readable) ----
            deg_head_rot = rotation_geodesic(R_head_pred, R_head_gt).mean().item() * (180.0 / np.pi)
            deg_optic_axis = angle_between(a_optic_pred, optic_axis_gt).mean().item() * (180.0 / np.pi)
            deg_visual_axis = angle_between(a_visual_pred, visual_axis_gt).mean().item() * (180.0 / np.pi)
            rmse_t_head = torch.sqrt(F.mse_loss(t_head_pred, t_head_gt)).item()  # cm
            rmse_center = torch.sqrt(F.mse_loss(c_eye_c, eyecenter3d_gt)).item() # cm

            err_r_eye = (r_eye_pred - beta["r_eye"]).abs().mean().item()         # cm
            err_r_iris = (r_iris_pred - beta["r_iris"]).abs().mean().item()      # cm
            err_r_cornea = (r_cornea_pred - beta["r_cornea"]).abs().mean().item()# cm
            err_d_cornea = (d_cornea_pred - beta["d_cornea"]).abs().mean().item()# cm

            # ---- Logging ----
            log_row = {
                "epoch": epoch, "step": step, "batch_size": args.batch_size,
                "loss_total": float(total_loss.item()),
                # raw losses
                "head_rot": float(losses["head_rot"].item()),
                "head_trans": float(losses["head_trans"].item()),
                "eye_center": float(losses["eye_center"].item()),
                "optic_axis": float(losses["optic_axis"].item()),
                "visual_axis": float(losses["visual_axis"].item()),
                "r_eye": float(losses["r_eye"].item()),
                "r_iris": float(losses["r_iris"].item()),
                "r_cornea": float(losses["r_cornea"].item()),
                "d_cornea": float(losses["d_cornea"].item()),
                "iris3d": float(losses["iris3d"].item()),
                "iris2d": float(losses["iris2d"].item()) if losses["iris2d"] is not None else 0.0,
                "pupil_center": float(losses["pupil_center"].item()),
                # weighted contributions
                "w_head_rot": float(weighted["head_rot"].item()),
                "w_head_trans": float(weighted["head_trans"].item()),
                "w_eye_center": float(weighted["eye_center"].item()),
                "w_optic_axis": float(weighted["optic_axis"].item()),
                "w_visual_axis": float(weighted["visual_axis"].item()),
                "w_r_eye": float(weighted["r_eye"].item()),
                "w_r_iris": float(weighted["r_iris"].item()),
                "w_r_cornea": float(weighted["r_cornea"].item()),
                "w_d_cornea": float(weighted["d_cornea"].item()),
                "w_iris3d": float(weighted["iris3d"].item()),
                "w_iris2d": float(weighted["iris2d"].item()) if "iris2d" in weighted else 0.0,
                "w_pupil_center": float(weighted["pupil_center"].item()),
                # sigmas
                "sigma_head_rot": float(sigmas["head_rot"].item()),
                "sigma_head_trans": float(sigmas["head_trans"].item()),
                "sigma_eye_center": float(sigmas["eye_center"].item()),
                "sigma_optic_axis": float(sigmas["optic_axis"].item()),
                "sigma_visual_axis": float(sigmas["visual_axis"].item()),
                "sigma_r_eye": float(sigmas["r_eye"].item()),
                "sigma_r_iris": float(sigmas["r_iris"].item()),
                "sigma_r_cornea": float(sigmas["r_cornea"].item()),
                "sigma_d_cornea": float(sigmas["d_cornea"].item()),
                "sigma_iris3d": float(sigmas["iris3d"].item()),
                "sigma_iris2d": float(sigmas["iris2d"].item()) if "iris2d" in sigmas else 0.0,
                "sigma_pupil_center": float(sigmas["pupil_center"].item()),
                # metrics
                "deg_head_rot": deg_head_rot,
                "deg_optic_axis": deg_optic_axis,
                "deg_visual_axis": deg_visual_axis,
                "rmse_t_head": rmse_t_head,
                "rmse_center": rmse_center,
                "err_r_eye": err_r_eye,
                "err_r_iris": err_r_iris,
                "err_r_cornea": err_r_cornea,
                "err_d_cornea": err_d_cornea,
            }
            # Print a succinct line every 10 iters
            if (step + 1) % 10 == 0:
                print(
                    f"Epoch {epoch} | Step {step+1}/{len(loader)} | "
                    f"Total {log_row['loss_total']:.3f} | "
                    f"Rot(deg) {deg_head_rot:.2f} | t_rmse(cm) {rmse_t_head:.3f} | "
                    f"center_rmse(cm) {rmse_center:.3f} | optic(deg) {deg_optic_axis:.2f} | "
                    f"visual(deg) {deg_visual_axis:.2f} | iris3d {log_row['iris3d']:.2f}"
                )

            # Write CSV
            csv_writer.writerow(log_row)
            logfile.flush()

        # ---- Save checkpoint ----
        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"raynet_epoch{epoch+1}.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'weights': weight_layer.state_dict(),
                'epoch': epoch,
                'args': vars(args)
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    logfile.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
