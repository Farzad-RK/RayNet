import argparse
import os
import csv
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from raynet import RayNet, GeomBounds   # RayNet from your new raynet.py
from dataset import GazeGeneDataset, MultiViewBatchSampler, multiview_collate


# =========================
# Math utilities (SO(3), angles)
# =========================
def rotation_geodesic(R_pred, R_gt, eps: float = 1e-7):
    """
    Geodesic distance between rotations (in radians).
    R_pred, R_gt: [..., 3, 3]
    """
    # M = R_gt^T * R_pred
    M = torch.matmul(R_gt.transpose(-1, -2), R_pred)
    trace = M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2]
    cos_val = (trace - 1.0) * 0.5
    cos_val = torch.clamp(cos_val, -1.0 + eps, 1.0 - eps)
    angle = torch.arccos(cos_val)
    return angle


def unit_vector(v, eps: float = 1e-8):
    """Normalize vectors safely: v / ||v||."""
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def angle_between(u, v, eps: float = 1e-7):
    """
    Angle (radians) between two (possibly unnormalized) vectors u and v.
    u, v: [..., 3]
    """
    u_n = unit_vector(u)
    v_n = unit_vector(v)
    cos_val = (u_n * v_n).sum(dim=-1)
    cos_val = torch.clamp(cos_val, -1.0 + eps, 1.0 - eps)
    return torch.arccos(cos_val)


def skew(v):
    """
    Skew-symmetric matrix [v]_x.
    v: [..., 3]
    """
    z = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    K = torch.stack([
        torch.stack([ z,   -vz,   vy], dim=-1),
        torch.stack([ vz,   z,   -vx], dim=-1),
        torch.stack([-vy,   vx,    z], dim=-1),
    ], dim=-2)
    return K


def so3_exp(w, eps: float = 1e-8):
    """
    Exponential map from axis-angle to rotation matrix (Rodrigues).
    w: [..., 3] (axis * angle)
    returns: [..., 3, 3]
    """
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)  # [..., 1]
    k = torch.where(theta > eps, w / theta, torch.zeros_like(w))
    K = skew(k)  # [..., 3, 3]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(K.shape)
    sin_th = torch.sin(theta)[..., None]
    cos_th = torch.cos(theta)[..., None]
    R = I + sin_th * K + (1.0 - cos_th) * (K @ K)
    R = torch.where((theta[..., None] < eps), I, R)
    return R


# =========================
# CLI / backbone
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="RayNet Training with Uncertainty Weighting (9 tasks)")
    parser.add_argument('--base_dir', type=str, required=True, help="Root of GazeGene dataset")
    parser.add_argument('--backbone_name', type=str, default="repnext_m3")
    parser.add_argument('--weight_path', type=str, default="./repnext_m3_pretrained.pt")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--samples_per_subject', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    parser.add_argument('--log_csv', type=str, default="train_log.csv")
    parser.add_argument('--checkpoint_freq', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', type=str, default="train", choices=["train", "test"],
                        help="Use official GazeGene train/test split")
    return parser.parse_args()


def get_backbone(backbone_name, weight_path, device):
    from backbone.repnext_utils import load_pretrained_repnext
    model = load_pretrained_repnext(backbone_name, weight_path)
    return model.to(device)


# =========================
# Training
# =========================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Subject split (official): 1..45 train, 46..55 test
    if args.split == "train":
        subject_ids = [f"subject{i}" for i in range(1, 46)]
    else:
        subject_ids = [f"subject{i}" for i in range(46, 56)]

    # Dataset & DataLoader:
    #   - MultiViewBatchSampler groups the 9 camera views per (subject, frame).
    #   - multiview_collate packs tensors into shapes [B, 9, ...].
    dataset = GazeGeneDataset(
        args.base_dir,
        subject_ids=subject_ids,
        samples_per_subject=args.samples_per_subject,
        transform=None,
        balance_attributes=['ethicity'],
        include_2d=False  # can switch to True if you add 2D reprojection losses
    )
    batch_sampler = MultiViewBatchSampler(dataset, batch_size=args.batch_size,
                                          balance_attributes=['ethicity'], shuffle=True)
    dataloader = DataLoader(dataset,
                            batch_sampler=batch_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            collate_fn=multiview_collate)

    # Backbone channels per RepNeXt variant
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
    model = RayNet(backbone, in_channels_list, panet_out_channels=256,
                   bounds=GeomBounds()).to(device)

    # ---- Uncertainty weighting (Kendall & Gal) ----
    # We learn 9 log-variances (one per loss). This is the simplest dynamic scheme
    # that handles all tasks without extra gradient computations.
    # s_i = log(σ_i^2) (free parameters). Effective weight = exp(-s_i).
    # Combined loss = sum_i [ exp(-s_i) * L_i + s_i ]  (ignoring constant 1/2 factors).
    NUM_TASKS = 9
    log_vars = torch.nn.Parameter(torch.zeros(NUM_TASKS, device=device), requires_grad=True)

    # Optimizer: include both model params and log_vars (with a slightly smaller LR for stability)
    optimizer = optim.AdamW([
        {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": [log_vars], "lr": args.lr * 0.5, "weight_decay": 0.0},
    ])

    # CSV logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logfile = open(args.log_csv, "w", newline="")
    csv_writer = csv.DictWriter(
        logfile,
        fieldnames=[
            "epoch", "step", "batch_size",
            # total
            "loss_total",
            # individual task losses (cm for translations/centers; radians for geodesic; cosine loss is unitless)
            "loss_head_rotation", "loss_head_translation",
            "loss_eyeball_center",
            "loss_optic_axis", "loss_visual_axis",
            "loss_eyeball_radius", "loss_iris_radius", "loss_cornea_radius", "loss_cornea_depth",
            # metrics (angles in degrees, distances in cm)
            "deg_head_rotation", "deg_optic_axis", "deg_visual_axis",
            "rmse_head_translation_cm", "rmse_eyeball_center_cm",
            "abs_err_eyeball_radius_cm", "abs_err_iris_radius_cm",
            "abs_err_cornea_radius_cm", "abs_err_cornea_depth_cm",
            # effective weights and uncertainties
            "w_head_rotation", "w_head_translation",
            "w_eyeball_center", "w_optic_axis", "w_visual_axis",
            "w_eyeball_radius", "w_iris_radius", "w_cornea_radius", "w_cornea_depth",
            "sigma_head_rotation", "sigma_head_translation",
            "sigma_eyeball_center", "sigma_optic_axis", "sigma_visual_axis",
            "sigma_eyeball_radius", "sigma_iris_radius", "sigma_cornea_radius", "sigma_cornea_depth",
        ]
    )
    csv_writer.writeheader()

    # Resume from last checkpoint (if any)
    start_epoch = 0
    ckpt_files = sorted([f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pth")])
    if ckpt_files:
        last_ckpt = os.path.join(args.checkpoint_dir, ckpt_files[-1])
        print(f"Resuming from checkpoint: {last_ckpt}")
        checkpoint = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer_model'])  # keep key name for compatibility
        # restore log_vars if present
        if 'log_vars' in checkpoint:
            with torch.no_grad():
                log_vars.copy_(checkpoint['log_vars'].to(device))
        start_epoch = checkpoint['epoch'] + 1

    # Constant eye-forward vector in local eye frame (e_z)
    e_z = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 1, 3)  # [1, 1, 1, 3]

    # ========================
    # Training loop
    # ========================
    for epoch in range(start_epoch, args.epochs):
        model.train()

        for step, batch in enumerate(dataloader):
            # --------------------
            # Prepare inputs and GT
            # --------------------
            # Inputs: images [B, 9, 3, H, W]
            images = batch["img"].to(device)
            batch_size, num_views = images.shape[0], images.shape[1]

            # Ground-truth (from collate)
            R_head_gt = batch["R_head"].to(device)           # [B, 9, 3, 3]
            t_head_gt = batch["t_head"].to(device)           # [B, 9, 3]
            eyeball_center_ccs_gt = batch["eyecenter3d"].to(device)  # [B, 9, 2, 3] (CCS; cm)
            optic_axis_gt = batch["optic_axis"].to(device)   # [B, 9, 2, 3]
            visual_axis_gt = batch["visual_axis"].to(device) # [B, 9, 2, 3]
            beta = {k: v.to(device) for k, v in batch["beta"].items()}  # subject-level params in HCS (cm)

            # --------------------
            # Forward pass
            # --------------------
            outputs = model(images)  # dict: 'per_view', 'frame'

            # Predicted per-view head pose
            R_head_pred = outputs["per_view"]["R_head"]  # [B, 9, 3, 3]
            t_head_pred = outputs["per_view"]["t_head"]  # [B, 9, 3]

            # Predicted frame-level eye parameters
            R_eye_local = outputs["frame"]["R_eye"]      # [B, 2, 3, 3]
            kappa_local = outputs["frame"]["kappa"]      # [B, 2, 3]  (axis-angle; radians)
            eye_center_hcs = outputs["frame"]["c_eye"]   # [B, 2, 3]  (HCS; cm)
            r_eyeball_pred = outputs["frame"]["r_eye"]   # [B, 1]     (cm)
            r_iris_pred = outputs["frame"]["r_iris"]     # [B, 1]     (cm)
            r_cornea_pred = outputs["frame"]["r_cornea"] # [B, 1]     (cm)
            d_cornea_pred = outputs["frame"]["d_cornea"] # [B, 1]     (cm)

            # --------------------
            # Build derived predictions (axes, transformed centers)
            # --------------------
            # Eye's forward axis in local eye coordinates
            a_eye_local = torch.matmul(R_eye_local, e_z.view(1, 1, 3, 1)).squeeze(-1)  # [B, 2, 3]

            # Kappa rotation
            R_kappa = so3_exp(kappa_local.view(-1, 3)).view(batch_size, 2, 3, 3)       # [B, 2, 3, 3]

            # Expand across the 9 cameras for comparison in CCS
            a_eye_local_exp = a_eye_local.unsqueeze(1).expand(-1, num_views, -1, -1)   # [B, 9, 2, 3]

            # Optic axis in CCS: a_optic = R_head * a_eye_local
            a_optic_pred = torch.matmul(
                R_head_pred.unsqueeze(2),                      # [B, 9, 1, 3, 3]
                a_eye_local_exp.unsqueeze(-1)                 # [B, 9, 2, 3, 1]
            ).squeeze(-1)                                      # [B, 9, 2, 3]

            # Visual axis in CCS: a_visual = R_head * (R_kappa * a_eye_local)
            a_visual_local = torch.matmul(R_kappa, a_eye_local.unsqueeze(-1)).squeeze(-1)  # [B, 2, 3]
            a_visual_local_exp = a_visual_local.unsqueeze(1).expand(-1, num_views, -1, -1) # [B, 9, 2, 3]
            a_visual_pred = torch.matmul(
                R_head_pred.unsqueeze(2), a_visual_local_exp.unsqueeze(-1)
            ).squeeze(-1)                                      # [B, 9, 2, 3]

            # Eyeball centers in CCS: c_C = R_head * c_H + t_head
            eye_center_hcs_exp = eye_center_hcs.unsqueeze(1).expand(-1, num_views, -1, -1)  # [B, 9, 2, 3]
            eye_center_ccs_pred = (
                torch.matmul(R_head_pred.unsqueeze(2), eye_center_hcs_exp.unsqueeze(-1)).squeeze(-1)
                + t_head_pred.unsqueeze(2)
            )  # [B, 9, 2, 3] (cm)

            # ====================
            # Individual losses (9 tasks)
            # ====================
            # 1) Head rotation geodesic (radians)
            head_rotation_loss = rotation_geodesic(R_head_pred, R_head_gt).mean()

            # 2) Head translation L2 (cm)
            head_translation_loss = F.mse_loss(t_head_pred, t_head_gt)

            # 3) Eyeball center L2 in CCS (cm)
            eyeball_center_loss = F.mse_loss(eye_center_ccs_pred, eyeball_center_ccs_gt)

            # 4) Optic axis alignment: 1 - cosine similarity (unitless)
            optic_cosine = (unit_vector(a_optic_pred) * unit_vector(optic_axis_gt)).sum(dim=-1)  # [B, 9, 2]
            optic_axis_loss = (1.0 - optic_cosine).mean()

            # 5) Visual axis alignment: 1 - cosine similarity (unitless)
            visual_cosine = (unit_vector(a_visual_pred) * unit_vector(visual_axis_gt)).sum(dim=-1)
            visual_axis_loss = (1.0 - visual_cosine).mean()

            # 6-9) Radii and cornea depth supervision from subject β (cm)
            eyeball_radius_loss = F.mse_loss(r_eyeball_pred, beta["r_eye"])
            iris_radius_loss    = F.mse_loss(r_iris_pred,    beta["r_iris"])
            cornea_radius_loss  = F.mse_loss(r_cornea_pred,  beta["r_cornea"])
            cornea_depth_loss   = F.mse_loss(d_cornea_pred,  beta["d_cornea"])

            # Stack all task losses into a vector L (length 9)
            loss_vector = torch.stack([
                head_rotation_loss,
                head_translation_loss,
                eyeball_center_loss,
                optic_axis_loss,
                visual_axis_loss,
                eyeball_radius_loss,
                iris_radius_loss,
                cornea_radius_loss,
                cornea_depth_loss
            ], dim=0)  # [9]

            # ====================
            # Uncertainty weighting (dynamic)
            # ====================
            # s_i = log σ_i^2 ; weight_i = exp(-s_i)
            # Total loss = Σ [ weight_i * L_i + s_i ]
            weights = torch.exp(-log_vars)
            weighted_losses = weights * loss_vector + log_vars
            loss_total = weighted_losses.sum()

            # --------------------
            # Backprop & optimize
            # --------------------
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # ====================
            # Metrics for logging / printing
            # ====================
            # Convert angles to degrees, distances remain in cm (GazeGene units)
            deg_head_rotation = rotation_geodesic(R_head_pred, R_head_gt).mean().item() * (180.0 / np.pi)
            deg_optic_axis    = angle_between(a_optic_pred,  optic_axis_gt).mean().item() * (180.0 / np.pi)
            deg_visual_axis   = angle_between(a_visual_pred, visual_axis_gt).mean().item() * (180.0 / np.pi)
            rmse_head_translation_cm = torch.sqrt(F.mse_loss(t_head_pred, t_head_gt)).item()
            rmse_eyeball_center_cm   = torch.sqrt(F.mse_loss(eye_center_ccs_pred, eyeball_center_ccs_gt)).item()

            # Absolute errors (cm) for radii/depth
            abs_err_eyeball_radius_cm = (r_eyeball_pred - beta["r_eye"]).abs().mean().item()
            abs_err_iris_radius_cm    = (r_iris_pred    - beta["r_iris"]).abs().mean().item()
            abs_err_cornea_radius_cm  = (r_cornea_pred  - beta["r_cornea"]).abs().mean().item()
            abs_err_cornea_depth_cm   = (d_cornea_pred  - beta["d_cornea"]).abs().mean().item()

            # Effective weights and sigmas (diagnostics)
            effective_weights = weights.detach().cpu().tolist()
            sigmas = torch.exp(0.5 * log_vars).detach().cpu().tolist()

            # --------------------
            # CSV logging
            # --------------------
            log_row = {
                "epoch": epoch, "step": step, "batch_size": images.shape[0],
                "loss_total": float(loss_total.item()),
                "loss_head_rotation": float(head_rotation_loss.item()),
                "loss_head_translation": float(head_translation_loss.item()),
                "loss_eyeball_center": float(eyeball_center_loss.item()),
                "loss_optic_axis": float(optic_axis_loss.item()),
                "loss_visual_axis": float(visual_axis_loss.item()),
                "loss_eyeball_radius": float(eyeball_radius_loss.item()),
                "loss_iris_radius": float(iris_radius_loss.item()),
                "loss_cornea_radius": float(cornea_radius_loss.item()),
                "loss_cornea_depth": float(cornea_depth_loss.item()),
                "deg_head_rotation": deg_head_rotation,
                "deg_optic_axis": deg_optic_axis,
                "deg_visual_axis": deg_visual_axis,
                "rmse_head_translation_cm": rmse_head_translation_cm,
                "rmse_eyeball_center_cm": rmse_eyeball_center_cm,
                "abs_err_eyeball_radius_cm": abs_err_eyeball_radius_cm,
                "abs_err_iris_radius_cm": abs_err_iris_radius_cm,
                "abs_err_cornea_radius_cm": abs_err_cornea_radius_cm,
                "abs_err_cornea_depth_cm": abs_err_cornea_depth_cm,
                "w_head_rotation": effective_weights[0],
                "w_head_translation": effective_weights[1],
                "w_eyeball_center": effective_weights[2],
                "w_optic_axis": effective_weights[3],
                "w_visual_axis": effective_weights[4],
                "w_eyeball_radius": effective_weights[5],
                "w_iris_radius": effective_weights[6],
                "w_cornea_radius": effective_weights[7],
                "w_cornea_depth": effective_weights[8],
                "sigma_head_rotation": sigmas[0],
                "sigma_head_translation": sigmas[1],
                "sigma_eyeball_center": sigmas[2],
                "sigma_optic_axis": sigmas[3],
                "sigma_visual_axis": sigmas[4],
                "sigma_eyeball_radius": sigmas[5],
                "sigma_iris_radius": sigmas[6],
                "sigma_cornea_radius": sigmas[7],
                "sigma_cornea_depth": sigmas[8],
            }
            csv_writer.writerow(log_row)
            logfile.flush()

            # --------------------
            # Console prints (every 10 steps)
            # --------------------
            if (step + 1) % 10 == 0:
                print(
                    f"Epoch {epoch} | Step {step+1}/{len(dataloader)} | "
                    f"Total {log_row['loss_total']:.5f} | "
                    f"Rot(rad) {log_row['loss_head_rotation']:.5f} | "
                    f"Trans(cm^2) {log_row['loss_head_translation']:.5f} | "
                    f"Center(cm^2) {log_row['loss_eyeball_center']:.5f} | "
                    f"Optic(1-cos) {log_row['loss_optic_axis']:.5f} | "
                    f"Visual(1-cos) {log_row['loss_visual_axis']:.5f} | "
                    f"r_eye(cm^2) {log_row['loss_eyeball_radius']:.5f} | "
                    f"r_iris(cm^2) {log_row['loss_iris_radius']:.5f} | "
                    f"r_cornea(cm^2) {log_row['loss_cornea_radius']:.5f} | "
                    f"d_cornea(cm^2) {log_row['loss_cornea_depth']:.5f} | "
                    f"| Weights: "
                    f"{effective_weights}"
                )

        # --------------------
        # Save checkpoint
        # --------------------
        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"raynet_epoch{epoch+1}.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer_model': optimizer.state_dict(),    # keep key name for compatibility
                'log_vars': log_vars.detach().cpu(),          # store uncertainties explicitly
                'epoch': epoch,
                'args': vars(args)
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    logfile.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
