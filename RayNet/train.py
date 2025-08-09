# train.py
import argparse
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
# If you run headless (SSH), uncomment:
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from raynet import RayNetStage1
from dataset import GazeGeneDataset, MultiViewBatchSampler

from head_pose.loss import multiview_headpose_losses
from gaze_vector.loss import multiview_gaze_vector_vmf_losses
from pupil_center.loss import multiview_pupil_center_losses  # SOTA pupil loss (geometry lifting)

# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser("RayNet Stage-1 (HeadPose + vMF Gaze + Geometric Pupil) with GradNorm+Kendall")
    p.add_argument('--base_dir', type=str, required=True)
    p.add_argument('--backbone_name', type=str, default="repnext_m3")
    p.add_argument('--weight_path', type=str, default="./repnext_m3_pretrained.pt")
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--samples_per_subject', type=int, default=None)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--checkpoint_dir', type=str, default="checkpoints_stage1")
    p.add_argument('--log_csv', type=str, default="train_stage1_log.csv")
    p.add_argument('--checkpoint_freq', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', type=str, default="train", choices=["train", "test"])
    p.add_argument('--plot_live', action="store_true")
    p.add_argument('--gradnorm_alpha', type=float, default=1.5)
    p.add_argument('--gaze_cons_w', type=float, default=0.2)
    p.add_argument('--plot_png_every', type=int, default=100)

    # Pupil loss weights / params
    p.add_argument('--pc_w_3d', type=float, default=1.0)
    p.add_argument('--pc_w_cons', type=float, default=0.3)
    p.add_argument('--pc_w_ellipse2d', type=float, default=0.2)
    p.add_argument('--pc_huber_delta_cm', type=float, default=0.5)
    return p.parse_args()


def get_backbone(backbone_name, weight_path, device):
    from backbone.repnext_utils import load_pretrained_repnext
    model = load_pretrained_repnext(backbone_name, weight_path)
    return model.to(device)


def iris_radius_tensor_from_subjects(dataset, subj_ids_flat, B, V, device):
    """
    Build [B*V] iris radius (cm) using dataset.attr_dict[subject]['iris_radius'].
    Fills small epsilon if missing to avoid degenerate geometry.
    """
    vals = []
    for s in subj_ids_flat:
        attrs = dataset.attr_dict.get(int(s), None)
        r = float(attrs.get('iris_radius', 0.0)) if attrs is not None else 0.0
        vals.append(max(r, 1e-3))  # clamp to epsilon
    t = torch.tensor(vals, dtype=torch.float32, device=device)  # [B*V]
    return t


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Subjects
    if args.split == "train":
        subject_ids = [f"subject{i}" for i in range(1, 46)]
    else:
        subject_ids = [f"subject{i}" for i in range(46, 56)]

    # Dataset / loader
    dataset = GazeGeneDataset(
        args.base_dir,
        subject_ids=subject_ids,
        samples_per_subject=args.samples_per_subject,
        transform=None,
        balance_attributes=['ethicity']
    )
    batch_sampler = MultiViewBatchSampler(dataset, batch_size=args.batch_size,
                                          balance_attributes=['ethicity'], shuffle=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                        num_workers=args.num_workers, pin_memory=True)

    # Backbone + Stage-1
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

    model = RayNetStage1(
        backbone=backbone,
        in_channels_list=in_channels_list,
        out_channels=256,
        ca_reduction=32,
        d_model=128,
        num_heads=4,
        grid=8,
        query_stride=4
    ).to(device)

    # ---- Tasks: HP, Gaze(vMF), Pupil (geometric) ----
    NUM_TASKS = 3
    TASK_IDX = {"hp": 0, "gaze": 1, "pc": 2}

    # GradNorm weights (learned)
    task_weights = nn.Parameter(torch.ones(NUM_TASKS, device=device), requires_grad=True)
    # Kendall log-variances (learned, optimized with model)
    log_vars = nn.Parameter(torch.zeros(NUM_TASKS, device=device), requires_grad=True)

    # Optimizers
    optimizer_model = optim.Adam(list(model.parameters()) + [log_vars], lr=args.lr)
    optimizer_weights = optim.Adam([task_weights], lr=args.lr)

    # Logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logfile = open(args.log_csv, "w", newline="")
    csv_writer = csv.DictWriter(logfile, fieldnames=[
        "epoch", "step", "batch_size",
        "hp_acc", "hp_cons",
        "gaze_acc_nll", "gaze_cons",
        "pc_acc", "pc_cons",
        "w_hp", "w_gaze", "w_pc",
        "s_hp", "s_gaze", "s_pc",
        "total_loss"
    ])
    csv_writer.writeheader()

    if args.plot_live:
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 7))
        plt.show(block=False)
        loss_hist = defaultdict(list)

    # GradNorm bookkeeping
    initial_losses = None
    alpha = args.gradnorm_alpha

    # Pick a shared parameter for GradNorm
    def pick_shared_param(net):
        if hasattr(net, "xattn") and hasattr(net.xattn, "q_proj"):
            return net.xattn.q_proj.weight
        return next(net.parameters())

    shared_param = pick_shared_param(model)

    step_global = 0
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(loader):
            step_global += 1

            # ---- Shapes / view grouping ----
            images = batch['img'].to(device)           # [B*V, C, H, W]
            B = images.shape[0] // 9
            V = 9

            # Subjects for iris radius
            subj_ids = batch['subject']
            if not torch.is_tensor(subj_ids):
                subj_ids = torch.tensor(subj_ids, dtype=torch.long)
            subj_ids = subj_ids.view(B, V)
            iris_r_cm = iris_radius_tensor_from_subjects(dataset, subj_ids.reshape(-1).tolist(), B, V, device)  # [B*V]

            # Intrinsics
            K = batch['intrinsic'].to(device).view(B, V, 3, 3)     # [B,V,3,3]
            K_flat = K.reshape(B * V, 3, 3)                         # [B*V,3,3]

            # GT
            hp_gt = batch['head_pose']['R'].to(device).view(B, V, 3, 3)
            gaze_vec_gt = batch['gaze']['gaze_C'].to(device).view(B, V, 3)
            gt_pupil_3d = batch['mesh']['pupil_center_3D'].to(device).view(B, V, 2, 3)
            eye_ctr_3d = batch['mesh']['eyeball_center_3D'].to(device).view(B, V, 2, 3)

            # Optional GT iris landmarks (2D) if you add them to dataset
            gt_iris_mesh_2d = None
            try:
                if 'iris_mesh_2D' in batch['mesh']:
                    gt_iris_mesh_2d = batch['mesh']['iris_mesh_2D'].to(device).view(B, V, 2, 100, 2)
            except Exception:
                gt_iris_mesh_2d = None

            # ---- Forward (per-view) ----
            out = model(images)  # Stage-1 outputs for B*V views

            # --- Head pose (6D) ---
            hp_pred = out["head_pose_6d"].view(B, V, 6)

            # --- Gaze vMF (mu,kappa) ---
            if "gaze" not in out:
                raise RuntimeError("Expected Stage-1 to return dict 'gaze' with keys ('mu','kappa').")
            gaze_mu = out["gaze"]["mu"].view(B, V, 3)
            gaze_k  = out["gaze"]["kappa"].view(B, V, 1)

            # --- Pupil (SOTA geometric) ---
            if "pupil" not in out:
                raise RuntimeError("SOTA pupil head required: expected out['pupil'] dict with ('ellipse','delta_cm', optional 'logvar').")
            pupil_pred = out["pupil"]
            # Ensure shapes are [B*V, ...]
            for key in ["ellipse", "delta_cm"]:
                if key not in pupil_pred:
                    raise RuntimeError(f"pupil dict missing key '{key}'.")
            # (loss accepts [B,2,6]/[B,2,1] and will expand across N views internally,
            #  but we let it handle that—just pass dict through)

            # ---- Per-task base losses ----
            hp_losses = multiview_headpose_losses(hp_pred, hp_gt)
            gaze_losses = multiview_gaze_vector_vmf_losses(
                {"mu": gaze_mu, "kappa": gaze_k},
                gaze_vec_gt,
                w_cons=args.gaze_cons_w
            )
            pc_losses = multiview_pupil_center_losses(
                pred=pupil_pred,
                gt_pupil_center_3d=gt_pupil_3d,
                intrinsics_K=K_flat,                 # [B*V,3,3]
                iris_radius_cm=iris_r_cm,           # [B*V]
                gt_iris_mesh_2d=gt_iris_mesh_2d,    # [B,V,2,100,2] or None
                w_3d=args.pc_w_3d,
                w_consistency=args.pc_w_cons,
                w_ellipse2d=args.pc_w_ellipse2d,
                huber_delta_cm=args.pc_huber_delta_cm,
            )

            # Combine accuracy + consistency per task
            L_hp   = hp_losses['accuracy']   + hp_losses['consistency']
            L_gaze = gaze_losses['accuracy'] + gaze_losses['consistency']
            L_pc   = pc_losses['accuracy']   + pc_losses['consistency']
            base_losses = torch.stack([L_hp, L_gaze, L_pc])  # [3]

            # Kendall uncertainty weighting
            s_vec = torch.stack([log_vars[0], log_vars[1], log_vars[2]])
            kendall_losses = torch.exp(-s_vec) * base_losses + s_vec

            # GradNorm – gradient norms wrt shared param
            G = []
            for i in range(NUM_TASKS):
                g_i = torch.autograd.grad(kendall_losses[i], shared_param,
                                          retain_graph=True, create_graph=False, allow_unused=False)[0]
                G.append(g_i.norm())
            G = torch.stack(G)

            # Phase 1: update model
            optimizer_model.zero_grad()
            model_loss = (task_weights.detach() * kendall_losses).sum()
            model_loss.backward()
            optimizer_model.step()

            if initial_losses is None:
                initial_losses = kendall_losses.detach()

            # GradNorm targets
            loss_ratios = kendall_losses.detach() / (initial_losses + 1e-8)
            avg_ratio = loss_ratios.mean()
            target = G.mean().detach() * (loss_ratios / (avg_ratio + 1e-8)) ** args.gradnorm_alpha

            # Phase 2: update task weights
            optimizer_weights.zero_grad()
            gradnorm_loss = F.l1_loss(task_weights * G, target)
            gradnorm_loss.backward()
            optimizer_weights.step()

            with torch.no_grad():
                task_weights.mul_(NUM_TASKS / (task_weights.sum() + 1e-8))

            # ---- Logging ----
            log = {
                "epoch": epoch, "step": step, "batch_size": args.batch_size,
                "hp_acc": float(hp_losses['accuracy']),
                "hp_cons": float(hp_losses['consistency']),
                "gaze_acc_nll": float(gaze_losses['accuracy']),
                "gaze_cons": float(gaze_losses['consistency']),
                "pc_acc": float(pc_losses['accuracy']),
                "pc_cons": float(pc_losses['consistency']),
                "w_hp": float(task_weights[TASK_IDX["hp"]].detach()),
                "w_gaze": float(task_weights[TASK_IDX["gaze"]].detach()),
                "w_pc": float(task_weights[TASK_IDX["pc"]].detach()),
                "s_hp": float(log_vars[0].detach()),
                "s_gaze": float(log_vars[1].detach()),
                "s_pc": float(log_vars[2].detach()),
                "total_loss": float(model_loss.detach()),
            }
            csv_writer.writerow(log)
            logfile.flush()

            # Live plot
            if args.plot_live:
                for k in ["hp_acc", "gaze_acc_nll", "pc_acc"]:
                    loss_hist[k].append(log[k])
                ax.clear()
                ax.plot(loss_hist["hp_acc"], label="HP acc")
                ax.plot(loss_hist["gaze_acc_nll"], label="Gaze NLL")
                ax.plot(loss_hist["pc_acc"], label="PC acc")
                ax.legend()
                ax.set_title(f"Epoch {epoch} Step {step} | Batch {args.batch_size}")
                ax.set_xlabel("Iterations")
                ax.set_ylabel("Loss")
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)
                if step_global % args.plot_png_every == 0:
                    try:
                        fig.savefig(os.path.join(args.checkpoint_dir, "training_plot.png"))
                    except Exception:
                        pass

            if (step + 1) % 10 == 0:
                print(
                    f"Epoch {epoch} | Step {step+1}/{len(loader)} | "
                    f"HP: {log['hp_acc']:.4f}/{log['hp_cons']:.4f} | "
                    f"Gaze NLL: {log['gaze_acc_nll']:.4f} Cons: {log['gaze_cons']:.4f} | "
                    f"PC: {log['pc_acc']:.4f}/{log['pc_cons']:.4f} | "
                    f"w: [{log['w_hp']:.2f},{log['w_gaze']:.2f},{log['w_pc']:.2f}] | "
                    f"s: [{log['s_hp']:.2f},{log['s_gaze']:.2f},{log['s_pc']:.2f}] | "
                    f"Total: {log['total_loss']:.4f}"
                )

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"stage1_epoch{epoch+1}.pth")
            torch.save({
                'model': model.state_dict(),
                'optimizer_model': optimizer_model.state_dict(),
                'optimizer_weights': optimizer_weights.state_dict(),
                'task_weights': task_weights.detach(),
                'log_vars': log_vars.detach(),
                'epoch': epoch,
                'args': vars(args)
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    logfile.close()
    print("Training complete.")

    if args.plot_live:
        plt.ioff()
        try:
            fig.savefig(os.path.join(args.checkpoint_dir, "training_plot_final.png"))
        except Exception:
            pass
        plt.show()


if __name__ == "__main__":
    main()
