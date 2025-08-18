# train.py
import argparse, os, csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

from dataset import GazeGeneDataset, MultiViewBatchSampler
from backbone.repnext_utils import load_pretrained_repnext
from iris_projection import iris_projection_loss, project_points
from iris_depth import depth_from_iris
from raynet import RayNet
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def parse_args():
    ap = argparse.ArgumentParser("FLAME-compliant RayNet (eye)")
    ap.add_argument('--base_dir', required=True)
    ap.add_argument('--backbone_name', default='repnext_m3')
    ap.add_argument('--weight_path', default='./repnext_m3_pretrained.pt')
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--samples_per_subject', type=int, default=None)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--checkpoint_dir', default='ckpts_eye')
    ap.add_argument('--log_csv', default='train_eye_proj.csv')
    ap.add_argument('--split', default='train', choices=['train','test'])
    ap.add_argument('--warmup_steps', type=int, default=5000)
    ap.add_argument('--eyeball_radius_cm', type=float, default=1.2)
    ap.add_argument('--iris_diameter_cm', type=float, default=1.17)
    return ap.parse_args()

def get_backbone(name, weight_path, device):
    m = load_pretrained_repnext(name, weight_path)
    return m.to(device)

def build_model(backbone, in_channels, args, device):
    return RayNet(
        backbone=backbone,
        in_channels_list=in_channels,
        n_iris_landmarks=100,
        eyeball_radius_cm=args.eyeball_radius_cm,
        iris_diameter_cm=args.iris_diameter_cm
    ).to(device)

def make_gt_iris_2d(gt_iris3d_LR, K):
    """
    gt_iris3d_LR: (B,2,N,3) camera-space 3D (GazeGene stores in camera coordinates)
    K: (B,3,3)
    Returns (B,2,N,2) pixels by pinhole projection
    """
    B, _, N, _ = gt_iris3d_LR.shape
    out = []
    for eye in [0,1]:
        P = gt_iris3d_LR[:, eye]  # (B,N,3)
        uvs = project_points(K, torch.eye(3, device=K.device).unsqueeze(0).repeat(B,1,1),
                                torch.zeros(B,3, device=K.device), P)
        out.append(uvs)
    return torch.stack(out, dim=1)  # (B,2,N,2)

def angular_loss(pred_dir, gt_dir, eps=1e-6):
    pred = pred_dir / (pred_dir.norm(dim=-1, keepdim=True) + eps)
    gt   = gt_dir   / (gt_dir.norm(dim=-1, keepdim=True) + eps)
    cosang = (pred * gt).sum(dim=-1).clamp(-1+1e-6, 1-1e-6)
    return torch.acos(cosang).mean()  # radians

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # split
    subjects = [f"subject{i}" for i in (range(1,46) if args.split=='train' else range(46,56))]

    # data
    ds = GazeGeneDataset(args.base_dir, subject_ids=subjects, samples_per_subject=args.samples_per_subject)
    sampler = MultiViewBatchSampler(ds, batch_size=args.batch_size, balance_attributes=None, shuffle=True)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)

    # backbone
    backbone_channels_dict = {
        'repnext_m0': [40, 80, 160, 320],
        'repnext_m1': [48, 96, 192, 384],
        'repnext_m2': [56, 112, 224, 448],
        'repnext_m3': [64, 128, 256, 512],
        'repnext_m4': [64, 128, 256, 512],
        'repnext_m5': [80, 160, 320, 640],
    }
    in_ch = backbone_channels_dict[args.backbone_name]
    backbone = get_backbone(args.backbone_name, args.weight_path, device)
    model = build_model(backbone, in_ch, args, device)

    optim_model = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(args.log_csv, 'w', newline='') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=[
            'epoch','step','batch',
            'L_iris2d','L_proj2d','L_pupil','L_gaze','L_depth','L_total',
            'median_s_px','depth_cm'
        ])
        writer.writeheader()

        global_step = 0
        model.train()
        for epoch in range(args.epochs):
            for step, batch in enumerate(loader):
                global_step += 1
                # prepare mini-batch
                imgs = batch['img'].to(device)            # (B*9,3,H,W)
                B9, _, H, W = imgs.shape
                B = B9 // 9
                imgs = imgs

                K   = batch['intrinsic'].to(device)       # (B*9,3,3)
                Rgt = batch['head_pose']['R'].to(device)  # (B*9,3,3)
                tgt = batch['head_pose']['t'].to(device)  # (B*9,3)

                # reshape to flat batch for our model; we treat every view as an independent sample
                out = model(imgs, K=K, head_pose_gt={'t': tgt}, image_size=(H, W),
                            global_step=global_step, warmup_steps=args.warmup_steps)

                # --- build 2D ground truth by projecting GT 3D iris (camera space) ---
                # dataset stores iris_mesh_3D as (B*9, 2, N, 3)
                gt_iris3d = batch['mesh']['iris_mesh_3D'].to(device)  # (B*9,2,N,3)
                gt_iris2d = make_gt_iris_2d(gt_iris3d, K)             # (B*9,2,N,2)

                # 2D iris head loss: compare predicted 2D landmarks to projected GT
                L_2d_left  = F.smooth_l1_loss(out['iris2d_L_px'], gt_iris2d[:, 0])
                L_2d_right = F.smooth_l1_loss(out['iris2d_R_px'], gt_iris2d[:, 1])
                L_iris2d = L_2d_left + L_2d_right

                # 3D->2D projection loss: project predicted 3D rings and compare to same 2D GT
                proj_losses = iris_projection_loss(
                    out['iris3d_L'], out['iris3d_R'], K,
                    gt_iris2d[:, 0], gt_iris2d[:, 1], R=None, t=None, w_proj=1.0
                )
                L_proj2d = proj_losses['total']

                # Pupil center loss (camera space)
                gt_pupil = batch['mesh']['pupil_center_3D'].to(device)  # (B*9,2,3)
                L_pupil = F.smooth_l1_loss(out['pupil_L'], gt_pupil[:, 0]) + \
                          F.smooth_l1_loss(out['pupil_R'], gt_pupil[:, 1])

                # Gaze direction loss (angle to GT visual axis if provided)
                if 'visual_axis_L' in batch['gaze'] and 'visual_axis_R' in batch['gaze']:
                    # average both eyes' visual axes as GT direction proxy
                    vL = batch['gaze']['visual_axis_L'].to(device)
                    vR = batch['gaze']['visual_axis_R'].to(device)
                    vGT = 0.5 * (vL + vR)
                    L_gaze = angular_loss(out['ray_dir'], vGT)
                else:
                    # fallback to gaze_C if provided
                    vGT = batch['gaze']['gaze_C'].to(device)
                    L_gaze = angular_loss(out['ray_dir'], vGT)

                # Depth from iris (MediaPipe). Use fx and clamp; warm-up detaches gradients through Z.
                fx = K[:, 0, 0]
                Z_cm, sL, sR = depth_from_iris(
                    fx_px=fx,
                    iris_diam_cm=args.iris_diameter_cm,
                    ring_L_px=out['iris2d_L_px'],
                    ring_R_px=out['iris2d_R_px'],
                    s_min_px=8.0, z_min_cm=20.0, z_max_cm=120.0,
                    detach_in_warmup=(global_step < args.warmup_steps)
                )
                # if dataset provides gaze depth, supervise lightly
                if 'gaze_depth' in batch['gaze']:
                    zGT = batch['gaze']['gaze_depth'].to(device)
                    L_depth = F.smooth_l1_loss(Z_cm, zGT)
                else:
                    L_depth = torch.tensor(0.0, device=device)

                # Total loss (weights can be tuned)
                L_total = 1.0 * L_iris2d + 1.0 * L_proj2d + 0.5 * L_pupil + 0.5 * L_gaze + 0.25 * L_depth

                optim_model.zero_grad()
                L_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim_model.step()

                # --- logging (convert to float) ---
                median_s = torch.median(0.5 * (sL + sR)).item()
                row = {
                    'epoch': int(epoch),
                    'step': int(step),
                    'batch': int(out['iris2d_L_px'].shape[0]),
                    'L_iris2d': float(L_iris2d.item()),
                    'L_proj2d': float(L_proj2d.item()),
                    'L_pupil':  float(L_pupil.item()),
                    'L_gaze':   float(L_gaze.item()),
                    'L_depth':  float(L_depth.item()) if isinstance(L_depth, torch.Tensor) else float(L_depth),
                    'L_total':  float(L_total.item()),
                    'median_s_px': float(median_s),
                    'depth_cm': float(Z_cm.mean().item()),
                }
                writer.writerow(row)
                fcsv.flush()

            # save checkpoint per epoch
            torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(
                args.checkpoint_dir, f'raynet_eye_epoch{epoch:03d}.pth'))

if __name__ == "__main__":
    main()
