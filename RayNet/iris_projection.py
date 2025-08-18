# iris_projection.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def project_points(K, R, t, P):
    """
    K: (B,3,3)
    R: (B,3,3)  world(head)->camera
    t: (B,3)
    P: (B,N,3)  points in world/head space?  Here we assume camera space already -> just K[x/z, y/z, 1]
       If points are camera-space, set R=I, t=0 and call this.
    Returns: (B,N,2) pixels
    """
    if P.dim() == 4:
        B, V, N, _ = P.shape
        P = P.view(B * V, N, 3)
        K = K.repeat_interleave(V, dim=0)
        R = R.repeat_interleave(V, dim=0)
        t = t.repeat_interleave(V, dim=0)

    B, N, _ = P.shape
    # if P is camera space, skip transform. Otherwise uncomment:
    # Pc = torch.einsum('bij,bnj->bni', R, P) + t[:, None, :]
    Pc = P
    X, Y, Z = Pc[..., 0], Pc[..., 1], torch.clamp(Pc[..., 2], min=1e-6)
    fx, fy = K[:, 0, 0], K[:, 1, 1]
    cx, cy = K[:, 0, 2], K[:, 1, 2]
    u = fx[:, None] * (X / Z) + cx[:, None]
    v = fy[:, None] * (Y / Z) + cy[:, None]
    return torch.stack([u, v], dim=-1)

def iris_projection_loss(pred_iris3d_L, pred_iris3d_R, K, gt_iris2d_L, gt_iris2d_R, R=None, t=None, w_proj=1.0):
    """
    Project predicted 3D iris to pixels and compare with 2D GT.
    pred_iris3d_*: (B,N,3) in camera space
    K: (B,3,3)
    gt_iris2d_*: (B,N,2) pixels
    Returns: dict with '2d_L', '2d_R', 'total'
    """
    if R is None:  # assume already camera space
        R = torch.eye(3, device=K.device).unsqueeze(0).repeat(K.size(0), 1, 1)
    if t is None:
        t = torch.zeros(K.size(0), 3, device=K.device)

    proj_L = project_points(K, R, t, pred_iris3d_L)  # (B,N,2)
    proj_R = project_points(K, R, t, pred_iris3d_R)

    L_loss = F.smooth_l1_loss(proj_L, gt_iris2d_L)
    R_loss = F.smooth_l1_loss(proj_R, gt_iris2d_R)
    total = w_proj * (L_loss + R_loss)

    return {"2d_L": L_loss, "2d_R": R_loss, "total": total}
