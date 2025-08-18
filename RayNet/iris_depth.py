# iris_depth.py
import torch

def _diameter_px_from_ring(pts_px):
    """
    pts_px: (B,N,2) pixel coordinates of iris ring
    Robust major-axis diameter estimate using PCA and clamp against outliers.
    Returns: (B,) diameter_px
    """
    B, N, _ = pts_px.shape
    mean = pts_px.mean(dim=1, keepdim=True)                   # (B,1,2)
    X = pts_px - mean                                         # (B,N,2)
    # covariance per batch
    C = torch.einsum('bni,bnj->bij', X, X) / (N - 1 + 1e-9)   # (B,2,2)
    eigvals, eigvecs = torch.linalg.eigh(C)                   # ascending
    # major axis direction
    major = eigvecs[:, :, 1]                                  # (B,2)
    # project points onto major axis
    proj = torch.einsum('bni,bi->bn', X, major)               # (B,N)
    d = proj.max(dim=1).values - proj.min(dim=1).values       # (B,)
    return torch.clamp(d, min=1.0)                            # avoid zeros

def depth_from_iris(fx_px, iris_diam_cm, ring_L_px, ring_R_px,
                    s_min_px=8.0, z_min_cm=20.0, z_max_cm=120.0,
                    detach_in_warmup=False):
    """
    MediaPipe-style: Z = f * D / s, averaged across eyes.
    fx_px: (B,) or scalar
    iris_diam_cm: scalar
    ring_*_px: (B,N,2)
    Returns: (B,) depth_cm, plus (B,) diameters for diagnostics
    """
    dL = _diameter_px_from_ring(ring_L_px)
    dR = _diameter_px_from_ring(ring_R_px)
    dL = torch.clamp(dL, min=s_min_px)
    dR = torch.clamp(dR, min=s_min_px)

    if isinstance(fx_px, (float, int)):
        fx_px = torch.tensor(fx_px, device=ring_L_px.device).expand_as(dL)

    zL = fx_px * iris_diam_cm / dL
    zR = fx_px * iris_diam_cm / dR
    Z = 0.5 * (zL + zR)
    Z = torch.clamp(Z, z_min_cm, z_max_cm)

    if detach_in_warmup:
        Z = Z.detach()
    return Z, dL, dR
