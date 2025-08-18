"""
depth_from_iris.py
MediaPipe-style monocular depth from iris diameter.

Z (cm) = f_px * D_cm / s_px

- f_px: focal length in pixels (from camera intrinsics K[0,0], K[1,1])
- D_cm: true iris diameter in centimeters (identity parameter)
- s_px: measured iris diameter in pixels from predicted 2D iris landmarks

References:
  - MediaPipe Iris blog (Google Research): similar triangles for depth.
"""

from typing import Tuple, Optional

import torch
import torch.nn.functional as F


def diameter_from_opposites(ring2d_px: torch.Tensor) -> torch.Tensor:
    """
    Robust diameter using opposite points (i, i+N/2).
    Args:
        ring2d_px: [B, N, 2] ordered contour (N even)
    Returns:
        diam_px: [B] maximum opposite-point distance in pixels
    """
    B, N, _ = ring2d_px.shape
    assert N % 2 == 0, "Iris ring must have an even number of points"
    i = torch.arange(N // 2, device=ring2d_px.device)
    A = ring2d_px[:, i, :]           # [B, N/2, 2]
    Bp = ring2d_px[:, i + N // 2, :] # [B, N/2, 2]
    d = torch.linalg.vector_norm(A - Bp, dim=-1)  # [B, N/2]
    return d.max(dim=1).values + 1e-6


def diameter_from_pca(ring2d_px: torch.Tensor) -> torch.Tensor:
    """
    Fallback diameter via PCA major axis (approx).
    Args:
        ring2d_px: [B, N, 2]
    Returns:
        diam_px: [B]
    """
    B, N, _ = ring2d_px.shape
    # center
    mu = ring2d_px.mean(dim=1, keepdim=True)  # [B,1,2]
    X = ring2d_px - mu
    # covariance
    cov = torch.einsum('bni,bnj->bij', X, X) / (N - 1 + 1e-6)  # [B,2,2]
    # eigenvalues
    eigvals = []
    for b in range(B):
        w, _ = torch.linalg.eigh(cov[b])
        eigvals.append(w)
    W = torch.stack(eigvals, dim=0)  # [B,2], ascending
    # major axis length ≈ 2*sqrt(lambda_max)
    return 2.0 * torch.sqrt(W[:, 1].clamp_min(1e-6))


def depth_from_iris_cm(
    iris2d_px: torch.Tensor,
    fx_px: torch.Tensor,
    iris_diam_cm: torch.Tensor,
    method: str = "auto",
) -> torch.Tensor:
    """
    Analytic depth (centimeters) from iris diameter.

    Args:
        iris2d_px   : [B, 2, N, 2] per-eye 2D iris ring in pixels
        fx_px       : [B] or [B,2] focal length(s) in pixels
        iris_diam_cm: [B] or [B,2] ground-truth iris diameter(s) in cm
        method      : "auto" | "opposites" | "pca"

    Returns:
        depth_cm: [B] average of left/right eye depth
    """
    L = iris2d_px[:, 0]  # [B,N,2]
    R = iris2d_px[:, 1]  # [B,N,2]

    if method in ("auto", "opposites"):
        dL = diameter_from_opposites(L)
        dR = diameter_from_opposites(R)
    elif method == "pca":
        dL = diameter_from_pca(L)
        dR = diameter_from_pca(R)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Broadcast fx and D as needed
    if fx_px.ndim == 1:
        fxL, fxR = fx_px, fx_px
    else:
        fxL, fxR = fx_px[:, 0], fx_px[:, 1]

    if iris_diam_cm.ndim == 1:
        DL, DR = iris_diam_cm, iris_diam_cm
    else:
        DL, DR = iris_diam_cm[:, 0], iris_diam_cm[:, 1]

    zL = fxL * DL / dL
    zR = fxR * DR / dR
    return 0.5 * (zL + zR)
