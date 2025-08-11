# pupil_center/intrinsics.py
# Utilities to map camera intrinsics through crop/resize/flip into the model's pixel space.

import os
import torch

INTR_DEBUG = os.environ.get("PUPIL_DEBUG", "0") not in ("0", "", "false", "False", "no", "No")

@torch.no_grad()
def scale_intrinsics_for_crop_resize(
    K: torch.Tensor,
    src_hw: torch.Tensor,
    crop_xywh: torch.Tensor = None,
    dst_hw: torch.Tensor = None,
    hflip: torch.Tensor = None,
):
    """
    Vectorized mapping of intrinsics into the model's image space.

    Args:
        K          : [N,3,3] intrinsics for the *source* image (pixels).
        src_hw     : [N,2] (H_src, W_src) of the source image (pixels).
        crop_xywh  : [N,4] optional ROI in source coords (x0,y0,w,h). If None, use full image.
        dst_hw     : [N,2] (H_dst, W_dst) after resize (the tensor fed to the model). If None, keep ROI size.
        hflip      : [N]   bool tensor; if True, a horizontal flip is applied after resize.

    Returns:
        K_eff      : [N,3,3] intrinsics in the model's pixel coordinates.
    """
    assert K.ndim == 3 and K.shape[1:] == (3,3), "K must be [N,3,3]"
    N = K.shape[0]
    device = K.device

    if crop_xywh is None:
        crop_xywh = torch.stack([torch.zeros(N, device=device),
                                 torch.zeros(N, device=device),
                                 src_hw[:,1].to(device),
                                 src_hw[:,0].to(device)], dim=-1)  # [N,4]
    if dst_hw is None:
        dst_hw = torch.stack([crop_xywh[:,3], crop_xywh[:,2]], dim=-1)  # [N,2] (H_dst,W_dst)
    if hflip is None:
        hflip = torch.zeros(N, dtype=torch.bool, device=device)

    x0, y0, w_roi, h_roi = crop_xywh.unbind(-1)
    H_dst, W_dst = dst_hw.unbind(-1)

    # 1) Translate principal point into the crop
    Kc = K.clone()
    Kc[:, 0, 2] -= x0
    Kc[:, 1, 2] -= y0

    # 2) Scale from ROI size to destination size
    sx = (W_dst / w_roi).clamp_min(1e-12)
    sy = (H_dst / h_roi).clamp_min(1e-12)
    Kc[:, 0, 0] *= sx
    Kc[:, 1, 1] *= sy
    Kc[:, 0, 2] *= sx
    Kc[:, 1, 2] *= sy

    # 3) Optional horizontal flip in destination space
    # u' = (W_dst - 1) - u -> multiply fx by -1 and move cx accordingly.
    if hflip.any():
        Kc[hflip, 0, 0] *= -1.0
        Kc[hflip, 0, 2] = (W_dst[hflip] - 1.0) - Kc[hflip, 0, 2]

    # 4) Zero skew and last row for safety
    Kc[:, 0, 1] = 0.0
    Kc[:, 2, :] = torch.tensor([0.0, 0.0, 1.0], device=device)

    if INTR_DEBUG:
        fx0, fy0 = K[:,0,0], K[:,1,1]
        fx1, fy1 = Kc[:,0,0], Kc[:,1,1]
        print(f"[Kwarp] src fx/fy mean: {float(fx0.mean()):.2f}/{float(fy0.mean()):.2f} "
              f"-> eff fx/fy mean: {float(fx1.mean()):.2f}/{float(fy1.mean()):.2f} "
              f"| sx mean {float(sx.mean()):.4f} sy mean {float(sy.mean()):.4f}")
        cx1, cy1 = Kc[:,0,2], Kc[:,1,2]
        print(f"[Kwarp] eff cx/cy mean: {float(cx1.mean()):.2f}/{float(cy1.mean()):.2f} "
              f"| dst W/H mean: {float(W_dst.float().mean()):.1f}/{float(H_dst.float().mean()):.1f}")

    return Kc
