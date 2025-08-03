
import torch
import torch.nn as nn

def orthogonalize_rotmat(rotmat):
    """
    Projects a batch of 3x3 matrices to the closest rotation matrix using SVD.

    Args:
        rotmat: [B, 3, 3]
    Returns:
        [B, 3, 3] (rotation matrices)
    """
    # SVD-based projection
    U, _, Vt = torch.linalg.svd(rotmat)
    R = torch.matmul(U, Vt)
    # Enforce det(R) = +1 (rotation, not reflection)
    det = torch.det(R)
    det_sign = det.sign().unsqueeze(-1).unsqueeze(-1)
    Vt = Vt * det_sign
    R = torch.matmul(U, Vt)
    return R