

import torch
import torch.nn.functional as F

def ortho6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D representation to 3×3 rotation matrices via Gram–Schmidt.
    x: [..., 6] tensor
    Returns: [..., 3, 3] rotation matrices
    """
    # flatten batch dims
    orig_shape = x.shape[:-1]
    x = x.view(-1, 6)

    a1 = x[:, 0:3]            # first 3 dims
    a2 = x[:, 3:6]            # last 3 dims

    # Normalize a1 to get the first basis vector
    b1 = F.normalize(a1, dim=1)  # shape [B,3]

    # Make a2 orthogonal to b1, then normalize
    dot12 = (b1 * a2).sum(dim=1, keepdim=True)       # [B,1]
    proj  = dot12 * b1                               # projection of a2 onto b1
    ortho = a2 - proj                                 # remove component along b1
    b2    = F.normalize(ortho, dim=1)                # second basis

    # Third basis is cross product (guaranteed orthogonal)
    b3 = torch.cross(b1, b2, dim=1)                  # [B,3]

    # Stack into rotation matrix
    R = torch.stack([b1, b2, b3], dim=-1)            # [B,3,3]

    # restore original batch shape
    return R.view(*orig_shape, 3, 3)
