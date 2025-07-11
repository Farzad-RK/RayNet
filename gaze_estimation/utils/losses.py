import torch
import torch.nn as nn
import torch.nn.functional as F

def rot6d_to_rotmat(d6):
    """Convert 6D rotation representation to 3x3 rotation matrix."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def extract_gaze_vector(rot_mat):
    """Extract gaze vector from rotation matrix."""
    return rot_mat[..., :, 2]

def gaze_loss(pred_6d, target_3d):
    """
    Compute gaze loss between predicted 6D rotation and target 3D gaze vector.
    
    Args:
        pred_6d: Predicted 6D rotation (batch_size, 6)
        target_3d: Target 3D gaze vector (batch_size, 2 or 3)
        
    Returns:
        torch.Tensor: Loss value
    """
    pred_rotmat = rot6d_to_rotmat(pred_6d)
    pred_gaze = extract_gaze_vector(pred_rotmat)
    pred_gaze = F.normalize(pred_gaze, dim=-1)
    target_3d = F.normalize(target_3d, dim=-1)
    
    # Handle 2D gaze vectors (add zero z-coordinate)
    if target_3d.shape[-1] == 2:
        zeros = torch.zeros_like(target_3d[..., :1], device=target_3d.device)
        target_3d = torch.cat([target_3d, zeros], dim=-1)
        
    return 1 - (pred_gaze * target_3d).sum(dim=-1).mean()

def angular_error(pred_6d, target_3d):
    """
    Compute angular error between predicted and target gaze directions.
    
    Args:
        pred_6d: Predicted 6D rotation (batch_size, 6)
        target_3d: Target 3D gaze vector (batch_size, 2 or 3)
        
    Returns:
        torch.Tensor: Angular error in degrees (batch_size,)
    """
    pred_rotmat = rot6d_to_rotmat(pred_6d)
    pred_gaze = extract_gaze_vector(pred_rotmat)
    pred_gaze = F.normalize(pred_gaze, dim=-1)
    target_3d = F.normalize(target_3d, dim=-1)
    
    # Handle 2D gaze vectors (add zero z-coordinate)
    if target_3d.shape[-1] == 2:
        zeros = torch.zeros_like(target_3d[..., :1], device=target_3d.device)
        target_3d = torch.cat([target_3d, zeros], dim=-1)
        
    dot = torch.clamp((pred_gaze * target_3d).sum(dim=-1), -1.0, 1.0)
    return torch.acos(dot) * 180 / torch.pi
