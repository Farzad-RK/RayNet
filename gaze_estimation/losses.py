import torch
import torch.nn as nn


def rot6d_to_rotmat(d6):
    a1 = d6[:, 0:3]
    a2 = d6[:, 3:6]
    b1 = nn.functional.normalize(a1, dim=-1)
    b2 = nn.functional.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)

def extract_gaze_vector(rot_mat):
    return rot_mat[:, :, 2]

def gaze_loss(pred_6d, target_3d):
    pred_rotmat = rot6d_to_rotmat(pred_6d)
    pred_gaze = extract_gaze_vector(pred_rotmat)
    pred_gaze = nn.functional.normalize(pred_gaze, dim=-1)
    target_3d = nn.functional.normalize(target_3d, dim=-1)
    if target_3d.shape[1] == 2:
        # Make sure zeros tensor is on the same device!
        zeros = torch.zeros_like(target_3d[:, :1], device=target_3d.device)
        target_3d = torch.cat([target_3d, zeros], dim=1)
    return 1 - (pred_gaze * target_3d).sum(dim=1).mean()

def angular_error(pred_6d, target_3d):
    pred_rotmat = rot6d_to_rotmat(pred_6d)
    pred_gaze = extract_gaze_vector(pred_rotmat)
    pred_gaze = nn.functional.normalize(pred_gaze, dim=-1)
    target_3d = nn.functional.normalize(target_3d, dim=-1)
    if target_3d.shape[1] == 2:
        zeros = torch.zeros_like(target_3d[:, :1], device=target_3d.device)
        target_3d = torch.cat([target_3d, zeros], dim=1)
    dot = torch.clamp((pred_gaze * target_3d).sum(dim=1), -1.0, 1.0)
    return torch.acos(dot) * 180 / torch.pi