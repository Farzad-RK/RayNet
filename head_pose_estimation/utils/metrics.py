import numpy as np
import torch

def calculate_mae(predictions, targets):
    """
    Calculate Mean Absolute Error (MAE) in degrees.
    
    Args:
        predictions: Predicted rotation matrices or vectors (N, 3, 3) or (N, 6)
        targets: Ground truth rotation matrices or vectors (N, 3, 3) or (N, 6)
        
    Returns:
        float: MAE in degrees
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Convert to rotation matrices if needed
    if predictions.shape[-1] == 6:  # 6D rotation representation
        predictions = _rotation_6d_to_matrix(torch.from_numpy(predictions)).numpy()
    if targets.shape[-1] == 6:  # 6D rotation representation
        targets = _rotation_6d_to_matrix(torch.from_numpy(targets)).numpy()
    
    # Calculate angular distance
    R_dot = np.einsum('nij,njk->nik', predictions, targets.transpose(0, 2, 1))
    trace = np.trace(R_dot, axis1=1, axis2=2)
    trace = np.clip((trace - 1) / 2, -1, 1)  # Ensure valid range for arccos
    angular_distance = np.arccos(trace) * 180 / np.pi
    
    return np.mean(angular_distance)

def calculate_accuracy(predictions, targets, threshold=5.0):
    """
    Calculate accuracy within a threshold.
    
    Args:
        predictions: Predicted rotation matrices or vectors
        targets: Ground truth rotation matrices or vectors
        threshold: Angular threshold in degrees
        
    Returns:
        float: Accuracy percentage
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Convert to rotation matrices if needed
    if predictions.shape[-1] == 6:  # 6D rotation representation
        predictions = _rotation_6d_to_matrix(torch.from_numpy(predictions)).numpy()
    if targets.shape[-1] == 6:  # 6D rotation representation
        targets = _rotation_6d_to_matrix(torch.from_numpy(targets)).numpy()
    
    # Calculate angular distance
    R_dot = np.einsum('nij,njk->nik', predictions, targets.transpose(0, 2, 1))
    trace = np.trace(R_dot, axis1=1, axis2=2)
    trace = np.clip((trace - 1) / 2, -1, 1)  # Ensure valid range for arccos
    angular_distance = np.arccos(trace) * 180 / np.pi
    
    # Calculate accuracy
    accuracy = np.mean(angular_distance <= threshold) * 100
    return accuracy

def _rotation_6d_to_matrix(rotation_6d):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    
    Args:
        rotation_6d: Tensor of shape (..., 6)
        
    Returns:
        Tensor of shape (..., 3, 3)
    """
    a1, a2 = rotation_6d[..., :3], rotation_6d[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)
