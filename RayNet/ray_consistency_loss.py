import torch
import torch.nn.functional as F

def multiview_ray_consistency_loss(origins, directions, gaze_points_from_ray):
    """
    Simplified ray consistency loss across multiple views.

    Args:
        origins: [B, N, 3] - pupil centers (origins of rays).
        directions: [B, N, 3] - normalized gaze directions.
        gaze_points_from_ray: [B, N, 3] - gaze points reconstructed from ray equation.

    Returns:
        scalar: ray consistency loss.
    """
    B, N, _ = origins.shape

    # Compute the mean gaze point reconstructed from rays across all views
    mean_gaze_point = gaze_points_from_ray.mean(dim=1, keepdim=True)  # [B, 1, 3]

    # Consistency: reconstructed gaze points should be close to the mean across views
    gaze_point_consistency = F.mse_loss(
        gaze_points_from_ray,
        mean_gaze_point.expand_as(gaze_points_from_ray)
    )

    return gaze_point_consistency
