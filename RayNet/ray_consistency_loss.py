import torch
import torch.nn.functional as F

def multiview_ray_consistency_loss(origins, directions, gaze_depths, gaze_points_pred):
    """
    Ray consistency loss: how well does the predicted ray (origin + d * direction) match the gaze point prediction,
    and how consistent are reconstructed gaze points across all views.

    Args:
        origins:         [B, N, 3] - predicted origins of rays (pupil centers)
        directions:      [B, N, 3] - predicted, normalized gaze directions
        gaze_depths:     [B, N]    - predicted depths for each ray
        gaze_points_pred:[B, N, 3] - predicted gaze point (direct regression, per view)
    Returns:
        dict: {
            "reconstruction": loss between (origin + depth * direction) and predicted gaze_point,
            "consistency": loss between all reconstructed gaze points and their mean,
            "total": weighted sum of above
        }
    """

    # [B, N, 3]: reconstruct the gaze point from the predicted ray
    gaze_point_from_ray = origins + gaze_depths.unsqueeze(-1) * directions

    # 1. Ray reconstruction loss: how close is reconstructed point to directly predicted point?
    ray_reconstruction_loss = F.mse_loss(gaze_point_from_ray, gaze_points_pred)

    # 2. Ray consistency loss: are all reconstructed points in the batch aligned?
    mean_recon = gaze_point_from_ray.mean(dim=1, keepdim=True)  # [B, 1, 3]
    ray_consistency_loss = F.mse_loss(gaze_point_from_ray, mean_recon.expand_as(gaze_point_from_ray))

    # 3. Optionally, add consistency between origins, directions, and depths themselves
    # (For more advanced constraint, but usually the two above suffice.)

    # 4. Combine with adjustable weights
    total = ray_reconstruction_loss + ray_consistency_loss

    return {
        "reconstruction": ray_reconstruction_loss,
        "consistency": ray_consistency_loss,
        "total": total
    }
