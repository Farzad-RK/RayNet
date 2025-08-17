import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class EyeballReconstructionLoss(nn.Module):
    """
    Loss for 3D eyeball reconstruction
    Since we don't have GT eyeball mesh, we use constraints from iris and pupil positions
    """

    def __init__(self,
                 center_weight: float = 1.0,
                 radius_weight: float = 0.5,
                 consistency_weight: float = 1.0,
                 regularization_weight: float = 0.01):
        super().__init__()
        self.center_weight = center_weight
        self.radius_weight = radius_weight
        self.consistency_weight = consistency_weight
        self.regularization_weight = regularization_weight

    def forward(self, eyeball_left, eyeball_right, gt_eyeball_centers, iris_landmarks):
        """
        Compute eyeball reconstruction losses
        Args:
            eyeball_left/right: dict with predicted eyeball parameters
            gt_eyeball_centers: (B, 2, 3) ground truth eyeball centers from GazeGene
            iris_landmarks: dict with iris landmarks (for consistency check)
        """
        losses = {}
        batch_size = gt_eyeball_centers.shape[0]

        # Eyeball center loss
        pred_centers = torch.stack([eyeball_left['center'], eyeball_right['center']], dim=1)
        center_loss = F.l1_loss(pred_centers, gt_eyeball_centers)
        losses['center'] = self.center_weight * center_loss

        # Radius consistency (both eyes should have similar radius)
        radius_diff = torch.abs(eyeball_left['radius'] - eyeball_right['radius'])
        losses['radius_consistency'] = self.radius_weight * radius_diff.mean()

        # Expected radius range (10-14mm, normalized)
        radius_mean = (eyeball_left['radius'] + eyeball_right['radius']) / 2
        radius_penalty = torch.relu(10.0 - radius_mean) + torch.relu(radius_mean - 14.0)
        losses['radius_range'] = self.radius_weight * radius_penalty.mean()

        # Iris should lie on eyeball surface
        if iris_landmarks is not None:
            for eye, eyeball in [('left', eyeball_left), ('right', eyeball_right)]:
                iris_points = iris_landmarks[eye]
                eyeball_center = eyeball['center']
                eyeball_radius = eyeball['radius']

                # Distance from iris points to eyeball center
                distances = torch.norm(iris_points - eyeball_center.unsqueeze(1), dim=2)

                # Should be close to eyeball radius
                surface_error = torch.abs(distances - eyeball_radius.unsqueeze(1))
                losses[f'{eye}_surface'] = self.consistency_weight * surface_error.mean()

        # Regularization on shape parameters
        shape_reg = 0
        if 'shape_coeffs' in eyeball_left:
            shape_reg += torch.mean(eyeball_left['shape_coeffs'] ** 2)
        if 'shape_coeffs' in eyeball_right:
            shape_reg += torch.mean(eyeball_right['shape_coeffs'] ** 2)
        losses['shape_regularization'] = self.regularization_weight * shape_reg

        losses['total'] = sum(losses.values())
        return losses


class IrisReconstructionLoss(nn.Module):
    """
    Loss for 3D iris mesh reconstruction
    """

    def __init__(self,
                 landmark_weight: float = 2.0,
                 pupil_weight: float = 1.0,
                 structure_weight: float = 0.5,
                 symmetry_weight: float = 0.1):
        super().__init__()
        self.landmark_weight = landmark_weight
        self.pupil_weight = pupil_weight
        self.structure_weight = structure_weight
        self.symmetry_weight = symmetry_weight

    def forward(self, pred_iris, gt_iris, pred_pupil, gt_pupil):
        """
        Compute iris reconstruction losses
        Args:
            pred_iris: dict with 'left' and 'right' (B, 100, 3)
            gt_iris: (B, 2, 100, 3) ground truth from GazeGene
            pred_pupil: dict with 'left' and 'right' (B, 3)
            gt_pupil: (B, 2, 3) ground truth pupil centers
        """
        losses = {}
        batch_size = gt_iris.shape[0]

        # Reshape ground truth
        gt_iris_left = gt_iris[:, 0]  # (B, 100, 3)
        gt_iris_right = gt_iris[:, 1]  # (B, 100, 3)
        gt_pupil_left = gt_pupil[:, 0]  # (B, 3)
        gt_pupil_right = gt_pupil[:, 1]  # (B, 3)

        # Iris landmark loss
        left_landmark_loss = F.l1_loss(pred_iris['left'], gt_iris_left)
        right_landmark_loss = F.l1_loss(pred_iris['right'], gt_iris_right)
        losses['left_landmarks'] = self.landmark_weight * left_landmark_loss
        losses['right_landmarks'] = self.landmark_weight * right_landmark_loss

        # Pupil center loss
        left_pupil_loss = F.l1_loss(pred_pupil['left'], gt_pupil_left)
        right_pupil_loss = F.l1_loss(pred_pupil['right'], gt_pupil_right)
        losses['left_pupil'] = self.pupil_weight * left_pupil_loss
        losses['right_pupil'] = self.pupil_weight * right_pupil_loss

        # Structural consistency - iris points should form circular pattern around pupil
        for eye, iris, pupil in [('left', pred_iris['left'], pred_pupil['left']),
                                 ('right', pred_iris['right'], pred_pupil['right'])]:
            # Distance from each iris point to pupil center
            distances = torch.norm(iris - pupil.unsqueeze(1), dim=2)

            # Variance should be low (circular pattern)
            distance_variance = torch.var(distances, dim=1)
            losses[f'{eye}_structure'] = self.structure_weight * distance_variance.mean()

            # Also check that distances are reasonable (2-6mm typical iris radius)
            mean_distance = distances.mean(dim=1)
            distance_penalty = torch.relu(2.0 - mean_distance) + torch.relu(mean_distance - 6.0)
            losses[f'{eye}_radius'] = self.structure_weight * distance_penalty.mean()

        # Symmetry loss (optional)
        if self.symmetry_weight > 0:
            # Mirror left eye to match right eye coordinate system
            left_mirrored = pred_iris['left'].clone()
            left_mirrored[:, :, 0] *= -1  # Flip x-coordinate

            # Compute correspondence (not exact match, but similar structure)
            symmetry_loss = F.mse_loss(
                torch.cdist(left_mirrored, left_mirrored),
                torch.cdist(pred_iris['right'], pred_iris['right'])
            )
            losses['symmetry'] = self.symmetry_weight * symmetry_loss

        losses['total'] = sum(losses.values())
        return losses


class GazeDirectionLoss(nn.Module):
    """
    Loss for gaze direction estimation with optical and visual axes
    """

    def __init__(self,
                 optical_weight: float = 1.0,
                 visual_weight: float = 1.5,
                 gaze_weight: float = 2.0,
                 angular_scale: float = 180.0 / 3.14159):  # Convert to degrees for better scale
        super().__init__()
        self.optical_weight = optical_weight
        self.visual_weight = visual_weight
        self.gaze_weight = gaze_weight
        self.angular_scale = angular_scale

    def angular_error(self, pred_vec, gt_vec):
        """Compute angular error in degrees"""
        pred_norm = F.normalize(pred_vec, dim=-1)
        gt_norm = F.normalize(gt_vec, dim=-1)

        cos_sim = torch.clamp(torch.sum(pred_norm * gt_norm, dim=-1), -1.0, 1.0)
        angle_rad = torch.acos(cos_sim)
        angle_deg = angle_rad * self.angular_scale

        return angle_deg

    def forward(self, predictions, ground_truth):
        """
        Compute gaze direction losses
        Args:
            predictions: dict with optical_axis_*, visual_axis_*, gaze_vector
            ground_truth: dict with GT from GazeGene
        """
        losses = {}

        # Optical axis loss for each eye
        optical_left_error = self.angular_error(
            predictions['optical_axis_left'],
            ground_truth['optic_axis_L']
        )
        optical_right_error = self.angular_error(
            predictions['optical_axis_right'],
            ground_truth['optic_axis_R']
        )

        losses['optical_left'] = self.optical_weight * optical_left_error.mean()
        losses['optical_right'] = self.optical_weight * optical_right_error.mean()

        # Visual axis loss for each eye
        visual_left_error = self.angular_error(
            predictions['visual_axis_left'],
            ground_truth['visual_axis_L']
        )
        visual_right_error = self.angular_error(
            predictions['visual_axis_right'],
            ground_truth['visual_axis_R']
        )

        losses['visual_left'] = self.visual_weight * visual_left_error.mean()
        losses['visual_right'] = self.visual_weight * visual_right_error.mean()

        # Combined gaze vector loss
        gaze_error = self.angular_error(
            predictions['gaze_vector'],
            ground_truth['gaze_C']
        )
        losses['gaze'] = self.gaze_weight * gaze_error.mean()

        # Also add L1 loss for better gradient flow
        gaze_l1 = F.l1_loss(
            F.normalize(predictions['gaze_vector'], dim=-1),
            F.normalize(ground_truth['gaze_C'], dim=-1)
        )
        losses['gaze_l1'] = gaze_l1

        # Log mean angular errors for monitoring
        losses['mean_angular_error'] = gaze_error.mean()

        losses['total'] = sum([v for k, v in losses.items() if k not in ['mean_angular_error']])
        return losses


class EyeballRotationConsistencyLoss(nn.Module):
    """
    Loss to ensure eyeball rotation is consistent with gaze direction
    """

    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight

    def forward(self, eyeball_left, eyeball_right, gaze_vectors):
        """
        Ensure eyeball rotation aligns with gaze direction
        """
        losses = {}

        # The eyeball's forward direction (typically z-axis after rotation)
        # should roughly align with the optical axis
        left_forward = eyeball_left['rotation'][:, :, 2]  # z-axis of rotation matrix
        right_forward = eyeball_right['rotation'][:, :, 2]

        # Compute alignment with optical axes
        left_alignment = 1.0 - torch.sum(
            F.normalize(left_forward, dim=-1) *
            F.normalize(gaze_vectors['optical_axis_left'], dim=-1),
            dim=-1
        )
        right_alignment = 1.0 - torch.sum(
            F.normalize(right_forward, dim=-1) *
            F.normalize(gaze_vectors['optical_axis_right'], dim=-1),
            dim=-1
        )

        losses['left_rotation_consistency'] = self.weight * left_alignment.mean()
        losses['right_rotation_consistency'] = self.weight * right_alignment.mean()

        # Ensure rotations are valid (orthogonal matrices)
        left_orth_loss = torch.mean(
            torch.abs(torch.matmul(eyeball_left['rotation'], eyeball_left['rotation'].transpose(-1, -2))
                      - torch.eye(3, device=eyeball_left['rotation'].device))
        )
        right_orth_loss = torch.mean(
            torch.abs(torch.matmul(eyeball_right['rotation'], eyeball_right['rotation'].transpose(-1, -2))
                      - torch.eye(3, device=eyeball_right['rotation'].device))
        )

        losses['left_orthogonality'] = 0.1 * left_orth_loss
        losses['right_orthogonality'] = 0.1 * right_orth_loss

        losses['total'] = sum(losses.values())
        return losses


class RayNetMultiViewLoss(nn.Module):
    """
    Multi-view consistency loss for RayNet
    """

    def __init__(self,
                 point_weight: float = 2.0,
                 depth_weight: float = 1.0,
                 triangulation_weight: float = 1.0):
        super().__init__()
        self.point_weight = point_weight
        self.depth_weight = depth_weight
        self.triangulation_weight = triangulation_weight

    def forward(self, predictions, ground_truth, batch_size, n_views=9):
        """
        Multi-view consistency losses
        Args:
            predictions: model outputs
            ground_truth: GT data
            batch_size: original batch size
            n_views: number of camera views
        """
        losses = {}

        # Reshape for multi-view
        ray_origins = predictions['ray_origin'].view(batch_size, n_views, 3)
        ray_directions = predictions['ray_direction'].view(batch_size, n_views, 3)
        gaze_depths = predictions['gaze_depth'].view(batch_size, n_views)
        gaze_points_gt = ground_truth['gaze_point'].view(batch_size, n_views, 3)

        # Predicted 3D gaze points from rays
        gaze_points_pred = ray_origins + gaze_depths.unsqueeze(-1) * ray_directions

        # Point reconstruction loss
        point_loss = F.l1_loss(gaze_points_pred, gaze_points_gt)
        losses['point'] = self.point_weight * point_loss

        # Depth estimation loss
        gt_depths = torch.norm(gaze_points_gt - ray_origins, dim=-1)
        depth_loss = F.l1_loss(gaze_depths, gt_depths)
        losses['depth'] = self.depth_weight * depth_loss

        # Multi-view triangulation consistency
        # All views should converge to similar 3D point
        if n_views > 1:
            # Compute pairwise ray intersections
            intersection_errors = []
            for i in range(n_views):
                for j in range(i + 1, n_views):
                    # Find closest point between rays from view i and j
                    o1, d1 = ray_origins[:, i], ray_directions[:, i]
                    o2, d2 = ray_origins[:, j], ray_directions[:, j]

                    # Compute closest points
                    w = o1 - o2
                    a = torch.sum(d1 * d1, dim=-1, keepdim=True)
                    b = torch.sum(d1 * d2, dim=-1, keepdim=True)
                    c = torch.sum(d2 * d2, dim=-1, keepdim=True)
                    d = torch.sum(d1 * w, dim=-1, keepdim=True)
                    e = torch.sum(d2 * w, dim=-1, keepdim=True)

                    denom = a * c - b * b
                    denom = torch.clamp(torch.abs(denom), min=1e-8)

                    s = torch.clamp((b * e - c * d) / denom, min=0)
                    t = torch.clamp((a * e - b * d) / denom, min=0)

                    p1 = o1 + s * d1
                    p2 = o2 + t * d2

                    # Distance between closest points (should be small)
                    error = torch.norm(p1 - p2, dim=-1)
                    intersection_errors.append(error)

            if intersection_errors:
                triangulation_error = torch.stack(intersection_errors).mean()
                losses['triangulation'] = self.triangulation_weight * triangulation_error

        losses['total'] = sum(losses.values())
        return losses


class CombinedRayNetLoss(nn.Module):
    """
    Combined loss for eye-focused RayNet training
    """

    def __init__(self,
                 eyeball_weight: float = 1.0,
                 iris_weight: float = 2.0,
                 gaze_weight: float = 2.0,
                 rotation_weight: float = 0.5,
                 multiview_weight: float = 1.0):
        super().__init__()

        self.eyeball_loss = EyeballReconstructionLoss()
        self.iris_loss = IrisReconstructionLoss()
        self.gaze_loss = GazeDirectionLoss()
        self.rotation_loss = EyeballRotationConsistencyLoss()
        self.multiview_loss = RayNetMultiViewLoss()

        self.weights = {
            'eyeball': eyeball_weight,
            'iris': iris_weight,
            'gaze': gaze_weight,
            'rotation': rotation_weight,
            'multiview': multiview_weight
        }

    def forward(self, predictions, ground_truth, is_multiview=False):
        """
        Compute all losses
        Args:
            predictions: model outputs
            ground_truth: GazeGene ground truth data
            is_multiview: whether this is a multi-view batch
        """
        all_losses = {}

        # Eyeball reconstruction loss
        eyeball_losses = self.eyeball_loss(
            predictions['eyeball_left'],
            predictions['eyeball_right'],
            ground_truth['mesh']['eyeball_center_3D'],
            predictions['iris_landmarks']
        )
        for k, v in eyeball_losses.items():
            all_losses[f'eyeball_{k}'] = self.weights['eyeball'] * v

        # Iris reconstruction loss
        iris_losses = self.iris_loss(
            predictions['iris_landmarks'],
            ground_truth['mesh']['iris_mesh_3D'],
            predictions['pupil_centers'],
            ground_truth['mesh']['pupil_center_3D']
        )
        for k, v in iris_losses.items():
            all_losses[f'iris_{k}'] = self.weights['iris'] * v

        # Gaze direction loss
        gaze_losses = self.gaze_loss(predictions, ground_truth['gaze'])
        for k, v in gaze_losses.items():
            all_losses[f'gaze_{k}'] = self.weights['gaze'] * v

        # Eyeball rotation consistency loss
        rotation_losses = self.rotation_loss(
            predictions['eyeball_left'],
            predictions['eyeball_right'],
            predictions
        )
        for k, v in rotation_losses.items():
            all_losses[f'rotation_{k}'] = self.weights['rotation'] * v

        # Multi-view consistency loss
        if is_multiview:
            batch_size = predictions['ray_origin'].shape[0] // 9
            multiview_losses = self.multiview_loss(
                predictions,
                ground_truth,
                batch_size,
                n_views=9
            )
            for k, v in multiview_losses.items():
                all_losses[f'multiview_{k}'] = self.weights['multiview'] * v

        # Total loss
        all_losses['total'] = sum([v for k, v in all_losses.items() if 'total' not in k])

        return all_losses