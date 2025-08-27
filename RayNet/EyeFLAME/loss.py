# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (
    normalize_vector, fit_plane_svd, geodesic_distance_rotmat,
    compute_plane_distances, ortho6d_to_rotmat
)


class GeometricUncertaintyWeighting(nn.Module):
    """
    Learn task-specific uncertainty to automatically balance loss terms
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall & Gal 2017)
    """

    def __init__(self, num_loss_terms=8):
        super().__init__()
        # Learn log variance for each loss term (prevents negative weights)
        self.log_vars = nn.Parameter(torch.zeros(num_loss_terms))
        self.loss_names = [
            'eyeball_l1', 'pupil_l1', 'iris_100_l1', 'head_gaze_l1',
            'optical_axis', 'perfect_planarity', 'perfect_circle', 'projection_2d'
        ]

    def forward(self, losses):
        """
        Args:
            losses: dict with loss terms in centimeters

        Returns:
            total_loss, weighting_info
        """
        loss_values = [losses[name] for name in self.loss_names]

        weighted_losses = []
        total_loss = 0

        for i, loss_val in enumerate(loss_values):
            # Uncertainty weighting: weight = 1/(2*σ²), regularization = log(σ)
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss_val + self.log_vars[i]
            weighted_losses.append(weighted_loss)
            total_loss += weighted_loss

        return total_loss, {
            'weighted_losses': dict(zip(self.loss_names, weighted_losses)),
            'learned_weights': torch.exp(-self.log_vars),  # Actual learned weights
            'uncertainties': torch.exp(self.log_vars)  # Learned uncertainties
        }


class GeometricScaleAwareWeighting:
    """
    Weight losses based on geometric scales and measurement units
    All measurements in centimeters as per GazeGene
    """

    def __init__(self):
        # Typical scales in centimeters for eye anatomy
        self.geometric_scales = {
            'eyeball_center': 0.1,  # ±1mm accuracy for eyeball center
            'pupil_center': 0.05,  # ±0.5mm accuracy for pupil center
            'iris_landmarks': 0.02,  # ±0.2mm accuracy for iris contour
            'head_gaze': 0.01,  # ±0.01 for unit vector (angular)
            'optical_axis': 0.01,  # ±0.01 rad ≈ 0.6° for axis direction
            'planarity': 1e-4,  # ±0.1mm planarity error (synthetic should be perfect)
            'circularity': 1e-4,  # ±0.1mm radius consistency
            'projection_2d': 2.0  # ±2 pixels in 2D (depends on camera resolution)
        }

    def compute_scale_normalized_weights(self, losses):
        """
        Normalize losses by their expected geometric scales
        """
        normalized_weights = {}

        # 3D reconstruction losses (in cm)
        normalized_weights['eyeball_l1'] = 1.0 / self.geometric_scales['eyeball_center']
        normalized_weights['pupil_l1'] = 1.0 / self.geometric_scales['pupil_center']
        normalized_weights['iris_100_l1'] = 1.0 / self.geometric_scales['iris_landmarks']
        normalized_weights['head_gaze_l1'] = 1.0 / self.geometric_scales['head_gaze']

        # Angular losses (unitless, but scale by expected precision)
        normalized_weights['optical_axis'] = 1.0 / self.geometric_scales['optical_axis']

        # Perfect geometric constraints (should be near-zero for synthetic)
        normalized_weights['perfect_planarity'] = 1.0 / self.geometric_scales['planarity']
        normalized_weights['perfect_circle'] = 1.0 / self.geometric_scales['circularity']

        # 2D projection losses (in pixels, convert to cm using camera params)
        normalized_weights['projection_2d'] = 1.0 / self.geometric_scales['projection_2d']

        return normalized_weights


class EyeFLAMELoss(nn.Module):
    """
    Complete geometric-aware loss function for EyeFLAME model
    Handles K=4 joints and 100 iris landmarks per eye
    """

    def __init__(self, use_uncertainty_weighting=True, use_scale_weighting=True):
        super().__init__()

        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.use_scale_weighting = use_scale_weighting

        # Initialize uncertainty weighting
        if use_uncertainty_weighting:
            self.uncertainty_weighter = GeometricUncertaintyWeighting(num_loss_terms=8)

        # Initialize scale weighting
        if use_scale_weighting:
            self.scale_weighter = GeometricScaleAwareWeighting()

        # Individual loss components
        self.geodesic_loss = GeodesicLoss()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, ground_truth, gazegene_subject_params, camera_params=None):
        """
        Compute complete geometric-aware loss

        Args:
            predictions: Dict from model forward pass
            ground_truth: Dict with GT annotations from GazeGene
            gazegene_subject_params: Subject-specific parameters
            camera_params: Camera parameters (optional)

        Returns:
            total_loss, detailed_losses
        """

        # === PRIMARY RECONSTRUCTION LOSSES (in cm) ===

        raw_losses = {}

        # 1. 3D Eyeball Centers (L1 in cm)
        raw_losses['eyeball_l1'] = self.l1_loss(
            predictions['eyeball_centers'],  # [B, 2, 3]
            ground_truth['eyeball_center_3D']  # [B, 2, 3]
        )

        # 2. 3D Pupil Centers (L1 in cm)
        raw_losses['pupil_l1'] = self.l1_loss(
            predictions['pupil_centers'],  # [B, 2, 3]
            ground_truth['pupil_center_3D']  # [B, 2, 3]
        )

        # 3. 100 Iris Landmarks per eye (L1 in cm) - Full contour
        raw_losses['iris_100_l1'] = self.l1_loss(
            predictions['iris_landmarks_100'],  # [B, 200, 3]
            ground_truth['iris_mesh_3D']  # [B, 200, 3]
        )

        # 4. Head Gaze Direction (L1 for unit vectors)
        raw_losses['head_gaze_l1'] = self.l1_loss(
            predictions['head_gaze_direction'],  # [B, 3]
            ground_truth['gaze_C']  # [B, 3]
        )

        # 5. Optical/Visual Axes (Angular loss)
        raw_losses['optical_axis'] = self.optical_axis_loss(
            predictions, ground_truth
        )

        # === PERFECT GEOMETRIC CONSTRAINTS (for synthetic data) ===

        # 6. Perfect Planarity (iris landmarks should lie exactly in plane)
        raw_losses['perfect_planarity'] = self.iris_perfect_planarity_loss(
            predictions['iris_landmarks_100']  # [B, 200, 3]
        )

        # 7. Perfect Circle Constraint (iris should form perfect circles)
        raw_losses['perfect_circle'] = self.iris_perfect_circle_loss(
            predictions['iris_landmarks_100'],  # [B, 200, 3]
            gazegene_subject_params['iris_radius']  # [B, 1]
        )

        # === 2D-3D CONSISTENCY ===

        # 8. 2D Projection Consistency (if camera params available)
        if camera_params is not None and 'projections_2d' in predictions:
            raw_losses['projection_2d'] = self.projection_consistency_loss(
                predictions, ground_truth, camera_params
            )
        else:
            raw_losses['projection_2d'] = torch.tensor(0.0, device=predictions['eyeball_centers'].device)

        # === JOINT CONSISTENCY LOSSES ===
        # These are additional losses not included in the main weighting scheme
        joint_losses = self.joint_consistency_losses(
            predictions, ground_truth
        )

        # Combine joint losses into a single term for weighting
        joint_consistency_total = sum(joint_losses.values()) if joint_losses else torch.tensor(0.0, device=predictions[
            'eyeball_centers'].device)

        # === GEOMETRIC-AWARE LOSS BALANCING ===

        # Only use the main 8 loss terms for uncertainty/scale weighting
        main_losses = {
            'eyeball_l1': raw_losses['eyeball_l1'],
            'pupil_l1': raw_losses['pupil_l1'],
            'iris_100_l1': raw_losses['iris_100_l1'],
            'head_gaze_l1': raw_losses['head_gaze_l1'],
            'optical_axis': raw_losses['optical_axis'],
            'perfect_planarity': raw_losses['perfect_planarity'],
            'perfect_circle': raw_losses['perfect_circle'],
            'projection_2d': raw_losses['projection_2d']
        }

        if self.use_uncertainty_weighting:
            uncertainty_loss, uncertainty_info = self.uncertainty_weighter(main_losses)
        else:
            uncertainty_loss = sum(main_losses.values())
            uncertainty_info = {'learned_weights': None, 'uncertainties': None}

        if self.use_scale_weighting:
            scale_weights = self.scale_weighter.compute_scale_normalized_weights(main_losses)
            scale_loss = sum(main_losses[key] * scale_weights[key] for key in main_losses.keys())
        else:
            scale_loss = sum(main_losses.values())
            scale_weights = None

        # Final loss combines weighted main losses + joint consistency losses
        if self.use_uncertainty_weighting and self.use_scale_weighting:
            total_loss = 0.7 * uncertainty_loss + 0.3 * scale_loss + 0.1 * joint_consistency_total
        elif self.use_uncertainty_weighting:
            total_loss = uncertainty_loss + 0.1 * joint_consistency_total
        elif self.use_scale_weighting:
            total_loss = scale_loss + 0.1 * joint_consistency_total
        else:
            total_loss = sum(main_losses.values()) + 0.1 * joint_consistency_total

        # Combine all losses for logging - ensure all values are scalars
        all_losses = {**raw_losses, **joint_losses}

        # Convert multi-element tensors to scalars for logging
        def ensure_scalar(v):
            if torch.is_tensor(v):
                if v.numel() == 1:
                    return v
                else:
                    return v.mean()  # Convert multi-element to scalar
            return v

        # Convert uncertainty weights and scale weights to scalars for logging
        uncertainty_weights_scalar = None
        if uncertainty_info.get('learned_weights') is not None:
            weights = uncertainty_info['learned_weights']
            uncertainty_weights_scalar = weights.mean() if torch.is_tensor(weights) else weights

        scale_weights_scalar = None
        if scale_weights is not None:
            if isinstance(scale_weights, dict):
                # Take mean of all scale weights
                scale_values = list(scale_weights.values())
                scale_weights_scalar = sum(scale_values) / len(scale_values) if scale_values else 0.0
            else:
                scale_weights_scalar = scale_weights.mean() if torch.is_tensor(scale_weights) else scale_weights

        return total_loss, {
            'raw_losses': {k: ensure_scalar(v) for k, v in all_losses.items()},
            'uncertainty_weights_mean': uncertainty_weights_scalar,  # Scalar version
            'scale_weights_mean': scale_weights_scalar,  # Scalar version
            'uncertainty_loss': ensure_scalar(uncertainty_loss),
            'scale_loss': ensure_scalar(scale_loss),
            'joint_consistency_total': ensure_scalar(joint_consistency_total),
            'num_loss_terms': len(main_losses),  # For reference
        }

    def optical_axis_loss(self, predictions, ground_truth):
        """
        Compute optical and visual axis alignment losses
        """
        # Optical axes loss
        pred_optical = predictions['optical_axes']  # [B, 2, 3]
        gt_optical = torch.stack([
            ground_truth['optic_axis_L'],  # [B, 3]
            ground_truth['optic_axis_R']  # [B, 3]
        ], dim=1)  # [B, 2, 3]

        optical_loss = torch.mean(1 - torch.sum(pred_optical * gt_optical, dim=-1))

        # Visual axes loss
        pred_visual = predictions['visual_axes']  # [B, 2, 3]
        gt_visual = torch.stack([
            ground_truth['visual_axis_L'],  # [B, 3]
            ground_truth['visual_axis_R']  # [B, 3]
        ], dim=1)  # [B, 2, 3]

        visual_loss = torch.mean(1 - torch.sum(pred_visual * gt_visual, dim=-1))

        return optical_loss + visual_loss

    def iris_perfect_planarity_loss(self, iris_landmarks_200):
        """
        Perfect planarity constraint for 100 iris landmarks per eye (in cm)

        Args:
            iris_landmarks_200: [B, 200, 3] iris landmarks (100 per eye)
        """
        batch_size = iris_landmarks_200.shape[0]
        planarity_loss = 0

        for eye_idx in range(2):  # Left, Right
            start_idx = eye_idx * 100
            end_idx = start_idx + 100
            landmarks = iris_landmarks_200[:, start_idx:end_idx]  # [B, 100, 3]

            for b in range(batch_size):
                # Fit plane through 100 landmarks for this batch item
                plane_normal, plane_d = fit_plane_svd(landmarks[b])  # landmarks[b] is [100, 3]

                # Distance of all points to plane (should be ~0 for synthetic)
                distances_to_plane = compute_plane_distances(
                    landmarks[b], plane_normal, plane_d
                )  # [100]

                # RMS planarity error in cm
                planarity_loss += torch.sqrt(torch.mean(distances_to_plane ** 2))

        return planarity_loss / (batch_size * 2)  # Average over batch and eyes

    def iris_perfect_circle_loss(self, iris_landmarks_200, iris_radius_cm):
        """
        Perfect circle constraint for 100 iris landmarks per eye (in cm)

        Args:
            iris_landmarks_200: [B, 200, 3] iris landmarks
            iris_radius_cm: [B, 1] iris radius in cm
        """
        batch_size = iris_landmarks_200.shape[0]
        circle_loss = 0

        for eye_idx in range(2):  # Left, Right
            start_idx = eye_idx * 100
            end_idx = start_idx + 100
            landmarks = iris_landmarks_200[:, start_idx:end_idx]  # [B, 100, 3]

            # Compute centroid for each batch item
            centroids = torch.mean(landmarks, dim=1)  # [B, 3]

            # Distances from centroid should equal iris_radius_cm
            distances = torch.norm(landmarks - centroids.unsqueeze(1), dim=-1)  # [B, 100]

            # RMS radius error in cm
            radius_errors = distances - iris_radius_cm.squeeze(-1).unsqueeze(-1)  # [B, 100]
            circle_loss += torch.sqrt(torch.mean(radius_errors ** 2))

        return circle_loss / 2  # Average over eyes

    def projection_consistency_loss(self, predictions, ground_truth, camera_params):
        """
        2D-3D projection consistency loss
        """
        if 'projections_2d' not in predictions:
            return torch.tensor(0.0, device=predictions['eyeball_centers'].device)

        proj_2d = predictions['projections_2d']
        total_proj_loss = 0

        # Eyeball centers 2D projection
        if 'eyeball_center_2D' in ground_truth:
            total_proj_loss += self.l1_loss(
                proj_2d['eyeball_centers_2d'],  # [B, 2, 2]
                ground_truth['eyeball_center_2D']  # [B, 2, 2]
            )

        # Pupil centers 2D projection
        if 'pupil_center_2D' in ground_truth:
            total_proj_loss += self.l1_loss(
                proj_2d['pupil_centers_2d'],  # [B, 2, 2]
                ground_truth['pupil_center_2D']  # [B, 2, 2]
            )

        # Iris landmarks 2D projection
        if 'iris_mesh_2D' in ground_truth:
            total_proj_loss += self.l1_loss(
                proj_2d['iris_landmarks_2d'],  # [B, 200, 2]
                ground_truth['iris_mesh_2D']  # [B, 200, 2]
            )

        return total_proj_loss

    def joint_consistency_losses(self, predictions, ground_truth):
        """
        Ensure joint parameters are consistent with expected relationships
        """
        joint_losses = {}

        if 'raw_joint_params' in predictions:
            joint_params = predictions['raw_joint_params']

            # Head gaze joint should match GazeGene 'gaze_C'
            joint_losses['head_gaze_consistency'] = self.l1_loss(
                joint_params['θ_head_gaze'],  # [B, 3]
                ground_truth['gaze_C']  # [B, 3]
            )

            # Rotation matrix orthogonality constraints
            if 'θ_head_pose' in joint_params:
                joint_losses['head_pose_orthogonal'] = self.rotation_orthogonality_loss(
                    joint_params['θ_head_pose']  # [B, 3, 3]
                )

            if 'θ_left_eyeball' in joint_params:
                joint_losses['left_eyeball_orthogonal'] = self.rotation_orthogonality_loss(
                    joint_params['θ_left_eyeball']  # [B, 3, 3]
                )

            if 'θ_right_eyeball' in joint_params:
                joint_losses['right_eyeball_orthogonal'] = self.rotation_orthogonality_loss(
                    joint_params['θ_right_eyeball']  # [B, 3, 3]
                )

        return joint_losses

    def rotation_orthogonality_loss(self, rotation_matrices):
        """
        Ensure rotation matrices are orthogonal (R @ R.T = I)

        Args:
            rotation_matrices: [B, 3, 3] rotation matrices

        Returns:
            orthogonality loss
        """
        batch_size = rotation_matrices.shape[0]
        identity = torch.eye(3, device=rotation_matrices.device).unsqueeze(0).expand(batch_size, -1, -1)

        # Compute R @ R.T
        should_be_identity = torch.bmm(rotation_matrices, rotation_matrices.transpose(1, 2))

        # Loss is deviation from identity
        orthogonal_loss = self.mse_loss(should_be_identity, identity)

        return orthogonal_loss


class GeodesicLoss(nn.Module):
    """
    Geodesic Loss for 3x3 rotation matrices (batch-wise).
    Computes the geodesic distance (angle, in radians) between rotation matrices.
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, m1, m2):
        """
        Args:
            m1: torch.Tensor, [B, 3, 3], predicted rotation matrices
            m2: torch.Tensor, [B, 3, 3], ground-truth rotation matrices
        Returns:
            mean geodesic loss (scalar)
        """
        return geodesic_distance_rotmat(m1, m2, eps=self.eps)


def multiview_joint_losses(pred_joint_params, gt_data, num_cameras=9):
    """
    Multi-view consistency losses for joint parameters

    Args:
        pred_joint_params: Dict with predicted joint parameters for multiple views
        gt_data: Ground truth data for multiple views
        num_cameras: Number of cameras (9 for GazeGene)

    Returns:
        dict: Multi-view consistency losses
    """

    geo_loss = GeodesicLoss()
    l1_loss = nn.L1Loss()

    losses = {}

    # Head pose consistency across views
    if 'head_pose_6d' in pred_joint_params:
        head_pose_6d = pred_joint_params['head_pose_6d']  # [B, num_cameras, 6]
        gt_head_rotmats = gt_data['head_R_mat']  # [B, num_cameras, 3, 3]

        # Convert predicted 6D to rotation matrices
        B, N, _ = head_pose_6d.shape
        pred_head_rotmats = ortho6d_to_rotmat(head_pose_6d.reshape(-1, 6)).reshape(B, N, 3, 3)

        # Per-view accuracy
        losses['head_pose_accuracy'] = geo_loss(
            pred_head_rotmats.reshape(-1, 3, 3),
            gt_head_rotmats.reshape(-1, 3, 3)
        )

        # Consistency: distance to mean prediction
        mean_pred = pred_head_rotmats.mean(dim=1)  # [B, 3, 3]
        losses['head_pose_consistency'] = geo_loss(
            pred_head_rotmats.reshape(-1, 3, 3),
            mean_pred.unsqueeze(1).expand(-1, N, -1, -1).reshape(-1, 3, 3)
        )

    # Head gaze consistency across views
    if 'head_gaze' in pred_joint_params:
        head_gaze = pred_joint_params['head_gaze']  # [B, num_cameras, 3]
        gt_head_gaze = gt_data['gaze_C']  # [B, num_cameras, 3]

        # Per-view accuracy
        losses['head_gaze_accuracy'] = l1_loss(head_gaze, gt_head_gaze)

        # Consistency: all views should predict similar gaze direction
        mean_gaze = head_gaze.mean(dim=1)  # [B, 3]
        losses['head_gaze_consistency'] = l1_loss(
            head_gaze,
            mean_gaze.unsqueeze(1).expand(-1, num_cameras, -1)
        )

    # Eyeball consistency (should be similar across views for same subject)
    for eye_side in ['left', 'right']:
        param_key = f'{eye_side}_eyeball_6d'
        if param_key in pred_joint_params:
            eyeball_6d = pred_joint_params[param_key]  # [B, num_cameras, 6]

            # Convert to rotation matrices
            B, N, _ = eyeball_6d.shape
            eyeball_rotmats = ortho6d_to_rotmat(eyeball_6d.reshape(-1, 6)).reshape(B, N, 3, 3)

            # Consistency across views
            mean_eyeball = eyeball_rotmats.mean(dim=1)  # [B, 3, 3]
            losses[f'{eye_side}_eyeball_consistency'] = geo_loss(
                eyeball_rotmats.reshape(-1, 3, 3),
                mean_eyeball.unsqueeze(1).expand(-1, N, -1, -1).reshape(-1, 3, 3)
            )

    return losses


def iris_landmark_quality_losses(predictions, ground_truth):
    """
    Specific quality losses for 100 iris landmarks per eye
    """
    iris_landmarks = predictions['iris_landmarks_100']  # [B, 200, 3]
    gt_iris = ground_truth['iris_mesh_3D']  # [B, 200, 3]

    losses = {}

    # Landmark ordering consistency (neighbors should be close)
    losses['iris_smoothness'] = iris_smoothness_loss(iris_landmarks)

    # Contour completeness (landmarks should form closed contours)
    losses['iris_closure'] = iris_closure_loss(iris_landmarks)

    # Scale consistency (iris should maintain proper size)
    losses['iris_scale'] = iris_scale_consistency_loss(iris_landmarks, ground_truth)

    return losses


def iris_smoothness_loss(iris_landmarks):
    """
    Ensure adjacent iris landmarks are smoothly connected
    """
    smoothness_loss = 0

    for eye_idx in range(2):  # Left, Right
        start_idx = eye_idx * 100
        landmarks = iris_landmarks[:, start_idx:start_idx + 100]  # [B, 100, 3]

        # Compute distances between adjacent landmarks (circular)
        next_landmarks = torch.roll(landmarks, shifts=-1, dims=1)  # [B, 100, 3]
        edge_lengths = torch.norm(landmarks - next_landmarks, dim=-1)  # [B, 100]

        # Smoothness: variance in edge lengths should be small
        smoothness_loss += torch.var(edge_lengths, dim=1).mean()  # Average over batch

    return smoothness_loss / 2  # Average over eyes


def iris_closure_loss(iris_landmarks):
    """
    Ensure iris contours form closed loops
    """
    closure_loss = 0

    for eye_idx in range(2):  # Left, Right
        start_idx = eye_idx * 100
        landmarks = iris_landmarks[:, start_idx:start_idx + 100]  # [B, 100, 3]

        # Distance between first and last landmark (should be small for closed contour)
        first_landmark = landmarks[:, 0]  # [B, 3]
        last_landmark = landmarks[:, -1]  # [B, 3]

        closure_distance = torch.norm(first_landmark - last_landmark, dim=-1)  # [B]
        closure_loss += closure_distance.mean()

    return closure_loss / 2  # Average over eyes


def iris_scale_consistency_loss(iris_landmarks, ground_truth):
    """
    Ensure iris maintains consistent scale relative to other eye structures

    Args:
        iris_landmarks: [B, 200, 3] iris landmarks (100 per eye)
        ground_truth: Dict containing subject parameters and eyeball info

    Returns:
        scale consistency loss
    """
    device = iris_landmarks.device
    batch_size = iris_landmarks.shape[0]

    # Extract subject-specific parameters
    expected_iris_radius = ground_truth.get('iris_radius')  # [B, 1] or [B]
    expected_eyeball_radius = ground_truth.get('eyeball_radius')  # [B, 1] or [B]
    eyeball_centers = ground_truth.get('eyeball_center_3D')  # [B, 2, 3]

    if expected_iris_radius is None or eyeball_centers is None:
        return torch.tensor(0.0, device=device)

    # Ensure proper shapes
    if expected_iris_radius.dim() == 2:
        expected_iris_radius = expected_iris_radius.squeeze(-1)  # [B]
    if expected_eyeball_radius is not None and expected_eyeball_radius.dim() == 2:
        expected_eyeball_radius = expected_eyeball_radius.squeeze(-1)  # [B]

    total_scale_loss = 0.0

    for eye_idx in range(2):  # Left, Right
        start_idx = eye_idx * 100
        eye_landmarks = iris_landmarks[:, start_idx:start_idx + 100]  # [B, 100, 3]
        eye_center = eyeball_centers[:, eye_idx]  # [B, 3]

        # Compute actual iris centroid and radius for this eye
        iris_centroid = torch.mean(eye_landmarks, dim=1)  # [B, 3]
        distances_to_centroid = torch.norm(eye_landmarks - iris_centroid.unsqueeze(1), dim=-1)  # [B, 100]
        actual_iris_radius = torch.mean(distances_to_centroid, dim=1)  # [B]

        # 1. Iris radius should match subject-specific parameter
        radius_consistency_loss = torch.mean(torch.abs(actual_iris_radius - expected_iris_radius))
        total_scale_loss += radius_consistency_loss

        # 2. Iris-to-eyeball scale ratio should be anatomically reasonable
        if expected_eyeball_radius is not None:
            iris_to_eyeball_ratio = actual_iris_radius / expected_eyeball_radius
            # Typical iris-to-eyeball ratio is around 0.05-0.07 (iris ~0.6cm, eyeball ~12cm)
            expected_ratio = expected_iris_radius / expected_eyeball_radius
            ratio_consistency_loss = torch.mean(torch.abs(iris_to_eyeball_ratio - expected_ratio))
            total_scale_loss += 0.5 * ratio_consistency_loss

        # 3. Distance from iris centroid to eyeball center should be reasonable
        iris_to_eyeball_distance = torch.norm(iris_centroid - eye_center, dim=-1)  # [B]
        # This should be approximately eyeball_radius (iris on surface)
        if expected_eyeball_radius is not None:
            surface_distance_loss = torch.mean(torch.abs(iris_to_eyeball_distance - expected_eyeball_radius))
            total_scale_loss += 0.3 * surface_distance_loss

    # 4. Left and right iris should have consistent scales
    left_landmarks = iris_landmarks[:, :100]  # [B, 100, 3]
    right_landmarks = iris_landmarks[:, 100:]  # [B, 100, 3]

    left_centroid = torch.mean(left_landmarks, dim=1)  # [B, 3]
    right_centroid = torch.mean(right_landmarks, dim=1)  # [B, 3]

    left_distances = torch.norm(left_landmarks - left_centroid.unsqueeze(1), dim=-1)  # [B, 100]
    right_distances = torch.norm(right_landmarks - right_centroid.unsqueeze(1), dim=-1)  # [B, 100]

    left_radius = torch.mean(left_distances, dim=1)  # [B]
    right_radius = torch.mean(right_distances, dim=1)  # [B]

    # Left and right iris should have similar radii
    lr_consistency_loss = torch.mean(torch.abs(left_radius - right_radius))
    total_scale_loss += 0.4 * lr_consistency_loss

    return total_scale_loss / 2  # Average over eyes