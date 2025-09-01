# model_depth_aware.py - COMPLETE FIXED VERSION
# EyeFLAME with proper depth handling and broadcasting fixes

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EyeFLAME_DepthAware(nn.Module):
    """
    EyeFLAME with Weak Perspective Projection - FIXED VERSION

    Key insight: Instead of predicting absolute 3D coordinates,
    predict normalized coordinates + scale factor (weak perspective model)
    This resolves the depth ambiguity issue.
    """

    def __init__(self, in_channels=256, hidden_dim=128, reduction=32):
        super().__init__()

        # Global feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # === WEAK PERSPECTIVE PARAMETERS ===
        self.weak_perspective_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # scale, tx, ty
        )

        # === NORMALIZED 3D STRUCTURE PREDICTION ===
        self.eyeball_centers_normalized = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 6)  # 2 eyes × 3 coords (normalized)
        )

        self.gaze_direction_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        self.optical_axes_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # 2 eyes × 3D direction
        )

        self.iris_params_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8)  # 2 eyes × (3D normal + radius_scale)
        )

        # Initialize weak perspective to reasonable defaults
        self._init_weak_perspective()

    def _init_weak_perspective(self):
        """Initialize weak perspective parameters to reasonable defaults"""
        wp_last = self.weak_perspective_head[-1]
        nn.init.zeros_(wp_last.weight)
        wp_last.bias.data = torch.tensor([1.0, 0.0, 0.0])

    def forward(self, fused_features, subject_params=None, camera_params=None):
        """
        Forward pass with weak perspective projection
        """
        batch_size = fused_features.shape[0]
        device = fused_features.device

        # Extract global features
        global_feats = self.global_pool(fused_features)
        global_feats = global_feats.view(batch_size, -1)

        # === WEAK PERSPECTIVE PARAMETERS ===
        weak_persp = self.weak_perspective_head(global_feats)  # [B, 3]
        scale = torch.sigmoid(weak_persp[:, 0]) * 2.0  # Scale factor [0, 2]
        translation_2d = weak_persp[:, 1:3]  # 2D translation in image plane

        # === NORMALIZED PREDICTIONS ===

        # 1. Normalized eyeball centers (as if at reference depth)
        eyeball_centers_norm = self.eyeball_centers_normalized(global_feats)  # [B, 6]
        eyeball_centers_norm = eyeball_centers_norm.view(batch_size, 2, 3)  # [B, 2, 3]

        # 2. Gaze direction (scale-invariant)
        gaze_direction = self.gaze_direction_head(global_feats)  # [B, 3]
        gaze_direction = F.normalize(gaze_direction, dim=-1)  # Unit vector

        # 3. Optical axes (scale-invariant)
        optical_axes = self.optical_axes_head(global_feats)  # [B, 6]
        optical_axes = optical_axes.view(batch_size, 2, 3)  # [B, 2, 3]
        optical_axes = F.normalize(optical_axes, dim=-1)  # Unit vectors

        # 4. Iris parameters
        iris_params = self.iris_params_head(global_feats)  # [B, 8]
        iris_normals = iris_params[:, :6].view(batch_size, 2, 3)  # [B, 2, 3]
        iris_normals = F.normalize(iris_normals, dim=-1)
        iris_radius_scales = torch.sigmoid(iris_params[:, 6:8])  # [B, 2] in [0,1]

        # === APPLY SCALE TO GET ACTUAL 3D COORDINATES ===

        # Reference depth from camera (5 meters = 500cm as per paper)
        reference_depth = 500.0

        # Scale normalized predictions to actual depth
        actual_depth = reference_depth * scale.unsqueeze(-1)  # [B, 1]

        # Transform eyeball centers to camera coordinates
        eyeball_centers_3d = eyeball_centers_norm.clone()
        eyeball_centers_3d[:, :, 2] = actual_depth.unsqueeze(1).expand(-1, 2, -1).squeeze(-1)  # Set Z
        eyeball_centers_3d[:, :, :2] *= scale.unsqueeze(-1).unsqueeze(-1)  # Scale X,Y

        # Add 2D translation in image plane
        eyeball_centers_3d[:, :, 0] += translation_2d[:, 0].unsqueeze(-1)
        eyeball_centers_3d[:, :, 1] += translation_2d[:, 1].unsqueeze(-1)

        # === GENERATE IRIS LANDMARKS ===

        iris_landmarks_100 = self.generate_iris_landmarks_scaled(
            eyeball_centers_3d, optical_axes, iris_normals,
            iris_radius_scales, subject_params, batch_size, device
        )

        # Pupil centers (center of iris)
        pupil_centers = torch.stack([
            torch.mean(iris_landmarks_100[:, :100], dim=1),  # Left
            torch.mean(iris_landmarks_100[:, 100:], dim=1)  # Right
        ], dim=1)  # [B, 2, 3]

        # === 2D PROJECTION WITH WEAK PERSPECTIVE ===

        projections_2d = None
        if camera_params is not None:
            projections_2d = self.weak_perspective_projection(
                eyeball_centers_3d, pupil_centers, iris_landmarks_100,
                scale, translation_2d, camera_params
            )

        # === VISUAL AXES ===

        visual_axes = self.apply_kappa(optical_axes, subject_params)

        return {
            'eyeball_centers': eyeball_centers_3d,  # [B, 2, 3] in CCS
            'pupil_centers': pupil_centers,  # [B, 2, 3] in CCS
            'iris_landmarks_100': iris_landmarks_100,  # [B, 200, 3] in CCS
            'optical_axes': optical_axes,  # [B, 2, 3]
            'visual_axes': visual_axes,  # [B, 2, 3]
            'head_gaze_direction': gaze_direction,  # [B, 3]
            'projections_2d': projections_2d,
            'weak_perspective': {
                'scale': scale,  # [B]
                'translation_2d': translation_2d,  # [B, 2]
                'normalized_centers': eyeball_centers_norm  # [B, 2, 3]
            }
        }

    def create_orthonormal_basis_robust(self, normal):
        """
        Create orthonormal basis vectors u, v given normal vector
        Fixed version with proper broadcasting

        Args:
            normal: [B, 3] batch of normal vectors (should be normalized)

        Returns:
            u: [B, 3] first orthonormal vector perpendicular to normal
            v: [B, 3] second orthonormal vector perpendicular to both normal and u
        """
        batch_size = normal.shape[0]
        device = normal.device

        # Choose a non-parallel vector for each normal
        temp = torch.zeros_like(normal)
        temp[:, 0] = 1.0

        # Check which normals are too parallel to [1, 0, 0]
        dot_x = torch.abs(normal[:, 0])

        # For those that are too parallel, use [0, 1, 0] instead
        mask = dot_x > 0.9
        temp[mask, :] = 0.0
        temp[mask, 1] = 1.0

        # Gram-Schmidt to create u perpendicular to normal
        dot_temp_normal = torch.sum(temp * normal, dim=-1, keepdim=True)  # [B, 1]
        u = temp - dot_temp_normal * normal  # [B, 3]
        u = F.normalize(u, dim=-1)  # [B, 3]

        # v = normal × u (cross product)
        v = torch.cross(normal, u, dim=-1)  # [B, 3]
        v = F.normalize(v, dim=-1)  # [B, 3]

        return u, v

    def generate_circle_on_plane(self, center, normal, radius, num_points, batch_size, device):
        """
        Generate circle points on a plane - FIXED version with proper broadcasting
        """
        # Ensure radius is properly shaped
        if not isinstance(radius, torch.Tensor):
            radius = torch.tensor(radius, device=device)
        if radius.dim() == 0:
            radius = radius.unsqueeze(0).expand(batch_size)
        elif radius.dim() == 1 and radius.shape[0] == 1:
            radius = radius.expand(batch_size)

        # Create orthonormal basis using robust method
        u, v = self.create_orthonormal_basis_robust(normal)  # [B, 3], [B, 3]

        # Generate angles for circle points
        angles = torch.linspace(0, 2 * np.pi, num_points, device=device)
        angles = angles.unsqueeze(0).expand(batch_size, -1)  # [B, num_points]

        cos_angles = torch.cos(angles)  # [B, num_points]
        sin_angles = torch.sin(angles)  # [B, num_points]

        # Generate points on circle with proper broadcasting
        center_expanded = center.unsqueeze(1)  # [B, 1, 3]
        u_expanded = u.unsqueeze(1)  # [B, 1, 3]
        v_expanded = v.unsqueeze(1)  # [B, 1, 3]
        radius_expanded = radius.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        cos_expanded = cos_angles.unsqueeze(-1)  # [B, num_points, 1]
        sin_expanded = sin_angles.unsqueeze(-1)  # [B, num_points, 1]

        points = (center_expanded +
                  radius_expanded * cos_expanded * u_expanded +
                  radius_expanded * sin_expanded * v_expanded)  # [B, num_points, 3]

        return points

    def generate_iris_landmarks_scaled(self, eyeball_centers, optical_axes,
                                       iris_normals, iris_radius_scales,
                                       subject_params, batch_size, device):
        """Generate iris landmarks with proper scaling - FIXED version"""

        # Get base iris radius from subject params
        if subject_params and 'iris_radius' in subject_params:
            base_iris_radius = subject_params['iris_radius']
            if base_iris_radius.dim() == 2:
                base_iris_radius = base_iris_radius.squeeze(-1)
        else:
            base_iris_radius = torch.ones(batch_size, device=device) * 0.6

        # Ensure proper shape
        if base_iris_radius.dim() == 0:
            base_iris_radius = base_iris_radius.unsqueeze(0).expand(batch_size)
        elif base_iris_radius.shape[0] == 1:
            base_iris_radius = base_iris_radius.expand(batch_size)

        # Get cornea distance
        if subject_params and 'cornea2center' in subject_params:
            cornea_dist = subject_params['cornea2center']
            if cornea_dist.dim() == 2:
                cornea_dist = cornea_dist.squeeze(-1)
        else:
            cornea_dist = torch.ones(batch_size, device=device) * 0.5

        # Ensure proper shape
        if cornea_dist.dim() == 0:
            cornea_dist = cornea_dist.unsqueeze(0).expand(batch_size)
        elif cornea_dist.shape[0] == 1:
            cornea_dist = cornea_dist.expand(batch_size)

        iris_landmarks_list = []

        for eye_idx in range(2):
            # Iris center along optical axis
            cornea_dist_expanded = cornea_dist.unsqueeze(-1)  # [B, 1]
            iris_center = eyeball_centers[:, eye_idx] + \
                          cornea_dist_expanded * optical_axes[:, eye_idx]  # [B, 3]

            # Scale iris radius for this eye
            iris_radius = base_iris_radius * (0.5 + 0.5 * iris_radius_scales[:, eye_idx])  # [B]

            # Generate circle on plane
            iris_landmarks = self.generate_circle_on_plane(
                iris_center,
                iris_normals[:, eye_idx],
                iris_radius,
                100,
                batch_size,
                device
            )  # [B, 100, 3]

            iris_landmarks_list.append(iris_landmarks)

        return torch.cat(iris_landmarks_list, dim=1)  # [B, 200, 3]

    def weak_perspective_projection(self, eyeball_centers, pupil_centers,
                                    iris_landmarks, scale, translation_2d,
                                    camera_params):
        """
        Project 3D points to 2D using weak perspective model
        This is more stable than full perspective for learning
        """
        K = camera_params.get('intrinsic_matrix')
        if K is None:
            return None

        batch_size = eyeball_centers.shape[0]

        # Handle different K matrix formats
        if K.dim() == 2:
            K = K.unsqueeze(0).expand(batch_size, -1, -1)
        elif K.shape[0] != batch_size:
            # If K has wrong batch size, expand or truncate
            if K.shape[0] == 1:
                K = K.expand(batch_size, -1, -1)
            else:
                K = K[:batch_size]

        projections = {}

        # Project using weak perspective approximation
        def project_weak(points_3d):
            # Ensure batch dimensions match
            if points_3d.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch: points {points_3d.shape[0]} vs expected {batch_size}")

            # Project: K @ points_3d^T
            if points_3d.dim() == 3:  # [B, N, 3]
                projected = torch.bmm(K, points_3d.transpose(1, 2))  # [B, 3, N]
                projected = projected.transpose(1, 2)  # [B, N, 3]
            else:  # [B, 2, 3] for eyeball/pupil centers
                # Reshape to [B, 2*3] then [B, 3, 2]
                points_reshaped = points_3d.reshape(batch_size, -1, 3).transpose(1, 2)  # [B, 3, M]
                projected = torch.bmm(K, points_reshaped)  # [B, 3, M]
                projected = projected.transpose(1, 2)  # [B, M, 3]
                projected = projected.reshape(points_3d.shape[0], points_3d.shape[1], 3)  # [B, 2, 3]

            # Use average Z for stability (weak perspective assumption)
            if projected.dim() == 3:
                z_avg = projected[..., 2:3].mean(dim=1, keepdim=True)  # [B, 1, 1]
                points_2d = projected[..., :2] / (z_avg + 1e-8)
            else:
                z_avg = projected[..., 2:3].mean(dim=1, keepdim=True)  # [B, 1, 1]
                points_2d = projected[..., :2] / (z_avg + 1e-8)

            return points_2d

        projections['eyeball_centers_2d'] = project_weak(eyeball_centers)
        projections['pupil_centers_2d'] = project_weak(pupil_centers)
        projections['iris_landmarks_2d'] = project_weak(iris_landmarks)

        return projections

    def apply_kappa(self, optical_axes, subject_params):
        """Apply kappa angle correction - 2D kappa (horizontal, vertical)"""
        if subject_params and 'L_kappa' in subject_params and 'R_kappa' in subject_params:
            visual_axes = optical_axes.clone()

            L_kappa = subject_params['L_kappa']  # Should be [B, 2]
            R_kappa = subject_params['R_kappa']  # Should be [B, 2]

            # Apply small angle approximation for kappa
            # Horizontal kappa rotates around Y axis (affects X and Z)
            # Vertical kappa rotates around X axis (affects Y and Z)

            # For left eye
            if L_kappa.dim() == 2 and L_kappa.shape[-1] >= 2:
                # Use only first 2 components if somehow it has more
                kappa_h = L_kappa[:, 0]  # [B] horizontal
                kappa_v = L_kappa[:, 1]  # [B] vertical

                # Apply rotations using small angle approximation
                # For small angles: sin(θ) ≈ θ, cos(θ) ≈ 1
                # But for accuracy, use actual sin/cos
                sin_h = torch.sin(kappa_h)
                cos_h = torch.cos(kappa_h)
                sin_v = torch.sin(kappa_v)
                cos_v = torch.cos(kappa_v)

                # Get original components
                x = visual_axes[:, 0, 0]
                y = visual_axes[:, 0, 1]
                z = visual_axes[:, 0, 2]

                # Apply horizontal rotation (around Y axis)
                new_x = x * cos_h - z * sin_h
                new_z = x * sin_h + z * cos_h

                # Apply vertical rotation (around X axis) to the new z
                new_y = y * cos_v - new_z * sin_v
                final_z = y * sin_v + new_z * cos_v

                visual_axes[:, 0, 0] = new_x
                visual_axes[:, 0, 1] = new_y
                visual_axes[:, 0, 2] = final_z

            # For right eye
            if R_kappa.dim() == 2 and R_kappa.shape[-1] >= 2:
                kappa_h = R_kappa[:, 0]
                kappa_v = R_kappa[:, 1]

                sin_h = torch.sin(kappa_h)
                cos_h = torch.cos(kappa_h)
                sin_v = torch.sin(kappa_v)
                cos_v = torch.cos(kappa_v)

                x = visual_axes[:, 1, 0]
                y = visual_axes[:, 1, 1]
                z = visual_axes[:, 1, 2]

                new_x = x * cos_h - z * sin_h
                new_z = x * sin_h + z * cos_h

                new_y = y * cos_v - new_z * sin_v
                final_z = y * sin_v + new_z * cos_v

                visual_axes[:, 1, 0] = new_x
                visual_axes[:, 1, 1] = new_y
                visual_axes[:, 1, 2] = final_z

            # Renormalize to unit vectors
            visual_axes = F.normalize(visual_axes, dim=-1)
            return visual_axes

        return optical_axes


# class WeakPerspectiveLoss(nn.Module):
#     """
#     Fixed loss function with proper shape handling
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.l1_loss = nn.L1Loss()
#         self.mse_loss = nn.MSELoss()
#
#     def forward(self, predictions, ground_truth, subject_params=None, camera_params=None):
#         losses = {}
#         device = predictions['eyeball_centers'].device
#
#         # Debug shapes
#         print("\n=== Loss Computation Debug ===")
#
#         # === 2D SUPERVISION (PRIMARY) ===
#         if 'projections_2d' in predictions and predictions['projections_2d'] is not None:
#
#             # Iris 2D loss
#             if 'iris_mesh_2D' in ground_truth:
#                 pred_iris_2d = predictions['projections_2d']['iris_landmarks_2d']
#                 gt_iris_2d = ground_truth['iris_mesh_2D']
#
#                 print(f"  Iris 2D - Pred: {pred_iris_2d.shape}, GT: {gt_iris_2d.shape}")
#
#                 # Ensure shapes match
#                 if pred_iris_2d.shape != gt_iris_2d.shape:
#                     print(f"  WARNING: Shape mismatch in iris 2D!")
#                     # Try to fix common issues
#                     if gt_iris_2d.dim() == 4 and gt_iris_2d.shape[1] == 2:
#                         gt_iris_2d = gt_iris_2d.reshape(gt_iris_2d.shape[0], -1, 2)
#                         print(f"  Fixed GT shape to: {gt_iris_2d.shape}")
#
#                 if pred_iris_2d.shape == gt_iris_2d.shape:
#                     losses['iris_2d'] = self.l1_loss(pred_iris_2d, gt_iris_2d)
#                 else:
#                     losses['iris_2d'] = torch.tensor(0.0, device=device)
#
#             # Eyeball 2D loss
#             if 'eyeball_center_2D' in ground_truth:
#                 pred_eyeball_2d = predictions['projections_2d']['eyeball_centers_2d']
#                 gt_eyeball_2d = ground_truth['eyeball_center_2D']
#
#                 print(f"  Eyeball 2D - Pred: {pred_eyeball_2d.shape}, GT: {gt_eyeball_2d.shape}")
#
#                 if pred_eyeball_2d.shape == gt_eyeball_2d.shape:
#                     losses['eyeball_2d'] = self.l1_loss(pred_eyeball_2d, gt_eyeball_2d)
#                 else:
#                     losses['eyeball_2d'] = torch.tensor(0.0, device=device)
#
#             # Pupil 2D loss
#             if 'pupil_center_2D' in ground_truth:
#                 pred_pupil_2d = predictions['projections_2d']['pupil_centers_2d']
#                 gt_pupil_2d = ground_truth['pupil_center_2D']
#
#                 print(f"  Pupil 2D - Pred: {pred_pupil_2d.shape}, GT: {gt_pupil_2d.shape}")
#
#                 if pred_pupil_2d.shape == gt_pupil_2d.shape:
#                     losses['pupil_2d'] = self.l1_loss(pred_pupil_2d, gt_pupil_2d)
#                 else:
#                     losses['pupil_2d'] = torch.tensor(0.0, device=device)
#
#         # === 3D SUPERVISION ===
#
#         # Eyeball 3D loss
#         pred_eyeball_3d = predictions['eyeball_centers']
#         gt_eyeball_3d = ground_truth['eyeball_center_3D']
#         print(f"  Eyeball 3D - Pred: {pred_eyeball_3d.shape}, GT: {gt_eyeball_3d.shape}")
#
#         if pred_eyeball_3d.shape == gt_eyeball_3d.shape:
#             losses['eyeball_3d'] = self.l1_loss(pred_eyeball_3d, gt_eyeball_3d)
#         else:
#             losses['eyeball_3d'] = torch.tensor(0.0, device=device)
#
#         # Pupil 3D loss
#         pred_pupil_3d = predictions['pupil_centers']
#         gt_pupil_3d = ground_truth['pupil_center_3D']
#         print(f"  Pupil 3D - Pred: {pred_pupil_3d.shape}, GT: {gt_pupil_3d.shape}")
#
#         if pred_pupil_3d.shape == gt_pupil_3d.shape:
#             losses['pupil_3d'] = self.l1_loss(pred_pupil_3d, gt_pupil_3d)
#         else:
#             losses['pupil_3d'] = torch.tensor(0.0, device=device)
#
#         # Iris 3D loss
#         pred_iris_3d = predictions['iris_landmarks_100']
#         gt_iris_3d = ground_truth['iris_mesh_3D']
#         print(f"  Iris 3D - Pred: {pred_iris_3d.shape}, GT: {gt_iris_3d.shape}")
#
#         if pred_iris_3d.shape == gt_iris_3d.shape:
#             losses['iris_3d'] = self.l1_loss(pred_iris_3d, gt_iris_3d)
#         else:
#             losses['iris_3d'] = torch.tensor(0.0, device=device)
#
#         # === ANGULAR LOSSES ===
#
#         # Gaze direction
#         if 'gaze_C' in ground_truth:
#             pred_gaze = predictions['head_gaze_direction']
#             gt_gaze = ground_truth['gaze_C']
#             print(f"  Gaze - Pred: {pred_gaze.shape}, GT: {gt_gaze.shape}")
#
#             if pred_gaze.shape == gt_gaze.shape:
#                 losses['gaze_direction'] = self.l1_loss(pred_gaze, gt_gaze)
#             else:
#                 losses['gaze_direction'] = torch.tensor(0.0, device=device)
#
#         # Optical axes
#         if 'optic_axis_L' in ground_truth and 'optic_axis_R' in ground_truth:
#             pred_optical = predictions['optical_axes']  # [B, 2, 3]
#             gt_optical_L = ground_truth['optic_axis_L']  # [B, 3]
#             gt_optical_R = ground_truth['optic_axis_R']  # [B, 3]
#
#             print(f"  Optical - Pred: {pred_optical.shape}, GT_L: {gt_optical_L.shape}, GT_R: {gt_optical_R.shape}")
#
#             # Compute cosine similarity for each eye
#             if pred_optical.shape[0] == gt_optical_L.shape[0]:
#                 cos_sim_L = torch.sum(pred_optical[:, 0] * gt_optical_L, dim=-1)
#                 cos_sim_R = torch.sum(pred_optical[:, 1] * gt_optical_R, dim=-1)
#                 losses['optical_axes'] = 2.0 - (torch.mean(cos_sim_L) + torch.mean(cos_sim_R))
#             else:
#                 losses['optical_axes'] = torch.tensor(0.0, device=device)
#
#         # === REGULARIZATION ===
#         if 'weak_perspective' in predictions:
#             wp = predictions['weak_perspective']
#             losses['scale_reg'] = torch.mean((wp['scale'] - 1.0) ** 2)
#             losses['trans_reg'] = torch.mean(wp['translation_2d'] ** 2)
#             losses['norm_centers_reg'] = torch.mean(torch.abs(wp['normalized_centers']))
#
#         # Add default values for any missing losses
#         default_loss_keys = ['iris_2d', 'eyeball_2d', 'pupil_2d', 'eyeball_3d',
#                              'pupil_3d', 'iris_3d', 'gaze_direction', 'optical_axes']
#         for key in default_loss_keys:
#             if key not in losses:
#                 losses[key] = torch.tensor(0.0, device=device)
#
#         total_loss = sum(losses.values())
#
#         print(f"  Total loss: {total_loss.item():.4f}")
#         print("=== End Loss Debug ===\n")
#
#         return total_loss, losses
