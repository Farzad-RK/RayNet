# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (
    create_orthonormal_basis, fit_plane_svd, normalize_vector,
    euler_to_rotation_matrix, transform_HCS_to_CCS,
    generate_iris_100_landmarks_in_plane, linear_blend_skinning
)


class EyeFLAME_Model(nn.Module):
    """
    EyeFLAME: FLAME-inspired 3D Eye Model with K=4 joints
    Joints: Head Pose, Head Gaze, Left Eyeball, Right Eyeball

    Args:
        in_channels (int): Input channels from fusion layer (256)
        hidden_dim (int): Hidden dimension for regression heads (128)
        reduction (int): Channel reduction factor (32)
    """

    def __init__(self, in_channels=256, hidden_dim=128, reduction=32):
        super().__init__()

        # === JOINT PARAMETER REGRESSION HEADS (K=4) ===

        # Joint 0: Head Pose (3 DOF - rotation matrix as 6D representation)
        self.head_pose_regression = JointRegressionHead(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            reduction=reduction,
            output_dim=6,  # 6D rotation representation
            joint_name="head_pose"
        )

        # Joint 1: Head Gaze (3 DOF - direction vector)
        self.head_gaze_regression = JointRegressionHead(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            reduction=reduction,
            output_dim=3,  # 3D direction vector
            joint_name="head_gaze"
        )

        # Joint 2: Left Eyeball (3 DOF - rotation)
        self.left_eyeball_regression = JointRegressionHead(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            reduction=reduction,
            output_dim=6,  # 6D rotation representation
            joint_name="left_eyeball"
        )

        # Joint 3: Right Eyeball (3 DOF - rotation)
        self.right_eyeball_regression = JointRegressionHead(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            reduction=reduction,
            output_dim=6,  # 6D rotation representation
            joint_name="right_eyeball"
        )

        # === LEARNABLE BLEND WEIGHTS FOR LINEAR BLEND SKINNING ===
        self.blend_weights = EyeFLAME_BlendWeights()

        # === TEMPLATE MESH PARAMETERS ===
        # These will be set during forward pass from subject parameters
        self.register_buffer('template_eyeball_centers', torch.zeros(2, 3))
        self.register_buffer('template_iris_landmarks', torch.zeros(200, 3))  # 100 per eye

    def forward(self, fused_features, subject_params=None, camera_params=None):
        """
        Forward pass through EyeFLAME model

        Args:
            fused_features: [B, 256, H, W] - Output from MultiScaleFusion
            subject_params: Dict with subject-specific parameters :
            camera_params: Camera parameters for 2D projection
        {
            'ID': int,                     # Subject ID from 1 to 56
            'gender': str,                 # ['F', 'M'] refers to female and male
            'ethicity': str,               # ['B', 'Y', 'W'] refers to Black, Yellow and White
            'eyecenter_L': np.array(3,)    # Left eyeball center coordinates under HCS
            'eyecenter_R': np.array(3,)    # Right eyeball center coordinates under HCS
            'eyeball_radius': float,       # Eyeball radius
            'iris_radius': float,          # Iris radius
            'cornea_radius': float,        # Cornea radius
            'cornea2center': float,        # Distance from cornea center to eyeball center
            'UVRadius': float,             # Normalized relative pupil size
            'L_kappa': np.array(3,),       # Euler angles of left eye kappa
            'R_kappa': np.array(3,)        # Euler angles of right eye kappa
        }

        Returns:
            dict: Complete eye structure predictions
        """
        batch_size = fused_features.shape[0]

        # === JOINT PARAMETER PREDICTION ===

        # Predict all joint parameters
        θ_head_pose_6d = self.head_pose_regression(fused_features)  # [B, 6]
        θ_head_gaze = self.head_gaze_regression(fused_features)  # [B, 3]
        θ_left_eyeball_6d = self.left_eyeball_regression(fused_features)  # [B, 6]
        θ_right_eyeball_6d = self.right_eyeball_regression(fused_features)  # [B, 6]

        # Convert 6D representations to rotation matrices
        θ_head_pose = ortho6d_to_rotmat(θ_head_pose_6d)  # [B, 3, 3]
        θ_left_eyeball = ortho6d_to_rotmat(θ_left_eyeball_6d)  # [B, 3, 3]
        θ_right_eyeball = ortho6d_to_rotmat(θ_right_eyeball_6d)  # [B, 3, 3]

        # Normalize head gaze direction
        θ_head_gaze = normalize_vector(θ_head_gaze)  # [B, 3]

        # === FORWARD KINEMATICS WITH K=4 JOINTS ===

        eye_structures = self.forward_kinematics(
            θ_joints={
                'head_pose': θ_head_pose,  # [B, 3, 3]
                'head_gaze': θ_head_gaze,  # [B, 3]
                'left_eyeball': θ_left_eyeball,  # [B, 3, 3]
                'right_eyeball': θ_right_eyeball  # [B, 3, 3]
            },
            β_shape=subject_params,
            batch_size=batch_size
        )

        # === 2D PROJECTION ===
        if camera_params is not None:
            eye_structures['projections_2d'] = self.project_to_2d(
                eye_structures, camera_params
            )

        # Store raw joint parameters for loss computation
        eye_structures['raw_joint_params'] = {
            'θ_head_pose_6d': θ_head_pose_6d,
            'θ_head_gaze': θ_head_gaze,
            'θ_left_eyeball_6d': θ_left_eyeball_6d,
            'θ_right_eyeball_6d': θ_right_eyeball_6d,
            'θ_head_pose': θ_head_pose,
            'θ_left_eyeball': θ_left_eyeball,
            'θ_right_eyeball': θ_right_eyeball
        }

        print("=== RAW MODEL OUTPUTS (before kinematics) ===")
        print(f"Head pose 6D range: {θ_head_pose_6d.min()} to {θ_head_pose_6d.max()}")
        print(f"Head gaze range: {θ_head_gaze.min()} to {θ_head_gaze.max()}")
        print(f"Left eyeball 6D range: {θ_left_eyeball_6d.min()} to {θ_left_eyeball_6d.max()}")
        print(f"Right eyeball 6D range: {θ_right_eyeball_6d.min()} to {θ_right_eyeball_6d.max()}")

        return eye_structures

    def forward_kinematics(self, θ_joints, β_shape, batch_size):
        """
        FLAME-style forward kinematics with K=4 joints

        Args:
            θ_joints: Dict of joint parameters
            β_shape: Subject-specific shape parameters (fixed from GazeGene)
            batch_size: Batch size

        Returns:
            dict: 3D eye structures
        """

        # === SETUP TEMPLATE STRUCTURES ===

        # Template eyeball centers in HCS (from subject parameters)
        template_eyeball_centers = torch.stack([
            β_shape['eyecenter_L'],  # [B, 3]
            β_shape['eyecenter_R']  # [B, 3]
        ], dim=1)  # [B, 2, 3]

        # Generate template iris landmarks (100 per eye, 200 total)
        template_iris_landmarks = self.generate_template_iris_landmarks(
            β_shape, batch_size
        )  # [B, 200, 3]

        # === JOINT TRANSFORMATIONS ===

        # Get blend weights for Linear Blend Skinning
        blend_weights = self.blend_weights()

        # Apply Linear Blend Skinning to transform template structures
        # Each vertex is influenced by all 4 joints with learned weights

        # Transform eyeball centers
        final_eyeball_centers = linear_blend_skinning(
            vertices=template_eyeball_centers,  # [B, 2, 3]
            joint_transforms=θ_joints,
            blend_weights=blend_weights['eyeball_centers']  # [2, 4]
        )

        # Transform iris landmarks
        final_iris_landmarks = linear_blend_skinning(
            vertices=template_iris_landmarks,  # [B, 200, 3]
            joint_transforms=θ_joints,
            blend_weights=blend_weights['iris_landmarks']  # [200, 4]
        )

        # === DERIVE ADDITIONAL STRUCTURES ===

        # Compute pupil centers (centroids of iris landmarks)
        pupil_centers = torch.stack([
            torch.mean(final_iris_landmarks[:, :100], dim=1),  # Left eye
            torch.mean(final_iris_landmarks[:, 100:], dim=1)  # Right eye
        ], dim=1)  # [B, 2, 3]

        # Compute optical axes (from eyeball centers to pupil centers)
        optical_axes = normalize_vector(
            pupil_centers - final_eyeball_centers
        )  # [B, 2, 3]

        # Compute visual axes (apply kappa corrections)
        L_kappa_rotation = euler_to_rotation_matrix(β_shape['L_kappa'])  # [B, 3, 3]
        R_kappa_rotation = euler_to_rotation_matrix(β_shape['R_kappa'])  # [B, 3, 3]

        visual_axes = torch.stack([
            torch.bmm(L_kappa_rotation, optical_axes[:, 0:1].transpose(1, 2)).squeeze(-1),
            torch.bmm(R_kappa_rotation, optical_axes[:, 1:2].transpose(1, 2)).squeeze(-1)
        ], dim=1)  # [B, 2, 3]

        print("=== BEFORE/AFTER KINEMATICS ===")
        print(f"Template eyeball centers range: {template_eyeball_centers.min()} to {template_eyeball_centers.max()}")
        print(f"Final eyeball centers range: {final_eyeball_centers.min()} to {final_eyeball_centers.max()}")

        return {
            'eyeball_centers': final_eyeball_centers,  # [B, 2, 3] in cm
            'pupil_centers': pupil_centers,  # [B, 2, 3] in cm
            'iris_landmarks_100': final_iris_landmarks,  # [B, 200, 3] in cm (100 per eye)
            'optical_axes': optical_axes,  # [B, 2, 3] unit vectors
            'visual_axes': visual_axes,  # [B, 2, 3] unit vectors
            'head_gaze_direction': θ_joints['head_gaze']  # [B, 3] unit vector
        }

    def generate_template_iris_landmarks(self, β_shape, batch_size):
        """
        Generate template iris landmarks (100 per eye) in HCS

        Args:
            β_shape: Subject-specific parameters
            batch_size: Batch size

        Returns:
            torch.Tensor: [B, 200, 3] iris landmarks in HCS
        """
        iris_landmarks_list = []

        for eye_idx in [0, 1]:  # Left, Right
            # Eyeball center in HCS
            if eye_idx == 0:
                eyeball_center = β_shape['eyecenter_L']  # [B, 3]
            else:
                eyeball_center = β_shape['eyecenter_R']  # [B, 3]

            # Generate 100 landmarks in perfect circle on iris plane
            iris_landmarks = generate_iris_100_landmarks_in_plane(
                eyeball_center=eyeball_center,
                iris_radius=β_shape['iris_radius'],  # [B, 1]
                cornea2center=β_shape['cornea2center'],  # [B, 1]
                batch_size=batch_size,
                device=eyeball_center.device
            )  # [B, 100, 3]

            iris_landmarks_list.append(iris_landmarks)

        # Concatenate left and right iris landmarks
        template_iris_landmarks = torch.cat(iris_landmarks_list, dim=1)  # [B, 200, 3]

        return template_iris_landmarks

    def project_to_2d(self, eye_structures, camera_params):
        """
        Project 3D eye structures to 2D using camera parameters

        Args:
            eye_structures: Dict of 3D structures
            camera_params: Camera parameters

        Returns:
            dict: 2D projections
        """
        projections_2d = {}

        # Project eyeball centers
        projections_2d['eyeball_centers_2d'] = project_3d_to_2d(
            eye_structures['eyeball_centers'], camera_params
        )

        # Project pupil centers
        projections_2d['pupil_centers_2d'] = project_3d_to_2d(
            eye_structures['pupil_centers'], camera_params
        )

        # Project iris landmarks
        projections_2d['iris_landmarks_2d'] = project_3d_to_2d(
            eye_structures['iris_landmarks_100'], camera_params
        )

        return projections_2d


class JointRegressionHead(nn.Module):
    """
    Individual regression head for each joint parameter
    """

    def __init__(self, in_channels, hidden_dim, reduction, output_dim, joint_name):
        super().__init__()

        self.joint_name = joint_name
        self.output_dim = output_dim

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Channel reduction
        self.channel_reduction = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # Hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(in_channels // reduction, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim // 2, output_dim)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] fused features

        Returns:
            torch.Tensor: [B, output_dim] joint parameters
        """
        # Global pooling
        x = self.global_pool(x)  # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C]

        # Channel reduction
        x = self.channel_reduction(x)  # [B, C//reduction]

        # Hidden layers
        x = self.hidden_layers(x)  # [B, hidden_dim//2]

        # Output
        output = self.output_layer(x)  # [B, output_dim]

        return output

    def _initialize_weights(self):
        """Initialize weights with appropriate scales for each joint type"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Special initialization for output layer based on joint type
        if self.joint_name in ['head_pose', 'left_eyeball', 'right_eyeball']:
            # For rotation outputs, initialize to identity-like values
            nn.init.normal_(self.output_layer.weight, 0, 0.01)
            if self.output_layer.bias is not None:
                # Initialize to approximate identity rotation in 6D representation
                self.output_layer.bias.data = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float32)
        elif self.joint_name == 'head_gaze':
            # For gaze direction, initialize to forward direction
            nn.init.normal_(self.output_layer.weight, 0, 0.01)
            if self.output_layer.bias is not None:
                self.output_layer.bias.data = torch.tensor([0, 0, -1], dtype=torch.float32)  # Forward gaze


class EyeFLAME_BlendWeights(nn.Module):
    """
    Learnable blend weights for Linear Blend Skinning (FLAME-style)
    K=4 joints: Head Pose, Head Gaze, Left Eyeball, Right Eyeball
    """

    def __init__(self):
        super().__init__()

        # Blend weights for eyeball centers (2 centers × 4 joints)
        # Initialize with anatomically reasonable weights
        initial_eyeball_weights = torch.tensor([
            [0.6, 0.2, 0.2, 0.0],  # Left eyeball: influenced by head_pose, head_gaze, left_eyeball
            [0.6, 0.2, 0.0, 0.2]  # Right eyeball: influenced by head_pose, head_gaze, right_eyeball
        ], dtype=torch.float32)

        self.eyeball_blend_weights = nn.Parameter(initial_eyeball_weights)

        # Blend weights for iris landmarks (200 landmarks × 4 joints)
        # 100 left eye landmarks + 100 right eye landmarks
        initial_iris_weights = torch.zeros(200, 4)

        # Left iris landmarks (0:100) - influenced by head_pose, head_gaze, left_eyeball
        initial_iris_weights[:100, :] = torch.tensor([0.4, 0.3, 0.3, 0.0])

        # Right iris landmarks (100:200) - influenced by head_pose, head_gaze, right_eyeball
        initial_iris_weights[100:, :] = torch.tensor([0.4, 0.3, 0.0, 0.3])

        self.iris_blend_weights = nn.Parameter(initial_iris_weights)

    def forward(self):
        """
        Returns normalized blend weights (sum to 1 for each vertex)
        """
        # Ensure weights sum to 1 for each vertex (FLAME constraint)
        eyeball_weights = F.softmax(self.eyeball_blend_weights, dim=1)  # [2, 4]
        iris_weights = F.softmax(self.iris_blend_weights, dim=1)  # [200, 4]

        return {
            'eyeball_centers': eyeball_weights,  # [2, 4]
            'iris_landmarks': iris_weights  # [200, 4]
        }


def ortho6d_to_rotmat(ortho6d):
    """
    Convert 6D orthogonal representation to rotation matrix

    Args:
        ortho6d: [B, 6] 6D representation

    Returns:
        torch.Tensor: [B, 3, 3] rotation matrices
    """
    x_raw = ortho6d[:, 0:3]  # [B, 3]
    y_raw = ortho6d[:, 3:6]  # [B, 3]

    x = normalize_vector(x_raw)  # [B, 3]
    z = torch.cross(x, y_raw, dim=1)  # [B, 3]
    z = normalize_vector(z)  # [B, 3]
    y = torch.cross(z, x, dim=1)  # [B, 3]

    # Stack to form rotation matrix
    rotmat = torch.stack([x, y, z], dim=2)  # [B, 3, 3]

    return rotmat


def project_3d_to_2d(points_3d, camera_params):
    """
    Project 3D points to 2D using camera parameters

    Args:
        points_3d: [B, N, 3] 3D points in camera coordinate system
        camera_params: Dict with 'intrinsic_matrix', etc.

    Returns:
        torch.Tensor: [B, N, 2] 2D projections
    """
    # This is a placeholder - implement based on your camera model
    # Typically: points_2d = K @ points_3d where K is intrinsic matrix

    batch_size, num_points, _ = points_3d.shape

    # Simple perspective projection (replace with your camera model)
    intrinsic = camera_params.get('intrinsic_matrix')  # [B, 3, 3] or [3, 3]

    if intrinsic is not None:
        # Handle both batch and single intrinsic matrix cases
        if intrinsic.dim() == 2:
            intrinsic = intrinsic.unsqueeze(0).expand(batch_size, -1, -1)

        # Project: [B, 3, 3] @ [B, N, 3].T = [B, 3, N] -> [B, N, 3]
        points_3d_homogeneous = points_3d  # Already in camera coordinates
        projected = torch.bmm(intrinsic, points_3d_homogeneous.transpose(1, 2))  # [B, 3, N]
        projected = projected.transpose(1, 2)  # [B, N, 3]

        # Convert to 2D by dividing by Z coordinate
        points_2d = projected[:, :, :2] / (projected[:, :, 2:3] + 1e-8)  # [B, N, 2]

        return points_2d
    else:
        # Fallback simple projection
        points_2d = points_3d[:, :, :2] / (points_3d[:, :, 2:3] + 1e-8)
        return points_2d