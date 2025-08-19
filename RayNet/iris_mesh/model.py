import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class IrisDetailAttention(nn.Module):
    """
    Fine-grained attention for iris texture and landmark details.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa

        return x


class EyeballGeometryBranch(nn.Module):
    """
    Predicts eyeball geometric parameters: center and radius for left/right eyes.
    """

    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 8)  # [left_center(3), left_radius(1), right_center(3), right_radius(1)]
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)
        params = self.fc(x)  # [B, 8]

        # Split into left and right eye parameters
        left_center = params[:, :3]  # [B, 3]
        left_radius = F.softplus(params[:, 3:4])  # [B, 1] - ensure positive
        right_center = params[:, 4:7]  # [B, 3]
        right_radius = F.softplus(params[:, 7:8])  # [B, 1] - ensure positive

        return {
            'eyeball_centers': torch.stack([left_center, right_center], dim=1),  # [B, 2, 3]
            'eyeball_radii': torch.stack([left_radius, right_radius], dim=1)  # [B, 2, 1]
        }


class IrisGeometryBranch(nn.Module):
    """
    Predicts iris-specific geometry: pupil centers, iris centers, and iris radii.
    """

    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Pupil center prediction
        self.pupil_fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6)  # [left_pupil(3), right_pupil(3)]
        )

        # Iris geometry prediction
        self.iris_fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 8)  # [left_center(3), left_radius(1), right_center(3), right_radius(1)]
        )

    def forward(self, x):
        pooled = self.pool(x).flatten(1)

        # Pupil centers
        pupil_params = self.pupil_fc(pooled)
        pupil_centers = pupil_params.view(-1, 2, 3)  # [B, 2, 3]

        # Iris geometry
        iris_params = self.iris_fc(pooled)
        iris_centers = iris_params[:, [0, 1, 2, 4, 5, 6]].view(-1, 2, 3)  # [B, 2, 3]
        iris_radii = F.softplus(iris_params[:, [3, 7]]).view(-1, 2, 1)  # [B, 2, 1]

        return {
            'pupil_centers': pupil_centers,
            'iris_centers': iris_centers,
            'iris_radii': iris_radii
        }


class SphericalRayBranch(nn.Module):
    """
    Predicts spherical coordinates (theta, phi) for each of the 100 iris landmarks.
    Uses radial structure to ensure anatomically plausible iris mesh.
    """

    def __init__(self, in_channels, hidden_dim, num_landmarks=100):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Predict radial structure (concentric circles + radial lines)
        self.radial_fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_landmarks * 4)  # [left_theta, left_phi, right_theta, right_phi] * 100
        )

        # Initialize with circular pattern
        self._init_circular_pattern()

    def _init_circular_pattern(self):
        """Initialize with a circular pattern for stable training."""
        with torch.no_grad():
            # Create concentric circles pattern
            angles = torch.linspace(0, 2 * math.pi, self.num_landmarks, dtype=torch.float32)
            # Small perturbations for different radial distances
            radial_offsets = torch.linspace(0.1, 0.9, self.num_landmarks)

            init_theta = angles
            init_phi = radial_offsets * 0.5  # Small angular variations

            # Initialize the last layer with this pattern
            init_pattern = torch.stack([init_theta, init_phi, init_theta, init_phi], dim=1).flatten()
            if hasattr(self.radial_fc[-1], 'weight'):
                self.radial_fc[-1].bias.data = init_pattern * 0.1

    def forward(self, x):
        pooled = self.pool(x).flatten(1)
        ray_params = self.radial_fc(pooled)  # [B, 400]

        # Reshape to [B, 2, 100, 2] - (left/right, landmarks, theta/phi)
        ray_coords = ray_params.view(-1, 2, self.num_landmarks, 2)

        # Apply constraints to ensure valid spherical coordinates
        theta = torch.sigmoid(ray_coords[..., 0]) * 2 * math.pi  # [0, 2π]
        phi = torch.sigmoid(ray_coords[..., 1]) * math.pi / 3  # [0, π/3] - limited range for iris

        return torch.stack([theta, phi], dim=-1)  # [B, 2, 100, 2]


class GeometricReconstructor(nn.Module):
    """
    Reconstructs 3D iris mesh from geometric parameters using anatomical constraints.
    """

    def __init__(self):
        super().__init__()

    def forward(self, eyeball_geometry, iris_geometry, spherical_rays):
        """
        Args:
            eyeball_geometry: dict with 'eyeball_centers' [B,2,3], 'eyeball_radii' [B,2,1]
            iris_geometry: dict with 'pupil_centers' [B,2,3], 'iris_centers' [B,2,3], 'iris_radii' [B,2,1]
            spherical_rays: [B, 2, 100, 2] - (theta, phi) coordinates
        Returns:
            iris_mesh_3d: [B, 2, 100, 3] - 3D coordinates of iris landmarks
        """
        batch_size = spherical_rays.shape[0]
        device = spherical_rays.device

        eyeball_centers = eyeball_geometry['eyeball_centers']  # [B, 2, 3]
        eyeball_radii = eyeball_geometry['eyeball_radii']  # [B, 2, 1]
        iris_centers = iris_geometry['iris_centers']  # [B, 2, 3]
        iris_radii = iris_geometry['iris_radii']  # [B, 2, 1]

        theta = spherical_rays[..., 0]  # [B, 2, 100]
        phi = spherical_rays[..., 1]  # [B, 2, 100]

        # Convert spherical coordinates to Cartesian on iris plane
        # Iris lies on eyeball sphere, so we use eyeball center as reference
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Create points on iris circle in local coordinate system
        local_x = iris_radii * cos_theta * cos_phi  # [B, 2, 100]
        local_y = iris_radii * sin_theta * cos_phi  # [B, 2, 100]
        local_z = iris_radii * sin_phi  # [B, 2, 100]

        # Stack to get local coordinates
        local_coords = torch.stack([local_x, local_y, local_z], dim=-1)  # [B, 2, 100, 3]

        # Transform to global coordinates relative to iris centers
        iris_mesh = iris_centers.unsqueeze(2) + local_coords  # [B, 2, 100, 3]

        # Project onto eyeball sphere for anatomical correctness
        vectors_to_center = iris_mesh - eyeball_centers.unsqueeze(2)  # [B, 2, 100, 3]
        distances = torch.norm(vectors_to_center, dim=-1, keepdim=True)  # [B, 2, 100, 1]
        normalized_vectors = vectors_to_center / (distances + 1e-8)

        # Project to eyeball surface
        projected_mesh = (eyeball_centers.unsqueeze(2) +
                          normalized_vectors * eyeball_radii.unsqueeze(2))  # [B, 2, 100, 3]

        return projected_mesh


class Iris2DProjectionBranch(nn.Module):
    """
    Predicts 2D iris landmarks directly in pixel space.
    Used for multi-scale supervision and projection consistency.
    """

    def __init__(self, in_channels, hidden_dim, num_landmarks=100):
        super().__init__()
        self.num_landmarks = num_landmarks

        # Use spatial features for 2D prediction
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )

        # Separate branches for left and right eye landmarks
        self.left_eye_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, num_landmarks * 2, 1),  # 100 landmarks * 2 coordinates
        )

        self.right_eye_head = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, num_landmarks * 2, 1),  # 100 landmarks * 2 coordinates
        )

        # Global average pooling for final prediction
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features):
        """
        Args:
            features: [B, C, H, W] - Spatial features
        Returns:
            iris_mesh_2d: [B, 2, 100, 2] - 2D pixel coordinates
        """
        x = self.conv_reduce(features)  # [B, hidden_dim//2, H, W]

        # Predict landmarks for each eye
        left_landmarks = self.left_eye_head(x)  # [B, 200, H, W]
        right_landmarks = self.right_eye_head(x)  # [B, 200, H, W]

        # Global pooling and reshape
        left_2d = self.pool(left_landmarks).flatten(1)  # [B, 200]
        right_2d = self.pool(right_landmarks).flatten(1)  # [B, 200]

        # Reshape to landmarks
        left_2d = left_2d.view(-1, self.num_landmarks, 2)  # [B, 100, 2]
        right_2d = right_2d.view(-1, self.num_landmarks, 2)  # [B, 100, 2]

        # Stack left and right
        iris_mesh_2d = torch.stack([left_2d, right_2d], dim=1)  # [B, 2, 100, 2]

        # Apply sigmoid to ensure coordinates are in [0, 1] range (normalized)
        iris_mesh_2d = torch.sigmoid(iris_mesh_2d)

        return iris_mesh_2d


class IrisMeshRegressionHead(nn.Module):
    """
    Complete geometrically-aware iris mesh regression head.
    Predicts both 3D and 2D iris landmarks with projection consistency.
    """

    def __init__(self, in_channels=256, hidden_dim=128, reduction=32, num_landmarks=100,
                 predict_2d=True):
        super().__init__()

        self.predict_2d = predict_2d

        # Feature processing
        self.coord_att = None  # Will be injected from your CoordAtt
        self.iris_attention = IrisDetailAttention(in_channels, reduction // 2)

        # Geometric prediction branches
        self.eyeball_branch = EyeballGeometryBranch(in_channels, hidden_dim)
        self.iris_branch = IrisGeometryBranch(in_channels, hidden_dim)
        self.ray_branch = SphericalRayBranch(in_channels, hidden_dim, num_landmarks)

        # 2D projection branch (optional)
        if self.predict_2d:
            self.projection_branch = Iris2DProjectionBranch(in_channels, hidden_dim, num_landmarks)

        # Geometric reconstruction
        self.reconstructor = GeometricReconstructor()

    def set_coord_attention(self, coord_att_module):
        """Inject your CoordAtt module."""
        self.coord_att = coord_att_module

    def project_3d_to_2d(self, points_3d, intrinsic_matrix, image_size=None):
        """
        Project 3D points to 2D pixel coordinates.

        Args:
            points_3d: [B, 2, 100, 3] - 3D iris landmarks
            intrinsic_matrix: [B, 3, 3] - Camera intrinsics
            image_size: (H, W) - Image dimensions for normalization

        Returns:
            points_2d: [B, 2, 100, 2] - 2D pixel coordinates (normalized [0,1])
        """
        batch_size, num_eyes, num_landmarks, _ = points_3d.shape

        # Reshape for batch processing
        points_flat = points_3d.view(batch_size, -1, 3)  # [B, 200, 3]

        # Project to 2D
        projected = torch.matmul(intrinsic_matrix, points_flat.transpose(-1, -2))  # [B, 3, 200]
        projected = projected.transpose(-1, -2)  # [B, 200, 3]

        # Normalize by depth
        points_2d_flat = projected[..., :2] / (projected[..., 2:3] + 1e-8)  # [B, 200, 2]

        # Reshape back
        points_2d = points_2d_flat.view(batch_size, num_eyes, num_landmarks, 2)  # [B, 2, 100, 2]

        # Normalize to [0, 1] if image size provided
        if image_size is not None:
            h, w = image_size
            points_2d[..., 0] /= w  # Normalize x
            points_2d[..., 1] /= h  # Normalize y

        return points_2d

    def forward(self, fused_features, intrinsic_matrix=None, image_size=None):
        """
        Args:
            fused_features: [B, C, H, W] - Output from your fusion module
            intrinsic_matrix: [B, 3, 3] - Camera intrinsics (optional, for projection)
            image_size: (H, W) - Image dimensions (optional, for normalization)
        Returns:
            dict with iris mesh and intermediate geometric parameters
        """
        # Apply attention mechanisms
        if self.coord_att is not None:
            features = self.coord_att(fused_features)
        else:
            features = fused_features

        features = self.iris_attention(features)

        # Predict geometric parameters
        eyeball_geometry = self.eyeball_branch(features)
        iris_geometry = self.iris_branch(features)
        spherical_rays = self.ray_branch(features)

        # Reconstruct 3D iris mesh
        iris_mesh_3d = self.reconstructor(eyeball_geometry, iris_geometry, spherical_rays)

        # Prepare output dictionary
        output = {
            'iris_mesh_3d': iris_mesh_3d,  # [B, 2, 100, 3] - Main output
            'eyeball_geometry': eyeball_geometry,  # Eyeball parameters
            'iris_geometry': iris_geometry,  # Iris parameters
            'spherical_rays': spherical_rays,  # Ray directions
            'pupil_centers_3d': iris_geometry['pupil_centers'],  # For compatibility
        }

        # Predict 2D landmarks if requested
        if self.predict_2d:
            iris_mesh_2d_direct = self.projection_branch(features)
            output['iris_mesh_2d_direct'] = iris_mesh_2d_direct  # [B, 2, 100, 2]

            # Also compute projected 2D from 3D if intrinsics available
            if intrinsic_matrix is not None:
                iris_mesh_2d_projected = self.project_3d_to_2d(
                    iris_mesh_3d, intrinsic_matrix, image_size
                )
                output['iris_mesh_2d_projected'] = iris_mesh_2d_projected  # [B, 2, 100, 2]

        return output