import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math


class EyeballModel(nn.Module):
    """
    Parametric eyeball model for 3D eye reconstruction
    Learns to predict eyeball shape and position from features
    """

    def __init__(self,
                 input_dim: int = 256,
                 n_shape_params: int = 10,
                 n_vertices: int = 642):  # Icosphere subdivision level 3
        super().__init__()

        self.n_shape_params = n_shape_params
        self.n_vertices = n_vertices

        # Initialize base eyeball mesh (unit sphere)
        self.register_buffer('base_vertices', self._create_icosphere(n_vertices))
        self.register_buffer('faces', self._create_icosphere_faces(n_vertices))

        # Shape basis for eyeball deformation (learned)
        self.shape_basis = nn.Parameter(torch.randn(n_shape_params, n_vertices, 3) * 0.01)

        # Networks for eyeball parameters
        self.eyeball_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )

        # Predict shape coefficients
        self.shape_predictor = nn.Linear(64, n_shape_params)

        # Predict eyeball center position
        self.center_predictor = nn.Linear(64, 3)

        # Predict eyeball radius (typically ~12mm)
        self.radius_predictor = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()  # Constrain between 0-1, scale later
        )

        # Predict rotation for eyeball orientation
        self.rotation_predictor = nn.Linear(64, 6)  # 6D rotation representation

    def _create_icosphere(self, n_vertices):
        """Create base icosphere vertices"""
        # Simplified - in practice, use proper icosphere generation
        # This is a placeholder for unit sphere vertices
        phi = torch.linspace(0, np.pi, int(np.sqrt(n_vertices)))
        theta = torch.linspace(0, 2 * np.pi, int(np.sqrt(n_vertices)))
        phi, theta = torch.meshgrid(phi, theta)

        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)

        vertices = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
        return vertices[:n_vertices]

    def _create_icosphere_faces(self, n_vertices):
        """Create icosphere face indices"""
        # Placeholder - implement proper icosphere topology
        n_faces = (n_vertices - 2) * 2
        faces = torch.randint(0, n_vertices, (n_faces, 3))
        return faces

    def forward(self, features):
        """
        Generate eyeball mesh from features
        Args:
            features: (B, C) encoded features
        Returns:
            dict with eyeball mesh and parameters
        """
        batch_size = features.shape[0]

        # Encode features
        encoded = self.eyeball_encoder(features)

        # Predict eyeball parameters
        shape_coeffs = self.shape_predictor(encoded)  # (B, n_shape_params)
        center = self.center_predictor(encoded)  # (B, 3)
        radius = 10.0 + 4.0 * self.radius_predictor(encoded)  # (B, 1) - scale to 10-14mm
        rotation_6d = self.rotation_predictor(encoded)  # (B, 6)

        # Convert 6D to rotation matrix
        rotation_matrix = self.ortho6d_to_rotmat(rotation_6d)  # (B, 3, 3)

        # Apply shape deformation
        vertices = self.base_vertices.unsqueeze(0).expand(batch_size, -1, -1)  # (B, V, 3)

        # Add shape basis deformations
        shape_deltas = torch.einsum('bs,svd->bvd', shape_coeffs, self.shape_basis)
        vertices = vertices + shape_deltas

        # Apply rotation
        vertices = torch.einsum('bvd,bkd->bvk', vertices, rotation_matrix)

        # Scale by radius
        vertices = vertices * radius.unsqueeze(1)

        # Translate to center position
        vertices = vertices + center.unsqueeze(1)

        return {
            'vertices': vertices,
            'faces': self.faces,
            'center': center,
            'radius': radius.squeeze(-1),
            'rotation': rotation_matrix,
            'shape_coeffs': shape_coeffs
        }

    def ortho6d_to_rotmat(self, ortho6d):
        """Convert 6D orthogonal representation to rotation matrix"""
        x_raw = ortho6d[:, :3]
        y_raw = ortho6d[:, 3:6]

        x = F.normalize(x_raw, dim=1)
        z = F.normalize(torch.cross(x, y_raw, dim=1), dim=1)
        y = F.normalize(torch.cross(z, x, dim=1), dim=1)

        return torch.stack([x, y, z], dim=2)


class IrisMeshDecoder(nn.Module):
    """
    Decoder for 3D iris mesh (100 landmarks from GazeGene)
    Also handles pupil center prediction
    """

    def __init__(self,
                 input_dim: int = 256,
                 n_landmarks: int = 100,
                 hidden_dim: int = 256):
        super().__init__()
        self.n_landmarks = n_landmarks

        # Shared encoder for both eyes
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Separate decoders for left and right iris
        self.left_iris_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_landmarks * 3)
        )

        self.right_iris_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_landmarks * 3)
        )

        # Pupil center predictors (3D position within iris)
        self.left_pupil_predictor = nn.Linear(hidden_dim, 3)
        self.right_pupil_predictor = nn.Linear(hidden_dim, 3)

        # Iris shape parameters (for regularization and control)
        self.iris_shape_predictor = nn.Linear(hidden_dim, 10)  # 10 shape parameters

    def forward(self, features, eyeball_info=None):
        """
        Decode iris landmarks and pupil centers
        Args:
            features: (B, C) encoded features
            eyeball_info: dict with eyeball centers and radii (optional, for constraint)
        Returns:
            dict with iris landmarks and pupil centers
        """
        batch_size = features.shape[0]

        # Encode features
        encoded = self.encoder(features)

        # Decode iris landmarks
        left_iris = self.left_iris_decoder(encoded).view(batch_size, self.n_landmarks, 3)
        right_iris = self.right_iris_decoder(encoded).view(batch_size, self.n_landmarks, 3)

        # Predict pupil centers
        left_pupil = self.left_pupil_predictor(encoded)
        right_pupil = self.right_pupil_predictor(encoded)

        # Predict iris shape parameters (for regularization)
        iris_shape = self.iris_shape_predictor(encoded)

        # Apply geometric constraints if eyeball info is provided
        if eyeball_info is not None:
            left_iris = self._constrain_to_eyeball(
                left_iris,
                eyeball_info['center'][:, 0] if eyeball_info['center'].dim() > 1 else eyeball_info['center'],
                eyeball_info['radius']
            )
            right_iris = self._constrain_to_eyeball(
                right_iris,
                eyeball_info['center'][:, 1] if eyeball_info['center'].dim() > 1 else eyeball_info['center'],
                eyeball_info['radius']
            )

        return {
            'iris_landmarks': {
                'left': left_iris,
                'right': right_iris
            },
            'pupil_centers': {
                'left': left_pupil,
                'right': right_pupil
            },
            'iris_shape_params': iris_shape
        }

    def _constrain_to_eyeball(self, iris_points, eyeball_center, eyeball_radius):
        """
        Project iris points onto eyeball surface
        Args:
            iris_points: (B, N, 3) iris landmark positions
            eyeball_center: (B, 3) eyeball center
            eyeball_radius: (B,) or (B, 1) eyeball radius
        """
        # Vector from eyeball center to each iris point
        vectors = iris_points - eyeball_center.unsqueeze(1)

        # Normalize and scale to eyeball surface
        distances = torch.norm(vectors, dim=2, keepdim=True)
        normalized = vectors / (distances + 1e-8)

        # Project onto sphere surface
        if eyeball_radius.dim() == 1:
            eyeball_radius = eyeball_radius.unsqueeze(1)

        constrained = eyeball_center.unsqueeze(1) + normalized * eyeball_radius.unsqueeze(1)

        return constrained


class GazeRayEstimator(nn.Module):
    """
    Estimate gaze rays from eyeball and iris geometry
    Computes optical axis, visual axis, and combined gaze vector
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        # Network for optical axis from iris geometry
        self.optical_axis_net = nn.Sequential(
            nn.Linear(100 * 3 + 3 + 3, hidden_dim),  # iris + pupil + eyeball center
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

        # Network for visual axis (includes foveal offset)
        self.visual_axis_net = nn.Sequential(
            nn.Linear(3 + 3, 32),  # optical axis + eye-specific features
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)
        )

        # Combined gaze from both eyes
        self.gaze_combiner = nn.Sequential(
            nn.Linear(6 + 6, 64),  # both visual axes + both eyeball centers
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)
        )

        # Gaze depth estimator
        self.depth_estimator = nn.Sequential(
            nn.Linear(3 + 3 + 6, 64),  # gaze vector + mean eye center + eye centers
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, iris_data, eyeball_data):
        """
        Compute gaze rays from eye geometry
        Args:
            iris_data: dict with iris_landmarks and pupil_centers
            eyeball_data: dict with eyeball centers
        Returns:
            dict with all gaze-related outputs
        """
        batch_size = iris_data['iris_landmarks']['left'].shape[0]
        results = {}

        # Process each eye
        for eye in ['left', 'right']:
            # Flatten iris landmarks
            iris_flat = iris_data['iris_landmarks'][eye].reshape(batch_size, -1)
            pupil = iris_data['pupil_centers'][eye]

            # Get eyeball center for this eye
            if eyeball_data['center'].dim() > 1 and eyeball_data['center'].shape[1] == 2:
                eyeball_center = eyeball_data['center'][:, 0 if eye == 'left' else 1]
            else:
                eyeball_center = eyeball_data['center']

            # Compute optical axis
            optical_input = torch.cat([iris_flat, pupil, eyeball_center], dim=1)
            optical_axis = F.normalize(self.optical_axis_net(optical_input), dim=1)

            # Compute visual axis (with foveal offset)
            visual_input = torch.cat([optical_axis, pupil - eyeball_center], dim=1)
            visual_offset = self.visual_axis_net(visual_input)
            visual_axis = F.normalize(optical_axis + 0.1 * visual_offset, dim=1)

            results[f'optical_axis_{eye}'] = optical_axis
            results[f'visual_axis_{eye}'] = visual_axis
            results[f'eyeball_center_{eye}'] = eyeball_center

        # Combine both eyes for final gaze
        combined_input = torch.cat([
            results['visual_axis_left'],
            results['visual_axis_right'],
            results['eyeball_center_left'],
            results['eyeball_center_right']
        ], dim=1)

        gaze_vector = F.normalize(self.gaze_combiner(combined_input), dim=1)
        results['gaze_vector'] = gaze_vector

        # Ray origin (mean of eyeball centers)
        ray_origin = (results['eyeball_center_left'] + results['eyeball_center_right']) / 2
        results['ray_origin'] = ray_origin
        results['ray_direction'] = gaze_vector

        # Estimate gaze depth
        depth_input = torch.cat([
            gaze_vector,
            ray_origin,
            results['eyeball_center_left'],
            results['eyeball_center_right']
        ], dim=1)
        gaze_depth = self.depth_estimator(depth_input).squeeze(-1)
        results['gaze_depth'] = gaze_depth

        # Compute 3D gaze point
        results['gaze_point_3d'] = ray_origin + gaze_depth.unsqueeze(-1) * gaze_vector

        return results


class RayNet(nn.Module):
    """
    Eye-focused RayNet for 3D eyeball/iris reconstruction and gaze estimation
    """

    def __init__(self,
                 backbone,
                 in_channels_list,
                 n_iris_landmarks: int = 100,
                 panet_out_channels: int = 256):
        super().__init__()

        # Visual backbone
        self.backbone = backbone

        # Feature pyramid network
        from panet import PANet
        from fusion import MultiScaleFusion
        self.panet = PANet(channels_list=in_channels_list, out_channels=panet_out_channels)
        self.fusion = MultiScaleFusion(in_channels=panet_out_channels, n_scales=4, out_channels=256)

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )

        # Eyeball reconstruction (both eyes)
        self.left_eyeball = EyeballModel(input_dim=256)
        self.right_eyeball = EyeballModel(input_dim=256)

        # Iris mesh decoder
        self.iris_decoder = IrisMeshDecoder(input_dim=256, n_landmarks=n_iris_landmarks)

        # Gaze ray estimator
        self.gaze_estimator = GazeRayEstimator(hidden_dim=128)

        # Head pose regressor (for completeness)
        self.head_pose_regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)  # 6D rotation representation
        )

    def forward(self, x):
        """
        Forward pass
        Args:
            x: (B, 3, H, W) input images
        Returns:
            dict with all outputs
        """
        # Extract visual features
        from torch.utils.checkpoint import checkpoint
        c0 = checkpoint(self.backbone.stem, x)
        c1 = checkpoint(self.backbone.stages[0], c0)
        c2 = checkpoint(self.backbone.stages[1], c1)
        c3 = checkpoint(self.backbone.stages[2], c2)
        c4 = checkpoint(self.backbone.stages[3], c3)

        features = [c1, c2, c3, c4]
        panet_features = self.panet(features)
        fused = self.fusion(panet_features)

        # Global encoding
        global_features = self.global_encoder(fused)

        # Reconstruct eyeballs
        left_eyeball = self.left_eyeball(global_features)
        right_eyeball = self.right_eyeball(global_features)

        # Combine eyeball information
        eyeball_info = {
            'center': torch.stack([left_eyeball['center'], right_eyeball['center']], dim=1),
            'radius': (left_eyeball['radius'] + right_eyeball['radius']) / 2,
            'left': left_eyeball,
            'right': right_eyeball
        }

        # Decode iris mesh and pupil centers
        iris_output = self.iris_decoder(global_features, eyeball_info)

        # Estimate gaze rays
        gaze_output = self.gaze_estimator(iris_output, eyeball_info)

        # Head pose
        head_pose_6d = self.head_pose_regressor(global_features)

        # Compile outputs
        outputs = {
            # Eyeball reconstruction
            'eyeball_left': left_eyeball,
            'eyeball_right': right_eyeball,
            'eyeball_centers': torch.stack([left_eyeball['center'], right_eyeball['center']], dim=1),

            # Iris and pupil
            'iris_landmarks': iris_output['iris_landmarks'],
            'pupil_centers': iris_output['pupil_centers'],
            'iris_shape_params': iris_output['iris_shape_params'],

            # Gaze outputs
            **gaze_output,

            # Head pose
            'head_pose_6d': head_pose_6d,

            # For compatibility with existing training code
            'pupil_center_3d': torch.stack([
                iris_output['pupil_centers']['left'],
                iris_output['pupil_centers']['right']
            ], dim=1),
            'origin': gaze_output['ray_origin'],
            'direction': gaze_output['ray_direction'],
            'gaze_vector_normalized': gaze_output['gaze_vector'],
            'gaze_point_from_ray': gaze_output['gaze_point_3d']
        }

        return outputs

    def cast_ray_to_plane(self, ray_origin, ray_direction, plane_point, plane_normal):
        """
        Cast ray to intersect with a plane (e.g., screen)
        """
        if plane_point.dim() == 1:
            plane_point = plane_point.unsqueeze(0).expand_as(ray_origin)
        if plane_normal.dim() == 1:
            plane_normal = plane_normal.unsqueeze(0).expand_as(ray_origin)

        numerator = torch.sum(plane_normal * (plane_point - ray_origin), dim=1)
        denominator = torch.sum(plane_normal * ray_direction, dim=1)

        t = numerator / (denominator + 1e-8)
        t = torch.clamp(t, min=0)

        intersection_points = ray_origin + t.unsqueeze(-1) * ray_direction

        return intersection_points, t


# Example usage
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Mock backbone for testing
    class MockBackbone:
        class Stage(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.conv = nn.Conv2d(in_c, out_c, 3, 2, 1)

            def forward(self, x):
                return self.conv(x)

        def __init__(self):
            self.stem = MockBackbone.Stage(3, 64)
            self.stages = nn.ModuleList([
                MockBackbone.Stage(64, 64),
                MockBackbone.Stage(64, 128),
                MockBackbone.Stage(128, 256),
                MockBackbone.Stage(256, 512)
            ])


    backbone = MockBackbone().to(device)
    in_channels_list = [64, 128, 256, 512]

    model = RayNet(
        backbone=backbone,
        in_channels_list=in_channels_list,
        n_iris_landmarks=100,
        panet_out_channels=256
    ).to(device)

    # Test forward pass
    x = torch.randn(2, 3, 448, 448).to(device)
    outputs = model(x)

    print("RayNet outputs:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
        elif isinstance(val, dict):
            print(f"  {key}:")
            for k, v in val.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.shape}")

    # Test ray casting
    screen_point = torch.tensor([0, 0, 500], dtype=torch.float32).to(device)
    screen_normal = torch.tensor([0, 0, 1], dtype=torch.float32).to(device)

    intersection, depth = model.cast_ray_to_plane(
        outputs['ray_origin'],
        outputs['ray_direction'],
        screen_point,
        screen_normal
    )
    print(f"\nScreen intersection: {intersection.shape}")
    print(f"Depth to screen: {depth}")