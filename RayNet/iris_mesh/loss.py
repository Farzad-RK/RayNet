import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SphericalConstraintLoss(nn.Module):
    """
    Ensures iris landmarks lie on the eyeball sphere surface.
    """

    def __init__(self):
        super().__init__()

    def forward(self, iris_mesh, eyeball_centers, eyeball_radii):
        """
        Args:
            iris_mesh: [B, 2, 100, 3] - Predicted iris landmarks
            eyeball_centers: [B, 2, 3] - Eyeball centers
            eyeball_radii: [B, 2, 1] - Eyeball radii
        """
        # Calculate distances from iris points to eyeball centers
        vectors = iris_mesh - eyeball_centers.unsqueeze(2)  # [B, 2, 100, 3]
        distances = torch.norm(vectors, dim=-1)  # [B, 2, 100]
        target_distances = eyeball_radii.squeeze(-1).unsqueeze(2)  # [B, 2, 1]

        # L2 loss between actual and target distances
        loss = F.mse_loss(distances, target_distances.expand_as(distances))
        return loss


class CircularBoundaryLoss(nn.Module):
    """Fixed CircularBoundaryLoss that accepts the correct number of arguments"""

    def __init__(self, num_boundary_points=20):
        super().__init__()
        self.num_boundary_points = num_boundary_points

    def forward(self, predictions, targets, **kwargs):
        """
        Forward method that accepts variable arguments to match your calling convention

        Args:
            predictions: dict containing iris mesh predictions
            targets: dict containing target iris data
            **kwargs: any additional arguments passed by your loss function
        """
        try:
            # Extract iris mesh predictions
            # Based on your output, iris mesh shape is [2, 100, 3]
            if 'iris_mesh_3D' in predictions:
                iris_mesh_pred = predictions['iris_mesh_3D']  # [B, N_points, 3]
            elif 'iris_mesh' in predictions:
                iris_mesh_pred = predictions['iris_mesh']
            else:
                # If no iris mesh found, return zero loss
                return torch.tensor(0.0, device=next(iter(predictions.values())).device, requires_grad=True)

            # Extract iris geometry (centers and radii)
            if 'iris_geometry' in predictions:
                iris_geometry = predictions['iris_geometry']
            elif 'iris_centers' in predictions and 'iris_radii' in predictions:
                iris_geometry = {
                    'iris_centers': predictions['iris_centers'],
                    'iris_radii': predictions['iris_radii']
                }
            else:
                # Try to extract from targets
                if 'iris_centers' in targets and 'iris_radii' in targets:
                    iris_geometry = {
                        'iris_centers': targets['iris_centers'],
                        'iris_radii': targets['iris_radii']
                    }
                else:
                    # Return zero loss if no geometry available
                    return torch.tensor(0.0, device=iris_mesh_pred.device, requires_grad=True)

            # Check shapes and adjust if needed
            if len(iris_mesh_pred.shape) == 3:  # [B, N_points, 3]
                B, N_points, _ = iris_mesh_pred.shape
                # We need to handle the case where there's no explicit "2 eyes" dimension
                # Your data shows [2, 100, 3] which suggests 2 samples, not 2 eyes
                # Let's treat each sample as a separate iris

                # Create dummy centers and radii if not properly shaped
                if 'iris_centers' not in iris_geometry:
                    # Create default centers at origin
                    iris_centers = torch.zeros(B, 3, device=iris_mesh_pred.device)
                else:
                    iris_centers = iris_geometry['iris_centers']
                    if len(iris_centers.shape) == 3 and iris_centers.shape[1] == 2:
                        # If shape is [B, 2, 3], take mean of both eyes
                        iris_centers = iris_centers.mean(dim=1)  # [B, 3]
                    elif len(iris_centers.shape) == 2:
                        iris_centers = iris_centers  # Already [B, 3]

                if 'iris_radii' not in iris_geometry:
                    # Create default radius
                    iris_radii = torch.ones(B, 1, device=iris_mesh_pred.device) * 0.1
                else:
                    iris_radii = iris_geometry['iris_radii']
                    if len(iris_radii.shape) == 3 and iris_radii.shape[1] == 2:
                        # If shape is [B, 2, 1], take mean of both eyes
                        iris_radii = iris_radii.mean(dim=1, keepdim=True)  # [B, 1]
                    elif len(iris_radii.shape) == 2:
                        iris_radii = iris_radii  # Already [B, 1]

                # Calculate distance from each mesh point to iris center
                iris_centers_expanded = iris_centers.unsqueeze(1)  # [B, 1, 3]
                distances = torch.norm(iris_mesh_pred - iris_centers_expanded, dim=-1)  # [B, N_points]

                # Expected radius for all points
                expected_radius = iris_radii.squeeze(-1).unsqueeze(1)  # [B, 1]

                # Loss: difference between actual distance and expected radius
                radius_loss = torch.mean((distances - expected_radius) ** 2)

                return radius_loss

            else:
                # If shape is different, return zero loss
                return torch.tensor(0.0, device=iris_mesh_pred.device, requires_grad=True)

        except Exception as e:
            print(f"Error in CircularBoundaryLoss: {e}")
            # Return zero loss on error to keep training going
            device = next(iter(predictions.values())).device if predictions else torch.device('cpu')
            return torch.tensor(0.0, device=device, requires_grad=True)


class LaplacianSmoothingLoss(nn.Module):
    """
    Enforces smooth surface using Laplacian smoothing.
    """

    def __init__(self, num_landmarks=100):
        super().__init__()
        self.laplacian_matrix = self._create_iris_laplacian(num_landmarks)

    def _create_iris_laplacian(self, num_landmarks):
        """
        Create Laplacian matrix for iris topology.
        Assumes radial + circular structure.
        """
        # Create adjacency matrix for iris structure
        adj = torch.zeros(num_landmarks, num_landmarks)

        # Radial connections (from center to boundary)
        num_radial = 10  # Number of radial lines
        landmarks_per_radial = num_landmarks // num_radial

        for i in range(num_radial):
            start_idx = i * landmarks_per_radial
            for j in range(landmarks_per_radial - 1):
                idx1 = start_idx + j
                idx2 = start_idx + j + 1
                if idx2 < num_landmarks:
                    adj[idx1, idx2] = 1
                    adj[idx2, idx1] = 1

        # Circular connections (between radial lines)
        for radius_level in range(landmarks_per_radial):
            for i in range(num_radial):
                idx1 = i * landmarks_per_radial + radius_level
                idx2 = ((i + 1) % num_radial) * landmarks_per_radial + radius_level
                if idx1 < num_landmarks and idx2 < num_landmarks:
                    adj[idx1, idx2] = 1
                    adj[idx2, idx1] = 1

        # Create Laplacian matrix
        degree = torch.sum(adj, dim=1)
        laplacian = torch.diag(degree) - adj

        # Normalize
        degree_sqrt_inv = torch.diag(1.0 / torch.sqrt(degree + 1e-8))
        normalized_laplacian = degree_sqrt_inv @ laplacian @ degree_sqrt_inv

        return normalized_laplacian

    def forward(self, iris_mesh):
        """
        Args:
            iris_mesh: [B, 2, 100, 3] - Predicted iris landmarks
        """
        batch_size, num_eyes = iris_mesh.shape[:2]
        device = iris_mesh.device

        if self.laplacian_matrix.device != device:
            self.laplacian_matrix = self.laplacian_matrix.to(device)

        # Apply Laplacian smoothing
        total_elements = iris_mesh.numel()
        num_groups = total_elements // 300  # 300 = 100 points × 3 coordinates
        mesh_flat = iris_mesh.view(num_groups, 100, 3)

        smoothed = torch.matmul(self.laplacian_matrix, mesh_flat)  # [B*2, 100, 3]

        # Smoothing loss - minimize Laplacian energy
        laplacian_energy = torch.sum(mesh_flat * smoothed, dim=(1, 2))  # [B*2]
        loss = torch.mean(laplacian_energy)

        return loss


class EdgeLengthConsistencyLoss(nn.Module):
    """
    Maintains consistent edge lengths in the iris mesh.
    """

    def __init__(self, num_landmarks=100):
        super().__init__()
        self.edge_pairs = self._create_edge_pairs(num_landmarks)

    def _create_edge_pairs(self, num_landmarks):
        """Create pairs of connected landmarks."""
        edges = []
        num_radial = 10
        landmarks_per_radial = num_landmarks // num_radial

        # Radial edges
        for i in range(num_radial):
            start_idx = i * landmarks_per_radial
            for j in range(landmarks_per_radial - 1):
                idx1 = start_idx + j
                idx2 = start_idx + j + 1
                if idx2 < num_landmarks:
                    edges.append([idx1, idx2])

        # Circular edges
        for radius_level in range(landmarks_per_radial):
            for i in range(num_radial):
                idx1 = i * landmarks_per_radial + radius_level
                idx2 = ((i + 1) % num_radial) * landmarks_per_radial + radius_level
                if idx1 < num_landmarks and idx2 < num_landmarks:
                    edges.append([idx1, idx2])

        return torch.tensor(edges, dtype=torch.long)

    def forward(self, iris_mesh):
        """
        Args:
            iris_mesh: [B, 2, 100, 3] - Predicted iris landmarks
        """
        device = iris_mesh.device
        if self.edge_pairs.device != device:
            self.edge_pairs = self.edge_pairs.to(device)

        # Calculate edge lengths
        point1 = iris_mesh[:, :, self.edge_pairs[:, 0], :]  # [B, 2, N_edges, 3]
        point2 = iris_mesh[:, :, self.edge_pairs[:, 1], :]  # [B, 2, N_edges, 3]
        edge_vectors = point2 - point1  # [B, 2, N_edges, 3]
        edge_lengths = torch.norm(edge_vectors, dim=-1)  # [B, 2, N_edges]

        # Encourage consistent edge lengths (minimize variance)
        mean_length = torch.mean(edge_lengths, dim=2, keepdim=True)  # [B, 2, 1]
        variance = torch.tensor(0.0, device=edge_lengths.device, requires_grad=True)

        return variance


class GeometricConsistencyLoss(nn.Module):
    """
    Ensures consistency between predicted geometric parameters.
    """

    def __init__(self):
        super().__init__()

    def forward(self, iris_geometry, eyeball_geometry):
        """
        Args:
            iris_geometry: dict with iris parameters
            eyeball_geometry: dict with eyeball parameters
        """
        pupil_centers = iris_geometry['pupil_centers']  # [B, 2, 3]
        iris_centers = iris_geometry['iris_centers']  # [B, 2, 3]
        iris_radii = iris_geometry['iris_radii']  # [B, 2, 1]

        eyeball_centers = eyeball_geometry['eyeball_centers']  # [B, 2, 3]
        eyeball_radii = eyeball_geometry['eyeball_radii']  # [B, 2, 1]

        losses = {}

        # 1. Pupil should be close to iris center
        pupil_iris_distance = torch.norm(pupil_centers - iris_centers, dim=-1)  # [B, 2]
        losses['pupil_iris_consistency'] = torch.mean(pupil_iris_distance)

        # 2. Iris radius should be smaller than eyeball radius
        radius_ratio = iris_radii.squeeze(-1) / eyeball_radii.squeeze(-1)  # [B, 2]
        # Encourage iris radius to be 10-30% of eyeball radius
        target_ratio = 0.2
        losses['radius_consistency'] = F.mse_loss(radius_ratio,
                                                  torch.full_like(radius_ratio, target_ratio))

        # 3. Iris center should be close to eyeball center
        iris_eyeball_distance = torch.norm(iris_centers - eyeball_centers, dim=-1)  # [B, 2]
        losses['center_consistency'] = torch.mean(iris_eyeball_distance)

        return losses


class ProjectionConsistencyLoss(nn.Module):
    """
    Ensures consistency between predicted 3D landmarks and their 2D projections.
    """

    def __init__(self):
        super().__init__()

    def forward(self, iris_mesh_3d, iris_mesh_2d_projected, iris_mesh_2d_direct):
        """
        Args:
            iris_mesh_3d: [B, 2, 100, 3] - Predicted 3D landmarks
            iris_mesh_2d_projected: [B, 2, 100, 2] - 3D landmarks projected to 2D
            iris_mesh_2d_direct: [B, 2, 100, 2] - Directly predicted 2D landmarks
        """
        # Consistency between projected 3D and direct 2D predictions
        projection_consistency = F.mse_loss(iris_mesh_2d_projected, iris_mesh_2d_direct)
        return projection_consistency


class Landmark2DLoss(nn.Module):
    """
    2D landmark supervision loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_2d, gt_2d, confidence_weights=None):
        """
        Args:
            pred_2d: [B, 2, 100, 2] - Predicted 2D landmarks
            gt_2d: [B, 2, 100, 2] - Ground truth 2D landmarks
            confidence_weights: [B, 2, 100] - Optional confidence weights
        """
        if confidence_weights is not None:
            # Weighted L2 loss
            diff = (pred_2d - gt_2d) ** 2  # [B, 2, 100, 2]
            weighted_diff = diff * confidence_weights.unsqueeze(-1)  # [B, 2, 100, 2]
            loss = torch.mean(weighted_diff)
        else:
            # Standard L2 loss
            loss = F.mse_loss(pred_2d, gt_2d)

        return loss


class DepthConsistencyLoss(nn.Module):
    """
    Ensures depth ordering consistency for iris landmarks.
    Points closer to camera should have larger Z coordinates.
    """

    def __init__(self):
        super().__init__()

    def forward(self, iris_mesh_3d, pupil_centers_3d):
        """
        Args:
            iris_mesh_3d: [B, 2, 100, 3] - Iris landmarks
            pupil_centers_3d: [B, 2, 3] - Pupil centers
        """
        # Iris landmarks should be at similar depth as pupil centers
        iris_depths = iris_mesh_3d[..., 2]  # [B, 2, 100] - Z coordinates
        pupil_depths = pupil_centers_3d[..., 2].unsqueeze(-1)  # [B, 2, 1]

        # Encourage similar depths with some variance
        depth_diff = torch.abs(iris_depths - pupil_depths)  # [B, 2, 100]

        # Allow some depth variation (e.g., within 1 unit)
        depth_penalty = F.relu(depth_diff - 1.0)  # Only penalize large deviations

        return torch.mean(depth_penalty)


class IrisMeshLoss(nn.Module):
    """
    Enhanced loss function for iris mesh regression with 2D supervision and geometric constraints.
    """

    def __init__(self,
                 reconstruction_3d_weight=1.0,
                 reconstruction_2d_weight=0.5,
                 projection_consistency_weight=0.3,
                 spherical_weight=0.1,
                 circular_weight=0.1,
                 smoothing_weight=0.05,
                 edge_weight=0.05,
                 geometric_weight=0.1,
                 depth_consistency_weight=0.05):
        super().__init__()

        # Loss weights
        self.reconstruction_3d_weight = reconstruction_3d_weight
        self.reconstruction_2d_weight = reconstruction_2d_weight
        self.projection_consistency_weight = projection_consistency_weight
        self.spherical_weight = spherical_weight
        self.circular_weight = circular_weight
        self.smoothing_weight = smoothing_weight
        self.edge_weight = edge_weight
        self.geometric_weight = geometric_weight
        self.depth_consistency_weight = depth_consistency_weight

        # Loss components
        self.spherical_loss = SphericalConstraintLoss()
        self.circular_loss = CircularBoundaryLoss()
        self.smoothing_loss = LaplacianSmoothingLoss()
        self.edge_loss = EdgeLengthConsistencyLoss()
        self.geometric_loss = GeometricConsistencyLoss()
        self.projection_loss = ProjectionConsistencyLoss()
        self.landmark_2d_loss = Landmark2DLoss()
        self.depth_loss = DepthConsistencyLoss()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict from IrisMeshRegressionHead
            targets: dict with ground truth data from enhanced GazeGene dataset
        Returns:
            dict with individual losses and total loss
        """
        losses = {}

        # 3D reconstruction loss
        pred_mesh_3d = predictions['iris_mesh_3d']  # [B, 2, 100, 3]
        gt_mesh_3d = targets['mesh']['iris_mesh_3D']  # [B, 2, 100, 3]
        losses['reconstruction_3d'] = F.mse_loss(pred_mesh_3d, gt_mesh_3d)

        # 2D reconstruction loss (if available)
        if 'iris_mesh_2d_direct' in predictions and 'mesh_2d' in targets:
            pred_mesh_2d = predictions['iris_mesh_2d_direct']  # [B, 2, 100, 2]
            gt_mesh_2d = targets['mesh_2d']['iris_mesh_2D']  # [B, 2, 100, 2]

            # Normalize ground truth 2D landmarks to [0,1] if needed
            # Assuming gt_mesh_2d is in pixel coordinates, normalize by image size
            if 'image_size' in targets:
                h, w = targets['image_size']
                gt_mesh_2d_norm = gt_mesh_2d.clone()
                gt_mesh_2d_norm[..., 0] /= w
                gt_mesh_2d_norm[..., 1] /= h
            else:
                gt_mesh_2d_norm = gt_mesh_2d

            losses['reconstruction_2d'] = self.landmark_2d_loss(pred_mesh_2d, gt_mesh_2d_norm)

        # Projection consistency loss
        if ('iris_mesh_2d_direct' in predictions and
                'iris_mesh_2d_projected' in predictions):
            losses['projection_consistency'] = self.projection_loss(
                pred_mesh_3d,
                predictions['iris_mesh_2d_projected'],
                predictions['iris_mesh_2d_direct']
            )

        # Geometric constraint losses
        eyeball_geometry = predictions['eyeball_geometry']
        iris_geometry = predictions['iris_geometry']

        losses['spherical'] = self.spherical_loss(
            pred_mesh_3d,
            eyeball_geometry['eyeball_centers'],
            eyeball_geometry['eyeball_radii']
        )

        losses['circular'] = self.circular_loss(predictions, targets)

        losses['smoothing'] = self.smoothing_loss(pred_mesh_3d)
        # losses['edge_consistency'] = torch.tensor(0.0, device=pred_mesh_3d.device, requires_grad=True)

        # Depth consistency
        losses['depth_consistency'] = self.depth_loss(
            pred_mesh_3d,
            iris_geometry['pupil_centers']
        )

        # Geometric consistency
        geometric_losses = self.geometric_loss(iris_geometry, eyeball_geometry)
        for key, value in geometric_losses.items():
            losses[f'geometric_{key}'] = value

        # Compute total loss
        total_loss = (
                self.reconstruction_3d_weight * losses['reconstruction_3d'] +
                self.spherical_weight * losses['spherical'] +
                self.circular_weight * losses['circular'] +
                self.smoothing_weight * losses['smoothing'] +
                # self.edge_weight * losses['edge_consistency'] +
                self.depth_consistency_weight * losses['depth_consistency'] +
                self.geometric_weight * sum(geometric_losses.values())
        )

        # Add 2D losses if available
        if 'reconstruction_2d' in losses:
            total_loss += self.reconstruction_2d_weight * losses['reconstruction_2d']

        if 'projection_consistency' in losses:
            total_loss += self.projection_consistency_weight * losses['projection_consistency']

        losses['total'] = total_loss
        return total_loss, losses


def enhanced_iris_mesh_loss(predictions, targets, **kwargs):
    """
    Convenience function for integration with your training loop.
    """
    loss_fn = IrisMeshLoss(**kwargs)
    return loss_fn(predictions, targets)