import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Import your existing modules
from backbone.repnext_utils import load_pretrained_repnext
from panet import PANet
from fusion import MultiScaleFusion
from head_pose.model import HeadPoseRegressionHead
from utils import ortho6d_to_rotmat

# Import new iris mesh components
from iris_mesh.model import IrisMeshRegressionHead
from iris_mesh.loss import iris_mesh_loss
from loss import multiview_headpose_losses

device = "cuda" if torch.cuda.is_available() else "cpu"


class RayNet(nn.Module):
    def __init__(self, backbone, in_channels_list, panet_out_channels=256):
        super().__init__()

        # Initialize the backbone
        self.backbone = backbone

        # Initialize PANet with all four stage channels
        self.panet = PANet(channels_list=in_channels_list, out_channels=panet_out_channels)

        # P2, P3, P4, P5 are the outputs of PANet
        self.fusion = MultiScaleFusion(in_channels=panet_out_channels, n_scales=4, out_channels=256)

        # --- Head pose regression head (keep existing) ---
        self.head_pose_regression = HeadPoseRegressionHead(in_channels=256, hidden_dim=128, reduction=32)

        # --- NEW: Iris mesh regression head ---
        self.iris_mesh_regression = IrisMeshRegressionHead(
            in_channels=256,
            hidden_dim=128,
            reduction=32,
            num_landmarks=100
        )

        # Inject coordinate attention if you have it
        # self.iris_mesh_regression.set_coord_attention(your_coord_att_module)

    def forward(self, x):
        # Backbone feature extraction
        c0 = checkpoint(self.backbone.stem, x)  # stride=4
        c1 = checkpoint(self.backbone.stages[0], c0)  # stride=4
        c2 = checkpoint(self.backbone.stages[1], c1)  # stride=8
        c3 = checkpoint(self.backbone.stages[2], c2)  # stride=16
        c4 = checkpoint(self.backbone.stages[3], c3)  # stride=32

        # All four stages used
        features = [c1, c2, c3, c4]

        # --- PANet & Fusion ---
        panet_features = self.panet(features)  # List of [B, C, H, W]
        fused = self.fusion(panet_features)  # [B, 256, H_fused, W_fused]

        # --- Head pose prediction ---
        head_pose_6d = self.head_pose_regression(fused)  # [B, 6] (6D pose vector)

        # --- Iris mesh prediction ---
        iris_results = self.iris_mesh_regression(fused)

        # Extract key outputs for ray computation
        pupil_centers_3d = iris_results['pupil_centers_3d']  # [B, 2, 3]
        iris_mesh_3d = iris_results['iris_mesh_3d']  # [B, 2, 100, 3]

        # Prepare output dictionary
        return {
            # Core predictions
            "head_pose_6d": head_pose_6d,
            "iris_mesh_3d": iris_mesh_3d,
            "pupil_centers_3d": pupil_centers_3d,

            # Geometric parameters for analysis/loss computation
            "eyeball_geometry": iris_results['eyeball_geometry'],
            "iris_geometry": iris_results['iris_geometry'],
            "spherical_rays": iris_results['spherical_rays'],

            # Features for potential extension
            "fused_features": fused,
            "panet_features": panet_features,
        }


class RayNetLoss(nn.Module):
    """
    Combined loss function for RayNet multi-task learning.
    """

    def __init__(self,
                 head_pose_weight=1.0,
                 iris_mesh_weight=1.0,
                 iris_loss_config=None):
        super().__init__()

        self.head_pose_weight = head_pose_weight
        self.iris_mesh_weight = iris_mesh_weight

        # Default iris loss configuration
        if iris_loss_config is None:
            iris_loss_config = {
                'reconstruction_weight': 1.0,
                'spherical_weight': 0.1,
                'circular_weight': 0.1,
                'smoothing_weight': 0.05,
                'edge_weight': 0.05,
                'geometric_weight': 0.1
            }
        self.iris_loss_config = iris_loss_config

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict from RayNet forward pass
            targets: dict with ground truth data from GazeGene dataset
        Returns:
            total_loss, individual_losses
        """
        losses = {}

        # 1. Head pose loss (existing)
        if "head_pose_6d" in predictions and "head_pose" in targets:
            # Convert head pose to rotation matrix for geodesic loss
            pred_6d = predictions["head_pose_6d"]  # [B, 6]
            gt_rotmat = targets["head_pose"]["R"]  # [B, 3, 3]

            # For multi-view, we might need to expand dimensions
            if len(pred_6d.shape) == 2:  # [B, 6]
                pred_6d = pred_6d.unsqueeze(1)  # [B, 1, 6]
            if len(gt_rotmat.shape) == 3:  # [B, 3, 3]
                gt_rotmat = gt_rotmat.unsqueeze(1)  # [B, 1, 3, 3]

            head_pose_losses = multiview_headpose_losses(pred_6d, gt_rotmat)
            losses['head_pose_accuracy'] = head_pose_losses['accuracy']
            losses['head_pose_consistency'] = head_pose_losses['consistency']

        # 2. Iris mesh loss (new)
        if "iris_mesh_3d" in predictions:
            iris_total_loss, iris_losses = iris_mesh_loss(
                predictions, targets, **self.iris_loss_config
            )
            losses['iris_total'] = iris_total_loss

            # Add individual iris losses for monitoring
            for key, value in iris_losses.items():
                if key != 'total':
                    losses[f'iris_{key}'] = value

        # Compute weighted total loss
        total_loss = 0.0
        if 'head_pose_accuracy' in losses:
            total_loss += self.head_pose_weight * (
                    losses['head_pose_accuracy'] +
                    0.1 * losses.get('head_pose_consistency', 0)
            )

        if 'iris_total' in losses:
            total_loss += self.iris_mesh_weight * losses['iris_total']

        losses['total'] = total_loss
        return total_loss, losses


def create_raynet_model(backbone_name="repnext_m3", weight_path="./repnext_m3_pretrained.pt"):
    """
    Factory function to create RayNet model with iris mesh capabilities.
    """
    # Load backbone
    repnext_model = load_pretrained_repnext(backbone_name, weight_path)
    repnext_model = repnext_model.to(device)

    # Channel configuration for different RepNeXt variants
    backbone_channels_dict = {
        'repnext_m0': [40, 80, 160, 320],
        'repnext_m1': [48, 96, 192, 384],
        'repnext_m2': [56, 112, 224, 448],
        'repnext_m3': [64, 128, 256, 512],
        'repnext_m4': [64, 128, 256, 512],
        'repnext_m5': [80, 160, 320, 640],
    }

    in_channels_list = backbone_channels_dict[backbone_name]

    # Create RayNet model
    model = RayNet(repnext_model, in_channels_list, panet_out_channels=256)
    model = model.to(device)

    return model


# Training utility functions
def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics for iris mesh and head pose.
    """
    metrics = {}

    # Head pose metrics
    if "head_pose_6d" in predictions and "head_pose" in targets:
        pred_rotmat = ortho6d_to_rotmat(predictions["head_pose_6d"])
        gt_rotmat = targets["head_pose"]["R"]

        # Angular error in degrees
        trace = torch.sum(pred_rotmat * gt_rotmat, dim=(1, 2))
        cos_angle = (trace - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1, 1)
        angle_error = torch.acos(cos_angle) * 180 / 3.14159
        metrics['head_pose_error_deg'] = torch.mean(angle_error)

    # Iris mesh metrics
    if "iris_mesh_3d" in predictions:
        pred_mesh = predictions["iris_mesh_3d"]  # [B, 2, 100, 3]
        gt_mesh = targets["mesh"]["iris_mesh_3D"]  # [B, 2, 100, 3]

        # L2 distance per landmark
        l2_errors = torch.norm(pred_mesh - gt_mesh, dim=-1)  # [B, 2, 100]
        metrics['iris_mesh_l2_error'] = torch.mean(l2_errors)
        metrics['iris_mesh_l2_std'] = torch.std(l2_errors)

        # Per-eye errors
        metrics['iris_mesh_left_error'] = torch.mean(l2_errors[:, 0, :])
        metrics['iris_mesh_right_error'] = torch.mean(l2_errors[:, 1, :])

    return metrics


# Example usage in training loop
def training_step_example(model, batch, loss_fn, optimizer):
    """
    Example training step for RayNet with iris mesh.
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    images = batch['img']  # [B, 3, H, W]
    predictions = model(images)

    # Compute loss
    total_loss, individual_losses = loss_fn(predictions, batch)

    # Backward pass
    total_loss.backward()
    optimizer.step()

    # Compute metrics
    with torch.no_grad():
        metrics = compute_metrics(predictions, batch)

    return {
        'total_loss': total_loss.item(),
        'losses': {k: v.item() for k, v in individual_losses.items()},
        'metrics': {k: v.item() for k, v in metrics.items()}
    }


if __name__ == '__main__':
    # Test the integrated model
    model = create_raynet_model("repnext_m3", "./repnext_m3_pretrained.pt")

    # Create loss function
    loss_fn = RayNetLoss(
        head_pose_weight=1.0,
        iris_mesh_weight=1.0,
        iris_loss_config={
            'reconstruction_weight': 1.0,
            'spherical_weight': 0.1,
            'circular_weight': 0.1,
            'smoothing_weight': 0.05,
            'edge_weight': 0.05,
            'geometric_weight': 0.1
        }
    )

    # Test forward pass
    x = torch.randn(2, 3, 448, 448).to(device)
    with torch.no_grad():
        outputs = model(x)

    print("=== RayNet with Iris Mesh Output Shapes ===")
    print(f"Head pose 6D: {outputs['head_pose_6d'].shape}")  # [2, 6]
    print(f"Iris mesh 3D: {outputs['iris_mesh_3d'].shape}")  # [2, 2, 100, 3]
    print(f"Pupil centers 3D: {outputs['pupil_centers_3d'].shape}")  # [2, 2, 3]
    print(f"Fused features: {outputs['fused_features'].shape}")  # [2, 256, H, W]

    print("\n=== Geometric Parameters ===")
    eyeball_geo = outputs['eyeball_geometry']
    print(f"Eyeball centers: {eyeball_geo['eyeball_centers'].shape}")  # [2, 2, 3]
    print(f"Eyeball radii: {eyeball_geo['eyeball_radii'].shape}")  # [2, 2, 1]

    iris_geo = outputs['iris_geometry']
    print(f"Iris centers: {iris_geo['iris_centers'].shape}")  # [2, 2, 3]
    print(f"Iris radii: {iris_geo['iris_radii'].shape}")  # [2, 2, 1]

    print(f"Spherical rays: {outputs['spherical_rays'].shape}")  # [2, 2, 100, 2]