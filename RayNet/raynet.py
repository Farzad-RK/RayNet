import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from backbone.repnext_utils import load_pretrained_repnext
from panet import PANet
from fusion import MultiScaleFusion
from coordatt import CoordAtt
from EyeFLAME.model import EyeFLAME_DepthAware

device = "cuda" if torch.cuda.is_available() else "cpu"


class RayNet_DepthAware(nn.Module):
    """
    RayNet with Depth-Aware EyeFLAME model
    Uses weak perspective projection to handle depth ambiguity
    """

    def __init__(self, backbone, in_channels_list, panet_out_channels=256):
        super().__init__()

        # Initialize the backbone
        self.backbone = backbone

        # Initialize PANet with all four stage channels
        self.panet = PANet(channels_list=in_channels_list, out_channels=panet_out_channels)

        # P2, P3, P4, P5 are the outputs of PANet
        self.fusion = MultiScaleFusion(in_channels=panet_out_channels, n_scales=4, out_channels=256)

        # --- NEW: Depth-Aware Eye FLAME model ---
        self.eye_FLAME_model = EyeFLAME_DepthAware(
            in_channels=256,
            hidden_dim=128,
            reduction=32,
        )

    def forward(self, x, subject_params=None, camera_params=None):
        """
        Forward pass through RayNet with depth-aware predictions

        Args:
            x: [B, 3, H, W] input images
            subject_params: Subject-specific anatomical parameters
            camera_params: Camera intrinsics for projection

        Returns:
            dict: Eye structure predictions with weak perspective
        """

        # Backbone feature extraction with gradient checkpointing
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

        # --- Depth-Aware Eye Structure Prediction ---
        eye_structures = self.eye_FLAME_model(
            fused,
            subject_params=subject_params,
            camera_params=camera_params,
        )

        return eye_structures


def create_raynet_model_with_depth_aware(backbone_name="repnext_m3", weight_path="./repnext_m3_pretrained.pt"):
    """
    Factory function to create RayNet with Depth-Aware EyeFLAME model

    Args:
        backbone_name: Name of RepNeXt variant
        weight_path: Path to pretrained weights

    Returns:
        RayNet_DepthAware model
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

    # Create RayNet with Depth-Aware model
    model = RayNet_DepthAware(repnext_model, in_channels_list, panet_out_channels=256)
    model = model.to(device)

    print(f"Created RayNet with Depth-Aware EyeFLAME")
    print(f"  Backbone: {backbone_name}")
    print(f"  Channels: {in_channels_list}")
    print(f"  Device: {device}")

    return model


# ============== Additional Utilities ==============

def verify_model_output(model, device):
    """
    Verify the depth-aware model produces expected outputs
    """
    print("\n=== Model Output Verification ===")

    # Create dummy inputs
    batch_size = 2
    dummy_img = torch.randn(batch_size, 3, 448, 448).to(device)

    dummy_subject_params = {
        'eyeball_radius': torch.ones(batch_size, 1).to(device) * 1.2,
        'iris_radius': torch.ones(batch_size, 1).to(device) * 0.6,
        'cornea_radius': torch.ones(batch_size, 1).to(device) * 0.78,
        'cornea2center': torch.ones(batch_size, 1).to(device) * 0.5,
        'L_kappa': torch.zeros(batch_size, 3).to(device),
        'R_kappa': torch.zeros(batch_size, 3).to(device),
    }

    dummy_camera_params = {
        'intrinsic_matrix': torch.tensor([
            [2000, 0, 1280],
            [0, 2000, 720],
            [0, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1).to(device)
    }

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_img, dummy_subject_params, dummy_camera_params)

    # Check outputs
    expected_keys = [
        'eyeball_centers', 'pupil_centers', 'iris_landmarks_100',
        'optical_axes', 'visual_axes', 'head_gaze_direction',
        'projections_2d', 'weak_perspective'
    ]

    print("Output keys:", list(output.keys()))

    for key in expected_keys:
        if key in output:
            if isinstance(output[key], dict):
                print(f"  {key}: dict with keys {list(output[key].keys())}")
            elif isinstance(output[key], torch.Tensor):
                print(f"  {key}: shape {output[key].shape}")
            else:
                print(f"  {key}: {type(output[key])}")

    # Check weak perspective parameters
    if 'weak_perspective' in output:
        wp = output['weak_perspective']
        print("\nWeak Perspective Parameters:")
        print(f"  Scale: {wp['scale'].mean().item():.3f} (expected ~1.0)")
        print(f"  Translation: {wp['translation_2d'].mean(dim=0).tolist()}")

    # Check 3D coordinates are in reasonable range
    if 'eyeball_centers' in output:
        eyeball_z = output['eyeball_centers'][:, :, 2]  # Z coordinate (depth)
        print(f"\nEyeball depth range: {eyeball_z.min().item():.1f} - {eyeball_z.max().item():.1f} cm")
        print(f"  (Expected around 500cm for 5m camera distance)")

    print("\n=== Verification Complete ===")

    return output


def analyze_multi_view_consistency(model, batch, device):
    """
    Analyze consistency across multiple camera views
    Important for resolving depth ambiguity
    """
    batch_size = batch['img'].shape[0]

    if batch_size % 9 != 0:
        print("Warning: Batch size not divisible by 9 (not multi-view)")
        return

    num_subjects = batch_size // 9

    print(f"\n=== Multi-View Consistency Analysis ===")
    print(f"Processing {num_subjects} subjects with 9 views each")

    # Process each subject
    for subj_idx in range(num_subjects):
        start_idx = subj_idx * 9
        end_idx = start_idx + 9

        # Get predictions for all 9 views
        predictions = []
        for view_idx in range(start_idx, end_idx):
            view_img = batch['img'][view_idx:view_idx + 1]
            # ... forward pass for single view
            # predictions.append(output)

        # Analyze consistency
        # ... compute variance across views

    print("=== Analysis Complete ===")


# ============== Debugging Tools ==============

class DebugHook:
    """
    Debug hook to monitor gradient flow and activations
    """

    def __init__(self, name):
        self.name = name
        self.gradients = []
        self.activations = []

    def register(self, module):
        module.register_forward_hook(self.forward_hook)
        module.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.activations.append({
                'mean': output.mean().item(),
                'std': output.std().item(),
                'shape': output.shape
            })

    def backward_hook(self, module, grad_input, grad_output):
        if isinstance(grad_output[0], torch.Tensor):
            self.gradients.append({
                'mean': grad_output[0].mean().item(),
                'std': grad_output[0].std().item(),
                'norm': grad_output[0].norm().item()
            })

    def print_stats(self):
        print(f"\n{self.name} Statistics:")
        if self.activations:
            latest_act = self.activations[-1]
            print(f"  Activation: mean={latest_act['mean']:.4f}, std={latest_act['std']:.4f}")
        if self.gradients:
            latest_grad = self.gradients[-1]
            print(f"  Gradient: mean={latest_grad['mean']:.4f}, norm={latest_grad['norm']:.4f}")


def add_debug_hooks(model):
    """
    Add debug hooks to monitor training
    """
    hooks = {}

    # Add hooks to key layers
    hooks['weak_perspective'] = DebugHook('WeakPerspective')
    hooks['weak_perspective'].register(model.eye_FLAME_model.weak_perspective_head)

    hooks['eyeball_centers'] = DebugHook('EyeballCenters')
    hooks['eyeball_centers'].register(model.eye_FLAME_model.eyeball_centers_normalized)

    return hooks