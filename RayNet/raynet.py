import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from backbone.repnext_utils import load_pretrained_repnext
from panet import PANet
from fusion import MultiScaleFusion
from coordatt import CoordAtt
from EyeFLAME.model import EyeFLAME_Model


device = "cuda" if torch.cuda.is_available() else "cpu"


class RayNet(nn.Module):
    def __init__(self, backbone, in_channels_list, panet_out_channels=256):
        super().__init__()

        # Initialize the backbone
        self.backbone = backbone

        # Coordinate attention module
        # self.coord_att_module = CoordAtt(256, 256, reduction=32)

        # Initialize PANet with all four stage channels
        self.panet = PANet(channels_list=in_channels_list, out_channels=panet_out_channels)

        # P2, P3, P4, P5 are the outputs of PANet
        self.fusion = MultiScaleFusion(in_channels=panet_out_channels, n_scales=4, out_channels=256)

        # --- NEW: Iris mesh regression head ---
        self.eye_FLAME_model = EyeFLAME_Model(
            in_channels=256,
            hidden_dim=128,
            reduction=32,
        )

    def forward(self, x, subject_params=None,camera_params=None):
        # Backbone feature extraction
        c0 = checkpoint(self.backbone.stem, x)  # stride=4
        c1 = checkpoint(self.backbone.stages[0], c0)  # stride=4
        c2 = checkpoint(self.backbone.stages[1], c1)  # stride=8
        c3 = checkpoint(self.backbone.stages[2], c2)  # stride=16
        c4 = checkpoint(self.backbone.stages[3], c3)  # stride=32

        # All four stages used
        features = [c1, c2, c3, c4]
        # Apply coordinate attention before feature fusion
        # features = self.coord_att_module(features)
        # --- PANet & Fusion ---
        panet_features = self.panet(features)  # List of [B, C, H, W]
        fused = self.fusion(panet_features)  # [B, 256, H_fused, W_fused]


        # --- Iris mesh prediction with 2D supervision ---
        eye_FLAME_results = self.eye_FLAME_model(
            fused,
            subject_params=subject_params,
            camera_params=camera_params,
        )
        """
        Kinematic parameters for eye structures
        eye_structures = 
         {
            'eyeball_centers': final_eyeball_centers,  # [B, 2, 3] in cm
            'pupil_centers': pupil_centers,  # [B, 2, 3] in cm
            'iris_landmarks_100': final_iris_landmarks,  # [B, 200, 3] in cm (100 per eye)
            'optical_axes': optical_axes,  # [B, 2, 3] unit vectors
            'visual_axes': visual_axes,  # [B, 2, 3] unit vectors
            'head_gaze_direction': θ_joints['head_gaze']  # [B, 3] unit vector,
        }
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
        """
        return eye_FLAME_results



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