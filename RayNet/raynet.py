"""
RayNet — Revised Architecture (v2)

Two-task model for gaze estimation and iris/pupil landmark detection:
  Task A: Iris + pupil landmark heatmaps (14 points via soft-argmax on P2)
  Task B: Optical axis regression (pitch/yaw on P5)

Backbone: RepNeXt-M3 (7.8M params)
Neck:     PANet (YOLOv8-style multi-scale fusion)
Attention: Coordinate Attention on P2 (landmarks) and P5 (gaze)

Input:    Normalized eye crop (3 x 224 x 224) via Zhang et al. 2018
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from backbone.repnext_utils import load_pretrained_repnext
from RayNet.panet import PANet
from RayNet.coordatt import CoordinateAttention
from RayNet.heads import IrisPupilLandmarkHead, OpticalAxisHead

device = "cuda" if torch.cuda.is_available() else "cpu"

# Channel configuration for RepNeXt variants
BACKBONE_CHANNELS = {
    'repnext_m0': [40, 80, 160, 320],
    'repnext_m1': [48, 96, 192, 384],
    'repnext_m2': [56, 112, 224, 448],
    'repnext_m3': [64, 128, 256, 512],
    'repnext_m4': [64, 128, 256, 512],
    'repnext_m5': [80, 160, 320, 640],
}


class RayNet(nn.Module):
    """
    RayNet v2: Two-task gaze estimation and landmark detection.

    Architecture:
        RepNeXt -> PANet -> {CoordAtt(P2) -> LandmarkHead,
                             CoordAtt(P5) -> GazeHead}
    """

    def __init__(self, backbone, in_channels_list, panet_out_channels=256,
                 n_landmarks=14):
        super().__init__()

        self.backbone = backbone
        self.panet = PANet(channels_list=in_channels_list,
                           out_channels=panet_out_channels)

        # Coordinate Attention on P2 (landmarks) and P5 (gaze)
        self.coord_att_p2 = CoordinateAttention(panet_out_channels)
        self.coord_att_p5 = CoordinateAttention(panet_out_channels)

        # Task heads
        self.landmark_head = IrisPupilLandmarkHead(
            in_ch=panet_out_channels, n_landmarks=n_landmarks)
        self.gaze_head = OpticalAxisHead(
            in_ch=panet_out_channels, hidden_dim=128)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) normalized eye crop

        Returns:
            dict with:
                'landmark_coords': (B, 14, 2) pixel coords in P2 feature space
                'landmark_heatmaps': (B, 14, H, W) raw logit heatmaps
                'gaze_vector': (B, 3) optical axis unit vector (normalized space)
                'gaze_angles': (B, 2) pitch/yaw in radians
        """
        # Backbone: 4-stage feature extraction
        c0 = checkpoint(self.backbone.stem, x, use_reentrant=False)
        c1 = checkpoint(self.backbone.stages[0], c0, use_reentrant=False)
        c2 = checkpoint(self.backbone.stages[1], c1, use_reentrant=False)
        c3 = checkpoint(self.backbone.stages[2], c2, use_reentrant=False)
        c4 = checkpoint(self.backbone.stages[3], c3, use_reentrant=False)

        features = [c1, c2, c3, c4]

        # PANet multi-scale fusion
        panet_out = self.panet(features)  # [P2, P3, P4, P5]
        p2 = panet_out[0]   # (B, 256, 56, 56) stride=4 for 224 input
        p5 = panet_out[-1]  # (B, 256, 7, 7)   stride=32

        # Coordinate Attention
        p2_att = self.coord_att_p2(p2)
        p5_att = self.coord_att_p5(p5)

        # Task A: Landmark detection on P2
        landmark_coords, landmark_heatmaps = self.landmark_head(p2_att)

        # Task B: Optical axis regression on P5
        gaze_vector, gaze_angles = self.gaze_head(p5_att)

        return {
            'landmark_coords': landmark_coords,
            'landmark_heatmaps': landmark_heatmaps,
            'gaze_vector': gaze_vector,
            'gaze_angles': gaze_angles,
        }


def create_raynet(backbone_name="repnext_m3", weight_path=None, n_landmarks=14):
    """
    Factory function to create RayNet v2.

    Args:
        backbone_name: RepNeXt variant name
        weight_path: path to pretrained backbone weights (JIT format)
        n_landmarks: number of landmarks (default 14: 10 iris + 4 pupil)

    Returns:
        RayNet model on device
    """
    if weight_path is not None:
        backbone = load_pretrained_repnext(backbone_name, weight_path)
    else:
        from backbone.repnext import create_repnext
        backbone = create_repnext(model_name=backbone_name, pretrained=False)

    backbone = backbone.to(device)
    in_channels_list = BACKBONE_CHANNELS[backbone_name]

    model = RayNet(backbone, in_channels_list, n_landmarks=n_landmarks)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    print(f"RayNet v2 created:")
    print(f"  Backbone: {backbone_name} ({BACKBONE_CHANNELS[backbone_name]})")
    print(f"  Landmarks: {n_landmarks}")
    print(f"  Total params: {total_params:.1f}M")
    print(f"  Trainable params: {trainable_params:.1f}M")
    print(f"  Device: {device}")

    return model
