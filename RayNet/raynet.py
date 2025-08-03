import torch
import torch.nn as nn

from backbone.repnext_utils import  load_pretrained_repnext
from panet import PANet
from fusion import MultiScaleFusion


from head_pose.model import HeadPoseRegressionHead
from gaze_vector.model import GazeVectorRegressionHead
from gaze_point.model import GazePointRegressionHead
from pupil_center.model import PupilCenterRegressionHead

device = "cuda" if torch.cuda.is_available() else "cpu"


class RayNet(nn.Module):
    def __init__(self, backbone, in_channels_list, panet_out_channels=256):
        super().__init__()

        # Initialize the backbone Note: RepNeXt architecture is being considered for now
        self.backbone = backbone

        # Initialize PANet with all four stage channels
        self.panet = PANet(channels_list=in_channels_list, out_channels=panet_out_channels)

        # P2, P3, P4, P5 are the outputs of PANet
        self.fusion = MultiScaleFusion(in_channels=panet_out_channels, n_scales=4, out_channels=256)

        # --- Head pose regression head ---
        self.head_pose_regression = HeadPoseRegressionHead(in_channels=256, hidden_dim=128, reduction=32)

        # --- Gaze vector regression head ---
        self.gaze_vector_regression = GazeVectorRegressionHead(in_channels=256, hidden_dim=128, reduction=32)

        # --- Gaze point regression head ---
        self.gaze_point_regression = GazePointRegressionHead(in_channels=256, hidden_dim=128, reduction=32)

        # pupil_center_regression = PupilCenterRegressionHead(in_channels=256, hidden_dim=128, reduction=32)
        self.pupil_center_regression = PupilCenterRegressionHead(in_channels=256, hidden_dim=128, reduction=32)


    def forward(self, x):
        c0 = self.backbone.stem(x)  # stride=4
        c1 = self.backbone.stages[0](c0)  # stride=4
        c2 = self.backbone.stages[1](c1)  # stride=8
        c3 = self.backbone.stages[2](c2)  # stride=16
        c4 = self.backbone.stages[3](c3)  # stride=32

        # All four stages used
        features = [c1, c2, c3, c4]

        # --- PANet & Fusion ---
        panet_features = self.panet(features)  # List of [B, C, H, W]
        fused = self.fusion(panet_features)  # [B, 256, H_fused, W_fused]

        # --- Head pose ---
        head_pose_6d = self.head_pose_regression(fused)     # [B, 6] (6D pose vector)
        # --- Gaze vector ---
        gaze_vector = self.gaze_vector_regression(fused)    # [B, 3] (gaze vector)
        # --- Gaze point ---
        gaze_point = self.gaze_point_regression(fused)      # [B, 3] (gaze point in 3D space)
        # --- Pupil center ---
        pupil_center = self.pupil_center_regression(fused) # [B, 2, 3] (left and right pupil centers in 3D space)


        return {
            "head_pose_6d": head_pose_6d,
            "gaze_vector_6d": gaze_vector,  # [B, 3] (gaze vector),
            "gaze_point_3d": gaze_point,  # [B, 3] (gaze point in 3D space),
            "pupil_center_3d": pupil_center,  # [B, 2, 3] (left and right pupil centers in 3D space),
        }


if __name__ == '__main__':
    backbone_name = "repnext_m3"
    weight_path = "./repnext_m3_pretrained.pt"
    repnext_model = load_pretrained_repnext("repnext_m3", weight_path)
    repnext_model = repnext_model.to(device)

    # Channels from each RepNeXt variant
    backbone_channels_dict = {
        'repnext_m0': [40, 80, 160, 320],
        'repnext_m1': [48, 96, 192, 384],
        'repnext_m2': [56, 112, 224, 448],
        'repnext_m3': [64, 128, 256, 512],
        'repnext_m4': [64, 128, 256, 512],
        'repnext_m5': [80, 160, 320, 640],
    }

    in_channels_list = backbone_channels_dict[backbone_name]
    raynet_model = RayNet(repnext_model, in_channels_list, panet_out_channels=256).to(device)

    x = torch.randn(2, 3, 448, 448).to(device)  # <-- put input on the right device!
    features = raynet_model(x)
    print("Head pose 6D shape:", features["head_pose_6d"].shape)  # Should be [2, 6]
    print("Gaze vector 6D shape:", features["gaze_vector_6d"].shape)  # Should be [2, 6]
    print("Gaze point 3D shape:", features["gaze_point_3d"].shape)  # Should be [2, 3]
    print("Pupil center 3D shape:", features["pupil_center_3d"].shape)  # Should be [2, 2, 3]
    print("Fused features shape:", features["fused"].shape)       # Should be [2, 256, H, W]
    for idx, f in enumerate(features["features"]):
        print(f"PANet output P{idx + 2}: {f.shape}")
