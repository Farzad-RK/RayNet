import torch
import torch.nn as nn
from backbone.repnext_utils import replace_batchnorm
from backbone.repnext import create_repnext
from panet import PANet

device = "cuda" if torch.cuda.is_available() else "cpu"


class RayNet(nn.Module):
    def __init__(self, backbone_name='repnext_m3', pretrained=True, panet_out_channels=256):
        super().__init__()

        # ---------- RepNext Loading ----------
        jit_model = torch.jit.load("repnext_m3_pretrained.pt", map_location="cpu")
        state_dict = jit_model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if
                      not k.startswith("head.head") and not k.startswith("head.head_dist")}
        model = create_repnext(model_name=backbone_name, pretrained=False)
        model.load_state_dict(state_dict, strict=False)
        replace_batchnorm(model)
        model = model.to(device)
        self.backbone = model

        # Channels from each RepNeXt variant
        backbone_channels_dict = {
            'repnext_m0': [40, 80, 160, 320],
            'repnext_m1': [48, 96, 192, 384],
            'repnext_m2': [56, 112, 224, 448],
            'repnext_m3': [64, 128, 256, 512],
            'repnext_m4': [64, 128, 256, 512],
            'repnext_m5': [80, 160, 320, 640],
        }
        print(backbone_channels_dict[backbone_name])

        in_channels_list = backbone_channels_dict[backbone_name]

        # Initialize PANet with all four stage channels
        self.panet = PANet(channels_list=in_channels_list, out_channels=panet_out_channels)

    def forward(self, x):
        c0 = self.backbone.stem(x)  # stride=4
        c1 = self.backbone.stages[0](c0)  # stride=4
        c2 = self.backbone.stages[1](c1)  # stride=8
        c3 = self.backbone.stages[2](c2)  # stride=16
        c4 = self.backbone.stages[3](c3)  # stride=32

        # All four stages used
        features = [c1, c2, c3, c4]

        panet_features = self.panet(features)

        return panet_features  # List of multi-scale fused features


if __name__ == '__main__':
    model = RayNet('repnext_m3', pretrained=False)
    model = model.to(device)  # (optional, you already did this internally)
    x = torch.randn(2, 3, 448, 448).to(device)  # <-- put input on the right device!
    features = model(x)
    for idx, f in enumerate(features):
        print(f"PANet output P{idx + 2}: {f.shape}")