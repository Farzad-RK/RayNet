import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFusion(nn.Module):
    """
    Fuses multiple scale feature maps into a unified feature map.
    - Upsamples all feature maps to the highest resolution (P2).
    - Concatenates along channel axis.
    - Reduces channels back to out_channels with 1x1 Conv.
    """
    def __init__(self, in_channels=256, n_scales=4, out_channels=256):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels * n_scales, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, features):
        # features: list of [B, C, H_i, W_i] (from PANet: [P2, P3, P4, P5])
        size = features[0].shape[-2:]  # spatial size of P2
        upsampled = [F.interpolate(f, size=size, mode='nearest') for f in features]
        fused = torch.cat(upsampled, dim=1)   # [B, 256*4, H, W]
        fused = self.act(self.bn(self.conv1x1(fused))) # [B, 256, H, W]
        return fused
