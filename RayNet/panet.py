import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # YOLOv8 uses SiLU by default

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class PANet(nn.Module):
    def __init__(self, channels_list, out_channels=256):
        """
        Args:
            channels_list (list[int]): Number of channels for backbone feature maps at each scale (from high-res to low-res).
            out_channels (int): Desired number of channels after fusion.
        """
        super().__init__()
        # Lateral layers to unify channels
        self.lateral_convs = nn.ModuleList([
            ConvBNReLU(in_ch, out_channels, kernel_size=1) for in_ch in channels_list
        ])

        # Top-down pathway
        self.top_down_convs = nn.ModuleList([
            ConvBNReLU(out_channels, out_channels, kernel_size=3) for _ in range(len(channels_list)-1)
        ])

        # Bottom-up pathway
        self.bottom_up_convs = nn.ModuleList([
            ConvBNReLU(out_channels, out_channels, kernel_size=3) for _ in range(len(channels_list)-1)
        ])

        self.downsamples = nn.ModuleList([
            ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=2) for _ in range(len(channels_list)-1)
        ])

    def forward(self, features):
        # features: [C3(high-res), C4, C5(low-res)]
        assert len(features) == len(self.lateral_convs), "Number of input features must match lateral conv layers."

        # Apply lateral convolutions
        feats = [l_conv(f) for f, l_conv in zip(features, self.lateral_convs)]

        # Top-down fusion
        td_feats = [feats[-1]]
        for i in range(len(feats)-2, -1, -1):
            size = feats[i].shape[-2:]
            td_feat = F.interpolate(td_feats[-1], size=size, mode='nearest') + feats[i]
            td_feats.append(self.top_down_convs[len(feats)-2 - i](td_feat))
        td_feats = td_feats[::-1]

        # Bottom-up fusion
        bu_feats = [td_feats[0]]
        for i in range(len(td_feats)-1):
            bu_feat = self.downsamples[i](bu_feats[-1]) + td_feats[i+1]
            bu_feats.append(self.bottom_up_convs[i](bu_feat))

        return bu_feats  # Multi-scale fused features
