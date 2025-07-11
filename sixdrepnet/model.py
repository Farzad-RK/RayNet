import math

import torch
from torch import nn

from backbone.repvgg import get_RepVGG_func_by_name
import utils

# Import repnext architectures
import backbone.repnext as repnext
from backbone.repnext_utils import replace_batchnorm


class SixDRepNet(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 pretrained=True):
        super(SixDRepNet, self).__init__()
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel

        self.linear_reg = nn.Linear(fea_dim, 6)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        return utils.compute_rotation_matrix_from_ortho6d(x)


class SixDRepNet2(nn.Module):
    def __init__(self, block, layers, fc_layers=1):
        self.inplanes = 64
        super(SixDRepNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.linear_reg = nn.Linear(512*block.expansion,6)
      


        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.linear_reg(x)        
        out = utils.compute_rotation_matrix_from_ortho6d(x)

        return out


class SixDRepNet_RepNeXt(nn.Module):
    """
    SixDRepNet model using any RepNeXt variant as the backbone.

    - Allows flexible swapping between repnext_m0 ... repnext_m5, or any compatible function.
    - Uses the RepNeXt output features and adds a global pooling + regression head for 6D pose.
    - BatchNorm fusion (reparameterization) is supported for efficient deployment.
    - Optionally loads pre-fused (JIT/scripted) backbone weights if provided.
    - The output is a rotation matrix computed via the original utils.compute_rotation_matrix_from_ortho6d.
    """

    def __init__(
            self,
            model_type="repnext_m4",  # e.g., "repnext_m4", "repnext_m5"
            pretrained=False,  # If True, load ImageNet-pretrained weights (if available)
            deploy=False,  # If True, fuse BN layers for inference efficiency
            backbone_weights_path=None  # Optional: path to a fused backbone (e.g., torch.jit.load)
    ):
        super(SixDRepNet_RepNeXt, self).__init__()

        # Dynamically get the model function from repnext
        backbone_fn = getattr(repnext, model_type)

        # 1. Instantiate the backbone with num_classes=0 to disable the default classifier head.
        self.backbone = backbone_fn(pretrained=pretrained, num_classes=0)

        # 2. Optionally fuse batchnorm layers for deployment (makes inference faster).
        if deploy:
            replace_batchnorm(self.backbone)

        # 3. Optionally load custom (e.g. JIT-fused or checkpointed) weights for the backbone.
        if backbone_weights_path is not None:
            print(f"Loading backbone weights from {backbone_weights_path}")
            # If using torch.jit.load, get the state dict
            state_dict = torch.jit.load(backbone_weights_path, map_location="cpu").state_dict()
            self.backbone.load_state_dict(state_dict, strict=False)

        # 4. Determine output feature dimension of backbone (needed for the regression head)
        self.fea_dim = getattr(self.backbone, 'num_features', 512)

        # 5. Define pooling and regression layers
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear_reg = nn.Linear(self.fea_dim, 6)

    def forward(self, x):
        """
        Forward pass:
        - Extract features using RepNeXt backbone (returns (B, C, H, W))
        - Apply global average pooling to get (B, C)
        - Pass through regression head to get 6D pose representation
        - Convert 6D pose to 3x3 rotation matrix using provided utility
        """
        feats = self.backbone.forward_features(x)
        pooled = self.gap(feats)
        pooled = torch.flatten(pooled, 1)
        out6d = self.linear_reg(pooled)
        rotmat = utils.compute_rotation_matrix_from_ortho6d(out6d)
        return rotmat