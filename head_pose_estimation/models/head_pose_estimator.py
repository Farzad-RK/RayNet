import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.backbone.repnext import repnext_m3, repnext_m4
from shared.backbone.repvgg import get_RepVGG_func_by_name
from shared.backbone.repnext_utils import replace_batchnorm

class HeadPoseEstimator(nn.Module):
    """
    Head pose estimation model using either RepVGG or RepNeXt as backbone.
    """
    def __init__(self, backbone_name='repnext_m4', pretrained_weights=None):
        super(HeadPoseEstimator, self).__init__()
        
        # Initialize backbone
        if 'repvgg' in backbone_name.lower():
            # For RepVGG
            repvgg_func = get_RepVGG_func_by_name(backbone_name)
            self.backbone = repvgg_func(deploy=False)
            in_features = 512  # Default for RepVGG
        else:
            # For RepNeXt
            if backbone_name == 'repnext_m3':
                self.backbone = repnext_m3(pretrained=False)
            elif backbone_name == 'repnext_m4':
                self.backbone = repnext_m4(pretrained=False)
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")
            in_features = self.backbone.num_features
            
            # Replace batch norm with custom version
            replace_batchnorm(self.backbone)
        
        # Regression head for 6D rotation
        self.rotation = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 6)  # 6D rotation
        )
        
        # Load pretrained weights if provided
        if pretrained_weights:
            self.load_pretrained_weights(pretrained_weights)
    
    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]  # Handle case where backbone returns tuple
        rotation = self.rotation(features)
        return rotation
    
    def load_pretrained_weights(self, weights_path):
        """Load pretrained weights from a checkpoint."""
        checkpoint = torch.load(weights_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle loading from different checkpoint formats
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        
        self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained weights from {weights_path}")

def get_head_pose_estimator(backbone_name='repnext_m4', pretrained_weights=None):
    """
    Factory function to create a head pose estimator model.
    
    Args:
        backbone_name: Name of the backbone architecture
        pretrained_weights: Path to pretrained weights
        
    Returns:
        HeadPoseEstimator: Initialized head pose estimation model
    """
    model = HeadPoseEstimator(backbone_name=backbone_name, 
                            pretrained_weights=pretrained_weights)
    return model
