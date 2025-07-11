import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.backbone.repnext import RepNeXt

class GazeEstimator(nn.Module):
    def __init__(self, backbone=None, pretrained_weights=None):
        super(GazeEstimator, self).__init__()
        
        # Use provided backbone or create a new one
        if backbone is not None:
            self.backbone = backbone
            # Freeze backbone parameters if using pre-trained weights
            if pretrained_weights is not None:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        else:
            # Create a new backbone if none provided
            self.backbone = RepNeXt()
            
        # Gaze estimation head (6D rotation output)
        in_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 6)  # 6D rotation output
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)

def get_gaze_estimator(backbone=None, pretrained_weights=None):
    """
    Factory function to create a gaze estimator model.
    
    Args:
        backbone: Optional pre-initialized backbone model
        pretrained_weights: Path to pre-trained weights for the gaze estimator
        
    Returns:
        GazeEstimator: Initialized gaze estimation model
    """
    model = GazeEstimator(backbone=backbone)
    
    if pretrained_weights is not None:
        state_dict = torch.load(pretrained_weights, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    
    return model
