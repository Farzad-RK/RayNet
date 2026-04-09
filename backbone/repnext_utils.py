"""
Utilities for RepNeXt backbone integration in SixDRepNet.

Currently includes:
- replace_batchnorm: Recursively fuses batchnorms using the .fuse() method
- load_pretrained_repnext: Loads a pretrained RepNeXt model and replaces batchnorm layers.

"""
import torch
from backbone.repnext import create_repnext


def replace_batchnorm(net):
    """
    Recursively replaces/fuses BatchNorm layers in the network modules that
    implement a .fuse() method (as in RepNeXt or RepVGG blocks).

    Args:
        net (torch.nn.Module): The model or submodule to fuse.
    """
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            # Replace this child module with its fused version.
            fused = child.fuse()
            setattr(net, child_name, fused)
            # Continue fusing recursively in the fused module
            replace_batchnorm(fused)
        else:
            # Recursively look for fuse-able submodules
            replace_batchnorm(child)

def load_pretrained_repnext(backbone_name, weight_path):
    model = create_repnext(
        backbone_name,
        pretrained=True,
        deploy=False
    )
    model = model.cuda(0)
    jit_model = torch.jit.load(weight_path, map_location=f"cuda:{0}")
    state_dict = jit_model.state_dict()
    model.backbone.load_state_dict(state_dict, strict=False)
    return model
