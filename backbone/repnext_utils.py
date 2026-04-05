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
    # Try TorchScript JIT first, fall back to regular state dict
    try:
        jit_model = torch.jit.load(weight_path, map_location="cpu")
        state_dict = jit_model.state_dict()
    except RuntimeError:
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(f"Cannot extract state_dict from {weight_path}")

    state_dict = {k: v for k, v in state_dict.items() if
                  not k.startswith("head.head") and not k.startswith("head.head_dist")}
    model = create_repnext(model_name=backbone_name, pretrained=False)
    model.load_state_dict(state_dict, strict=False)
    replace_batchnorm(model)
    return model
