"""
Utilities for RepNeXt backbone integration in SixDRepNet.

Currently includes:
- replace_batchnorm: Recursively fuses batchnorms using the .fuse() method
  (if implemented by RepNeXt blocks).

You can add more RepNeXt-specific helpers here as your project grows.
"""

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
