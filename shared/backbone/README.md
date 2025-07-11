# Shared Backbone Models

This directory contains shared backbone models used across different tasks like head pose estimation and gaze estimation.

## Attribution

The backbone implementations are based on the following repositories:

1. **RepVGG**
   - Original Paper: [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
   - Original Implementation: [DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)

2. **RepNeXt**
   - Original Paper: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
   - Original Implementation: [facebookresearch/ResNeXt](https://github.com/facebookresearch/ResNeXt)

## Models

- `repvgg.py`: Implementation of RepVGG backbone
- `repnext.py`: Implementation of RepNeXt backbone
- `se_block.py`: Squeeze-and-Excitation block implementation
- `repnext_utils.py`: Utility functions for RepNeXt

## Usage

These backbones can be imported and used as follows:

```python
from shared.backbone.repnext import RepNeXt
from shared.backbone.repvgg import RepVGG

# Initialize model
model = RepNeXt(num_blocks=[3, 4, 6, 3], cardinality=32)
# or
model = RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5])
```

## License

The code in this directory is released under the MIT License, following the original repositories' licenses.
