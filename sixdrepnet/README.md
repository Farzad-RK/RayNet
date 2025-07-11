# SixDRepNet - 6D Rotation Representation for Head Pose Estimation

This module implements the SixDRepNet model for head pose estimation, which predicts 6D rotation representation for head pose estimation.

## Attribution

This implementation is based on the original SixDRepNet paper and code:

- **Paper**: [6D Rotation Representation For Unconstrained Head Pose Estimation](https://ieeexplore.ieee.org/document/9873203)
- **Original Repository**: [thohemp/6DRepNet](https://github.com/thohemp/6DRepNet)
- **License**: MIT License

## Model Architecture

The model uses either RepVGG or RepNeXt as the backbone network, followed by regression heads for rotation prediction. The key components are:

- **Backbone**: Feature extraction using RepVGG or RepNeXt
- **Regression Heads**: Predict 6D rotation representation
- **Loss Function**: Combines geodesic loss for rotation and L1 loss for translation

## Usage

### Training

To train the model:

```bash
python sixdrepnet/train.py \
    --dataset_path /path/to/dataset \
    --output_dir ./output \
    --batch_size 32 \
    --epochs 100 \
    --backbone repnext_m4  # or repvgg_a0
```

### Evaluation

To evaluate a trained model:

```bash
python sixdrepnet/test.py \
    --dataset_path /path/to/test/dataset \
    --snapshot /path/to/checkpoint.pth \
    --output_dir ./results
```

### Using the Model

```python
from sixdrepnet.model import SixDRepNet
import torch

# Initialize model
model = SixDRepNet(backbone_name='repnext_m4')
model.eval()

# Prepare input (example)
input_tensor = torch.randn(1, 3, 224, 224)  # Batch of 1 RGB image

# Forward pass
with torch.no_grad():
    R = model(input_tensor)  # Output is 6D rotation representation
```

## Shared Backbone

The backbone implementation is shared with other modules in the `shared/backbone` directory, which contains:

- RepVGG implementation
- RepNeXt implementation
- Utility functions for model conversion and training

## License

This code is released under the MIT License, following the original repository's license.
