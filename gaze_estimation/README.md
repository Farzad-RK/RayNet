# Gaze Estimation Module

This module implements gaze estimation using a shared backbone with the head pose estimation model. It's designed to work with the ARGaze dataset and supports both training and evaluation.

## Directory Structure

```
gaze_estimation/
├── data/
│   └── argaze_dataset.py      # Dataset loading and preprocessing
├── models/
│   └── gaze_estimator.py      # Gaze estimation model definition
├── utils/
│   └── losses.py              # Loss functions and metrics
├── train.py                   # Training script
├── test.py                    # Testing and evaluation script
└── README.md                  # This file
```

## Installation

1. Make sure you have the required dependencies installed:
   ```bash
   pip install torch torchvision numpy pandas tqdm
   ```

2. Clone the repository and install the package in development mode:
   ```bash
   git clone <repository-url>
   cd headpose-estimation
   pip install -e .
   ```

## Usage

### Training

To train the gaze estimation model:

```bash
python -m gaze_estimation.train \
    --data_root /path/to/argaze/dataset \
    --output_dir ./output \
    --batch_size 32 \
    --epochs 30 \
    --lr 1e-4 \
    --pretrained_backbone /path/to/pretrained/backbone.pth
```

### Testing

To evaluate a trained model:

```bash
python -m gaze_estimation.test \
    --data_root /path/to/argaze/dataset \
    --checkpoint /path/to/checkpoint.pth \
    --output_file results.csv
```

### Using the Model

You can load and use a trained model in your Python code:

```python
from gaze_estimation.models.gaze_estimator import get_gaze_estimator
import torch

# Load model
model = get_gaze_estimator(pretrained_weights='/path/to/checkpoint.pth')
model.eval()

# Prepare input (example)
input_tensor = torch.randn(1, 3, 224, 224)  # Batch of 1 RGB image

# Forward pass
with torch.no_grad():
    output = model(input_tensor)  # Output is 6D rotation representation
```

## Model Architecture

The gaze estimation model consists of:

1. A shared backbone (RepNeXt) for feature extraction
2. A lightweight head for predicting 6D rotation representation
3. Loss function that optimizes for angular error in gaze direction

## Dataset

The module expects the ARGaze dataset in the following structure:

```
argaze/
├── P1/
│   ├── P1_S1/
│   │   ├── P1_S1_C1/         # Camera 1 images
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   └── target.npy        # Gaze vectors
│   └── ...
├── P2/
└── ...
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
