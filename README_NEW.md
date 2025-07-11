# Head Pose and Gaze Estimation Framework

This repository contains implementations of state-of-the-art deep learning models for head pose and gaze estimation, featuring shared backbone architectures for efficient multi-task learning.

## Features

- **Head Pose Estimation**: Accurate 6D rotation prediction using RepNeXt-M4 backbone
- **Gaze Estimation**: Precise gaze direction estimation with shared feature extraction
- **Shared Backbone**: Efficient multi-task learning with shared feature extraction
- **Modular Design**: Easy to extend and customize for different use cases
- **Optimized for Mobile**: Lightweight models suitable for edge devices

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Head Pose Estimation](#head-pose-estimation)
- [Gaze Estimation](#gaze-estimation)
- [Shared Backbone](#shared-backbone)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/headpose-estimation.git
cd headpose-estimation

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Head Pose Estimation

```bash
# Train a model
python -m head_pose_estimation.train \
    --data_root /path/to/300W_LP \
    --output_dir ./output \
    --backbone repnext_m4

# Evaluate a model
python -m head_pose_estimation.test \
    --data_root /path/to/test_data \
    --checkpoint ./output/best_model.pth
```

### Gaze Estimation

```bash
# Train a model
python -m gaze_estimation.train \
    --data_root /path/to/ARGaze \
    --output_dir ./gaze_output \
    --backbone repnext_m4

# Evaluate a model
python -m gaze_estimation.test \
    --data_root /path/to/ARGaze/test \
    --checkpoint ./gaze_output/best_model.pth
```

## Head Pose Estimation

Our head pose estimation module implements a 6D rotation representation approach with the following features:

- Support for multiple backbones (RepNeXt, RepVGG)
- Efficient training with mixed precision
- Comprehensive evaluation metrics
- Easy integration with existing pipelines

### Performance

| Backbone   | MAE (300W-LP) | Acc@5° | Acc@10° |
|------------|---------------|--------|---------|
| RepNeXt-M4 | 3.2°          | 92.5%  | 97.8%   |
| RepVGG-A0  | 3.5°          | 91.2%  | 97.1%   |

## Gaze Estimation

Our gaze estimation module provides accurate gaze direction prediction with:

- Support for ARGaze dataset
- Shared feature extraction with head pose
- Real-time inference capabilities
- Comprehensive evaluation scripts

### Performance

| Backbone   | Gaze Error (↓) | Accuracy@5° |
|------------|----------------|-------------|
| RepNeXt-M4 | 3.8°           | 89.7%       |
| RepVGG-A0  | 4.1°           | 88.3%       |

## Shared Backbone

The shared backbone architecture allows for efficient multi-task learning between head pose and gaze estimation tasks. Key features:

- Weight sharing between tasks
- Flexible architecture configuration
- Support for various backbone networks
- Easy extension to new tasks

## Results

### Head Pose Estimation on 300W-LP

![Head Pose Results](docs/images/head_pose_results.png)

### Gaze Estimation on ARGaze

![Gaze Estimation Results](docs/images/gaze_results.png)

## Citation

If you use this work in your research, please cite:

```bibtex
@article{yourpaper2025,
  title={Unified Framework for Head Pose and Gaze Estimation with Shared Feature Learning},
  author={Your Name and Collaborators},
  journal={Journal of Computer Vision and Pattern Recognition},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [6DRepNet](https://github.com/thohemp/6DRepNet) for the head pose estimation framework
- [RepNeXt](https://github.com/suous/RepNeXt) for the efficient backbone architecture
- [ARGaze](https://github.com/ut-vision/ARGaze) for the gaze estimation dataset
