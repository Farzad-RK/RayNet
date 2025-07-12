# ARGaze-RepNeXt: Gaze Estimation with RepNeXt Backbone on the ARGaze Dataset

This repository contains code for **gaze estimation** using the [ARGaze dataset](https://arxiv.org/abs/2207.02541) and [RepNeXt](https://arxiv.org/abs/2205.15018) backbone. The code is modular, supporting various training and evaluation methods including **LOSO cross-validation** and standard training/testing splits.

---

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Dataset: ARGaze](#dataset-argaze)
- [Model: RepNeXt Backbone](#model-repnext-backbone)
- [Training](#training)
- [Testing/Evaluation](#testingevaluation)
- [Logging and Results](#logging-and-results)
- [Citations](#citations)
- [Acknowledgements](#acknowledgements)
- [References](#references)

---

## Overview

- **ARGaze-RepNeXt** is designed for efficient and reproducible eye gaze estimation using deep learning.
- Modular codebase: easy to swap models, datasets, training strategies.
- Main backbone: [RepNeXt](https://arxiv.org/abs/2205.15018) (2022).
- Dataset: [ARGaze](https://arxiv.org/abs/2207.02541) (2022).

---

## Repository Structure

```

ARGaze-RepNeXt/
│
├── dataset.py         # Dataset loader for ARGaze
├── transforms.py      # Training and testing transforms
├── losses.py          # Custom losses and metrics
├── utils.py           # Helper functions (e.g., BN fusion)
├── train.py           # Training logic and scripts (LOSO and standard)
├── test.py            # Testing/evaluation script
├── config.py          # Centralized configuration
├── repnext.py         # RepNeXt model implementation
├── requirements.txt   # Python dependencies
├── README.md
└── ARGaze\_logs/       # Training/validation logs and checkpoints (created during training)

````

---

## Getting Started

### 1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/ARGaze-RepNeXt.git
cd ARGaze-RepNeXt
````

### 2. **Install Requirements**

```bash
pip install -r requirements.txt
```

### 3. **Download the ARGaze Dataset**

* **Official website:** [ARGaze Dataset](https://github.com/CSG-UESTC/ARGaze)
* Please follow their license and citation requirements.
* Place the extracted data under `./dataset/ARGaze/`

### 4. **Pretrained RepNeXt Weights**

* Download RepNeXt pretrained weights and place under the root directory as `repnext_m3_pretrained.pt` (see [RepNeXt repo](https://github.com/DingXiaoH/RepNeXt-pytorch) for details).

---

## Dataset: ARGaze

* **Citation:**

  > Wu, X., Liu, J., Sun, W., Dong, W., & Wang, X. (2022).
  > **ARGaze: A Large-Scale Gaze Dataset with Angular Resolution and Gaze Estimation Baselines**.
  > *arXiv preprint arXiv:2207.02541*.
  > [Paper link](https://arxiv.org/abs/2207.02541)

* **Structure:**
  Each subject has multiple sessions, each containing images and a `target.npy` of gaze labels.

---

## Model: RepNeXt Backbone

* **Citation:**

  > Ding, X., Zhang, X., Han, J., Ding, G., & Xie, S. (2022).
  > **RepNeXt: Making Convolutional Networks Greater with Re-parameterization**.
  > *arXiv preprint arXiv:2205.15018*.
  > [Paper link](https://arxiv.org/abs/2205.15018)

* **Usage:**
  RepNeXt is used as the backbone for the gaze estimation model.
  The model outputs a 6D representation for gaze direction.

## Model Usage

We now support creating RepNext models by name using the `create_repnext` function from `backbone.repnext`. For example:

```python
from backbone.repnext import create_repnext

model = create_repnext('repnext_m3', pretrained=False, num_classes=6)
```

Available model names: `repnext_m0`, `repnext_m1`, `repnext_m2`, `repnext_m3`, `repnext_m4`, `repnext_m5`.

---

## Training

### Training with LOSO Cross-Validation

The `train.py` script performs Leave-One-Subject-Out (LOSO) cross-validation for gaze estimation using RepNeXt models.

#### Key Features
- 🏗️ **Model Selection**: Choose from multiple RepNeXt variants (m0-m5)
- 🔄 **Resumable Training**: Automatically skips completed folds
- 💾 **Checkpointing**: Saves periodic checkpoints (every 5 epochs)
- 📊 **Comprehensive Logging**: Tracks training loss and validation MAE
- ⏱️ **Time Estimation**: Predicts remaining training time

#### Command Line Arguments
```bash
python train.py \
    --train_samples 2000 \
    --test_samples 1000 \
    --batch_size 32 \
    --epochs 30 \
    --model_type repnext_m3 \
    --pretrained_weights path/to/weights.pt \
    --save_dir ./ARGaze_logs \
    --ckpt_dir ./ARGaze_checkpoints
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--train_samples` | Samples per training subject | 2000 |
| `--test_samples` | Samples per test subject | 1000 |
| `--batch_size` | Training batch size | 32 |
| `--epochs` | Number of training epochs | 30 |
| `--model_type` | RepNeXt variant (m0-m5) | repnext_m3 |
| `--pretrained_weights` | Path to pretrained weights | '' |
| `--save_dir` | Directory for logs and models | ./ARGaze_logs |
| `--ckpt_dir` | Directory for periodic checkpoints | ./ARGaze_checkpoints |

#### Output Files
- `training_log.csv`: CSV log of all training metrics
- `model_{subject}.pth`: Best model for each subject
- `{subject}_epoch{N}.pth`: Periodic checkpoints (every 5 epochs)

#### Example Workflow
1. **Start training**:
```bash
python train.py --model_type repnext_m4 --epochs 50
```

2. **Resume interrupted training**:
```bash
# Automatically skips completed folds
python train.py --model_type repnext_m4 --epochs 50
```

3. **Analyze results**:
```python
import pandas as pd
logs = pd.read_csv("./ARGaze_logs/training_log.csv")
print(logs.groupby('fold')['val_mae'].min().mean())
```

#### Implementation Notes
- Uses cosine annealing learning rate scheduling
- Automatically fuses batchnorm layers for efficiency
- Handles CUDA memory optimization when available
- Provides detailed per-batch progress reporting

---

## Testing/Evaluation

To test a trained model on one or more subjects:

```bash
# Single subject
python test.py --subjects P1 --checkpoint_path ./ARGaze_logs/model_P1.pth

# All subjects (LOSO-style)
python test.py --subjects all --checkpoint_dir ./ARGaze_logs --save_results loso_results.csv
```

Additional arguments:

* `--save_predictions ./angle_preds` saves per-image angle errors.
* See `test.py --help` for all options.

---

## Logging and Results

* **Training logs:**
  CSV logs are saved under `./ARGaze_logs/` (default, can be configured).
* **Checkpoints:**
  Model weights are saved per subject, e.g., `model_P3.pth`, under `./ARGaze_logs/` or specified checkpoint directory.
* **Validation metrics:**
  Mean angular error (MAE) per subject is reported at the end of each training fold and collected in the log file.
* **Example log output:**

  ```
  | fold | epoch | train_loss | val_mae |
  |------|-------|------------|---------|
  | P1   | 1     | 0.328      | 5.12    |
  | P1   | 2     | 0.224      | 4.89    |
  ...
  ```

---

## Citations

If you use this code or results in your work, **please cite both ARGaze and RepNeXt:**

```bibtex
@article{wu2022argaze,
  title={ARGaze: A Large-Scale Gaze Dataset with Angular Resolution and Gaze Estimation Baselines},
  author={Wu, Xiaoming and Liu, Jian and Sun, Wei and Dong, Weijie and Wang, Xinghao},
  journal={arXiv preprint arXiv:2207.02541},
  year={2022}
}

@article{ding2022repnext,
  title={RepNeXt: Making Convolutional Networks Greater with Re-parameterization},
  author={Ding, Xiaohan and Zhang, Xiangyu and Han, Jungong and Ding, Guiguang and Xie, Saining},
  journal={arXiv preprint arXiv:2205.15018},
  year={2022}
}
```

---

## Acknowledgements

* **ARGaze dataset** authors for providing a rich benchmark for gaze estimation.
* **RepNeXt** authors for the backbone and pretrained weights.
* Pytorch, torchvision, and the open-source gaze estimation community.

---

## References

* [ARGaze Dataset GitHub](https://github.com/CSG-UESTC/ARGaze)
* [RepNeXt Official Repo](https://github.com/DingXiaoH/RepNeXt-pytorch)
* [Pytorch](https://pytorch.org/)
* [PIL](https://python-pillow.org/)
* [TQDM](https://tqdm.github.io/)

---

## Contact

Open issues or contact [yourname](mailto:your@email.com) for questions, bugs, or feature requests.

---

```
*If you would like to further customize this README (e.g., add example outputs, badges, colab demo, or figures), let me know!*

---

If you have a specific log file, you can include a *log example* or visualization in the README as well. Let me know if you want that added!
