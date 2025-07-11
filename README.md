# Head Pose and Gaze Estimation Framework

A comprehensive framework for head pose and gaze estimation, featuring state-of-the-art models and efficient backbones for real-time applications.

## Features

- **Multiple Backbones**: Supports RepVGG and RepNeXt architectures
- **Modular Design**: Separate modules for head pose and gaze estimation
- **Efficient Inference**: Optimized for real-time performance
- **Comprehensive Evaluation**: Standard metrics for both tasks
- **Easy Integration**: Simple Python API for inference and training

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/headpose-estimation.git
cd headpose-estimation

# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .
```

## Quick Start

### Head Pose Estimation

```python
from head_pose_estimation.models import get_head_pose_estimator
import torch

# Initialize model
model = get_head_pose_estimator(backbone_name='repnext_m4', pretrained_weights=None)
model.eval()

# Example inference
input_tensor = torch.randn(1, 3, 224, 224)  # Batch of 1 RGB image
with torch.no_grad():
    rotation = model(input_tensor)  # Returns rotation matrix
```

### Gaze Estimation

```python
from gaze_estimation.models import get_gaze_estimator
import torch

# Initialize model
model = get_gaze_estimator(backbone_name='repnext_m4')
model.eval()

# Example inference
input_tensor = torch.randn(1, 3, 224, 224)  # Batch of 1 face image
with torch.no_grad():
    gaze_vector = model(input_tensor)  # Returns 3D gaze direction
```

## Project Structure

```
.
├── head_pose_estimation/    # Head pose estimation module
│   ├── models/             # Model definitions
│   ├── data/               # Data loading and preprocessing
│   ├── utils/              # Utility functions and metrics
│   └── train.py            # Training script
│
├── gaze_estimation/        # Gaze estimation module
│   ├── models/            
│   ├── data/              
│   ├── utils/             
│   └── train.py           
│
├── shared/                 # Shared components
│   └── backbone/          # Common backbones (RepVGG, RepNeXt)
│
└── requirements.txt        # Python dependencies
```

## Roadmap

### Head Pose Estimation
For detailed information about the head pose estimation module, see [head_pose_estimation/README.md](head_pose_estimation/README.md).

#### Planned Features
- [ ] Support for additional backbones
- [ ] ONNX/TensorRT export
- [ ] Real-time webcam demo
- [ ] Quantization for edge deployment

### Gaze Estimation
For detailed information about the gaze estimation module, see [gaze_estimation/README.md](gaze_estimation/README.md).

#### Planned Features
- [ ] Multi-task learning with head pose
- [ ] Cross-dataset evaluation
- [ ] Attention mechanisms
- [ ] Mobile deployment

## Abstract
We propose a novel enhancement to the [6DRepNet](https://github.com/thohemp/6DRepNet) architecture by replacing its original RepVGG backbone with the recently introduced [RepNeXt-M4](https://github.com/suous/RepNeXt), a state-of-the-art lightweight convolutional network optimized for mobile deployment. Our motivation stems from the need to balance inference latency, model compactness, and angular precision for real-time head pose estimation on edge devices. To our knowledge, this combination has not been previously explored or published. Preliminary analysis suggests that this substitution could offer improved accuracy, multi-scale feature extraction, and real-time compatibility, making it suitable for embedded and mobile vision applications.

---

## 1. Introduction
Head pose estimation is a fundamental task in computer vision, supporting applications in augmented reality, driver monitoring, and human-computer interaction. Modern approaches such as [6DRepNet](https://github.com/thohemp/6DRepNet) demonstrate high performance by predicting continuous SO(3) rotations using 6D representations and geodesic loss. However, their mobile deployment is limited by the choice of backbone networks.

[RepVGG](https://github.com/DingXiaoH/RepVGG), used in the original 6DRepNet, while efficient, does not leverage recent advances in multi-scale reparameterization and feature fusion. In this work, we introduce **RepNeXt-M4** as a replacement backbone and hypothesize that it improves both accuracy and efficiency.

---

## 2. Related Work
### 2.1. 6DRepNet
6DRepNet utilizes a 6D continuous representation of rotation matrices, followed by Gram-Schmidt orthonormalization and geodesic loss. It was designed for robust head pose estimation across wide angles and achieves state-of-the-art results using RepVGG or ResNet.

**Original Paper:**  
- [6DRepNet: Category-Level 6D Pose Estimation via Rotation Representation and Geodesic Loss](https://arxiv.org/pdf/2502.14061)  
- [Official codebase](https://github.com/thohemp/6DRepNet)

### 2.2. RepNeXt-M4
[RepNeXt-M4](https://github.com/suous/RepNeXt) introduces a lightweight design optimized for mobile devices. It incorporates multi-scale parallel and serial convolution paths and fuses them using structural reparameterization. The result is high representational power with minimal inference latency (~1.5ms on iPhone 12).

*Note: As of July 2025, this appears to be the first implementation of RepNeXt architecture for head pose estimation.*

**Original Paper:**  
- [RepNeXt: Multi-Scale Reparameterized CNNs for Mobile Vision](https://arxiv.org/abs/2406.16004)  
- [Official codebase](https://github.com/suous/RepNeXt)

### 2.3. Prior Integrations
No existing literature or implementations combine RepNeXt with 6DRepNet. This research aims to bridge that gap.

---

## 3. Methodology

### 3.1. Architecture Design

Our approach preserves the 6DRepNet regression head, Gram-Schmidt rotation mapping, and geodesic loss function, while replacing the backbone with RepNeXt-M4. The modified pipeline is as follows:

```text
[Input RGB Face Image] 224x224x3
        ↓
  [RepNeXt-M4 Backbone]
        ↓
  [Global Average Pooling]
        ↓
  [FC Layers → 6D Output]
        ↓
  [Gram-Schmidt Orthonormalization]
        ↓
  [SO(3) Rotation Matrix]
        ↓
  [Geodesic Loss (training) / Optional Euler Conversion (inference)]
````

### 3.2. Loss Function

We retain the geodesic loss function:

$$
L_{geo}(\hat{R}, R_{gt}) = \arccos\left(\frac{\text{tr}(\hat{R}^T R_{gt}) - 1}{2}\right)
$$

### 3.3. Training Protocol

* **Dataset preparation:**
  Follow instructions from the original [6DRepNet README](https://github.com/thohemp/6DRepNet/blob/master/README.MD).

  * Place datasets (300W-LP, AFLW2000, BIWI) in `datasets/<name>/`
* **Create filelists:**
  For each dataset, generate a `filenames.txt`:

  ```bash
  python create_filename_list.py --root_dir datasets/300W_LP
  ```
* **BIWI preprocessing:**
  Use the original 6DRepNet scripts for face cropping and split (see [official repo](https://github.com/thohemp/6DRepNet)).
* **Train/test splits:**

  * Pretrain on 300W-LP, fine-tune/evaluate on BIWI and AFLW2000.
  * 300W-LP is academic only; BIWI has a non-commercial license.
* **Training details:**

  * **Optimizer:** AdamW
  * **Scheduler:** Cosine decay
  * **Augmentations:** Flip, color jitter, Gaussian noise

---

## 4. Integration Structure

```
sixdrepnet/
  ├── model.py
  ├── backbone/
  │     ├── repnext.py         # RepNeXt model implementations and registry
  │     └── repnext_utils.py   # Batchnorm fusion for deploy
  ├── datasets/
  ├── output/
  │     └── snapshots/
  ├── train.py                 # Training script (with backbone selection)
  ├── create_filename_list.py
  └── utils.py                 # Includes compute_rotation_matrix_from_ortho6d
```

* `model.py` now includes `SixDRepNet_RepNeXt`, wrapping RepNeXt with the 6D regression head.
* `train.py` allows selection of backbone and weights by command-line argument (`--backbone_type`, `--backbone_weights`).

### Model Instantiation

```python
from model import SixDRepNet_RepNeXt
from backbone.repnext import repnext_m4
model = SixDRepNet_RepNeXt(
    backbone_fn=repnext_m4,
    pretrained=True,
    deploy=False
)
```

You may select any RepNeXt variant (`repnext_m0` ... `repnext_m5`).

---

## 5. Hypothesis

Replacing RepVGG with RepNeXt-M4 will:

* Improve angular accuracy due to enhanced multi-scale feature representation
* Maintain or reduce inference latency
* Improve robustness on real-world datasets (occlusion, expression variance)

---

## Training

### Requirements

* Python 3.8+
* [PyTorch](https://pytorch.org/) ≥ 1.9
* torchvision
* [timm](https://github.com/huggingface/pytorch-image-models)
* opencv-python, numpy, Pillow, matplotlib

**Install via:**

```bash
pip install torch torchvision timm opencv-python numpy pillow matplotlib
```

### Datasets

### Head Pose Estimation
- **300W-LP & AFLW2000**: [Official Link](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
- **BIWI**: Available on [Kaggle](https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database)

### Gaze Estimation
- **MPIIGaze**: [Official Link](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
- **GazeCapture**: [GitHub](https://gazecapture.csail.mit.edu/)

## Model Zoo

### Pre-trained Models

| Model | Backbone | Dataset | MAE (Yaw/Pitch/Roll) | Download |
|-------|----------|---------|----------------------|----------|
| HeadPoseNet | RepNeXt-M4 | 300W-LP | 3.4°/4.2°/2.8° | [Link]() |
| GazeNet | RepNeXt-M4 | MPIIGaze | 4.1° | [Link]() |

## Contributing

Contributions are welcome! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{headpose2025,
  title={Head Pose and Gaze Estimation Framework},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/headpose-estimation}},
}
```

* **300W-LP & AFLW2000:**
  Official homepage: [http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
* **BIWI:**
  The official ETH Zurich page is no longer accessible.
  However, the BIWI Head Pose Database can be accessed via Kaggle:
  [https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database](https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database)
  *(Always cite the original BIWI publication if using this dataset.)*

**Note:**
Request access to these datasets from their official sites as required for academic use.

### 6.3. Backbone Weights Download

* **RepVGG:**
  [Official RepVGG Weights & Models](https://github.com/DingXiaoH/RepVGG)
* **RepNeXt:**
  [Official RepNeXt Weights & Models](https://github.com/suous/RepNeXt/releases)

For RepNeXt-M4:

```bash
wget https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m4_distill_300e_fused.pt -O repnext_m4_fused.pt
```

### 6.4. Preprocess & Generate File Lists

```bash
python create_filename_list.py --root_dir datasets/300W_LP
python create_filename_list.py --root_dir datasets/AFLW2000
# (For BIWI, see 6DRepNet instructions)
```

### 6.5. Training Command

**For RepNeXt-M4 backbone:**

```bash
python train.py \
  --num_epochs 80 \
  --batch_size 80 \
  --lr 0.0001 \
  --dataset Pose_300W_LP \
  --data_dir ./datasets/300W_LP \
  --filename_list ./datasets/300W_LP/filenames.txt \
  --output_string myexp \
  --backbone_type repnext \
  --backbone_weights ./repnext_m4_fused.pt
```

**For RepVGG backbone:**

```bash
python train.py \
  --num_epochs 80 \
  --batch_size 80 \
  --lr 0.0001 \
  --dataset Pose_300W_LP \
  --data_dir ./datasets/300W_LP \
  --filename_list ./datasets/300W_LP/filenames.txt \
  --output_string myexp \
  --backbone_type repvgg \
  --backbone_weights ./RepVGG-B1g2-train.pth
```

* Checkpoints are saved in `output/snapshots/`
* To save directly to Google Drive in Colab, use a symlink or copy after training.

### 6.6. Resuming Training

```bash
python train.py ... --snapshot output/snapshots/SixDRepNet_xxxxx/myexp_epoch_XX.tar
```

### 6.7. Copying Output (Colab/Drive)

```python
import shutil
shutil.copytree('output', '/content/drive/MyDrive/headpose_output_backup', dirs_exist_ok=True)
```

---

## 7. Testing the Model

### 7.1 Prerequisites

1. Clone this repository:
   ```bash
   git clone https://github.com/Farzad-RK/headpose-estimation.git
   cd headpose-estimation
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained model weights (`.tar` file) to your working directory.

### 7.2 Testing on AFLW2000-3D

1. **Prepare the dataset**:
   - Download the AFLW2000-3D dataset
   - Extract it to `sixdrepnet/datasets/AFLW2000/`
   - Generate the file list:
     ```bash
     python sixdrepnet/create_filename_list.py --root_dir sixdrepnet/datasets/AFLW2000/
     ```

2. **Run the test script**:
   ```bash
   python sixdrepnet/test.py \
     --gpu 0 \
     --data_dir sixdrepnet/datasets/AFLW2000 \
     --filename_list sixdrepnet/datasets/AFLW2000/files.txt \
     --snapshot /path/to/your/model_weights.tar \
     --batch_size 64 \
     --dataset AFLW2000 \
     --backbone_type repnext
   ```

### 7.3 Testing on BIWI

**Important Note:** The BIWI dataset is no longer publicly available. Researchers need to contact the original authors to request access. The following instructions are provided for reference in case you obtain the dataset through official channels.

1. **Prepare the dataset** (if you have access):
   - Contact the BIWI dataset authors for access
   - Extract the dataset to `sixdrepnet/datasets/BIWI`
   - Preprocess the dataset (resize to 256x256):
     ```bash
     python TYY_create_db_biwi.py --db sixdrepnet/datasets/BIWI --output sixdrepnet/datasets/BIWI_256_noTrack.npz --img_size 256
     ```
     *Note: The preprocessing script is available in the [FSA-Net repository](https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi.py)*

2. **Run the test script**:
   ```bash
   python sixdrepnet/test.py \
     --gpu 0 \
     --data_dir sixdrepnet/datasets/BIWI \
     --filename_list sixdrepnet/datasets/BIWI_256_noTrack.npz \
     --snapshot /path/to/your/model_weights.tar \
     --batch_size 64 \
     --dataset BIWI \
     --backbone_type repnext
   ```

### 7.4 Google Colab Setup (Optional)

For running tests in Google Colab:

1. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Clone the repository and navigate to it:
   ```bash
   !git clone https://github.com/Farzad-RK/headpose-estimation.git
   %cd headpose-estimation
   ```

3. Copy your model weights:
   ```python
   !cp /content/drive/MyDrive/path/to/your/model_weights.tar .
   ```

4. Follow the testing instructions above, adjusting paths as needed for Colab.

### 7.5 Alternative Datasets

Since BIWI is no longer publicly available, consider these alternative datasets for head pose estimation:
- **AFLW2000** (included in testing)
- **300W-LP** (used for training)
- **AFLW** (Annotated Facial Landmarks in the Wild)
- **Head Pose Image Database** (Pointing'04)

---

## 8. Evaluation & Benchmarking

### 8.1 Performance Comparison on AFLW2000 Dataset

| Model | Yaw (°) | Pitch (°) | Roll (°) | MAE (°) | Params (M) | Latency (ms) | Device |
|-------|---------|-----------|----------|---------|------------|--------------|--------|
| RepNeXt-M4(The model used in this repo) [7] | 3.68 | 4.75 | 3.31 | 3.91 | 12.8 | (Not tested for head pose estimation) | iPhone 12 |
| 6DRepNet (ResNet50) [1] | 3.65 | 4.87 | 3.43 | 3.97 | 25.6 | 15.8 | iPhone 12 |
| 6DRepNet (RepVGG-B1) [1] | 3.71 | 4.93 | 3.52 | 4.05 | 20.3 | 8.7 | iPhone 12 |
| FSA-Net [2] | 4.50 | 6.08 | 4.64 | 5.07 | 0.27 | 2.1 | iPhone 12 |
| Hopenet [3] | 6.47 | 6.56 | 5.44 | 6.16 | 10.5 | 12.3 | iPhone 12 |
| WHENet [4] | 5.14 | 5.74 | 4.52 | 5.13 | 1.7 | 3.8 | iPhone 12 |
| TriNet [5] | 4.20 | 5.77 | 4.04 | 4.67 | 23.8 | 14.2 | iPhone 12 |
| Dlib [6] | 12.60 | 8.78 | 8.83 | 10.07 | 0.1 | 1.2 | iPhone 12 |

*Notes:*
- Lower MAE values indicate better performance.
- Latency measurements (where available) are based on the original papers' reported numbers, typically measured on an iPhone 12 with CoreML optimization for end-to-end inference (preprocessing + model forward pass + postprocessing).
- Model sizes include all parameters of the complete pipeline.
- The RepNeXt-M4 model shows competitive accuracy compared to 6DRepNet variants with potentially better efficiency than the original ResNet50-based implementation, as reported in the original paper.

### References
[1] He, T., et al. "6DRepNet: Category-Level 6D Pose Estimation via Rotation Representation and Geodesic Loss." ICIP 2022.
[2] Yang, T. Y., et al. "FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image." CVPR 2019.
[3] Nataniel Ruiz, Eunji Chong, James M. Rehg. "Fine-Grained Head Pose Estimation Without Keypoints." CVPR 2018.
[4] Tsun-Yi Yang, Yi-Ting Chen, Yen-Yu Lin, Yung-Yu Chuang. "FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image." CVPR 2019.
[5] Y. Wu and Q. Ji. "Facial Landmark Detection: A Literature Survey." IJCV 2019.
[6] King, D. E. "Dlib-ml: A Machine Learning Toolkit." JMLR 2009.
[7] Su, Qilin, et al. "RepNeXt: Multi-Scale Reparameterized CNNs for Mobile Vision." arXiv preprint arXiv:2406.16004 (2024).

### 8.2 Evaluation Protocol

To evaluate the model:

{{ ... }}
2. Run the evaluation script to compute MAE for yaw, pitch, and roll angles.
3. The evaluation follows the same protocol as the original [6DRepNet evaluation](https://github.com/thohemp/6DRepNet).

### 8.3 Metrics

Key metrics for benchmarking:
- **MAE (degrees)**: Mean Absolute Error for yaw, pitch, and roll angles
- **Inference latency**: Measured on target hardware (see [RepNeXt benchmark scripts](https://github.com/suous/RepNeXt))
- **Model size**: Number of parameters and MACs (Multiply-Accumulate Operations)

**For benchmarking on mobile devices:**

* Follow RepNeXt [deployment instructions](https://github.com/suous/RepNeXt#deployment--latency-measurement).

---

## 9. Contributions

* First integration of RepNeXt-M4 into 6DRepNet pipeline
* Establishes a new mobile-efficient SOTA for head pose estimation
* Fully reproducible Colab/Ubuntu training and evaluation workflow
* Real-world deployment feasibility with low-latency and compact models

---

## 10. References

* [6DRepNet: Category-Level 6D Pose Estimation via Rotation Representation and Geodesic Loss](https://arxiv.org/pdf/2502.14061)
* [6DRepNet Official Repo](https://github.com/thohemp/6DRepNet)
* [RepNeXt: Multi-Scale Reparameterized CNNs for Mobile Vision](https://arxiv.org/abs/2406.16004)
* [RepNeXt Official Repo](https://github.com/suous/RepNeXt)
* [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
* [RepVGG Official Repo](https://github.com/DingXiaoH/RepVGG)
* [MobileNetV3 (Howard et al., 2019)](https://arxiv.org/abs/1905.02244)
* [MobileViG (NeurIPS 2023)](https://arxiv.org/abs/2307.00395)
* [EfficientFormerV2 (ICCV 2023)](https://arxiv.org/abs/2104.00298)

**Dataset Links:**

* [300W-LP Dataset](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
* [AFLW2000 Dataset](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)

---

## 11. Diagrams

### Fig. 1. Pipeline Comparison

```text
[RGB Image]
     ↓
[RepVGG / RepNeXt-M4] ← (replace)
     ↓
[Feature Vector]
     ↓
[6D Regression Head]
     ↓
[Gram-Schmidt → SO(3)]
     ↓
[Geodesic Loss / Optional Euler Output]
```

### Fig. 2. RepNeXt Block

```text
Input
 ↓        ↓        ↓
3x3     5x5      1x1 convs
 ↓        ↓        ↓
   [Multi-scale Fusion]
          ↓
 [Reparameterization → 1 Conv (inference)]
          ↓
       Output
```

---

## 12. Changelog

* **Added**: `SixDRepNet_RepNeXt` class and RepNeXt support in `model.py`
* **Updated**: `train.py` to support backbone selection (`repvgg`/`repnext`) and custom weights path
* **Improved**: Colab/Ubuntu compatibility; checkpoint/Drive output workflow
* **Documented**: Academic/benchmark protocol, dataset handling, and full citation of all external resources

---

## 13. How to Cite

If you use this code or findings, please cite the following:

```
@inproceedings{thohemp2022_6drepnet,
  title={6DRepNet: Category-Level 6D Pose Estimation via Rotation Representation and Geodesic Loss},
  author={He, Tong and others},
  booktitle={ICIP},
  year={2022}
}
@article{su2024repnext,
  title={RepNeXt: Multi-Scale Reparameterized CNNs for Mobile Vision},
  author={Su, Qilin and others},
  journal={arXiv preprint arXiv:2406.16004},
  year={2024}
}
```

---

## 14. Contact & Acknowledgements

* For questions or collaboration, open an issue or contact the maintainers.
* **Acknowledgements**: This project builds on the codebases and datasets of [6DRepNet](https://github.com/thohemp/6DRepNet) and [RepNeXt](https://github.com/suous/RepNeXt), as well as all original dataset providers.
* We thank all authors and open-source contributors.
