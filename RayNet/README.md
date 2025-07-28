## RayNet: GazeGene Dataset Loader & Multi-View Sampler

This module provides PyTorch Dataset and Sampler classes for the GazeGene synthetic gaze estimation dataset, 
as used in RayNet. It is designed for efficient, balanced, multi-camera training of deep gaze models.

Main Features:
--------------
- Loads images, mesh, gaze, headpose, and camera info from GazeGene directory structure.
- Returns samples as PyTorch tensors for direct use with RepNeXt and BiFPN backbones.
- Supports selecting a subset of frames per subject for faster experiments.
- Provides balanced sampling over subject attributes (e.g., ethnicity, gender, eye color).
- Provides multi-view batches: each batch contains all 9 camera views for a single (subject, frame).

-----------------------------------------------------------------------------------
GazeGeneDataset
-----------------------------------------------------------------------------------

class GazeGeneDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for GazeGene synthetic gaze estimation dataset.

    Args:
        base_dir (str): Path to GazeGene_FaceCrops directory.
        subject_ids (list, optional): List of subject IDs to include (e.g., ['subject1', ...]).
        camera_ids (list, optional): List of camera indices (0-8) to include. Default: all 9.
        samples_per_subject (int, optional): Number of unique frames per subject to load. 
                                             If None, loads all frames.
        transform (callable, optional): Optional transform to be applied on images.
        balance_attributes (list, optional): List of attribute names (e.g., ['ethicity', 'gender'])
                                             for balanced sampling.
        seed (int): Random seed for reproducible sampling.

    Returns:
        Each __getitem__ returns a dict with:
            'img'        : Image tensor [3, H, W]
            'subject'    : Subject numeric ID
            'camera'     : Camera index (0-8)
            'frame_idx'  : Frame index within subject
            'mesh'       : Dict of mesh tensors (eyeball_center_3D, pupil_center_3D, iris_mesh_3D)
            'gaze'       : Dict of gaze vectors (gaze_C, visual_axis_L/R, optic_axis_L/R)
            'gaze_point' : 3D gaze point (if available)
            'head_pose'  : Dict of rotation matrix 'R' and translation 't'
            'intrinsic'  : Intrinsic camera matrix for cropped image
            'attributes' : Subject-level attributes (ethnicity, gender, etc.)
    """

-----------------------------------------------------------------------------------
MultiViewBatchSampler
-----------------------------------------------------------------------------------

class MultiViewBatchSampler(torch.utils.data.Sampler):
    """
    Custom batch sampler for multi-camera training.

    Each batch consists of all 9 camera views for a given (subject, frame_idx).
    Optionally balances sampling over subject attributes, for robust, unbiased training.

    Args:
        dataset (GazeGeneDataset): The dataset instance.
        balance_attributes (list, optional): List of attribute names for balanced grouping.
        shuffle (bool): Whether to shuffle batches each epoch.

    Yields:
        List of 9 indices (one per camera) for a single (subject, frame_idx).
    """

-----------------------------------------------------------------------------------
Example Usage
-----------------------------------------------------------------------------------

```python
from torch.utils.data import DataLoader

# Initialize dataset (50 random frames per subject, balance by ethnicity)
dataset = GazeGeneDataset(
    base_dir='./GazeGene_FaceCrops',
    samples_per_subject=50,
    transform=None,               # or torchvision transforms
    balance_attributes=['ethicity']
)

# Initialize multi-view sampler and loader
batch_sampler = MultiViewBatchSampler(dataset, balance_attributes=['ethicity'], shuffle=True)
loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

# Iterate over batches
for batch in loader:
    # batch['img'] is a list of 9 tensors [3, H, W] (all camera views)
    # batch['gaze']['gaze_C'] is a list of 9 vectors
    print(batch['img'][0].shape, batch['gaze']['gaze_C'][0])
    break


```
Absolutely! Here’s a **ready-to-paste Markdown documentation** for your **PANet** and **RayNet** modules, including channel mapping, theoretical background, and input/output details.

---

# RayNet Architecture Documentation

## Overview

**RayNet** is a modular neural network architecture for robust eye gaze, head pose, and eye mesh estimation.
It combines a deep, efficient **RepNeXt** backbone (flexible size), with a **YOLOv8-style PANet** for powerful multi-scale feature fusion.
This design enables detailed local (eye) and contextual (face, head) understanding for multi-task learning.

---

## Table of Contents

1. [RayNet Architecture](#raynet-architecture)
2. [RepNeXt Backbone](#repnext-backbone)
3. [PANet Neck](#panet-neck)
4. [Channel & Feature Size Mapping](#channel--feature-size-mapping)
5. [Input/Output Example](#inputoutput-example)
6. [Theory and Design Rationale](#theory-and-design-rationale)
7. [Usage Example](#usage-example)

---

## RayNet Architecture

* **Backbone:** RepNeXt (choose from m0-m5, various widths and depths)
* **Neck:** PANet (Path Aggregation Network, YOLOv8-style)
* **Multi-task heads:** (to be defined) for gaze, head pose, and mesh estimation.

```
Input (3x448x448) 
   │
RepNeXt backbone (4 stages)
   ├── C1: stride=4  (high-res, low semantic, e.g., eyes)
   ├── C2: stride=8
   ├── C3: stride=16
   └── C4: stride=32 (low-res, high semantic, e.g., head/global)
   │
PANet (lateral, top-down, bottom-up fusion)
   └── Outputs P2, P3, P4, P5: multi-scale, unified-channel features
```

---

## RepNeXt Backbone

The **RepNeXt** backbone is a scalable convolutional network.
You can select from different model sizes (m0, m1, ..., m5) to trade off between speed and accuracy.

| Model       | C1 | C2  | C3  | C4  |
| ----------- | -- | --- | --- | --- |
| repnext\_m0 | 40 | 80  | 160 | 320 |
| repnext\_m1 | 48 | 96  | 192 | 384 |
| repnext\_m2 | 56 | 112 | 224 | 448 |
| repnext\_m3 | 64 | 128 | 256 | 512 |
| repnext\_m4 | 64 | 128 | 256 | 512 |
| repnext\_m5 | 80 | 160 | 320 | 640 |

* **C1–C4:** Feature maps at four different resolutions, extracted after each major backbone stage.

---

## PANet Neck

**PANet** (Path Aggregation Network) is used for fusing multi-scale features.

* **Lateral 1x1 conv:** Unifies all backbone outputs to the same number of channels (e.g., 256).
* **Top-down fusion:** Passes global, semantic context to high-resolution features (fine detail).
* **Bottom-up fusion:** Brings detail from high-res features into coarser features (robust context).

> **References:**
>
> * Liu et al., ["Path Aggregation Network for Instance Segmentation"](https://arxiv.org/abs/1803.01534)
> * [YOLOv8 PANet implementation](https://docs.ultralytics.com/models/yolov8/#architecture)

---

## Channel & Feature Size Mapping

**Assuming input image size 448×448**:

| Layer       | Channels      | Feature Size | Stride |
| ----------- | ------------- | ------------ | ------ |
| Input       | 3             | 448×448      | 1      |
| C1 (stage0) | e.g. 64       | 112×112      | 4      |
| C2 (stage1) | e.g. 128      | 56×56        | 8      |
| C3 (stage2) | e.g. 256      | 28×28        | 16     |
| C4 (stage3) | e.g. 512      | 14×14        | 32     |
| PANet P2    | 256 (default) | 112×112      | 4      |
| PANet P3    | 256           | 56×56        | 8      |
| PANet P4    | 256           | 28×28        | 16     |
| PANet P5    | 256           | 14×14        | 32     |

* Channels per stage change depending on RepNeXt version (see [RepNeXt Backbone](#repnext-backbone)).
* **PANet always outputs the same number of channels per scale** (default: 256).

---

## Input/Output Example

**Example with repnext\_m3 and input \[B, 3, 448, 448]:**

```python
features = model(x)
for idx, f in enumerate(features):
    print(f"PANet output P{idx+2}: {f.shape}")
```

**Outputs:**

```
PANet output P2: torch.Size([B, 256, 112, 112])
PANet output P3: torch.Size([B, 256, 56, 56])
PANet output P4: torch.Size([B, 256, 28, 28])
PANet output P5: torch.Size([B, 256, 14, 14])
```

---

## Theory and Design Rationale

* **Why multi-scale features?**

  * Tasks like gaze and mesh estimation need fine detail (P2), while head pose benefits from global context (P5).
  * PANet allows learning at all scales, improving robustness and accuracy.

* **Why unify channels?**

  * Different backbone stages have different channel widths; PANet uses 1x1 conv to align them for addition and fusion.

* **Why top-down & bottom-up?**

  * Top-down: passes semantic info to details.
  * Bottom-up: reinforces context and details at all scales.

---

## Usage Example

```python
from raynet import RayNet

# Create the model (choose backbone size)
model = RayNet(backbone_name='repnext_m3', pretrained=False)
model = model.cuda()  # or .to(device)

# Dummy input
x = torch.randn(2, 3, 448, 448).cuda()
features = model(x)

# features: list of 4 tensors (P2-P5), each [B, 256, H, W]
for idx, f in enumerate(features):
    print(f"PANet output P{idx+2}: {f.shape}")
```

---

## Downstream Multi-Task Heads

The four output feature maps (P2-P5) can be passed to custom task-specific heads for:

* **Gaze vector & gaze point regression**
* **Head pose estimation**
* **Eye mesh (vertex regression or heatmap)**

Simply add separate head modules to the RayNet model as required.

---

## Extending to Other RepNeXt Variants

To use a different backbone (e.g., `repnext_m5`), change:

```python
model = RayNet('repnext_m5', pretrained=False)
```

All PANet fusion logic adapts automatically to the channel widths of your selected RepNeXt variant.

---

## References

* [Path Aggregation Network for Instance Segmentation](https://arxiv.org/abs/1803.01534)
* [YOLOv8 Model Architecture](https://docs.ultralytics.com/models/yolov8/#architecture)
* [RepNeXt: RepVGG-style next generation ConvNets](https://github.com/slightech-research/RepNeXt)

---

**Author:** *Your Name or Lab*
**Last updated:** *YYYY-MM-DD*

---

Let me know if you want this as a separate README file, want additional figures/diagrams, or need inline code docstrings as well!

