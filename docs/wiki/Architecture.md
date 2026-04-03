# Architecture

RayNet v2 is a two-task convolutional architecture: it predicts 14 eye landmarks and the optical axis direction from a 224x224 normalized eye crop.

## Pipeline Overview

```
Input Image (B, 3, 224, 224)
        |
   RepNeXt-M3 Backbone
   (4 stages, strides 4/8/16/32)
        |
   [C1, C2, C3, C4]
        |
   PANet (Path Aggregation Network)
   Top-down + Bottom-up fusion
        |
   [P2, P3, P4, P5]  (all 256 channels)
        |
   +----+----+
   |         |
CoordAtt  CoordAtt
  (P2)      (P5)
   |         |
Landmark   Optical
  Head     Axis Head
   |         |
(B,14,2)  (B,3) unit vec
coords    gaze direction
```

## Backbone: RepNeXt

RepNeXt is a family of lightweight CNNs using reparameterized depthwise convolutions. RayNet defaults to **RepNeXt-M3**.

### Variants

| Model | Channels (C1/C2/C3/C4) | Params | Notes |
|-------|------------------------|--------|-------|
| `repnext_m0` | 40 / 80 / 160 / 320 | ~2.8 M | Fastest |
| `repnext_m1` | 48 / 96 / 192 / 384 | ~4.2 M | |
| `repnext_m2` | 56 / 112 / 224 / 448 | ~5.5 M | |
| `repnext_m3` | 64 / 128 / 256 / 512 | **7.8 M** | **Default** |
| `repnext_m4` | 64 / 128 / 256 / 512 | ~7.8 M | Different depth |
| `repnext_m5` | 80 / 160 / 320 / 640 | ~12.0 M | Largest |

### Output Feature Maps

For a 224x224 input with RepNeXt-M3:

| Stage | Output Shape | Stride |
|-------|-------------|--------|
| C1 | `(B, 64, 56, 56)` | 4 |
| C2 | `(B, 128, 28, 28)` | 8 |
| C3 | `(B, 256, 14, 14)` | 16 |
| C4 | `(B, 512, 7, 7)` | 32 |

### Loading Pretrained Weights

```python
from RayNet.raynet import create_raynet

model = create_raynet(
    backbone_name='repnext_m3',
    weight_path='/path/to/repnext_m3_imagenet.pt',  # optional
    n_landmarks=14,
)
```

## PANet (Path Aggregation Network)

YOLOv8-style multi-scale fusion with SiLU activations. Projects all backbone stages to 256 channels, then fuses top-down and bottom-up.

**Lateral projections** (1x1 conv):
```
C1 (64ch)  -> P2 (256ch, 56x56)
C2 (128ch) -> P3 (256ch, 28x28)
C3 (256ch) -> P4 (256ch, 14x14)
C4 (512ch) -> P5 (256ch, 7x7)
```

**Top-down path** (upsample + add):
```
P5 -> upsample(2x) + P4 -> upsample(2x) + P3 -> upsample(2x) + P2
```

**Bottom-up path** (downsample + add):
```
P2 -> downsample(2x) + P3 -> downsample(2x) + P4 -> downsample(2x) + P5
```

Output: `[P2, P3, P4, P5]` all with 256 channels.

Source: `RayNet/panet.py`

## Coordinate Attention

Applied to **P2** (landmarks) and **P5** (gaze). Encodes spatial position information via directional pooling along x and y axes, then recalibrates channel responses. Preferred over SE-Net for spatial landmark localization.

Source: `RayNet/coordatt.py` (Hou et al., 2021)

## Task Head A: Iris/Pupil Landmark Detection

**Input**: `P2_att` (B, 256, 56, 56)

### Architecture

```
P2_att (B, 256, 56, 56)
    |
    +------+------+
    |             |
Heatmap Branch  Offset Branch
Conv(256,128)   Conv(256,128)
BatchNorm+ReLU  BatchNorm+ReLU
Conv(128, 14)   Conv(128, 28)
    |             |
 (B,14,56,56)  (B,28,56,56)
    |             |
 sigmoid       reshape to
    |          (B,14,2,56,56)
    |             |
    +------+------+
           |
      Soft-Argmax
    (differentiable)
           |
    (B, 14, 2)  <-- landmark coordinates in feature space
```

### Soft-Argmax with Offset Refinement

1. Apply sigmoid to heatmap logits
2. Compute soft-argmax (expected spatial position weighted by heatmap)
3. Add learned sub-pixel offset from offset branch
4. Output: `(B, 14, 2)` coordinates in feature-map space (0 to 56)

### 14 Landmarks

| Index | Type | Source |
|-------|------|--------|
| 0-9 | Iris contour | 100 iris mesh points, subsampled at indices [0, 10, 20, ..., 90] |
| 10-13 | Pupil boundary | 4 iris points closest to 2D pupil center |

Source: `RayNet/heads.py:IrisPupilLandmarkHead`

## Task Head B: Optical Axis Regression

**Input**: `P5_att` (B, 256, 7, 7)

### Architecture

```
P5_att (B, 256, 7, 7)
    |
AdaptiveAvgPool2d(1)
    |
 (B, 256, 1, 1) -> flatten -> (B, 256)
    |
Linear(256, 128) + ReLU
    |
Linear(128, 2)  -> (B, 2)  pitch, yaw in radians
    |
Spherical -> Cartesian conversion
    |
 (B, 3)  unit vector (gaze direction)
```

### Angle-to-Vector Conversion

```python
pitch, yaw = angles[:, 0], angles[:, 1]
gaze_x = -torch.cos(pitch) * torch.sin(yaw)
gaze_y = -torch.sin(pitch)
gaze_z = -torch.cos(pitch) * torch.cos(yaw)
gaze_vector = stack([gaze_x, gaze_y, gaze_z])  # (B, 3)
gaze_vector = F.normalize(gaze_vector, dim=-1)
```

Source: `RayNet/heads.py:OpticalAxisHead`

## Full Tensor Shape Trace

```
Input:                              (B, 3, 224, 224)

RepNeXt C1 (stride 4):             (B,  64,  56, 56)
RepNeXt C2 (stride 8):             (B, 128,  28, 28)
RepNeXt C3 (stride 16):            (B, 256,  14, 14)
RepNeXt C4 (stride 32):            (B, 512,   7,  7)

PANet P2:                           (B, 256,  56, 56)
PANet P3:                           (B, 256,  28, 28)
PANet P4:                           (B, 256,  14, 14)
PANet P5:                           (B, 256,   7,  7)

CoordAtt(P2):                       (B, 256,  56, 56)
CoordAtt(P5):                       (B, 256,   7,  7)

Landmark heatmaps:                  (B,  14,  56, 56)
Landmark offsets:                   (B,  28,  56, 56)
Landmark coords (feature space):   (B,  14,   2)

Gaze angles:                        (B,   2)
Gaze vector:                        (B,   3)
```

## Model Output Dictionary

```python
predictions = model(images)  # images: (B, 3, 224, 224)

predictions = {
    'landmark_coords':    (B, 14, 2),    # feature-map space [0, 56)
    'landmark_heatmaps':  (B, 14, 56, 56),  # raw logits (before sigmoid)
    'gaze_vector':        (B, 3),        # unit vector in normalized space
    'gaze_angles':        (B, 2),        # pitch, yaw in radians
}
```

### Coordinate Space Conversions

```
Feature space (0-56)  * 4.0  =  Normalized pixel space (0-224)
                                        |
                                   M_norm_inv (homography)
                                        |
                                Original camera pixel space
```
