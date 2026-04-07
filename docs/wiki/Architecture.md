# Architecture

RayNet v4 is a two-task convolutional architecture with geometry-conditioned multi-view fusion: it predicts 14 eye landmarks and the optical axis direction from a 224×224 face crop.

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
   |    AdaptiveAvgPool
   |         |
   |   LandmarkGazeBridge ← cross-attend(P5_pooled, P2)
   |         |
   |   CameraEmbedding(R_cam, T_cam)
   |         |
   |   CrossViewAttention (geometry-conditioned)
   |         |
Landmark   Optical
  Head     Axis Head
   |         |
(B,14,2)  (B,3) unit vec
coords    gaze direction
```

## v3 → v4 Changes

| Component | v3 | v4 |
|-----------|----|----|
| Input size | 448×448 | **224×224** |
| CrossViewAttention | Blind (no camera info) | **Geometry-conditioned** (CameraEmbedding) |
| Landmark-Gaze coupling | Decoupled (P2 and P5 independent) | **Cross-attention bridge** (LandmarkGazeBridge) |
| Ray constraint | None | **Ray-to-target loss** (origin + depth × direction = target) |
| Normalization | None (raw crops) | **Easy-Norm** available (MAGE-style, bounding-box only) |

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

For a 224×224 input with RepNeXt-M3:

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

**Lateral projections** (1×1 conv):
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

## LandmarkGazeBridge (v4)

Cross-attention module that ties the landmark and gaze tasks in a shared latent space. P5 gaze features (query) attend to P2 landmark features (key/value).

```
P2_att (B, 256, 56, 56) → AdaptiveAvgPool2d(7) → (B, 256, 7, 7) → flatten → (B, 49, 256) [key/value]
P5_pooled (B, 256) → unsqueeze → (B, 1, 256) [query]
    → MultiheadAttention (4 heads, pre-norm)
    → residual connection
    → (B, 256) gaze features enriched with landmark context
```

~0.4M params. Source: `RayNet/raynet.py:LandmarkGazeBridge`

## CameraEmbedding (v4)

Encodes camera extrinsics into a d_model-dim vector for geometry-conditioned cross-view attention.

```
R_cam (B, 3, 3) → flatten → (B, 9)
T_cam (B, 3)
    → concat → (B, 12)
    → Linear(12, 64) + ReLU + Linear(64, 256)
    → (B, 256) camera embedding
```

Added to pooled features before CrossViewAttention (additive fusion).

~0.02M params. Source: `RayNet/raynet.py:CameraEmbedding`

## CrossViewAttention (v4: geometry-conditioned)

Pre-norm Transformer Encoder for cross-view gaze feature fusion. Now receives camera embedding for geometric grounding.

```
pooled (B, 256) + cam_embed (B, 256)  [additive fusion]
    → reshape to (G, V, 256)
    → TransformerEncoder (2 layers, 4 heads, GELU)
    → reshape to (B, 256)
```

Single-view (n_views=1) bypasses the encoder. ~1.05M params.

Source: `RayNet/raynet.py:CrossViewAttention`

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
LandmarkGazeBridge(P2_att)   [v4: cross-attend to landmarks]
    |
CameraEmbedding + CrossViewAttention  [v4: geometry-conditioned]
    |
Linear(256, 128) + ReLU
    |
Linear(128, 2)  -> (B, 2)  pitch, yaw in radians
    |
Spherical -> Cartesian conversion
    |
 (B, 3)  unit vector (gaze direction)
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

LandmarkGazeBridge:                 (B, 256) [cross-attended]
CameraEmbedding:                    (B, 256) [additive]
CrossViewAttention:                 (B, 256) [multi-view fused]

Gaze angles:                        (B,   2)
Gaze vector:                        (B,   3)
```

## Model Output Dictionary

```python
predictions = model(images, n_views=9, R_cam=R_cam, T_cam=T_cam)

predictions = {
    'landmark_coords':    (B, 14, 2),    # feature-map space [0, 56)
    'landmark_heatmaps':  (B, 14, 56, 56),  # raw logits (before sigmoid)
    'gaze_vector':        (B, 3),        # unit vector in camera coordinate space
    'gaze_angles':        (B, 2),        # pitch, yaw in radians
}
```

### Coordinate Space Conversions

```
Feature space (0-56)  * 4.0  =  Pixel space (0-224)
                                        |
                                   M_norm_inv (Easy-Norm homography)
                                        |
                                Original camera pixel space
```
