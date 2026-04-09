# Architecture

RayNet v4.1 is a multi-task architecture with two independent backbones: it predicts 14 eye landmarks, optical axis direction, and implicit head pose from a 224×224 face crop. The pose backbone is gradient-isolated from the main backbone to prevent adversarial optimization between tasks.

## Pipeline Overview

```
Input Image (B, 3, 224, 224)
        |
        +---------------------------+
        |                           |
   RepNeXt-M3 Backbone         RepNeXt-M1 Backbone
   (main: landmarks + gaze)    (pose: head orientation)
   [C1, C2, C3, C4]            [C1, C2, C3, C4]
        |                           |
   PANet Neck                  CoordAtt(C4)
   [P2, P3, P4, P5]                |
        |                      GlobalAvgPool
   +----+----+                      |
   |         |                 Linear(384→256)
CoordAtt  CoordAtt                  |
  (P2)      (P5)            pose_feat (B, 256)
   |         |              + pose_head → 6D rotation (aux loss)
   |    GAP(P5)                     |
   |         |                      |
   |   LandmarkGazeBridge           |
   |   (P5 attends P2)             |
   |         |                      |
   |   CameraEmbed(R_cam,T_cam) + pose_feat  [additive fusion]
   |         |
   |   CrossViewAttention (geometry-conditioned)
   |         |
Landmark   Optical
  Head     Axis Head
   |         |
(B,14,2)  (B,3) unit vec      (B,6) 6D rotation
coords    gaze direction       pred_pose_6d (aux)
```

## Version History

| Component | v3 | v4 | v4.1 |
|-----------|----|----|------|
| Input size | 448×448 | 224×224 | 224×224 |
| Backbones | 1 (M3) | 1 (M3) | **2 (M3 + M1)** |
| Head pose | None | Explicit head_R input | **Implicit PoseEncoder** (MAGE-style) |
| Pose representation | — | 9D (flat matrix) | **6D (Gram-Schmidt)** |
| Pose loss | — | L1 on matrix | **Geodesic on SO(3)** |
| CrossViewAttention | Blind | Camera + head_R | **Camera + learned pose** |
| Landmark-Gaze | Decoupled | LandmarkGazeBridge | LandmarkGazeBridge |
| Ray constraint | None | Ray-to-target loss | Ray-to-target loss |
| head_R at inference | N/A | Required | **Not needed** |

## Backbone: RepNeXt (Dual)

RepNeXt is a family of lightweight CNNs using reparameterized depthwise convolutions with multi-scale feature extraction.

### Main Backbone: RepNeXt-M3 (landmarks + gaze)

Pretrained weights: `repnext_m3_distill_300e.pth` (ImageNet-1K, distilled, 80.7% top-1)

| Stage | Output Shape | Stride | Channels |
|-------|-------------|--------|----------|
| C1 | `(B, 64, 56, 56)` | 4 | 64 |
| C2 | `(B, 128, 28, 28)` | 8 | 128 |
| C3 | `(B, 256, 14, 14)` | 16 | 256 |
| C4 | `(B, 512, 7, 7)` | 32 | 512 |

7.8M params, 1.3G MACs, 1.11ms on iPhone 12.

### Pose Backbone: RepNeXt-M1 (head pose)

Pretrained weights: `repnext_m1_distill_300e.pth` (ImageNet-1K, distilled, 78.8% top-1)

| Stage | Output Shape | Stride | Channels |
|-------|-------------|--------|----------|
| C1 | `(B, 48, 56, 56)` | 4 | 48 |
| C2 | `(B, 96, 28, 28)` | 8 | 96 |
| C3 | `(B, 192, 14, 14)` | 16 | 192 |
| C4 | `(B, 384, 7, 7)` | 32 | 384 |

4.8M params, 0.8G MACs, 0.86ms on iPhone 12. Gradient-isolated from main backbone.

### Weight Selection

| Format | File Pattern | Use Case |
|--------|-------------|----------|
| **Training** (`.pth`) | `repnext_mN_distill_300e.pth` | Fine-tuning (has BatchNorm, multi-branch) |
| Fused (`.pt`) | `repnext_mN_distill_300e_fused.pt` | Inference-only (JIT, BN fused) — **cannot fine-tune** |

Always use **distilled, non-fused** (`.pth`) weights for training. Distilled weights have richer internal representations from teacher model supervision.

### RepNeXt Variant Comparison

| Model | Params | MACs | Latency | Top-1 | Role |
|-------|--------|------|---------|-------|------|
| M0 | 2.3M | 0.4G | 0.59ms | 74.2% | Too weak for pose |
| **M1** | **4.8M** | **0.8G** | **0.86ms** | **78.8%** | **Pose backbone** |
| M2 | 6.5M | 1.1G | 1.00ms | 80.1% | Alternative pose (heavier) |
| **M3** | **7.8M** | **1.3G** | **1.11ms** | **80.7%** | **Main backbone** |

Source: `backbone/repnext.py`, `backbone/repnext_utils.py`

## PANet (Path Aggregation Network)

YOLOv8-style multi-scale fusion with SiLU activations. Projects all main backbone stages to 256 channels, then fuses top-down and bottom-up. Applied to main backbone only (not pose backbone).

```
Top-down:  P5 → upsample(2x) + P4 → upsample(2x) + P3 → upsample(2x) + P2
Bottom-up: P2 → downsample(2x) + P3 → downsample(2x) + P4 → downsample(2x) + P5
```

Output: `[P2, P3, P4, P5]` all with 256 channels.

Source: `RayNet/panet.py`

## Coordinate Attention

Applied to **P2** (landmarks), **P5** (gaze), and **pose backbone C4** (head pose). Encodes spatial position via directional pooling along x and y axes. Captures face asymmetry (yaw) and vertical tilt (pitch) — critical cues for pose estimation.

Source: `RayNet/coordatt.py` (Hou et al., 2021)

## PoseEncoder (v4.1, MAGE-inspired)

Separate RepNeXt-M1 backbone that learns implicit head pose features from the image. No explicit head pose input needed at inference.

```
Image (B, 3, 224, 224)
    |
RepNeXt-M1 (4 stages, gradient checkpointed)
    |
C4 (B, 384, 7, 7)
    |
CoordinateAttention(384)
    |
AdaptiveAvgPool2d(1) → (B, 384)
    |
Linear(384, 256) → pose_feat (B, 256)   [fused with cam_embed]
    |
Linear(256, 6) → pred_pose_6d (B, 6)    [auxiliary geodesic loss]
```

### 6D Rotation Representation

The pose head outputs 6 values representing the first two columns of the rotation matrix. Gram-Schmidt orthogonalization reconstructs the full 3×3 R:

```
a1 = r6d[:, 0:3]  (first column)
a2 = r6d[:, 3:6]  (second column)

b1 = normalize(a1)
b2 = normalize(a2 - (b1·a2) * b1)   # orthogonalize
b3 = b1 × b2                         # cross product

R = [b1, b2, b3]  (proper rotation, det=+1)
```

This is the optimal continuous representation of SO(3) for neural networks (Zhou et al., CVPR 2019). Unlike quaternions or Euler angles, it has no discontinuities or gimbal lock.

### Geodesic Loss

The pose prediction is supervised by geodesic distance on SO(3):

```
L_geo = arccos( (tr(R_pred^T @ R_gt) - 1) / 2 )
```

This measures the actual angle needed to rotate from predicted to GT orientation, respecting the SO(3) manifold. L1/L2 on matrix elements doesn't respect rotation geometry.

~4.9M params total (backbone + CoordAtt + projection + aux head).

Source: `RayNet/raynet.py:PoseEncoder`, `RayNet/losses.py:geodesic_loss`

## LandmarkGazeBridge (v4)

Cross-attention: P5 gaze features (query) attend to P2 landmark features (key/value).

```
P2_att (B, 256, 56, 56) → AvgPool(7) → flatten → (B, 49, 256) [key/value]
P5_pooled (B, 256) → unsqueeze → (B, 1, 256) [query]
    → MultiheadAttention (4 heads, pre-norm)
    → residual → (B, 256)
```

**Note on GazeGene augmentation**: The random crop translation means P2 landmark features encode positions in augmented crop space, while gaze is in CCS. The bridge may learn unstable spatial correlations. See staged training strategy in train.py for mitigation.

~0.4M params. Source: `RayNet/raynet.py:LandmarkGazeBridge`

## CameraEmbedding (v4.1)

Encodes camera extrinsics only (no head pose — that's learned by PoseEncoder).

```
R_cam (B, 3, 3) → flatten → (B, 9)
T_cam (B, 3)
    → concat → (B, 12)
    → MLP(12→64→256)
    → (B, 256) + pose_feat (B, 256)  [additive fusion]
    → cam_embed for CrossViewAttention
```

~0.02M params. Source: `RayNet/raynet.py:CameraEmbedding`

## CrossViewAttention

Pre-norm Transformer Encoder for cross-view gaze feature fusion. Receives fused camera + pose embedding.

```
pooled (B, 256) + cam_embed (B, 256)
    → reshape to (G, V, 256)   where G = B/V groups, V = 9 views
    → TransformerEncoder (2 layers, 4 heads, GELU)
    → reshape to (B, 256)
```

Single-view (n_views=1) bypasses the encoder. ~1.05M params.

Source: `RayNet/raynet.py:CrossViewAttention`

## Task Head A: Iris/Pupil Landmark Detection

**Input**: `P2_att` (B, 256, 56, 56)

Heatmap + offset branches → soft-argmax for differentiable subpixel landmark coordinates.

| Index | Type | Source |
|-------|------|--------|
| 0-9 | Iris contour | 100 iris mesh points, subsampled at [0, 10, ..., 90] |
| 10-13 | Pupil boundary | 4 iris points closest to 2D pupil center |

Source: `RayNet/heads.py:IrisPupilLandmarkHead`

## Task Head B: Optical Axis Regression

**Input**: `P5_att` (B, 256, 7, 7) → pool → bridge → cross-view → FC → pitch/yaw → unit 3D vector

Source: `RayNet/heads.py:OpticalAxisHead`

## Full Tensor Shape Trace

```
Input:                              (B, 3, 224, 224)

=== Main Backbone (RepNeXt-M3) ===
C1 (stride 4):                     (B,  64,  56, 56)
C2 (stride 8):                     (B, 128,  28, 28)
C3 (stride 16):                    (B, 256,  14, 14)
C4 (stride 32):                    (B, 512,   7,  7)

PANet P2:                           (B, 256,  56, 56)
PANet P5:                           (B, 256,   7,  7)

CoordAtt(P2):                       (B, 256,  56, 56)
CoordAtt(P5):                       (B, 256,   7,  7)

Landmark heatmaps:                  (B,  14,  56, 56)
Landmark coords:                    (B,  14,   2)

GAP(P5_att):                        (B, 256)
LandmarkGazeBridge:                 (B, 256)

=== Pose Backbone (RepNeXt-M1) ===
C4 (stride 32):                     (B, 384,   7,  7)
CoordAtt(C4):                       (B, 384,   7,  7)
pose_feat:                          (B, 256)
pred_pose_6d:                       (B,   6)

=== Fusion ===
CameraEmbed + pose_feat:            (B, 256)
CrossViewAttention:                  (B, 256)

=== Outputs ===
Gaze vector:                        (B,   3)
Gaze angles:                        (B,   2)
```

## Model Output Dictionary

```python
predictions = model(images, n_views=9, R_cam=R_cam, T_cam=T_cam)

predictions = {
    'landmark_coords':   (B, 14, 2),      # feature-map space [0, 56)
    'landmark_heatmaps': (B, 14, 56, 56), # raw logits
    'gaze_vector':       (B, 3),          # unit vector in CCS
    'gaze_angles':       (B, 2),          # pitch, yaw in radians
    'pred_pose_6d':      (B, 6),          # 6D rotation (for aux loss)
}
```

## Parameter Budget

| Component | Params | Notes |
|-----------|--------|-------|
| RepNeXt-M3 (main) | 7.8M | Landmarks + gaze features |
| PANet | ~1.0M | Multi-scale fusion |
| CoordAtt ×3 | ~0.1M | P2, P5, pose C4 |
| LandmarkGazeBridge | ~0.4M | Cross-attention |
| CameraEmbedding | ~0.02M | Extrinsics encoding |
| CrossViewAttention | ~1.05M | Multi-view transformer |
| RepNeXt-M1 (pose) | 4.8M | Head pose features |
| PoseEncoder head | ~0.1M | Projection + 6D output |
| Task heads | ~0.3M | Landmarks + gaze FC |
| **Total** | **~15.6M** | |
