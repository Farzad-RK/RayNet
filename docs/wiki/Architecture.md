# Architecture

RayNet v4.1 is a multi-task architecture with two gradient-isolated backbones. It predicts 14 eye landmarks, optical axis direction, and implicit head pose (9D: 6D rotation + 3D translation) from a 224x224 face crop.

## Pipeline Overview

```
Input Image (B, 3, 224, 224)
        |
        +---------------------------+
        |                           |
   RepNeXt-M3 Backbone         RepNeXt-M1 Backbone
   (main: landmarks + gaze)    (pose: head orientation + translation)
   [C1, C2, C3, C4]            [C1, C2, C3, C4]
        |                           |
   PANet Neck                  CoordAtt(C4)
   [P2, P3, P4, P5]                |
        |                      GlobalAvgPool
   +----+----+                      |
   |         |                 Linear(384->256)
CoordAtt  CoordAtt                  |
  (P2)      (P5)            pose_feat (B, 256)
   |         |              + pose_head -> 9D pose (aux losses)
   |    GAP(P5)             |   6D rotation (geodesic loss)
   |         |              |   3D translation (log-depth SmoothL1)
   |   LandmarkGazeBridge   |
   |   (P5 attends P2)     |
   |   [Stage 3 only]      |
   |         |              |
   |   CameraEmbed(R_cam,T_cam) + pose_feat  [additive fusion]
   |         |
   |   CrossViewAttention (geometry-conditioned transformer)
   |         |
Landmark   Optical
  Head     Axis Head
   |         |
(B,14,2)  (B,3) unit vec     (B,6) 6D rotation  (B,3) 3D translation
coords    gaze direction      pred_pose_6d        pred_pose_t
```

## Version History

| Component | v3 | v4 | v4.1 |
|-----------|----|----|------|
| Input size | 448x448 | 224x224 | 224x224 |
| Backbones | 1 (M3) | 1 (M3) | **2 (M3 + M1)** |
| Head pose | None | Explicit head_R input | **Implicit PoseEncoder** (MAGE-style) |
| Pose representation | -- | 9D (flat matrix) | **9D (6D Gram-Schmidt + 3D tanh/exp)** |
| Pose rotation loss | -- | L1 on matrix | **Geodesic on SO(3)** |
| Pose translation loss | -- | None | **SmoothL1 (xy) + log-SmoothL1 (z)** |
| CrossViewAttention | Blind | Camera + head_R | **Camera + learned pose** |
| Landmark-Gaze | Decoupled | LandmarkGazeBridge | LandmarkGazeBridge (Stage 3 only) |
| Ray constraint | None | Ray-to-target loss | Ray-to-target loss |
| head_R at inference | N/A | Required | **Not needed** |
| Gradient clipping | max_norm=1.0 | max_norm=1.0 | **Phase-dependent (5.0 / 2.0)** |

---

## Backbone: RepNeXt (Dual)

RepNeXt is a family of lightweight CNNs using reparameterized depthwise convolutions with multi-scale feature extraction. Both backbones use **distilled, non-fused** (`.pth`) weights for training.

### Main Backbone: RepNeXt-M3 (landmarks + gaze)

Pretrained weights: `repnext_m3_distill_300e.pth` (ImageNet-1K, distilled, 80.7% top-1)

| Stage | Output Shape | Stride | Channels |
|-------|-------------|--------|----------|
| C1 | `(B, 64, 56, 56)` | 4 | 64 |
| C2 | `(B, 128, 28, 28)` | 8 | 128 |
| C3 | `(B, 256, 14, 14)` | 16 | 256 |
| C4 | `(B, 512, 7, 7)` | 32 | 512 |

7.8M params, 1.3G MACs. Features are fused by PANet before feeding task heads.

### Pose Backbone: RepNeXt-M1 (head pose)

Pretrained weights: `repnext_m1_distill_300e.pth` (ImageNet-1K, distilled, 78.8% top-1)

| Stage | Output Shape | Stride | Channels |
|-------|-------------|--------|----------|
| C1 | `(B, 48, 56, 56)` | 4 | 48 |
| C2 | `(B, 96, 28, 28)` | 8 | 96 |
| C3 | `(B, 192, 14, 14)` | 16 | 192 |
| C4 | `(B, 384, 7, 7)` | 32 | 384 |

4.8M params, 0.8G MACs. **Gradient-isolated** from the main backbone -- pose gradients don't interfere with landmark/gaze feature learning.

### Why Gradient Isolation?

v4 training showed adversarial optimization when pose, landmark, and gaze tasks shared one backbone: train gaze improved (79 deg -> 25 deg) while val gaze worsened (43 deg -> 80 deg). The shared representation couldn't satisfy conflicting gradient directions. Following MAGE (CVPR 2025), which uses a separate ResNet-18 for pose, we use an independent lightweight RepNeXt-M1.

### Weight Selection

| Format | File Pattern | Use Case |
|--------|-------------|----------|
| **Training** (`.pth`) | `repnext_mN_distill_300e.pth` | Fine-tuning (has BatchNorm, multi-branch) |
| Fused (`.pt`) | `repnext_mN_distill_300e_fused.pt` | Inference-only (JIT, BN fused) -- **cannot fine-tune** |

### RepNeXt Variant Comparison

| Model | Params | MACs | Top-1 | Role |
|-------|--------|------|-------|------|
| M0 | 2.3M | 0.4G | 74.2% | Too weak for pose |
| **M1** | **4.8M** | **0.8G** | **78.8%** | **Pose backbone** |
| M2 | 6.5M | 1.1G | 80.1% | Alternative pose (heavier) |
| **M3** | **7.8M** | **1.3G** | **80.7%** | **Main backbone** |

Source: `backbone/repnext.py`, `backbone/repnext_utils.py`

---

## PANet (Path Aggregation Network)

YOLOv8-style multi-scale fusion with SiLU activations. Projects all main backbone stages to 256 channels, then fuses top-down and bottom-up. Applied to main backbone only (not pose backbone).

```
Top-down:  P5 -> upsample(2x) + P4 -> upsample(2x) + P3 -> upsample(2x) + P2
Bottom-up: P2 -> downsample(2x) + P3 -> downsample(2x) + P4 -> downsample(2x) + P5
```

Output: `[P2, P3, P4, P5]` all with 256 channels.

Source: `RayNet/panet.py`

---

## Coordinate Attention

Applied to **P2** (landmarks), **P5** (gaze), and **pose backbone C4** (head pose). Encodes spatial position via directional pooling along x and y axes separately, preserving geometric context critical for landmark localization and pose estimation. Captures face asymmetry (yaw) and vertical tilt (pitch).

Source: `RayNet/coordatt.py` (Hou et al., 2021)

---

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
AdaptiveAvgPool2d(1) -> (B, 384)
    |
Linear(384, 256) -> pose_feat (B, 256)   [fused with cam_embed]
    |
Linear(256, 9) -> pose_out (B, 9)
    |
    +-- [:6] -> pred_pose_6d (B, 6)       [6D rotation, aux geodesic loss]
    +-- [6:] -> raw translation (B, 3)
                  |
                  +-- tanh(tx) -> [-1, 1]   (image-plane horizontal)
                  +-- tanh(ty) -> [-1, 1]   (image-plane vertical)
                  +-- exp(tz)  -> (0, +inf)  (depth)
                  |
                  -> pred_pose_t (B, 3)    [aux translation loss]
```

### 9D Pose Representation

The pose head outputs 9 values: 6 for rotation + 3 for translation.

**6D Rotation** (Zhou et al., CVPR 2019): The first 6 values represent the first two columns of the rotation matrix. Gram-Schmidt orthogonalization reconstructs the full 3x3 R:

```
a1 = r6d[:, 0:3]  (first column)
a2 = r6d[:, 3:6]  (second column)

b1 = normalize(a1)
b2 = normalize(a2 - (b1 . a2) * b1)   # orthogonalize
b3 = b1 x b2                           # cross product

R = [b1, b2, b3]  (proper rotation, det=+1)
```

This is the optimal continuous representation of SO(3) for neural networks. Unlike quaternions or Euler angles, it has no discontinuities or gimbal lock.

**3D Translation** with normalization:

| Component | Activation | Range | Purpose |
|-----------|-----------|-------|---------|
| tx | `tanh` | [-1, 1] | Image-plane horizontal offset |
| ty | `tanh` | [-1, 1] | Image-plane vertical offset |
| tz | `exp` | (0, +inf) | Depth (trained in log-space for scale invariance) |

The `exp` activation for depth pairs naturally with the log-space SmoothL1 loss: `log(exp(raw)) = raw`, so the depth loss effectively becomes `SmoothL1(raw, log(gt_z))`.

**GT normalization requirement**: Ground-truth `head_t` must be pre-normalized to match: tx, ty in [-1, 1], tz as positive metric depth.

### Geodesic Loss (Rotation)

```
L_geo = arccos( (tr(R_pred^T @ R_gt) - 1) / 2 )
```

Measures the actual rotation angle between predicted and GT orientation on the SO(3) manifold. Unlike L1/L2 on matrix elements, geodesic loss treats all rotation axes equally.

### Translation Loss

```
L_trans = SmoothL1(pred_xy, gt_xy) + SmoothL1(log(pred_z), log(gt_z))
```

Log-space comparison for depth makes the loss scale-invariant -- a 10% depth error at 50cm is penalized equally to 10% at 5m.

~4.9M params total (backbone + CoordAtt + projection + aux head).

Source: `RayNet/raynet.py:PoseEncoder`, `RayNet/losses.py:geodesic_loss`, `RayNet/losses.py:translation_loss`

---

## LandmarkGazeBridge (v4)

Cross-attention module where P5 gaze features (query) attend to P2 landmark features (key/value). This ties gaze and landmark tasks in a shared latent space so the gaze head can leverage spatial landmark geometry.

```
P2_att (B, 256, 56, 56) -> AvgPool(7) -> flatten -> (B, 49, 256) [key/value]
P5_pooled (B, 256) -> unsqueeze -> (B, 1, 256) [query]
    -> MultiheadAttention (4 heads, pre-norm, residual)
    -> (B, 256)
```

**Disabled in Stage 1 and Stage 2** to prevent poisoning 3D geometry with augmentation artifacts from the image-space crop. GazeGene's random crop translation means P2 landmark features encode positions in augmented crop space, while gaze is in CCS. Only enabled in Stage 3 after gaze is verified to converge independently.

~0.4M params. Source: `RayNet/raynet.py:LandmarkGazeBridge`

---

## CameraEmbedding (v4.1)

Encodes camera extrinsics into a d_model-dim vector. No head pose information -- that's learned implicitly by PoseEncoder.

```
R_cam (B, 3, 3) -> flatten -> (B, 9)
T_cam (B, 3)
    -> concat -> (B, 12)
    -> MLP(12 -> 64 -> 256)
    -> (B, 256) + pose_feat (B, 256)  [additive fusion]
    -> cam_embed for CrossViewAttention
```

Camera extrinsics tell the transformer WHERE each camera is. Pose features tell it HOW the head is oriented. Combined, they provide full geometric context for cross-view reasoning.

~0.02M params. Source: `RayNet/raynet.py:CameraEmbedding`

---

## CrossViewAttention

Pre-norm Transformer Encoder for cross-view gaze feature fusion. Receives fused camera + pose embedding.

```
pooled (B, 256) + cam_embed (B, 256)
    -> reshape to (G, V, 256)   where G = B/V groups, V = 9 views
    -> TransformerEncoder (2 layers, 4 heads, d_ff=512, GELU, dropout=0.1)
    -> reshape to (B, 256)
```

Self-attention across V views allows the model to learn multi-view consistency through attention, not just post-hoc loss penalties. Single-view (n_views=1) bypasses the encoder (identity).

~1.05M params. Source: `RayNet/raynet.py:CrossViewAttention`

---

## Task Head A: Iris/Pupil Landmark Detection

**Input**: `P2_att` (B, 256, 56, 56) -- CoordAtt-enhanced P2 from PANet

Architecture: Heatmap branch (Conv->BN->ReLU->Conv) + offset branch -> soft-argmax for differentiable subpixel coordinates.

| Index | Type | Source |
|-------|------|--------|
| 0-9 | Iris contour | 100 iris mesh points, subsampled at [0, 10, ..., 90] |
| 10-13 | Pupil boundary | 4 iris points closest to 2D pupil center |

**Output**: `landmark_coords` (B, 14, 2) in feature-map space [0, 56) and `landmark_heatmaps` (B, 14, 56, 56) raw logits.

Source: `RayNet/heads.py:IrisPupilLandmarkHead`

---

## Task Head B: Optical Axis Regression

**Input**: `P5_att` (B, 256, 7, 7) -- CoordAtt-enhanced P5 from PANet

Processing pipeline:
1. `pool_features`: GAP -> (B, 256)
2. LandmarkGazeBridge (if enabled): cross-attend to P2 landmarks -> (B, 256)
3. CameraEmbed + pose_feat fusion -> cam_embed
4. CrossViewAttention with cam_embed -> (B, 256)
5. `predict_from_pooled`: FC(256->128->2) -> pitch/yaw -> unit 3D vector

The head is split into `pool_features()` and `predict_from_pooled()` to allow the bridge and cross-view attention to be inserted between them.

**Output**: `gaze_vector` (B, 3) unit vector in CCS, `gaze_angles` (B, 2) pitch/yaw in radians.

Source: `RayNet/heads.py:OpticalAxisHead`

---

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
LandmarkGazeBridge [Stage 3]:       (B, 256)

=== Pose Backbone (RepNeXt-M1) ===
C4 (stride 32):                     (B, 384,   7,  7)
CoordAtt(C4):                       (B, 384,   7,  7)
pose_feat:                          (B, 256)
pred_pose_6d:                       (B,   6)
pred_pose_t:                        (B,   3)

=== Fusion ===
CameraEmbed + pose_feat:            (B, 256)
CrossViewAttention:                  (B, 256)

=== Outputs ===
Gaze vector:                        (B,   3)
Gaze angles:                        (B,   2)
```

## Model Output Dictionary

```python
predictions = model(images, n_views=9, R_cam=R_cam, T_cam=T_cam, use_bridge=True)

predictions = {
    'landmark_coords':   (B, 14, 2),      # feature-map space [0, 56)
    'landmark_heatmaps': (B, 14, 56, 56), # raw logits
    'gaze_vector':       (B, 3),          # unit vector in CCS
    'gaze_angles':       (B, 2),          # pitch, yaw in radians
    'pred_pose_6d':      (B, 6),          # 6D rotation (for aux geodesic loss)
    'pred_pose_t':       (B, 3),          # 3D translation (for aux translation loss)
}
```

## Parameter Budget

| Component | Params | Notes |
|-----------|--------|-------|
| RepNeXt-M3 (main) | 7.8M | Landmarks + gaze features |
| PANet | ~1.0M | Multi-scale fusion |
| CoordAtt x3 | ~0.1M | P2, P5, pose C4 |
| LandmarkGazeBridge | ~0.4M | Cross-attention (Stage 3 only) |
| CameraEmbedding | ~0.02M | Extrinsics encoding |
| CrossViewAttention | ~1.05M | Multi-view transformer (2 layers) |
| RepNeXt-M1 (pose) | 4.8M | Head pose features (gradient-isolated) |
| PoseEncoder head | ~0.1M | Projection + 9D output |
| Task heads | ~0.3M | Landmarks + gaze FC |
| **Total** | **~15.6M** | |
