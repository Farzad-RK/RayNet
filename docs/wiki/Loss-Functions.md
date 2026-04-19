# Loss Functions

All loss functions used in RayNet v5 training.

## Overview

```
Total Loss = lam_lm           * Landmark_Loss
           + lam_gaze         * Gaze_Loss
           + lam_eyeball      * Eyeball_Center_Loss
           + lam_pupil        * Pupil_Center_Loss
           + lam_geom_angular * Geometric_Angular_Loss
           + lam_pose         * Pose_Rotation_Loss (geodesic)
           + lam_trans        * Pose_Translation_Loss (log-depth SmoothL1)
           + lam_ray          * Ray_Target_Loss
           + mv_weight * (lam_reproj * Gaze_Ray_Consistency
                        + lam_mask  * Landmark_Shape_Consistency)
```

---

## 1. Landmark Loss

**Source**: `RayNet/losses.py:landmark_loss`

Combines heatmap regression with coordinate regression for robust landmark detection.

### Heatmap MSE

Ground-truth heatmaps are generated as 2D Gaussians centered at each landmark:

```
GT_heatmap(x, y) = exp(-((x - cx)^2 + (y - cy)^2) / (2 * sigma^2))
```

Where `(cx, cy)` is the ground-truth landmark position in feature-map space (56x56).

```
L_heatmap = MSE(sigmoid(pred_heatmaps), GT_heatmaps)
```

**Sigma schedule**: starts at 2.0 (Phase 1, broader targets) and tightens to 1.0 (Phase 3, sub-pixel precision).

### Coordinate L1

Direct L1 regression on predicted vs ground-truth coordinates in feature-map space (0 to 56):

```
L_coord = L1(pred_coords, gt_coords)
```

### Combined

```
L_landmark = L_heatmap + L_coord
```

---

## 2. Gaze Loss

**Source**: `RayNet/losses.py:gaze_loss`

L1 loss on predicted vs ground-truth unit gaze vectors in camera coordinate space (CCS). This follows the GazeGene paper (Sec 4.1.1).

```
L_gaze = L1(pred_gaze, gt_gaze)     (both unit vectors in CCS)
```

### Why L1 Instead of Angular (acos)?

`torch.acos(cosine_similarity)` has an infinite gradient singularity at cos_sim = +/-1:

```
d/dx acos(x) = -1/sqrt(1 - x^2)  -->  infinity when x = +/-1
```

L1 on unit vectors is numerically stable everywhere, consistent with the GazeGene paper, and monotonically related to angular error.

### Angular Error (Metrics Only)

Angular error is computed for logging using the `atan2` formulation, but is **not backpropagated**:

```python
cross = cross_product(pred, gt)
dot = dot_product(pred, gt)
angle = atan2(||cross||, dot)       # stable everywhere
```

**Source**: `RayNet/losses.py:angular_error`

---

## 3. Eyeball Center Loss

**Source**: `RayNet/losses.py:eyeball_center_loss`

L1 loss on predicted vs GT eyeball center in camera coordinate space (CCS, centimeters). From GazeGene Sec 4.2.2, loss 1.

```
L_eyeball = L1(pred_eyeball_3d, gt_eyeball_3d)     (both in CCS, cm)
```

Active in Stage 2+ (off in Stage 1 where gaze is disabled).

---

## 4. Pupil Center Loss

**Source**: `RayNet/losses.py:pupil_center_loss`

L1 loss on predicted vs GT pupil center in CCS (centimeters). From GazeGene Sec 4.2.2, loss 2.

```
L_pupil = L1(pred_pupil_3d, gt_pupil_3d)     (both in CCS, cm)
```

Active in Stage 2+ (off in Stage 1).

---

## 5. Geometric Angular Loss

**Source**: `RayNet/losses.py:geometric_angular_loss`

Angular error between the optical axis derived from predicted 3D geometry and the GT optical axis (GazeGene Sec 4.2.2, loss 4).

```
pred_optical_axis = normalize(pred_pupil - pred_eyeball)
L_geom_angular = angular_error(pred_optical_axis, gt_optical_axis)
```

This ensures geometric consistency — the model cannot cheat by predicting correct eyeball/pupil positions but wrong relative direction. Uses `atan2` for numerical stability.

Delayed until Phase 3 in Stage 2 (geometry must converge first). Active earlier in Stage 3 where bridges provide additional structure.

---

## 6. Pose Rotation Loss (Geodesic)

**Source**: `RayNet/losses.py:pose_prediction_loss`, `RayNet/losses.py:geodesic_loss`

Auxiliary loss on the PoseEncoder's 6D rotation prediction. The 6D output is reconstructed to a 3x3 rotation matrix via Gram-Schmidt, then compared to GT using geodesic distance on SO(3).

```
R_pred = gram_schmidt(pred_6d)              # (B, 3, 3)
R_diff = R_pred^T @ R_gt
L_geo  = arccos( (trace(R_diff) - 1) / 2 ) # mean over batch
```

This measures the actual rotation angle needed to go from predicted to GT orientation, respecting the SO(3) manifold geometry. Numerically stabilized with clamp to avoid NaN from arccos at +/-1.

**Source**: `RayNet/losses.py:rotation_6d_to_matrix`, `RayNet/losses.py:geodesic_loss`

---

## 7. Pose Translation Loss (Log-Depth SmoothL1)

**Source**: `RayNet/losses.py:translation_loss`

Auxiliary loss on the PoseEncoder's 3D translation prediction. Uses separate treatment for image-plane offsets (xy) and depth (z):

```
L_trans = SmoothL1(pred_xy, gt_xy) + SmoothL1(log(pred_z), log(gt_z))
```

| Component | Pred activation | GT range | Loss |
|-----------|----------------|----------|------|
| tx, ty | `tanh` -> [-1, 1] | [-1, 1] (normalized) | SmoothL1 |
| tz | `exp` -> (0, +inf) | positive metric depth | SmoothL1 in log-space |

Log-space comparison for depth makes the loss **scale-invariant**: a 10% depth error at 50cm is penalized equally to 10% at 5m. Since `log(exp(raw)) = raw`, the depth loss effectively becomes `SmoothL1(raw, log(gt_z))`.

**GT normalization requirement**: Ground-truth `head_t` from GazeGene (`head_T_vec`) is in centimeters. It must be pre-normalized so tx, ty are in [-1, 1] and tz is positive depth before the loss sees it.

---

## 8. Ray-to-Target Loss

**Source**: `RayNet/losses.py:ray_target_loss`

Geometric constraint that ties gaze direction to a physical target location:

```
target_hat = eyeball_center + gaze_depth * pred_gaze
L_ray = L1(target_hat, gt_gaze_target)
```

Uses ground-truth depth along the gaze ray and known eye center to reconstruct the 3D gaze target from the predicted gaze direction. This provides an explicit geometric signal beyond direction-only loss.

Active in Stage 2 Phase 2+ and Stage 3 Phase 2+.

---

## 9. Gaze Ray Consistency Loss

**Source**: `RayNet/multiview_loss.py:gaze_ray_consistency_loss`

Enforces that predicted gaze vectors from different camera views of the same subject, transformed to world frame via camera extrinsics, point in the same direction.

```
g_world_v = R_cam_v^T @ g_v           # transform to world frame
g_mean = normalize(mean(g_world))      # group consensus
L_consist = L1(g_world, detach(g_mean))
```

Uses `R_cam` (static camera extrinsics) for the world-frame transformation. All operations involve unit vectors -- numerically stable under AMP float16.

See [[Multi-View Consistency]] for details.

---

## 10. Landmark Shape Consistency Loss

**Source**: `RayNet/multiview_loss.py:landmark_shape_consistency_loss`

Enforces that the spatial pattern of predicted landmarks is consistent across camera views. Translation-and-scale-invariant (Procrustes-style):

```
pts_centered = pts - centroid
pts_normalized = pts_centered / rms_distance
L_shape = SmoothL1(pts_normalized_i, pts_normalized_j)
```

See [[Multi-View Consistency]] for details.

---

## Total Loss Assembly

**Source**: `RayNet/losses.py:total_loss`

```python
total = lam_lm * landmark_loss + lam_gaze * gaze_loss

# 3D eyeball structure (active when lam_eyeball/pupil/geom_angular > 0)
total += lam_eyeball * eyeball_center_loss
total += lam_pupil * pupil_center_loss
total += lam_geom_angular * geometric_angular_loss

# Ray-to-target (active when lam_ray > 0 and GT available)
total += lam_ray * ray_target_loss

# Pose rotation (active when lam_pose > 0 and pose encoder exists)
total += lam_pose * pose_prediction_loss

# Pose translation (active when lam_trans > 0 and GT available)
total += lam_trans * translation_loss

# Multi-view consistency (added in train loop when multiview enabled)
total += mv_weight * (lam_reproj * gaze_consist + lam_mask * shape_consist)
```

The `mv_weight` ramps linearly from 0 to 1 over the first 10 epochs to smooth multi-view loss activation.

### Logged Components

| Component | Key in CSV | Description |
|-----------|-----------|-------------|
| Total loss | `loss` / `train_total` | Weighted sum of all active losses |
| Landmark | `landmark` / `train_landmark` | Heatmap MSE + coord L1 (normalized by feat area) |
| Angular error | `angular_deg` / `train_angular_deg` | Degrees (metric only, not in loss) |
| Eyeball center | `eyeball` | L1 on 3D eyeball center (cm) |
| Pupil center | `pupil` | L1 on 3D pupil center (cm) |
| Geometric angular | `geom_angular` | Angular error from predicted geometry (rad) |
| Gaze consistency | `gaze_consist` / `train_reproj` | Multi-view gaze ray agreement |
| Shape consistency | `shape` / `train_mask` | Multi-view landmark shape agreement |
| Ray-to-target | `ray_target` / `train_ray_target` | Gaze ray geometric constraint |
| Pose rotation | `pose` / `train_pose` | Geodesic loss in radians |
| Translation | `translation` / `train_translation` | Log-depth SmoothL1 |

---

## Loss Weight Schedule (All Stages)

### Stage 1: Landmark + Pose Baseline

| Phase | Epochs | lam_lm | lam_gaze | lam_eyeball | lam_pupil | lam_geom_angular | lam_pose | lam_trans | lam_ray | Multi-view |
|-------|--------|--------|----------|-------------|-----------|-----------------|----------|-----------|---------|------------|
| 1 | 1-8 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 0.5 | 0.0 | Off |
| 2 | 9-15 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 | Off |

### Stage 2: Eye-Crop Gaze on Frozen Face Path

The face path (`shared_stem` + `landmark_branch` + `pose_branch`) is frozen in phases 1-2 so `lam_lm = lam_pose = lam_trans = 0` — those losses have no gradient sink. Phase 3 unfreezes everything for gentle joint fine-tuning.

| Phase | Epochs | `freeze_face` | lam_lm | lam_gaze | lam_eyeball | lam_pupil | lam_geom_angular | lam_pose | lam_trans | lam_ray | lam_reproj | lam_mask |
|-------|--------|---------------|--------|----------|-------------|-----------|-----------------|----------|-----------|---------|------------|----------|
| 1 | 1-8 | True | 0.0 | 1.0 | 0.3 | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2 | 9-15 | True | 0.0 | 1.0 | 0.5 | 0.5 | 0.2 | 0.0 | 0.0 | 0.2 | 0.1 | 0.05 |
| 3 | 16-25 | False | 0.5 | 1.0 | 0.5 | 0.5 | 0.3 | 0.3 | 0.3 | 0.3 | 0.1 | 0.05 |

### Stage 3: Full Pipeline with Bridges

| Phase | Epochs | lam_lm | lam_gaze | lam_eyeball | lam_pupil | lam_geom_angular | lam_pose | lam_trans | lam_ray | lam_reproj | lam_mask |
|-------|--------|--------|----------|-------------|-----------|-----------------|----------|-----------|---------|------------|----------|
| 1 | 1-5 | 1.0 | 0.3 | 0.3 | 0.3 | 0.1 | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 |
| 2 | 6-15 | 1.0 | 0.5 | 0.5 | 0.5 | 0.2 | 0.5 | 0.5 | 0.1 | 0.05 | 0.02 |
| 3 | 16-25 | 0.5 | 1.0 | 0.5 | 0.5 | 0.3 | 0.5 | 0.5 | 0.3 | 0.1 | 0.05 |

---

## Gradient Clipping

Gradient clipping varies by training phase to balance multi-task learning:

| Phase | max_norm | Rationale |
|-------|----------|-----------|
| 1 | 5.0 | Aggressive -- allows large gradients during multi-task warmup |
| 2+ | 2.0 | Conservative -- prevents gaze/pose gradient interference during fine-tuning |
