# Loss Functions

All loss functions used in RayNet v2 training.

## Overview

```
Total Loss = lam_lm * Landmark_Loss
           + lam_gaze * Angular_Loss
           + lam_reproj * Reprojection_Consistency_Loss    (Phase 2+)
           + lam_mask * Triangulation_Masking_Loss          (Phase 2+)
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

Direct L1 regression on predicted vs ground-truth coordinates:

```
L_coord = L1(pred_coords, gt_coords)
```

Both are in feature-map space (0 to 56).

### Combined

```
L_landmark = L_heatmap + L_coord
```

---

## 2. Angular Loss

**Source**: `RayNet/losses.py:angular_loss`

Measures the angular error between predicted and ground-truth optical axis directions.

```
cos_sim = dot(pred_gaze, gt_gaze)     (both unit vectors)
angle = arccos(clamp(cos_sim, -1, 1))
L_angular = mean(angle)               (in radians)
```

**Why L1 on angle instead of cosine similarity?** Cosine similarity saturates for large errors early in training (cos(pi) = -1), providing weak gradients. L1 on the angle gives a constant gradient magnitude regardless of error size, which is more robust during early training when predictions may be nearly random.

---

## 3. Total Single-View Loss

**Source**: `RayNet/losses.py:total_loss`

```python
total = lam_lm * landmark_loss(pred_hm, pred_coords, gt_coords, H, W, sigma)
      + lam_gaze * angular_loss(pred_gaze, gt_gaze)
```

Returns `(total_loss, components_dict)` where components contains detached values for logging:
- `landmark_loss`
- `angular_loss`
- `angular_loss_deg` (converted to degrees)
- `total_loss`

---

## 4. Reprojection Consistency Loss

**Source**: `RayNet/multiview_loss.py:reprojection_consistency_loss`

Enforces geometric consistency: landmarks predicted in camera i, when reprojected into camera j, should match camera j's own predictions.

### Pipeline

For each randomly sampled camera pair (i, j):

1. **Denormalize** both cameras' predictions from feature space to original pixel space via `M_norm_inv`
2. **Unproject** camera i's landmarks to 3D:
   ```
   ray = K_i_inv @ [u, v, 1]^T
   P_3d = ray * Z_i    (Z from GT eyeball center)
   ```
3. **Transform** to camera j's frame:
   ```
   R_rel = R_j @ R_i^T
   t_rel = T_j - R_rel @ T_i
   P_j = R_rel @ P_i + t_rel
   ```
4. **Project** into camera j's image: `[u', v'] = K_j @ P_j / P_j_z`
5. **L1 loss** between reprojected points and camera j's predictions

Both directions (i->j and j->i) are computed and averaged. Default: 2 random pairs per batch.

### Depth Source

- **Option A** (current): GT eyeball center Z component in camera coordinates, clamped to min 100mm
- **Option B** (future): Model's own iris depth estimate from ellipse fitting

---

## 5. Triangulation Masking Loss

**Source**: `RayNet/multiview_loss.py:triangulation_masking_loss`

Masks one camera and uses two others to triangulate the eye position, then verifies it against the masked camera's prediction.

### Pipeline

1. Randomly select 3 cameras: `mask_cam`, `tri_a`, `tri_b`
2. Compute landmark **centroids** (mean of 14 points) as eye center proxy in original pixel space
3. Build projection matrices: `P = K @ [R | T]` for `tri_a` and `tri_b`
4. **Triangulate** via DLT (Direct Linear Transform):
   - Build 4x4 system from cross-product equations
   - Solve via SVD (last column of V)
   - **Detach** the result (no gradients through SVD)
5. **Project** triangulated 3D point into `mask_cam`'s image
6. **L1 loss** between projected point and `mask_cam`'s predicted centroid

### Why Detach?

SVD gradients are numerically unstable for the smallest singular value. By detaching the triangulated point, gradients flow only through the masked camera's predictions. The triangulated point acts as a pseudo-ground-truth anchor.

---

## 6. Combined Multi-View Loss

**Source**: `RayNet/multiview_loss.py:multiview_consistency_loss`

```python
L_multiview = lam_reproj * L_reprojection + lam_mask * L_triangulation
```

Returns `(total_mv_loss, {'reproj_loss': ..., 'mask_loss': ...})`.

---

## Loss Weight Schedule

| Phase | Epochs | lam_lm | lam_gaze | lam_reproj | lam_mask |
|-------|--------|--------|----------|------------|----------|
| 1 | 1-5 | 1.0 | 0.0 | 0.0 | 0.0 |
| 2 | 6-15 | 1.0 | 0.3 | 0.1 | 0.05 |
| 3 | 16-30 | 0.5 | 0.5 | 0.2 | 0.1 |

**Rationale**:
- Phase 1: Landmark-only. Gaze head receives no gradients.
- Phase 2: Gaze introduced gently (0.3). Multi-view at low weight (0.1/0.05) to avoid destabilizing landmarks.
- Phase 3: Equal task weighting. Multi-view at full strength. Sigma tightened for fine-grained precision.

---

## Regularization Losses (EyeFLAME Module)

The experimental EyeFLAME model (`RayNet/EyeFLAME/loss.py`) uses additional losses:

| Loss | Formula | Purpose |
|------|---------|---------|
| `scale_reg` | `mean((scale - 1)^2)` | Keep weak perspective scale near 1 |
| `trans_reg` | `mean(translation^2)` | Prevent translation drift |
| `norm_centers_reg` | `mean(abs(normalized_centers))` | Regularize 3D structure |

These are not used in the main RayNet pipeline.
