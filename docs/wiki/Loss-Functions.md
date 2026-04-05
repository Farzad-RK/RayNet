# Loss Functions

All loss functions used in RayNet v2 training.

## Overview

```
Total Loss = lam_lm * Landmark_Loss
           + lam_gaze * Gaze_Loss                          (Phase 2+)
           + lam_gaze_consist * Gaze_Ray_Consistency_Loss   (Phase 2+)
           + lam_shape * Landmark_Shape_Consistency_Loss     (Phase 2+)
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

## 2. Gaze Loss

**Source**: `RayNet/losses.py:gaze_loss`

L1 loss on predicted vs ground-truth unit gaze vectors in normalized space. This follows the GazeGene paper (Sec 4.1.1), which uses L1 loss between predicted 3D unit head gaze vector and ground truth.

```
L_gaze = L1(pred_gaze, gt_gaze)     (both unit vectors in normalized space)
```

### Why L1 Instead of Angular (acos)?

Previous versions used `torch.acos(cosine_similarity)` as the training loss. This has an **infinite gradient singularity** at cos_sim = +/-1:

```
d/dx acos(x) = -1/sqrt(1 - x^2)  -->  infinity when x = +/-1
```

When gaze predictions start aligning with ground truth (cos_sim approaches 1.0), the gradient explodes and causes NaN. This was observed empirically at batch ~500 of Phase 2 training.

L1 on unit vectors is:
- Numerically stable everywhere (bounded gradients)
- Consistent with the GazeGene paper's methodology
- Monotonically related to angular error for unit vectors

### Angular Error (Metrics Only)

Angular error is computed for logging/metrics using the numerically stable `atan2` formulation, but is **not backpropagated**:

```python
cross = cross_product(pred, gt)
dot = dot_product(pred, gt)
angle = atan2(||cross||, dot)       # stable everywhere, no singularity
```

**Source**: `RayNet/losses.py:angular_error`

---

## 3. Total Single-View Loss

**Source**: `RayNet/losses.py:total_loss`

```python
total = lam_lm * landmark_loss(pred_hm, pred_coords, gt_coords, H, W, sigma)
      + lam_gaze * gaze_loss(pred_gaze, gt_gaze)
```

Returns `(total_loss, components_dict)` where components contains detached values for logging:
- `landmark_loss`
- `angular_loss` (in radians, from `angular_error`, detached)
- `angular_loss_deg` (converted to degrees)
- `total_loss`

---

## 4. Gaze Ray Consistency Loss

**Source**: `RayNet/multiview_loss.py:gaze_ray_consistency_loss`

Enforces that predicted gaze vectors from different camera views of the same subject point in the same world-frame direction.

### Algorithm

For each multi-view group (same subject, same frame, 9 cameras):

1. Each view `v` predicts a gaze vector `g_v` in **normalized space**
2. Transform to world frame: `g_world_v = R_norm_v^T @ g_v` (R_norm is orthogonal)
3. Compute group mean direction: `g_mean = normalize(mean(g_world))`
4. **L1 loss** between each view's world gaze and the detached group mean

```
L_gaze_consist = L1(g_world, detach(g_mean))
```

### Why This Works

All operations involve **unit vectors** (bounded [-1, 1]). No matrix inversions, no large-scale 3D coordinates, no SVD. Numerically stable under AMP float16.

### Why Not Pixel-Space Reprojection?

Earlier versions used a pixel-space reprojection approach (unproject landmarks to 3D, transform between camera frames, reproject to 2D). This was numerically unstable because:

- Camera intrinsic values (focal lengths ~500-2000) caused precision loss in float16
- Depth values (eyeball at ~5m = 5000mm) amplified errors
- Matrix inversions (`torch.linalg.inv(K)`) produced garbage in float16
- SVD (`torch.linalg.svd`) doesn't support float16 on CUDA
- Even in float32, large coordinate values caused inf/NaN in the loss

The ray-based approach avoids all of these issues by working entirely with unit vectors in normalized space.

---

## 5. Landmark Shape Consistency Loss

**Source**: `RayNet/multiview_loss.py:landmark_shape_consistency_loss`

Enforces that the spatial pattern of predicted landmarks is consistent across views, using a translation-and-scale-invariant comparison (Procrustes-style).

### Algorithm

For each sampled camera pair (i, j):

1. Extract landmarks in feature-map space (no denormalization needed)
2. **Center**: subtract centroid of 14 landmarks per view
3. **Scale-normalize**: divide by RMS distance from centroid
4. **Smooth L1 loss** between the normalized shapes

```
pts_centered = pts - mean(pts)
pts_normalized = pts_centered / rms_distance
L_shape = SmoothL1(pts_normalized_i, pts_normalized_j)
```

### Why Translation/Scale Invariant?

Each camera view has a different normalization warp (M_norm), so raw coordinates are not directly comparable. By removing translation and scale, we compare only the relative **shape** of the landmark configuration, which should be anatomically consistent across views.

---

## 6. Combined Multi-View Loss

**Source**: `RayNet/multiview_loss.py:multiview_consistency_loss`

```python
L_multiview = lam_gaze_consist * L_gaze_ray_consistency
            + lam_shape * L_landmark_shape_consistency
```

Returns `(total_mv_loss, {'gaze_consist_loss': ..., 'shape_loss': ...})`.

---

## Loss Weight Schedule

| Phase | Epochs | lam_lm | lam_gaze | lam_gaze_consist | lam_shape |
|-------|--------|--------|----------|------------------|-----------|
| 1 | 1-5 | 1.0 | 0.0 | 0.0 | 0.0 |
| 2 | 6-15 | 1.0 | 0.3 | 0.1 | 0.05 |
| 3 | 16-30 | 0.5 | 0.5 | 0.2 | 0.1 |

**Rationale**:
- **Phase 1**: Landmark-only warmup. Each camera view is treated as an independent sample (no view grouping). The gaze head receives no gradients — its weights remain at initialization.
- **Phase 2**: Gaze loss (L1 on unit vectors) introduced at 0.3 weight. Multi-view consistency activated with the grouped multiview dataloader (9 cameras per group). Gaze ray consistency ensures cross-view directional agreement; shape consistency regularizes landmark structure.
- **Phase 3**: Equal task weighting (0.5/0.5). Multi-view weights doubled. Sigma tightened for sub-pixel landmark precision.

---

## Methodology Alignment with GazeGene Paper

The loss design follows the GazeGene paper (Bao et al., CVPR 2025):

| Aspect | Paper (Sec 4.1) | Our Implementation |
|--------|-----------------|-------------------|
| Normalization | Zhang et al. 2018 | Yes, `normalize_sample()` with R_norm |
| Image size | 224 x 224 | 224 x 224 |
| Gaze target | 3D unit vector in normalized space | `optical_axis_norm = R_norm @ optical_axis` |
| Training loss | L1 between predicted and GT gaze vectors | `gaze_loss = F.l1_loss(pred, gt)` |
| Optimizer | Adam, betas=(0.5, 0.95), lr=1e-4 | AdamW, betas=(0.5, 0.95), lr varies by phase |
| Backbone | ResNet-18/50 | RepNeXt-M3 |

**Key design principle**: All training losses operate in **normalized space** (unit vectors, feature-map coordinates). Raw 3D camera coordinates and pixel-space geometric operations are never used in the loss computation, avoiding the numerical instability documented in Section 4.2 of the paper where normalization distortion along the z-axis makes 3D coordinate regression ill-conditioned.

---

## Regularization Losses (EyeFLAME Module)

The experimental EyeFLAME model (`RayNet/EyeFLAME/loss.py`) uses additional losses:

| Loss | Formula | Purpose |
|------|---------|---------|
| `scale_reg` | `mean((scale - 1)^2)` | Keep weak perspective scale near 1 |
| `trans_reg` | `mean(translation^2)` | Prevent translation drift |
| `norm_centers_reg` | `mean(abs(normalized_centers))` | Regularize 3D structure |

These are not used in the main RayNet pipeline.
