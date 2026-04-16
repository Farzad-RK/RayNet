# Multi-View Consistency

RayNet v5 exploits the 9 synchronized cameras in GazeGene to impose geometric constraints during training. This provides "free" supervision that enforces anatomically consistent predictions across viewpoints.

**Source**: `RayNet/multiview_loss.py`

## Overview

Two ray-based losses enforce multi-view consistency using unit vectors and camera extrinsics:

| Loss | Purpose | Weight | Active |
|------|---------|--------|--------|
| **Gaze ray consistency** | Gaze from all views, transformed to world frame via R_cam, should agree | lam_reproj | Stage 2+ Phase 2+ |
| **Landmark shape consistency** | Relative landmark patterns should be anatomically consistent | lam_mask | Stage 2+ Phase 2+ |

All operations use **unit vectors and feature-map coordinates** -- no raw 3D camera-space coordinates. Numerically stable under AMP float16.

## Design: Why Ray-Based?

An earlier implementation used pixel-space geometric operations (unproject via K, transform between frames, reproject). This was numerically unstable:

1. **Large coordinate values**: focal lengths ~500-2000, depths ~5000mm cause precision loss in float16
2. **Matrix inversions**: `inv(K)` in float16 produces garbage
3. **SVD limitations**: not supported in float16 on CUDA

The ray-based approach works entirely with unit vectors and orthogonal rotations -- bounded inputs, bounded outputs, no inversions.

---

## Batch Structure

### Without Multi-View (Phase 1)

Each camera view is an independent sample:

```
Standard batch: (B, ...)    e.g. B=504 random samples
```

### With Multi-View (Phase 2+)

The dataloader ensures 9 consecutive samples form one (subject, frame) group:

```
Multi-view batch: (G*9, ...)    e.g. G=16 groups, 144 total samples
```

`reshape_multiview` converts `(G*9, ...)` -> `(G, 9, ...)` for cross-view operations.

---

## Gaze Ray Consistency Loss

**Source**: `RayNet/multiview_loss.py:gaze_ray_consistency_loss`

### Principle

For the same subject looking at the same target, all 9 camera views should predict gaze vectors that, when transformed to world frame, point in the same direction.

### Algorithm

1. Each view `v` predicts gaze vector `g_v` in **camera coordinate space** (CCS, unit vector)
2. Transform to world frame using camera extrinsics:
   ```
   g_world_v = R_cam_v^T @ g_v
   ```
   (`R_cam` is orthogonal, so `R_cam^T = R_cam^{-1}`)
3. Normalize the world-frame vectors
4. Compute group mean: `g_mean = normalize(mean(g_world_v) over views)`
5. L1 loss between each view's world gaze and the detached group mean:
   ```
   L = L1(g_world, detach(g_mean))
   ```

### Why R_cam (not R_norm)?

RayNet uses camera extrinsics (`R_cam` from `camera_info.pkl`) rather than per-frame normalization rotation (`R_norm`). `R_cam` is **static per camera** -- it never changes across frames for the same camera. This provides more stable consistency targets than frame-dependent normalization rotations.

### Gradient Flow

```
pred_gaze (CCS)                 <-- gradients flow here
    |
R_cam^T (fixed rotation)        <-- no learnable params
    |
g_world (world frame)
    |
g_mean (detached)                <-- no gradients through mean
    |
L1 Loss                          <-- loss signal
```

Detaching `g_mean` ensures each view receives gradients to move toward group consensus, without the mean being pulled by any single view.

---

## Landmark Shape Consistency Loss

**Source**: `RayNet/multiview_loss.py:landmark_shape_consistency_loss`

### Principle

The spatial arrangement of 14 eye landmarks should be anatomically consistent regardless of camera view. Absolute positions differ, but relative shape should be preserved.

### Algorithm

For each sampled camera pair (i, j):

1. Extract predicted landmarks in feature-map space
2. **Center**: subtract centroid (mean of 14 points)
3. **Scale-normalize**: divide by RMS distance from centroid
4. **Smooth L1 loss** between normalized shapes

```
pts_c = pts - mean(pts, axis=landmarks)     # center
scale = rms(||pts_c||)                       # scale factor
pts_n = pts_c / scale                        # normalized shape
L = SmoothL1(pts_n_i, pts_n_j)
```

Procrustes-style normalization removes translation and scale, comparing only intrinsic landmark shape.

---

## Combined Multi-View Loss

**Source**: `RayNet/multiview_loss.py:multiview_consistency_loss`

```python
L_multiview = lam_gaze_consist * gaze_ray_consistency_loss(pred_gaze, R_cam)
            + lam_shape * landmark_shape_consistency_loss(pred_coords)
```

### Inputs Required

| Input | Shape | Source |
|-------|-------|--------|
| `pred_gaze` | (B, 3) | Model output (`gaze_vector`) |
| `pred_coords` | (B, N, 2) | Model output (`landmark_coords`) |
| `R_cam` | (B, 3, 3) | Dataset (`R_cam` field, camera extrinsics) |

No camera intrinsics (K), inverse warps, or 3D coordinates needed.

### Smooth Ramp

Multi-view loss is ramped linearly over the first 10 epochs:

```python
mv_weight = min(1.0, epoch / 10.0)
loss += mv_weight * mv_loss
```

This prevents sudden loss spikes when multi-view is first activated.

---

## Weight Schedule

Weights are set per stage/phase in `STAGE_CONFIGS`:

| Stage | Phase | lam_reproj (gaze consist) | lam_mask (shape) |
|-------|-------|--------------------------|-----------------|
| 1 | all | 0.0 | 0.0 |
| 2 | 1 | 0.0 | 0.0 |
| 2 | 2 | 0.05 | 0.02 |
| 2 | 3 | 0.1 | 0.05 |
| 3 | 1 | 0.0 | 0.0 |
| 3 | 2 | 0.05 | 0.02 |
| 3 | 3 | 0.1 | 0.05 |

---

## AMP Compatibility

All operations are **float16-safe**:
- L1/SmoothL1 on unit vectors: bounded inputs, bounded gradients
- R_cam matrix-vector multiply: orthogonal matrix, unit vector output
- No matrix inversions, no SVD, no large-scale coordinates

## Batch Requirements

Multi-view losses require `batch_size % 9 == 0`. If the batch is not divisible by 9 (e.g., last batch), the loss returns 0 gracefully.
