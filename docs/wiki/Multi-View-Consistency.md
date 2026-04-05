# Multi-View Consistency

RayNet v2 exploits the 9 synchronized cameras in GazeGene to impose geometric constraints during training. This provides "free" supervision that enforces anatomically consistent predictions across viewpoints.

**Source**: `RayNet/multiview_loss.py`

## Overview

Two ray-based losses enforce multi-view consistency using unit vectors in normalized space:

| Loss | Purpose | Weight | Active |
|------|---------|--------|--------|
| **Gaze ray consistency** | Gaze predictions from all views, transformed to world frame, should agree in direction | 0.1 -> 0.2 | Phase 2+ |
| **Landmark shape consistency** | Relative landmark patterns should be anatomically consistent across views | 0.05 -> 0.1 | Phase 2+ |

All operations use **unit vectors and normalized coordinates** — never raw 3D camera-space coordinates. This ensures numerical stability under AMP float16.

## Design Philosophy

### Why Not Pixel-Space Reprojection?

An earlier implementation used pixel-space geometric operations: unprojecting landmarks to 3D via camera intrinsics, transforming between camera frames, and reprojecting into target views. This approach was **numerically unstable** for several reasons:

1. **Large coordinate values**: Camera intrinsics have focal lengths ~500-2000, eyeball depths ~5000mm. These magnitudes cause catastrophic precision loss in float16 (only ~3 decimal digits).
2. **Matrix inversions**: `torch.linalg.inv(K)` on intrinsic matrices in float16 produces garbage.
3. **SVD limitations**: `torch.linalg.svd` doesn't support float16 on CUDA (`svd_cuda_gesvdjBatched not implemented for 'Half'`).
4. **Depth entanglement**: As noted in the GazeGene paper (Sec 4.2), without normalization, 3D coordinate regression is "ill-conditioned" — the network must simultaneously infer 3D position and gaze direction from a single 2D projection, with small depth errors causing massive downstream errors.

### The Ray-Based Solution

Following the GazeGene paper's methodology (Sec 4.1), all computations stay in **normalized space**:

- Gaze vectors are unit vectors (bounded [-1, 1])
- Landmark coordinates are in feature-map space (0-56)
- Cross-view consistency is enforced through R_norm transformations, not 3D reprojection

This eliminates all sources of numerical instability.

## Batch Structure

### Phase 1: Independent Samples

In Phase 1 (landmark warmup), multi-view is disabled. Each camera angle is treated as an **independent sample** — the standard dataloader draws randomly from all subjects, frames, and cameras with no view grouping:

```
Standard batch: (B, ...)    e.g. B=2048 random samples from any camera
```

This maximizes sample diversity for landmark training.

### Phase 2+: Multi-View Groups

When multi-view is activated, the dataloader ensures 9 consecutive samples form one (subject, frame) group:

```
Multi-view batch: (G*9, ...)    e.g. G=64 groups, 576 total samples
```

The `reshape_multiview` utility converts `(G*9, ...)` -> `(G, 9, ...)` for cross-view operations.

The multiview dataloader is created **lazily** — only when Phase 2 begins — to avoid downloading shards unnecessarily during Phase 1.

---

## Gaze Ray Consistency Loss

**Source**: `RayNet/multiview_loss.py:gaze_ray_consistency_loss`

### Principle

For the same subject looking at the same target, all 9 camera views should predict gaze vectors that, when transformed to a common world frame, point in the same direction.

### Algorithm

1. Each view `v` predicts gaze vector `g_v` in **normalized space** (unit vector)
2. Transform to world frame using the normalization rotation:
   ```
   g_world_v = R_norm_v^T @ g_v
   ```
   (`R_norm` is orthogonal, so `R_norm^T = R_norm^{-1}`)
3. Normalize the world-frame vectors
4. Compute group mean: `g_mean = normalize(mean(g_world_v) over views)`
5. L1 loss between each view's world gaze and the detached group mean:
   ```
   L = L1(g_world, detach(g_mean))
   ```

### Gradient Flow

```
pred_gaze (normalized space)    <-- gradients flow here
    |
R_norm^T (fixed rotation)       <-- no learnable params
    |
g_world (world frame)
    |
    v
g_mean (detached)               <-- no gradients through mean
    |
L1 Loss                         <-- loss signal
```

Detaching `g_mean` ensures each view receives gradients to move toward the group consensus, without the mean itself being pulled by any single view.

---

## Landmark Shape Consistency Loss

**Source**: `RayNet/multiview_loss.py:landmark_shape_consistency_loss`

### Principle

The spatial arrangement of 14 eye landmarks (10 iris + 4 pupil) should be anatomically consistent regardless of which camera view captures them. While the absolute pixel positions differ due to different normalization warps, the **relative shape** should be preserved.

### Algorithm

For each sampled camera pair (i, j):

1. Extract predicted landmarks in feature-map space (no denormalization needed)
2. **Center**: subtract the centroid (mean of 14 points)
3. **Scale-normalize**: divide by RMS distance from centroid
4. **Smooth L1 loss** between normalized shapes

```
pts_c = pts - mean(pts, axis=landmarks)     # center
scale = rms(||pts_c||)                       # scale factor
pts_n = pts_c / scale                        # normalized shape
L = SmoothL1(pts_n_i, pts_n_j)              # compare shapes
```

### Why Procrustes-Style Normalization?

Each camera view has a different Zhang normalization warp (M_norm), producing different absolute coordinates for the same anatomical landmarks. By removing translation (centering) and scale, we compare only the **intrinsic shape** — the relative positions of landmarks to each other. This is invariant to the per-view normalization warp.

### Why Smooth L1?

`SmoothL1` (Huber loss) is less sensitive to outlier landmark pairs than raw L1, providing more stable gradients during early training when predictions may be noisy.

---

## Combined Multi-View Loss

**Source**: `RayNet/multiview_loss.py:multiview_consistency_loss`

```python
L_multiview = lam_gaze_consist * gaze_ray_consistency_loss(pred_gaze, R_norm)
            + lam_shape * landmark_shape_consistency_loss(pred_coords)
```

### Inputs Required

| Input | Shape | Source |
|-------|-------|--------|
| `pred_gaze` | (B, 3) | Model output (`gaze_vector`) |
| `pred_coords` | (B, N, 2) | Model output (`landmark_coords`) |
| `R_norm` | (B, 3, 3) | Dataset (`R_norm` field) |

No camera intrinsics (K), extrinsics (R_cam, T_cam), inverse warps (M_norm_inv), or 3D coordinates (eyeball_center_3d) are needed.

---

## Weight Schedule

| Phase | Epochs | lam_gaze_consist | lam_shape | Dataloader |
|-------|--------|-------------------|-----------|------------|
| 1 | 1-5 | 0.0 | 0.0 | Standard (random, independent views) |
| 2 | 6-15 | 0.1 | 0.05 | Multi-view (9 cameras per group) |
| 3 | 16-30 | 0.2 | 0.1 | Multi-view (9 cameras per group) |

---

## Implementation Details

### AMP Compatibility

All multi-view loss operations are **float16-safe**:
- L1 loss on unit vectors: bounded inputs, bounded gradients
- R_norm matrix-vector multiply: orthogonal matrix, unit vector output
- No matrix inversions, no SVD, no large-scale coordinates

No `autocast(enabled=False)` wrapper is needed (unlike the previous pixel-space approach).

### Batch Requirements

Multi-view losses require `batch_size % 9 == 0`. If the batch size is not divisible by 9 (e.g., last batch), the multi-view loss returns 0 gracefully.

### `mv_groups` Parameter

Controls the number of multi-view groups per batch. Actual batch size = `mv_groups * 9`.

| Profile | mv_groups | Batch size | Batches/epoch (828K samples) |
|---------|-----------|------------|------------------------------|
| default | 2 | 18 | 46,000 |
| t4 | 4 | 36 | 23,000 |
| l4 | 8 | 72 | 11,500 |
| a100 | 16 | 144 | 5,750 |
| h100 | 32 | 288 | 2,875 |

Can be overridden with `--mv_groups N` (e.g., `--mv_groups 64` for batch size 576 on A100).

---

## Verification Protocol

### Normalization Roundtrip

Run `sanity_check_roundtrip(dataset, n_samples=50)` to verify:
1. Normalize 2D landmarks with M
2. Denormalize with M_inv
3. Compare with originals

**Must be < 2px error**.

### Expected Training Behavior

| Metric | Phase 1 | Phase 2 (early) | Phase 2 (late) |
|--------|---------|-----------------|----------------|
| gaze_mv | 0.0 | ~0.19 | ~0.15 |
| shape | 0.0 | ~0.05 | ~0.04 |
| angular_deg | ~42° (random) | ~20° | ~10-12° |

Gaze ray consistency loss starts high because each view independently predicts somewhat random gaze directions. As the gaze head learns, cross-view predictions converge in world frame.
