# Multi-View Consistency

RayNet v2 exploits the 9 synchronized cameras in GazeGene to impose geometric constraints during training. This provides "free" supervision that resolves depth ambiguity and enforces anatomically consistent predictions across viewpoints.

**Source**: `RayNet/multiview_loss.py`

## Overview

Two complementary losses are used:

| Loss | Purpose | Weight | Active |
|------|---------|--------|--------|
| **Reprojection consistency** | Predictions from cam_i, reprojected to cam_j, should match cam_j's own predictions | 0.1 -> 0.2 | Phase 2+ |
| **Triangulation masking** | Triangulate eye from two cameras, project into a third (masked) camera, compare | 0.05 -> 0.1 | Phase 2+ |

Both losses operate in **original camera pixel space**, not normalized space.

## Batch Structure

Multi-view batches are structured as:

```
Standard batch: (B, ...)           e.g. B=512 random samples
Multi-view:     (G*9, ...)         e.g. G=2 groups, 18 total samples
```

The `reshape_multiview` utility converts `(G*9, ...)` -> `(G, 9, ...)` for cross-view operations.

## Denormalization Pipeline

Before any geometric operation, predictions must be converted from feature space to original camera pixels:

```
Feature coords (B, 14, 2) in [0, 56)
        |
    * 4.0 (stride)
        |
Normalized pixels (B, 14, 2) in [0, 224)
        |
    M_norm_inv (homography, per sample)
        |
Original camera pixels (B, 14, 2)
```

This is done by `denormalize_landmarks_to_original_px()` using batched homogeneous transforms.

---

## Reprojection Consistency Loss

### Algorithm

For each forward pass, sample `n_pairs=2` random camera pairs from the 9 available:

**For pair (i, j):**

1. **Denormalize** both cameras' predicted landmarks to original pixel space

2. **Unproject** camera i's 2D landmarks to 3D:
   ```
   ray = K_i_inv @ [u, v, 1]^T       # normalized ray direction
   P_3d = ray * Z_i                   # scale by depth
   ```
   Where `Z_i` = GT eyeball center Z-component in camera i coords (clamped >= 100mm)

3. **Transform** from camera i frame to camera j frame:
   ```
   R_rel = R_j @ R_i^T                # relative rotation
   t_rel = T_j - R_rel @ T_i          # relative translation
   P_j = R_rel @ P_i + t_rel
   ```

4. **Project** into camera j's image:
   ```
   [u', v', z'] = K_j @ P_j
   pixel_j = [u'/z', v'/z']
   ```

5. **L1 loss** between reprojected landmarks and camera j's own predictions

6. Repeat in reverse (j -> i) and average both directions

**Final loss**: average over all sampled pairs.

### Depth Strategy

**Option A (current)**: Use GT eyeball center Z component from the dataset. This is the depth of the eyeball center in each camera's coordinate system. All 14 landmarks are assumed to lie approximately at this depth (valid because the iris is roughly planar and close to the eyeball center).

**Option B (future)**: Use the model's geometric iris depth estimate from ellipse fitting (`geometry.py:metric_pupil_diameter`), making the loss fully self-supervised.

---

## Triangulation Masking Loss

### Algorithm

1. **Select 3 cameras** randomly: `mask_cam`, `tri_a`, `tri_b`

2. **Compute centroids** of the 14 predicted landmarks in original pixel space for all 3 cameras. The centroid serves as an eye center proxy.

3. **Build projection matrices** for the two triangulation cameras:
   ```
   P_a = K_a @ [R_a | T_a]    shape: (G, 3, 4)
   P_b = K_b @ [R_b | T_b]    shape: (G, 3, 4)
   ```

4. **Triangulate** via DLT (Direct Linear Transform):
   - Build 4x4 system from the cross-product constraints:
     ```
     A[0] = x_a * P_a[2] - P_a[0]
     A[1] = y_a * P_a[2] - P_a[1]
     A[2] = x_b * P_b[2] - P_b[0]
     A[3] = y_b * P_b[2] - P_b[1]
     ```
   - Solve `A @ X = 0` via SVD: solution is last column of V
   - Dehomogenize to get 3D world point
   - **Detach** the result (no gradients through SVD)

5. **Project** triangulated point into masked camera:
   ```
   P_mask_cam = R_mask @ P_world + T_mask
   pixel = K_mask @ P_mask_cam / z
   ```

6. **L1 loss** between projected point and masked camera's predicted centroid

### Why This Works

This loss teaches the network that its predictions must be geometrically consistent across views. If camera A and B both predict eye landmarks that triangulate to a 3D point, that point should reproject correctly into camera C. This is supervision that comes "for free" from the multi-camera geometry.

---

## Implementation Details

### Gradient Flow

```
Model predictions          <-- gradients flow here
    |
Denormalization (M_inv)    <-- fixed (no learnable params)
    |
Unproject (K_inv, Z)       <-- fixed camera params
    |
Transform (R_rel, t_rel)   <-- fixed camera params
    |
Project (K)                <-- fixed camera params
    |
L1 Loss                    <-- loss signal
```

For the triangulation loss, the triangulated 3D point is **detached**, so gradients only flow through the masked camera's prediction branch.

### Numerical Stability

- Depth Z clamped to minimum 100mm (prevents division by tiny numbers)
- Dehomogenization adds 1e-8 to denominator
- Projection Z clamped to minimum 1.0 (prevents negative/zero depth)

### Batch Requirements

Multi-view losses require `batch_size % 9 == 0`. If the batch size is not divisible by 9 (e.g., last batch), the multi-view loss returns 0 gracefully.

---

## Verification Protocol

### Normalization Roundtrip

Run `sanity_check_roundtrip(dataset, n_samples=50)` to verify:
1. Normalize 2D landmarks with M
2. Denormalize with M_inv
3. Compare with originals

**Must be < 2px error**.

### Synthetic Validation

To validate the reprojection pipeline independently:
1. Choose a known 3D point
2. Project into camera i and camera j using known K, R, T
3. Run through the reprojection loss pipeline
4. Loss should be approximately zero

### Camera Pair Baseline

If all cameras predict perfectly consistent landmarks, the reprojection loss should be near zero. In practice, early training will show high reprojection loss that decreases as predictions become geometrically consistent.
