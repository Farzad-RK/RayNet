# Normalization

RayNet uses **per-frame image normalization** (Zhang et al. 2018) to remove depth ambiguity from the learning task. Every eye crop is warped to a virtual camera at a canonical distance and focal length.

**Source**: `RayNet/normalization.py`

## Why Normalize?

Without normalization, the same eye at different distances produces images of different scales. The network would need to implicitly learn depth estimation before it can localize landmarks or estimate gaze. Normalization removes this confound by placing every eye at the same virtual distance.

## Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `d_norm` | 600 mm | Canonical distance from camera to eye |
| `f_norm` | 960 px | Canonical focal length |
| `img_size` | 224 px | Output image dimensions |

## Step-by-Step Pipeline

### Input

- `image`: Original camera frame (H, W, 3) BGR
- `K`: Camera intrinsic matrix (3, 3)
- `R_head`: Head rotation matrix (3, 3) in camera coords
- `t_eye`: Eye center position (3,) in camera coords (mm)

### 1. Distance-Based Scaling

Compute the actual distance and build a scaling matrix:

```
z_actual = ||t_eye||
S = diag([1, 1, d_norm / z_actual])
```

This scales the Z-axis to move the eye from its actual distance to the canonical distance.

### 2. Virtual Camera Rotation

Build a rotation matrix so the virtual camera directly faces the eye center:

```
z_axis = t_eye / z_actual              # points toward eye
x_axis = cross([0, 1, 0], z_axis)     # horizontal
x_axis = x_axis / ||x_axis||
y_axis = cross(z_axis, x_axis)        # vertical
R_norm = stack([x_axis, y_axis, z_axis])   # (3, 3)
```

### 3. Canonical Intrinsics

```
K_norm = [f_norm,  0,       112]     # 112 = 224 / 2
         [0,       f_norm,  112]
         [0,       0,       1  ]
```

### 4. Perspective Warp

Combine into a single homography:

```
M = K_norm @ R_norm @ S @ K_inv
```

Apply to the image:

```python
img_norm = cv2.warpPerspective(image, M, (224, 224))
```

### Output

- `img_norm`: (224, 224, 3) normalized eye crop
- `R_norm`: (3, 3) the normalization rotation matrix (needed for denormalization)

## Gaze Direction Transformation

Gaze directions must be transformed between camera space and normalized space:

### Camera -> Normalized (training)

```python
gaze_norm = R_norm @ gaze_cam
gaze_norm = gaze_norm / ||gaze_norm||
```

The network is trained to predict gaze in **normalized space**.

### Normalized -> Camera (inference)

```python
gaze_cam = R_norm^T @ gaze_norm
gaze_cam = gaze_cam / ||gaze_cam||
```

## Landmark Warping

2D landmarks in the original image are warped to normalized space using the same homography M:

```python
# Homogeneous coordinates
pts_h = [u, v, 1]^T
warped_h = M @ pts_h
warped = warped_h[:2] / warped_h[2]    # dehomogenize
```

**Source**: `normalization.py:warp_points_2d`

## Inverse Warp (Denormalization)

To convert predictions back to original camera pixel space (needed for multi-view consistency):

```python
M_inv = inv(M)
# Apply M_inv to normalized pixel coordinates
original_px = dehomogenize(M_inv @ [u_norm, v_norm, 1]^T)
```

The dataset stores `M_norm_inv` in each sample for this purpose.

## Sanity Check

The normalization pipeline includes a round-trip sanity check (`multiview_loss.py:sanity_check_roundtrip`):

1. Take original 2D landmarks
2. Warp to normalized space with M
3. Warp back with M_inv
4. Compare with original

**Acceptance criterion**: Maximum pixel error < 2.0 px across all test samples.

This check runs automatically at training startup.

## Coordinate Spaces Summary

```
Original camera pixels
    |
    |  M (perspective warp)
    v
Normalized image pixels (224x224)
    |
    |  * 0.25 (= 56/224)
    v
Feature map space (56x56)           <-- model predictions live here
    |
    |  * 4.0 (= 224/56)
    v
Normalized image pixels (224x224)
    |
    |  M_inv (inverse warp)
    v
Original camera pixels              <-- multi-view geometry lives here
    |
    |  K_inv * Z (unproject)
    v
Camera 3D coordinates (mm)          <-- triangulation lives here
```
