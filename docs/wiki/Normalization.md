# Normalization

RayNet v5 uses **Easy-Norm** (MAGE, Bao et al. CVPR 2025) for image normalization. Unlike the previous Zhang et al. 2018 approach, Easy-Norm requires only the face center (from bounding box or landmarks) — no head pose needed.

This is critical for the GazeGene dataset, where random translation and scaling augmentation during face cropping breaks camera intrinsics, making traditional Zhang normalization invalid.

**Source**: `RayNet/normalization.py`

## Why Easy-Norm?

| Property | Zhang Normalization | Easy-Norm |
|----------|-------------------|-----------|
| Requires head pose | Yes (R_head, t_eye) | **No** |
| Requires accurate K | Yes | Tolerant of augmented K |
| Works with GazeGene crops | No (intrinsics broken) | **Yes** |
| Rotation | Full 3D warp | Z-axis toward face center |

## Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `f_norm` | 960 px | Canonical focal length |
| `d_norm` | 600 mm | Canonical distance from camera to face |
| `img_size` | 224 px | Output image dimensions |

## Split Pipeline

RayNet v5 uses task-specific normalization:

| Task | Normalization | Reason |
|------|--------------|--------|
| **Gaze regression** | Full (rotation + scaling) | Remove head pose confound |
| **Iris/pupil landmarks** | Partial (scaling only, no roll) | Preserve iris appearance, avoid OCR artifacts |
| **3D geometry (triangulation)** | None | Stay in original CCS, avoid z-axis distortion |

## Easy-Norm Pipeline

### Input

- `image`: Face crop (H, W, 3) BGR
- `K`: Camera intrinsic matrix (3, 3) — can be approximate
- `t_face`: Face center position (3,) in camera coordinates

### 1. Face Distance

```
z_actual = ||t_face||
```

### 2. Virtual Camera Rotation

Build a rotation matrix so the virtual camera's z-axis points at the face center:

```
z_axis = t_face / ||t_face||
x_axis = cross([0, 1, 0], z_axis) / ||cross||
y_axis = cross(z_axis, x_axis)
R_norm = stack([x_axis, y_axis, z_axis])   # (3, 3)
```

### 3. Depth Scaling

```
S = diag([1, 1, d_norm / z_actual])
```

### 4. Canonical Intrinsics

```
K_norm = [f_norm,  0,       112]     # 112 = 224 / 2
         [0,       f_norm,  112]
         [0,       0,       1  ]
```

### 5. Homography

```
M = K_norm @ S @ R_norm @ K_inv
```

Apply to the image:

```python
img_norm = cv2.warpPerspective(image, M, (224, 224))
```

### Output

- `img_norm`: (224, 224, 3) normalized face image
- `R_norm`: (3, 3) normalization rotation (for gaze vector transform)
- `M_inv`: (3, 3) inverse homography (for denormalization)

## Gaze Direction Transformation

### Camera -> Normalized (training)

```python
gaze_norm = R_norm @ gaze_cam
gaze_norm = gaze_norm / ||gaze_norm||
```

### Normalized -> Camera (inference)

```python
gaze_cam = R_norm^T @ gaze_norm
gaze_cam = gaze_cam / ||gaze_cam||
```

## Partial Normalization (Landmarks)

For landmark detection, only depth scaling is applied (no rotation):

```python
S_partial = diag([d_norm / z_actual, d_norm / z_actual, 1])
M_partial = K_norm @ S_partial @ K_inv
img_scaled = cv2.warpPerspective(image, M_partial, (224, 224))
```

This preserves iris ellipse appearance and avoids artificial Ocular Counter-Rolling (OCR) artifacts.

## Landmark Warping

2D landmarks in the original image are warped to normalized space:

```python
warped = warp_points_2d(landmarks, M)
```

**Source**: `normalization.py:warp_points_2d`

## Coordinate Spaces Summary

```
Original camera pixels
    |
    |  M (Easy-Norm homography)
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
    |  M_inv (inverse homography)
    v
Original camera pixels              <-- multi-view geometry lives here
    |
    |  K_inv * Z (unproject)
    v
Camera 3D coordinates (mm)          <-- triangulation lives here
```
