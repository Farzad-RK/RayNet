# Geometry and Kappa

Geometric post-processing for converting optical axis predictions to visual axis, computing metric pupil diameter, and projecting gaze to screen coordinates.

## Kappa Angle

The **kappa angle** is the angular offset between the optical axis (line from eyeball center through pupil center) and the visual axis (line from fovea through nodal point to the fixation target). RayNet predicts the **optical axis** and converts to visual axis using kappa.

### Anatomy

```
                          fixation point
                               *
                              /
                  visual axis/
                            /
        +---------+--------/---+
        |         |       / o  |  <- fovea (off-center)
        | eyeball | pupil/     |
        | center  |     /      |
        +---------+----/-------+
                      /
         optical axis/
                    /
                   * pupil center
```

The visual axis passes through the fovea (displaced ~5 deg nasally from the optical axis).

### Population Mean Kappa

| Angle | Value | Direction |
|-------|-------|-----------|
| Yaw (horizontal) | ~4.0 deg | Nasal |
| Pitch (vertical) | ~1.0 deg | Superior |
| Roll (torsional) | **0.0 deg** | Always zero |

**Source**: `RayNet/kappa.py`

### Hard-Zero Roll

Roll is **always zeroed** in `build_R_kappa()` regardless of the dataset value. Torsional rotation of the eyeball is negligible and can introduce artificial distortion in geometric computations.

```python
def build_R_kappa(kappa_angles):
    yaw = float(kappa_angles[0])
    pitch = float(kappa_angles[1])
    # Roll is IGNORED — index 2 not used
    R_yaw = rotation_matrix_y(yaw)
    R_pitch = rotation_matrix_x(pitch)
    return R_yaw @ R_pitch   # (3, 3)
```

### Optical -> Visual Axis Conversion

**At inference time**, convert the predicted optical axis to visual axis:

```python
# Zero-calibration (population mean)
R_kappa = build_R_kappa([np.deg2rad(4.0), np.deg2rad(1.0), 0.0])
visual_axis = R_kappa @ optical_axis

# Per-subject calibration (if kappa is known)
R_kappa = build_R_kappa(subject_kappa_angles)
visual_axis = R_kappa @ optical_axis
```

### Ground-Truth Optical Axis

The training target is computed from 3D geometry:

```python
def ground_truth_optical_axis(eyeball_center, pupil_center):
    direction = pupil_center - eyeball_center
    return direction / np.linalg.norm(direction)
```

This is the anatomical optical axis in camera coordinate space. It is then transformed to normalized space via `R_norm` for supervision.

---

## Metric Pupil Diameter

**Source**: `RayNet/geometry.py:metric_pupil_diameter`

Estimates the physical pupil diameter in millimeters using the iris as a geometric reference.

### Algorithm

1. **Fit ellipse to iris** (Fitzgibbon's algebraic method):
   ```
   iris_pts_2d: (10, 2) iris landmarks in pixel space
   -> ellipse parameters (center, axes, angle)
   -> apparent_iris_radius = sqrt(semi_major * semi_minor)
   ```

2. **Fit ellipse to pupil**:
   ```
   pupil_pts_2d: (4, 2) pupil landmarks
   -> apparent_pupil_radius = sqrt(semi_major * semi_minor)
   ```

3. **Estimate depth from iris size**:
   ```
   Z_mm = focal_length * iris_radius_mm / apparent_iris_radius_px
   ```
   Where `iris_radius_mm = 5.9 mm` (anatomical constant).

4. **Scale pupil to metric**:
   ```
   pupil_diameter_mm = 2 * apparent_pupil_radius * Z_mm / focal_length
   ```

### Usage

```python
from RayNet.geometry import metric_pupil_diameter

Z_mm, pupil_diam_mm = metric_pupil_diameter(
    iris_pts_2d,     # (10, 2) or (N, 2)
    pupil_pts_2d,    # (4, 2) or (M, 2)
    K,               # (3, 3) camera intrinsics
    iris_radius_mm=5.9,
)
```

---

## Gaze-to-Screen Projection

**Source**: `RayNet/geometry.py:gaze_to_screen_point`

Computes where the gaze ray intersects a known screen plane.

### Algorithm

Ray-plane intersection:

```
t = (screen_point - gaze_origin) . screen_normal / (gaze_direction . screen_normal)
hit_3d = gaze_origin + t * gaze_direction
screen_uv = project hit_3d onto screen axes (2D coordinates on screen)
```

### Usage

```python
from RayNet.geometry import gaze_to_screen_point

screen_uv, hit_3d = gaze_to_screen_point(
    gaze_origin,      # (3,) eye center in world/camera coords
    gaze_direction,    # (3,) unit gaze vector
    screen_normal,     # (3,) outward normal of the screen
    screen_point,      # (3,) any point on the screen plane
    screen_axes,       # (2, 3) horizontal and vertical axes of the screen
)
```

---

## Ellipse Fitting

**Source**: `RayNet/geometry.py:fit_ellipse_algebraic`

Uses Fitzgibbon's direct algebraic fitting method:
1. Build the design matrix from 2D point coordinates
2. Solve the constrained eigenvalue problem (`4ac - b^2 = 1`)
3. Extract ellipse parameters (center, semi-axes, rotation angle)

This is used internally by `metric_pupil_diameter` and can also be used for iris ellipse analysis at inference time.

---

## Coordinate Systems

| System | Abbreviation | Convention |
|--------|-------------|------------|
| World Coordinate System | WCS | Defined by the GazeGene capture setup |
| Camera Coordinate System | CCS | Origin at camera, Z forward, Y down, X right |
| Head Coordinate System | HCS | Defined by head pose (R_head, T_head) |
| Normalized Space | NCS | After Zhang 2018 normalization |

### Transformations

```
WCS -> CCS:     P_cam = R_cam @ P_world + T_cam
CCS -> NCS:     gaze_norm = R_norm @ gaze_cam
NCS -> CCS:     gaze_cam = R_norm^T @ gaze_norm
Optical -> Visual:  visual = R_kappa @ optical
```

All 3D geometric computations (triangulation, reprojection) should be done in **original camera space (CCS)**, not in normalized space, to avoid z-axis distortion from the perspective warp.
