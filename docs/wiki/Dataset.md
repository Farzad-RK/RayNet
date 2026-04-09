# Dataset

RayNet v4.1 is built for the **GazeGene** multi-camera gaze dataset. This page describes the data format, directory layout, pickle file schemas, and how data flows into the model.

## GazeGene Overview

- **Cameras**: 9 synchronized cameras per subject (IDs 0-8)
- **Subjects**: 56+ subjects with calibrated eye parameters
- **Per frame**: Face crops from each camera with full 3D annotations
- **Annotations**: 3D eye landmarks (100 iris mesh points), eyeball/pupil centers, head pose (R + T), camera calibration, gaze labels
- **Units**: centimeters (all 3D coordinates)

## Directory Layout

```
GazeGene_FaceCrops/
├── subject1/
│   ├── subject_label.pkl           # Subject-level attributes
│   ├── camera_info.pkl             # Camera intrinsics/extrinsics (9 cameras)
│   ├── labels/
│   │   ├── complex_label_camera0.pkl   # 3D/2D eye annotations
│   │   ├── ...
│   │   ├── complex_label_camera8.pkl
│   │   ├── gaze_label_camera0.pkl      # Head pose + gaze annotations
│   │   ├── ...
│   │   └── gaze_label_camera8.pkl
│   └── images/
│       ├── cam0_frame0.jpg
│       └── ...
├── subject2/
├── ...
└── subject56/
```

## Pickle File Schemas

### `subject_label.pkl`

Subject-level attributes (constant across all frames):

```python
{
    'ethnicity': str,              # e.g. "Asian", "Caucasian"
    'gender': str,                 # "M" or "F"
    'eye_color_L': str,            # e.g. "Brown"
    'eye_color_R': str,
    'eyecenter_L': ndarray(3,),    # Left eye center in world coords (cm)
    'eyecenter_R': ndarray(3,),    # Right eye center in world coords (cm)
    'L_kappa': ndarray(3,),        # [yaw, pitch, roll] in radians
    'R_kappa': ndarray(3,),        # [yaw, pitch, roll] in radians
    'iris_radius': float,          # Iris radius in mm (~5.9)
}
```

**Note**: Kappa roll (index 2) is always zeroed during training. See [[Geometry and Kappa]].

### `camera_info.pkl`

Per-subject camera calibration (list of 9 camera dicts):

```python
[
    {
        'cam_id': 0,
        'intrinsic_matrix': ndarray(3, 3),   # Camera K matrix
        'R_mat': ndarray(3, 3),              # World-to-camera rotation
        'T_vec': ndarray(3,),                # World-to-camera translation
    },
    # ... cameras 1-8
]
```

### `complex_label_camera{N}.pkl`

3D/2D eye annotations for one camera across all frames:

```python
{
    'img_path': list[str],                    # Relative image paths, length N_frames
    'eyeball_center_3D': ndarray(N, 2, 3),   # [Left, Right] eye centers in CCS (cm)
    'pupil_center_3D': ndarray(N, 2, 3),     # [L, R] pupil centers in CCS (cm)
    'iris_mesh_3D': ndarray(N, 2, 100, 3),   # [L, R] 100-point iris contour in CCS
    'iris_mesh_2D': ndarray(N, 2, 100, 2),   # [L, R] 100-point iris in pixel coords
    'pupil_center_2D': ndarray(N, 2, 2),     # [L, R] pupil centers in pixels
    'intrinsic_matrix_cropped': ndarray(N, 3, 3),  # K matrix after face crop
}
```

### `gaze_label_camera{N}.pkl`

Head pose and gaze annotations for one camera across all frames:

```python
{
    'head_R_mat': ndarray(N, 3, 3),    # Head rotation in camera frame
    'head_T_vec': ndarray(N, 3),       # Head translation in camera frame (cm)
    'optic_axis_L': ndarray(N, 3),     # Left optical axis unit vector, CCS
    'optic_axis_R': ndarray(N, 3),     # Right optical axis unit vector, CCS
    'visual_axis_L': ndarray(N, 3),    # Left visual axis unit vector, CCS
    'visual_axis_R': ndarray(N, 3),    # Right visual axis unit vector, CCS
    'gaze_C': ndarray(N, 3),           # Unit head gaze direction, CCS
    'gaze_target': ndarray(N, 3),      # 3D gaze target position, CCS (cm)
    'gaze_depth': ndarray(N,),         # Vergence depth (scalar)
}
```

**CCS** = Camera Coordinate System (origin at camera, Z forward, Y down, X right).

---

## GazeGeneDataset Class

**Source**: `RayNet/dataset.py`

### Constructor

```python
GazeGeneDataset(
    base_dir: str,                    # Path to GazeGene_FaceCrops/
    subject_ids: list[int] = None,    # Which subjects (default: all)
    camera_ids: list[int] = None,     # Which cameras (default: 0-8, all 9)
    samples_per_subject: int = None,  # Max frames per subject (default: all)
    eye: str = 'L',                   # 'L' or 'R'
    img_size: int = 224,              # Output image size
    augment: bool = False,            # Enable augmentation
    seed: int = 42,                   # Random seed
)
```

### Output per Sample

`__getitem__` returns a dictionary:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `image` | `(3, 224, 224)` | uint8 | RGB face crop (normalized to float in train loop) |
| `landmark_coords` | `(14, 2)` | float32 | Landmarks in feature-map space (56x56) |
| `landmark_coords_px` | `(14, 2)` | float32 | Landmarks in pixel space (224x224) |
| `optical_axis` | `(3,)` | float32 | GT optical axis unit vector in CCS |
| `R_kappa` | `(3, 3)` | float32 | Kappa rotation matrix |
| `K` | `(3, 3)` | float32 | Camera intrinsic matrix (cropped) |
| `R_cam` | `(3, 3)` | float32 | Camera extrinsic rotation |
| `T_cam` | `(3,)` | float32 | Camera extrinsic translation |
| `head_R` | `(3, 3)` | float32 | Head pose rotation matrix |
| `head_t` | `(3,)` | float32 | Head translation vector (cm) |
| `eyeball_center_3d` | `(3,)` | float32 | Eye center in CCS (cm) |
| `gaze_target` | `(3,)` | float32 | 3D gaze target in CCS (cm) |
| `gaze_depth` | scalar | float32 | Vergence depth |
| `subject` | scalar | int | Subject ID |
| `cam_id` | scalar | int | Camera ID (0-8) |
| `frame_idx` | scalar | int | Frame index |

### Processing Pipeline in `__getitem__`

1. Load BGR image from disk, resize to 224x224 if needed
2. Look up camera parameters (K, R_cam, T_cam) from `camera_info.pkl`
3. Extract GT optical axis from `gaze_label` (pre-computed by GazeGene, CCS unit vector)
4. Subsample iris: 100 -> 10 points at indices [0, 10, 20, ..., 90]
5. Select 4 pupil boundary points (closest iris points to 2D pupil center)
6. Concatenate: 10 iris + 4 pupil = **14 landmarks**
7. Scale landmarks from image space to pixel space (224x224)
8. Scale to feature-map space: `landmarks_px / 4.0` (P2 stride=4, so 224/4=56)
9. Build R_kappa from subject attributes (roll zeroed)
10. Convert image to uint8 RGB tensor (train loop does `.float().div_(255.0)`)
11. Apply augmentation if enabled (color jitter + random translation)

### Data Augmentation

When `augment=True`:
- **Color jitter**: brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
- **Random translation**: ~5% shift (matching GazeGene paper Sec 4.1.3)

---

## MDS Shard Schema

**Source**: `RayNet/streaming/convert_to_mds.py`

MDS shards store all dataset fields in binary format for streaming:

| Column | MDS Type | Description |
|--------|----------|-------------|
| `image` | `bytes` | JPEG-encoded 224x224 face crop |
| `landmark_coords` | `ndarray` | (14, 2) feature-map space |
| `landmark_coords_px` | `ndarray` | (14, 2) pixel space |
| `optical_axis` | `ndarray` | (3,) unit vector CCS |
| `R_kappa` | `ndarray` | (3, 3) |
| `K` | `ndarray` | (3, 3) intrinsics |
| `R_cam` | `ndarray` | (3, 3) extrinsics rotation |
| `T_cam` | `ndarray` | (3,) extrinsics translation |
| `eyeball_center_3d` | `ndarray` | (3,) CCS |
| `head_R` | `ndarray` | (3, 3) head pose rotation |
| `head_t` | `ndarray` | (3,) head translation |
| `gaze_target` | `ndarray` | (3,) CCS |
| `gaze_depth` | `float32` | scalar |
| `subject` | `int` | subject ID |
| `cam_id` | `int` | camera ID |
| `frame_idx` | `int` | frame index |

When `multiview_grouped=True`, samples are written so 9 consecutive samples form one (subject, frame) group, enabling multi-view batch construction during streaming.

---

## Batch Samplers

### Standard Batching

Random samples from any subject/camera/frame:

```python
create_dataloaders(..., ensure_multiview=False, batch_size=512)
```

### Multi-View Batching

Groups all 9 camera views of the same (subject, frame) into a batch:

```python
create_dataloaders(..., ensure_multiview=True, batch_size=16)
# Actual batch size = 16 groups * 9 cameras = 144 samples
```

### Collate Function

`gazegene_collate_fn` stacks tensor fields and collects scalars into lists:

```python
# Tensor fields -> torch.stack
['image', 'landmark_coords', 'landmark_coords_px', 'optical_axis',
 'R_kappa', 'K', 'R_cam', 'T_cam', 'head_R', 'head_t',
 'eyeball_center_3d', 'gaze_target', 'gaze_depth']

# Scalar fields -> list
['subject', 'cam_id', 'frame_idx']
```
