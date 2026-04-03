# Dataset

RayNet v2 is built for the **GazeGene** multi-camera gaze dataset. This page describes the data format, directory layout, pickle file schemas, and how to load data for training.

## GazeGene Overview

- **Cameras**: 9 synchronized cameras per subject (IDs 0-8)
- **Subjects**: 56+ subjects with calibrated eye parameters
- **Per frame**: Eye crops from each camera with full 3D annotations
- **Annotations**: 3D eye landmarks (100 iris mesh points), eyeball/pupil centers, head pose, camera calibration

## Directory Layout

```
GazeGene_FaceCrops/
├── subject1/
│   ├── subject_label.pkl           # Subject-level attributes
│   ├── camera_info.pkl             # Camera intrinsics/extrinsics (9 cameras)
│   ├── labels/
│   │   ├── complex_label_camera0.pkl   # 3D/2D eye annotations
│   │   ├── complex_label_camera1.pkl
│   │   ├── ...
│   │   ├── complex_label_camera8.pkl
│   │   ├── gaze_label_camera0.pkl      # Head pose annotations
│   │   ├── gaze_label_camera1.pkl
│   │   ├── ...
│   │   └── gaze_label_camera8.pkl
│   └── images/
│       ├── cam0_frame0.jpg
│       ├── cam0_frame1.jpg
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
    'eyecenter_L': ndarray(3,),    # Left eye center in world coords (mm)
    'eyecenter_R': ndarray(3,),    # Right eye center in world coords (mm)
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
    'eyeball_center_3D': ndarray(N, 2, 3),   # [Left, Right] eye centers in CCS (mm)
    'pupil_center_3D': ndarray(N, 2, 3),     # [L, R] pupil centers in CCS (mm)
    'iris_mesh_3D': ndarray(N, 2, 100, 3),   # [L, R] 100-point iris contour in CCS
    'iris_mesh_2D': ndarray(N, 2, 100, 2),   # [L, R] 100-point iris in pixel coords
    'pupil_center_2D': ndarray(N, 2, 2),     # [L, R] pupil centers in pixels
    'intrinsic_matrix_cropped': ndarray(N, 3, 3),  # K matrix (may vary per frame)
}
```

**CCS** = Camera Coordinate System (origin at camera, Z forward, Y down, X right).

### `gaze_label_camera{N}.pkl`

Head pose for one camera across all frames:

```python
{
    'head_R_mat': ndarray(N, 3, 3),   # Head rotation in camera frame
    'head_T_vec': ndarray(N, 3),      # Head translation in camera frame
}
```

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
    d_norm: float = 600,              # Normalization distance (mm)
    f_norm: float = 960,              # Normalization focal length (px)
    img_size: int = 224,              # Output image size
    augment: bool = False,            # Enable augmentation
    seed: int = 42,                   # Random seed
)
```

### Output per Sample

`__getitem__` returns a dictionary after full pre-processing (normalization, landmark warping):

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `image` | `(3, 224, 224)` | float32 | Normalized RGB crop, values [0, 1] |
| `landmark_coords` | `(14, 2)` | float32 | Landmarks in feature space (56x56) |
| `landmark_coords_px` | `(14, 2)` | float32 | Landmarks in pixel space (224x224) |
| `optical_axis` | `(3,)` | float32 | GT optical axis (normalized space) |
| `R_norm` | `(3, 3)` | float32 | Normalization rotation matrix |
| `R_kappa` | `(3, 3)` | float32 | Kappa rotation matrix |
| `K` | `(3, 3)` | float32 | Camera intrinsic matrix |
| `R_cam` | `(3, 3)` | float32 | Camera extrinsic rotation |
| `T_cam` | `(3,)` | float32 | Camera extrinsic translation |
| `M_norm_inv` | `(3, 3)` | float32 | Inverse normalization warp |
| `eyeball_center_3d` | `(3,)` | float32 | Eye center in camera coords (mm) |
| `subject` | scalar | int | Subject ID |
| `cam_id` | scalar | int | Camera ID (0-8) |
| `frame_idx` | scalar | int | Frame index |

### Processing Pipeline in `__getitem__`

1. Load BGR image from disk
2. Look up camera parameters (K, R_cam, T_cam) from `camera_info.pkl`
3. Compute eye center in camera coordinates: `t_eye = R_cam @ eyecenter_world + T_cam`
4. **Per-frame normalization** (Zhang et al. 2018) -> normalized 224x224 crop + R_norm
5. Compute warp matrix M for landmark transformation
6. Compute GT optical axis: `(pupil_3d - eyeball_3d) / ||...||`
7. Transform optical axis to normalized space: `R_norm @ optical_axis`
8. Subsample iris: 100 -> 10 points at indices [0, 10, 20, ..., 90]
9. Select 4 pupil boundary points (closest iris points to pupil center)
10. Concatenate: 10 iris + 4 pupil = **14 landmarks**
11. Warp landmarks to normalized image space using M
12. Scale to feature-map space: `landmarks * (56 / 224) = landmarks * 0.25`
13. Convert image to RGB float32 tensor [0, 1]
14. Apply augmentation if enabled (brightness jitter, Gaussian noise)

### Data Augmentation

When `augment=True`:
- Random brightness: scale by [0.8, 1.2] with 50% probability
- Random Gaussian noise: sigma=0.02 with 50% probability

---

## Batch Samplers

### Standard Batching

Random samples from any subject/camera/frame. Used in Phase 1.

```python
create_dataloaders(..., ensure_multiview=False, batch_size=512)
```

### Multi-View Batching

Groups all 9 camera views of the same (subject, frame) into a batch. Used in Phases 2-3.

```python
create_dataloaders(..., ensure_multiview=True, batch_size=2)
# Actual batch size = 2 groups * 9 cameras = 18 samples
```

**MultiViewBatchSampler** (`dataset.py`):
- Uses `index_by_key` dict mapping `(subject_id, frame_idx)` -> list of dataset indices
- When `ensure_multiview=True`, only includes groups with all 9 cameras present
- Yields batches of `batch_size * 9` indices

### Collate Function

`gazegene_collate_fn` stacks tensor fields and collects scalars into lists:

```python
# Tensor fields -> torch.stack
['image', 'landmark_coords', 'landmark_coords_px', 'optical_axis',
 'R_norm', 'R_kappa', 'K', 'R_cam', 'T_cam', 'M_norm_inv', 'eyeball_center_3d']

# Scalar fields -> list
['subject', 'cam_id', 'frame_idx']
```

---

## Data Loading Example

```python
from RayNet.dataset import GazeGeneDataset, create_dataloaders

# Quick check
ds = GazeGeneDataset(
    base_dir='/path/to/GazeGene_FaceCrops',
    subject_ids=[1, 2, 3],
    samples_per_subject=10,
    eye='L',
)
sample = ds[0]
print(sample['image'].shape)           # (3, 224, 224)
print(sample['landmark_coords'].shape) # (14, 2)
print(sample['optical_axis'].shape)    # (3,)

# Full training loaders
train_loader, val_loader = create_dataloaders(
    base_dir='/path/to/GazeGene_FaceCrops',
    train_subjects=list(range(1, 47)),
    val_subjects=list(range(47, 57)),
    batch_size=512,
    num_workers=4,
    eye='L',
    ensure_multiview=False,
)
```
