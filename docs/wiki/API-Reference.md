# API Reference

Function signatures for all public modules in RayNet v5.

---

## `RayNet/raynet_v5.py` — Model

```python
class RayNetV5(nn.Module):
    def forward(
        self, x,
        n_views=1,
        R_cam=None, T_cam=None,
        face_bbox=None,
        use_landmark_bridge=True,      # no-op in Quad-M1 (back-compat)
        use_pose_bridge=True,          # False → zero pose stream into GazeFusionBlock
    ) -> dict
        # x: (B, 3, 224, 224)  (B = n_views * mv_groups)
        # R_cam: (B, 3, 3)     camera extrinsic rotation, or None
        # T_cam: (B, 3)        camera extrinsic translation, or None
        # face_bbox: (B, 3)    (x_p, y_p, L_x) from Intrinsic Delta
        # Internally rescales predicted landmark_coords from 56-space to 224-space,
        # detaches them, and passes them to the gaze branch as the eye-crop anchor.
        # Returns: {
        #   'landmark_coords':   (B, 14, 2),     # 56x56 space
        #   'landmark_heatmaps': (B, 14, 56, 56),
        #   'eyeball_center':    (B, 3),         # CCS, cm
        #   'pupil_center':      (B, 3),         # CCS, cm
        #   'gaze_vector':       (B, 3),         # unit optical axis
        #   'gaze_angles':       (B, 2),         # pitch, yaw
        #   'pred_pose_6d':      (B, 6),
        #   'pred_pose_t':       (B, 3),
        # }

def create_raynet_v5(
    backbone_weight_path=None,
    cross_view_cfg=None,
    n_landmarks=14,
) -> RayNetV5
    # Instantiates the Quad-M1 model. Creates FOUR RepNeXt-M1 instances:
    #   shared    → stem + s0 + s1 feed landmark & pose
    #   landmark  → s2 + s3 for the landmark branch encoder
    #   pose      → s2 + s3 for the pose branch encoder
    #   eye       → full M1 (stem + s0..s3) dedicated to the gaze branch,
    #               operating on 112x112 landmark-guided eye crops
    # All four start from the same pretrained backbone_weight_path.
```

## `RayNet/eye_crop.py` — Differentiable Eye Crop

```python
class EyeCropModule(nn.Module):
    def __init__(self, out_size=112, pad_frac=0.25, min_half_size=24.0)
    def forward(self, image, landmarks_px) -> Tensor
        # image:         (B, 3, H, W) float
        # landmarks_px:  (B, N, 2) in pixel space of `image`
        # Returns:       (B, 3, out_size, out_size)
        # Implementation: square bbox around landmark centroid with pad_frac
        # expansion, clamped to min_half_size. Sampled via F.affine_grid +
        # F.grid_sample (align_corners=True, padding_mode='zeros').
```

## `RayNet/coordatt.py` — Attention

```python
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels: int, reduction=32)
    def forward(self, x: Tensor) -> Tensor
```

---

## `RayNet/losses.py` — Loss Functions

```python
def gaussian_heatmaps(coords, H, W, sigma=2.0) -> Tensor
def landmark_loss(pred_hm, pred_coords, gt_coords, feat_H, feat_W, sigma=2.0) -> Tensor
def gaze_loss(pred_gaze, gt_gaze) -> Tensor         # L1 on unit vectors
def angular_error(pred_gaze, gt_gaze) -> Tensor     # metrics only

def eyeball_center_loss(pred_eyeball, gt_eyeball) -> Tensor   # L1 on 3D centers (cm)
def pupil_center_loss(pred_pupil, gt_pupil) -> Tensor         # L1 on 3D centers (cm)
def geometric_angular_loss(pred_eyeball, pred_pupil, gt_optical_axis) -> Tensor  # atan2-stable

def rotation_6d_to_matrix(r6d) -> Tensor            # (B, 6) -> (B, 3, 3)
def matrix_to_rotation_6d(R) -> Tensor              # (B, 3, 3) -> (B, 6)
def geodesic_loss(pred_R, gt_R) -> Tensor
def pose_prediction_loss(pred_6d, gt_head_R) -> Tensor
def translation_loss(pred_t, gt_t, eps=1e-6) -> Tensor

def ray_target_loss(pred_gaze, eyeball_center, gaze_target, gaze_depth) -> Tensor

def total_loss(
    pred_hm, pred_coords, pred_gaze,
    gt_coords, gt_gaze,
    feat_H, feat_W,
    lam_lm=1.0, lam_gaze=0.5, sigma=2.0,
    lam_eyeball=0.0, pred_eyeball=None, gt_eyeball=None,
    lam_pupil=0.0,   pred_pupil=None,   gt_pupil=None,
    lam_geom_angular=0.0,
    lam_ray=0.0, eyeball_center=None, gaze_target=None, gaze_depth=None,
    lam_pose=0.0,  pred_pose_6d=None, gt_head_R=None,
    lam_trans=0.0, pred_pose_t=None,  gt_head_t=None,
) -> tuple[Tensor, dict]
    # Returns: (total_loss, components_dict)
    # Components include: landmark_loss, angular_loss, angular_loss_deg,
    #                     eyeball_loss, pupil_loss, geom_angular_loss,
    #                     ray_target_loss, pose_loss, pose_loss_deg,
    #                     translation_loss, total_loss
```

## `RayNet/multiview_loss.py` — Multi-View Losses

```python
def reshape_multiview(tensor, n_views=9) -> Tensor
def gaze_ray_consistency_loss(pred_gaze, R_cam, n_pairs=3) -> Tensor
def landmark_shape_consistency_loss(pred_coords, n_pairs=3) -> Tensor

def multiview_consistency_loss(
    pred_gaze, pred_coords, R_cam,
    lam_gaze_consist=1.0, lam_shape=0.5, n_views=9,
) -> tuple[Tensor, dict]
```

---

## `RayNet/kappa.py` — Kappa Angles

```python
def build_R_kappa(kappa_angles) -> ndarray
def ground_truth_optical_axis(eyeball_center, pupil_center) -> ndarray
def optical_to_visual(optical_axis, R_kappa) -> ndarray
```

## `RayNet/geometry.py` — Geometric Post-Processing

```python
def fit_ellipse_algebraic(points_2d) -> tuple
def metric_pupil_diameter(iris_pts_2d, pupil_pts_2d, K,
                          iris_radius_mm=5.9) -> tuple[float, float]
def gaze_to_screen_point(gaze_origin, gaze_direction,
                         screen_normal, screen_point, screen_axes
                        ) -> tuple[ndarray, ndarray]
```

---

## `RayNet/dataset.py` — Data Loading

```python
class GazeGeneDataset(Dataset):
    def __init__(self, base_dir, subject_ids=None, camera_ids=None,
                 samples_per_subject=None, eye='L',
                 img_size=224, augment=False, seed=42)
    def __getitem__(self, idx) -> dict         # includes `face_bbox_gt`, `K`, `intrinsic_original`
    def __len__(self) -> int

class MultiViewBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=1, shuffle=True, ensure_multiview=True)

def gazegene_collate_fn(batch: list[dict]) -> dict

def create_dataloaders(base_dir, train_subjects, val_subjects,
                        batch_size=4, num_workers=4,
                        samples_per_subject=None, eye='L',
                        ensure_multiview=False
                       ) -> tuple[DataLoader, DataLoader]
```

## `RayNet/streaming/` — MosaicML Streaming + MinIO

### `RayNet/streaming/dataset.py`

```python
class StreamingGazeGeneDataset(StreamingDataset):
    def __init__(self, transform=None, samples_per_subject=None, **kwargs)
    def __getitem__(self, idx) -> dict

def create_streaming_dataloaders(
    remote_train, remote_val,
    local_cache='./mds_cache',
    batch_size=512, num_workers=4,
    transform=None, val_transform=None,
    shuffle_train=True, pin_memory=True,
    prefetch_factor=2, persistent_workers=False,
    samples_per_subject=None,
    **streaming_kwargs,
) -> tuple[DataLoader, DataLoader]

def create_multiview_streaming_dataloaders(
    remote_train, remote_val,
    local_cache='./mds_cache',
    mv_groups=2, num_workers=4,
    transform=None,
    samples_per_subject=None,
    **streaming_kwargs,
) -> tuple[DataLoader, DataLoader]
    # batch_size = mv_groups * 9  (9-camera groups preserved)
```

### `RayNet/streaming/convert_to_mds.py`

```python
MDS_COLUMNS: dict                     # includes 'face_bbox_gt' and 'intrinsic_original'

def convert_to_mds(dataset, output_dir, split='train',
                   multiview_grouped=True) -> int

def convert_to_mds_chunked(data_dir, output_dir, subject_ids,
                           split='train', multiview_grouped=True,
                           samples_per_subject=None, eye='L',
                           chunk_size=3) -> int
```

---

## `RayNet/hardware_profiles.py` — Hardware Config + Accelerate

```python
HARDWARE_PROFILES: dict          # 'default', 't4', 'l4', 'a10g', 'v100', 'a100',
                                  # 'h100', 'kaggle_t4x2', 'multi_node_t4'
AMP_DTYPE_MAP: dict              # 'float16' | 'bfloat16' | 'float32' → torch dtype

def apply_hardware_profile(args) -> dict
def setup_hardware(hw: dict, device) -> None
def build_accelerator() -> Accelerator
    # Constructs Accelerator(mixed_precision='no') with
    # DistributedDataParallelKwargs(find_unused_parameters=True).
    # Required for Stage 2 P1/P2 where the frozen face path has params
    # with requires_grad=False.
```

## `RayNet/train.py` — Training

```python
STAGE_CONFIGS: dict              # stages 1, 2, 3 with per-phase configs (incl. 'freeze_face')

def get_phase(epoch: int) -> int
def get_phase_config(epoch: int) -> dict

def _unwrap_raynet(model) -> RayNetV5                      # strips DDP + torch.compile
def set_face_frozen(model, frozen: bool) -> None
    # Freezes / unfreezes shared_stem + landmark_branch + pose_branch.
    # When frozen: requires_grad_(False) + .eval() on each submodule, so BN
    # running stats also stop drifting. Re-applied inside train_one_epoch
    # after model.train() so the freeze survives the per-epoch toggle.

def _filter_compatible_state(src_sd, target_sd) -> tuple[dict, list[str]]
    # Drops shape-mismatched tensors during cross-stage forks.
def _optimizer_state_compatible(saved_state, new_optimizer) -> bool
    # Per-group param count check; guards optimizer.load_state_dict.

def train_one_epoch(model, train_loader, optimizer, device, epoch, cfg,
                    scaler=None, grad_accum_steps=1, amp_enabled=False,
                    amp_dtype=torch.float16, batch_csv_writer=None,
                    n_views=1) -> dict

def validate(model, val_loader, device, epoch, cfg, amp_enabled=False,
             amp_dtype=torch.float16, n_views=1) -> dict

def train(args) -> None
def parse_args() -> Namespace
```
