# API Reference

Function signatures for all public modules in RayNet v4.1.

---

## `RayNet/raynet.py` -- Model

```python
class RayNet(nn.Module):
    def __init__(self, backbone, in_channels_list, panet_out_channels=256,
                 n_landmarks=14, cross_view_cfg=None,
                 pose_backbone=None, pose_backbone_channels=None)
    def forward(self, x, n_views=1, R_cam=None, T_cam=None, use_bridge=True) -> dict
        # x: (B, 3, 224, 224)
        # n_views: 1=single-view, 9=multi-view
        # R_cam: (B, 3, 3) camera extrinsic rotation, or None
        # T_cam: (B, 3) camera extrinsic translation, or None
        # use_bridge: enable LandmarkGazeBridge (False for Stage 1/2)
        # Returns: {
        #   'landmark_coords':   (B, 14, 2),
        #   'landmark_heatmaps': (B, 14, 56, 56),
        #   'gaze_vector':       (B, 3),
        #   'gaze_angles':       (B, 2),
        #   'pred_pose_6d':      (B, 6) or None,
        #   'pred_pose_t':       (B, 3) or None,
        # }

class CrossViewAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=512, dropout=0.1, n_layers=2)
    def forward(self, x, n_views, cam_embed=None) -> Tensor
        # x: (B, d_model), n_views: int, cam_embed: (B, d_model) or None
        # Returns: (B, d_model)

class CameraEmbedding(nn.Module):
    def __init__(self, d_model=256)
    def forward(self, R_cam, T_cam) -> Tensor
        # R_cam: (B, 3, 3), T_cam: (B, 3)
        # Returns: (B, d_model)

class PoseEncoder(nn.Module):
    def __init__(self, pose_backbone, pose_feat_dim, d_model=256)
    def forward(self, x) -> tuple[Tensor, Tensor, Tensor]
        # x: (B, 3, 224, 224)
        # Returns: (pose_feat (B, d_model), pred_pose_6d (B, 6), pred_pose_t (B, 3))

class LandmarkGazeBridge(nn.Module):
    def __init__(self, d_model=256, n_heads=4)
    def forward(self, p5_pooled, p2_feat) -> Tensor
        # p5_pooled: (B, D), p2_feat: (B, D, H, W)
        # Returns: (B, D)

def create_raynet(
    core_backbone_name='repnext_m3',
    core_backbone_weight_path=None,
    pose_backbone_weight_path=None,
    n_landmarks=14,
    cross_view_cfg=None,
    pose_backbone_name='repnext_m1',
) -> RayNet
```

## `RayNet/heads.py` -- Task Heads

```python
class IrisPupilLandmarkHead(nn.Module):
    def __init__(self, in_ch=256, n_landmarks=14)
    def forward(self, feat) -> tuple[Tensor, Tensor]
        # feat: (B, 256, 56, 56)
        # Returns: coords (B, 14, 2), heatmaps (B, 14, 56, 56)

class OpticalAxisHead(nn.Module):
    def __init__(self, in_ch=256, hidden_dim=128)
    def pool_features(self, feat) -> Tensor
        # (B, C, H, W) -> (B, C)
    def predict_from_pooled(self, pooled) -> tuple[Tensor, Tensor]
        # (B, C) -> (gaze_vector (B, 3), angles (B, 2))
    def forward(self, feat) -> tuple[Tensor, Tensor]
        # (B, C, H, W) -> (gaze_vector (B, 3), angles (B, 2))
```

## `RayNet/panet.py` -- Multi-Scale Fusion

```python
class PANet(nn.Module):
    def __init__(self, channels_list: list[int], out_channels=256)
    def forward(self, features: list[Tensor]) -> list[Tensor]
        # features: [C1, C2, C3, C4] from backbone
        # Returns: [P2, P3, P4, P5] all with out_channels
```

## `RayNet/coordatt.py` -- Attention

```python
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels: int, reduction=32)
    def forward(self, x: Tensor) -> Tensor
```

---

## `RayNet/losses.py` -- Loss Functions

```python
def gaussian_heatmaps(coords, H, W, sigma=2.0) -> Tensor
    # coords: (B, N, 2) -> heatmaps: (B, N, H, W)

def landmark_loss(pred_hm, pred_coords, gt_coords, feat_H, feat_W, sigma=2.0) -> Tensor

def gaze_loss(pred_gaze, gt_gaze) -> Tensor
    # L1 on unit gaze vectors (CCS). Both (B, 3) -> scalar

def angular_error(pred_gaze, gt_gaze) -> Tensor
    # atan2-based angular error for METRICS ONLY (not backpropagated)

def rotation_6d_to_matrix(r6d) -> Tensor
    # (B, 6) -> (B, 3, 3) via Gram-Schmidt

def matrix_to_rotation_6d(R) -> Tensor
    # (B, 3, 3) -> (B, 6)

def geodesic_loss(pred_R, gt_R) -> Tensor
    # (B, 3, 3), (B, 3, 3) -> scalar (mean geodesic distance in radians)

def pose_prediction_loss(pred_6d, gt_head_R) -> Tensor
    # (B, 6), (B, 3, 3) -> scalar (6D -> matrix -> geodesic)

def translation_loss(pred_t, gt_t, eps=1e-6) -> Tensor
    # (B, 3), (B, 3) -> scalar
    # SmoothL1 on xy + log-space SmoothL1 on z

def ray_target_loss(pred_gaze, eyeball_center, gaze_target, gaze_depth) -> Tensor
    # pred_gaze (B, 3), eyeball_center (B, 3), gaze_target (B, 3), gaze_depth (B,)
    # -> scalar (L1 ray-target consistency)

def total_loss(
    pred_hm, pred_coords, pred_gaze,
    gt_coords, gt_gaze,
    feat_H, feat_W,
    lam_lm=1.0, lam_gaze=0.5, sigma=2.0,
    lam_ray=0.0, eyeball_center=None, gaze_target=None, gaze_depth=None,
    lam_pose=0.0, pred_pose_6d=None, gt_head_R=None,
    lam_trans=0.0, pred_pose_t=None, gt_head_t=None,
) -> tuple[Tensor, dict]
    # Returns: (total_loss, components_dict)
    # Components: landmark_loss, angular_loss, angular_loss_deg, total_loss,
    #             ray_target_loss, pose_loss, pose_loss_deg, translation_loss
```

## `RayNet/multiview_loss.py` -- Multi-View Losses

All operations use unit vectors and camera extrinsics. Float16-safe under AMP.

```python
def reshape_multiview(tensor, n_views=9) -> Tensor
    # (B*V, ...) -> (G, V, ...)

def gaze_ray_consistency_loss(pred_gaze, R_cam, n_pairs=3) -> Tensor
    # pred_gaze: (G, V, 3) unit gaze vectors in CCS
    # R_cam: (G, V, 3, 3) camera extrinsic rotation matrices
    # Transforms to world frame via R_cam^T, L1 loss vs group mean

def landmark_shape_consistency_loss(pred_coords, n_pairs=3) -> Tensor
    # pred_coords: (G, V, N, 2) landmarks in feature-map space
    # Procrustes-style shape comparison (SmoothL1)

def multiview_consistency_loss(
    pred_gaze, pred_coords, R_cam,
    lam_gaze_consist=1.0, lam_shape=0.5, n_views=9,
) -> tuple[Tensor, dict]
    # Returns: (total_mv_loss, {'gaze_consist_loss', 'shape_loss'})
```

---

## `RayNet/kappa.py` -- Kappa Angles

```python
def build_R_kappa(kappa_angles) -> ndarray
    # kappa_angles: [yaw, pitch, (roll)] -> R_kappa: (3, 3)
    # Roll is ALWAYS ZEROED

def ground_truth_optical_axis(eyeball_center, pupil_center) -> ndarray
    # Both (3,) -> unit vector (3,)

def optical_to_visual(optical_axis, R_kappa) -> ndarray
    # (3,) optical -> (3,) visual axis
```

## `RayNet/geometry.py` -- Geometric Post-Processing

```python
def fit_ellipse_algebraic(points_2d) -> tuple
    # (N, 2) -> (center, axes, angle)

def metric_pupil_diameter(iris_pts_2d, pupil_pts_2d, K,
                           iris_radius_mm=5.9) -> tuple[float, float]
    # Returns: (Z_mm, pupil_diameter_mm)

def gaze_to_screen_point(gaze_origin, gaze_direction,
                          screen_normal, screen_point, screen_axes
                         ) -> tuple[ndarray, ndarray]
    # Returns: (screen_uv (2,), hit_point (3,))
```

---

## `RayNet/dataset.py` -- Data Loading

```python
class GazeGeneDataset(Dataset):
    def __init__(self, base_dir, subject_ids=None, camera_ids=None,
                 samples_per_subject=None, eye='L',
                 img_size=224, augment=False, seed=42)
    def __getitem__(self, idx) -> dict
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

## `RayNet/streaming/` -- MosaicML Streaming + MinIO

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
    # batch_size = mv_groups * 9
```

### `RayNet/streaming/convert_to_mds.py`

```python
MDS_COLUMNS: dict   # Column schema (see Dataset page)

def convert_to_mds(dataset, output_dir, split='train',
                    multiview_grouped=True) -> int

def convert_to_mds_chunked(data_dir, output_dir, subject_ids,
                           split='train', multiview_grouped=True,
                           samples_per_subject=None, eye='L',
                           chunk_size=3) -> int
```

---

## `RayNet/train.py` -- Training

```python
HARDWARE_PROFILES: dict     # 'default', 't4', 'l4', 'a10g', 'v100', 'a100', 'h100'
STAGE_CONFIGS: dict         # stages 1, 2, 3 with per-phase configs

def get_phase(epoch: int) -> int
def get_phase_config(epoch: int) -> dict
def apply_hardware_profile(args) -> dict
def setup_hardware(hw: dict, device) -> None

def train_one_epoch(model, train_loader, optimizer, device, epoch, cfg,
                    scaler=None, grad_accum_steps=1, amp_enabled=False,
                    amp_dtype=torch.float16, batch_csv_writer=None,
                    n_views=1) -> dict

def validate(model, val_loader, device, epoch, cfg, amp_enabled=False,
             amp_dtype=torch.float16, n_views=1) -> dict

def train(args) -> None
def parse_args() -> Namespace
```
