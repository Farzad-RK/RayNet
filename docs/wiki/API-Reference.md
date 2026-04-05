# API Reference

Function signatures for all public modules in RayNet v2.

---

## `RayNet/raynet.py` — Model

```python
class RayNet(nn.Module):
    def __init__(self, backbone, panet_channels=256, n_landmarks=14)
    def forward(self, x: Tensor) -> dict
        # x: (B, 3, 224, 224)
        # Returns: {
        #   'landmark_coords':   (B, 14, 2),
        #   'landmark_heatmaps': (B, 14, 56, 56),
        #   'gaze_vector':       (B, 3),
        #   'gaze_angles':       (B, 2),
        # }

def create_raynet(backbone_name='repnext_m3', weight_path=None, n_landmarks=14) -> RayNet
```

## `RayNet/heads.py` — Task Heads

```python
class IrisPupilLandmarkHead(nn.Module):
    def __init__(self, in_channels=256, n_landmarks=14)
    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor]
        # feat: (B, 256, 56, 56)
        # Returns: coords (B, 14, 2), heatmaps (B, 14, 56, 56)

class OpticalAxisHead(nn.Module):
    def __init__(self, in_channels=256, hidden_dim=128)
    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor]
        # feat: (B, 256, 7, 7)
        # Returns: gaze_vector (B, 3), angles (B, 2)
```

## `RayNet/panet.py` — Multi-Scale Fusion

```python
class PANet(nn.Module):
    def __init__(self, in_channels_list: list[int], out_channels=256)
    def forward(self, features: list[Tensor]) -> list[Tensor]
        # features: [C1, C2, C3, C4] from backbone
        # Returns: [P2, P3, P4, P5] all with out_channels
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
    # coords: (B, N, 2) -> heatmaps: (B, N, H, W)

def landmark_loss(pred_hm, pred_coords, gt_coords, feat_H, feat_W, sigma=2.0) -> Tensor

def gaze_loss(pred_gaze, gt_gaze) -> Tensor
    # L1 loss on unit gaze vectors (following GazeGene paper Sec 4.1.1)
    # Both (B, 3) unit vectors -> scalar

def angular_error(pred_gaze, gt_gaze) -> Tensor
    # atan2-based angular error for METRICS ONLY (not backpropagated)
    # Numerically stable everywhere (no acos singularity)
    # Both (B, 3) -> scalar (mean angular error in radians)

def total_loss(pred_hm, pred_coords, pred_gaze, gt_coords, gt_gaze,
               feat_H, feat_W, lam_lm=1.0, lam_gaze=0.5, sigma=2.0
              ) -> tuple[Tensor, dict]
    # Returns: (total_loss, {'landmark_loss', 'angular_loss', 'angular_loss_deg', 'total_loss'})
    # angular_loss/angular_loss_deg are detached metrics (not in loss computation graph)
```

## `RayNet/multiview_loss.py` — Multi-View Losses (Ray-Based)

All operations use unit vectors in normalized space. No raw 3D coordinates, no matrix
inversions, no SVD. Float16-safe under AMP.

```python
def reshape_multiview(tensor, n_views=9) -> Tensor
    # (B*V, ...) -> (G, V, ...)

def gaze_ray_consistency_loss(pred_gaze, R_norm, n_pairs=3) -> Tensor
    # pred_gaze: (G, V, 3) unit gaze vectors in normalized space
    # R_norm: (G, V, 3, 3) normalization rotation matrices
    # Transforms to world frame via R_norm^T, L1 loss vs group mean

def landmark_shape_consistency_loss(pred_coords, n_pairs=3) -> Tensor
    # pred_coords: (G, V, N, 2) landmarks in feature-map space
    # Translation/scale-invariant (Procrustes-style) shape comparison

def multiview_consistency_loss(
    pred_gaze,       # (B_total, 3) predicted gaze unit vectors
    pred_coords,     # (B_total, N, 2) predicted landmark coords
    R_norm,          # (B_total, 3, 3) normalization rotation matrices
    lam_gaze_consist=1.0, lam_shape=0.5, n_views=9
) -> tuple[Tensor, dict]
    # Returns: (total_mv_loss, {'gaze_consist_loss', 'shape_loss'})

def sanity_check_roundtrip(dataset, n_samples=50, threshold_px=2.0) -> tuple[float, bool]
```

---

## `RayNet/normalization.py` — Image Normalization

```python
def normalize_sample(image, K, R_head, t_eye,
                     d_norm=600, f_norm=960, size=224
                    ) -> tuple[ndarray, ndarray]
    # Returns: (img_norm (224,224,3), R_norm (3,3))

def denormalize_gaze(gaze_norm, R_norm) -> ndarray
    # (3,) normalized -> (3,) camera space

def normalize_gaze(gaze_cam, R_norm, R_head=None) -> ndarray
    # (3,) camera space -> (3,) normalized

def compute_normalization_matrix(K, t_eye,
                                  d_norm=600, f_norm=960, size=224
                                 ) -> tuple[ndarray, ndarray]
    # Returns: (M (3,3), R_norm (3,3))

def warp_points_2d(points_2d, M) -> ndarray
    # (N, 2) -> (N, 2) warped through homography M
```

## `RayNet/kappa.py` — Kappa Angles

```python
def build_R_kappa(kappa_angles) -> ndarray
    # kappa_angles: [yaw, pitch, (roll)] -> R_kappa: (3, 3)
    # Roll is ALWAYS ZEROED

def ground_truth_optical_axis(eyeball_center, pupil_center) -> ndarray
    # Both (3,) -> unit vector (3,)

def optical_to_visual(optical_axis, R_kappa) -> ndarray
    # (3,) optical -> (3,) visual axis
```

## `RayNet/geometry.py` — Geometric Post-Processing

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

## `RayNet/dataset.py` — Data Loading

```python
class GazeGeneDataset(Dataset):
    def __init__(self, base_dir, subject_ids=None, camera_ids=None,
                 samples_per_subject=None, eye='L',
                 d_norm=600, f_norm=960, img_size=224,
                 augment=False, seed=42)
    def __getitem__(self, idx) -> dict
    def __len__(self) -> int

class MultiViewBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=1, shuffle=True, ensure_multiview=True)
    def __iter__(self) -> Iterator[list[int]]
    def __len__(self) -> int

def gazegene_collate_fn(batch: list[dict]) -> dict

def create_dataloaders(base_dir, train_subjects, val_subjects,
                        batch_size=4, num_workers=4,
                        samples_per_subject=None, eye='L',
                        ensure_multiview=False
                       ) -> tuple[DataLoader, DataLoader]
```

## `RayNet/webdataset_utils.py` — WebDataset Streaming

```python
def create_webdataset_shards(dataset, output_dir, samples_per_shard=1000,
                              split='train', multiview_grouped=True) -> int

def push_shards_to_hub(shard_dir, repo_id, split='train', private=True) -> None

def create_streaming_dataloader(urls, batch_size=512, num_workers=4,
                                 shuffle=True, epoch_length=None) -> DataLoader

def create_multiview_streaming_dataloader(urls, mv_groups=2, num_workers=4,
                                           shuffle=True, epoch_length=None) -> DataLoader

def hf_hub_shard_urls(repo_id, split='train', n_shards=None,
                       shard_pattern=None) -> str

class MultiViewStreamingDataset(IterableDataset):
    def __init__(self, urls, shuffle=True)
    def __iter__(self) -> Iterator[dict]
```

## `RayNet/streaming/` — MosaicML Streaming + MinIO

### `RayNet/streaming/dataset.py` — Streaming Dataset & Loaders

```python
class StreamingGazeGeneDataset(StreamingDataset):
    def __init__(self, remote=None, local=None, split=None,
                 shuffle=True, batch_size=None, **kwargs)
    def __getitem__(self, idx) -> dict
        # Returns dict matching GazeGeneDataset.__getitem__ format:
        # image (3,224,224), landmark_coords (14,2), optical_axis (3,),
        # R_norm (3,3), R_kappa (3,3), K (3,3), R_cam (3,3), T_cam (3,),
        # M_norm_inv (3,3), eyeball_center_3d (3,), subject, cam_id, frame_idx

def create_streaming_dataloaders(
    remote_train, remote_val,
    local_cache='./mds_cache',
    batch_size=512, num_workers=4,
    shuffle_train=True, pin_memory=True,
    prefetch_factor=2, persistent_workers=False,
    **streaming_kwargs,
) -> tuple[DataLoader, DataLoader]
    # remote_train/val: 's3://bucket/train', local path, or 'gs://...'
    # S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY env vars
    # must be set for MinIO/S3 access

def create_multiview_streaming_dataloaders(
    remote_train, remote_val,
    local_cache='./mds_cache',
    mv_groups=2, num_workers=4,
    **streaming_kwargs,
) -> tuple[DataLoader, DataLoader]
    # batch_size = mv_groups * 9
    # Requires shards created with multiview_grouped=True
```

### `RayNet/streaming/convert_to_mds.py` — MDS Shard Creation

```python
MDS_COLUMNS: dict   # Column schema: image->jpeg, landmarks->ndarray, etc.

def convert_to_mds(dataset, output_dir, split='train',
                    multiview_grouped=True) -> int
    # dataset: GazeGeneDataset instance
    # multiview_grouped: write 9-camera groups consecutively
    # Returns: number of samples written
    # Compression: zstd, hashes: SHA-1, shard limit: 128 MB

# CLI: python -m RayNet.streaming.convert_to_mds \
#          --data_dir PATH --output_dir PATH --split train \
#          --subject_start 1 --subject_end 46 \
#          [--samples_per_subject N] [--eye L|R] [--no_multiview_group]
```

### `RayNet/streaming/minio_utils.py` — MinIO Upload & Config

```python
def configure_minio_env(endpoint_url, access_key, secret_key) -> None
    # Sets S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    # Must be called BEFORE creating StreamingDataset instances

def upload_to_minio(shard_dir, bucket, prefix='',
                     endpoint='http://localhost:9000',
                     access_key=None, secret_key=None,
                     make_bucket=True) -> None
    # Uploads all files in shard_dir to s3://bucket/prefix/

def minio_shard_url(bucket, prefix, endpoint=None) -> str
    # Returns 's3://bucket/prefix' URL for mosaicml-streaming

def verify_minio_connection(endpoint='http://localhost:9000',
                             access_key=None, secret_key=None) -> list[str]
    # Returns list of bucket names, or raises on failure

# CLI: python -m RayNet.streaming.minio_utils upload \
#          --shard_dir PATH --bucket gazegene --prefix train
#      python -m RayNet.streaming.minio_utils verify \
#          --endpoint http://localhost:9000
```

---

## `RayNet/train.py` — Training

```python
HARDWARE_PROFILES: dict     # 'default', 't4', 'l4', 'a10g', 'v100', 'a100', 'h100'
PHASE_CONFIG: dict          # phases 1, 2, 3

def get_phase(epoch: int) -> int
def get_phase_config(epoch: int) -> dict
def apply_hardware_profile(args) -> dict
def setup_hardware(hw: dict, device) -> None

def train_one_epoch(model, train_loader, optimizer, device, epoch, cfg,
                    scaler=None, grad_accum_steps=1, amp_enabled=False) -> dict

def validate(model, val_loader, device, epoch, cfg, amp_enabled=False) -> dict

def train(args) -> None
def parse_args() -> Namespace
```
