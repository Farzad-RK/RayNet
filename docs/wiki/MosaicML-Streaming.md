# MosaicML Streaming + MinIO

RayNet supports [MosaicML Streaming](https://github.com/mosaicml/streaming) as a high-performance alternative to WebDataset. MDS (Mosaic Dataset Shard) format provides deterministic shuffling, resumable training, and native S3/MinIO support.

Combined with a self-hosted **MinIO** server, this gives you full control over your data pipeline without depending on Hugging Face Hub.

**Source**: `RayNet/streaming/`

## Architecture

```
Local Machine                    MinIO Server                  Colab / Cloud
+------------------+           +------------------+           +------------------+
| GazeGene dataset |           | Docker: MinIO    |           | StreamingDataset |
|        |         |           |                  |           |        |         |
|  convert_to_mds  |  upload   | s3://gazegene/   |  stream   |  DataLoader      |
|  -----> MDS dir  | --------> |   train/*.mds    | --------> |  -----> model    |
|                  |           |   val/*.mds      |           |                  |
+------------------+           +------------------+           +------------------+
```

## Why MosaicML Streaming over WebDataset?

| Feature | WebDataset | MosaicML Streaming |
|---------|------------|-------------------|
| Shuffle | Buffer-based (approximate) | **Deterministic** across workers/nodes |
| Resume | No (restarts from scratch) | **Yes** (exact sample-level) |
| Multi-node | Manual shard assignment | **Built-in** (auto-partitioning) |
| S3/MinIO | via pipe:curl | **Native** (boto3-based) |
| Local cache | Manual | **Automatic** (with eviction) |
| Epoch boundary | Approximate | **Exact** |

---

## Step 1: Convert Dataset to MDS

```bash
# Training (subjects 1-46)
python -m RayNet.streaming.convert_to_mds \
    --data_dir /path/to/GazeGene_FaceCrops \
    --output_dir ./mds_shards/train \
    --split train \
    --subject_start 1 --subject_end 46

# Validation (subjects 47-56)
python -m RayNet.streaming.convert_to_mds \
    --data_dir /path/to/GazeGene_FaceCrops \
    --output_dir ./mds_shards/val \
    --split val \
    --subject_start 47 --subject_end 56
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | required | Path to GazeGene dataset |
| `--output_dir` | required | Output directory for MDS shards |
| `--split` | `train` | Split name (metadata) |
| `--subject_start` | `1` | First subject ID |
| `--subject_end` | `46` | Last subject ID |
| `--samples_per_subject` | `None` (all) | Limit frames per subject |
| `--eye` | `L` | Which eye |
| `--no_multiview_group` | `False` | Disable multi-view consecutive ordering |

### MDS Shard Format

Each shard contains pre-processed samples with zstd compression:

| Column | MDS Type | Shape |
|--------|----------|-------|
| `image` | `jpeg` | 224x224x3 |
| `landmark_coords` | `ndarray` | (14, 2) |
| `landmark_coords_px` | `ndarray` | (14, 2) |
| `optical_axis` | `ndarray` | (3,) |
| `R_norm` | `ndarray` | (3, 3) |
| `R_kappa` | `ndarray` | (3, 3) |
| `K` | `ndarray` | (3, 3) |
| `R_cam` | `ndarray` | (3, 3) |
| `T_cam` | `ndarray` | (3,) |
| `M_norm_inv` | `ndarray` | (3, 3) |
| `eyeball_center_3d` | `ndarray` | (3,) |
| `subject` | `int` | scalar |
| `cam_id` | `int` | scalar |
| `frame_idx` | `int` | scalar |

Shard size limit: 128 MB. Integrity hashes: SHA-1.

---

## Step 2: Deploy MinIO

See `deploy/README.md` for full instructions.

### Quick Start

```bash
cd deploy/
cp .env.example .env
# Edit .env — set a strong MINIO_ROOT_PASSWORD

docker compose up -d
```

- S3 API: `http://localhost:9000`
- Web Console: `http://localhost:9001`

### Upload Shards

```bash
python -m RayNet.streaming.minio_utils upload \
    --shard_dir ./mds_shards/train \
    --bucket gazegene --prefix train

python -m RayNet.streaming.minio_utils upload \
    --shard_dir ./mds_shards/val \
    --bucket gazegene --prefix val
```

### Verify

```bash
python -m RayNet.streaming.minio_utils verify
```

Or browse http://localhost:9001.

---

## Step 3: Stream in Training

### Configure MinIO Connection

```python
from RayNet.streaming.minio_utils import configure_minio_env

configure_minio_env(
    endpoint_url='http://YOUR_SERVER_IP:9000',
    access_key='minioadmin',
    secret_key='your-password',
)
```

Or set environment variables directly:
```bash
export S3_ENDPOINT_URL=http://YOUR_SERVER_IP:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=your-password
```

### Create DataLoaders

```python
from RayNet.streaming import create_streaming_dataloaders

train_loader, val_loader = create_streaming_dataloaders(
    remote_train='s3://gazegene/train',
    remote_val='s3://gazegene/val',
    local_cache='/tmp/mds_cache',
    batch_size=2048,
    num_workers=8,
)

for batch in train_loader:
    images = batch['image']              # (2048, 3, 224, 224)
    landmarks = batch['landmark_coords'] # (2048, 14, 2)
    gaze = batch['optical_axis']         # (2048, 3)
    # ... train step
```

### Multi-View DataLoaders

For phases that need 9-camera groups:

```python
from RayNet.streaming.dataset import create_multiview_streaming_dataloaders

mv_train, mv_val = create_multiview_streaming_dataloaders(
    remote_train='s3://gazegene/train',
    remote_val='s3://gazegene/val',
    mv_groups=16,     # 16 groups * 9 cameras = 144 samples/batch
    num_workers=8,
)
```

Requires shards created with `multiview_grouped=True` (the default).

---

## Colab Example

```python
# Cell 1: Install
!pip install -q mosaicml-streaming minio

# Cell 2: Configure MinIO
import os
os.environ['S3_ENDPOINT_URL'] = 'http://YOUR_SERVER:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-password'

# Cell 3: Create loaders
from RayNet.streaming import create_streaming_dataloaders

train_loader, val_loader = create_streaming_dataloaders(
    remote_train='s3://gazegene/train',
    remote_val='s3://gazegene/val',
    batch_size=2048,
    num_workers=8,
)

# Cell 4: Train (same loop as train.py)
for batch in train_loader:
    ...
```

---

## Local-Only Mode (No MinIO)

You can also stream from local MDS shards without MinIO:

```python
train_loader, val_loader = create_streaming_dataloaders(
    remote_train='./mds_shards/train',   # local path
    remote_val='./mds_shards/val',
    batch_size=512,
)
```

---

## Caching Behavior

MosaicML Streaming automatically caches downloaded shards in `local_cache`:

- First epoch: downloads shards on demand (streaming)
- Subsequent epochs: reads from local cache (fast)
- Cache is persistent across restarts

For Colab with limited disk, set `local_cache='/tmp/mds_cache'` (cleared on runtime restart).

---

## Performance Tips

| Setting | Recommendation |
|---------|---------------|
| `num_workers` | 8 for A100, 4 for consumer GPUs |
| `prefetch_factor` | 4 for streaming, 2 for local |
| `batch_size` | 2048 (A100) or 512 (default) |
| Shard size | 128 MB (default, good balance) |
| Compression | zstd (default, fast decode) |
| Network | MinIO on same LAN as training = best performance |
