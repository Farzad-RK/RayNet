# RayNet Deploy — MinIO + MosaicML Streaming

Self-hosted S3-compatible object storage for streaming GazeGene training data and storing model checkpoints.  MinIO serves as the single storage backend for both the MDS dataset shards and the training checkpoints.

## Architecture

```
+------------------+         +---------------------+         +------------------+
|   Local Machine  |         |    MinIO Server      |         |  Remote GPU      |
|                  |         |    (Docker)          |         |  (Colab / VM)    |
|  GazeGene data   | ------> |                     | ------> |                  |
|  convert to MDS  | upload  |  s3://gazegene/      | stream  |  MDS Streaming   |
|  upload to MinIO |         |    train/            |         |  -> DataLoader   |
|                  |         |    val/              |         |  -> RayNet train |
|                  |         |                     |         |                  |
|                  |         |  s3://raynet-ckpt/   | <------ |  Checkpoints     |
|                  |         |    checkpoints/      |  save   |  -> latest.pt    |
|                  |         |      run_20260405/   |         |  -> best_model.pt|
|                  |         |        metadata.json |         |  -> metadata.json|
+------------------+         +---------------------+         +------------------+
```

## Quick Start

### 1. Start MinIO

```bash
cd deploy/
cp .env.example .env
# Edit .env — change MINIO_ROOT_PASSWORD!

docker compose up -d
```

MinIO is now running:
- **S3 API**: http://localhost:9000
- **Web Console**: http://localhost:9001

### 2. Convert Dataset to MDS

```bash
# Training split (subjects 1-46)
python -m RayNet.streaming.convert_to_mds \
    --data_dir /path/to/GazeGene_FaceCrops \
    --output_dir ./mds_shards/train \
    --split train \
    --subject_start 1 --subject_end 46

# Validation split (subjects 47-56)
python -m RayNet.streaming.convert_to_mds \
    --data_dir /path/to/GazeGene_FaceCrops \
    --output_dir ./mds_shards/val \
    --split val \
    --subject_start 47 --subject_end 56
```

### 3. Upload Shards to MinIO

```bash
python -m RayNet.streaming.minio_utils upload \
    --shard_dir ./mds_shards/train \
    --bucket gazegene \
    --prefix train \
    --endpoint http://localhost:9000

python -m RayNet.streaming.minio_utils upload \
    --shard_dir ./mds_shards/val \
    --bucket gazegene \
    --prefix val \
    --endpoint http://localhost:9000
```

### 4. Verify

```bash
python -m RayNet.streaming.minio_utils verify --endpoint http://localhost:9000
```

Or open the web console at http://localhost:9001 (login with your `.env` credentials).

### 5. Train with MDS Streaming + MinIO Checkpoints

This is the recommended workflow.  Data streams from MinIO via MDS and checkpoints are saved back to MinIO, all through the same endpoint:

```bash
# Set MinIO credentials
export S3_ENDPOINT_URL=http://YOUR_SERVER_IP:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=your-password

# Train with MDS streaming and MinIO checkpoints
python -m RayNet.train \
    --mds_streaming \
    --mds_train s3://gazegene/train \
    --mds_val s3://gazegene/val \
    --minio_endpoint http://YOUR_SERVER_IP:9000 \
    --ckpt_bucket raynet-checkpoints \
    --ckpt_every 5 \
    --profile a100 \
    --backbone repnext_m3
```

### 6. Resume After Interruption

If training is interrupted, resume from exactly where it stopped:

```bash
python -m RayNet.train \
    --mds_streaming \
    --mds_train s3://gazegene/train \
    --mds_val s3://gazegene/val \
    --minio_endpoint http://YOUR_SERVER_IP:9000 \
    --ckpt_bucket raynet-checkpoints \
    --run_id run_20260405_143022 \
    --resume \
    --profile a100
```

The `--run_id` must match the original run (printed at startup, or find it via `CheckpointManager.list_runs()`).

## Checkpoint Storage

When `--ckpt_bucket` is provided, checkpoints are uploaded to MinIO and organized by run:

```
s3://raynet-checkpoints/checkpoints/<run_id>/
    metadata.json            # Run config + per-epoch metrics
    latest.pt                # Overwritten every epoch (for --resume)
    best_model.pt            # Best validation loss
    checkpoint_epoch5.pt     # Periodic snapshot
    checkpoint_epoch10.pt
    ...
```

### What Each Checkpoint Contains

| File | Contents | Purpose |
|------|----------|---------|
| `latest.pt` | model + optimizer + scheduler + scaler + metrics | Resume training |
| `best_model.pt` | model + optimizer + scheduler + scaler + val_loss | Deploy / fine-tune |
| `checkpoint_epochN.pt` | model + optimizer + scheduler + scaler + metrics | Historical snapshot |
| `metadata.json` | run config, per-epoch train/val metrics, best epoch | Experiment tracking |

### Checkpoint Frequency

Controlled by `--ckpt_every N` (default: 5):

- `latest.pt` is saved **every epoch** (always, for robust resume)
- `checkpoint_epochN.pt` is saved **every N epochs**
- `best_model.pt` is saved **whenever validation loss improves**

```bash
# Save a named checkpoint every 2 epochs
python -m RayNet.train --ckpt_bucket raynet-checkpoints --ckpt_every 2 ...

# Save only latest + best (no periodic)
python -m RayNet.train --ckpt_bucket raynet-checkpoints --ckpt_every 999 ...
```

### Using CheckpointManager Directly

```python
from RayNet.streaming import CheckpointManager

mgr = CheckpointManager(
    bucket='raynet-checkpoints',
    endpoint='http://localhost:9000',
    run_id='run_20260405_143022',
)

# List all runs
print(mgr.list_runs())

# List checkpoints for a run
print(mgr.list_checkpoints())

# Load best model for inference
state = mgr.load_best(map_location='cuda')
model.load_state_dict(state['model_state_dict'])

# View run metadata (config + metrics)
meta = mgr.get_metadata()
print(meta['config'])
print(meta['best'])
```

## All Training Modes

RayNet supports three data loading modes.  All three work with MinIO checkpoints:

| Mode | Flag | Data Source | Best For |
|------|------|-------------|----------|
| Local | `--data_dir /path` | Local disk | Small-scale, debugging |
| MDS streaming | `--mds_streaming` | MinIO/S3 MDS shards | Production training |
| WebDataset | `--streaming` | HuggingFace / HTTP | HuggingFace Hub datasets |

### MDS Streaming (recommended)

```bash
python -m RayNet.train \
    --mds_streaming \
    --mds_train s3://gazegene/train \
    --mds_val s3://gazegene/val \
    --ckpt_bucket raynet-checkpoints \
    --profile a100
```

### Local Disk

```bash
python -m RayNet.train \
    --data_dir /path/to/GazeGene_FaceCrops \
    --ckpt_bucket raynet-checkpoints \
    --profile default
```

### Colab Notebook Setup

```python
import os
os.environ['S3_ENDPOINT_URL'] = 'http://YOUR_SERVER_IP:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-password'

from RayNet.streaming import create_streaming_dataloaders, CheckpointManager

# Data from MinIO
train_loader, val_loader = create_streaming_dataloaders(
    remote_train='s3://gazegene/train',
    remote_val='s3://gazegene/val',
    batch_size=2048,
    num_workers=8,
)

# Checkpoints to MinIO
ckpt = CheckpointManager(bucket='raynet-checkpoints')
```

## CLI Reference — Checkpoint & Streaming Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mds_streaming` | off | Stream MDS shards from MinIO/S3 |
| `--mds_train` | — | MDS remote URL for training (e.g. `s3://gazegene/train`) |
| `--mds_val` | — | MDS remote URL for validation |
| `--minio_endpoint` | `$S3_ENDPOINT_URL` | MinIO endpoint URL |
| `--ckpt_bucket` | — | MinIO bucket for checkpoints (enables MinIO storage) |
| `--ckpt_prefix` | `checkpoints` | Key prefix under the bucket |
| `--ckpt_every` | `5` | Save named checkpoint every N epochs |
| `--run_id` | auto-generated | Run ID for checkpoint grouping |
| `--resume` | off | Resume from `latest.pt` of the given `--run_id` |

## Exposing MinIO Externally

For Colab/remote access, MinIO must be reachable from the internet.

### Option A: Direct Port Exposure

If your server has a public IP, just use port 9000:

```python
os.environ['S3_ENDPOINT_URL'] = 'http://YOUR_PUBLIC_IP:9000'
```

### Option B: Ngrok Tunnel (for testing)

```bash
ngrok http 9000
# Use the ngrok URL as S3_ENDPOINT_URL
```

### Option C: Nginx + TLS (production)

Uncomment the nginx service in `docker-compose.yml` and provide TLS certificates. Then:

```python
os.environ['S3_ENDPOINT_URL'] = 'https://minio.yourdomain.com'
```

## Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | MinIO server + bucket initialization |
| `.env.example` | Configuration template (copy to `.env`) |
| `setup_minio.sh` | Manual bucket setup script |
| `README.md` | This file |

### Streaming & Checkpoint Modules

| File | Purpose |
|------|---------|
| `RayNet/streaming/dataset.py` | MDS streaming dataset + dataloaders |
| `RayNet/streaming/convert_to_mds.py` | Convert GazeGene to MDS shards |
| `RayNet/streaming/minio_utils.py` | Upload shards, configure MinIO env |
| `RayNet/streaming/checkpoint.py` | CheckpointManager for MinIO storage |
| `RayNet/train.py` | Training script with all modes |

## Requirements

- Docker and Docker Compose v2
- Python packages: `mosaicml-streaming`, `minio`

```bash
pip install mosaicml-streaming minio
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Connection refused` from Colab | Check firewall allows port 9000 inbound |
| `Access denied` | Verify AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY match .env |
| Slow uploads | MinIO is limited by your upload bandwidth, not MinIO itself |
| `Bucket not found` | Run `./setup_minio.sh` or check the minio-init container logs |
| `SSL error` with http:// endpoint | Set `S3_USE_SSL=0` in environment |
| `--resume` fails with "not found" | Check `--run_id` matches exactly (case-sensitive) |
| Checkpoints not appearing in MinIO | Verify `--ckpt_bucket` is set and MinIO is reachable |
