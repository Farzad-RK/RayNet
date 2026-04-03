# RayNet Deploy — MinIO + MosaicML Streaming

Self-hosted S3-compatible object storage for streaming GazeGene training data to remote environments (Google Colab, cloud VMs, etc.).

## Architecture

```
+------------------+         +------------------+         +------------------+
|   Local Machine  |         |   MinIO Server   |         |  Google Colab    |
|                  |         |   (Docker)       |         |  (A100 GPU)      |
|  GazeGene data   | ------> |                  | ------> |                  |
|  convert to MDS  | upload  |  s3://gazegene/  | stream  |  StreamingDataset|
|  upload to MinIO |         |    train/        |         |  -> DataLoader   |
+------------------+         |    val/          |         |  -> RayNet train |
                             +------------------+         +------------------+
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

### 5. Stream from Colab

In your Colab notebook:

```python
import os
os.environ['S3_ENDPOINT_URL'] = 'http://YOUR_SERVER_IP:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-password'

from RayNet.streaming import create_streaming_dataloaders

train_loader, val_loader = create_streaming_dataloaders(
    remote_train='s3://gazegene/train',
    remote_val='s3://gazegene/val',
    batch_size=2048,
    num_workers=8,
)
```

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
