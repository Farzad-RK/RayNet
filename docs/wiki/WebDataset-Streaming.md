# WebDataset Streaming

RayNet supports streaming training data from local `.tar` shards or from Hugging Face Hub using [WebDataset](https://github.com/webdataset/webdataset). This enables cloud training (e.g., Colab A100) without copying the full dataset.

**Source**: `RayNet/webdataset_utils.py`

## Overview

The pipeline has three stages:

```
1. Create shards    Local GazeGene dataset -> .tar files
2. Push to Hub      .tar files -> HF Hub dataset repo
3. Stream           HF Hub -> WebDataset -> PyTorch DataLoader
```

---

## 1. Create Shards

Convert the local GazeGene dataset into WebDataset `.tar` shard files. Each sample is **pre-processed** (normalized, landmarks warped) so streaming avoids duplicating the normalization logic.

### CLI

```bash
# Training split (subjects 1-46)
python -m RayNet.webdataset_utils create_shards \
    --data_dir /path/to/GazeGene_FaceCrops \
    --output_dir ./shards/train \
    --split train \
    --subject_start 1 \
    --subject_end 46 \
    --samples_per_shard 1000 \
    --eye L

# Validation split (subjects 47-56)
python -m RayNet.webdataset_utils create_shards \
    --data_dir /path/to/GazeGene_FaceCrops \
    --output_dir ./shards/val \
    --split val \
    --subject_start 47 \
    --subject_end 56
```

### Shard Contents

Each sample in a `.tar` file contains:

| File | Format | Shape | Description |
|------|--------|-------|-------------|
| `image.jpg` | JPEG (quality 95) | 224x224x3 | Normalized RGB eye crop |
| `landmark_coords.npy` | NumPy | (14, 2) | Feature-space landmarks |
| `landmark_coords_px.npy` | NumPy | (14, 2) | Pixel-space landmarks |
| `optical_axis.npy` | NumPy | (3,) | GT optical axis |
| `R_norm.npy` | NumPy | (3, 3) | Normalization rotation |
| `R_kappa.npy` | NumPy | (3, 3) | Kappa rotation |
| `K.npy` | NumPy | (3, 3) | Camera intrinsics |
| `R_cam.npy` | NumPy | (3, 3) | Camera extrinsics R |
| `T_cam.npy` | NumPy | (3,) | Camera extrinsics T |
| `M_norm_inv.npy` | NumPy | (3, 3) | Inverse warp matrix |
| `eyeball_center_3d.npy` | NumPy | (3,) | Eye center in CCS |
| `metadata.json` | JSON | - | `{subject, cam_id, frame_idx}` |

### Sample Key Format

```
{subject:04d}_{cam:01d}_{frame:06d}
e.g. "0001_3_000042"
```

### Multi-View Grouping

When `multiview_grouped=True` (default), all 9 views of the same `(subject, frame)` are placed **consecutively** within a shard, sorted by cam_id. This enables the multi-view streaming loader to read 9 consecutive samples as one group.

Groups with fewer than 9 cameras are placed at the end of the shard sequence.

### Shard Naming

```
gazegene-train-000000.tar
gazegene-train-000001.tar
...
gazegene-val-000000.tar
```

---

## 2. Push to Hugging Face Hub

Upload shards to a HF dataset repository:

```bash
# Training shards
python -m RayNet.webdataset_utils push \
    --shard_dir ./shards/train \
    --repo_id YOUR_USERNAME/gazegene-wds \
    --split train \
    --private

# Validation shards
python -m RayNet.webdataset_utils push \
    --shard_dir ./shards/val \
    --repo_id YOUR_USERNAME/gazegene-wds \
    --split val
```

This creates the repository structure on HF Hub:

```
YOUR_USERNAME/gazegene-wds/
├── train/
│   ├── gazegene-train-000000.tar
│   ├── gazegene-train-000001.tar
│   └── ...
└── val/
    ├── gazegene-val-000000.tar
    └── ...
```

### Authentication

Login first:
```bash
huggingface-cli login
```

Or in Python/notebook:
```python
from huggingface_hub import notebook_login
notebook_login()
```

---

## 3. Stream During Training

### Standard Streaming (single-view)

```python
from RayNet.webdataset_utils import create_streaming_dataloader

loader = create_streaming_dataloader(
    urls="pipe:curl -sL https://huggingface.co/datasets/USER/gazegene-wds/resolve/main/train/gazegene-train-{000000..000099}.tar",
    batch_size=512,
    num_workers=4,
    shuffle=True,
)

for batch in loader:
    images = batch['image']       # (512, 3, 224, 224)
    landmarks = batch['landmark_coords']  # (512, 14, 2)
    ...
```

### Multi-View Streaming

For phases that need multi-view batches:

```python
from RayNet.webdataset_utils import create_multiview_streaming_dataloader

mv_loader = create_multiview_streaming_dataloader(
    urls="pipe:curl -sL .../train/gazegene-train-{000000..000099}.tar",
    mv_groups=2,        # 2 groups * 9 cameras = 18 samples per batch
    num_workers=4,
    shuffle=True,
)

for batch in mv_loader:
    images = batch['image']  # (18, 3, 224, 224)  flat: 2 groups * 9 views
    ...
```

The `MultiViewStreamingDataset` reads 9 consecutive samples (same subject+frame), stacks them, and the collate function flattens groups into the expected `(G*9, ...)` format.

### URL Formats

| Source | URL Pattern |
|--------|-------------|
| Local files | `./shards/train/gazegene-train-{000000..000099}.tar` |
| HF Hub | `pipe:curl -sL https://huggingface.co/datasets/USER/REPO/resolve/main/train/gazegene-train-{000000..000099}.tar` |
| S3 | `pipe:aws s3 cp s3://bucket/train/gazegene-train-{000000..000099}.tar -` |

### URL Builder Helper

```python
from RayNet.webdataset_utils import hf_hub_shard_urls

url = hf_hub_shard_urls(
    repo_id='USER/gazegene-wds',
    split='train',
    n_shards=100,
)
# Returns: "pipe:curl -sL https://huggingface.co/datasets/USER/gazegene-wds/resolve/main/train/gazegene-train-{000000..000099}.tar"
```

---

## Training with Streaming

### Via CLI

```bash
python -m RayNet.train \
    --profile a100 \
    --streaming \
    --dataset_url "pipe:curl -sL https://huggingface.co/datasets/USER/gazegene-wds/resolve/main/train/gazegene-train-{000000..000099}.tar" \
    --val_dataset_url "pipe:curl -sL https://huggingface.co/datasets/USER/gazegene-wds/resolve/main/val/gazegene-val-{000000..000019}.tar" \
    --epochs 30
```

### Via Colab Notebook

See `notebooks/train_colab_a100.ipynb` cells 4-6 for streaming setup.

---

## Shard Creation API

For programmatic shard creation:

```python
from RayNet.dataset import GazeGeneDataset
from RayNet.webdataset_utils import create_webdataset_shards, push_shards_to_hub

# Create dataset
ds = GazeGeneDataset(
    base_dir='/path/to/GazeGene_FaceCrops',
    subject_ids=list(range(1, 47)),
    eye='L',
    augment=False,
)

# Write shards
n_shards = create_webdataset_shards(
    dataset=ds,
    output_dir='./shards/train',
    samples_per_shard=1000,
    split='train',
    multiview_grouped=True,    # group 9 views consecutively
)

# Push to HF Hub
push_shards_to_hub(
    shard_dir='./shards/train',
    repo_id='USER/gazegene-wds',
    split='train',
    private=True,
)
```

---

## Performance Notes

- Shard size of 1000 samples is a good balance between I/O efficiency and shuffle quality
- Streaming shuffle buffer of 1000 samples provides reasonable randomization
- For multi-view streaming, shards must be created with `multiview_grouped=True`
- JPEG quality 95 preserves visual quality while reducing shard size ~5x vs PNG
- Streaming adds ~10-20% overhead vs local disk due to JPEG decode + numpy deserialization
