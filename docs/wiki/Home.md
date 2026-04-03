# RayNet v2

RayNet v2 is a real-time gaze estimation system built for the [GazeGene](https://github.com/gazegene) multi-camera dataset. It jointly predicts **iris/pupil landmarks** and the **optical axis** from a single normalized eye crop, then uses multi-view geometric constraints across 9 synchronized cameras to resolve depth ambiguity.

## Key Features

| Feature | Details |
|---------|---------|
| Backbone | RepNeXt-M3 (7.8 M params) |
| Input | 224 x 224 normalized eye crop |
| Task A | 14 landmarks (10 iris + 4 pupil) via soft-argmax heatmaps |
| Task B | Optical axis unit vector (pitch + yaw) |
| Multi-view | Reprojection consistency + triangulation masking loss |
| Training | 3-phase progressive schedule with dynamic loss weighting |
| Hardware | Default profile + A100 profile (fp16, torch.compile, TF32) |
| Streaming | WebDataset shards via Hugging Face Hub |

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/RayNet.git
cd RayNet
pip install -r requirements.txt
```

### 2. Train (local dataset)

```bash
python -m RayNet.train \
    --data_dir /path/to/GazeGene_FaceCrops \
    --backbone repnext_m3 \
    --profile default \
    --epochs 30
```

### 3. Train on A100 (streaming from HF Hub)

```bash
python -m RayNet.train \
    --profile a100 \
    --streaming \
    --dataset_url "pipe:curl -sL https://huggingface.co/datasets/USER/gazegene-wds/resolve/main/train/gazegene-train-{000000..000099}.tar" \
    --val_dataset_url "pipe:curl -sL https://huggingface.co/datasets/USER/gazegene-wds/resolve/main/val/gazegene-val-{000000..000019}.tar" \
    --epochs 30
```

### 4. Create WebDataset shards

```bash
python -m RayNet.webdataset_utils create_shards \
    --data_dir /path/to/GazeGene_FaceCrops \
    --output_dir ./shards/train \
    --split train \
    --subject_start 1 --subject_end 46

python -m RayNet.webdataset_utils push \
    --shard_dir ./shards/train \
    --repo_id YOUR_USERNAME/gazegene-wds \
    --split train
```

## Project Structure

```
RayNet/
├── backbone/                   # RepNeXt backbone (M0-M5)
│   ├── repnext.py
│   ├── repnext_utils.py
│   └── se_block.py
├── RayNet/                     # Core module
│   ├── raynet.py               # Main model
│   ├── panet.py                # Path Aggregation Network
│   ├── coordatt.py             # Coordinate Attention
│   ├── heads.py                # Landmark + Gaze heads
│   ├── losses.py               # Landmark + Angular losses
│   ├── multiview_loss.py       # Multi-view consistency losses
│   ├── normalization.py        # Zhang et al. 2018 normalization
│   ├── kappa.py                # Kappa angle handling
│   ├── geometry.py             # Pupil diameter, gaze-to-screen
│   ├── dataset.py              # GazeGeneDataset + samplers
│   ├── train.py                # Training script
│   ├── webdataset_utils.py     # WebDataset shard creation + streaming
│   ├── streaming/              # MosaicML Streaming + MinIO integration
│   │   ├── convert_to_mds.py   # Convert dataset to MDS format
│   │   ├── dataset.py           # StreamingGazeGeneDataset
│   │   └── minio_utils.py       # MinIO upload + configuration
│   ├── EyeFLAME/               # Experimental FLAME-based model
│   ├── head_pose/              # Head pose estimation module
│   └── iris_mesh/              # Iris mesh regression module
├── deploy/                     # Docker Compose for MinIO
│   ├── docker-compose.yml
│   ├── .env.example
│   ├── setup_minio.sh
│   └── README.md
├── notebooks/
│   └── train_colab_a100.ipynb  # Colab A100 training notebook
├── docs/wiki/                  # This wiki
└── requirements.txt
```

## Performance Targets

| Metric | Baseline (ResNet-18) | RayNet v2 Target |
|--------|---------------------|------------------|
| Iris 2D error (px) | 1.84 | < 1.3 |
| Optical axis error (deg) | 4.98 | < 4.0 |
| Eyeball 3D error (cm) | 0.11 | < 0.09 |
| Parameters | 11.7 M | 7.8 M |
| Latency | ~15 ms | < 10 ms |

## Wiki Pages

| Page | Contents |
|------|----------|
| [[Architecture]] | Model architecture, backbone, PANet, task heads, tensor shapes |
| [[Dataset]] | GazeGene format, pickle files, data loading, splits |
| [[Normalization]] | Zhang et al. 2018 per-frame normalization pipeline |
| [[Training Guide]] | Full training instructions, phases, CLI, hardware profiles |
| [[Loss Functions]] | All loss functions with formulas and weighting |
| [[Multi-View Consistency]] | Reprojection loss, triangulation masking, geometry |
| [[WebDataset Streaming]] | Shard creation, HF Hub upload, streaming dataloaders |
| [[MosaicML Streaming]] | MDS shards, MinIO deployment, high-performance streaming |
| [[Geometry and Kappa]] | Kappa angles, pupil diameter, gaze-to-screen projection |
| [[API Reference]] | Function signatures for all public modules |
