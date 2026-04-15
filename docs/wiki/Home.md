# RayNet v5

RayNet v5 is a real-time multi-task gaze estimation system for the [GazeGene](https://github.com/gazegene) multi-camera dataset. From a single 224×224 face crop plus a face bounding box, it jointly predicts **iris/pupil landmarks**, **3D eyeball geometry** (eyeball + pupil centers → optical axis), and **head pose** (6D rotation + 3D translation), with multi-view supervision across 9 synchronized cameras.

## Key Features

| Feature | Details |
|---------|---------|
| Architecture | Triple-M1: shared stem + 3 dedicated RepNeXt-M1 branches |
| Params | ~17.1M total |
| Input | (3, 224, 224) face crop + (3,) face bbox `(x_p, y_p, L_x)` |
| Task A | 14 iris/pupil landmarks via U-Net decoder with attention gates, 56×56 heatmaps |
| Task B | Explicit 3D eyeball geometry: eyeball center + pupil center → optical axis |
| Task C | 9D head pose: 6D rotation (Gram-Schmidt) + 3D translation |
| MAGE | BoxEncoder + FusionBlock provide gaze origin from face bbox — no MediaPipe-468 at inference |
| Multi-view | Geometry-conditioned CrossViewAttention + ray consistency losses |
| Bridges | Landmark cross-attention + pose SHMA modulation (zero-init, active from epoch 1) |
| Training | Staged curriculum with per-phase loss weights and CosineAnnealingLR |
| Streaming | MDS shards via MosaicML Streaming + MinIO |

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/RayNet.git
cd RayNet
pip install -r requirements.txt
```

### 2. Train (local, Stage 1 — landmark + pose baseline)

```bash
python -m RayNet.train \
    --data_dir /path/to/GazeGene_FaceCrops \
    --stage 1 \
    --core_backbone_weight_path /path/to/repnext_m1_distill_300e.pth \
    --profile default \
    --epochs 20
```

### 3. Train on A100 (MDS streaming from MinIO)

```bash
export S3_ENDPOINT_URL=http://YOUR_SERVER_IP:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=your-password

python -m RayNet.train \
    --mds_streaming \
    --mds_train s3://gazegene/train \
    --mds_val   s3://gazegene/val \
    --stage 3 \
    --profile a100 \
    --core_backbone_weight_path /path/to/repnext_m1_distill_300e.pth \
    --ckpt_bucket raynet-checkpoints \
    --minio_endpoint http://YOUR_SERVER_IP:9000
```

### 4. Create MDS shards from local GazeGene

```bash
python -m RayNet.streaming.convert_to_mds \
    --data_dir /path/to/GazeGene_FaceCrops \
    --output_dir ./mds_shards/train \
    --split train \
    --subject_start 1 --subject_end 46
```

## Project Structure

```
RayNet/
├── backbone/                   # RepNeXt-M0..M5
├── RayNet/
│   ├── raynet_v5.py            # Triple-M1 model (shared stem + 3 branches + MAGE)
│   ├── coordatt.py             # Coordinate Attention
│   ├── losses.py               # Landmark + angular + 3D structure + pose losses
│   ├── multiview_loss.py       # Gaze ray + landmark shape consistency
│   ├── kappa.py                # Kappa angle handling
│   ├── geometry.py             # Pupil diameter, gaze-to-screen
│   ├── dataset.py              # GazeGeneDataset + samplers (local)
│   ├── normalization.py        # Zhang et al. 2018 normalization (inference only)
│   ├── streaming/              # MosaicML Streaming + MinIO integration
│   ├── train.py                # Staged training script
│   └── inference.py            # Inference + visualization
├── deploy/
├── docs/wiki/
└── requirements.txt
```

## Version History

| Version | Key Changes |
|---------|-------------|
| v3 | Single backbone (M3), 448×448 input, no pose |
| v4 | 224×224 input, CameraEmbedding, LandmarkGazeBridge, PANet neck |
| v4.1 | Dual backbone (M3+M1), implicit PoseEncoder, 9D pose, 3-stage training |
| **v5** | **Triple-M1 branches, shared stem, explicit 3D eyeball geometry, MAGE BoxEncoder/FusionBlock, no PANet** |

## Performance Targets

| Metric | Baseline (ResNet-18) | RayNet v5 Target |
|--------|---------------------|------------------|
| Iris 2D error (px) | 1.84 | < 1.3 |
| Optical axis error (°) | 4.98 | < 4.0 |
| Eyeball 3D error (cm) | 0.11 | < 0.09 |
| Parameters | 11.7M | ~17.1M |

## Wiki Pages

| Page | Contents |
|------|----------|
| [[Architecture]] | Triple-M1, shared stem, branch encoders, MAGE BoxEncoder, tensor shapes |
| [[Dataset]] | GazeGene format, pickle files, data loading, MDS shard schema |
| [[Training Guide]] | Staged training, phases, loss weights, hardware profiles, CLI |
| [[Loss Functions]] | Landmark, gaze, 3D eyeball structure, pose, multi-view consistency |
| [[Multi-View Consistency]] | Gaze ray + landmark shape consistency across 9 cameras |
| [[MosaicML Streaming]] | MDS shards, MinIO deployment, streaming dataloaders |
| [[Geometry and Kappa]] | Kappa angles, Intrinsic Delta method, pupil diameter, gaze-to-screen |
| [[API Reference]] | Function signatures for all public modules |
