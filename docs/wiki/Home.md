# RayNet v4.1

RayNet v4.1 is a real-time multi-task gaze estimation system built for the [GazeGene](https://github.com/gazegene) multi-camera dataset. It jointly predicts **iris/pupil landmarks**, the **optical axis**, and **implicit head pose** from a single 224x224 face crop, using two gradient-isolated backbones and geometry-conditioned cross-view attention across 9 synchronized cameras.

## Key Features

| Feature | Details |
|---------|---------|
| Main backbone | RepNeXt-M3 (7.8M params) — landmarks + gaze |
| Pose backbone | RepNeXt-M1 (4.8M params) — implicit head pose (gradient-isolated) |
| Input | 224x224 GazeGene face crop |
| Task A | 14 landmarks (10 iris + 4 pupil) via soft-argmax heatmaps on P2 |
| Task B | Optical axis unit vector (pitch + yaw) on P5 |
| Aux | 9D head pose: 6D rotation (Gram-Schmidt) + 3D translation (tanh/exp) |
| Multi-view | Geometry-conditioned CrossViewAttention + ray consistency losses |
| Bridge | LandmarkGazeBridge: P5 gaze attends to P2 landmarks (stage 3 only) |
| Training | 3-stage curriculum with per-phase loss weights and gradient clipping |
| Hardware | 7 profiles (default, T4, L4, A10G, V100, A100, H100) |
| Streaming | MDS shards via MosaicML Streaming + MinIO |

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/RayNet.git
cd RayNet
pip install -r requirements.txt
```

### 2. Train (Stage 1 — Landmark + Pose baseline)

```bash
python -m RayNet.train \
    --data_dir /path/to/GazeGene_FaceCrops \
    --stage 1 \
    --core_backbone repnext_m3 \
    --pose_backbone repnext_m1 \
    --core_backbone_weight_path /path/to/repnext_m3_distill_300e.pth \
    --pose_backbone_weight_path /path/to/repnext_m1_distill_300e.pth \
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
    --mds_val s3://gazegene/val \
    --stage 2 \
    --profile a100 \
    --core_backbone_weight_path /path/to/repnext_m3_distill_300e.pth \
    --pose_backbone_weight_path /path/to/repnext_m1_distill_300e.pth \
    --ckpt_bucket raynet-checkpoints \
    --minio_endpoint http://YOUR_SERVER_IP:9000
```

### 4. Create MDS shards

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
├── backbone/                   # RepNeXt backbone (M0-M5)
│   ├── repnext.py
│   ├── repnext_utils.py
│   └── se_block.py
├── RayNet/                     # Core module
│   ├── raynet.py               # Main model (RayNet, PoseEncoder, CrossViewAttention, etc.)
│   ├── panet.py                # Path Aggregation Network
│   ├── coordatt.py             # Coordinate Attention
│   ├── heads.py                # IrisPupilLandmarkHead + OpticalAxisHead
│   ├── losses.py               # All losses (landmark, gaze, geodesic, translation, ray)
│   ├── multiview_loss.py       # Gaze ray + landmark shape consistency
│   ├── kappa.py                # Kappa angle handling
│   ├── geometry.py             # Pupil diameter, gaze-to-screen
│   ├── dataset.py              # GazeGeneDataset + samplers
│   ├── train.py                # Staged training script
│   ├── streaming/              # MosaicML Streaming + MinIO integration
│   │   ├── convert_to_mds.py   # Convert dataset to MDS format
│   │   ├── dataset.py          # StreamingGazeGeneDataset
│   │   ├── minio_utils.py      # MinIO upload + configuration
│   │   └── checkpoint.py       # MinIO checkpoint manager
│   └── normalization.py        # Zhang et al. 2018 normalization (inference only)
├── deploy/                     # Docker Compose for MinIO
├── notebooks/
├── docs/wiki/                  # This wiki
└── requirements.txt
```

## Version History

| Version | Key Changes |
|---------|-------------|
| v3 | Single backbone (M3), 448x448 input, no pose, no bridge |
| v4 | 224x224 input, CameraEmbedding, LandmarkGazeBridge, ray-to-target loss |
| **v4.1** | **Dual backbone (M3+M1), implicit PoseEncoder (MAGE-style), 9D pose (6D rotation + 3D translation), geodesic + translation loss, per-phase gradient clipping, 3-stage training** |

## Performance Targets

| Metric | Baseline (ResNet-18) | RayNet v4.1 Target |
|--------|---------------------|------------------|
| Iris 2D error (px) | 1.84 | < 1.3 |
| Optical axis error (deg) | 4.98 | < 4.0 |
| Eyeball 3D error (cm) | 0.11 | < 0.09 |
| Parameters | 11.7M | ~15.6M (dual backbone) |
| Latency | ~15 ms | < 10 ms (edge, single backbone) |

## Wiki Pages

| Page | Contents |
|------|----------|
| [[Architecture]] | Dual backbone, PANet, PoseEncoder, 9D pose, CrossViewAttention, task heads, tensor shapes |
| [[Dataset]] | GazeGene format, pickle files, data loading, MDS shard schema |
| [[Training Guide]] | 3-stage training, phases, loss weights, gradient clipping, hardware profiles, CLI |
| [[Loss Functions]] | Landmark, gaze, geodesic rotation, translation, ray-to-target, multi-view consistency |
| [[Multi-View Consistency]] | Gaze ray consistency, landmark shape consistency, camera extrinsics |
| [[WebDataset Streaming]] | Shard creation, HF Hub upload, streaming dataloaders |
| [[MosaicML Streaming]] | MDS shards, MinIO deployment, high-performance streaming |
| [[Geometry and Kappa]] | Kappa angles, pupil diameter, gaze-to-screen projection |
| [[API Reference]] | Function signatures for all public modules |
