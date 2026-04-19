# RayNet v5

RayNet v5 is a real-time multi-task gaze estimation system for the [GazeGene](https://github.com/gazegene) multi-camera dataset. From a single 224×224 face crop plus a face bounding box, it jointly predicts **iris/pupil landmarks**, **3D eyeball geometry** (eyeball + pupil centers → optical axis), and **head pose** (6D rotation + 3D translation), with multi-view supervision across 9 synchronized cameras.

## Key Features

| Feature | Details |
|---------|---------|
| Architecture | Quad-M1: shared stem + 3 face branches + dedicated eye-crop gaze backbone |
| Params | ~17M total |
| Input | (3, 224, 224) face crop + (3,) face bbox `(x_p, y_p, L_x)` |
| Task A | 14 iris/pupil landmarks via U-Net decoder with attention gates, 56×56 heatmaps |
| Task B | Explicit 3D eyeball geometry: eyeball center + pupil center → optical axis |
| Task C | 9D head pose: 6D rotation (Gram-Schmidt) + 3D translation |
| Eye crop | 112×112 landmark-guided differentiable crop (`F.affine_grid`/`F.grid_sample`) feeds a private full RepNeXt-M1 |
| MAGE | BoxEncoder provides gaze origin from face bbox — no MediaPipe-468 at inference |
| Multi-view | Geometry-conditioned CrossViewAttention + ray consistency losses |
| Fusion | Zero-init residual `GazeFusionBlock(eye, pose, bbox)` with eye as anchor |
| Training | Staged curriculum with `freeze_face` in Stage 2 P1/P2 and CosineAnnealingLR |
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
    --pose_backbone_weight_path /path/to/repnext_m1_distill_300e.pth \
    --profile default
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
    --pose_backbone_weight_path /path/to/repnext_m1_distill_300e.pth \
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
│   ├── raynet_v5.py            # Quad-M1 model (shared stem + 3 branches + eye backbone)
│   ├── eye_crop.py             # Differentiable 112×112 landmark-guided eye crop
│   ├── coordatt.py             # Coordinate Attention
│   ├── losses.py               # Landmark + angular + 3D structure + pose losses
│   ├── multiview_loss.py       # Gaze ray + landmark shape consistency
│   ├── kappa.py                # Kappa angle handling
│   ├── geometry.py             # Pupil diameter, gaze-to-screen
│   ├── dataset.py              # GazeGeneDataset + samplers (local)
│   ├── normalization.py        # Easy-Norm (MAGE) (inference only)
│   ├── streaming/              # MosaicML Streaming + MinIO integration
│   ├── hardware_profiles.py    # Profiles + Accelerator (find_unused_parameters=True)
│   ├── train.py                # Staged training script + freeze_face helper
│   └── inference.py            # Inference + visualization
├── deploy/
├── docs/wiki/
├── docs/experiments/           # Per-run training logs + analyses
└── requirements.txt
```

## Version History

| Version | Key Changes |
|---------|-------------|
| v3 | Single backbone (M3), 448×448 input, no pose |
| v4 | 224×224 input, CameraEmbedding, LandmarkGazeBridge, PANet neck |
| v4.1 | Dual backbone (M3+M1), implicit PoseEncoder, 9D pose, 3-stage training |
| v5 (Triple-M1) | Shared stem + 3 RepNeXt-M1 branches, explicit 3D eyeball geometry, MAGE BoxEncoder, bridges (landmark x-attn + pose SHMA). Gaze shared the 14×14 feature map with landmark/pose. |
| **v5 (Quad-M1, eye-crop gaze)** | **Gaze branch owns a full RepNeXt-M1 fed by a 112×112 landmark-guided eye crop. Bridges deleted. Stage 2 freezes the face path so gaze trains on a stable predicted-landmark distribution. See the [2026-04 experiments](#key-experiments) that drove this pivot.** |

## Key Experiments

See [`docs/experiments/README.md`](../experiments/README.md) for per-run analyses.

| Run | Stage | Key result |
|-----|-------|-----------|
| [`raynet_v5_500_samples_per_subject`](../experiments/raynet_v5_500_samples_per_subject/) | S1 baseline (15 ep, kaggle_t4x2) | `val_landmark_px` 7.92 → **2.64** (best E14); **-33%** vs 200 samples/subject. Gaze disabled (`lam_gaze=0`). |
| [`raynet_v5_S2_fork_500_samples_per_subject`](../experiments/raynet_v5_S2_fork_500_samples_per_subject/) | S2 fork of above (Triple-M1 gaze) | `val_angular` floored at **~42°** across 8 epochs (P1 E5 dip to 26° was a transient cosine-LR outlier; P2 settled at 41-44°). Diagnosed: gaze bottlenecked by stride-16 shared feature map. → Pivot to Quad-M1 eye-crop architecture. |

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
| [[Architecture]] | Quad-M1, shared stem, branch encoders, eye crop + fusion, tensor shapes |
| [[Dataset]] | GazeGene format, pickle files, data loading, MDS shard schema |
| [[Training Guide]] | Staged training, freeze_face curriculum, hardware profiles, cross-stage forking |
| [[Loss Functions]] | Landmark, gaze, 3D eyeball structure, pose, multi-view consistency |
| [[Normalization]] | Easy-Norm (MAGE), split pipeline, coordinate spaces |
| [[Multi-View Consistency]] | Gaze ray + landmark shape consistency across 9 cameras |
| [[MosaicML Streaming]] | MDS shards, MinIO deployment, streaming dataloaders |
| [[Geometry and Kappa]] | Kappa angles, Intrinsic Delta method, pupil diameter, gaze-to-screen |
| [[API Reference]] | Function signatures for all public modules |
