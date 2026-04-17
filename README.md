# RayNet v5 — Multi-Task Gaze Estimation with Explicit 3D Eyeball Geometry

Multi-task deep learning for gaze estimation, iris/pupil landmarks, and head pose on the [GazeGene](https://github.com/gazegene) dataset. Trained with multi-view geometric supervision.

## Architecture (v5 Triple-M1)

```
Input: (3, 224, 224) GazeGene face crop + (3,) face bbox (x_p, y_p, L_x)
  │
  ▼
SharedStem  (RepNeXt-M1 stem + stages[0..1])    48→96ch, 28x28   ~1.5M
  │
  ├──────────────┬───────────────┬───────────────┐
  ▼              ▼               ▼
Landmark Branch  Gaze Branch     Pose Branch   (each: RepNeXt-M1 stages[2..3])
  U-Net decoder  Eyeball center  6D rotation +
  + attention    + pupil center  3D translation
  gates          → optical axis
  14 pts @56x56                  (gradient-isolated)
        │              │               │
        └──────────────┴──────────── cross-attention / pose-modulation
                       │
                    MAGE BoxEncoder + FusionBlock (face bbox)
```

- Backbone: **RepNeXt-M1** (4.8M per instance), shared stem + 3 dedicated branches.
- Total: **~17M params**.
- Each task branch is extended separately to prevent gradient conflict.
- Zero-init bridges active from epoch 1 (no cold-start).
- Replaces v4 PANet / LandmarkGazeBridge / dual backbone.

## Key Design

- **Explicit 3D eyeball geometry**: predict `eyeball_center_3d` + `pupil_center_3d`, derive `optical_axis = normalize(pupil − eyeball)`. Supervised with L1 on centers + angular loss.
- **MAGE integration**: BoxEncoder consumes the face bounding box `(x_p, y_p, L_x)` derived from the Intrinsic Delta method — no MediaPipe-468 dependency at inference.
- **Multi-view consistency**: 9-camera GazeGene batches; CrossViewAttention + PoseEncoder fusion at train + val.
- **Gradient isolation**: pose branch uses a detached copy of the shared stem; each task owns its high-level features.

## Data

Two modes:

- **Local**: `--data_dir /path/to/GazeGene_FaceCrops`
- **MDS streaming** (MosaicML + MinIO/S3): `--mds_streaming --mds_train s3://gazegene/train --mds_val s3://gazegene/val`

Convert local GazeGene → MDS shards:

```bash
python -m RayNet.streaming.convert_to_mds \
    --data_dir /path/to/GazeGene_FaceCrops \
    --output_dir ./mds_shards/train --split train
```

## Training

Staged schedule (see `RayNet/train.py:STAGE_CONFIGS`): landmark + pose baseline → add gaze → full pipeline with landmark/pose bridges active.

```bash
python -m RayNet.train \
  --mds_streaming \
  --mds_train s3://gazegene/train \
  --mds_val   s3://gazegene/val \
  --core_backbone_weight_path ./ptrained_models/repnext_m1_distill_300e.pth \
  --profile a100 \
  --stage 3 \
  --epochs 30 \
  --samples_per_subject 200
```

Stages are `--stage 1|2|3`. All stages use the same v5 model; only loss weights and active bridges change.

## Project Structure

```
RayNet/
├── backbone/                  # RepNeXt variants
├── RayNet/
│   ├── raynet_v5.py           # Triple-M1 model (shared stem + 3 branches + MAGE)
│   ├── coordatt.py            # Coordinate Attention
│   ├── multiview_loss.py      # Cross-view consistency
│   ├── losses.py              # Landmark + angular + geometry + pose losses
│   ├── dataset.py             # GazeGene loader (local)
│   ├── streaming/             # MosaicML MDS streaming + MinIO
│   ├── normalization.py       # Zhang 2018 image normalization
│   ├── train.py               # Staged training
│   └── inference.py           # Inference + visualization
├── docs/                      # Experiments, wiki, figures
└── deploy/                    # ONNX/TensorRT export
```

## Inference

Run `RayNet/inference.py` with a trained checkpoint to reproduce visualizations under `docs/`. The script is preserved through cleanup and exercises the same v5 forward path as training.

## Target Metrics (GazeGene benchmark)

| Metric | GazeGene ResNet-18 | RayNet v5 Target |
|--------|-------------------|------------------|
| Iris 2D (px) | 1.84 | < 1.3 |
| Optical axis (°) | 4.98 | < 4.0 |
| Eyeball 3D (cm) | 0.11 | < 0.09 |
| Pupil 3D (cm) | 0.15 | < 0.12 |
| Parameters (M) | 11.7 | ~17.1 |
