# RayNet v5 — Multi-Task Gaze Estimation with Explicit 3D Eyeball Geometry

Multi-task deep learning for gaze estimation, iris/pupil landmarks, head pose, and **anatomical eye-region segmentation** on the [GazeGene](https://github.com/gazegene) dataset. Trained with multi-view geometric supervision and parallel MTL from epoch 1.

## Architecture (v5 Triple-M1, FPANet landmark + AERI gaze)

```
Input: (3, 224, 224) GazeGene face crop + (3,) face bbox (x_p, y_p, L_x)
  │
  ▼
SharedStem  (RepNeXt-M1 stem + stages[0..1])    48→96ch, 28x28   ~0.21M
  │     (landmark-owned: only landmark loss backprops here)
  │
  ├───────────────┬───────────────────┬────────────────────┐
  ▼               ▼ (s1.detach)       ▼ (s1.detach)
Landmark Branch   Pose Branch         Gaze Branch
(M1 s2+s3)        (M1 s2+s3)          (M1 s2+s3)
PANet (P2..P5)    CoordAtt+pool       ┌────────────────────┐
→ P2 refine +     ⊕ BoxEncoder(bbox)  │ FPNAERIHead (PANet) │
  heatmap+offset  (zero-init residual)│ → iris_logits       │
14 pts @56×56     → pose_feat         │ → eyeball_logits    │
                  → 6D rot + 3D t     │   (both @ 56×56)    │
                                      └────────────────────┘
                                                 │
                                       eyeball mask gates
                                       the 7×7 gaze map:
                                       gaze_s3 *= (GLOBAL_FLOOR +
                                         (1−GLOBAL_FLOOR) · pool₇(M))
                                                 │
                                                 ▼
                                       GazeFusion(gaze, pose_feat)
                                       (zero-init residual)
                                                 │
                                                 ▼
                                       Eyeball + pupil 3D →
                                       optical_axis =
                                         normalize(pupil − eyeball)
```

- Backbone: **RepNeXt-M1** (~4.8M per instance). One shared stem + three independent `s2+s3` branches.
- Total: **~18.7M params** (SharedStem 0.21M, LandmarkBranch 6.21M, PoseBranch 4.69M, GazeBranch 6.47M, CrossView+Cam 1.07M).
- **Triple-M1** refers to the three task-specific branches above a single shared stem — not an eye-crop encoder. There is no pixel-level cropping anywhere in the forward path.

## Key Design

- **PANet multi-scale fusion** for landmark and AERI: each branch carries its own `FeaturePyramidNetwork` that fuses all four backbone strides (P2..P5) to a uniform `fpn_ch` width via 1×1 lateral + top-down + bottom-up passes, all Conv-BN-SiLU. The landmark and AERI heads consume the fused P2 (`fpn_ch` × 56 × 56). Replaces the prior single-stage U-Net decoders that plateaued above 1 px on landmark sub-pixel accuracy.
- **AERI (Anatomical Eye Region Isolation)**: the gaze branch's `FPNAERIHead` produces binary iris + eyeball segmentation logits at 56×56 plus a `fpn_ch`-channel `d1` tensor. The predicted eyeball mask is downsampled to 7×7 and used as a soft attention gate on the gaze bottleneck (with `GLOBAL_FLOOR = 0.5`), so the pooled gaze vector is eye-region-dominant without any geometric crop. Masks are baked into the MDS shards from GazeGene anatomy (see `RayNet/streaming/eye_masks.py`).
- **Landmark-owned stem + full gradient isolation**: pose and gaze both read `s1.detach()` (and `s0.detach()` where used as a skip), so gaze/pose gradients never reach the shared low-level encoder. Landmark loss is the sole driver of `SharedStem`.
- **Parallel MTL from epoch 1**: no sequential freeze stages. Landmark + pose + gaze + AERI segmentation + head translation are all active from the first step. Phase transitions adjust loss weights and LR only.
- **Explicit 3D eyeball geometry**: predict `eyeball_center` + `pupil_center`, derive `optical_axis = normalize(pupil − eyeball)` (GazeGene Sec 4.2.2). Supervised with L1 on centers + angular loss on the derived axis.
- **MAGE integration inside PoseBranch**: `BoxEncoder` consumes `(x_p, y_p, L_x)` from the Intrinsic Delta method (see `dataset.py::__getitem__` and `docs/wiki/Geometry-and-Kappa.md`). Fused into pose via a zero-init residual so pose predictions at step 0 come from the CNN feature alone and bbox signal ramps in as the encoder trains.
- **Multi-view consistency**: 9-camera GazeGene batches; `CrossViewAttention` + `CameraEmbedding` fuse across views; gaze-ray + landmark-shape consistency losses supervise agreement (phase 2+).

## Data

Two modes:

- **Local**: `--data_dir /path/to/GazeGene_FaceCrops`
- **MDS streaming** (MosaicML + MinIO/S3): `--mds_streaming --mds_train s3://gazegene/train --mds_val s3://gazegene/val`

Convert local GazeGene → MDS shards (baked-in iris + eyeball masks are written automatically):

```bash
python -m RayNet.streaming.convert_to_mds \
    --data_dir /path/to/GazeGene_FaceCrops \
    --output_dir ./mds_shards/train --split train
```

## Training

Single stage, three phases, all losses active from epoch 1 (see `RayNet/train.py::PHASE_CONFIG`):

| Phase | Epochs | Purpose | LR | Multi-view loss |
|-------|--------|---------|----|-----------------|
| 1 | 1–8   | warmup — all losses active, moderate weights | 5e-4 | off |
| 2 | 9–16  | main — full weights + multi-view consistency | 3e-4 | on |
| 3 | 17–25 | fine-tune — lower LR, gaze emphasis | 1e-4 | on |

Phase transitions carry over optimizer momentum and rebuild only the `CosineAnnealingLR` for the new phase window. Gradient clipping is `max_norm=5.0` in phase 1 (lets the multi-task gradient settle) and `max_norm=2.0` afterwards.

```bash
# Single GPU
python -m RayNet.train \
  --mds_streaming \
  --mds_train s3://gazegene/train \
  --mds_val   s3://gazegene/val \
  --core_backbone_weight_path ./ptrained_models/repnext_m1_distill_300e.pth \
  --profile t4 \
  --samples_per_subject 500 \
  --epochs 25

# Multi-GPU (Kaggle 2× T4 / similar)
accelerate launch --multi_gpu --num_processes 2 \
  -m RayNet.train \
  --mds_streaming --mds_train s3://gazegene/train --mds_val s3://gazegene/val \
  --core_backbone_weight_path ./ptrained_models/repnext_m1_distill_300e.pth \
  --profile kaggle_t4x2 \
  --samples_per_subject 500 \
  --epochs 25 \
  --ckpt_bucket raynet-checkpoints --ckpt_prefix checkpoints
```

No `--stage` flag. No `freeze_face`. Fork/warmstart/resume machinery is preserved for cross-architecture migrations and hyperparameter branching.

## Project Structure

```
RayNet/
├── backbone/                  # RepNeXt variants
├── RayNet/
│   ├── raynet_v5.py           # Triple-M1 model (shared stem + 3 branches + AERI)
│   ├── coordatt.py            # Coordinate Attention
│   ├── multiview_loss.py      # Cross-view consistency
│   ├── losses.py              # Landmark + angular + geometry + pose + seg losses
│   ├── dataset.py             # GazeGene loader (local) + AERI mask rendering
│   ├── streaming/
│   │   ├── eye_masks.py       # AERI iris + eyeball mask renderer
│   │   ├── convert_to_mds.py  # Shard writer (masks baked into shards)
│   │   └── dataset.py         # MosaicML MDS streaming reader
│   ├── normalization.py       # Easy-Norm (MAGE) image normalization
│   ├── hardware_profiles.py   # Profiles + Accelerator
│   ├── train.py               # Parallel MTL training (3 phases, 1 stage)
│   └── inference.py           # Inference + visualization
├── docs/                      # Experiments, wiki, figures
└── deploy/                    # ONNX/TensorRT export
```

## Inference

Run `RayNet/inference.py` with a trained checkpoint to reproduce visualizations under `docs/`. The AERI masks are exposed as `iris_mask_logits` / `eyeball_mask_logits` in the forward dict and can be visualised alongside the landmarks + optical axis.

## Target Metrics (GazeGene benchmark)

| Metric | GazeGene ResNet-18 | RayNet v5 Target |
|--------|-------------------|------------------|
| Iris 2D (px) | 1.84 | < 1.3 |
| Optical axis (°) | 4.98 | < 4.0 |
| Eyeball 3D (cm) | 0.11 | < 0.09 |
| Pupil 3D (cm) | 0.15 | < 0.12 |
| Iris mask IoU | — | > 0.85 (auxiliary) |
| Eyeball mask IoU | — | > 0.90 (auxiliary) |
| Parameters (M) | 11.7 | ~18.7 |

The S1 500-spc reference from the prior Quad-M1 design (`docs/experiments/raynet_v5_500_samples_per_subject/`, `val_landmark_px` = 2.64 at best epoch 14) is the landmark-accuracy baseline to beat under the new architecture.
