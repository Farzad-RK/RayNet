# RayNet v5 — Multi-Task Gaze Estimation with Explicit 3D Eyeball Geometry

Multi-task deep learning for gaze estimation, iris/pupil landmarks, and head pose on the [GazeGene](https://github.com/gazegene) dataset. Trained with multi-view geometric supervision.

## Architecture (v5 Quad-M1, eye-crop gaze)

```
Input: (3, 224, 224) GazeGene face crop + (3,) face bbox (x_p, y_p, L_x)
  │
  ▼
SharedStem  (RepNeXt-M1 stem + stages[0..1])    48→96ch, 28x28   ~0.21M
  │
  ├──────────────┬───────────────┐
  ▼              ▼
Landmark Branch  Pose Branch     (each: RepNeXt-M1 stages[2..3])
  U-Net decoder  6D rotation +
  + attention    3D translation
  gates          (gradient-isolated: reads s1.detach())
  14 pts @56×56
        │
        │  predicted landmarks (×4 to pixel space, .detach())
        ▼
  EyeCropModule (112×112, 25% pad, min half-size 24px)
        │
        ▼
  EyeBackbone  (private full RepNeXt-M1, stem + s0..s3)       ~5M
        │
        ▼
  GazeFusionBlock(eye, pose, box)   zero-init residual, eye as anchor
        │
        ▼
  Eyeball center + pupil center → optical_axis = normalize(pupil − eyeball)
                                               ▲
                                               │
                                     MAGE BoxEncoder (face bbox)
```

- Backbone: **RepNeXt-M1** (~4.8M per instance); shared stem + 3 face branches + 1 private eye backbone.
- Total: **~17M params** (SharedStem 0.21M, LandmarkBranch 6.18M, PoseBranch 4.45M, GazeBranch 5.04M, CrossView+Cam 1.07M).
- Gaze branch now has its **own** full M1 fed by a landmark-guided 112×112 eye crop — it does not share the face's 14×14 feature map anymore.
- Landmarks feeding the crop are always `.detach()`-ed; during Stage 2 P1/P2 the face path is frozen so the crop input distribution matches inference.

## Key Design

- **Eye-crop gaze**: the ~42° `val_angular` ceiling in the old Triple-M1 design came from the iris occupying 2-3 cells of the stride-16 face feature map. A differentiable landmark-guided crop + dedicated M1 puts the iris back in a native 28×28 stem map. Driving experiments: `docs/experiments/raynet_v5_500_samples_per_subject/` (S1 baseline, `val_landmark_px` 2.64) and `docs/experiments/raynet_v5_S2_fork_500_samples_per_subject/` (showed the 42° ceiling).
- **Explicit 3D eyeball geometry**: predict `eyeball_center` + `pupil_center`, derive `optical_axis = normalize(pupil − eyeball)`. Supervised with L1 on centers + angular loss.
- **MAGE integration**: BoxEncoder consumes the face bounding box `(x_p, y_p, L_x)` derived from the Intrinsic Delta method — no MediaPipe-468 dependency at inference.
- **Multi-view consistency**: 9-camera GazeGene batches; CrossViewAttention + CameraEmbedding fuse across views, gaze-ray + landmark-shape consistency losses supervise agreement.
- **Gradient isolation**: pose branch reads `s1.detach()`; landmarks into the eye crop are `.detach()`-ed so gaze loss never flows back through the face path.
- **Freeze-face curriculum (Stage 2 P1/P2)**: `shared_stem + landmark_branch + pose_branch` held at `.eval()` + `requires_grad=False` (BN stats frozen) while only the eye-crop gaze branch trains; released in P3 for joint fine-tuning.

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

Staged schedule (see `RayNet/train.py:STAGE_CONFIGS`):

1. **Stage 1** — landmark + pose baseline (gaze disabled, `val_angular` ~42° throughout). Recommended recipe: 500 samples/subject, 15 epochs, `kaggle_t4x2`; drives `val_landmark_px` to ≈ 2.64 (see `docs/experiments/raynet_v5_500_samples_per_subject/`).
2. **Stage 2** — eye-crop gaze on top of the frozen face path. P1 gaze warmup (lr 3e-4, face frozen, single-view) → P2 + multi-view + geometric angular (lr 1e-4, face frozen) → P3 joint fine-tuning (lr 5e-5, face unfrozen). The freeze-face design was chosen after `docs/experiments/raynet_v5_S2_fork_500_samples_per_subject/` showed the old Triple-M1 Stage 2 flooring at ~42°.
3. **Stage 3** — full pipeline with optional bridges + MAGE box encoder (retained for comparison experiments).

```bash
# Stage 1 (landmark + pose baseline)
python -m RayNet.train \
  --mds_streaming \
  --mds_train s3://gazegene/train \
  --mds_val   s3://gazegene/val \
  --core_backbone_weight_path ./ptrained_models/repnext_m1_distill_300e.pth \
  --profile kaggle_t4x2 \
  --stage 1 \
  --samples_per_subject 500

# Stage 2 (forks from a Stage 1 checkpoint)
python -m RayNet.train \
  --mds_streaming --mds_train s3://gazegene/train --mds_val s3://gazegene/val \
  --core_backbone_weight_path ./ptrained_models/repnext_m1_distill_300e.pth \
  --profile kaggle_t4x2 --stage 2 \
  --fork_from s3://raynet-checkpoints/checkpoints/<stage1_run_id>/best_model.pt \
  --run_id <new_stage2_run_id>
```

Stages are `--stage 1|2|3`. All stages use the same v5 Quad-M1 model; only loss weights, the `freeze_face` flag, and (optionally) the active bridges change. Multi-GPU launches go through `accelerate launch --multi_gpu`; `build_accelerator()` sets `find_unused_parameters=True` so DDP tolerates the frozen face path in Stage 2 P1/P2.

## Project Structure

```
RayNet/
├── backbone/                  # RepNeXt variants
├── RayNet/
│   ├── raynet_v5.py           # Quad-M1 model (shared stem + 3 branches + eye backbone)
│   ├── eye_crop.py            # Differentiable 112×112 landmark-guided eye crop
│   ├── coordatt.py            # Coordinate Attention
│   ├── multiview_loss.py      # Cross-view consistency
│   ├── losses.py              # Landmark + angular + geometry + pose losses
│   ├── dataset.py             # GazeGene loader (local)
│   ├── streaming/             # MosaicML MDS streaming + MinIO
│   ├── normalization.py       # Easy-Norm (MAGE) image normalization
│   ├── hardware_profiles.py   # Profiles + Accelerator (find_unused_parameters=True)
│   ├── train.py               # Staged training + set_face_frozen helper
│   └── inference.py           # Inference + visualization
├── docs/                      # Experiments, wiki, figures
└── deploy/                    # ONNX/TensorRT export
```

## Inference

Run `RayNet/inference.py` with a trained checkpoint to reproduce visualizations under `docs/`. The script is preserved through cleanup and exercises the same v5 forward path as training.

## Target Metrics (GazeGene benchmark)

| Metric | GazeGene ResNet-18 | RayNet v5 Target | S1 500 spc (reference) |
|--------|-------------------|------------------|------------------------|
| Iris 2D (px) | 1.84 | < 1.3 | **2.64** (best E14, 15 ep) |
| Optical axis (°) | 4.98 | < 4.0 | 42.5° (gaze disabled in S1) |
| Eyeball 3D (cm) | 0.11 | < 0.09 | — (S2+ only) |
| Pupil 3D (cm) | 0.15 | < 0.12 | — (S2+ only) |
| Parameters (M) | 11.7 | ~17 | ~17 |

The "S1 500 spc (reference)" column is from `docs/experiments/raynet_v5_500_samples_per_subject/` (Stage 1, `--samples_per_subject 500`, 15 epochs, `kaggle_t4x2`). Stage 2 (eye-crop gaze) runs from there.
