# RayNet v2 — Stereo Gaze Estimation, Iris Landmark Detection & Pupillometry

A two-task deep learning model for gaze estimation and iris/pupil landmark detection on the [GazeGene](https://github.com/gazegene) dataset. Designed for real-time behavioural analysis including cognitive load measurement via pupillometry.

## Architecture

```
Input: Normalized eye crop (3 × 224 × 224)
  │
  ▼
RepNeXt-M3 Backbone (4 stages, 7.8M params)
  ├── C1: stride=4  → (64, 56, 56)
  ├── C2: stride=8  → (128, 28, 28)
  ├── C3: stride=16 → (256, 14, 14)
  └── C4: stride=32 → (512, 7, 7)
  │
  ▼
PANet Neck (YOLOv8-style multi-scale fusion)
  ├── P2: (256, 56, 56)  ← landmarks
  ├── P3: (256, 28, 28)
  ├── P4: (256, 14, 14)
  └── P5: (256, 7, 7)    ← gaze
  │
  ├──────────────────────────────┐
  ▼                              ▼
CoordinateAttention(P2)     CoordinateAttention(P5)
  │                              │
  ▼                              ▼
Iris/Pupil Landmark Head    Optical Axis Head
  Heatmap + Soft-Argmax       GAP → FC → pitch/yaw
  14 landmarks (px)           → unit 3D vector
```

### Two Core Tasks

| Task | Output | Downstream Use |
|------|--------|----------------|
| **A: Iris/Pupil Landmarks** | 14 heatmap-based landmarks (10 iris + 4 pupil) | Pupil diameter (mm), depth (mm), screen projection |
| **B: Optical Axis** | Unit 3D gaze direction in normalized space | Gaze ray, screen coordinate, visual axis (with kappa) |

### Removed Tasks (vs v1)
- **Head pose**: handled by pretrained 6DRepNet (separate model)
- **Gaze depth**: replaced by geometric depth from iris ellipse — more accurate, no learning needed
- **Gaze point**: derived analytically from optical axis + depth

## Key Design Decisions

### Normalization (Zhang et al. 2018)
Every frame is warped to a virtual camera at canonical distance (600mm) and focal length (960px). This removes depth ambiguity — the network only needs to solve direction, not position. Normalization is per-frame using current head pose.

### Correct Training Target
The network predicts the **optical axis** (eyeball center → pupil center), NOT the head gaze vector or visual axis. The visual axis requires subject-specific kappa angles, which are unavailable at inference without calibration.

### Kappa Handling
- **Training**: use GT kappa from GazeGene (with roll zeroed)
- **Inference (zero-cal)**: use population mean kappa (yaw=4°, pitch=1°)
- **Inference (calibrated)**: use per-person kappa from 5-point calibration

### Heatmap + Soft-Argmax Landmarks
Replaces direct coordinate regression with spatial heatmaps + differentiable soft-argmax + learned offset refinement. This is the key improvement for subpixel landmark accuracy.

## Training

### Progressive 3-Phase Schedule

| Phase | Epochs | λ_landmark | λ_gaze | LR | Description |
|-------|--------|-----------|--------|-----|-------------|
| 1 | 1-5 | 1.0 | 0.0 | 1e-3 | Landmark warmup |
| 2 | 6-15 | 1.0 | 0.3 | 5e-4 | Introduce gaze |
| 3 | 16-30 | 0.5 | 0.5 | 1e-4 | Balanced fine-tuning |

### Usage

```bash
python RayNet/train.py \
  --data_dir /path/to/gazegene \
  --backbone repnext_m3 \
  --weight_path ./repnext_m3_pretrained.pt \
  --batch_size 512 \
  --epochs 30 \
  --samples_per_subject 200
```

### Dataset Split
- **Train**: subjects 1–46 (up to 200 samples/subject)
- **Val**: subjects 47–56

## Target Metrics

Benchmarked against GazeGene's ResNet-18 baseline (Table 5 in paper):

| Metric | GazeGene ResNet-18 | RayNet v2 Target |
|--------|-------------------|-----------------|
| Iris 2D (px) | 1.84 | < 1.3 |
| Optical axis (°) | 4.98 | < 4.0 |
| Eyeball 3D (cm) | 0.11 | < 0.09 |
| Pupil 3D (cm) | 0.15 | < 0.12 |
| Parameters (M) | 11.7 | 7.8 |
| Latency target (ms) | ~15 | < 10 (edge) |

## Post-Processing (Geometric, No Learning)

### Metric Pupil Diameter
Once accurate iris landmarks are available, pupil diameter in mm is computed via projective geometry using the iris as a known-size ruler:

```
Z_depth = focal_length × iris_radius_mm / apparent_iris_radius_px
pupil_mm = 2 × apparent_pupil_radius_px × Z_depth / focal_length
```

At target landmark accuracy (<1.3px), this achieves ~0.03mm precision — comparable to desktop eye trackers for cognitive load measurement.

### Screen Gaze Point
Derived from optical axis + geometric depth via ray-plane intersection. No separate learned head needed.

## Project Structure

```
RayNet/
├── backbone/              # RepNeXt backbone (m0-m5 variants)
│   ├── repnext.py
│   ├── repnext_utils.py
│   └── se_block.py
├── RayNet/                # Core model
│   ├── raynet.py          # Main model (backbone + PANet + heads)
│   ├── panet.py           # Path Aggregation Network
│   ├── coordatt.py        # Coordinate Attention module
│   ├── heads.py           # IrisPupilLandmarkHead + OpticalAxisHead
│   ├── losses.py          # Landmark + angular loss functions
│   ├── dataset.py         # GazeGene loader with normalization
│   ├── normalization.py   # Zhang 2018 image normalization
│   ├── kappa.py           # Kappa angle handling
│   ├── geometry.py        # Metric pupil diameter & gaze-to-screen
│   ├── train.py           # Training script (3-phase progressive)
│   └── utils.py           # Rotation utilities
├── gaze_estimation/       # ARGaze gaze estimation module
├── sixdrepnet/            # 6DRepNet head pose (pretrained)
├── export_onnx.py         # ONNX export
├── requirements.txt
└── README.md
```

## Future Work

1. **Multi-view consistency loss** — cross-view reprojection using all 9 GazeGene cameras (self-supervised)
2. **Stereo teacher + knowledge distillation** — multi-view teacher → monocular student (requires mono < 5° first)
3. **5-point calibration** — lightweight affine correction for per-person kappa and camera geometry
4. **Temporal pupil tracking** — Hampel outlier rejection + Savitzky-Golay smoothing for cognitive load index
