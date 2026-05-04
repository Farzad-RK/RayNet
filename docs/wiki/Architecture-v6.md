# Architecture v6 — Decoupled GazeGene + OpenEDS pipeline

> Status: shipped 2026-05. v6 supersedes v5.x. The v5 pipeline is preserved
> in commit history for ablation reference.

> **v6.1 update (2026-05-04).** Adds the **macro / micro gaze split**: GazeGene
> trains macro gaze (`gaze_C`, head→target) via a new `MacroGazeHead` that fuses
> pose with the predicted 3D eyeball anchor; OpenEDS owns micro gaze (refraction-
> aware visual axis from real iris geometry, computed downstream of the
> segmenter + torsion stages). Also ships the OpenEDS MDS converter parallel
> to the GazeGene one so both pipelines share streaming ergonomics.

> **v6.2.1 hotfix (2026-05-05).** Two latent 448-resolution bugs found while
> investigating the regression in `run_20260504_182102` (val_landmark_px=2.91 px
> vs historical sub-pixel):
> 1. **`raynet_v5.py:GazeBranch.forward`** hardcoded `eye_attn_7 =
>    F.adaptive_avg_pool2d(scheduled_mask, 7)`. The literal `7` is the M1+224
>    stride-32 spatial size; at 448 input `gaze_s3` is 14×14 and the multiply
>    crashes. Fixed: pool to ``gaze_s3.shape[-2:]`` so the gate auto-adapts
>    to any input resolution.
> 2. **`streaming/convert_to_mds.py:image_to_jpeg_bytes`** defaulted to
>    ``resize_to=(224, 224)``. When re-sharding with ``--img_size 448`` the
>    dataset emits 448×448 tensors but the converter silently downsizes the
>    JPEG to 224 — producing a shard with a 224 image alongside ``K_cropped``
>    and ``landmark_coords`` calibrated for 448. Fixed: default ``resize_to=
>    None`` (preserve native).
> 3. **`dataset.py:MASK_SIZE = 56`** hardcoded, so the AERI mask GT was
>    always rendered at 56×56 even when the model's P2 output was 112×112
>    at 448 input. Replaced with ``_mask_size(img_size) = img_size // 4`` so
>    GT masks stay aligned with model outputs at any resolution.
>
> **Reshard required** for v6.2.1 to take effect at 448. Shards written
> with the previous broken converter must be regenerated.

> **v6.2 update (2026-05-05).** Architecture simplification + bridge upgrade:
> - **M3 default backbone** (M1 → M3, embed=64/128/256/512). `--backbone {m1,m3}` CLI flag, M3 default for v6.2.
> - **448-pixel face crops** supported via `--img_size {224,448}` (reshard required for 448).
> - **EyeballRadiusHead** — predicts per-subject globe radius R_s in cm; supervised by GazeGene `subject_label.eyeball_radius`. Anchors the OpenEDS torsion two-sphere model.
> - **High-resolution eye-patch crop** replaces binary AERI masks as the bridge to OpenEDS. Crop covers the eyelid + lashes vicinity (5/14 of face crop ≈ 80px @ 224, 160px @ 448) so the OpenEDS-style segmenter has the same field of view it sees during real-IR training.
> - **Pose translation head removed** — global translation now comes from the predicted `eyeball_center_3d` (Macro-Locator). PoseBranch outputs rotation only.
> - **Tiny RITnet variant** (`build_ritnet_tiny`, ~0.5M params) for the OpenEDS segmenter, plus a **geometric prior channel** helper (`make_geometric_prior_channel`) that turns the projected 3D eyeball centre into a 2D Gaussian ROI seeded into the segmenter's second input channel.

## Why v6

The v5 pipeline trained gaze, segmentation, and pose **jointly** on synthetic GazeGene MetaHuman renderings. Two structural problems showed up at the 14-18° gaze plateau visible in `run_20260430_002809`:

1. **Texture poisoning.** The AERI segmentation head is forced to predict iris/sclera boundaries from RGB pixels. Even when the GT mask is geometric (rendered from `iris_mesh_3D` + the eyeball sphere), the head can only fit it by reading MetaHuman pixel patterns. Real-world IR cornea optics are nothing like UE5 specular reflections, and the gap shows up as cross-domain failure when the trained model hits noisy IR sensors.

2. **Single-source supervision.** Both 3D-anchor and gaze-direction supervision came from the same synthetic source. There was no real-IR signal anywhere in the pipeline, so any artefact specific to the renderer (perfectly-clean iris gradients, shadowless lashes) became a learning shortcut.

v6 splits these into **two encoders trained on disjoint datasets**, joined only at inference time:

- **Macro-Locator** (GazeGene → skeleton geometry). RepNeXt-M3 + landmark/pose/gaze heads. Outputs 3D eyeball centre, pupil centre, iris contour mesh, optical axis, head pose. **No texture supervision.**
- **Foveal Segmenter** (OpenEDS → real-IR boundaries). RITnet-style encoder. Outputs 4-class semantic masks at native 400×640 resolution.
- **Torsion + Temporal** (OpenEDS sequences). Classical iris-polar patch matching + dilated TCN for cyclotorsion, gaze smoothing, blink/saccade classification.

## Top-level diagram

```
                         ┌─ GazeGene synthetic (TriCam {1, 6, 8}) ─┐
   skeleton path         │                                          │
   ──────────────        │   shared_stem (M3)                       │
                         │       ↓                                  │
                         │   landmark_branch ──→ 14 landmarks      │
                         │       ↓                                  │
                         │   pose_branch ─────→ R, t (head pose)   │
                         │       ↓                                  │
                         │   gaze_branch                            │
                         │       ├── eyeball_center_3d              │
                         │       ├── pupil_center_3d                │
                         │       ├── iris_mesh_3d (100×3)           │
                         │       ├── gaze_geom = norm(p − e)        │
                         │       ├── gaze_direct = norm(MLP(pooled))│
                         │       └── gaze_fused = norm(geom+direct) │
                         │                                          │
                         │   CrossViewAttention (TriCam=3 views)   │
                         └──────────────────────────────────────────┘

                                    ▲ inference-only bridge

                         ┌─ OpenEDS real-IR (FovalNet preprocessed) ─┐
   foveal/temporal path  │                                            │
   ───────────────────   │   RITnet-style segmenter                   │
                         │       ↓                                    │
                         │   pupil/iris/sclera masks @ 400×640       │
                         │       ↓                                    │
                         │   classical IrisPolarTorsion              │
                         │       ↓                                    │
                         │   per-frame e_t = [gaze, ellipse, torsion,│
                         │                    pupil_area, ...]       │
                         │       ↓                                    │
                         │   TCN (causal dilated, RF=61 frames)      │
                         │       ↓                                    │
                         │   smoothed_gaze, torsion_residual,        │
                         │   blink_logit, movement_class             │
                         └────────────────────────────────────────────┘
```

## TriCam {1, 6, 8}

The 9-cam GazeGene rig is a 3×3 spatial array (front/middle/back rings × left/centre/right columns). The triplet `{1, 6, 8}` was selected by maximising 3D triangulation area against `docs/camera_info.pkl`:

| Triplet | 3D triangulation area (cm²) | Notes |
|---|---|---|
| **{1, 6, 8}** | **85 406** | Optimal — front-centre + back corners. Wide z-baseline (379 cm), uniform pairwise baselines (433-450 cm). |
| {0, 2, 7} | 83 099 | Mirror — front corners + back centre. |
| {3, 4, 5} | 16 747 | **Degenerate** — all middle ring (z = -162), zero z-spread. Was the v5 default. |

See `RayNet/train.py:TRICAM_IDS` for the constant. Switching is one line in train config.

## T_vec convention

**Units: centimetres**, per the GazeGene CVPR2025 dataset description. The previous suspicion about millimetres was wrong — the rig is a multi-camera mocap-scale layout (~5 m wide) with telephoto cameras (fx ≈ 21 549, FOV ≈ 6.8°).

### Numerical scale fix

`T_cam` magnitudes are O(100–600) cm; `R_cam` entries are bounded [-1, 1]. Concatenating raw values into the `CameraEmbedding`'s `Linear(12, 64)` saturates the post-ReLU on `T_cam` alone. Fix lives in `RayNet/raynet_v5.py:CameraEmbedding`:

```python
T_norm = T_cam / 100.0   # cm → m, brings to O(1) like R_cam
x = torch.cat([R_cam.flatten(1), T_norm], dim=-1)
```

Geometric semantics elsewhere (multiview loss, dataset GT, eyeball-centre reprojection) keep cm.

## Skeleton-only GazeGene supervision

The decoupling rule is enforced at the loss-weight level (no architectural firewall is needed):

```
GazeGene phase weights:
    lam_iris_seg     = 0.0   # foveal seg → OpenEDS only
    lam_eyeball_seg  = 0.0   # foveal seg → OpenEDS only
    lam_lm           > 0     # 14-landmark skeleton (sparse, geometric)
    lam_iris_mesh    > 0     # 100-vertex iris ring (geometric)
    lam_eyeball/pupil > 0    # 3D anchors (geometric)
    lam_pose / lam_trans > 0 # head pose (geometric)
    lam_gaze*        > 0     # gaze readouts (mean-of-two)
```

The AERI head is constructed but receives no gradient on synthetic batches; it acts as a soft attention prior whose weights are inherited from whatever state Phase 1 ends in.

## Macro vs micro gaze (v6.1)

GazeGene exposes two gaze paradigms; v6.1 trains **only the macro one on synthetic data** to keep micro-gaze (foveal-refraction) supervision real-IR.

| Paradigm | Frame | GT field | Trained on | Loss |
|---|---|---|---|---|
| **Macro (head)** | head → target, in CCS, kappa-free | `gaze_C` | **GazeGene** | `macro_gaze_loss` (L1 on unit) |
| **Micro (eye)** | optical axis: eyeball → cornea/pupil | `optic_axis_{L,R}` | (geometric consistency only — not a primary signal) | `geometric_angular_loss` |
| **Micro (visual)** | cornea → fovea, refraction-aware | derived from real IR | **OpenEDS** (downstream) | future — needs torsion stage and per-deployment intrinsics |

The new `MacroGazeHead` (in `RayNet/raynet_v5.py`) fuses two inputs:

```
gaze_macro = MacroGazeHead(pose_feat,             # (B, d_model)
                           eyeball_center_3d.detach())   # (B, 3)
```

`pose_feat` carries the head-pose embedding; the predicted `eyeball_center_3d` provides the geometric origin. Detaching the eyeball anchor on the way in stops macro-gaze gradients from over-riding the metric anchor regression — `eyeball_center_loss` remains the only signal that shapes the eyeball-fc weights.

Why this split:
- The macro/head signal is robust to photorealistic-but-imperfect MetaHuman renderings; head pose extracts well from the synthetic distribution because face-shape variation is high.
- Micro/visual axis supervision needs real corneal optics — UE5 specular highlights are not what a real IR sensor sees through a refracting cornea. Training visual-axis from synthetic textures *is* the texture-poisoning failure mode we want to avoid.

Active in Phase 2/3 at weight `lam_gaze_macro = 1.0` (peer with `lam_gaze` on the fused readout). The optical-axis supervision (geom + direct + visual) keeps its v6 weights — those losses target a *geometric consistency* objective rather than a primary direction signal, so leaving them on does not contradict the macro/micro split.

---

## Mean-of-two gaze fusion

Following 3DGazeNet (Sec 7), `GeometricGazeHead` emits three gaze vectors:

| Readout | Formula | Strength |
|---|---|---|
| `gaze_geom` | `normalize(pupil_3d − eyeball_3d)` | Anchor-grounded; robust on profile views. |
| `gaze_direct` | `normalize(direct_fc(pooled))` | Direct regression; fast warmup, accurate near-frontal. |
| `gaze_fused` | `normalize(gaze_geom + gaze_direct)` | Canonical signal — used as `gaze_vector` everywhere downstream. |

Independent sub-supervisions on `gaze_geom` and `gaze_direct` (lambdas 0.5 each in Phase 2/3) prevent collapse; the fused readout carries the full `lam_gaze` weight.

## Iris-mesh M-target

Implements 3DGazeNet's M-target (their Eq 1-2). New head: `IrisMeshHead` regresses `(B, 100, 3)` iris-contour vertices in CCS. Supervised with vertex L1 (`lam_iris_mesh`, paper λ_v = 0.1) + edge-length L2 (`lam_iris_edge`, paper λ_e = 0.01).

3DGazeNet's ablation (Tab 3) shows M+V outperforming V alone on **all 4** within-dataset benchmarks. GazeGene already provides the GT (`iris_mesh_3D` field, `(2 eyes × 100 verts × 3)`); the dataset loader indexes the active eye. **Reshard required** to land the new GT in MDS (see "Reshard" below).

## Visual-axis (kappa-corrected) supervision

The optical axis is anatomy (eyeball→cornea); the visual axis is the line through the fovea. They differ by a per-subject kappa angle (~±2°). For medical-grade gaze, collapsing them costs ~1° of accuracy.

GazeGene ships per-subject `L_kappa` / `R_kappa` Euler angles in `subject_label.pkl`. `RayNet.kappa.build_R_kappa` converts them to a 3×3 matrix at dataset load. The new loss applies `R_kappa` to `gaze_geom` and supervises against `gt_visual_axis`:

```
visual_pred = normalize(R_kappa @ gaze_geom)
L_visual    = L1(visual_pred, gt_visual_axis)
```

Active in Phase 2+ at weight 0.5.

## OpenEDS pipeline

### Dataset

FovalNet-preprocessed format: `openEDS/openEDS/S_<id>/<frame>.png` (400×640 grayscale IR) paired with `<frame>.npy` (400×640 uint8 4-class mask).

### MDS shards (parallel to GazeGene)

`RayNet/openeds/convert_to_mds.py` mirrors the GazeGene converter so both datasets share streaming ergonomics. Schema:

| Column | Type | Notes |
|---|---|---|
| `image` | bytes | Raw PNG (no re-encoding — keeps the 1-channel grayscale faithful). |
| `mask` | ndarray (H, W) uint8 | 4-class index. `[0,0,0,0]` placeholder when unlabelled. |
| `subject` | int | Parsed from `S_<int>`. |
| `frame_idx` | int | Original integer frame index. |
| `has_mask` | int | 1 if paired `.npy` exists, 0 otherwise. |

**Sequence grouping**: pass `--sequence_grouped` (on by default) to write per-subject frames in monotonic `frame_idx` order. The streaming sequence reader requires this and reads with `shuffle=False` so contiguous windows correspond to real temporal sequences.

```bash
# Default 80/20 subject split
python -m RayNet.openeds.convert_to_mds \
    --data_dir /path/to/openEDS/openEDS \
    --output_dir /path/to/mds_openeds/train \
    --split train --sequence_grouped

python -m RayNet.openeds.convert_to_mds \
    --data_dir /path/to/openEDS/openEDS \
    --output_dir /path/to/mds_openeds/val \
    --split val --sequence_grouped
```

### Streaming readers

`RayNet/openeds/streaming.py` provides:

- `StreamingOpenEDSSegDataset` — flat per-frame iterator. `require_labelled=True` by default skips unlabelled frames via the existing `NonEmptyBatchLoader`.
- `StreamingOpenEDSSequenceDataset` — sliding window of `window` consecutive frames per item. Sanity-checks per window that all frames share one `subject` id; returns `None` (and the empty-batch wrapper drops it) for windows that cross a subject boundary.

Loader builders mirror the GazeGene helpers:

- `create_openeds_seg_streaming_dataloaders(remote_train, remote_val, ...)`
- `create_openeds_sequence_streaming_dataloaders(remote_train, remote_val, window=64, ...)`

| | |
|---|---|
| Subjects | 191 |
| Frames | ~93k images, ~62k labelled |
| Native size | 400 × 640 |
| Stride-32-friendly crop | 416 × 640 (zero-pad) or 384 × 640 (centre-crop) |
| Class map | 0=bg, 1=sclera, 2=iris, 3=pupil |

Loaders in `RayNet/openeds/dataset.py`:
- `OpenEDSSegDataset` — flat per-frame iterator for the segmenter.
- `OpenEDSSequenceDataset` — sliding-window per-subject sequences for the TCN (default window 64 frames, ~640 ms at 100 Hz).

### Foveal segmenter

`RayNet/openeds/segmenter.py:RITnetStyleSegmenter`. 1-channel input, 4-class softmax output, dense-block encoder + transposed-conv decoder.

Default config (`base_channels=32, growth_rate=16`) is ~2.3M params. For a tighter RITnet-faithful build set `base_channels=16, growth_rate=8` (~0.5M params).

Loss: `combined_loss = cross_entropy + dice_weight * soft_dice` (background excluded from Dice). Class frequencies on OpenEDS are roughly `[0.80, 0.12, 0.06, 0.02]`; pass `class_weights=tensor([0.5, 1.0, 2.0, 4.0])` if pupil recall is too low under uniform CE.

### Torsion (3DeepVOG-style)

`RayNet/openeds/torsion.py:IrisPolarTorsion`. Stateful per-session estimator; classical, no learnable params:

1. Fit ellipses to iris and pupil masks (OpenCV least-squares).
2. Extract iris annulus (`r_pupil + 5%` to `r_iris − 5%`) into a `(64, 360)` polar rubbersheet via `cv2.remap`.
3. Slice into 8 angular patches of 30° each.
4. **Rolling reference**: average polar of first 30 non-blink frames seeds the per-session reference.
5. Per-patch NCC against the reference within a ±15° φ search → per-patch shift.
6. Discard patches whose NCC peak falls below 0.4 (occluded by eyelid).
7. Weighted-median of accepted patch shifts → torsion angle in degrees.

Self-supervised pretraining helper (`torsion_self_supervised_pretext`) generates synthetic `(rotated_polar, target_Δφ)` pairs for warming up a learnable variant.

### TCN temporal block

`RayNet/openeds/temporal.py:TCNTemporalBlock`. Causal dilated 1D conv stack:

| | |
|---|---|
| Layers | 4 residual blocks (2 causal convs each) |
| Kernel size | 3 |
| Dilations | (1, 2, 4, 8) |
| Receptive field | **61 frames** — call `model.receptive_field()` to verify |
| Hidden dim | 128 |
| Params | 399k |
| Outputs | `smoothed_gaze (3)`, `torsion_residual (1)`, `blink_logit (1)`, `movement_class (4)` |

Pseudo-label helpers:
- `derive_pseudo_blink_labels(pupil_areas)` — per-frame pupil-area below 30% of session median.
- `derive_pseudo_saccade_labels(gaze_seq, threshold_deg_per_frame=0.3)` — frame-to-frame angular velocity threshold.

OpenEDS itself ships no GT for blink or movement type, so the TCN trains on these pseudo-labels plus a smoothing/consistency objective on the gaze and torsion residuals.

### Training entry point

`RayNet/openeds/train.py` provides a thin CLI:

```bash
# Stage S1 — segmenter
python -m RayNet.openeds.train --root /path/to/openEDS --stage seg --epochs 30

# Stage S2 — TCN (requires segmenter checkpoint + per-frame gaze pipeline)
python -m RayNet.openeds.train --root /path/to/openEDS --stage tcn --window 64
```

The OpenEDS dataset is **not** present on the host that holds the GazeGene shards in this repo's reference checkout — the harness is intended to run on a separate machine that has the Kaggle FovalNet preprocessed dump.

## Reshard required

v6 introduces two new MDS columns: `iris_mesh_3d (100, 3)` and `visual_axis (3,)`. Older shards predate this schema and the streaming loader fills them with NaN tensors so the train loop can detect "no GT" via `torch.isnan(...)` and skip the relevant losses without crashing. To enable the full v6 supervision schedule:

```bash
python -m RayNet.streaming.convert_to_mds \
    --data_dir /path/to/GazeGene_FaceCrops \
    --output_dir /path/to/mds_shards_v6 \
    --multiview_grouped
```

Until reshard is complete, set `lam_iris_mesh = lam_iris_edge = lam_gaze_visual = 0` in `PHASE_CONFIG` so the new heads are constructed but produce no loss term.

## Files modified / added

| Type | Path | Purpose |
|---|---|---|
| modified | `RayNet/raynet_v5.py` | `IrisMeshHead`, **`MacroGazeHead`** (v6.1), mean-of-two `GeometricGazeHead`, T_vec normalisation in `CameraEmbedding`, model output dict expanded |
| modified | `RayNet/losses.py` | `iris_mesh_loss`, `iris_edge_loss`, `visual_axis_loss`, **`macro_gaze_loss`** (v6.1), `total_loss` signature |
| modified | `RayNet/train.py` | `TRICAM_IDS`, v6/v6.1 PHASE_CONFIG, batch GT plumbing (incl. `gaze_c`), NaN-sentinel detection |
| modified | `RayNet/dataset.py` | Emit `iris_mesh_3d`, `visual_axis`, `gaze_c`, `camera_ids` filter |
| modified | `RayNet/streaming/dataset.py` | Plumb new GT, NaN sentinel, TriCam filter, n_views inference |
| modified | `RayNet/streaming/convert_to_mds.py` | Schema additions (incl. `gaze_c`) |
| added | `RayNet/openeds/__init__.py` | Package exports |
| added | `RayNet/openeds/dataset.py` | `OpenEDSSegDataset`, `OpenEDSSequenceDataset` |
| added | `RayNet/openeds/segmenter.py` | `RITnetStyleSegmenter`, `combined_loss` |
| added | `RayNet/openeds/torsion.py` | `IrisPolarTorsion`, `torsion_self_supervised_pretext` |
| added | `RayNet/openeds/temporal.py` | `TCNTemporalBlock`, pseudo-label helpers |
| added | `RayNet/openeds/train.py` | OpenEDS-only CLI training harness |
| **added (v6.1)** | `RayNet/openeds/convert_to_mds.py` | OpenEDS → MDS shard converter (sequence-grouped) |
| **added (v6.1)** | `RayNet/openeds/streaming.py` | `StreamingOpenEDSSegDataset`, `StreamingOpenEDSSequenceDataset` + builders |
| added/updated | `docs/wiki/Loss-Functions.md` | Sections 11-14 (visual axis, iris mesh, iris edge, macro gaze), v6.1 phase table |
| added | `docs/wiki/Architecture-v6.md` | This document |

## Quick start

### GazeGene shards + training

```bash
# 1. (re)shard with the v6 / v6.1 schema (adds iris_mesh_3d, visual_axis, gaze_c)
python -m RayNet.streaming.convert_to_mds \
    --data_dir /path/to/GazeGene_FaceCrops \
    --output_dir /path/to/mds_shards_v6/train --split train --multiview_grouped

python -m RayNet.streaming.convert_to_mds \
    --data_dir /path/to/GazeGene_FaceCrops \
    --output_dir /path/to/mds_shards_v6/val --split val --multiview_grouped

# 2. Train with v6.1 defaults (TriCam {1,6,8}, T_vec normalised, macro+micro gaze)
python -m RayNet.train \
    --mds_streaming \
    --mds_train  /path/to/mds_shards_v6/train \
    --mds_val    /path/to/mds_shards_v6/val \
    --output_dir ./runs/v6_1_first
```

### OpenEDS shards + training

```bash
# 1. Shard the FovalNet preprocessed OpenEDS (sequence_grouped on by default)
python -m RayNet.openeds.convert_to_mds \
    --data_dir /path/to/openEDS/openEDS \
    --output_dir /path/to/mds_openeds/train --split train

python -m RayNet.openeds.convert_to_mds \
    --data_dir /path/to/openEDS/openEDS \
    --output_dir /path/to/mds_openeds/val --split val

# 2. Train the segmenter (4-class, 1-channel, 416x640 zero-padded)
python -m RayNet.openeds.train --root /path/to/openEDS/openEDS \
    --stage seg --epochs 30
```

The TriCam filter, T_vec normalisation, and v6/v6.1 phase weights are enabled by default. The CLI surface is unchanged from v5 — most v5 invocations Just Work, only the shard schema requires a refresh.

## Open work

- **Iris-mesh M-target ablation** — re-run the latest experiment with iris-mesh + edge losses to quantify the angular-error improvement (3DGazeNet's ablation suggests +30%-class generalisation in cross-dataset, smaller within-dataset).
- **OpenEDS segmenter training** — run on the Kaggle host once the dataset is mounted; report mean IoU per class and pupil sub-pixel boundary accuracy.
- **Torsion learnable variant** — a small CNN that consumes the polar rubbersheet and regresses Δφ directly; expected to improve under partial occlusion.
- **TCN supervised heads** — once the segmenter and per-frame gaze run on OpenEDS, train the TCN's blink/saccade/movement heads against pseudo-labels.
- **v6 inference fusion** — the bridge between the GazeGene-trained skeleton and the OpenEDS-trained foveal/torsion heads is currently inference-only. The geometric prior (project predicted eyeball centre into the OpenEDS frame to seed the segmentation prior) requires per-deployment camera intrinsics and is left to the deployment harness.
