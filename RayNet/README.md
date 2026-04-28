# RayNet v5 — Model Package

**Triple-M3 + AERI + HRFH-α** architecture. A shared low-level encoder (landmark-owned) and three task-specific branches that all consume the full 224×224 face crop. The gaze branch carries its own mini U-Net (`AERIHead`) producing **iris + eyeball** binary segmentation logits at 56×56. The two masks are combined into a saliency map (`0.65·iris + 0.35·eyeball`); the saliency is blended with a uniform field through an `α` schedule, the result both gates the gaze bottleneck (7×7) **with a 0.5 floor** AND modulates the high-resolution `d1` decoder tensor (56×56, 64ch) **with the same 0.5 floor** to produce a 64-d → 128-d **foveal vector** that represents sub-pixel iris/pupil dynamics. The floors are the eyelid-occlusion mitigation: when AERI mis-fires (e.g. drowsy-eye drift), the gaze pathway still receives ≥50% of the underlying features. See `raynet_v5.py`.

The backbone was promoted from **RepNeXt-M1** to **RepNeXt-M3** (embed_dim=64,128,256,512; depth=3,3,13,2) because the M1 variant could not reach sub-pixel landmark accuracy on full-face crops; the prior sub-pixel result was achieved with M3 + the full dataset, so the capacity gap is now closed.

## Components

- **SharedStem** — RepNeXt-M3 stem + stages[0..1]. 3→64→128ch, 28×28. ~0.37M params. Intermediate maps (`s0`, `s1`) are exposed as U-Net skip connections for the landmark branch. Pose and gaze branches receive `s1.detach()` so only landmark loss backprops here.
- **Landmark Branch** — RepNeXt-M3 stages[2..3] + U-Net decoder with attention gates. Predicts 14 iris/pupil heatmaps at 56×56 with soft-argmax + offset refinement. ~10.36M params.
- **Pose Branch** — RepNeXt-M3 stages[2..3] on `s1.detach()`, CoordAtt + pooled feature fused with `BoxEncoder(face_bbox)` via a zero-init residual. Predicts 6D rotation + 3D translation. `face_bbox` is **optional at inference** — when omitted the BoxEncoder residual zeroes out and pose collapses to CNN features. ~7.47M params.
- **Gaze Branch** — RepNeXt-M3 stages[2..3] on `s1.detach()` + **AERIHead** (mini U-Net, 2-class: iris + eyeball). Pipeline:
  1. AERI predicts iris/eyeball logits at 56×56 plus the decoder tensor `d1` (64ch, 56×56).
  2. `saliency = 0.65·sigmoid(iris) + 0.35·sigmoid(eyeball)`.
  3. `scheduled_mask = α·saliency + (1−α)·1` — α controlled by `get_scheduled_alpha(epoch)` in `train.py`.
  4. **HRFH harvesting** with eyelid-occlusion floors:
     - Global gate at 7×7: `gate = GLOBAL_FLOOR + (1−GLOBAL_FLOOR)·pool₇(scheduled_mask)`, `GLOBAL_FLOOR = 0.5`. Multiplied into the 7×7 bottleneck → 512-d global vector → LayerNorm.
     - Foveal gate at 56×56: `gate = FOVEAL_FLOOR + (1−FOVEAL_FLOOR)·scheduled_mask`, `FOVEAL_FLOOR = 0.5`. Multiplied into `d1` → pooled to 64-d → `Linear→128 → GELU → LayerNorm` → stochastic depth (`FOVEAL_DROP_P=0.10`, train only).
  5. `[global ‖ foveal_proj]` (640-d) → `Linear → 256` gaze_feat.
  6. `GazeFusionBlock` folds in `pose_feat` via a zero-init residual.
  7. `CrossViewAttention` (when `n_views > 1`).
  8. `GeometricGazeHead` predicts `eyeball_center` and `pupil_center`; optical axis is `normalize(pupil − eyeball)`. ~10.76M params.
- **BoxEncoder (MAGE)** — `(x_p, y_p, L_x)` → `d_model` via 3→64→128→256 MLP with GELU. Consumed by the **pose** branch (not gaze). Provides head-pose prior; optional at inference.
- **CrossViewAttention** — 9-camera geometric attention conditioned on `R_cam`, `T_cam`. Identity short-circuit when `n_views ≤ 1`. ~1.07M params including the camera embedding.

### Why floors? (eyelid-occlusion failure mode)

In the M1 architecture the global gate was `0.25 + 0.75·M` (25% floor) and the foveal gate was a pure multiply `M` (0% floor). When an eyelid partially covers the sclera at inference time, AERI's iris/eyeball masks shrink, the saliency map collapses toward zero, and the foveal pathway loses ~70% of its magnitude (with α=0.7 saturation, mask floor = 1−α = 0.3, so foveal_feat ≈ 0.3·pool(d1)). This produced the "drowsy-eye drift" pattern observed in `run_20260427_205327` — the model relies on a crisp iris/sclera boundary as a shortcut, and degrades catastrophically when that boundary is occluded.

Raising both floors to 0.5 means: AERI can still amplify the eye region by up to 1× (gate range 0.5→1.0), but the gaze branch always receives at least half of its underlying features unconditionally. This shifts the AERI signal from "feature gate" toward "feature emphasiser" and makes the model robust to imperfect masks at inference.

## Forward Signature

```python
from RayNet.raynet_v5 import create_raynet_v5

model = create_raynet_v5(
    backbone_weight_path='./ptrained_models/repnext_m3_distill_300e.pth',  # or None for random init
    n_landmarks=14,
)

out = model(
    images,              # (B, 3, 224, 224), B = n_views * mv_groups
    n_views=9,
    R_cam=R_cam,         # (B, 3, 3) camera rotation
    T_cam=T_cam,         # (B, 3)    camera translation
    face_bbox=face_bbox, # (B, 3)    (x_p, y_p, L_x) — optional at inference
    aeri_alpha=0.9,      # 0..1     AERI saliency vs. uniform-field blend
)
```

Outputs: `landmark_coords`, `landmark_heatmaps`, `iris_mask_logits`, `eyeball_mask_logits`, `eyeball_center`, `pupil_center`, `gaze_vector`, `gaze_vector_sv`, `gaze_angles`, `pred_pose_6d`, `pred_pose_t`.

No eye-crop anchor is passed: the entire forward path is pixel-crop-free. AERI provides the eye-region inductive bias via a soft mask instead of a hard crop.

### Inference without external face crops

The reference `inference.py` runs MediaPipe (Haar fallback) inside the module. The caller passes a full frame; the module detects the face, square-crops with a 1.3× expansion, runs the model, projects landmarks/masks/gaze back to frame coordinates, and draws the bounding box that was actually fed to the model. `face_bbox` is synthesised from the detected pixels (`mage_bbox_from_pixels`) assuming a centred principal point, or set to `None` to bypass the BoxEncoder entirely.

## Intrinsic Delta for BoxEncoder GT

`face_bbox` is derived from the two camera intrinsics (original and cropped) that GazeGene already stores — no manual labels. `K_cropped` is calibrated for the native 448×448 JPG, so the dataset rescales it to the 224-space tensor before computing `(x_p, y_p, L_x)`. See `dataset.py::__getitem__` and `docs/wiki/Geometry-and-Kappa.md`.

## AERI Masks

Iris and eyeball binary masks are **baked into MDS shards** at conversion time (see `streaming/eye_masks.py` + `streaming/convert_to_mds.py`). Each sample carries `iris_mask` and `eyeball_mask` as `uint8` arrays at 56×56. The dataset loader passes them to the training loop; `losses.mask_seg_loss` applies BCE-with-logits against the AERI head outputs.

## Losses

`losses.total_loss` combines, with phase-dependent weights:

- Landmark heatmap + coordinate L1 (`lam_lm`, `lam_heatmap`)
- Optical-axis angular + gaze-vector L1 (`lam_gaze`)
- GazeGene 3D structure: `lam_eyeball`, `lam_pupil`, `lam_geom_angular`
- Pose: 6D geodesic (`lam_pose`) + translation (`lam_trans`)
- Optional ray-to-target reprojection (`lam_ray`, `lam_reproj`)
- **AERI segmentation**: `lam_iris_seg`, `lam_eyeball_seg` (BCE-with-logits at 56×56)

Multi-view consistency is applied by `multiview_loss.multiview_consistency_loss` on 9-camera batches in phases 2 and 3.

## Three-Phase Training Schedule

The schedule is now interpreted as a **branch-staged curriculum** rather than parallel MTL. All branches are wired into the same forward graph, but loss weights + selective freezing isolate one objective at a time. See `train.py::PHASE_CONFIG` and `train.py::get_scheduled_alpha`.

| Phase | Epochs | Active branches | Frozen | LR | Multi-view |
|-------|--------|-----------------|--------|----|------------|
| 1 | 1–15 | landmark + AERI iris/eyeball seg + headpose | gaze branch (encoder, fusion, head) | 5e-4 | off |
| 2 | 16–30 | gaze branch + AERI fine-tune | shared stem + landmark + pose | 3e-4 | off (monocular only) |
| 3 | 31–50 | gaze + pose + AERI + multi-view fusion | shared stem + landmark | 5e-5 | on (5-epoch ramp) |

**Phase 1** establishes the landmark foundation (val_landmark_px ≤ 2.2 px) and crisp iris/eyeball masks (val seg loss ≤ 0.005 / 0.012). The gaze loss is OFF (`lam_gaze = lam_eyeball = lam_pupil = lam_geom_angular = 0`); seg weights are lifted to 1.0 to drive masks to convergence before HRFH-α consumes them. P1 was extended from 8 to 15 epochs because under-trained landmarks cap downstream gaze quality — the gaze branch consumes `s1` (and the AERI head sits on the same shared stem), so any sub-pixel error in P1 propagates.

**Phase 2** trains gaze monocularly. `aeri_alpha` ramps 0.4 → 0.7 over the first 3 epochs of P2 (epochs 16-18), then holds at 0.7 — do NOT ramp during fine-tune. `lam_iris_seg = lam_eyeball_seg = 0.5` keeps the masks stable while gaze adapts. CrossViewAttention is short-circuited (`n_views=1`); use a single-view dataloader.

**Phase 3** turns on `multiview=True`, `lam_reproj=0.1`, `lam_mask=0.05`, with `mv_weight = min(1, max(0, (epoch − 30) / 5))` ramping the consistency loss in over the first 5 epochs. The shared stem and landmark branch are frozen (`freeze_set='face_kept'`) and `lam_lm = 0` — landmark fine-tune in P3 risks pulling the shared stem in a direction that helps sub-pixel landmark error at the cost of the gaze representation. Pose stays trainable (no shared params with gaze). `aeri_alpha` stays constant at 0.7.

Phase transitions preserve optimizer momentum and rebuild only the `CosineAnnealingLR` for the new phase window. Gradient clipping is `max_norm=5.0` in phase 1 and `max_norm=2.0` afterwards. AMP uses GradScaler only when dtype is fp16; bf16 skips the scaler.

Fork/warmstart/resume machinery is preserved for cross-architecture migrations via `_filter_compatible_state` (drops shape-mismatched tensors) and `_optimizer_state_compatible` (guards `optimizer.load_state_dict` across parameter-group changes). `--warmstart_phase` lets a Phase-2 fork start at the right LR/loss weights when warmstarting from a Phase-1 checkpoint without replaying P1.

## AERI-α Schedule (`get_scheduled_alpha`)

α controls how much the saliency mask gates gaze features vs. the uniform field. `scheduled_mask = α·saliency + (1−α)·1`.

  - `α = 0.4` (Phase 1, epochs 1-15) — gaze branch is frozen, value is moot but kept low to avoid suppressing AERI seg supervision on training-only masks.
  - `α = 0.4 → 0.7` linear ramp over P2 epochs 16-18 (3 epochs).
  - `α = 0.7` (P2 fine-tune + Phase 3, epochs 19+) — held constant.
  - The previously-shipped `0.4 → 0.9` linear ramp during the cosine LR decay caused validation drift in `triple_m1_aeri_iris_eyeball_500spc_run_20260423_101115` (val_angular climbs from 12.4° at epoch 28 to 15.3° at epoch 35 as α approaches 0.9). Hold α constant during fine-tune.

## Data Loaders

- `dataset.create_dataloaders` — local disk, renders AERI masks on the fly.
- `streaming.create_multiview_streaming_dataloaders` — MosaicML MDS streaming with 9-grouped batches; masks are read directly from the shard.

## Parameters

~30.0M total: SharedStem 0.37M, LandmarkBranch 10.36M, PoseBranch 7.47M, GazeBranch 10.76M (includes AERIHead), CrossViewAttention + CameraEmbedding 1.07M. The ~1.6× jump over the prior M1 budget (~18.7M) is the M1→M3 promotion; this is the capacity headroom needed for sub-pixel landmark accuracy.

## Inference

```bash
python -m RayNet.inference --checkpoint best_model.pt --webcam
python -m RayNet.inference --checkpoint best_model.pt --input clip.mp4 --output annotated.mp4
python -m RayNet.inference \
    --ckpt_bucket raynet-checkpoints \
    --minio_endpoint http://204.168.238.119:9000 \
    --run_id triple_m1_aeri_iris_eyeball_500spc_run_20260423_101115 \
    --ckpt_file best_model.pt --webcam
```

The inference module embeds face detection (MediaPipe → Haar fallback). It draws: detected face box, 14 iris/pupil landmarks, AERI iris (green) and eyeball (yellow) masks overlaid at 56×56 → upsampled to crop, gaze arrow from the eye-center, RGB pose axes, pitch/yaw/translation overlay. Toggle the mask overlay with the `m` key. Use `--no_masks` to start with the overlay off.
