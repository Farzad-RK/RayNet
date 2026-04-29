# RayNet v5 — Model Package

**Triple-M1 + FPANet landmark + AERI + HRFH-α** architecture. A shared low-level encoder (landmark-owned) and three task-specific branches that all consume the full 224×224 face crop. The landmark branch and the gaze branch's AERI head each carry a private **PANet** (`FeaturePyramidNetwork`) that fuses the four backbone strides — P2/56², P3/28², P4/14², P5/7² — to a uniform `fpn_ch` width via 1×1 lateral + top-down (semantic injection) + bottom-up (high-resolution amplification) passes. The landmark head consumes the fused P2 (default 128 channels @ 56×56) and the AERI head emits **iris + eyeball** binary segmentation logits from the same fused P2 plus a `fpn_ch`-channel decoder tensor `d1`. The two masks are combined into a saliency map (`0.65·iris + 0.35·eyeball`); the saliency is blended with a uniform field through an `α` schedule, the result both gates the gaze bottleneck (7×7) **with a 0.5 floor** AND modulates the high-resolution `d1` (56×56) **with the same 0.5 floor** to produce a `fpn_ch`-d → 96-d **foveal vector** that represents sub-pixel iris/pupil dynamics. The floors are the eyelid-occlusion mitigation: when AERI mis-fires (e.g. drowsy-eye drift), the gaze pathway still receives ≥50% of the underlying features. See `raynet_v5.py`.

The backbone is **RepNeXt-M1** (embed_dim=48,96,192,384; depth=3,3,15,2). An earlier M1→M3 promotion was reverted: the `Tripple_M3_run_20260428_130241` benchmark with matched training config showed M3 offered no measurable gain over M1, so the 30M-param M3 budget was wasteful. Sub-pixel landmark accuracy is recoverable at the M1 budget once the dataset size is sufficient (`samples_per_subject = 500` cut val_landmark_px from 3.93 to 2.64 on M1 — a 33% drop with no architecture change).

## Components

- **SharedStem** — RepNeXt-M1 stem + stages[0..1]. 3→48→96ch, 28×28. Intermediate maps (`s0`, `s1`) feed the landmark branch's PANet directly and the gaze branch's PANet via `s0.detach()` / `s1.detach()`, so only landmark loss backprops into the stem.
- **Landmark Branch** — RepNeXt-M1 stages[2..3] + private `FeaturePyramidNetwork` (PANet) over `[s0, s1, s2, s3]` → fused P2 (`fpn_ch`@56) → 2× Conv-BN-SiLU refine → heatmap + offset 1×1 heads → soft-argmax + offset refinement. The PANet's top-down + bottom-up fusion is what makes sub-pixel landmark accuracy reachable; the prior single-stage U-Net decoder plateaued well above 1 px.
- **Pose Branch** — RepNeXt-M1 stages[2..3] on `s1.detach()`, CoordAtt + pooled feature fused with `BoxEncoder(face_bbox)` via a zero-init residual. Predicts 6D rotation + 3D translation. `face_bbox` is **optional at inference** — when omitted the BoxEncoder residual zeroes out and pose collapses to CNN features.
- **Gaze Branch** — RepNeXt-M1 stages[2..3] on `s1.detach()` + **FPNAERIHead** (private PANet over `[s0_det, s1_det, gaze_s2, gaze_s3]`, 2-class: iris + eyeball). Pipeline:
  1. AERI predicts iris/eyeball logits at 56×56 plus the decoder tensor `d1` (`fpn_ch`@56).
  2. `saliency = 0.65·sigmoid(iris) + 0.35·sigmoid(eyeball)`.
  3. `scheduled_mask = α·saliency + (1−α)·1` — α controlled by `get_scheduled_alpha(epoch)` in `train.py`.
  4. **HRFH harvesting** with eyelid-occlusion floors:
     - Global gate at 7×7: `gate = GLOBAL_FLOOR + (1−GLOBAL_FLOOR)·pool₇(scheduled_mask)`, `GLOBAL_FLOOR = 0.5`. Multiplied into the 7×7 bottleneck → 384-d global vector → LayerNorm.
     - Foveal gate at 56×56: `gate = FOVEAL_FLOOR + (1−FOVEAL_FLOOR)·scheduled_mask`, `FOVEAL_FLOOR = 0.5`. Multiplied into `d1` → pooled to `fpn_ch`-d → `Linear→96 → GELU → LayerNorm` → stochastic depth (`FOVEAL_DROP_P=0.10`, train only).
  5. `[global ‖ foveal_proj]` (480-d) → `Linear → d_model` gaze_feat.
  6. `GazeFusionBlock` folds in `pose_feat` via a zero-init residual.
  7. `CrossViewAttention` (when `n_views > 1`).
  8. `GeometricGazeHead` predicts `eyeball_center` and `pupil_center`; optical axis is `normalize(pupil − eyeball)`.
- **BoxEncoder (MAGE)** — `(x_p, y_p, L_x)` → `d_model` via 3→64→128→256 MLP with GELU. Consumed by the **pose** branch (not gaze). Provides head-pose prior; optional at inference.
- **CrossViewAttention** — 9-camera geometric attention conditioned on `R_cam`, `T_cam`. Identity short-circuit when `n_views ≤ 1`.

## Eyelid-occlusion robustness (three-pronged)

OpenFace's CLNF — a part-based model on a much older backbone — handles partial eyelid occlusion gracefully because each landmark's local detector reasons over surrounding anatomy (eyebrow, lashes, sclera-skin edge) rather than a crisp iris/sclera boundary. RayNet now matches that inductive bias through three coordinated changes:

1. **HRFH-α floors (0.5 / 0.5)**. In the M1 architecture as originally shipped, the global gate was `0.25 + 0.75·M` (25% floor) and the foveal gate was a pure multiply `M` (0% floor). When an eyelid covers part of the sclera, AERI's iris/eyeball masks shrink, the saliency collapses toward zero, and the foveal pathway loses up to ~70% of its magnitude (with α=0.7 saturation, mask floor = 1−α = 0.3, so foveal_feat ≈ 0.3·pool(d1)). This produced the "drowsy-eye drift" pattern in `run_20260427_205327`. Raising both floors to 0.5 means AERI still amplifies the eye region (gate range 0.5→1.0), but the gaze branch always receives at least half of its features unconditionally, so the head can never be starved.

2. **BCE + soft-Dice mask loss** (`losses.mask_seg_loss`). BCE alone is biased toward "background everywhere" because every pixel costs the same and the eyeball is only ~5–10% of a 56×56 face crop (iris ≈ 1–2%). Soft-Dice (`1 − 2·|P∩G| / (|P|+|G|)`) is area-normalised and forces the head to learn the full silhouette shape. The combined loss is `BCE + dice_weight·soft_Dice` with `dice_weight=1.0` and Laplace smoothing `dice_eps=1.0`. Set `dice_weight=0` to A/B against the previous BCE-only behaviour.

3. **Eyelid-occlusion augmentation** (`streaming/occlusion_aug.py`). The GT eyeball mask in the MDS shards is the **theoretical full silhouette** (no eyelid clip — see `streaming/eye_masks.py`). We exploit that by painting a synthetic, skin-toned, feathered band over the top portion of the visible eye region in the **image** while leaving the GT mask untouched. The seg head then receives `(occluded image, full silhouette)` pairs and is forced to extrapolate the silhouette from anatomical context. This is gated by the `--eyelid_occlusion_p` CLI flag (default `0.30` → 30% of training samples), with `cover_frac_range=(0.20, 0.55)`, a 4-pixel feathered bottom edge, and ±10% multiplicative skin-tone jitter so the model can't memorise the augmenter's palette. The augmentation is applied per-sample in `StreamingGazeGeneDataset.__getitem__`, **before** Normalize, and only to the train split.

These three are independent — drop any one and the others still help — but the combination is what reproduces OpenFace-style robustness to drowsy / partial-blink frames without sacrificing accuracy on fully-open eyes.

## Forward Signature

```python
from RayNet.raynet_v5 import create_raynet_v5

model = create_raynet_v5(
    backbone_weight_path='./ptrained_models/repnext_m1_distill_300e.pth',  # or None for random init
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

Iris and eyeball binary masks are **baked into MDS shards** at conversion time (see `streaming/eye_masks.py` + `streaming/convert_to_mds.py`). Each sample carries `iris_mask` and `eyeball_mask` as `uint8` arrays at 56×56. The masks are the **theoretical, un-occluded** silhouettes — eyelid clipping is intentionally *not* applied at conversion time so that downstream training can teach the AERI head to predict through occluders (see `eye_masks.py` docstring). The dataset loader passes them to the training loop; `losses.mask_seg_loss` applies BCE + soft-Dice against the AERI head outputs.

## Losses

`losses.total_loss` combines, with phase-dependent weights:

- Landmark heatmap + coordinate L1 (`lam_lm`, `lam_heatmap`)
- Optical-axis angular + gaze-vector L1 (`lam_gaze`)
- GazeGene 3D structure: `lam_eyeball`, `lam_pupil`, `lam_geom_angular`
- Pose: 6D geodesic (`lam_pose`) + translation (`lam_trans`)
- Optional ray-to-target reprojection (`lam_ray`, `lam_reproj`)
- **AERI segmentation**: `lam_iris_seg`, `lam_eyeball_seg` (BCE + soft-Dice at 56×56; `mask_seg_loss(dice_weight=1.0, dice_eps=1.0)`)

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

~18.7M total at the RepNeXt-M1 budget. The Triple-M1 design replicates `stages[2..3]` per branch (landmark / pose / gaze), so the per-branch heads carry their own mid+late stages on top of the shared 48→96-channel stem. The exact `nn.Module.numel()` breakdown is printed at startup by `train.py`. The earlier M1→M3 promotion (~30.0M) was reverted because matched-config benchmarks showed no measurable accuracy gain over M1 — capacity was not the bottleneck, dataset size was (`samples_per_subject=500`).

## Hardware Profiles

Profiles live in `hardware_profiles.py::HARDWARE_PROFILES`. Choose one with `--profile <name>`. The CLI flags `--batch_size`, `--mv_groups`, `--num_workers`, `--grad_accum_steps`, `--no_compile` override individual fields. `mv_groups` is the **number of 9-view groups per batch**, so `batch_size = mv_groups × 9`.

| Profile | VRAM | mv_groups | batch_size | grad_accum | AMP | TF32 | compile |
|---------|------|-----------|------------|------------|-----|------|---------|
| `t4` | 16 GB | 16 | 144 | 2 | fp16 | – | – |
| `v100` | 16/32 GB | 16 | 144 | 2 | fp16 | – | – |
| `kaggle_t4x2` | 2× 16 GB | 16 (per-GPU) | 144 (per-GPU) | 2 | fp16 | – | – |
| `multi_node_t4` | 2× 16 GB (2 nodes) | 16 (per-GPU) | 144 (per-GPU) | 2 | fp16 | – | – |
| `a10g` | 24 GB | 32 | 288 | 1 | bf16 | ✓ | – |
| `l4` | 24 GB | **48** | **432** | 1 | bf16 | ✓ | – |
| `a100` | 80 GB | **224** | **2016** | 1 | bf16 | ✓ | ✓ |
| `h100` | 80 GB | 256 | 2304 | 1 | bf16 | ✓ | ✓ |

**`l4` (24 GB)** is tuned for ≤1 GB headroom on the full Triple-M1 + AERI + HRFH-α model. If you see `peak_alloc < 22 GB` after warm-up in the per-epoch memory probe (below), nudge with `--mv_groups 56` (504 samples). If you OOM, drop to `--mv_groups 40`.

**`a100` (80 GB)** is anchored against the H100 80 GB profile (256 mv_groups, 2304 samples) — the A100 has the same VRAM but slightly less raw bandwidth, so it sits one notch below at 224 mv_groups (2016 samples) with the 1 GB safety margin. If `peak_alloc < 78 GB`, you can match H100 exactly with `--mv_groups 256`. **40 GB A100s** should override with `--mv_groups 64` (576 samples).

`bfloat16` is preferred over `fp16` on Ada/Ampere/Hopper because it has FP32's exponent range — the gaze pipeline contains `exp`/`log` ops (geometric angular loss, soft-Dice) that can overflow fp16 at large batch sizes.

### Peak-memory probe

`train.py` prints per-epoch memory stats after each train epoch (rank 0 only):

```
GPU mem (rank 0): peak 22.8 / total 23.0 GB  (headroom 0.2 GB)
```

`torch.cuda.max_memory_allocated()` is read at the end of each train epoch and `reset_peak_memory_stats()` is called immediately after, so each line reflects only that epoch's peak. Use this to verify the 1 GB margin on your actual hardware and to dial `mv_groups` up/down from the profile defaults.

### Distributed training (Accelerate)

Distribution is via HuggingFace Accelerate. AMP stays under `torch.amp.autocast` + `GradScaler` (handled in `train.py`); `Accelerator(mixed_precision='no')` is intentional. `find_unused_parameters=True` is required by the Stage 2 / Phase 2 selective-freeze curriculum.

```bash
# Kaggle 2× T4 (single node)
accelerate launch --multi_gpu --num_processes 2 \
    -m RayNet.train --profile kaggle_t4x2 --mds_streaming ...

# Two single-T4 machines (NCCL over TCP)
# On main node ($MAIN_IP):
accelerate launch --multi_gpu --num_machines 2 --num_processes 2 \
    --machine_rank 0 --main_process_ip $MAIN_IP --main_process_port 29500 \
    -m RayNet.train --profile multi_node_t4 ...
# On the second node, same command with --machine_rank 1.
```

`batch_size` and `mv_groups` in the distributed profiles are **per-process**. Global effective batch = `batch_size × num_processes × grad_accum_steps`.

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
