# Training Guide

Complete guide to training RayNet v5, covering staged training, hardware profiles, MDS streaming, checkpointing, and resume.

## Prerequisites

1. **GazeGene dataset** -- either:
   - Local disk: raw GazeGene_FaceCrops directory
   - MinIO/S3: MDS shards (see [[MosaicML Streaming]])
2. Python 3.9+ with CUDA-capable GPU
3. Dependencies installed: `pip install -r requirements.txt`
4. Pretrained backbone weights (distilled, non-fused `.pth` format):
   - `repnext_m1_distill_300e.pth` (used for all branches — shared stem + 3 branch encoders)

## Data Split

| Split | Subjects | Purpose |
|-------|----------|---------|
| Train | 1 - 46 | Model training |
| Val | 47 - 56 | Validation and model selection |

Each subject has up to ~2000 frames across 9 cameras (~18,000 samples/subject).

---

## 3-Stage Training Strategy

RayNet v5 uses a staged training curriculum to establish baselines before combining tasks. Each stage validates a specific hypothesis before proceeding.

> **Auto-cap**: `--epochs` is automatically capped to the stage's last configured epoch. You can pass `--epochs 30` (the default) for any stage and it will stop at the right epoch.

### Stage 1: Landmark + Pose Baseline (no gaze)

**Purpose**: Validate shared stem + landmark + pose branches learn useful features independently. The gaze branch receives no gradient — `val_angular_deg` will stay ~42° (GazeGene's natural mean optical-axis direction) throughout.

**Recommended recipe** (from [`docs/experiments/raynet_v5_500_samples_per_subject/`](../experiments/raynet_v5_500_samples_per_subject/)): `--samples_per_subject 500`, `--profile kaggle_t4x2`, 15 epochs. This drove `val_landmark_px` from **3.93 → 2.64** (best at E14) — a 33% absolute improvement over the previous 200-samples recipe. Total wall time ≈ 10.7 h on 2× T4.

**Expectations**:
- Landmark px error < 4px by epoch 5, < 3px by epoch 8, best ≈ 2.6px around epoch 14
- Pose translation L1 down by ≥ 10× between epoch 1 and epoch 8 (e.g. 0.089 → 0.002 cm on the reference run)
- Anomaly: landmark px stuck > 5px after epoch 8 = backbone / shared-stem init failed to load
- `val_angular_deg ≈ 42.5°` throughout is **expected** (`lam_gaze=0` in both phases)

```bash
python -m RayNet.train --stage 1 --profile t4 \
    --core_backbone_weight_path /path/to/repnext_m1_distill_300e.pth \
    --pose_backbone_weight_path /path/to/repnext_m1_distill_300e.pth \
    --data_dir /path/to/GazeGene_FaceCrops
```

| Phase | Epochs | lam_lm | lam_pose | lam_trans | lam_gaze | Bridge | Multi-view | max_norm |
|-------|--------|--------|----------|-----------|----------|--------|------------|----------|
| 1 | 1-8 | 1.0 | 0.5 | 0.5 | 0.0 | Off | Off | 5.0 |
| 2 | 9-15 | 1.0 | 1.0 | 1.0 | 0.0 | Off | Off | 2.0 |

### Stage 2: Eye-Crop Gaze on Top of a Frozen Face Path

**Purpose**: Fit only the dedicated eye-crop gaze branch on top of the Stage 1 landmark + pose baseline, then unfreeze for joint fine-tuning. This is the Quad-M1 curriculum that replaced the Triple-M1 Stage 2 — the old schedule stalled at `val_angular ≈ 42°` because the gaze branch shared the stride-16 face feature map (see [`docs/experiments/raynet_v5_S2_fork_500_samples_per_subject/`](../experiments/raynet_v5_S2_fork_500_samples_per_subject/)).

**Expectations**:
- P1 (face frozen, gaze warmup only): `val_angular_deg` drops below the Triple-M1 ~42° floor by epoch 8.
- P2 (face frozen + multi-view + geometric angular): gaze keeps improving while eyeball / pupil L1 converge.
- P3 (face unfrozen, joint fine-tuning): landmarks should *not* regress meaningfully from the S1 baseline (`val_landmark_px` stays ≤ ~3px); if they do, drop `lam_lm` or shorten P3.
- Anomalies:
  - P1 stuck at 42° → check that `freeze_face=True` is actually being applied (the training log prints `FROZEN (gaze-only training)` at each phase transition and on every `train_one_epoch` call).
  - NaN in P2 after multi-view ramp → too-hot `lam_geom_angular` before geometry converges; verify the Stage 1 pupil/eyeball centers are calibrated.

```bash
python -m RayNet.train --stage 2 --profile kaggle_t4x2 \
    --core_backbone_weight_path /path/to/repnext_m1_distill_300e.pth \
    --data_dir /path/to/GazeGene_FaceCrops \
    --fork_from s3://raynet-checkpoints/checkpoints/<stage1_run_id>/best_model.pt \
    --run_id <new_stage2_run_id>
```

| Phase | Epochs | `freeze_face` | lam_lm | lam_gaze | lam_eyeball | lam_pupil | lam_geom_angular | lam_pose | lam_trans | lam_ray | lam_reproj | lam_mask | lr | Multi-view |
|-------|--------|---------------|--------|----------|-------------|-----------|------------------|----------|-----------|---------|------------|----------|----|------------|
| 1 | 1-8 | **True** | 0.0 | 1.0 | 0.3 | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3e-4 | Off |
| 2 | 9-15 | **True** | 0.0 | 1.0 | 0.5 | 0.5 | 0.2 | 0.0 | 0.0 | 0.2 | 0.1 | 0.05 | 1e-4 | On |
| 3 | 16-25 | False | 0.5 | 1.0 | 0.5 | 0.5 | 0.3 | 0.3 | 0.3 | 0.3 | 0.1 | 0.05 | 5e-5 | On |

Phases 1-2 train only the gaze branch (`EyeCropModule` + `EyeBackbone` + `GazeFusionBlock` + `GeometricGazeHead` + `CrossViewAttention` + `CameraEmbedding`). The shared stem, landmark branch, and pose branch are moved to `.eval()` + `requires_grad_(False)` via `train.set_face_frozen(model, True)`, so BatchNorm running stats also freeze and the predicted-landmark distribution the eye crop depends on stays fixed. Phase 3 releases the face path for gentle joint fine-tuning at a conservative learning rate.

### Stage 3: Full Pipeline with Bridges + MAGE BoxEncoder

**Purpose**: Activate inter-branch bridges (landmark cross-attention + pose SHMA modulation) and BoxEncoder fusion. Compare val gaze to Stage 2.

Only run after Stage 2 shows gaze converging on validation. Bridges are zero-init from checkpoint (never trained in S1/S2) — they start as identity and learn gradually.

```bash
python -m RayNet.train --stage 3 --profile t4 \
    --core_backbone_weight_path /path/to/repnext_m1_distill_300e.pth \
    --pose_backbone_weight_path /path/to/repnext_m1_distill_300e.pth \
    --data_dir /path/to/GazeGene_FaceCrops
```

| Phase | Epochs | lam_lm | lam_gaze | lam_eyeball | lam_pupil | lam_geom_angular | lam_pose | lam_trans | lam_ray | Bridge | Multi-view |
|-------|--------|--------|----------|-------------|-----------|-----------------|----------|-----------|---------|--------|------------|
| 1 | 1-5 | 1.0 | 0.3 | 0.3 | 0.3 | 0.1 | 0.5 | 0.5 | 0.0 | On | Off |
| 2 | 6-15 | 1.0 | 0.5 | 0.5 | 0.5 | 0.2 | 0.5 | 0.5 | 0.1 | On | On |
| 3 | 16-25 | 0.5 | 1.0 | 0.5 | 0.5 | 0.3 | 0.5 | 0.5 | 0.3 | On | On |

**Decision rule**: If bridge helps (val gaze improves over Stage 2), keep it. If bridge hurts (val gaze worse than Stage 2), remove it.

---

## Phase Transitions

At each phase boundary, the optimizer and scheduler are **recreated**:
- Optimizer: AdamW with `betas=(0.5, 0.95)`, `weight_decay=1e-4`
- Scheduler: CosineAnnealingLR per phase (T_max = phase duration)
- Learning rate: set per phase (see tables above)
- `set_face_frozen(model, cfg['freeze_face'])` is called at every transition and re-applied inside `train_one_epoch` after `model.train()` so the freeze survives the per-epoch `.train()` toggle.

## Cross-Stage Forking

Stage 2 typically starts from a Stage 1 checkpoint via `--fork_from`. Because the Quad-M1 architecture differs from the old Triple-M1 Stage 1 checkpoints (new `EyeBackbone`, 3-input `GazeFusionBlock` replacing the old 2-input `FusionBlock`), two safety nets in `train.py` handle the migration:

- **`_filter_compatible_state(src_sd, target_sd)`** drops any tensor whose shape no longer matches the current model — e.g. the old `gaze_branch.fusion_block.proj.0.weight` (256×512) gives way to the new 256×768 `GazeFusionBlock.proj.0.weight`. The dropped keys are printed at load time so the migration is auditable.
- **`_optimizer_state_compatible(saved_state, new_optimizer)`** checks per-group parameter counts; if they differ, the fork falls back to a fresh AdamW state (with a printed warning) rather than crashing with *"loaded state dict contains a parameter group that doesn't match the size of optimizer's group"*.

Cross-stage forks therefore always keep whatever backbone weights survive shape-wise and always rebuild the optimizer state when the architecture has changed.

---

## Gradient Clipping

Varies by phase for multi-task learning:

| Phase | max_norm | Rationale |
|-------|----------|-----------|
| Phase 1 | 5.0 | Aggressive -- large multi-task gradients during warmup |
| Phase 2+ | 2.0 | Conservative -- prevents gaze/pose interference during fine-tuning |

---

## MDS Streaming from MinIO (Recommended)

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

---

## Hardware Profiles

Select with `--profile NAME`. CLI flags override profile values.

| Profile | Batch | mv_groups | AMP | Compile | Workers | Grad Accum | Target GPU |
|---------|-------|-----------|-----|---------|---------|------------|------------|
| default | 504 | 56 | off | off | 4 | 1 | CPU / testing |
| t4 | 144 | 16 | fp16 | off | 2 | 2 | T4 (16GB) |
| l4 | 288 | 32 | bf16 | off | 4 | 1 | L4 (24GB) |
| a10g | 288 | 32 | bf16 | off | 4 | 1 | A10G (24GB) |
| v100 | 144 | 16 | fp16 | off | 4 | 2 | V100 (16-32GB) |
| a100 | 1152 | 128 | bf16 | on | 8 | 1 | A100 (40-80GB) |
| h100 | 2304 | 256 | bf16 | on | 8 | 1 | H100 (80GB) |
| kaggle_t4x2 | 144 per GPU | 16 | fp16 | off | 2 | 2 | 2× T4 via `accelerate launch --multi_gpu --num_processes 2` |
| multi_node_t4 | 144 per GPU | 16 | fp16 | off | 2 | 2 | 2 machines × 1× T4 each (NCCL over TCP) |

Batch sizes are sized for the 224×224 input and ~17M-param Quad-M1. Distributed profiles (`kaggle_t4x2`, `multi_node_t4`) are per-process; global effective batch is `batch_size × num_processes × grad_accum_steps`.

### DDP + frozen face path

`hardware_profiles.build_accelerator()` constructs the `Accelerator` with `DistributedDataParallelKwargs(find_unused_parameters=True)`. This is required whenever `freeze_face=True` (Stage 2 P1/P2) because DDP's default reducer fails with *"Expected to have finished reduction in the prior iteration before starting a new one"* when a subset of parameters (here the frozen face path) never receives gradients. The flag enables graph-aware reduction that tolerates unused params; cost is one extra graph traversal per step, negligible against a full M1 forward pass.

---

## Checkpoint System (MinIO)

When `--ckpt_bucket` is provided, checkpoints are stored on MinIO:

```
s3://raynet-checkpoints/checkpoints/<run_id>/
    metadata.json            # Run config + per-epoch metrics
    latest.pt                # Overwritten every epoch (for --resume)
    best_model.pt            # Best validation loss
    checkpoint_epoch5.pt     # Periodic snapshot (every --ckpt_every epochs)
    batch_log.csv            # Per-batch loss log (uploaded each epoch)
```

### Resuming Training

```bash
# Resume from latest checkpoint
python -m RayNet.train \
    --mds_streaming --mds_train s3://gazegene/train --mds_val s3://gazegene/val \
    --ckpt_bucket raynet-checkpoints \
    --run_id run_20260405_025128 \
    --resume

# Resume from specific checkpoint
python -m RayNet.train \
    --mds_streaming --mds_train s3://gazegene/train --mds_val s3://gazegene/val \
    --ckpt_bucket raynet-checkpoints \
    --run_id run_20260405_025128 \
    --resume_from checkpoint_epoch5.pt
```

---

## CLI Reference

```
python -m RayNet.train [OPTIONS]

Data:
  --data_dir PATH               GazeGene dataset path (required if not streaming)
  --output_dir PATH             Output directory (default: ./results)
  --samples_per_subject INT     Max frames per subject (default: 200)
  --eye {L,R}                   Which eye (default: L)

MDS Streaming:
  --mds_streaming               Stream MDS shards from MinIO/S3
  --mds_train URL               MDS path for training (e.g. s3://gazegene/train)
  --mds_val URL                 MDS path for validation

WebDataset Streaming:
  --streaming                   Use WebDataset streaming
  --dataset_url URL             Train shard URL pattern
  --val_dataset_url URL         Val shard URL pattern

Model:
  --core_backbone NAME          Main backbone: repnext_m0 through repnext_m5 (default: repnext_m3)
  --pose_backbone NAME          Pose backbone: repnext_m0/m1/m2 or "none" (default: repnext_m1)
  --core_backbone_weight_path   Pretrained main backbone weights (.pth, non-fused)
  --pose_backbone_weight_path   Pretrained pose backbone weights (.pth, non-fused)

Training:
  --stage {1,2,3}               Training stage (default: 3)
  --epochs INT                  Total epochs (default: 30, auto-capped to stage max)
  --no_multiview                Disable multi-view (ablation)
  --gaze_only                   Disable landmark loss (ablation)

Hardware:
  --profile NAME                Hardware profile (default, t4, l4, a10g, v100, a100, h100)
  --no_compile                  Disable torch.compile
  --batch_size INT              Override profile batch size
  --num_workers INT             Override dataloader workers
  --mv_groups INT               Multi-view groups per batch (batch = mv_groups * 9)
  --grad_accum_steps INT        Gradient accumulation steps

Checkpoints:
  --ckpt_bucket BUCKET          MinIO bucket (enables cloud checkpointing)
  --ckpt_prefix PREFIX          Key prefix (default: checkpoints)
  --ckpt_every N                Named checkpoint every N epochs (default: 5)
  --minio_endpoint URL          MinIO endpoint (default: $S3_ENDPOINT_URL)
  --run_id ID                   Run ID for checkpoint grouping
  --resume                      Resume from latest.pt of --run_id
  --resume_from FILE            Resume from specific checkpoint file
```

---

## Output

### Console Output

```
Epoch   8 | Phase 2 | lr 3.97e-04 | Train: loss=0.1307 lm=0.0875 ang=16.70deg gaze_mv=0.1985 shape=0.0604 ray=0.0312 pose=0.1524 trans=0.0891 | Val: loss=0.1433 lm_px=0.60px ang=18.48deg
```

### Training Log CSV (`training_log.csv`)

| Column | Description |
|--------|-------------|
| `epoch` | Epoch number |
| `phase` | Training phase (1, 2, or 3) |
| `lr` | Current learning rate |
| `train_total` | Total training loss |
| `train_landmark` | Landmark loss (normalized by feature area) |
| `train_angular_deg` | Gaze angular error in degrees (metric) |
| `train_reproj` | Gaze ray consistency loss |
| `train_mask` | Landmark shape consistency loss |
| `train_ray_target` | Ray-to-target loss |
| `train_pose` | Geodesic pose rotation loss |
| `train_translation` | Translation loss |
| `val_total` | Validation total loss |
| `val_landmark` | Validation landmark loss |
| `val_angular_deg` | Validation gaze error in degrees |
| `val_landmark_px` | Validation landmark error in pixels |

### Batch Log CSV (`batch_log.csv`)

Per-batch granularity: `epoch, batch, loss, landmark, angular_deg, gaze_consist, shape, ray_target, pose, translation, lr`

---

## Mixed Precision Notes

- **Forward + loss**: computed under `autocast(dtype=float16/bfloat16)`
- **Backward**: scaled by `GradScaler` to prevent underflow
- **Optimizer step**: unscaled, clipped (phase-dependent max_norm), then stepped via `scaler.step()`
- **Multi-view losses**: fully float16-compatible (unit vector operations only)
- **Geodesic loss**: uses `clamp` for numerical stability in `arccos`

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM | Reduce `--batch_size` or `--mv_groups`, or increase `--grad_accum_steps` |
| NaN loss | Check data pipeline. Geodesic loss is clamped; gaze uses L1 (no singularity) |
| Slow first epoch with torch.compile | Normal (~60s compilation on first forward pass) |
| Phase 2 low GPU utilization | Increase `--mv_groups` |
| `val_angular_deg ≈ 42°` in Stage 1 | Expected -- gaze branch receives no gradients (`lam_gaze=0.0`); ~42° is GazeGene's natural mean optical-axis direction |
| Pose loss stuck > 30° | Pose backbone not learning face geometry; check pretrained weights |
| Stage 2 gaze stuck at 42° | Ceiling from the old Triple-M1 shared-feature design; the current Quad-M1 + `freeze_face` curriculum was the fix. If still stuck, verify the training log prints `FROZEN (gaze-only training)` at the P1/P2 boundary |
| DDP "did not receive grad for rank N: 0 1 2 3..." | `build_accelerator` must pass `find_unused_parameters=True`; required whenever `freeze_face=True` (Stage 2 P1/P2) |
| `ValueError: loaded state dict contains a parameter group that doesn't match the size of optimizer's group` when forking | Fork is from a pre-Quad-M1 checkpoint; the new guard `_optimizer_state_compatible` falls back to fresh AdamW state. Confirm the warning appears in the log |
| MDS streaming hangs | Check MinIO connectivity; ensure `S3_ENDPOINT_URL` is set |
| Landmark > 4px at Stage 1 end | Try `--samples_per_subject 500` (proven recipe) and confirm the default auto-caps `--epochs` to 15 for Stage 1 |
| `KeyError: N` in `get_phase_config` | Epochs exceed the stage's configured range; `--epochs` should be auto-capped but check your `--stage` flag |
