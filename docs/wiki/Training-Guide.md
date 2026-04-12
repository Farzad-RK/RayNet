# Training Guide

Complete guide to training RayNet v4.1, covering staged training, hardware profiles, MDS streaming, checkpointing, and resume.

## Prerequisites

1. **GazeGene dataset** -- either:
   - Local disk: raw GazeGene_FaceCrops directory
   - MinIO/S3: MDS shards (see [[MosaicML Streaming]])
   - HuggingFace Hub: WebDataset shards (see [[WebDataset Streaming]])
2. Python 3.9+ with CUDA-capable GPU
3. Dependencies installed: `pip install -r requirements.txt`
4. Pretrained backbone weights (distilled, non-fused `.pth` format):
   - `repnext_m3_distill_300e.pth` (main backbone)
   - `repnext_m1_distill_300e.pth` (pose backbone)

## Data Split

| Split | Subjects | Purpose |
|-------|----------|---------|
| Train | 1 - 46 | Model training |
| Val | 47 - 56 | Validation and model selection |

Each subject has up to ~2000 frames across 9 cameras (~18,000 samples/subject).

---

## 3-Stage Training Strategy

RayNet v4.1 uses a staged training curriculum to establish baselines before combining tasks. Each stage validates a specific hypothesis before proceeding.

### Stage 1: Landmark + Pose Baseline (no gaze)

**Purpose**: Validate both backbones learn useful representations independently.

**Expectations**:
- Landmark px error < 5px by epoch 10
- Pose geodesic < 10 deg by epoch 15
- Anomaly: pose stuck > 30 deg = backbone not learning face geometry

```bash
python -m RayNet.train --stage 1 --profile t4 \
    --core_backbone_weight_path /path/to/repnext_m3_distill_300e.pth \
    --pose_backbone_weight_path /path/to/repnext_m1_distill_300e.pth \
    --data_dir /path/to/GazeGene_FaceCrops \
    --epochs 20
```

| Phase | Epochs | lam_lm | lam_pose | lam_trans | lam_gaze | Bridge | Multi-view | max_norm |
|-------|--------|--------|----------|-----------|----------|--------|------------|----------|
| 1 | 1-10 | 1.0 | 0.5 | 0.5 | 0.0 | Off | Off | 5.0 |
| 2 | 11-20 | 1.0 | 1.0 | 1.0 | 0.0 | Off | Off | 2.0 |

### Stage 2: Add Gaze, No Bridge

**Purpose**: Test gaze learning without the crop-poisoned LandmarkGazeBridge.

**Expectations**:
- Gaze angular error improving on BOTH train and val (no divergence)
- Val gaze < 20 deg by epoch 15 = gaze learns from appearance alone
- Anomaly: train gaze improving but val worsening = adversarial optimization in shared backbone

```bash
python -m RayNet.train --stage 2 --profile t4 \
    --core_backbone_weight_path /path/to/repnext_m3_distill_300e.pth \
    --pose_backbone_weight_path /path/to/repnext_m1_distill_300e.pth \
    --data_dir /path/to/GazeGene_FaceCrops \
    --epochs 25
```

| Phase | Epochs | lam_lm | lam_gaze | lam_pose | lam_trans | lam_ray | Bridge | Multi-view | max_norm |
|-------|--------|--------|----------|----------|-----------|---------|--------|------------|----------|
| 1 | 1-5 | 1.0 | 0.1 | 0.5 | 0.5 | 0.0 | Off | Off | 5.0 |
| 2 | 6-15 | 1.0 | 0.5 | 0.5 | 0.5 | 0.1 | Off | On | 2.0 |
| 3 | 16-25 | 0.5 | 1.0 | 0.5 | 0.5 | 0.3 | Off | On | 2.0 |

### Stage 3: Full Pipeline with Bridge

**Purpose**: Test if LandmarkGazeBridge helps or hurts. Compare val gaze to Stage 2.

Only run after Stage 2 shows gaze converging on validation. Same loss weights as Stage 2, but with `use_bridge=True`.

```bash
python -m RayNet.train --stage 3 --profile t4 \
    --core_backbone_weight_path /path/to/repnext_m3_distill_300e.pth \
    --pose_backbone_weight_path /path/to/repnext_m1_distill_300e.pth \
    --data_dir /path/to/GazeGene_FaceCrops \
    --epochs 25
```

**Decision rule**: If bridge helps (val gaze improves over Stage 2), keep it. If bridge hurts (val gaze worse than Stage 2), remove it.

---

## Phase Transitions

At each phase boundary, the optimizer and scheduler are **recreated**:
- Optimizer: AdamW with `betas=(0.5, 0.95)`, `weight_decay=1e-4`
- Scheduler: CosineAnnealingLR per phase (T_max = phase duration)
- Learning rate: set per phase (see tables above)

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
| l4 | 288 | 32 | fp16 | on | 4 | 1 | L4 (24GB) |
| a10g | 288 | 32 | fp16 | on | 4 | 1 | A10G (24GB) |
| v100 | 144 | 16 | fp16 | off | 4 | 2 | V100 (16-32GB) |
| a100 | 1152 | 128 | bf16 | on | 8 | 1 | A100 (40-80GB) |
| h100 | 2304 | 256 | bf16 | on | 8 | 1 | H100 (80GB) |

Batch sizes are optimized for 224x224 input and the dual-backbone architecture (~15.6M params).

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
  --epochs INT                  Total epochs (default: 30)
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
| Gaze error ~42 deg in Stage 1 | Expected -- gaze head receives no gradients |
| Pose loss stuck > 30 deg | Pose backbone not learning face geometry; check weights |
| Train gaze improves, val worsens | Adversarial optimization -- try Stage 2 (no bridge) first |
| MDS streaming hangs | Check MinIO connectivity; ensure `S3_ENDPOINT_URL` is set |
