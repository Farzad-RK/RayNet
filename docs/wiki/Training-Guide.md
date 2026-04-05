# Training Guide

Complete guide to training RayNet v2, covering local training, MDS streaming from MinIO, hardware profiles, checkpointing, and resume.

## Prerequisites

1. **GazeGene dataset** — either:
   - Local disk: raw GazeGene_FaceCrops directory
   - MinIO/S3: MDS shards (see [[MosaicML Streaming]])
   - HuggingFace Hub: WebDataset shards (see [[WebDataset Streaming]])
2. Python 3.9+ with CUDA-capable GPU
3. Dependencies installed: `pip install -r requirements.txt`

## Data Split

| Split | Subjects | Purpose |
|-------|----------|---------|
| Train | 1 - 46 | Model training |
| Val | 47 - 56 | Validation and model selection |

Each subject has up to ~2000 frames across 9 cameras (~18,000 samples/subject).

---

## Basic Training (Local Dataset)

```bash
python -m RayNet.train \
    --data_dir /path/to/GazeGene_FaceCrops \
    --backbone repnext_m3 \
    --output_dir ./results \
    --epochs 30 \
    --eye L
```

This uses the `default` hardware profile (batch_size=512, no AMP, no compile).

---

## MDS Streaming from MinIO (Recommended)

Stream data from MinIO with checkpoints saved back to MinIO:

```bash
export S3_ENDPOINT_URL=http://YOUR_SERVER_IP:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=your-password

python -m RayNet.train \
    --mds_streaming \
    --mds_train s3://gazegene/train \
    --mds_val s3://gazegene/val \
    --ckpt_bucket raynet-checkpoints \
    --minio_endpoint http://YOUR_SERVER_IP:9000 \
    --profile a100 \
    --backbone repnext_m3 \
    --weight_path /path/to/repnext_m3_pretrained.pt
```

See `deploy/README.md` for MinIO setup.

---

## 3-Phase Progressive Schedule

Training follows a curriculum that stabilizes landmark detection before introducing gaze regression and multi-view consistency.

### Phase 1: Landmark Warmup (Epochs 1-5)

| Parameter | Value |
|-----------|-------|
| Lambda landmark | 1.0 |
| Lambda gaze | **0.0** (frozen) |
| Lambda gaze_consist | 0.0 |
| Lambda shape | 0.0 |
| Learning rate | 1e-3 |
| Heatmap sigma | 2.0 |
| Multi-view | **Off** |
| Dataloader | Standard (random single-view batches) |

**Purpose**: Train the landmark head in isolation. Each camera view is treated as an **independent sample** — the dataloader draws randomly from all subjects, frames, and cameras with no view grouping. This maximizes sample diversity. Larger sigma makes the heatmap target broader, easier to learn initially.

**Expected results**: Landmark pixel error drops from ~1.1px to ~0.6px. Angular gaze error stays at ~42° (random, since gaze head receives no gradients).

### Phase 2: Introduce Gaze + Multi-View (Epochs 6-15)

| Parameter | Value |
|-----------|-------|
| Lambda landmark | 1.0 |
| Lambda gaze | **0.3** |
| Lambda gaze_consist | **0.1** |
| Lambda shape | **0.05** |
| Learning rate | 5e-4 |
| Heatmap sigma | 1.5 |
| Multi-view | **On** |
| Dataloader | Multi-view (9 cameras per group) |

**Purpose**: Gaze L1 loss (on unit vectors in normalized space, following GazeGene paper Sec 4.1.1) is introduced at 0.3 weight. The multiview MDS loader is created **lazily** at the start of Phase 2, grouping 9 camera views per sample. Gaze ray consistency enforces directional agreement across views in world frame. Landmark shape consistency regularizes the spatial pattern of predictions.

**Expected results**: Gaze error drops rapidly (42° -> ~20° in first epoch, continuing to ~10-12° by epoch 15). Landmark accuracy is maintained.

### Phase 3: Balanced Fine-Tuning (Epochs 16-30)

| Parameter | Value |
|-----------|-------|
| Lambda landmark | **0.5** |
| Lambda gaze | **0.5** |
| Lambda gaze_consist | **0.2** |
| Lambda shape | **0.1** |
| Learning rate | 1e-4 |
| Heatmap sigma | 1.0 |
| Multi-view | **On** |
| Dataloader | Multi-view |

**Purpose**: Equal task weighting. Tighter heatmap sigma demands sub-pixel precision. Multi-view weights at full strength. Lower learning rate for fine-grained convergence.

### Phase Transitions

At each phase boundary, the optimizer and scheduler are **recreated**:
- Optimizer: AdamW with `betas=(0.5, 0.95)`, `weight_decay=1e-4`
- Scheduler: CosineAnnealingLR per phase (T_max = phase duration)

---

## Loss Functions

### Gaze Loss (L1 on Unit Vectors)

Following the GazeGene paper (Sec 4.1.1), gaze is trained with **L1 loss on unit vectors** in normalized space:

```python
L_gaze = L1(pred_gaze, gt_gaze)    # both (B, 3) unit vectors
```

Angular error is computed with `atan2` for metrics/logging only — not backpropagated. This avoids the `torch.acos` gradient singularity at cos_sim = +/-1, which caused NaN during training.

### Multi-View Losses (Ray-Based)

All multi-view losses operate in **normalized space with unit vectors** — no raw 3D coordinates, no matrix inversions, no SVD. This ensures numerical stability under AMP float16.

| Loss | Description |
|------|-------------|
| **Gaze ray consistency** | `R_norm^T @ pred_gaze` should agree across 9 views (L1 vs group mean) |
| **Landmark shape consistency** | Translation/scale-invariant landmark patterns should match across views (Smooth L1) |

See [[Loss Functions]] and [[Multi-View Consistency]] for details.

---

## Hardware Profiles

Select a profile with `--profile NAME`. CLI flags override profile values.

### Available Profiles

| Profile | Batch | mv_groups | AMP | Compile | Workers | Grad Accum | Target GPU |
|---------|-------|-----------|-----|---------|---------|------------|------------|
| default | 512 | 2 | off | off | 4 | 1 | Consumer (RTX 3090/4090) |
| t4 | 512 | 4 | fp16 | off | 2 | 1 | T4 (16GB) |
| l4 | 1024 | 8 | fp16 | on | 4 | 1 | L4 (24GB) |
| a10g | 1024 | 8 | fp16 | on | 4 | 1 | A10G (24GB) |
| v100 | 512 | 4 | fp16 | off | 4 | 1 | V100 (16-32GB) |
| a100 | 2048 | 16 | fp16 | on | 8 | 2 | A100 (80GB) |
| h100 | 4096 | 32 | bf16 | on | 8 | 2 | H100 (80GB) |

### A100 Optimizations

Applied automatically with `--profile a100`:
- `torch.backends.cuda.matmul.allow_tf32 = True`
- `torch.backends.cudnn.allow_tf32 = True`
- `torch.set_float32_matmul_precision('high')`
- Model wrapped with `torch.compile()`
- Training uses `GradScaler` + `autocast(dtype=float16)`

### Multi-View GPU Utilization

In Phase 2+, the multiview loader uses `mv_groups * 9` as batch size (much smaller than Phase 1). To increase GPU utilization, override with `--mv_groups`:

```bash
# A100: increase from default 16 to 64 (batch 576, ~50-60GB GPU)
python -m RayNet.train --profile a100 --mv_groups 64 ...
```

---

## Checkpoint System (MinIO)

When `--ckpt_bucket` is provided, checkpoints are stored on MinIO organized by run:

```
s3://raynet-checkpoints/checkpoints/<run_id>/
    metadata.json            # Run config + per-epoch metrics
    latest.pt                # Overwritten every epoch (for --resume)
    best_model.pt            # Best validation loss
    checkpoint_epoch5.pt     # Periodic snapshot (every --ckpt_every epochs)
```

### Checkpoint Contents

| File | Contents | Purpose |
|------|----------|---------|
| `latest.pt` | model + optimizer + scheduler + scaler + metrics | Resume training |
| `best_model.pt` | model + optimizer + scheduler + scaler + val_loss | Deploy / fine-tune |
| `checkpoint_epochN.pt` | model + optimizer + scheduler + scaler + metrics | Historical snapshot |
| `metadata.json` | run config, per-epoch train/val metrics, best epoch | Experiment tracking |

### Checkpoint Frequency

Controlled by `--ckpt_every N` (default: 5):

- `latest.pt` is saved **every epoch**
- `checkpoint_epochN.pt` is saved **every N epochs**
- `best_model.pt` is saved **whenever validation loss improves**

---

## Resuming Training

### Resume from Latest

```bash
python -m RayNet.train \
    --mds_streaming --mds_train s3://gazegene/train --mds_val s3://gazegene/val \
    --ckpt_bucket raynet-checkpoints \
    --minio_endpoint http://YOUR_SERVER_IP:9000 \
    --profile a100 \
    --run_id run_20260405_025128 \
    --resume
```

### Resume from Specific Checkpoint

Use `--resume_from` to load a specific checkpoint file (e.g., to roll back to a known-good state):

```bash
python -m RayNet.train \
    --mds_streaming --mds_train s3://gazegene/train --mds_val s3://gazegene/val \
    --ckpt_bucket raynet-checkpoints \
    --minio_endpoint http://YOUR_SERVER_IP:9000 \
    --profile a100 \
    --run_id run_20260405_025128 \
    --resume_from checkpoint_epoch5.pt
```

This loads the model, optimizer, scheduler, and scaler state from epoch 5 and resumes at epoch 6. Useful when later epochs produced corrupted state (e.g., from NaN losses).

---

## WebDataset Streaming (HF Hub)

For training from HuggingFace Hub without MinIO:

```bash
python -m RayNet.train \
    --profile a100 \
    --streaming \
    --dataset_url "pipe:curl -sL https://huggingface.co/datasets/USER/gazegene-wds/resolve/main/train/gazegene-train-{000000..000099}.tar" \
    --val_dataset_url "pipe:curl -sL https://huggingface.co/datasets/USER/gazegene-wds/resolve/main/val/gazegene-val-{000000..000019}.tar" \
    --epochs 30
```

See [[WebDataset Streaming]] for shard creation and upload instructions.

---

## CLI Reference

```
python -m RayNet.train [OPTIONS]

Data:
  --data_dir PATH               Path to GazeGene dataset (required if not streaming)
  --output_dir PATH             Output directory (default: ./results)
  --samples_per_subject INT     Max frames per subject (default: 200)
  --eye {L,R}                   Which eye (default: L)

MDS Streaming:
  --mds_streaming               Stream MDS shards from MinIO/S3
  --mds_train URL               MDS remote URL for training (e.g. s3://gazegene/train)
  --mds_val URL                 MDS remote URL for validation

WebDataset Streaming:
  --streaming                   Use WebDataset streaming
  --dataset_url URL             Train shard URL pattern
  --val_dataset_url URL         Val shard URL pattern

Model:
  --backbone NAME               repnext_m0 through repnext_m5 (default: repnext_m3)
  --weight_path PATH            Pretrained backbone weights

Hardware:
  --profile NAME                Hardware profile (default, t4, l4, a10g, v100, a100, h100)
  --no_compile                  Disable torch.compile

Checkpoints:
  --ckpt_bucket BUCKET          MinIO bucket for checkpoints (enables MinIO storage)
  --ckpt_prefix PREFIX          Key prefix under bucket (default: checkpoints)
  --ckpt_every N                Save named checkpoint every N epochs (default: 5)
  --minio_endpoint URL          MinIO endpoint (default: $S3_ENDPOINT_URL)
  --run_id ID                   Run ID for checkpoint grouping (auto-generated if omitted)
  --resume                      Resume from latest.pt of --run_id
  --resume_from FILE            Resume from specific checkpoint file (e.g. checkpoint_epoch5.pt)

Overrides:
  --batch_size INT              Override profile batch size
  --epochs INT                  Total epochs (default: 30)
  --num_workers INT             Dataloader workers
  --mv_groups INT               Multi-view groups per batch (batch = mv_groups * 9)
  --grad_accum_steps INT        Gradient accumulation steps
```

---

## Output

### Console Output

```
Epoch   8 | Phase 2 | lr 3.97e-04 | Train: loss=0.1307 lm=0.0875 ang=16.70deg gaze_mv=0.1985 shape=0.0604 | Val: loss=0.1433 lm_px=0.60px ang=18.48deg
```

| Field | Description |
|-------|-------------|
| `loss` | Total training loss |
| `lm` | Landmark loss component |
| `ang` | Gaze angular error in degrees (metric only, not loss) |
| `gaze_mv` | Gaze ray consistency loss (Phase 2+) |
| `shape` | Landmark shape consistency loss (Phase 2+) |
| `lm_px` | Validation landmark error in pixels (224x224 space) |

### Training Log CSV Columns

| Column | Description |
|--------|-------------|
| `epoch` | Epoch number |
| `phase` | Training phase (1, 2, or 3) |
| `lr` | Current learning rate |
| `train_total` | Total training loss |
| `train_landmark` | Landmark loss component |
| `train_angular_deg` | Gaze angular error in degrees |
| `train_reproj` | Gaze ray consistency loss (0 in Phase 1) |
| `train_mask` | Landmark shape consistency loss (0 in Phase 1) |
| `val_total` | Total validation loss |
| `val_landmark` | Validation landmark loss |
| `val_angular_deg` | Validation gaze error in degrees |
| `val_landmark_px` | Validation landmark error in pixels |

---

## Gradient Clipping

All training uses `clip_grad_norm_(max_norm=1.0)` to prevent gradient explosions.

## Mixed Precision Notes

- **Forward + loss**: computed under `autocast(dtype=float16)` (or `bfloat16` for H100)
- **Backward**: scaled by `GradScaler` to prevent underflow
- **Optimizer step**: unscaled, then clipped, then stepped via `scaler.step()`
- **Multi-view losses**: fully float16-compatible (unit vector operations only)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM on A100 | Reduce `--batch_size` or `--mv_groups`, or increase `--grad_accum_steps` |
| NaN loss | Should not occur with current L1 gaze loss. If it does, check data pipeline. |
| Slow first epoch with torch.compile | Normal. Compilation happens on first forward pass (~60s) |
| Phase 2 low GPU utilization | Increase `--mv_groups` (e.g., 64 for A100) |
| MDS streaming hangs | Check MinIO connectivity; ensure `S3_ENDPOINT_URL` is set |
| `--resume` loads wrong epoch | Use `--resume_from checkpoint_epochN.pt` for a specific epoch |
| `lr_scheduler.step()` warning | Cosmetic on resume, does not affect training |
| Gaze error ~42° in Phase 1 | Expected — gaze head receives no gradients in Phase 1 |
