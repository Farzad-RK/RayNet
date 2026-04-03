# Training Guide

Complete guide to training RayNet v2, covering local training, A100 optimization, streaming, and the Colab notebook.

## Prerequisites

1. **GazeGene dataset** on local disk, or WebDataset shards on HF Hub (see [[WebDataset Streaming]])
2. Python 3.9+ with CUDA-capable GPU
3. Dependencies installed: `pip install -r requirements.txt`

## Data Split

| Split | Subjects | Purpose |
|-------|----------|---------|
| Train | 1 - 46 | Model training |
| Val | 47 - 56 | Validation and model selection |

Each subject has up to ~1000 frames across 9 cameras (up to ~9000 samples/subject).

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

## 3-Phase Progressive Schedule

Training follows a curriculum that stabilizes landmark detection before introducing gaze regression and multi-view consistency:

### Phase 1: Landmark Warmup (Epochs 1-5)

| Parameter | Value |
|-----------|-------|
| Lambda landmark | 1.0 |
| Lambda gaze | **0.0** (frozen) |
| Lambda reproj | 0.0 |
| Lambda mask | 0.0 |
| Learning rate | 1e-3 |
| Heatmap sigma | 2.0 |
| Multi-view | **Off** |
| Dataloader | Standard (random single-view batches) |

**Purpose**: Train the landmark head in isolation. Larger sigma makes the heatmap target broader, easier to learn initially.

### Phase 2: Introduce Gaze + Multi-View (Epochs 6-15)

| Parameter | Value |
|-----------|-------|
| Lambda landmark | 1.0 |
| Lambda gaze | **0.3** |
| Lambda reproj | **0.1** |
| Lambda mask | **0.05** |
| Learning rate | 5e-4 |
| Heatmap sigma | 1.5 |
| Multi-view | **On** |
| Dataloader | Multi-view (9 cameras per group) |

**Purpose**: Gently introduce the gaze loss and multi-view consistency. Landmarks are now stable enough for cross-view reprojection to be meaningful.

### Phase 3: Balanced Fine-Tuning (Epochs 16-30)

| Parameter | Value |
|-----------|-------|
| Lambda landmark | **0.5** |
| Lambda gaze | **0.5** |
| Lambda reproj | **0.2** |
| Lambda mask | **0.1** |
| Learning rate | 1e-4 |
| Heatmap sigma | 1.0 |
| Multi-view | **On** |
| Dataloader | Multi-view |

**Purpose**: Equal weighting of both tasks. Tighter heatmap sigma demands sub-pixel precision. Multi-view weights are at full strength.

### Phase Transitions

At each phase boundary, the optimizer and scheduler are **recreated**:
- Optimizer: AdamW with `betas=(0.5, 0.95)`, `weight_decay=1e-4`
- Scheduler: CosineAnnealingLR per phase (T_max = phase duration)

---

## Hardware Profiles

### Default Profile

For consumer GPUs (RTX 3090, 4090, etc.):

```
batch_size:         512
mv_groups:          2          (18 samples in multi-view phases)
num_workers:        4
AMP:                off
grad_accum_steps:   1
torch.compile:      off
TF32:               off
```

### A100 Profile

For NVIDIA A100 80GB:

```
batch_size:         2048
mv_groups:          16         (144 samples in multi-view phases)
num_workers:        8
AMP:                fp16
grad_accum_steps:   2          (effective batch: 4096)
torch.compile:      on
TF32:               on
prefetch_factor:    4
persistent_workers: on
```

**A100 optimizations applied automatically:**
- `torch.backends.cuda.matmul.allow_tf32 = True`
- `torch.backends.cudnn.allow_tf32 = True`
- `torch.set_float32_matmul_precision('high')`
- Model wrapped with `torch.compile()`
- Training uses `GradScaler` + `autocast(dtype=float16)`

### Selecting a Profile

```bash
# Default
python -m RayNet.train --data_dir /data --profile default

# A100
python -m RayNet.train --data_dir /data --profile a100
```

CLI flags override profile values:
```bash
python -m RayNet.train --profile a100 --batch_size 1024 --no_compile
```

---

## Streaming Training (HF Hub)

For cloud/Colab training without local dataset copies:

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

## Google Colab A100 Notebook

A ready-to-run notebook is provided at `notebooks/train_colab_a100.ipynb`.

### Setup

1. Open in Colab and change runtime to **A100 GPU**
2. Edit `CONFIG['hf_repo_id']` to point to your dataset
3. Run all cells

### What the Notebook Does

| Cell | Action |
|------|--------|
| 1 | Verify A100 GPU and VRAM |
| 2 | Install deps, clone repo, HF login |
| 3 | A100-optimized config (batch 2048, fp16, grad accum 2, torch.compile) |
| 4 | Load dataset (streaming from HF Hub or local) |
| 5 | Create model with torch.compile |
| 6 | Full 3-phase training loop with AMP |
| 7 | Loss curve plots |
| 8 | Save to Google Drive or push to HF Hub |
| Appendix | One-time shard creation from local dataset |

---

## CLI Reference

```
python -m RayNet.train [OPTIONS]

Data:
  --data_dir PATH               Path to GazeGene dataset (required if not --streaming)
  --output_dir PATH             Output directory (default: ./results)
  --samples_per_subject INT     Max frames per subject (default: 200)
  --eye {L,R}                   Which eye (default: L)

Streaming:
  --streaming                   Use WebDataset streaming
  --dataset_url URL             Train shard URL pattern
  --val_dataset_url URL         Val shard URL pattern

Model:
  --backbone NAME               repnext_m0 through repnext_m5 (default: repnext_m3)
  --weight_path PATH            Pretrained backbone weights

Hardware:
  --profile {default,a100}      Hardware profile (default: default)
  --no_compile                  Disable torch.compile

Overrides:
  --batch_size INT              Override profile batch size
  --epochs INT                  Total epochs (default: 30)
  --num_workers INT             Dataloader workers
  --mv_groups INT               Multi-view groups per batch (batch = mv_groups * 9)
  --grad_accum_steps INT        Gradient accumulation steps
```

---

## Output Directory

Each run creates a timestamped directory:

```
results/run_20260403_143000/
├── training_log.csv            # Per-epoch metrics
├── best_model.pt               # Lowest validation loss
└── checkpoint_epoch{5,10,...}.pt  # Periodic checkpoints
```

### Checkpoint Format

```python
{
    'epoch': int,
    'phase': int,
    'model_state_dict': OrderedDict,
    'val_loss': float,
    'val_landmark_px': float,         # best_model.pt only
    'val_angular_deg': float,         # best_model.pt only
    'optimizer_state_dict': OrderedDict,  # periodic checkpoints only
    'scaler_state_dict': dict,            # if AMP was used
    'profile': str,
}
```

### Loading a Checkpoint

```python
from RayNet.raynet import create_raynet

model = create_raynet(backbone_name='repnext_m3', n_landmarks=14)
ckpt = torch.load('results/run_xxx/best_model.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
```

---

## Training Log CSV Columns

| Column | Description |
|--------|-------------|
| `epoch` | Epoch number |
| `phase` | Training phase (1, 2, or 3) |
| `lr` | Current learning rate |
| `train_total` | Total training loss |
| `train_landmark` | Landmark loss component |
| `train_angular_deg` | Gaze angular error in degrees |
| `train_reproj` | Reprojection consistency loss (0 in Phase 1) |
| `train_mask` | Triangulation masking loss (0 in Phase 1) |
| `val_total` | Total validation loss |
| `val_landmark` | Validation landmark loss |
| `val_angular_deg` | Validation gaze error in degrees |
| `val_landmark_px` | Validation landmark error in pixels (224x224 space) |

---

## Gradient Clipping

All training uses `clip_grad_norm_(max_norm=1.0)` to prevent gradient explosions, particularly important when multi-view losses are introduced in Phase 2.

## Mixed Precision Notes

- **Forward + loss**: computed under `autocast(dtype=float16)`
- **Backward**: scaled by `GradScaler` to prevent underflow
- **Optimizer step**: unscaled, then clipped, then stepped via `scaler.step()`
- **Multi-view geometry** (matrix inversions, SVD): remains numerically stable because the triangulation output is detached and camera parameters are float32

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM on A100 | Reduce `--batch_size` or increase `--grad_accum_steps` |
| NaN loss in Phase 2 | Reduce `lam_reproj` to 0.05, check camera calibration |
| Slow first epoch with torch.compile | Normal. Compilation happens on first forward pass (~60s) |
| Multi-view loader too slow | Increase `--num_workers`, use `persistent_workers` |
| Streaming stalls | Check HF Hub URL pattern, ensure shards exist |
