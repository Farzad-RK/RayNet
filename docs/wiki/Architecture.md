# Architecture

RayNet v5 — Triple-M1 multi-task architecture with a shared low-level encoder and three dedicated RepNeXt-M1 branches. Source: `RayNet/raynet_v5.py`.

## Diagram

```
Input: (3, 224, 224) face crop + (3,) face bbox (x_p, y_p, L_x)
                         │
                ┌────────▼─────────┐
                │   SharedStem     │  RepNeXt-M1 stem + stages[0..1]
                │ 3→48→96ch, 28x28 │  ~1.5M params
                └────────┬─────────┘
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
  Landmark Branch   Gaze Branch    Pose Branch  (RepNeXt-M1 stages[2..3], ~3.3M each)
        │                │               │
        │                ▼               ▼
        │          SHMA modulation   6D rot + 3D trans head
        │          ← pose features   (operates on detached shared-stem features)
        │                ▲
        │                │  Landmark cross-attention (zero-init bridge)
        │                │
        ▼                ▼
  U-Net decoder    Eyeball center head
  + attention      Pupil center head
  gates            → optical_axis = normalize(pupil − eyeball)
  (14 landmarks,
   56x56 heatmaps)
                         ▲
                ┌────────┴─────────┐
                │  MAGE BoxEncoder + FusionBlock
                │  (x_p, y_p, L_x) → 256d mixed with pose features
                └──────────────────┘
```

## Components

### SharedStem

RepNeXt-M1 stem (3→48ch, stride 4, 56×56) + stage[0] (48→48ch, 56×56) + stage[1] (48→96ch, 28×28). Intermediate maps are exposed as skip connections for the landmark U-Net decoder.

### Task Branches

Each branch runs RepNeXt-M1 stages[2..3] (96→192→384ch, 7×7) starting from the shared 96-channel feature. Branches are independent to prevent gradient conflict across tasks.

- **Landmark branch**: U-Net decoder with attention gates upsamples from 7×7 back to 56×56 with skips from the shared-stem intermediates. Predicts 14 heatmaps → soft-argmax + learned offset refinement.
- **Gaze branch**: global-pooled 384d features → eyeball_center and pupil_center heads (both 3D camera-space). Optical axis is derived analytically as `normalize(pupil − eyeball)`. Enriched by pose-conditioned SHMA modulation and landmark cross-attention.
- **Pose branch**: operates on a gradient-detached copy of the shared stem output. Predicts 6D rotation (Gram-Schmidt) + 3D translation.

### MAGE Integration

- **BoxEncoder**: `(x_p, y_p, L_x)` → 64 → 128 → 256 via 3 linear layers + GELU.
- **FusionBlock**: zero-init residual that adds BoxEncoder output to pose features, providing gaze-origin information without requiring 468-point MediaPipe landmarks at inference.

### CrossViewAttention

9-camera attention conditioned on `R_cam` and `T_cam`. Exercised when batches are delivered as 9-grouped multi-view tuples (see `streaming.create_multiview_streaming_dataloaders`).

## Forward Pass

```python
out = model(
    images,                 # (B, 3, 224, 224), B = mv_groups * 9
    n_views=9,
    R_cam=R_cam,            # (B, 3, 3)
    T_cam=T_cam,            # (B, 3)
    face_bbox=face_bbox,    # (B, 3) (x_p, y_p, L_x)
    use_landmark_bridge=True,
    use_pose_bridge=True,
)
```

Outputs:

| Key | Shape | Description |
|-----|-------|-------------|
| `landmark_heatmaps` | (B, 14, 56, 56) | per-landmark heatmaps |
| `landmark_coords` | (B, 14, 2) | subpixel `(x, y)` |
| `eyeball_center_3d` | (B, 3) | camera-space |
| `pupil_center_3d` | (B, 3) | camera-space |
| `optical_axis` | (B, 3) | unit vector |
| `pose_6d` | (B, 6) | Gram-Schmidt rotation |
| `pose_t` | (B, 3) | translation (tanh + exp) |

## Tensor Shapes

```
Shared stem       -> (B, 96,  28, 28)
Branch stages[2]  -> (B, 192, 14, 14)
Branch stages[3]  -> (B, 384,  7,  7)
Landmark decoder  -> (B, 14,  56, 56)
Gaze head         -> (B, 3), (B, 3), (B, 3)
Pose head         -> (B, 6), (B, 3)
```

## Parameter Budget

| Module | Params |
|--------|--------|
| SharedStem | ~1.5M |
| Landmark branch + U-Net decoder | ~4.0M |
| Gaze branch + heads + bridges | ~3.9M |
| Pose branch + head | ~3.5M |
| BoxEncoder + FusionBlock | ~0.3M |
| CrossViewAttention | ~3.9M |
| **Total** | **~17.1M** |

## Gradient Flow

- Task branches do not share high-level weights — each owns stages[2..3].
- The pose branch reads shared-stem features through `.detach()` so pose-only gradients cannot perturb the shared encoder.
- Landmark and gaze bridges are **zero-init** residuals: they contribute nothing at initialization but can be learned from epoch 1, avoiding the cold-start schedule v4 needed.

## Key Differences from v4.1

- No PANet — each branch carries its own high-level encoder path.
- U-Net decoder with attention gates replaces heatmap-on-P2.
- Explicit 3D eyeball geometry replaces black-box pitch/yaw regression.
- MAGE BoxEncoder/FusionBlock replaces CameraEmbedding.
- Bridges active from epoch 1 (zero-init); no `use_bridge` toggle.
- Single RepNeXt-M1 variant used for all branches.
