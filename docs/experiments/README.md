# Experiments

Each subdirectory is one training run — its `training_log.csv`, `batch_log.csv`, and (when recorded) `metadata.json` are the source of truth. This page summarises the two pivotal runs that drove the move from the old Triple-M1 design to the current Quad-M1 eye-crop gaze architecture.

## `raynet_v5_500_samples_per_subject/` — Stage 1 baseline (2026-04-18)

Stage 1 landmark + pose baseline on the Quad-M1 model with an aggressive data budget.

| Setting | Value |
|---------|-------|
| Stage | 1 (landmark + pose, gaze disabled) |
| Samples / subject | 500 |
| Eye | L |
| Epochs | 15 (P1: 1-8, P2: 9-15) |
| Profile | `kaggle_t4x2` (2× T4, per-GPU batch 144, fp16, grad_accum 2) |
| Phase LR | P1 lr 1e-3, P2 lr 3e-4 (CosineAnnealingLR per phase) |
| Loss weights | P1: lam_lm=1.0, lam_pose=0.5, lam_trans=0.5; P2: lam_lm=1.0, lam_pose=1.0, lam_trans=1.0 |
| Wall time | ~10.7 h |

### Result

`val_landmark_px` drops from 7.92 (E1) to **2.64** (E14, best). `val_angular_deg` stays at ≈ 42.5° throughout because the gaze branch gets no gradient in Stage 1 (`lam_gaze=0`).

```
Epoch:    1     2     3     4     5     6     7     8     9    10    11    12    13    14    15
val_px: 7.92  4.80  4.38  4.17  3.37  3.16  3.00  2.95  3.07  2.96  3.15  2.83  2.67  2.64  2.64
val_ang:42.7  42.7  42.7  42.6  42.6  42.6  42.6  42.6  42.6  42.5  42.5  42.5  42.5  42.6  42.5
```

### Takeaway

500 samples/subject on the Quad-M1 face path is ≈ **33% better** on iris-px than the earlier 200-samples baseline (3.93 → 2.64). This is the Stage 1 checkpoint we fork Stage 2 from.

## `raynet_v5_S2_fork_500_samples_per_subject/` — Stage 2 fork that revealed the 42° ceiling (2026-04-18)

Stage 2 fork of the run above, **using the old Triple-M1 gaze design** (gaze shared the stride-16 face feature map with landmark/pose). We stopped this run at epoch 8 to diagnose a stagnant `val_angular_deg`.

| Setting | Value |
|---------|-------|
| Stage | 2 fork (Triple-M1 gaze) |
| Fork source | `raynet_v5_500_samples_per_subject` E14 best |
| Epochs observed | 8 (P1: 1-5, P2: 6-8, stopped mid-P2) |
| Profile | `kaggle_t4x2` |

### Result

```
Epoch:        1      2      3      4      5      6      7      8
val_ang:   66.3   56.5   48.0   55.1   26.2   44.6   41.8   42.4    # degrees
val_px:    3.30   3.54   2.70   2.62   2.57   2.84   2.89   2.58
```

`val_angular_deg` settled into a narrow band around 42-45° after the multi-view ramp kicked in (P2). The brief E5 dip to 26° coincided with the final LR-0 step of P1's cosine schedule and did not persist into P2 — the gaze branch never broke meaningfully below the ~42° floor.

### Takeaway

The ceiling is a spatial-resolution bottleneck, not a loss-weighting problem. The gaze branch in the old Triple-M1 design consumed the same `(B, 384, 7, 7)` high-level feature map produced from the 224×224 face — the iris occupies 2-3 cells of that map, which is not enough for gaze discrimination. This run drove the pivot to the Quad-M1 eye-crop design (see [[Architecture]]): the gaze branch now owns a full private RepNeXt-M1 fed by a 112×112 landmark-guided eye crop, and Stage 2 P1/P2 freeze the face path so the gaze encoder trains against a stable predicted-landmark distribution.

## Summary of the pivot

```
Triple-M1 (v5.0)                           Quad-M1 (v5, current)
─────────────────                          ──────────────────────
shared_stem + 3 branches                   shared_stem + landmark + pose
gaze shares stages[2..3] on face           gaze owns full private M1 on 112×112 eye crop
bridges: landmark x-attn + pose SHMA       bridges deleted
Stage 2: multitask from scratch            Stage 2 P1/P2: freeze_face, gaze-only training
                                           Stage 2 P3: unfreeze for joint fine-tuning
val_angular floored at ~42°                — target: break the 42° ceiling
```
