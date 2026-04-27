# Architecture

RayNet v5 (current) — **Triple-M1 + AERI + HRFH-α**. Three task-specific RepNeXt-M1 stage-2/3 encoders (landmark, pose, gaze) sit on a shared stem (RepNeXt-M1 stem + stages[0..1]). The gaze branch carries a private mini U-Net (`AERIHead`) that produces iris and eyeball binary masks at 56×56 plus a 48-channel decoder tensor `d1`. The masks are combined into a saliency, blended with a uniform field through an α schedule, and used twice: once at 7×7 to gate the gaze bottleneck (global features), once at 56×56 to gate `d1` (foveal features). The two pooled vectors are concatenated and projected into the 256-d gaze feature consumed by `GazeFusionBlock`. Source: `RayNet/raynet_v5.py`, `RayNet/streaming/eye_masks.py`.

## Why Triple-M1 + AERI replaced Quad-M1

The earlier Quad-M1 cascade (eye-crop branch + landmark refinement head, see `docs/experiments/raynet_QUAd-M1_Stage2_New_Arch_500_samples_per_subject/`) had the differentiable affine eye-crop generating interpolation artefacts on val (train/val distribution gap from the affine grid being deterministic in train but operating on jittered landmark predictions in val), and the landmark refinement head was operating on a feature pyramid that already had to satisfy gaze. The eye crop itself drove a hard 9-13° val_angular floor that loss reweighting did not move.

**AERI** removes the differentiable crop and replaces it with a soft mask gate at 56×56. The gaze branch shares the full-face crop with landmark/pose, so train and val see the same input distribution. Iris/eyeball segmentation is a clean MSGazeNet-style auxiliary signal supervised by per-frame masks baked into the MDS shards. **HRFH-α** then harvests the high-resolution `d1` tensor (48ch × 56² = 16× the token density of the 384ch × 7² bottleneck), gates it with the same scheduled saliency, and produces a 48-d foveal vector that captures sub-pixel iris/pupil dynamics — the part Quad-M1 was getting from its dedicated landmark refinement head.

The progressive-training advantage that Quad-M1 enjoyed (pose-only Phase 1 → full Phase 2 with reproj/mask) is now replicated by the **branch-staged curriculum** in `train.py`: Phase 1 freezes the gaze branch and trains landmark + AERI seg + headpose; Phase 2 freezes everything except gaze + AERI for monocular gaze; Phase 3 unfreezes all branches and turns on multi-view fusion at a 5×–10× lower LR than P2. See `RayNet/README.md` for the schedule table.

## HRFH-α (High-Resolution Feature Harvesting with Alpha Scheduling)

A standard architecture regresses gaze from the stride-32 bottleneck (`(B, 384, 7, 7)`), where the iris occupies 1-3 cells. That ceiling appeared as a stubborn ~20° val_angular asymptote in the v5.0 Triple-M1 fork. HRFH harvests features at 56×56 instead and does so on a clean (segmented) eye region:

1. **AERIHead** decodes the gaze branch's pyramid (s3=384@7, s2=192@14) up to 56×56 with attention-gated U-Net blocks. It emits two binary mask logits (iris, eyeball) and exposes the 48-channel `d1` decoder tensor.
2. **Saliency**: `0.8·sigmoid(iris) + 0.2·sigmoid(eyeball)` — iris-centric, prioritising the high-entropy pupil/iris boundary.
3. **α schedule** (`get_scheduled_alpha(epoch)` in `train.py`): `scheduled_mask = α·saliency + (1−α)·1`. Low α at warmup keeps the gaze pathway looking at the global field; α is held constant in fine-tune (the previously-shipped 0.4→0.9 ramp during the cosine LR decay caused validation drift; see `docs/experiments/triple_m1_aeri_iris_eyeball_500spc_run_20260423_101115`).
4. **Stochastic mask dropout** (training only, 10% of batches): replace the scheduled mask with the uniform field. Prevents the gaze head from overfitting to specific pixel-mask correlations.
5. **Two-scale gating**:
   - Pool the scheduled mask to 7×7 → multiply the gaze bottleneck `(384, 7, 7)` → CoordAtt → AdaptiveAvgPool → 384-d **global vector**.
   - Multiply the 56×56 scheduled mask onto `d1` → AdaptiveAvgPool → 48-d **foveal vector**.
6. Concatenate `[global ‖ foveal]` (432-d), `LayerNorm`, `Linear → 256` → `gaze_feat` consumed by `GazeFusionBlock`.

The 16× higher token density of the foveal path (48 channels × 3136 cells vs. 384 channels × 49 cells) is what gives the gaze branch its sub-pixel iris signal — replacing the role the Quad-M1 LandmarkRefinementHead used to play.

## Diagram

```
Input: (3, 224, 224) face crop  +  (3,) face bbox (x_p, y_p, L_x)  [optional at inference]
                         │
                ┌────────▼─────────┐
                │   SharedStem     │  RepNeXt-M1 stem + stages[0..1]
                │ 3→48→96ch, 28×28 │  ~0.21M params; landmark-owned
                └─┬───────┬───────┬┘
                  │       │       │
       s0,s1 (lm) │  s1.detach()  │  s0.detach(), s1.detach()
                  ▼       ▼       ▼
   ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────────────┐
   │ LandmarkBranch   │  │ PoseBranch      │  │ GazeBranch               │
   │ M1 s2+s3 +       │  │ M1 s2+s3 +      │  │ M1 s2+s3 +               │
   │ U-Net @56 +      │  │ CoordAtt +      │  │ AERIHead (mini U-Net) +  │
   │ AttGates +       │  │ pool + proj +   │  │ HRFH-α gating @ 7 + 56 + │
   │ heatmap+offset   │  │ BoxEncoder      │  │ CoordAtt + pool +        │
   │                  │  │ (zero-init      │  │ Concat[global,foveal] +  │
   │                  │  │  residual)      │  │ LN + Linear → 256        │
   │                  │  │ → pose_feat,    │  │ → GazeFusionBlock(pose)  │
   │ → 14 landmarks   │  │   6D, t (m)     │  │ → CrossViewAttention(*)  │
   │   @ 56×56        │  │                 │  │ → GeometricGazeHead      │
   └──────────────────┘  └────────┬────────┘  └──────────┬───────────────┘
                                  │                       │
                                  └─────► pose_feat ─────►│
                                                          ▼
                                       eyeball_center, pupil_center
                                       optical_axis = normalize(pupil − eyeball)

  (*) CrossViewAttention identity short-circuits when n_views == 1.
```

## Components

### SharedStem

RepNeXt-M1 stem (3→48ch, stride 4, 56×56) + stage[0] (48→48ch, 56×56) + stage[1] (48→96ch, 28×28). Intermediate maps are exposed as skip connections for the landmark U-Net decoder. ~0.21M params.

### Task Branches

- **Landmark branch**: RepNeXt-M1 stages[2..3] + U-Net decoder with attention gates. Upsamples 7×7 → 56×56 with skips from shared-stem intermediates. Predicts 14 heatmaps → soft-argmax + learned offset refinement. **Owns the shared stem** — the only branch whose loss reaches `s0`/`s1`. ~6.18M params.
- **Pose branch**: RepNeXt-M1 stages[2..3] on `s1.detach()`. CoordAtt + pool + projection, fused with `BoxEncoder(face_bbox)` via a zero-init residual. Predicts 6D rotation (Gram-Schmidt) + 3D translation. The `face_bbox` argument is **optional**: when `None` the BoxEncoder fork zeroes out and pose collapses to CNN features (see `PoseBranch.forward`). ~4.69M params.
- **Gaze branch**: RepNeXt-M1 stages[2..3] on `s1.detach()` + `AERIHead` (mini U-Net) + HRFH-α gating + `GazeFusionBlock` + `CrossViewAttention` (when applicable) + `GeometricGazeHead`. ~6.53M params.

### AERIHead

Mini U-Net decoder operating on the gaze branch's own pyramid. Uses `s0.detach()` and `s1.detach()` from the shared stem as skip connections (gradient blocked so AERI loss does not leak into the landmark-owned stem). Returns `(iris_logits, eyeball_logits, d1)` where `d1` is the 48-channel 56×56 decoder tensor. Supervised by per-frame iris and eyeball binary masks baked into the MDS shards (`streaming/eye_masks.py`).

### GazeFusionBlock

Zero-init residual fusion with the post-HRFH gaze feature as the anchor:

```
residual = MLP([gaze_feat, pose_feat])    # last linear is zero-init
out      = gaze_feat + residual
```

Pose features ramp in via gradient signal rather than being baked in by a pretrained bridge. `pose_feat` is detached before being added to the optional `cam_embed` so gaze loss cannot backpropagate into PoseBranch through the cross-view side path (see `RayNetV5.forward`).

### BoxEncoder (MAGE)

`(x_p, y_p, L_x) → 64 → 128 → 256` MLP with GELU. Lives **inside `PoseBranch`** (head pose and face bbox are both rigid-geometry cues; keeping bbox inside pose removes redundant inputs from the gaze fusion block). At inference the bbox is either synthesised from a detected face box assuming a centred principal point (`inference.mage_bbox_from_pixels`) or omitted entirely.

### CrossViewAttention

2-layer pre-norm Transformer encoder for cross-view gaze fusion. Identity short-circuit when `n_views <= 1`. Exercised when batches are delivered as N-grouped multi-view tuples (see `streaming.create_multiview_streaming_dataloaders`).

## Forward Pass

```python
out = model(
    images,              # (B, 3, 224, 224), B = mv_groups * n_views
    n_views=9,
    R_cam=R_cam,         # (B, 3, 3) — optional
    T_cam=T_cam,         # (B, 3)    — optional
    face_bbox=face_bbox, # (B, 3) (x_p, y_p, L_x) — optional
    aeri_alpha=0.7,      # 0..1 saliency vs. uniform-field blend
)
```

Internally (`RayNetV5.forward`):

1. `s0, s1 = SharedStem(images)`.
2. `landmark_coords, landmark_heatmaps = LandmarkBranch(s0, s1)` — only path that backprops into the stem.
3. `pose_feat, pose_6d, pose_t = PoseBranch(s1.detach(), face_bbox)`.
4. `cam_embed = CameraEmbedding(R_cam, T_cam) + pose_feat.detach()` (when extrinsics passed).
5. `iris_logits, eyeball_logits, d1 = AERIHead(s0.detach(), s1.detach(), gaze_s2, gaze_s3)`.
6. `scheduled_mask = α·(0.8·sigmoid(iris) + 0.2·sigmoid(eyeball)) + (1−α)·1`.
7. Optional stochastic mask dropout (training only).
8. `global = pool(CoordAtt(gaze_s3 · pool₇(scheduled_mask)·0.75 + 0.25))` — 384-d.
9. `foveal = pool(d1 · scheduled_mask)` — 48-d.
10. `gaze_feat = Linear(LayerNorm(Concat[global, foveal]))` — 256-d.
11. `pooled_sv = GazeFusionBlock(gaze_feat, pose_feat)`.
12. `pooled = CrossViewAttention(pooled_sv, n_views, cam_embed)`.
13. `eyeball_center, pupil_center, optical_axis = GeometricGazeHead(pooled)` and (when `n_views > 1`) `sv_optical_axis = GeometricGazeHead(pooled_sv)`.

## Outputs

| Key | Shape | Description |
|-----|-------|-------------|
| `landmark_heatmaps` | (B, 14, 56, 56) | per-landmark heatmaps (face-frame) |
| `landmark_coords` | (B, 14, 2) | subpixel `(x, y)` in 56×56 face space |
| `iris_mask_logits` | (B, 56, 56) | AERI iris segmentation logits |
| `eyeball_mask_logits` | (B, 56, 56) | AERI eyeball segmentation logits |
| `eyeball_center` | (B, 3) | camera-space, cm |
| `pupil_center` | (B, 3) | camera-space, cm |
| `gaze_vector` | (B, 3) | unit optical axis = normalize(pupil − eyeball) |
| `gaze_vector_sv` | (B, 3) or None | single-view (pre-CrossViewAttention) optical axis (only when `n_views > 1`) |
| `gaze_angles` | (B, 2) | pitch, yaw |
| `pred_pose_6d` | (B, 6) | Gram-Schmidt rotation |
| `pred_pose_t` | (B, 3) | translation (meters, direct linear) |

`gaze_vector_sv` exists so the GeometricGazeHead is supervised on the single-view pathway too — validation runs `n_views=1`, which bypasses CrossViewAttention. Without `lam_gaze_sv > 0` in P3 the val metric would lag the train metric by a wide margin.

## Tensor Shapes

```
Shared stem      -> s0:(B,48,56,56), s1:(B,96,28,28)
Branch stages[2] -> (B, 192, 14, 14)
Branch stages[3] -> (B, 384,  7,  7)
Landmark decoder -> (B, 14, 56, 56) heatmaps
AERI decoder     -> iris/eyeball logits (B, 56, 56), d1 (B, 48, 56, 56)
Gaze fusion      -> [global(384) ‖ foveal(48)] → LN → Linear → (B, 256)
Pose head        -> (B, 6), (B, 3)
Gaze head        -> (B, 3), (B, 3), (B, 3)
```

## Parameter Budget

| Module | Params |
|--------|--------|
| SharedStem | 0.21M |
| LandmarkBranch (M1 s2+s3 + U-Net + heads) | 6.18M |
| PoseBranch (M1 s2+s3 + CoordAtt + BoxEncoder + head) | 4.69M |
| GazeBranch (M1 s2+s3 + AERIHead + HRFH fusion + GeometricGazeHead) | 6.53M |
| CrossViewAttention + CameraEmbedding | 1.07M |
| **Total** | **~18.7M** |

## Gradient Flow

- Task branches do not share high-level weights — each owns its own RepNeXt-M1 stages[2..3].
- Only the **landmark branch's** loss reaches the shared stem. Pose branches see `s1.detach()`. Gaze branch sees `s0.detach()` and `s1.detach()`.
- `pose_feat` is detached before being added to `cam_embed` so gaze loss cannot leak into PoseBranch through the cross-view side path.
- `GazeFusionBlock` last linear is zero-init: at initialisation `gaze_feat` flows through unchanged and the pose contribution ramps in via gradient signal.
- `BoxEncoder`'s residual is also zero-init: the model trains correctly with `face_bbox=None` because that branch contributes zero at init and the optimiser learns to use it only when bbox is present.

## Three-Phase Branch-Staged Curriculum

Replaces the v5.0 "all losses on from epoch 1" parallel-MTL schedule. The detach-based gradient isolation is preserved; the **freeze schedule** and **loss-weight schedule** together implement progressive training:

| Phase | Active branches | Frozen modules | Multi-view | LR |
|-------|-----------------|----------------|------------|----|
| 1 (1-8)   | landmark + AERI seg + headpose                      | gaze branch (encoder + fusion + head) — AERI head stays trainable | off | 5e-4 |
| 2 (9-18)  | gaze branch + AERI fine-tune                        | shared stem + landmark + pose                                     | off (n_views=1) | 3e-4 |
| 3 (19-35) | all                                                 | none                                                              | on  | 3e-5 — 5e-5 |

`set_face_frozen`-style helpers must do `requires_grad_(False)` **and** `.eval()` on the frozen modules (especially BatchNorm) so running stats do not drift under the new training distribution.

DDP's default reducer expects every parameter to receive a gradient every step, so `hardware_profiles.build_accelerator()` constructs the `Accelerator` with `DistributedDataParallelKwargs(find_unused_parameters=True)`.

## Validation EMA

Validation runs through a `torch.optim.swa_utils.AveragedModel` (weight EMA, decay `--ema_decay`, default 0.999). Only parameters are EMA'd; BN running stats are mirrored from the live model after every optimiser step so the shadow model uses consistent statistics. This smooths out per-step weight noise in the reported metrics.

## Inference

`RayNet/inference.py` runs MediaPipe (Haar fallback) inside the module — the caller passes a full frame, the module detects, square-crops with a 1.3× expansion (matching the GazeGene crop convention), runs the v5 model, and visualises landmarks, AERI iris/eyeball masks, gaze arrow from the eye-center, RGB pose axes, and the bounding box that was actually fed to the model. `face_bbox` is synthesised from the detected pixels under a centred-principal-point assumption (`mage_bbox_from_pixels`).

## Key Differences from v5.0 (Triple-M1, parallel MTL) and Quad-M1

- **AERI replaces the differentiable eye crop.** No `EyeCropModule`, no `EyeBackbone`, no `LandmarkRefinementHead`. The eye-region inductive bias is now a soft 56×56 mask gate; the sub-pixel iris signal that the LandmarkRefinementHead used to provide is now produced by HRFH harvesting on the AERI `d1` tensor.
- **HRFH-α gating** (`d1` × scheduled saliency → 48-d foveal vector) added on top of the existing 7×7 bottleneck gating.
- **Branch-staged curriculum** (Phase 1 freezes gaze, Phase 2 freezes everything except gaze + AERI, Phase 3 fine-tunes all). v5.0's parallel MTL is retained as a fallback (set all phase configs to "no freeze") but is no longer the default.
- **`face_bbox` is optional at inference.** BoxEncoder residual is zero-init so omitting it leaves PoseBranch correct.
- BoxEncoder retained; bridge toggles `use_landmark_bridge` / `use_pose_bridge` are kept only for config back-compat — coarse landmarks are always used to anchor the eye crop, and `use_pose_bridge=False` zeros the pose stream into the fusion block.
- Stage 2 face-freeze is now permanent across P1–P3 (no unfreeze in P3). The gaze branch + refinement head have enough capacity to refine on their own; unfreezing risks moving the coarse-landmark distribution out from under the stable eye crop.
