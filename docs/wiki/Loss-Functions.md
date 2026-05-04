# Loss Functions

All loss functions used in RayNet v5/v6 training.

> **v6 update (2026-05).** Three new losses ship with the decoupled-pipeline
> refactor: `Iris_Mesh_Loss` + `Iris_Edge_Loss` (3DGazeNet M-target) and
> `Visual_Axis_Loss` (kappa-corrected gaze). Two existing gaze readouts are
> now mean-of-two fused (`gaze_geom`, `gaze_direct`, `gaze_fused`) with
> independent sub-supervisions. AERI segmentation losses (`iris_seg`,
> `eyeball_seg`) are zeroed across all GazeGene phases — foveal-texture
> supervision moved to the OpenEDS-only stage to prevent MetaHuman
> texture-poisoning.

## Overview

```
Total Loss = lam_lm           * Landmark_Loss
           + lam_gaze         * Gaze_Loss            (canonical: gaze_fused)
           + lam_gaze_geom    * Gaze_Loss(gaze_geom)
           + lam_gaze_direct  * Gaze_Loss(gaze_direct)
           + lam_gaze_visual  * Visual_Axis_Loss     (kappa-corrected)
           + lam_eyeball      * Eyeball_Center_Loss
           + lam_pupil        * Pupil_Center_Loss
           + lam_geom_angular * Geometric_Angular_Loss
           + lam_iris_mesh    * Iris_Mesh_Loss       (3DGazeNet vertex L1)
           + lam_iris_edge    * Iris_Edge_Loss       (3DGazeNet edge L2)
           + lam_pose         * Pose_Rotation_Loss (geodesic)
           + lam_trans        * Pose_Translation_Loss (log-depth SmoothL1)
           + lam_ray          * Ray_Target_Loss
           + mv_weight * (lam_reproj * Gaze_Ray_Consistency
                        + lam_mask  * Landmark_Shape_Consistency)
```

---

## 1. Landmark Loss

**Source**: `RayNet/losses.py:landmark_loss`

Combines heatmap regression with coordinate regression for robust landmark detection.

### Heatmap MSE

Ground-truth heatmaps are generated as 2D Gaussians centered at each landmark:

```
GT_heatmap(x, y) = exp(-((x - cx)^2 + (y - cy)^2) / (2 * sigma^2))
```

Where `(cx, cy)` is the ground-truth landmark position in feature-map space (56x56).

```
L_heatmap = MSE(sigmoid(pred_heatmaps), GT_heatmaps)
```

**Sigma schedule**: starts at 2.0 (Phase 1, broader targets) and tightens to 1.0 (Phase 3, sub-pixel precision).

### Coordinate L1

Direct L1 regression on predicted vs ground-truth coordinates in feature-map space (0 to 56):

```
L_coord = L1(pred_coords, gt_coords)
```

### Combined

```
L_landmark = L_heatmap + L_coord
```

---

## 2. Gaze Loss (mean-of-two fused readout)

**Source**: `RayNet/losses.py:gaze_loss`

L1 on predicted vs ground-truth unit gaze vectors in CCS. This follows the GazeGene paper (Sec 4.1.1).

```
L_gaze = L1(pred_gaze, gt_gaze)     (both unit vectors in CCS)
```

### Mean-of-two readouts (v6)

`GeometricGazeHead` emits **three** gaze vectors that are unit-normalised independently:

| Key | Definition | Failure mode it covers |
|---|---|---|
| `gaze_geom` | `normalize(pupil_center - eyeball_center)` | Anchor predictions accurate but direct head saturated under heavy head pose. |
| `gaze_direct` | `normalize(direct_fc(pooled))` | Profile/occluded views where the 3D anchors are noisy. |
| `gaze_fused` | `normalize(gaze_geom + gaze_direct)` | Canonical training/inference signal. Aliased as `gaze_vector`. |

This mirrors 3DGazeNet (Sec 7, supplementary Fig 7d), which reports the mean-of-two consistently outperforming either head in isolation on profile-view examples.

The training loss decomposes:

```
lam_gaze         * gaze_loss(gaze_fused, gt)         # canonical
+ lam_gaze_geom    * gaze_loss(gaze_geom,   gt)        # keeps geom head honest
+ lam_gaze_direct  * gaze_loss(gaze_direct, gt)        # keeps direct head honest
+ lam_gaze_visual  * visual_axis_loss(...)             # see Section 11
```

Independent sub-supervisions on `gaze_geom` and `gaze_direct` prevent one head from collapsing into a passthrough of the other.

### Why L1 instead of angular (acos)?

`torch.acos(cosine_similarity)` has an infinite gradient singularity at `cos_sim = ±1`:

```
d/dx acos(x) = -1/sqrt(1 - x^2)  -->  infinity when x = +/-1
```

L1 on unit vectors is numerically stable everywhere, consistent with the GazeGene paper, and monotonically related to angular error.

### Angular error (metrics only)

Angular error is computed for logging using the `atan2` formulation, but is **not backpropagated**:

```python
cross = cross_product(pred, gt)
dot = dot_product(pred, gt)
angle = atan2(||cross||, dot)       # stable everywhere
```

**Source**: `RayNet/losses.py:angular_error`

---

## 3. Eyeball Center Loss

**Source**: `RayNet/losses.py:eyeball_center_loss`

L1 loss on predicted vs GT eyeball center in camera coordinate space (CCS, centimeters). From GazeGene Sec 4.2.2, loss 1.

```
L_eyeball = L1(pred_eyeball_3d, gt_eyeball_3d)     (both in CCS, cm)
```

Active in Stage 2+ (off in Stage 1 where gaze is disabled).

---

## 4. Pupil Center Loss

**Source**: `RayNet/losses.py:pupil_center_loss`

L1 loss on predicted vs GT pupil center in CCS (centimeters). From GazeGene Sec 4.2.2, loss 2.

```
L_pupil = L1(pred_pupil_3d, gt_pupil_3d)     (both in CCS, cm)
```

Active in Stage 2+ (off in Stage 1).

---

## 5. Geometric Angular Loss

**Source**: `RayNet/losses.py:geometric_angular_loss`

Angular error between the optical axis derived from predicted 3D geometry and the GT optical axis (GazeGene Sec 4.2.2, loss 4).

```
pred_optical_axis = normalize(pred_pupil - pred_eyeball)
L_geom_angular = angular_error(pred_optical_axis, gt_optical_axis)
```

This ensures geometric consistency — the model cannot cheat by predicting correct eyeball/pupil positions but wrong relative direction. Uses `atan2` for numerical stability.

Delayed until Phase 3 in Stage 2 (geometry must converge first). Active earlier in Stage 3 where bridges provide additional structure.

---

## 6. Pose Rotation Loss (Geodesic)

**Source**: `RayNet/losses.py:pose_prediction_loss`, `RayNet/losses.py:geodesic_loss`

Auxiliary loss on the PoseEncoder's 6D rotation prediction. The 6D output is reconstructed to a 3x3 rotation matrix via Gram-Schmidt, then compared to GT using geodesic distance on SO(3).

```
R_pred = gram_schmidt(pred_6d)              # (B, 3, 3)
R_diff = R_pred^T @ R_gt
L_geo  = arccos( (trace(R_diff) - 1) / 2 ) # mean over batch
```

This measures the actual rotation angle needed to go from predicted to GT orientation, respecting the SO(3) manifold geometry. Numerically stabilized with clamp to avoid NaN from arccos at +/-1.

**Source**: `RayNet/losses.py:rotation_6d_to_matrix`, `RayNet/losses.py:geodesic_loss`

---

## 7. Pose Translation Loss (Log-Depth SmoothL1)

**Source**: `RayNet/losses.py:translation_loss`

Auxiliary loss on the PoseEncoder's 3D translation prediction. Uses separate treatment for image-plane offsets (xy) and depth (z):

```
L_trans = SmoothL1(pred_xy, gt_xy) + SmoothL1(log(pred_z), log(gt_z))
```

| Component | Pred activation | GT range | Loss |
|-----------|----------------|----------|------|
| tx, ty | `tanh` -> [-1, 1] | [-1, 1] (normalized) | SmoothL1 |
| tz | `exp` -> (0, +inf) | positive metric depth | SmoothL1 in log-space |

Log-space comparison for depth makes the loss **scale-invariant**: a 10% depth error at 50cm is penalized equally to 10% at 5m. Since `log(exp(raw)) = raw`, the depth loss effectively becomes `SmoothL1(raw, log(gt_z))`.

**GT normalization requirement**: Ground-truth `head_t` from GazeGene (`head_T_vec`) is in centimeters. It must be pre-normalized so tx, ty are in [-1, 1] and tz is positive depth before the loss sees it.

---

## 8. Ray-to-Target Loss

**Source**: `RayNet/losses.py:ray_target_loss`

Geometric constraint that ties gaze direction to a physical target location:

```
target_hat = eyeball_center + gaze_depth * pred_gaze
L_ray = L1(target_hat, gt_gaze_target)
```

Uses ground-truth depth along the gaze ray and known eye center to reconstruct the 3D gaze target from the predicted gaze direction. This provides an explicit geometric signal beyond direction-only loss.

Active in Stage 2 Phase 2+ and Stage 3 Phase 2+.

---

## 9. Gaze Ray Consistency Loss

**Source**: `RayNet/multiview_loss.py:gaze_ray_consistency_loss`

Enforces that predicted gaze vectors from different camera views of the same subject, transformed to world frame via camera extrinsics, point in the same direction.

```
g_world_v = R_cam_v^T @ g_v           # transform to world frame
g_mean = normalize(mean(g_world))      # group consensus
L_consist = L1(g_world, detach(g_mean))
```

Uses `R_cam` (static camera extrinsics) for the world-frame transformation. All operations involve unit vectors -- numerically stable under AMP float16.

See [[Multi-View Consistency]] for details.

---

## 10. Landmark Shape Consistency Loss

**Source**: `RayNet/multiview_loss.py:landmark_shape_consistency_loss`

Enforces that the spatial pattern of predicted landmarks is consistent across camera views. Translation-and-scale-invariant (Procrustes-style):

```
pts_centered = pts - centroid
pts_normalized = pts_centered / rms_distance
L_shape = SmoothL1(pts_normalized_i, pts_normalized_j)
```

See [[Multi-View Consistency]] for details.

---

## 11. Visual-Axis Loss (kappa-corrected, v6)

**Source**: `RayNet/losses.py:visual_axis_loss`

Predicted optical axis is rotated by the per-subject kappa matrix to obtain the predicted visual axis, then L1-compared against the GT visual axis from GazeGene.

```
visual_axis_pred = R_kappa @ optical_axis_pred
visual_axis_pred = normalize(visual_axis_pred)
L_visual = L1(visual_axis_pred, gt_visual_axis)
```

The optical axis is the eyeball-centre → cornea/pupil line (anatomy). The visual axis is the line from the cornea centre through the fovea — the line that defines where the subject is *looking*. The two differ by a per-subject kappa angle (typically ±1-2°). For medical-grade gaze precision, collapsing the two costs ~1° of accuracy, comparable to a sub-arcminute pupillometry budget.

GazeGene ships per-subject `L_kappa` / `R_kappa` Euler angles in `subject_label.pkl`; `RayNet.kappa.build_R_kappa` converts them to a 3×3 rotation matrix at dataset load time. Per-frame batches carry both `R_kappa` and `visual_axis` (per-eye) GT.

**Convention**: `visual_axis = R_kappa @ optical_axis`. This is supervised on `gaze_geom` (not `gaze_fused`) because the optical-axis interpretation only applies to the geometry-derived branch.

Active in Phase 2+ (off in Phase 1 where the gaze branch is frozen).

---

## 12. Iris Mesh Loss (3DGazeNet M-target, v6)

**Source**: `RayNet/losses.py:iris_mesh_loss`

L1 between predicted and GT iris-contour vertices in CCS (centimetres):

```
L_iris_mesh = L1(pred_iris_3d, gt_iris_3d)        # (B, 100, 3)
```

This is the M-target that 3DGazeNet (Eq 1) shows is the dominant generalization lever — their ablation Table 3 reports M+V (Mesh + Vector) outperforming V alone in 4/4 datasets. The mesh acts as a redundancy regulariser: vector regression alone provides exact label supervision but is brittle on profile views, while the mesh constrains predictions to plausible 3D iris geometry.

The GT comes from GazeGene's `iris_mesh_3D` field (shape `(2 eyes × 100 verts × 3)`); the dataset loader indexes the active eye to produce `(100, 3)` per sample. The supervision is geometric (synthetic-iris-mesh GT, **not** iris-pixel texture), so it is consistent with the v6 skeleton-only GazeGene stage.

3DGazeNet's λ_v = 0.1 is the recommended starting weight.

Active in Phase 2+ (gaze branch must be unfrozen for the head to receive gradient).

---

## 13. Iris Edge Loss (3DGazeNet edge-length regulariser, v6)

**Source**: `RayNet/losses.py:iris_edge_loss`

L2 (MSE) between predicted and GT iris ring edge lengths:

```
edges_pred = ||v_i - v_{(i+1) mod 100}||         # 100 edges
edges_gt   = ||gt_v_i - gt_v_{(i+1) mod 100}||
L_iris_edge = MSE(edges_pred, edges_gt)
```

The 100-vertex iris contour is a closed polygon; comparing edge lengths preserves local topology even when absolute vertex positions are off. Without this regulariser, the vertex L1 alone allows the iris ring to fold or self-intersect (a degenerate solution that minimises L1 by collapsing all vertices toward their mean).

3DGazeNet's λ_e = 0.01 is the recommended starting weight.

---

## 14a. Eyeball Radius Loss (v6.2)

**Source**: `RayNet/losses.py:eyeball_radius_loss`

L1 between predicted and GT per-subject eyeball radius (cm):

```
L_eyeball_radius = L1(pred_radius_cm, gt_radius_cm)
```

The new `EyeballRadiusHead` (in `RayNet/raynet_v5.py`) regresses a scalar from the pooled gaze feature with bias init at 1.2 cm (population mean). GazeGene's `subject_label.pkl` ships `eyeball_radius` per subject (1.15-1.25 cm typical); making it a *predicted* scalar lets the OpenEDS torsion stage build the kinematic two-sphere model with the correct globe radius rather than a population-average constant.

Active in Phase 2/3 at weight 0.2 (small because the L1 is in cm scale).

---

## 14. Macro (Head) Gaze Loss (v6.1)

**Source**: `RayNet/losses.py:macro_gaze_loss`

L1 on unit head-gaze vectors. GazeGene's `gaze_C` is the unit direction from the head centre to the gaze target, in CCS — a *macro* signal, head-coordinate, kappa-free, and distinct from per-eye optical/visual axes which carry the *micro* foveal signal.

```
L_gaze_macro = L1(pred_gaze_macro, gt_gaze_c)     (both unit vectors in CCS)
```

The new `MacroGazeHead` (in `RayNet/raynet_v5.py`) fuses two inputs to predict `gaze_macro`:

```
pose_feat (B, d_model)  ← head pose embedding from PoseBranch
eyeball_center_3d (B, 3)  ← regressed CCS centimetres (DETACHED on the way in)
        ↓ concat → Linear → GELU → Linear → unit-normalise
gaze_macro (B, 3)
```

Why detach the eyeball anchor: `eyeball_center_loss` is the only signal we want to shape the eyeball-fc regression. Letting macro gaze gradients flow back through the anchor would mix two objectives in one parameter group.

### Decoupling rationale

- **Macro gaze on synthetic GazeGene**: the head→target vector is robust to photorealistic-but-imperfect MetaHuman renderings — head pose lifts well from the synthetic distribution.
- **Micro gaze (visual axis) on real OpenEDS IR**: refraction-aware foveal direction needs real corneal optics; UE5 specular highlights are exactly the texture-poisoning hazard.

See [[Architecture-v6]] § "Macro vs micro gaze" for the full split.

Active in Phase 2/3 at weight 1.0 (peer with `lam_gaze` on the fused readout).

---

## Total Loss Assembly

**Source**: `RayNet/losses.py:total_loss`

```python
total = lam_lm * landmark_loss + lam_gaze * gaze_loss(gaze_fused)

# Mean-of-two sub-supervisions (v6)
total += lam_gaze_geom   * gaze_loss(gaze_geom)
total += lam_gaze_direct * gaze_loss(gaze_direct)
total += lam_gaze_visual * visual_axis_loss(gaze_geom, R_kappa, gt_visual)

# Macro (head) gaze — GazeGene gaze_C (v6.1)
total += lam_gaze_macro * macro_gaze_loss(gaze_macro, gt_gaze_c)

# 3D eyeball structure (active when lam_eyeball/pupil/geom_angular > 0)
total += lam_eyeball      * eyeball_center_loss
total += lam_pupil        * pupil_center_loss
total += lam_geom_angular * geometric_angular_loss

# 3DGazeNet M-target (v6)
total += lam_iris_mesh * iris_mesh_loss
total += lam_iris_edge * iris_edge_loss

# Ray-to-target (active when lam_ray > 0 and GT available)
total += lam_ray * ray_target_loss

# Pose rotation (active when lam_pose > 0 and pose encoder exists)
total += lam_pose * pose_prediction_loss

# Pose translation (active when lam_trans > 0 and GT available)
total += lam_trans * translation_loss

# Multi-view consistency (added in train loop when multiview enabled)
total += mv_weight * (lam_reproj * gaze_consist + lam_mask * shape_consist)
```

The `mv_weight` ramps linearly from 0 to 1 over the first 10 epochs to smooth multi-view loss activation.

### Logged Components

| Component | Key in CSV | Description |
|-----------|-----------|-------------|
| Total loss | `loss` / `train_total` | Weighted sum of all active losses |
| Landmark | `landmark` / `train_landmark` | Heatmap MSE + coord L1 (normalized by feat area) |
| Angular error | `angular_deg` / `train_angular_deg` | Degrees (metric only, not in loss) |
| Eyeball center | `eyeball` | L1 on 3D eyeball center (cm) |
| Pupil center | `pupil` | L1 on 3D pupil center (cm) |
| Geometric angular | `geom_angular` | Angular error from predicted geometry (rad) |
| **Iris mesh (v6)** | `iris_mesh_loss` | Vertex L1 between predicted/GT 100-vert iris ring (cm) |
| **Iris edge (v6)** | `iris_edge_loss` | Edge-length L2 on the same ring (topology regulariser) |
| **Gaze geom (v6)** | `gaze_geom_loss` | L1 on `gaze_geom` sub-readout |
| **Gaze direct (v6)** | `gaze_direct_loss` | L1 on `gaze_direct` sub-readout |
| **Gaze visual (v6)** | `gaze_visual_loss` | L1 on kappa-corrected visual axis |
| **Gaze macro (v6.1)** | `gaze_macro_loss` | L1 on head→target unit vector (`gaze_C`) |
| Gaze consistency | `gaze_consist` / `train_reproj` | Multi-view gaze ray agreement |
| Shape consistency | `shape` / `train_mask` | Multi-view landmark shape agreement |
| Ray-to-target | `ray_target` / `train_ray_target` | Gaze ray geometric constraint |
| Pose rotation | `pose` / `train_pose` | Geodesic loss in radians |
| Translation | `translation` / `train_translation` | Log-depth SmoothL1 |

---

## v6 Loss Weight Schedule (current)

The v6 schedule replaces the older Stage-2/Stage-3 split. All phases share the same `PHASE_CONFIG` block in `RayNet/train.py` and run on the **TriCam {1, 6, 8}** subset (see [[Multi-View Consistency]]). AERI segmentation losses are **0 across all phases** — the foveal-segmentation supervision moves to a separate OpenEDS-only training stage to prevent MetaHuman texture-poisoning.

| Phase | Epochs | lam_lm | lam_gaze | lam_gaze_geom | lam_gaze_direct | lam_gaze_visual | **lam_gaze_macro** | lam_eyeball | lam_pupil | lam_geom_angular | lam_iris_mesh | lam_iris_edge | lam_pose | lam_trans | lam_reproj | lam_mask | freeze | Multi-view |
|-------|--------|--------|----------|---------------|-----------------|-----------------|--------------------|-------------|-----------|-----------------|---------------|---------------|----------|-----------|------------|----------|--------|------------|
| 1 | 1-15 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 0.5 | 0.0 | 0.0 | gaze_only | Off |
| 2 | 16-30 | 0.0 | 2.0 | 0.5 | 0.5 | 0.5 | **1.0** | 0.3 | 0.3 | 0.5 | 0.1 | 0.01 | 0.0 | 0.0 | 0.0 | 0.0 | face_only | Off |
| 3 | 31-50 | 0.0 | 2.0 | 0.5 | 0.5 | 0.5 | **1.0** | 0.4 | 0.4 | 0.5 | 0.05 | 0.005 | 0.3 | 0.3 | 0.1 | 0.05 | face_kept | On |

### Notes

- **Phase 1**: skeleton geometry only — landmarks + head pose. Gaze branch (and the iris-mesh head it hosts) is frozen. AERI seg is off because foveal supervision is OpenEDS-only.
- **Phase 2**: monocular gaze + 3D anchors + iris-mesh + visual-axis. Face path frozen so the M3 stem doesn't drift while gaze fits. All v6 lambdas at peak weight here.
- **Phase 3**: TriCam multi-view fine-tune. `lam_iris_mesh` halved to let multi-view gaze consistency dominate. Pose + translation come back online for joint fine-tune.

### Foveal-segmentation losses (deferred)

`lam_iris_seg` and `lam_eyeball_seg` are present in `total_loss` for backward-compatibility but **set to 0 across all GazeGene phases**. The foveal-segmentation pipeline trains separately on OpenEDS via `RayNet/openeds/train.py` — see [[Architecture-v6]] for the dataset-decoupling rationale.

---

## Gradient Clipping

Gradient clipping varies by training phase to balance multi-task learning:

| Phase | max_norm | Rationale |
|-------|----------|-----------|
| 1 | 5.0 | Aggressive -- allows large gradients during multi-task warmup |
| 2+ | 2.0 | Conservative -- prevents gaze/pose gradient interference during fine-tuning |
