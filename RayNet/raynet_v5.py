"""
RayNet v5 — Triple-M1 architecture with full-face branches + FPANet
landmark + AERI gaze.

Three task-specific branches, all consuming the 224x224 full-face crop
through a single shared stem:

  Landmark branch (OWNS the shared stem):
    SharedStem (M1 stem+s0+s1) → LandmarkBranchEncoder (M1 s2+s3)
      → FeaturePyramidNetwork (PANet) over [s0, s1, s2, s3]
      → fused P2 (fpn_ch × 56 × 56) → refine + heatmap + offset heads
      → soft-argmax → 14 sub-pixel landmark coords.
    Only task whose gradient backpropagates into the shared stem —
    pose and gaze both detach() their stem input so the low-level
    representation is steered by landmark loss alone. The PANet
    top-down + bottom-up fusion is what gives the head sub-pixel
    resolution; the prior single-stage U-Net decoder plateaued well
    above 1 px and could not see global facial geometry while
    resolving the iris contour.

  Pose branch (gradient-isolated from stem + fuses MAGE bbox):
    s1.detach() → PoseBranchEncoder (M1 s2+s3) → coord-att + pool + proj
    ⊕ BoxEncoder(face_bbox)   (zero-init residual)
      → pose_feat (d_model), 6D rotation, 3D translation.
    BoxEncoder lives inside PoseBranch now — the face bbox is
    head-pose information, not gaze information.

  Gaze branch (gradient-isolated + AERI):
    s1.detach() → GazeBranchEncoder (M1 s2+s3)
      → FPNAERIHead — its own PANet over [s0_det, s1_det, gz_s2, gz_s3]
         producing iris + eyeball mask logits @ 56×56 plus a
         fpn_ch-wide d1 for HRFH harvesting.
      → AERI attention: predicted eyeball mask gates the gaze feature
         map so the pooled vector is dominated by eye-region features
         without any geometric cropping.
      → GazeFusionBlock(gaze_feat, pose_feat, zero-init residual)
      → CrossViewAttention (when n_views > 1)
      → GeometricGazeHead → eyeball_center + pupil_center +
         optical_axis = normalize(pupil - eyeball) (GazeGene Sec 4.2.2).

AERI (Anatomical Eye Region Isolation) replaces the previous eye-crop
encoder: instead of cropping at the pixel level and running a private
RepNeXt on the patch, the gaze branch learns WHERE the eye is via
the iris + eyeball binary masks baked into the MDS shards, and uses
the predicted mask to modulate its own feature map. Benefits:
  - No differentiable cropping / affine grid — removes interpolation
    artefacts that were observed to regress val_angular.
  - Gaze shares the full-face view that landmark/pose see, closing
    the Stage 1 ↔ Stage 2 train/val distribution gap.
  - The segmentation head is a clean MSGazeNet-style auxiliary signal:
    iris + eyeball BCE at 56x56.

Triple-M1 means three branch-specific RepNeXt-M1 s2+s3 encoders plus
one shared stem (which is structurally stem+s0+s1 of a fourth M1).
Weights are loaded identically into all four instances; gradients
diverge during training through task-specific losses and the pose/gaze
detach(). RepNeXt-M1 (embed_dim=(48,96,192,384), depth=(3,3,15,2)) is
deliberately the smaller variant: an A/B comparison against M3 in
docs/experiments/Tripple_M3_run_20260428_130241 vs the equivalent M1
run showed no convergence advantage from the ~1.6× parameter budget,
so dataset size — not capacity — is the constraint that matters.

GazeGene 3D Eyeball Structure losses:
  1. L1 on eyeball_center_3d
  2. L1 on pupil_center_3d
  3. L1 on iris contour (handled by the 10 iris landmarks in the
     14-landmark head — same subsample as in GazeGene subject_label)
  4. Angular error between optical_axis = normalize(pupil - eyeball)
     and GT optical axis.

Parallel training from epoch 1: no sequential freeze phases. Landmark,
pose, gaze, and AERI segmentation are all active simultaneously. The
detach() on the stem is the ONLY isolation mechanism — no .eval() /
requires_grad dance.

Input:  (3, 224, 224) GazeGene face crop + (3,) face bounding box
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from backbone.repnext import create_repnext
from RayNet.coordatt import CoordinateAttention

device = "cuda" if torch.cuda.is_available() else "cpu"

# RepNeXt-M1: embed_dim=(48, 96, 192, 384), depth=(3, 3, 15, 2)
M1_CHANNELS = [48, 96, 192, 384]
# RepNeXt-M3: embed_dim=(64, 128, 256, 512), depth=(3, 3, 13, 2)
# Larger backbone for sub-pixel iris precision + drowsy-eye occlusion
# robustness (GLOBAL/FOVEAL_FLOOR=0.5). v6.2 default.
M3_CHANNELS = [64, 128, 256, 512]
BACKBONE_CHANNELS = {
    'm1': M1_CHANNELS,
    'm3': M3_CHANNELS,
}


# ─── Shared Stem ────────────────────────────────────────────────────

class SharedStem(nn.Module):
    """
    Shared low-level encoder: RepNeXt-M1 stem + stages[0] + stages[1].

    Extracts edges, textures, and low-level facial features shared by
    all three task branches. Output:
        s0: (B, 48, 56, 56)  — skip for U-Net decoders
        s1: (B, 96, 28, 28)  — input to every branch encoder

    Only the landmark branch's backward pass reaches the stem —
    pose and gaze detach s1 (and s0, when used as a decoder skip) —
    so the stem is effectively a landmark-owned low-level encoder.
    """

    def __init__(self, stem, stage0, stage1):
        super().__init__()
        self.stem = stem
        self.stage0 = stage0
        self.stage1 = stage1

    def forward(self, x):
        x = checkpoint(self.stem, x, use_reentrant=False)
        s0 = checkpoint(self.stage0, x, use_reentrant=False)
        s1 = checkpoint(self.stage1, s0, use_reentrant=False)
        return s0, s1


# ─── Branch Encoder ─────────────────────────────────────────────────

class BranchEncoder(nn.Module):
    """
    Task-specific encoder: RepNeXt-M1 stages[2] + stages[3].

    Takes shared stem output (96ch, 28x28) and produces task-specific
    features at two scales:
        s2: 192ch at 14x14
        s3: 384ch at 7x7
    """

    def __init__(self, stage2, stage3):
        super().__init__()
        self.stage2 = stage2
        self.stage3 = stage3

    def forward(self, s1):
        s2 = checkpoint(self.stage2, s1, use_reentrant=False)
        s3 = checkpoint(self.stage3, s2, use_reentrant=False)
        return s2, s3


# ─── MAGE Box Encoder (lives inside PoseBranch) ────────────────────

class BoxEncoder(nn.Module):
    """
    MAGE-style face bounding box encoder (Sec 3.2).

    Encodes (x_p, y_p, L_x) — face centre in normalised camera
    coordinates plus a focal-ratio scale proxy — into a d_model
    embedding. In v5 this is consumed by PoseBranch (head pose and
    face-bbox are both rigid-geometry cues; keeping bbox inside pose
    removes redundant inputs from the gaze fusion block).
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
        )

    def forward(self, bbox):
        return self.mlp(bbox)


# ─── Feature Pyramid (PANet) ───────────────────────────────────────

# FPN_CHANNELS sets the uniform pyramid width. 128 is the working
# default — enough capacity for sub-pixel landmark regression while
# keeping P2 activations (fpn_ch * 56 * 56) at ~1 GB per branch under
# bf16 + the a100_40gb batch (1152 samples). The FPANet whitepaper /
# user spec is 256ch; bumping to 256 quadruples activation memory and
# pushes the 40 GB profile out of headroom — switch only if running on
# 80 GB or larger, by passing `fpn_ch=256` through the factory.
FPN_CHANNELS = 128


class FeaturePyramidNetwork(nn.Module):
    """
    Path Aggregation Network (PANet) over four backbone strides.

    Inputs are P2..P5 (strides 4, 8, 16, 32 for a 224×224 face crop):
      P2  s0 / shared stem stage[0]      48ch @ 56×56
      P3  s1 / shared stem stage[1]      96ch @ 28×28
      P4  s2 / branch encoder stage[2]  192ch @ 14×14
      P5  s3 / branch encoder stage[3]  384ch @  7× 7

    Two passes — the multi-scale fusion that 1-stage U-Net decoders
    lack:

      1. Top-down (FPN, semantic injection):
           T_5 = smooth(lat(P_5))
           T_i = smooth(lat(P_i) + Upsample(T_{i+1}))   for i = 4, 3, 2

      2. Bottom-up (PAN, high-resolution amplification):
           B_2 = T_2
           B_i = smooth(T_i + Downsample(B_{i-1}))      for i = 3, 4, 5

    All convs are Conv-BN-SiLU. The lateral 1×1 reprojects each input
    to `out_ch`; the smoothing and downsampling 3×3s build the pyramid.
    Returns [B_2, B_3, B_4, B_5] at strides 4, 8, 16, 32 — heads pick
    whichever level they need (landmark + AERI both use B_2).
    """

    def __init__(self, in_channels, out_ch=FPN_CHANNELS):
        super().__init__()
        self.out_ch = out_ch
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
            )
            for c in in_channels
        ])
        # Top-down smoothing — applied after lateral + upsample sum.
        # The deepest level (T_5) also passes through one 3×3 for symmetry.
        self.smooth_td = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
            )
            for _ in in_channels
        ])
        # Stride-2 3×3 downsampling for the bottom-up path. One per
        # level transition (P2→P3, P3→P4, P4→P5).
        self.downsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
            )
            for _ in range(len(in_channels) - 1)
        ])
        # Bottom-up smoothing on each fused level above P2.
        self.smooth_bu = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
            )
            for _ in range(len(in_channels) - 1)
        ])

    def forward(self, feats):
        # feats: [P2, P3, P4, P5] — low-stride to high-stride.
        L = [self.lateral[i](f) for i, f in enumerate(feats)]
        n = len(L)

        # Top-down: P5 → P4 → P3 → P2.
        T = [None] * n
        T[n - 1] = self.smooth_td[n - 1](L[n - 1])
        for i in range(n - 2, -1, -1):
            up = F.interpolate(T[i + 1], size=L[i].shape[2:],
                               mode='bilinear', align_corners=False)
            T[i] = self.smooth_td[i](L[i] + up)

        # Bottom-up: P2 → P3 → P4 → P5.
        B = [None] * n
        B[0] = T[0]
        for i in range(1, n):
            down = self.downsample[i - 1](B[i - 1])
            B[i] = self.smooth_bu[i - 1](T[i] + down)
        return B  # [P2, P3, P4, P5] all at out_ch.


# ─── FPN-based Landmark Branch ─────────────────────────────────────

class FPNLandmarkBranch(nn.Module):
    """
    Landmark detection driven by PANet multi-scale fusion.

    Pipeline:
      BranchEncoder(s1) → (s2, s3)
      FPN([s0, s1, s2, s3]) → [P2, P3, P4, P5]   (all `fpn_ch`)
      P2 → CoordinateAttention (row/column spatial gating)
      → heatmap (n_landmarks × H × W)
      → offset  (n_landmarks·2 × H × W)
      → soft-argmax (β=100) + offset refinement → 14 sub-pixel coords.

    The fused P2 is what the old U-Net's d1 used to be, but augmented
    with the global semantic context that P5 injects through the
    top-down path and the high-resolution emphasis that the bottom-up
    path adds. CoordinateAttention on top of P2 is what the
    sub-pixel-accurate ablation/subpixel_landmark run used; it
    encodes horizontal/vertical positional cues directly into the
    feature map and was the head architecture that reached 0.53 px
    val_landmark_px in run_20260405_025128.

    Heatmap stride is 4: a 448×448 input produces a 112×112 heatmap
    (one cell per native pixel) — required to recover the previous
    sub-pixel result. A 224×224 input produces 56×56 (each cell spans
    4 image pixels) and so caps achievable image-space precision.

    14 landmarks = 10 iris contour + 4 pupil boundary, GazeGene Sec 4.1.
    """

    def __init__(self, n_landmarks=14, fpn_ch=FPN_CHANNELS,
                 backbone_channels=None):
        super().__init__()
        self.encoder = None  # set by factory
        self.n_landmarks = n_landmarks
        in_ch = list(backbone_channels) if backbone_channels else M1_CHANNELS
        self.fpn = FeaturePyramidNetwork(in_ch, out_ch=fpn_ch)

        self.coord_att = CoordinateAttention(fpn_ch)
        # Per-head Conv-BN-ReLU refinement matches the
        # ablation/4th_april IrisPupilLandmarkHead (heads.py:24-33)
        # that hit 0.53 px val_landmark_px. Each head gets its own BN
        # so heatmap and offset normalisation statistics don't share.
        head_mid = 128
        self.heatmap = nn.Sequential(
            nn.Conv2d(fpn_ch, head_mid, 3, padding=1),
            nn.BatchNorm2d(head_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_mid, n_landmarks, 1),
        )
        self.offset = nn.Sequential(
            nn.Conv2d(fpn_ch, head_mid, 3, padding=1),
            nn.BatchNorm2d(head_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_mid, n_landmarks * 2, 1),
        )

    def forward(self, s0, s1, s2, s3):
        feats = self.fpn([s0, s1, s2, s3])
        x = self.coord_att(feats[0])  # P2 fused, with row/column attention
        hm = self.heatmap(x)
        off = self.offset(x)
        coords = self._soft_argmax(hm, off)
        return coords, hm

    def _soft_argmax(self, hm, off):
        # β = 100 collapses softmax(logits) into a near-hard argmax at
        # the peak cell. Coord L1 then flows almost entirely into the
        # offset head, which learns the (dx, dy) sub-pixel residual at
        # that cell — the same scheme that produced 0.53 px in
        # run_20260405_025128. β = 1 (Integral Pose Regression) was
        # tried in v5/v6 but spreads the expectation across the heatmap
        # tail and stalled landmark precision well above 1 px.
        B, N, H, W = hm.shape
        flat = hm.view(B, N, -1)
        weight = F.softmax(flat * 100.0, dim=-1).view(B, N, H, W)

        gx = torch.arange(W, dtype=torch.float32, device=hm.device)
        gy = torch.arange(H, dtype=torch.float32, device=hm.device)

        x = (weight * gx[None, None, None, :]).sum(dim=[2, 3])
        y = (weight * gy[None, None, :, None]).sum(dim=[2, 3])

        off2 = off.view(B, N, 2, H * W)
        idx = (y.long() * W + x.long()).clamp(0, H * W - 1)
        dx = off2[:, :, 0, :].gather(2, idx.unsqueeze(2)).squeeze(2)
        dy = off2[:, :, 1, :].gather(2, idx.unsqueeze(2)).squeeze(2)

        return torch.stack([x + dx, y + dy], dim=-1)


# ─── FPN-based AERI Segmentation Head ──────────────────────────────

class FPNAERIHead(nn.Module):
    """
    Anatomical Eye Region Isolation head built on PANet fusion.

    Replaces the previous mini-U-Net AERI decoder. Operates on the
    gaze branch's own four-level pyramid (s0, s1, gaze_s2, gaze_s3) —
    the gaze branch detaches s0/s1 upstream so AERI loss never leaks
    into the landmark-owned shared stem.

    Pipeline:
      FPN([s0, s1, gaze_s2, gaze_s3]) → [P2, P3, P4, P5]
      P2 (fpn_ch × 56 × 56) → Conv-BN-SiLU refine → d1
      d1 → 1×1 Conv → 2 logit channels (iris, eyeball) @ 56×56

    Returns:
      iris_logits     (B, 56, 56)
      eyeball_logits  (B, 56, 56)
      d1              (B, fpn_ch, 56, 56) — the fused, refined P2; what
        HRFH-α harvests for the foveal pathway. Wider than the legacy
        48-channel U-Net d1, with multi-scale context injected by the
        top-down + bottom-up pyramid passes.
    """

    def __init__(self, fpn_ch=FPN_CHANNELS, backbone_channels=None):
        super().__init__()
        in_ch = list(backbone_channels) if backbone_channels else M1_CHANNELS
        self.fpn = FeaturePyramidNetwork(in_ch, out_ch=fpn_ch)
        self.refine = nn.Sequential(
            nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_ch),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Conv2d(fpn_ch, 2, 1)

    def forward(self, s0, s1, s2, s3):
        feats = self.fpn([s0, s1, s2, s3])
        d1 = self.refine(feats[0])
        logits = self.head(d1)
        iris = logits[:, 0]
        eyeball = logits[:, 1]
        return iris, eyeball, d1


# ─── Gaze Fusion + Geometric Head ───────────────────────────────────

class GazeFusionBlock(nn.Module):
    """
    Fuse gaze-branch features with the pose-branch embedding.

    gaze_feat is the anchor; pose_feat enters through a zero-init
    residual projection so the gaze head predicts from the gaze
    pathway alone at the start of training and learns to blend in
    pose/bbox signal gradually.
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)

    def forward(self, gaze_feat, pose_feat):
        residual = self.proj(torch.cat([gaze_feat, pose_feat], dim=-1))
        return gaze_feat + residual


class GeometricGazeHead(nn.Module):
    """
    GazeGene 3D Eyeball Structure Estimation head (Sec 4.2.2) +
    3DGazeNet-style mean-of-two gaze fusion.

    Three gaze readouts:
      - gaze_geom   = normalize(pupil_center - eyeball_center)
        Derived purely from the predicted 3D anchors. Robust under
        profile head pose because it inherits the geometric prior from
        L_eyeball + L_pupil.
      - gaze_direct: regressed by a sibling Linear head, then unit-
        normalised. Robust on near-frontal / high-resolution faces
        where the trunk has direct iris signal.
      - gaze_fused  = normalize(0.5 * (gaze_geom + gaze_direct))
        Mean-of-two as in 3DGazeNet (Sec 7, supplementary Fig 7d).
        Used as the canonical training/inference signal once both
        sub-heads have warmed up.

    Predictions returned:
      - eyeball_center_3d (B, 3)
      - pupil_center_3d   (B, 3)
      - gaze_geom         (B, 3) unit
      - gaze_direct       (B, 3) unit
      - gaze_fused        (B, 3) unit
      - gaze_angles       (B, 2) pitch/yaw of gaze_fused
    """

    def __init__(self, d_model=256, hidden_dim=128):
        super().__init__()
        self.eyeball_fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.pupil_fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        # Direct gaze regressor — operates on the same pooled feature
        # that feeds eyeball/pupil. Output is a free 3-vector that gets
        # unit-normalised post-hoc; the loss is on the normalised vector
        # (consistent with `gaze_loss` in losses.py).
        self.direct_fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, pooled):
        eyeball_center = self.eyeball_fc(pooled)
        pupil_center = self.pupil_fc(pooled)

        gaze_geom = F.normalize(pupil_center - eyeball_center, dim=-1)
        gaze_direct = F.normalize(self.direct_fc(pooled), dim=-1)
        gaze_fused = F.normalize(gaze_geom + gaze_direct, dim=-1)

        # gaze_angles uses the fused gaze (post-warmup canonical signal)
        x, y, z = gaze_fused[:, 0], gaze_fused[:, 1], gaze_fused[:, 2]
        pitch = torch.asin((-y).clamp(-1 + 1e-6, 1 - 1e-6))
        yaw = torch.atan2(-x, -z)
        gaze_angles = torch.stack([pitch, yaw], dim=-1)
        return (eyeball_center, pupil_center,
                gaze_geom, gaze_direct, gaze_fused, gaze_angles)


class IrisMeshHead(nn.Module):
    """
    3DGazeNet-style M-target: regress N_v iris-contour vertices in CCS
    (centimetres). GazeGene supplies `iris_mesh_3D` of shape
    (N_v=100, 3) per eye, stored as (2 eyes, 100, 3) in `complex_label`.

    The head predicts (B, N_v, 3) directly. Pairing it with the vertex
    L1 + edge-length L2 losses (see losses.py:iris_mesh_loss /
    iris_edge_loss) reproduces 3DGazeNet's M+V training target — the
    single largest generalization win in their ablation (Tab 3, M+V vs
    V alone). The supervision is geometric (synthetic-iris-mesh GT,
    NOT iris-pixel texture), so it is consistent with our skeleton-
    only GazeGene stage.
    """

    N_VERTICES = 100

    def __init__(self, d_model=256, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.N_VERTICES * 3),
        )

    def forward(self, pooled):
        verts_flat = self.fc(pooled)
        return verts_flat.view(-1, self.N_VERTICES, 3)


class EyeballRadiusHead(nn.Module):
    """
    Predicts the per-subject eyeball radius (R_s, cm) from the pooled
    gaze feature.

    GazeGene `subject_label.pkl` ships per-subject anatomy:
    ``eyeball_radius`` (typical 1.15 - 1.25 cm), ``cornea_radius``,
    ``cornea2center``. We currently treat ``eyeball_radius`` as a
    fixed prior in the iris/eyeball mask renderer; v6.2 makes it a
    *predicted* scalar so the Macro-Locator can refine it per-subject
    at inference time, which the OpenEDS torsion stage consumes when
    constructing the kinematic two-sphere eye model.

    Output is unconstrained; supervise with ``eyeball_radius_loss``
    (L1 against GazeGene's ground-truth scalar). The head ships a
    sensible bias init (``DEFAULT_EYEBALL_RADIUS_CM = 1.2``) so the
    initial prediction is anatomically plausible.
    """

    DEFAULT_EYEBALL_RADIUS_CM = 1.2

    def __init__(self, d_model: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.constant_(self.fc[-1].bias, self.DEFAULT_EYEBALL_RADIUS_CM)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # Squeeze the trailing dim so callers get (B,) not (B, 1).
        return self.fc(pooled).squeeze(-1)


class MacroGazeHead(nn.Module):
    """
    Macro (head) gaze head — GazeGene `gaze_C` paradigm.

    GazeGene ships two gaze paradigms (per the CVPR2025 dataset doc):
      1. **`gaze_C` — head gaze**: a unit vector from the head centre
         to the gaze target, in CCS. Lives in *head* coordinates,
         insensitive to per-eye optical-axis kappa offsets.
      2. **`optic_axis_{L,R}` — optical axis**: per-eye line through
         (eyeball_centre → cornea_centre / pupil), CCS.

    v6 supervises both, with the v6.1 split assigning **macro gaze
    (`gaze_C`) to GazeGene** (synthetic, photorealistic, large head-
    pose distribution) and **micro gaze (visual axis from real iris
    refraction) to OpenEDS** (real IR, sub-pixel iris boundaries).
    The macro head fuses pose with the predicted 3D eyeball anchor:

        pose_feat (B, d_model)        ← head pose embedding
        eyeball_center_3d (B, 3)      ← regressed in CCS, cm
                ↓ concat
        Linear(d_model + 3 → hidden)
                ↓ GELU
        Linear(hidden → 3) → unit-normalise → gaze_macro

    The 3D eyeball anchor enters the head explicitly so the macro
    direction inherits the geometric prior from `eyeball_center_loss`,
    not just the appearance-derived pose embedding.
    """

    def __init__(self, d_model: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, pose_feat: torch.Tensor,
                eyeball_center_3d: torch.Tensor) -> torch.Tensor:
        # Detach the eyeball anchor on the way in so the macro-gaze loss
        # cannot back-propagate into the GeometricGazeHead's eyeball_fc
        # via this side path. The eyeball_center is independently
        # supervised by `eyeball_center_loss`; mixing gradients here
        # would let macro gaze override the metric anchor regression.
        x = torch.cat([pose_feat, eyeball_center_3d.detach()], dim=-1)
        return F.normalize(self.fc(x), dim=-1)


# ─── Gaze Branch ────────────────────────────────────────────────────

class GazeBranch(nn.Module):
    """
    AERI + HRFH-α gaze branch.

    Pipeline (operating on gradient-isolated shared-stem features):
      s0.detach(), s1.detach() from SharedStem
      s1.detach() → BranchEncoder → gaze_s2 (14x14), gaze_s3 (7x7)
      FPNAERIHead(s0, s1, gaze_s2, gaze_s3)
        → iris_logits, eyeball_logits, d1 (B, fpn_ch, 56, 56)
      saliency = SALIENCY_IRIS_W * sigmoid(iris)
               + SALIENCY_EYE_W  * sigmoid(eye)
      scheduled_mask = α * saliency + (1 - α) * 1
      Global path: gaze_s3 * (GLOBAL_FLOOR + (1-GLOBAL_FLOOR) * pool₇(scheduled_mask))
                   → CoordAtt → pool → 384-d → LayerNorm
      Foveal path: pool(d1 * (FOVEAL_FLOOR + (1-FOVEAL_FLOOR) * scheduled_mask))
                   → fpn_ch-d → Linear→96 → GELU → LayerNorm
                   → stochastic depth (training only, FOVEAL_DROP_P)
                   — identity at val.
      [global ‖ foveal_proj] (480-d) → Linear → 256-d gaze_feat
      GazeFusionBlock(gaze_feat, pose_feat)  (zero-init residual)
      CrossViewAttention (when n_views > 1)
      GeometricGazeHead → eyeball_center, pupil_center, optical_axis
    """

    # Saliency mix: iris is ~8x8 cells of a 56x56 map, eyeball is ~25x15.
    # 0.65/0.35 keeps the iris dominant without crowding out the limbus
    # context that the geometric head uses to localise eyeball_center.
    SALIENCY_IRIS_W = 0.65
    SALIENCY_EYE_W = 0.35

    # Foveal stochastic-depth drop probability (training only).
    # Replaces the previous train-only mask_dropout, which created a
    # train/val asymmetry on the GLOBAL path. Stochastic depth on the
    # foveal contribution still regularises the high-resolution channel
    # without ever flipping the global mask gate at val.
    FOVEAL_DROP_P = 0.10

    # ── Eyelid-occlusion robustness floors ───────────────────────────
    # When the eyelid partially covers the sclera, AERI's iris/eyeball
    # masks shrink, the saliency map collapses, and the gating signal
    # below cannot recover the missing eye signal at inference time.
    # The "floor" terms cap how much the saliency mask is allowed to
    # SUPPRESS each path — i.e. when AERI predicts ~0 saliency the gate
    # still passes (FLOOR) of the underlying features through, and only
    # the residual (1-FLOOR) is gated by saliency.  This breaks the
    # shortcut where the model becomes catastrophically dependent on a
    # crisp iris/sclera boundary, while still letting AERI amplify the
    # eye region by up to 1×.  Increase a floor → less AERI dependence.
    GLOBAL_FLOOR = 0.50   # was effectively 0.25 (gate = 0.25 + 0.75·M)
    FOVEAL_FLOOR = 0.50   # was effectively 0.00 (gate = M; pure suppression)
    # ── Eyelid-occlusion robustness floors ───────────────────────────

    def __init__(self, d_model=256, fpn_ch=FPN_CHANNELS,
                 backbone_channels=None):
        super().__init__()
        self.encoder = None  # set by factory (BranchEncoder)
        bb = list(backbone_channels) if backbone_channels else M1_CHANNELS
        self.aeri = FPNAERIHead(fpn_ch=fpn_ch, backbone_channels=bb)

        # Backbone s3 channel count (M1=384, M3=512). Drives the
        # coordinate-attention input width below.
        s3_ch = bb[-1]
        self.coord_att = CoordinateAttention(s3_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # --- AERI / HRFH-α fusion ---
        # The fpn_ch-d raw foveal vector (FPNAERIHead d1) is projected
        # to 96-d before being concatenated with the 384-d global
        # vector. The 96/(384+96) = 20% foveal share preserves the
        # sub-pixel iris signal HRFH was designed to extract while
        # keeping the global context dominant. Each path is normalised
        # separately so the global stats don't drown the foveal stats
        # inside a shared LN. The legacy mini-U-Net used a 48-channel
        # d1; the FPN d1 is wider (`fpn_ch`, default 128) and carries
        # multi-scale context from the top-down + bottom-up passes.
        self.foveal_proj = nn.Sequential(
            nn.Linear(fpn_ch, 96),
            nn.GELU(),
        )
        self.global_norm = nn.LayerNorm(s3_ch)
        self.foveal_norm = nn.LayerNorm(96)
        # global (s3_ch) + foveal_proj (96) → d_model
        self.proj = nn.Linear(s3_ch + 96, d_model)
        # --- AERI / HRFH-α fusion ---

        self.fusion_block = GazeFusionBlock(d_model)
        self.head = GeometricGazeHead(d_model)
        # 3DGazeNet M-target — iris contour mesh in CCS (100 verts × 3)
        self.iris_mesh_head = IrisMeshHead(d_model)
        # GazeGene gaze_C — macro (head) gaze from pose + 3D eyeball anchor
        self.macro_gaze_head = MacroGazeHead(d_model)
        # Per-subject eyeball radius (R_s, cm) — anchors the OpenEDS
        # torsion two-sphere model.
        self.eyeball_radius_head = EyeballRadiusHead(d_model)

    def forward(self, s0_detached, s1_detached, pose_feat,
                    cross_view_attn=None, n_views=1, cam_embed=None,
                    aeri_alpha=0.4):
        gaze_s2, gaze_s3 = self.encoder(s1_detached)

        # 1. AERI segmentation + d1 harvest
        iris_logits, eyeball_logits, d1 = self.aeri(
            s0_detached, s1_detached, gaze_s2, gaze_s3)

        iris_mask = torch.sigmoid(iris_logits).unsqueeze(1)
        eye_mask = torch.sigmoid(eyeball_logits).unsqueeze(1)

        # 2. Saliency: iris-dominant but with enough limbus context for
        #    the geometric head to localise eyeball_center.
        saliency_56 = (self.SALIENCY_IRIS_W * iris_mask
                       + self.SALIENCY_EYE_W * eye_mask)

        # 3. α schedule: blend saliency with a uniform field. Caller
        #    decides α via the curriculum (see train.get_scheduled_alpha).
        uniform_mask = torch.ones_like(saliency_56)
        scheduled_mask = (aeri_alpha * saliency_56
                          + (1.0 - aeri_alpha) * uniform_mask)

        # 4. Global path — gate the stride-32 bottleneck. The GLOBAL_FLOOR
        #    baseline keeps non-eye context contributing FLOOR of the
        #    signal even when AERI collapses to an empty mask (e.g.
        #    eyelid covering sclera at inference). NB: no stochastic
        #    dropout here — keeping the global path identical between
        #    train and val fixes the asymmetry that drove P3 val drift.
        #
        # v6.2 — pool to ``gaze_s3``'s actual spatial dim instead of a
        # hardcoded 7. At 224 input that's 7×7 (M1/M3 stride-32) and
        # behaves identically to before; at 448 input it's 14×14, which
        # is the bug that previously silently broke any non-224 forward.
        s3_h, s3_w = gaze_s3.shape[-2], gaze_s3.shape[-1]
        eye_attn = F.adaptive_avg_pool2d(scheduled_mask, (s3_h, s3_w))
        global_gate = self.GLOBAL_FLOOR + (1.0 - self.GLOBAL_FLOOR) * eye_attn
        gaze_s3_gated = gaze_s3 * global_gate
        global_feat = self.pool(self.coord_att(gaze_s3_gated)).flatten(1)
        global_feat = self.global_norm(global_feat)

        # 5. Foveal path — gate d1 at 56x56 then pool to 48-d.
        #    FOVEAL_FLOOR matches GLOBAL_FLOOR so the high-resolution
        #    iris/pupil signal degrades gracefully (not catastrophically)
        #    when AERI mis-fires under eyelid occlusion. The previous
        #    pure multiply (gate=M) collapsed foveal_feat to ~30% of d1's
        #    natural magnitude whenever saliency≈0, which is what made
        #    the inference-time drowsy-eye pattern brittle.
        foveal_gate = self.FOVEAL_FLOOR + (1.0 - self.FOVEAL_FLOOR) * scheduled_mask
        foveal_feat = self.pool(d1 * foveal_gate).flatten(1)
        foveal_feat = self.foveal_proj(foveal_feat)
        foveal_feat = self.foveal_norm(foveal_feat)

        # 6. Stochastic depth on the foveal contribution only.
        #    Drop the foveal vector with prob FOVEAL_DROP_P during
        #    training; identity at val. Forces the global pathway to
        #    stay competent without making the global mask gate change
        #    behaviour between train and val.
        if self.training and self.FOVEAL_DROP_P > 0:
            keep = (torch.rand(foveal_feat.shape[0], 1,
                               device=foveal_feat.device)
                    >= self.FOVEAL_DROP_P).float()
            foveal_feat = foveal_feat * keep / (1.0 - self.FOVEAL_DROP_P)

        # 7. Concat and project to d_model.
        gaze_feat = self.proj(torch.cat([global_feat, foveal_feat], dim=1))

        # pooled_sv is the pre-CrossViewAttention (single-view) representation.
        # Returned separately so the training loop can supervise the single-view
        # pathway directly, preventing val degradation from CrossViewAttention
        # train/val asymmetry (train uses 9-view fusion; val bypasses it).
        pooled_sv = self.fusion_block(gaze_feat, pose_feat)

        pooled = pooled_sv
        if cross_view_attn is not None:
            pooled = cross_view_attn(pooled_sv, n_views, cam_embed)

        (eyeball_center, pupil_center,
         gaze_geom, gaze_direct, gaze_fused, gaze_angles) = self.head(pooled)

        # 3DGazeNet M-target — iris-contour mesh in CCS (B, 100, 3).
        iris_mesh_3d = self.iris_mesh_head(pooled)

        # Macro gaze (GazeGene gaze_C) — fused from head pose + 3D
        # eyeball anchor. Independent of `gaze_geom` / `gaze_direct`
        # so the optical-axis branch is not constrained by macro
        # supervision (and vice versa).
        gaze_macro = self.macro_gaze_head(pose_feat, eyeball_center)

        # Per-subject eyeball radius (R_s, cm) — anchors torsion stage.
        eyeball_radius = self.eyeball_radius_head(pooled)

        # Single-view gaze prediction (no CrossViewAttention).
        # Only computed when multi-view fusion is actually active (n_views > 1)
        # so there is no overhead in single-view inference / validation.
        sv_gaze_fused = None
        if n_views > 1:
            _, _, _, _, sv_gaze_fused, _ = self.head(pooled_sv)

        return (eyeball_center, pupil_center,
                gaze_geom, gaze_direct, gaze_fused, gaze_angles,
                iris_mesh_3d, gaze_macro, eyeball_radius,
                iris_logits, eyeball_logits, sv_gaze_fused)


# ─── Pose Branch (now fuses BoxEncoder) ────────────────────────────

class PoseBranch(nn.Module):
    """
    Head pose estimation branch with MAGE bbox fusion.

    Takes gradient-detached shared-stem features to prevent pose
    gradients from contaminating the landmark-owned stem. Encodes the
    face bbox with BoxEncoder and fuses it into the pose embedding —
    both are rigid-geometry cues that share the same downstream head.

    Outputs:
      pose_feat (B, d_model): embedding consumed by GazeBranch
      pred_pose_6d (B, 6): 6D rotation → Gram-Schmidt → SO(3)

    v6.2: head translation is no longer regressed here. The Macro-
    Locator (predicted ``eyeball_center_3d`` in CCS) carries global
    translation, so the dedicated translation head was redundant and
    its absence reduces PoseBranch by one Linear output dim.
    """

    def __init__(self, d_model=256, backbone_channels=None):
        super().__init__()
        self.encoder = None  # set by factory (BranchEncoder)
        bb = list(backbone_channels) if backbone_channels else M1_CHANNELS
        s3_ch = bb[-1]
        self.coord_att = CoordinateAttention(s3_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(s3_ch, d_model)

        self.box_encoder = BoxEncoder(d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Zero-init the residual so pose_feat = cnn_feat at init
        # (bbox signal ramps in as the box_encoder + fuse train).
        nn.init.zeros_(self.fuse[2].weight)
        nn.init.zeros_(self.fuse[2].bias)

        self.head = nn.Linear(d_model, 6)  # 6D rotation only (v6.2)

    def forward(self, s1_detached, face_bbox=None):
        s2, s3 = self.encoder(s1_detached)
        s3_att = self.coord_att(s3)
        cnn_feat = self.proj(self.pool(s3_att).flatten(1))   # (B, d_model)

        if face_bbox is not None:
            box_feat = self.box_encoder(face_bbox)
        else:
            box_feat = torch.zeros_like(cnn_feat)
        residual = self.fuse(torch.cat([cnn_feat, box_feat], dim=-1))
        pose_feat = cnn_feat + residual

        pred_pose_6d = self.head(pose_feat)
        return pose_feat, pred_pose_6d


# ─── Cross-View and Camera Modules ─────────────────────────────────

class CrossViewAttention(nn.Module):
    """
    Pre-norm Transformer Encoder for cross-view gaze feature fusion.
    Single-view (n_views=1) bypasses (identity).
    """

    def __init__(self, d_model=256, n_heads=4, d_ff=512, dropout=0.1,
                 n_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

    def forward(self, x, n_views, cam_embed=None):
        if n_views <= 1:
            return x
        if cam_embed is not None:
            x = x + cam_embed
        B, D = x.shape
        G = B // n_views
        x = x.view(G, n_views, D)
        x = self.encoder(x)
        return x.view(G * n_views, D)


class CameraEmbedding(nn.Module):
    """Encode R_cam (3x3) + T_cam (3) → d_model embedding."""

    # GazeGene T_vec is in centimetres (CVPR2025 dataset README) with rig
    # positions of magnitude ~30-600 cm. R_cam entries are unit-norm in
    # [-1, 1]. Concatenating raw values produces a 12-D vector where 3
    # dims dominate by ~2-3 orders of magnitude, which (a) saturates the
    # initial Linear(12, 64) post-ReLU on T_cam alone and (b) starves
    # cross-view attention of rotation cues. Dividing T_cam by 100
    # (cm → m) puts both modalities on a comparable [-6, 6] scale without
    # changing any geometric semantics elsewhere — T_cam in cm is still
    # what dataset.py, multiview_loss.py, and any future cross-view
    # reprojection code see.
    T_NORM_CM_PER_M = 100.0

    def __init__(self, d_model=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )

    def forward(self, R_cam, T_cam):
        T_norm = T_cam / self.T_NORM_CM_PER_M
        x = torch.cat([R_cam.flatten(1), T_norm], dim=-1)
        return self.mlp(x)


# ─── RayNet V5 Main Model ──────────────────────────────────────────

class RayNetV5(nn.Module):
    """
    RayNet v5 Triple-M1: full-face branches + AERI gaze + parallel MTL.
    """

    def __init__(self, shared_stem, landmark_branch, gaze_branch,
                 pose_branch, cross_view_cfg=None):
        super().__init__()
        self.shared_stem = shared_stem
        self.landmark_branch = landmark_branch
        self.gaze_branch = gaze_branch
        self.pose_branch = pose_branch

        cv_cfg = cross_view_cfg or {}
        cv_cfg.setdefault('d_model', 256)
        self.cross_view_attn = CrossViewAttention(**cv_cfg)
        self.camera_embedding = CameraEmbedding(d_model=cv_cfg['d_model'])

    def forward(self, x, n_views=1, R_cam=None, T_cam=None,
                face_bbox=None,aeri_alpha=0.2, **_unused):
        """
        Args:
            x: (B, 3, 224, 224) face crop.
            n_views: views per group (1=single, 9=multi-view).
            R_cam, T_cam: camera extrinsics (for CrossViewAttention
                cam_embed — optional).
            face_bbox: (B, 3) MAGE face bbox [x_p, y_p, L_x], consumed
                by PoseBranch.
            **_unused: swallows legacy use_landmark_bridge /
                use_pose_bridge flags that Stage 3 configs still pass.

        Returns:
            dict of predictions (see bottom).
        """
        s0, s1 = self.shared_stem(x)

        # Landmark branch — only gradient path into the shared stem.
        lm_s2, lm_s3 = self.landmark_branch.encoder(s1)
        landmark_coords, landmark_heatmaps = self.landmark_branch(
            s0, s1, lm_s2, lm_s3)

        # Pose branch — gradient-isolated + bbox fusion. v6.2 removed
        # the translation head; head pose carries rotation only.
        pose_feat, pred_pose_6d = self.pose_branch(
            s1.detach(), face_bbox=face_bbox)

        # Gaze branch — gradient-isolated + AERI attention.
        # pose_feat is detached before cam_embed so gaze loss cannot
        # backpropagate into pose_branch through this side path.
        cam_embed = None
        if R_cam is not None and T_cam is not None:
            cam_embed = self.camera_embedding(R_cam, T_cam) + pose_feat.detach()

        (eyeball_center, pupil_center,
         gaze_geom, gaze_direct, gaze_fused, gaze_angles,
         iris_mesh_3d, gaze_macro, eyeball_radius,
         iris_mask_logits, eyeball_mask_logits,
         sv_gaze_fused) = self.gaze_branch(
            s0.detach(), s1.detach(), pose_feat,
            cross_view_attn=self.cross_view_attn,
            n_views=n_views,
            cam_embed=cam_embed,
            aeri_alpha=aeri_alpha,
        )

        return {
            # Landmarks
            'landmark_coords': landmark_coords,        # (B, 14, 2) @ 56x56
            'landmark_heatmaps': landmark_heatmaps,    # (B, 14, 56, 56)
            # AERI segmentation
            'iris_mask_logits': iris_mask_logits,      # (B, 56, 56)
            'eyeball_mask_logits': eyeball_mask_logits,
            # Gaze (GazeGene 3D structure)
            'eyeball_center': eyeball_center,          # (B, 3) cm, CCS
            'pupil_center': pupil_center,              # (B, 3) cm, CCS
            # Three gaze readouts (3DGazeNet mean-of-two fusion).
            # `gaze_vector` aliases the fused signal so existing call
            # sites (multiview_loss, val metrics) consume the canonical
            # output without code changes.
            'gaze_geom': gaze_geom,                    # (B, 3) unit, from anchors
            'gaze_direct': gaze_direct,                # (B, 3) unit, regressed
            'gaze_fused': gaze_fused,                  # (B, 3) unit, mean-of-two
            'gaze_vector': gaze_fused,                 # canonical alias
            'gaze_angles': gaze_angles,                # (B, 2) pitch/yaw
            # 3DGazeNet M-target — iris-contour mesh in CCS, cm
            'iris_mesh_3d': iris_mesh_3d,              # (B, 100, 3)
            # GazeGene macro (head) gaze — fused from pose + eyeball anchor
            'gaze_macro': gaze_macro,                  # (B, 3) unit, CCS
            # Per-subject eyeball radius (R_s, cm) — feeds OpenEDS torsion
            'eyeball_radius': eyeball_radius,          # (B,) cm
            # Single-view fused gaze (pre-CrossViewAttention); None when n_views==1
            'gaze_vector_sv': sv_gaze_fused,           # (B, 3) unit or None
            # Pose (v6.2 — rotation only, translation handled by Macro-Locator)
            'pred_pose_6d': pred_pose_6d,              # (B, 6)
        }


# ─── Factory Functions ──────────────────────────────────────────────

def _split_m1_backbone(m1):
    return m1.stem, m1.stages[0], m1.stages[1], m1.stages[2], m1.stages[3]


def create_raynet_v5(backbone_weight_path=None, cross_view_cfg=None,
                     n_landmarks=14, backbone='m1'):
    """
    RayNet v5/v6 factory with switchable backbone.

    Four RepNeXt instances are created and split:
      - bb_shared   → stem + s0 + s1          (SharedStem)
      - bb_landmark → s2 + s3                  (LandmarkBranch enc)
      - bb_pose     → s2 + s3                  (PoseBranch enc)
      - bb_gaze     → s2 + s3                  (GazeBranch enc)
    Each branch produces its own 7x7 / 14x14 feature pyramid so training
    doesn't force one set of features to satisfy three objectives.

    Args:
        backbone_weight_path: optional pretrained checkpoint path.
        cross_view_cfg: kwargs for CrossViewAttention.
        n_landmarks: landmark count for the heatmap head.
        backbone: 'm1' (default — embed_dim=(48,96,192,384), depth=
            (3,3,15,2)) or 'm3' (embed_dim=(64,128,256,512), depth=
            (3,3,13,2)). The v6.2 default for medical-grade gaze is
            'm3'; M1 is preserved for ablation A/Bs.
    """
    if backbone not in BACKBONE_CHANNELS:
        raise ValueError(
            f"backbone must be one of {sorted(BACKBONE_CHANNELS)}, "
            f"got {backbone!r}")
    backbone_name = f'repnext_{backbone}'
    bb_channels = BACKBONE_CHANNELS[backbone]

    def _make_backbone(weight_path):
        if weight_path:
            from backbone.repnext_utils import load_pretrained_repnext
            return load_pretrained_repnext(backbone_name, weight_path)
        return create_repnext(backbone_name, pretrained=False)

    bb_shared = _make_backbone(backbone_weight_path)
    bb_landmark = _make_backbone(backbone_weight_path)
    bb_pose = _make_backbone(backbone_weight_path)
    bb_gaze = _make_backbone(backbone_weight_path)

    stem, s0, s1, _, _ = _split_m1_backbone(bb_shared)
    _, _, _, lm_s2, lm_s3 = _split_m1_backbone(bb_landmark)
    _, _, _, ps_s2, ps_s3 = _split_m1_backbone(bb_pose)
    _, _, _, gz_s2, gz_s3 = _split_m1_backbone(bb_gaze)

    shared_stem = SharedStem(stem, s0, s1)

    landmark_branch = FPNLandmarkBranch(
        n_landmarks=n_landmarks, backbone_channels=bb_channels)
    landmark_branch.encoder = BranchEncoder(lm_s2, lm_s3)

    pose_branch = PoseBranch(d_model=256, backbone_channels=bb_channels)
    pose_branch.encoder = BranchEncoder(ps_s2, ps_s3)

    gaze_branch = GazeBranch(d_model=256, backbone_channels=bb_channels)
    gaze_branch.encoder = BranchEncoder(gz_s2, gz_s3)

    model = RayNetV5(
        shared_stem=shared_stem,
        landmark_branch=landmark_branch,
        gaze_branch=gaze_branch,
        pose_branch=pose_branch,
        cross_view_cfg=cross_view_cfg,
    )
    model = model.to(device)

    def _count(module):
        return sum(p.numel() for p in module.parameters()) / 1e6

    total = _count(model)
    print(f"RayNet v6 (Triple-{backbone.upper()}, FPANet landmark + AERI gaze) created:")
    print(f"  SharedStem:      {_count(shared_stem):.2f}M")
    print(f"  LandmarkBranch:  {_count(landmark_branch):.2f}M "
          f"(encoder {_count(landmark_branch.encoder):.2f}M + PANet "
          f"+ heatmap/offset heads)")
    print(f"  PoseBranch:      {_count(pose_branch):.2f}M "
          f"(encoder + CoordAtt + BoxEncoder + head)")
    print(f"  GazeBranch:      {_count(gaze_branch):.2f}M "
          f"(encoder {_count(gaze_branch.encoder):.2f}M + FPN-AERI "
          f"+ CoordAtt + fusion + geometric head)")
    print(f"  CrossView+Cam:   "
          f"{_count(model.cross_view_attn) + _count(model.camera_embedding):.2f}M")
    print(f"  Total:           {total:.1f}M params")
    print(f"  Device:          {device}")
    return model
