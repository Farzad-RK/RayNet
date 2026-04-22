"""
RayNet v5 — Triple-M1 architecture with full-face branches + AERI gaze.

Three task-specific branches, all consuming the 224x224 full-face crop
through a single shared stem:

  Landmark branch (OWNS the shared stem):
    SharedStem (M1 stem+s0+s1) → LandmarkBranchEncoder (M1 s2+s3)
      → U-Net decoder (56x56) → 14 landmark heatmaps + soft-argmax.
    Only task whose gradient backpropagates into the shared stem —
    pose and gaze both detach() their stem input so the low-level
    representation is steered by landmark loss alone.

  Pose branch (gradient-isolated from stem + fuses MAGE bbox):
    s1.detach() → PoseBranchEncoder (M1 s2+s3) → coord-att + pool + proj
    ⊕ BoxEncoder(face_bbox)   (zero-init residual)
      → pose_feat (d_model), 6D rotation, 3D translation.
    BoxEncoder lives inside PoseBranch now — the face bbox is
    head-pose information, not gaze information.

  Gaze branch (gradient-isolated + AERI):
    s1.detach() → GazeBranchEncoder (M1 s2+s3)
      → AERIHead (mini U-Net → iris + eyeball mask logits @ 56x56)
      → AERI attention: predicted eyeball mask gates the gaze feature
         map so the pooled vector is dominated by eye-region features
         without any geometric cropping.
      → GazeFusionBlock(gaze_feat, pose_feat, zero-init residual)
      → CrossViewAttention (when n_views > 1)
      → GeometricGazeHead → eyeball_center + pupil_center +
         optical_axis = normalize(pupil - eyeball) (GazeGene Sec 4.2.2).

AERI (Anatomical Eye Region Isolation) replaces the previous eye-crop
encoder: instead of cropping at the pixel level and running a private
RepNeXt-M1 on the patch, the gaze branch learns WHERE the eye is via
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
detach().

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

# RepNeXt-M1: embed_dim=(48, 96, 192, 384), depth=(2, 2, 6, 2)
M1_CHANNELS = [48, 96, 192, 384]


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


# ─── U-Net Landmark Branch ──────────────────────────────────────────

class AttentionGate(nn.Module):
    """Attention gate for skip connections (Oktay et al., 2018)."""

    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.W_x = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        g = self.W_g(F.interpolate(gate, skip.shape[2:],
                                   mode='bilinear', align_corners=False))
        x = self.W_x(skip)
        att = self.psi(g + x)
        return skip * att


class UNetDecoderBlock(nn.Module):
    """U-Net decoder block: upsample + attention-gated skip + double conv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=False)
        self.att_gate = AttentionGate(in_ch, skip_ch, inter_ch=skip_ch // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.att_gate(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetLandmarkBranch(nn.Module):
    """
    Landmark detection via U-Net encoder-decoder with attention gates.

    Encoder: BranchEncoder on SharedStem s1 output (192ch@14, 384ch@7).
    Decoder: 7→14→28→56 with skip connections from s2, s1, s0.
    Output: 14 heatmaps @ 56x56 + soft-argmax coordinates + subpixel
    offset refinement.

    14 landmarks = 10 iris contour + 4 pupil boundary, GazeGene Sec 4.1.
    """

    def __init__(self, n_landmarks=14):
        super().__init__()
        self.encoder = None  # set by factory
        self.n_landmarks = n_landmarks

        self.dec3 = UNetDecoderBlock(in_ch=384, skip_ch=192, out_ch=192)
        self.dec2 = UNetDecoderBlock(in_ch=192, skip_ch=96,  out_ch=96)
        self.dec1 = UNetDecoderBlock(in_ch=96,  skip_ch=48,  out_ch=48)

        self.heatmap = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, n_landmarks, 1),
        )
        self.offset = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, n_landmarks * 2, 1),
        )

    def forward(self, s0, s1, s2, s3):
        d3 = self.dec3(s3, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)

        hm = self.heatmap(d1)
        off = self.offset(d1)
        coords = self._soft_argmax(hm, off)
        return coords, hm

    def _soft_argmax(self, hm, off):
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


# ─── AERI Segmentation Head ────────────────────────────────────────

class AERIHead(nn.Module):
    """
    Anatomical Eye Region Isolation head (MSGazeNet-style).

    Mini U-Net built on the gaze branch's own feature pyramid,
    producing two binary-segmentation logit maps at 56x56:
      - iris_logits    (channel 0)
      - eyeball_logits (channel 1)

    The decoder reuses the landmark-style UNetDecoderBlock, with s0/s1
    from the shared stem as detached skips (the same tensors landmark
    uses, but with gradient blocked so AERI loss doesn't leak into the
    stem — the stem is landmark-owned).

    Supervised by iris + eyeball binary masks baked into the MDS
    shards (see RayNet.streaming.eye_masks). The predicted eyeball
    mask is also used inside GazeBranch as a soft attention gate on
    the 7x7 gaze feature map, making the pooled gaze vector
    eye-region-dominant without any geometric cropping.
    """

    def __init__(self):
        super().__init__()
        self.dec3 = UNetDecoderBlock(in_ch=384, skip_ch=192, out_ch=192)
        self.dec2 = UNetDecoderBlock(in_ch=192, skip_ch=96,  out_ch=96)
        self.dec1 = UNetDecoderBlock(in_ch=96,  skip_ch=48,  out_ch=48)
        self.head = nn.Conv2d(48, 2, 1)

    def forward(self, s0, s1, s2, s3):
        d3 = self.dec3(s3, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)
        logits = self.head(d1)                # (B, 2, 56, 56)
        iris = logits[:, 0]                   # (B, 56, 56)
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
    GazeGene 3D Eyeball Structure Estimation head (Sec 4.2.2).

    Predicts two 3D points that geometrically define the optical axis:
      - eyeball_center_3d (B, 3)
      - pupil_center_3d   (B, 3)
    optical_axis = normalize(pupil_center - eyeball_center).
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

    def forward(self, pooled):
        eyeball_center = self.eyeball_fc(pooled)
        pupil_center = self.pupil_fc(pooled)

        optical_axis = F.normalize(pupil_center - eyeball_center, dim=-1)

        x, y, z = optical_axis[:, 0], optical_axis[:, 1], optical_axis[:, 2]
        pitch = torch.asin((-y).clamp(-1 + 1e-6, 1 - 1e-6))
        yaw = torch.atan2(-x, -z)
        gaze_angles = torch.stack([pitch, yaw], dim=-1)
        return eyeball_center, pupil_center, optical_axis, gaze_angles


# ─── Gaze Branch ────────────────────────────────────────────────────

class GazeBranch(nn.Module):
    """
    AERI-based gaze branch.

    Pipeline (operating on gradient-isolated shared-stem features):
        s0.detach(), s1.detach() (from SharedStem)
        s1.detach() → BranchEncoder → gaze_s2 (14x14), gaze_s3 (7x7)
        AERIHead(s0, s1, gaze_s2, gaze_s3)
          → iris_logits, eyeball_logits (both 56x56)
        AERI attention:
          effective_mask = 0.25 + 0.75 * sigmoid(eyeball_logits)
          gaze_s3 ← gaze_s3 * avg_pool(effective_mask → 7x7)
        CoordinateAttention → AdaptiveAvgPool → Linear → gaze_feat
        GazeFusionBlock(gaze_feat, pose_feat) (zero-init residual for pose)
        CrossViewAttention (when n_views > 1)
        GeometricGazeHead → eyeball_center, pupil_center, optical_axis
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.encoder = None  # set by factory (BranchEncoder)
        self.aeri = AERIHead()

        self.coord_att = CoordinateAttention(384)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(384, d_model)
        # Mask-Weighted Foveal Pooling
        # 384 (global context from s3) + 48 (foveal precision from d1) = 432
        self.proj = nn.Linear(384 + 48, d_model)

        self.fusion_block = GazeFusionBlock(d_model)
        self.head = GeometricGazeHead(d_model)

    def forward(self, s0_detached, s1_detached, pose_feat,
                    cross_view_attn=None, n_views=1, cam_embed=None,
                    aeri_alpha=0.2):  # Default to low influence for safety
        gaze_s2, gaze_s3 = self.encoder(s1_detached)

        # iris_logits, eyeball_logits = self.aeri(
        #     s0_detached, s1_detached, gaze_s2, gaze_s3)
        #
        #
        # # AERI attention on the 7x7 bottleneck. Downsample the 56x56
        # # eyeball sigmoid to 7x7 via area-average, apply baseline +
        # # foreground scaling so the background still contributes 25%
        # # (prevents zero-mask collapse early in training when AERI is
        # # still random).
        # eye_attn_56 = torch.sigmoid(eyeball_logits).unsqueeze(1)
        # eye_attn_7 = F.adaptive_avg_pool2d(eye_attn_56, 7)
        # gaze_s3 = gaze_s3 * (0.25 + 0.75 * eye_attn_7)
        #
        # feat = self.coord_att(gaze_s3)
        # gaze_feat = self.proj(self.pool(feat).flatten(1))
        # 1. Unpack d1 from AERI
        #
        # iris_logits, eyeball_logits, d1 = self.aeri(
        #     s0_detached, s1_detached, gaze_s2, gaze_s3)
        #
        # # 2. Create the 56x56 probability mask
        # eye_attn_56 = torch.sigmoid(eyeball_logits).unsqueeze(1)
        #
        # # 3. Apply global attention to the 7x7 bottleneck (keep existing logic)
        # eye_attn_7 = F.adaptive_avg_pool2d(eye_attn_56, 7)
        # gaze_s3 = gaze_s3 * (0.25 + 0.75 * eye_attn_7)
        #
        # # 4. Extract Global Features (1D vector)
        # feat = self.coord_att(gaze_s3)
        # global_feat = self.pool(feat).flatten(1)  # Shape: (B, 384)
        #
        # # 5. Extract Foveal Features (The SOTA Fix)
        # # Multiply high-res d1 by the eye mask to remove background noise
        # foveal_map = d1 * eye_attn_56  # Shape: (B, 48, 56, 56)
        # foveal_feat = self.pool(foveal_map).flatten(1)  # Shape: (B, 48)
        #
        # # 6. Fuse and Project
        # fused_feat = torch.cat([global_feat, foveal_feat], dim=1)  # Shape: (B, 432)
        # gaze_feat = self.proj(fused_feat)  # Shape: (B, d_model)

        # 1. High-Res Features from AERI
        iris_logits, eyeball_logits, d1 = self.aeri(
            s0_detached, s1_detached, gaze_s2, gaze_s3)

        iris_mask = torch.sigmoid(iris_logits).unsqueeze(1)
        eye_mask = torch.sigmoid(eyeball_logits).unsqueeze(1)

        # 2. Saliency Calculation
        saliency_56 = (0.8 * iris_mask) + (0.2 * eye_mask)

        # 3. Refined Influence Scheduling (AERI Influence)
        # We blend saliency with a uniform field.
        # When alpha is low, the model sees the whole 56x56 field equally.
        uniform_mask = torch.ones_like(saliency_56)
        scheduled_mask = (aeri_alpha * saliency_56) + ((1.0 - aeri_alpha) * uniform_mask)

        # 4. Stochastic Masking (The "Anti-Shortcut" Trigger)
        # Prevents the model from relying 100% on specific pixel-mask correlations
        if self.training:
            # 10% chance to drop the mask influence entirely for a batch
            mask_dropout = (torch.rand(1, device=scheduled_mask.device) > 0.1).float()
            scheduled_mask = scheduled_mask * mask_dropout + (1.0 - mask_dropout) * uniform_mask

        # 5. Feature Extraction
        eye_attn_7 = F.adaptive_avg_pool2d(scheduled_mask, 7)
        gaze_s3 = gaze_s3 * (0.25 + 0.75 * eye_attn_7)
        global_feat = self.pool(self.coord_att(gaze_s3)).flatten(1)

        foveal_map = d1 * scheduled_mask
        foveal_feat = self.pool(foveal_map).flatten(1)

        # 6. Stabilized Fusion
        fused_feat = torch.cat([global_feat, foveal_feat], dim=1)
        fused_feat = self.fusion_norm(fused_feat)
        gaze_feat = self.proj(fused_feat)

        # pooled_sv is the pre-CrossViewAttention (single-view) representation.
        # Returned separately so the training loop can supervise the single-view
        # pathway directly, preventing val degradation from CrossViewAttention
        # train/val asymmetry (train uses 9-view fusion; val bypasses it).
        pooled_sv = self.fusion_block(gaze_feat, pose_feat)

        pooled = pooled_sv
        if cross_view_attn is not None:
            pooled = cross_view_attn(pooled_sv, n_views, cam_embed)

        eyeball_center, pupil_center, optical_axis, gaze_angles = \
            self.head(pooled)

        # Single-view gaze prediction (no CrossViewAttention).
        # Only computed when multi-view fusion is actually active (n_views > 1)
        # so there is no overhead in single-view inference / validation.
        sv_optical_axis = None
        if n_views > 1:
            _, _, sv_optical_axis, _ = self.head(pooled_sv)

        return (eyeball_center, pupil_center, optical_axis, gaze_angles,
                iris_logits, eyeball_logits, sv_optical_axis)


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
      pred_pose_t  (B, 3): translation in meters (raw linear head)
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.encoder = None  # set by factory (BranchEncoder)
        self.coord_att = CoordinateAttention(384)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(384, d_model)

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

        self.head = nn.Linear(d_model, 9)  # 6D rot + 3D translation

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

        out = self.head(pose_feat)
        pred_pose_6d = out[:, :6]
        pred_pose_t = out[:, 6:]  # meters
        return pose_feat, pred_pose_6d, pred_pose_t


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

    def __init__(self, d_model=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )

    def forward(self, R_cam, T_cam):
        x = torch.cat([R_cam.flatten(1), T_cam], dim=-1)
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

        # Pose branch — gradient-isolated + bbox fusion.
        pose_feat, pred_pose_6d, pred_pose_t = self.pose_branch(
            s1.detach(), face_bbox=face_bbox)

        # Gaze branch — gradient-isolated + AERI attention.
        # pose_feat is detached before cam_embed so gaze loss cannot
        # backpropagate into pose_branch through this side path.
        cam_embed = None
        if R_cam is not None and T_cam is not None:
            cam_embed = self.camera_embedding(R_cam, T_cam) + pose_feat.detach()

        (eyeball_center, pupil_center, optical_axis, gaze_angles,
         iris_mask_logits, eyeball_mask_logits,
         sv_optical_axis) = self.gaze_branch(
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
            'gaze_vector': optical_axis,               # (B, 3) unit
            'gaze_angles': gaze_angles,                # (B, 2) pitch/yaw
            # Single-view gaze (pre-CrossViewAttention); None when n_views==1
            'gaze_vector_sv': sv_optical_axis,         # (B, 3) unit or None
            # Pose
            'pred_pose_6d': pred_pose_6d,              # (B, 6)
            'pred_pose_t': pred_pose_t,                # (B, 3) m
        }


# ─── Factory Functions ──────────────────────────────────────────────

def _split_m1_backbone(m1):
    return m1.stem, m1.stages[0], m1.stages[1], m1.stages[2], m1.stages[3]


def create_raynet_v5(backbone_weight_path=None, cross_view_cfg=None,
                     n_landmarks=14):
    """
    Triple-M1 factory.

    Four RepNeXt-M1 instances are created and split:
      - m1_shared   → stem + s0 + s1          (SharedStem, 1.5M)
      - m1_landmark → s2 + s3                  (LandmarkBranch enc, 3.3M)
      - m1_pose     → s2 + s3                  (PoseBranch enc, 3.3M)
      - m1_gaze     → s2 + s3                  (GazeBranch enc, 3.3M)
    "Triple-M1" refers to the three task-specific s2+s3 branches above
    the single shared stem. Each branch produces its own 7x7 / 14x14
    feature pyramid, so training doesn't force one set of features
    to satisfy three objectives at once.
    """

    def _make_m1(weight_path):
        if weight_path:
            from backbone.repnext_utils import load_pretrained_repnext
            return load_pretrained_repnext('repnext_m1', weight_path)
        return create_repnext('repnext_m1', pretrained=False)

    m1_shared = _make_m1(backbone_weight_path)
    m1_landmark = _make_m1(backbone_weight_path)
    m1_pose = _make_m1(backbone_weight_path)
    m1_gaze = _make_m1(backbone_weight_path)

    stem, s0, s1, _, _ = _split_m1_backbone(m1_shared)
    _, _, _, lm_s2, lm_s3 = _split_m1_backbone(m1_landmark)
    _, _, _, ps_s2, ps_s3 = _split_m1_backbone(m1_pose)
    _, _, _, gz_s2, gz_s3 = _split_m1_backbone(m1_gaze)

    shared_stem = SharedStem(stem, s0, s1)

    landmark_branch = UNetLandmarkBranch(n_landmarks=n_landmarks)
    landmark_branch.encoder = BranchEncoder(lm_s2, lm_s3)

    pose_branch = PoseBranch(d_model=256)
    pose_branch.encoder = BranchEncoder(ps_s2, ps_s3)

    gaze_branch = GazeBranch(d_model=256)
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
    print(f"RayNet v5 (Triple-M1, AERI gaze) created:")
    print(f"  SharedStem:      {_count(shared_stem):.2f}M")
    print(f"  LandmarkBranch:  {_count(landmark_branch):.2f}M "
          f"(encoder {_count(landmark_branch.encoder):.2f}M + U-Net decoder)")
    print(f"  PoseBranch:      {_count(pose_branch):.2f}M "
          f"(encoder + CoordAtt + BoxEncoder + head)")
    print(f"  GazeBranch:      {_count(gaze_branch):.2f}M "
          f"(encoder {_count(gaze_branch.encoder):.2f}M + AERI U-Net + "
          f"CoordAtt + fusion + geometric head)")
    print(f"  CrossView+Cam:   "
          f"{_count(model.cross_view_attn) + _count(model.camera_embedding):.2f}M")
    print(f"  Total:           {total:.1f}M params")
    print(f"  Device:          {device}")
    return model
