"""
RayNet v5 — Quad-M1 architecture with dedicated eye-crop gaze encoder.

Four task-specific paths:
  1. SharedStem (M1 stem+s0+s1) — low-level features for landmark + pose
  2. LandmarkBranch (M1 s2+s3 → U-Net decoder) — 14 iris/pupil landmarks
     at 56x56 resolution with attention-gated skip connections.
  3. GazeBranch — dedicated pipeline:
        predicted landmarks → EyeCropModule (differentiable, 112x112)
          → full RepNeXt-M1 EyeBackbone (private to gaze)
          → GazeFusionBlock(eye, pose, bbox) → GeometricGazeHead
     The gaze branch does NOT share features with landmark/pose; it
     gets a high-resolution view of the iris through landmark-guided
     cropping. This breaks the face-level 14x14 bottleneck that was
     capping val_angular near 42 degrees.
  4. PoseBranch (M1 s2+s3) — 6D rotation + 3D translation from
     gradient-isolated shared-stem features.

Backbone: 4 x RepNeXt-M1 (~4.8M each, ~19M total backbone)
  shared (stem+s0+s1), landmark (s2+s3), pose (s2+s3), eye (all 4 stages).

MAGE integration (Sec 3.2):
  BoxEncoder(face_bbox) feeds into the gaze fusion block alongside
  the pose embedding and eye-crop feature, providing gaze-origin
  information without requiring 468-point MediaPipe at inference.

GazeGene 3D Eyeball Structure Estimation (Sec 4.2.2):
  Predict eyeball_center_3d AND pupil_center_3d, derive
  optical_axis = normalize(pupil - eyeball).

Stage 2 curriculum:
  Phases 1-2 freeze shared_stem + landmark_branch + pose_branch and
  train only the gaze branch; phase 3 unfreezes for joint refinement.
  Predicted landmarks are always detached before feeding the eye crop
  to eliminate train/test distribution gap.

Input:  (3, 224, 224) GazeGene face crop + (3,) face bounding box
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from backbone.repnext import create_repnext
from RayNet.coordatt import CoordinateAttention
from RayNet.eye_crop import EyeCropModule

device = "cuda" if torch.cuda.is_available() else "cpu"

# RepNeXt-M1: embed_dim=(48, 96, 192, 384), depth=(2, 2, 6, 2)
M1_CHANNELS = [48, 96, 192, 384]


# ─── Shared Stem ────────────────────────────────────────────────────

class SharedStem(nn.Module):
    """
    Shared low-level encoder: RepNeXt-M1 stem + stages[0] + stages[1].

    Extracts edges, textures, and low-level facial features shared by
    all three task branches. Output: 96ch at 28x28 (stride 8 for 224 input).

    Intermediate features at each stage are returned for U-Net skip
    connections in the landmark branch.

    ~1.5M params.
    """

    def __init__(self, stem, stage0, stage1):
        super().__init__()
        self.stem = stem        # 3→48ch, stride 4 (56x56)
        self.stage0 = stage0    # 48→48ch, no downsample (56x56)
        self.stage1 = stage1    # 48→96ch, downsample 2x (28x28)

    def forward(self, x):
        """
        Returns:
            s0: (B, 48, 56, 56)  — skip connection for U-Net
            s1: (B, 96, 28, 28)  — input to all three branches + skip
        """
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
        s2: 192ch at 14x14 (stride 16)
        s3: 384ch at 7x7   (stride 32)

    ~3.3M params per branch.
    """

    def __init__(self, stage2, stage3):
        super().__init__()
        self.stage2 = stage2    # 96→192ch, downsample 2x (14x14)
        self.stage3 = stage3    # 192→384ch, downsample 2x (7x7)

    def forward(self, s1):
        """
        Args:
            s1: (B, 96, 28, 28) from SharedStem

        Returns:
            s2: (B, 192, 14, 14)
            s3: (B, 384, 7, 7)
        """
        s2 = checkpoint(self.stage2, s1, use_reentrant=False)
        s3 = checkpoint(self.stage3, s2, use_reentrant=False)
        return s2, s3


# ─── MAGE-style Box Encoder + Fusion Block ─────────────────────────

class BoxEncoder(nn.Module):
    """
    MAGE-style face bounding box encoder (Sec 3.2).

    Encodes face bounding box parameters (x_p, y_p, L_x) into a
    d_model-dimensional embedding that carries positional/spatial
    information about the face center location and the coordinate
    system rotation during Easy-Norm normalization.

    At inference, only a fast face bounding box detector is needed
    (YOLO/Haar/MediaPipe face detection) — no 468-point landmark mesh.

    Architecture: 3 → 64 → 128 → d_model with GELU activations.
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
        """
        Args:
            bbox: (B, 3) face bounding box [x_p, y_p, L_x]
                  x_p, y_p = face center in image coordinates
                  L_x = face width (proxy for depth/scale)

        Returns:
            (B, d_model) bounding box embedding
        """
        return self.mlp(bbox)


# ─── U-Net Landmark Branch ──────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Attention gate for skip connections (Oktay et al., 2018).

    Learns which spatial locations in the skip features are relevant
    for the current decoder level, suppressing noise from irrelevant
    regions.
    """

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
    """
    U-Net decoder block: upsample + attention-gated skip + double conv.

    Uses bilinear upsampling (no checkerboard artifacts from ConvTranspose).
    """

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

    Encoder path: BranchEncoder (M1 stages 2-3) produces 192ch@14x14
    and 384ch@7x7. Decoder upsamples back to 56x56 with skip connections
    from the shared stem and branch encoder.

    Output: 14 landmark heatmaps + coordinates via soft-argmax with
    subpixel offset refinement.

    14 landmarks: 10 iris contour + 4 pupil boundary points.
    """

    def __init__(self, n_landmarks=14):
        super().__init__()
        self.encoder = None  # Set by RayNetV5.__init__
        self.n_landmarks = n_landmarks

        # Decoder: 7x7 → 14x14 → 28x28 → 56x56
        # in_ch = upsampled, skip_ch = from encoder, out_ch = decoder output
        self.dec3 = UNetDecoderBlock(in_ch=384, skip_ch=192, out_ch=192)  # 7→14
        self.dec2 = UNetDecoderBlock(in_ch=192, skip_ch=96,  out_ch=96)   # 14→28
        self.dec1 = UNetDecoderBlock(in_ch=96,  skip_ch=48,  out_ch=48)   # 28→56

        # Landmark heatmap head
        self.heatmap = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, n_landmarks, 1),
        )

        # Subpixel offset refinement
        self.offset = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, n_landmarks * 2, 1),
        )

    def forward(self, s0, s1, s2, s3):
        """
        Args:
            s0: (B, 48, 56, 56)  from SharedStem stage0 (skip)
            s1: (B, 96, 28, 28)  from SharedStem stage1 (skip)
            s2: (B, 192, 14, 14) from landmark BranchEncoder (skip)
            s3: (B, 384, 7, 7)   from landmark BranchEncoder (bottleneck)

        Returns:
            coords: (B, 14, 2) landmark pixel coordinates in 56x56 space
            heatmaps: (B, 14, 56, 56) raw logit heatmaps
        """
        # Decoder with attention-gated skip connections
        d3 = self.dec3(s3, s2)   # (B, 192, 14, 14)
        d2 = self.dec2(d3, s1)   # (B, 96, 28, 28)
        d1 = self.dec1(d2, s0)   # (B, 48, 56, 56)

        # Heatmaps + soft-argmax
        hm = self.heatmap(d1)    # (B, 14, 56, 56)
        off = self.offset(d1)    # (B, 28, 56, 56)
        coords = self._soft_argmax(hm, off)

        return coords, hm

    def _soft_argmax(self, hm, off):
        """Differentiable soft-argmax with subpixel offset refinement."""
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


# ─── Gaze Branch with Explicit 3D Geometry ──────────────────────────

class EyeBackbone(nn.Module):
    """
    Dedicated full RepNeXt-M1 backbone for the eye-crop branch.

    Unlike the other branches which share the stem+s0+s1 path, the eye
    encoder takes a 112x112 landmark-guided eye patch and runs ALL four
    RepNeXt stages privately. This gives the gaze pathway a high-res
    view of the iris/pupil, breaking the face-level 14x14 feature-map
    bottleneck that was capping val_angular near 42 degrees when the
    gaze branch shared low-level features with landmark/pose.

    At 112x112 input the final feature map is 384ch @ ~4x4.

    ~4.8M params.
    """

    def __init__(self, stem, stage0, stage1, stage2, stage3):
        super().__init__()
        self.stem = stem
        self.stage0 = stage0
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3

    def forward(self, x):
        x = checkpoint(self.stem, x, use_reentrant=False)
        x = checkpoint(self.stage0, x, use_reentrant=False)
        x = checkpoint(self.stage1, x, use_reentrant=False)
        x = checkpoint(self.stage2, x, use_reentrant=False)
        x = checkpoint(self.stage3, x, use_reentrant=False)
        return x


class GazeFusionBlock(nn.Module):
    """
    Merge three d_model streams into the gaze embedding:
      - eye_feat:  dedicated eye-crop encoder output (anchor)
      - pose_feat: head-pose embedding from PoseBranch
      - box_feat:  face-bbox encoding from MAGE BoxEncoder

    Eye feature is the residual anchor; pose and box add corrective
    information. Output projection is zero-init, so at start the model
    predicts purely from the eye crop — pose/box enter gradually
    through training without disrupting the eye-encoder signal.
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)

    def forward(self, eye_feat, pose_feat, box_feat):
        fused = self.proj(torch.cat([eye_feat, pose_feat, box_feat], dim=-1))
        return eye_feat + fused


class GeometricGazeHead(nn.Module):
    """
    GazeGene-style 3D Eyeball Structure Estimation head (Sec 4.2.2).

    Predicts two 3D points that define the optical axis geometrically:
      - eyeball_center_3d (B, 3): eyeball center in CCS (cm)
      - pupil_center_3d (B, 3): pupil center in CCS (cm)

    The optical axis is derived from geometry:
        optical_axis = normalize(pupil_center - eyeball_center)

    This is physically grounded: the optical axis IS the line from
    eyeball center through the pupil. By predicting both endpoints
    and deriving the direction, the model is forced to learn
    anatomically consistent 3D structure.

    Supervised by 4 losses (GazeGene Sec 4.2.2):
      1. L1 on eyeball_center_3d vs GT
      2. L1 on pupil_center_3d vs GT
      3. L1 on iris contour 2D (from landmark branch, not here)
      4. Angular error between optical_axis (from predicted geometry)
         and GT optical axis
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
        """
        Args:
            pooled: (B, d_model) gaze features after modulation/attention

        Returns:
            eyeball_center: (B, 3) predicted eye center in CCS (cm)
            pupil_center: (B, 3) predicted pupil center in CCS (cm)
            optical_axis: (B, 3) unit gaze direction (derived from geometry)
            gaze_angles: (B, 2) pitch, yaw in radians (for visualization)
        """
        eyeball_center = self.eyeball_fc(pooled)
        pupil_center = self.pupil_fc(pooled)

        # Derive optical axis from predicted 3D structure
        # This is the key GazeGene insight: direction comes from geometry
        optical_axis = F.normalize(pupil_center - eyeball_center, dim=-1)

        # Derive pitch/yaw for visualization / backward compat
        x = optical_axis[:, 0]
        y = optical_axis[:, 1]
        z = optical_axis[:, 2]
        pitch = torch.asin((-y).clamp(-1 + 1e-6, 1 - 1e-6))
        yaw = torch.atan2(-x, -z)
        gaze_angles = torch.stack([pitch, yaw], dim=-1)

        return eyeball_center, pupil_center, optical_axis, gaze_angles


class GazeBranch(nn.Module):
    """
    Gaze branch with dedicated landmark-guided eye-crop encoder.

    Pipeline:
        image + landmarks_px
          → EyeCropModule (differentiable, 112x112, 25% pad)
          → EyeBackbone (full RepNeXt-M1, private to gaze)
          → CoordAtt + pool + proj → eye_feat (d_model)
        + BoxEncoder(face_bbox) → box_feat (d_model)
        + pose_feat (from PoseBranch, can be zeroed via use_pose_bridge)
          → GazeFusionBlock (eye anchor, zero-init residual for pose/box)
          → CrossViewAttention (if n_views > 1)
          → GeometricGazeHead → eyeball_center + pupil_center + optical_axis

    Rationale: the face-level 14x14 feature map was a hard spatial
    ceiling for gaze — iris occupied 2-3 cells. By cropping to the eye
    and running a private backbone, the iris gets a native 28x28 stem
    map with far more spatial detail. The zero-init fusion ensures that
    pose and bbox add refinement without disrupting the eye signal.
    """

    def __init__(self, d_model=256, eye_out_size=112, eye_pad_frac=0.25):
        super().__init__()
        self.eye_crop = EyeCropModule(
            out_size=eye_out_size, pad_frac=eye_pad_frac)
        self.encoder = None  # EyeBackbone, set by factory
        self.coord_att = CoordinateAttention(384)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(384, d_model)

        self.box_encoder = BoxEncoder(d_model)
        self.fusion_block = GazeFusionBlock(d_model)

        self.head = GeometricGazeHead(d_model)

    def forward(self, image, landmarks_px, pose_feat, face_bbox=None,
                cross_view_attn=None, n_views=1, cam_embed=None):
        """
        Args:
            image: (B, 3, H, W) raw face image (H=W=224 expected)
            landmarks_px: (B, 14, 2) predicted landmarks in PIXEL space
                          of `image`. Caller should detach to keep
                          gradients from flowing back to the landmark
                          branch during Stage 2 (landmark branch frozen).
            pose_feat: (B, d_model) from PoseBranch
            face_bbox: (B, 3) MAGE face bbox [x_p, y_p, L_x] or None
            cross_view_attn: optional CrossViewAttention module
            n_views: views per group (1=single, 9=multi-view)
            cam_embed: (B, d_model) camera embedding or None

        Returns:
            eyeball_center: (B, 3)
            pupil_center: (B, 3)
            optical_axis: (B, 3)
            gaze_angles: (B, 2)
        """
        eye = self.eye_crop(image, landmarks_px)            # (B, 3, 112, 112)
        feat = self.encoder(eye)                            # (B, 384, h, w)
        feat = self.coord_att(feat)
        eye_feat = self.proj(self.pool(feat).flatten(1))    # (B, d_model)

        if face_bbox is not None:
            box_feat = self.box_encoder(face_bbox)
        else:
            box_feat = torch.zeros_like(eye_feat)

        pooled = self.fusion_block(eye_feat, pose_feat, box_feat)

        if cross_view_attn is not None:
            pooled = cross_view_attn(pooled, n_views, cam_embed)

        return self.head(pooled)


# ─── Pose Branch ────────────────────────────────────────────────────

class PoseBranch(nn.Module):
    """
    Head pose estimation branch: encoder + CoordAtt + pool + heads.

    Takes gradient-detached shared stem features to prevent pose
    gradients from steering the shared low-level representation
    away from what landmark and gaze need.

    Outputs:
      - pose_feat (B, d_model): implicit pose embedding for gaze modulation
      - pred_pose_6d (B, 6): 6D rotation (Gram-Schmidt → SO(3))
      - pred_pose_t (B, 3): translation in meters (raw linear)
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.encoder = None  # Set by RayNetV5.__init__
        self.coord_att = CoordinateAttention(384)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(384, d_model)
        self.head = nn.Linear(d_model, 9)  # 6D rotation + 3D translation

    def forward(self, s1_detached):
        """
        Args:
            s1_detached: (B, 96, 28, 28) detached from SharedStem

        Returns:
            pose_feat: (B, d_model) for gaze modulation
            pred_pose_6d: (B, 6)
            pred_pose_t: (B, 3) in meters
        """
        s2, s3 = self.encoder(s1_detached)
        s3_att = self.coord_att(s3)
        pooled = self.pool(s3_att).flatten(1)
        pose_feat = self.proj(pooled)

        out = self.head(pose_feat)
        pred_pose_6d = out[:, :6]
        pred_pose_t = out[:, 6:]  # raw linear, meters

        return pose_feat, pred_pose_6d, pred_pose_t


# ─── Cross-View and Camera Modules (reused from v4.1) ──────────────

class CrossViewAttention(nn.Module):
    """
    Pre-norm Transformer Encoder for cross-view gaze feature fusion.
    Identical to v4.1. Single-view (n_views=1) bypasses (identity).
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
    """Encode R_cam (3x3) + T_cam (3) → d_model embedding. Reused from v4.1."""

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
    RayNet v5: Quad-M1 architecture with dedicated eye-crop gaze encoder.

    Architecture:
        Image (224x224) ─→ SharedStem (M1 stem+s0+s1)
                            ├── LandmarkBranch (M1 s2+s3 → U-Net → 14 landmarks)
                            └── PoseBranch     (M1 s2+s3 → 6D rot + translation)
                                                  ↑ gradient detached from shared stem

        Image (224x224) + predicted landmarks (detached)
            ─→ GazeBranch:
                EyeCropModule → 112x112 eye patch
                  → EyeBackbone (full RepNeXt-M1, private to gaze)
                  → GazeFusionBlock(eye, pose, bbox, zero-init residual)
                  → GeometricGazeHead (eyeball_center, pupil_center, optical_axis)

    Landmarks are predicted by the landmark branch, detached, rescaled
    from 56-space to 224-space, and passed to the gaze branch as the
    eye-crop anchor. During Stage 2 phases 1-2 the landmark/pose/stem
    path is frozen (.eval() + requires_grad=False) so the gaze branch
    learns from a stable predicted-landmark distribution — the same
    distribution it sees at inference.
    """

    def __init__(self, shared_stem, landmark_branch, gaze_branch,
                 pose_branch, cross_view_cfg=None, landmark_feat_size=56):
        super().__init__()
        self.shared_stem = shared_stem
        self.landmark_branch = landmark_branch
        self.gaze_branch = gaze_branch
        self.pose_branch = pose_branch
        # Feature-map size the landmark branch predicts coords in
        # (used to rescale coords back to pixel space for eye crop).
        self.landmark_feat_size = landmark_feat_size

        # Cross-view and camera modules
        cv_cfg = cross_view_cfg or {}
        cv_cfg.setdefault('d_model', 256)
        self.cross_view_attn = CrossViewAttention(**cv_cfg)
        self.camera_embedding = CameraEmbedding(d_model=cv_cfg['d_model'])

    def forward(self, x, n_views=1, R_cam=None, T_cam=None,
                face_bbox=None,
                use_landmark_bridge=True, use_pose_bridge=True):
        """
        Args:
            x: (B, 3, 224, 224) face crop
            n_views: views per group (1=single, 9=multi-view)
            R_cam: (B, 3, 3) camera rotation or None
            T_cam: (B, 3) camera translation or None
            face_bbox: (B, 3) face bounding box [x_p, y_p, L_x] or None
                       MAGE-style: provides gaze origin info without
                       external face landmark detector at inference
            use_landmark_bridge: no-op in v5 eye-crop design (landmarks
                are always used to anchor the eye crop). Kept for
                phase-config backward compatibility.
            use_pose_bridge: if False, zero the pose stream into the
                gaze fusion block (gaze predicts from eye crop + bbox
                only, useful for isolating the pose signal).

        Returns:
            dict with all predictions
        """
        # Shared low-level features (landmark + pose consume these)
        s0, s1 = self.shared_stem(x)

        # === Pose Branch (gradient-isolated from shared stem) ===
        pose_feat, pred_pose_6d, pred_pose_t = self.pose_branch(s1.detach())

        # === Landmark Branch ===
        lm_s2, lm_s3 = self.landmark_branch.encoder(s1)
        landmark_coords, landmark_heatmaps = self.landmark_branch(
            s0, s1, lm_s2, lm_s3)

        # === Gaze Branch (dedicated eye-crop encoder) ===
        # Camera embedding
        cam_embed = None
        if R_cam is not None and T_cam is not None:
            cam_embed = self.camera_embedding(R_cam, T_cam)
            cam_embed = cam_embed + pose_feat  # fuse camera + pose

        # Landmarks live in 56x56 feature space; the eye crop operates
        # in the 224-pixel image frame of `x`. Detach so gaze gradients
        # cannot flow back through the landmark branch during Stage 2
        # (when the landmark branch is frozen).
        img_size = x.shape[-1]
        feat_size = landmark_coords.new_tensor(
            float(self.landmark_feat_size))
        landmarks_px = landmark_coords.detach() * (img_size / feat_size)

        gaze_pose_feat = (
            pose_feat if use_pose_bridge else torch.zeros_like(pose_feat)
        )

        eyeball_center, pupil_center, optical_axis, gaze_angles = \
            self.gaze_branch(
                x, landmarks_px, gaze_pose_feat,
                face_bbox=face_bbox,
                cross_view_attn=self.cross_view_attn,
                n_views=n_views,
                cam_embed=cam_embed,
            )

        return {
            # Landmarks
            'landmark_coords': landmark_coords,        # (B, 14, 2) in 56x56 space
            'landmark_heatmaps': landmark_heatmaps,    # (B, 14, 56, 56)
            # Gaze (GazeGene 3D eyeball structure)
            'eyeball_center': eyeball_center,          # (B, 3) CCS, cm
            'pupil_center': pupil_center,              # (B, 3) CCS, cm
            'gaze_vector': optical_axis,               # (B, 3) unit vector (derived)
            'gaze_angles': gaze_angles,                # (B, 2) pitch, yaw
            # Pose
            'pred_pose_6d': pred_pose_6d,              # (B, 6)
            'pred_pose_t': pred_pose_t,                # (B, 3) meters
        }


# ─── Factory Functions ──────────────────────────────────────────────

def _split_m1_backbone(m1):
    """Split a RepNeXt-M1 into stem/stage components."""
    return m1.stem, m1.stages[0], m1.stages[1], m1.stages[2], m1.stages[3]


def create_raynet_v5(backbone_weight_path=None, cross_view_cfg=None,
                     n_landmarks=14):
    """
    Factory: create RayNet v5 with optional pretrained M1 backbone weights.

    Creates 4 RepNeXt-M1 instances. Shares stem+stages[0-1] from the
    first; takes stages[2-3] from each for the three branches.

    All four M1 instances start with the same pretrained weights (if
    provided). During training, the shared stem stays shared while each
    branch's stages 2-3 diverge through task-specific gradients.

    Args:
        backbone_weight_path: path to pretrained RepNeXt-M1 weights.
            None = random init (for inference from checkpoint).
        cross_view_cfg: kwargs for CrossViewAttention
        n_landmarks: number of landmarks (default 14)

    Returns:
        RayNetV5 model
    """
    def _make_m1(weight_path):
        if weight_path:
            from backbone.repnext_utils import load_pretrained_repnext
            return load_pretrained_repnext('repnext_m1', weight_path)
        return create_repnext('repnext_m1', pretrained=False)

    # 4 M1 instances:
    #   shared   → stem + s0 + s1 feed landmark & pose
    #   landmark → s2 + s3 for the landmark branch encoder
    #   pose     → s2 + s3 for the pose branch encoder
    #   eye      → FULL M1 (stem + s0..s3) dedicated to the gaze branch,
    #              operating on 112x112 landmark-guided eye crops
    m1_shared = _make_m1(backbone_weight_path)
    m1_landmark = _make_m1(backbone_weight_path)
    m1_pose = _make_m1(backbone_weight_path)
    m1_eye = _make_m1(backbone_weight_path)

    # Split into components
    stem, s0, s1, _, _ = _split_m1_backbone(m1_shared)
    _, _, _, lm_s2, lm_s3 = _split_m1_backbone(m1_landmark)
    _, _, _, ps_s2, ps_s3 = _split_m1_backbone(m1_pose)
    eye_stem, eye_s0, eye_s1, eye_s2, eye_s3 = _split_m1_backbone(m1_eye)

    # Assemble
    shared_stem = SharedStem(stem, s0, s1)

    landmark_branch = UNetLandmarkBranch(n_landmarks=n_landmarks)
    landmark_branch.encoder = BranchEncoder(lm_s2, lm_s3)

    gaze_branch = GazeBranch(d_model=256)
    gaze_branch.encoder = EyeBackbone(
        eye_stem, eye_s0, eye_s1, eye_s2, eye_s3)

    pose_branch = PoseBranch(d_model=256)
    pose_branch.encoder = BranchEncoder(ps_s2, ps_s3)

    model = RayNetV5(
        shared_stem=shared_stem,
        landmark_branch=landmark_branch,
        gaze_branch=gaze_branch,
        pose_branch=pose_branch,
        cross_view_cfg=cross_view_cfg,
    )
    model = model.to(device)

    # Print summary
    def _count(module):
        return sum(p.numel() for p in module.parameters()) / 1e6

    total = _count(model)
    print(f"RayNet v5 (Quad-M1, eye-crop gaze) created:")
    print(f"  SharedStem:      {_count(shared_stem):.2f}M")
    print(f"  LandmarkBranch:  {_count(landmark_branch):.2f}M "
          f"(encoder {_count(landmark_branch.encoder):.2f}M + U-Net decoder)")
    print(f"  GazeBranch:      {_count(gaze_branch):.2f}M "
          f"(EyeBackbone {_count(gaze_branch.encoder):.2f}M + CoordAtt + "
          f"fusion + geometric head)")
    print(f"  PoseBranch:      {_count(pose_branch):.2f}M "
          f"(encoder + CoordAtt + head)")
    print(f"  CrossView+Cam:   {_count(model.cross_view_attn) + _count(model.camera_embedding):.2f}M")
    print(f"  Total:           {total:.1f}M params")
    print(f"  Device:          {device}")

    return model
