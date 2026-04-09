"""
RayNet v4.1 — Multi-task gaze estimation, landmark detection, and pose.

  Task A: Iris + pupil landmark heatmaps (14 points via soft-argmax on P2)
  Task B: Optical axis regression (pitch/yaw on P5) with Cross-View Attention
  Aux:    Implicit head pose prediction (MAGE-style, separate backbone)

Main backbone:  RepNeXt-M3 (7.8M params) — landmarks + gaze
Pose backbone:  RepNeXt-M1 (4.8M params) — implicit head pose (gradient-isolated)
Neck:           PANet (YOLOv8-style multi-scale fusion, main backbone only)
Attention:      Coordinate Attention on P2 (landmarks) and P5 (gaze)
Cross-View:     Geometry-conditioned Transformer Encoder on pooled P5
Bridge:         LandmarkGazeBridge cross-attention (P5 attends to P2)
Pose:           PoseEncoder — separate RepNeXt-M1 backbone learns implicit head
                pose from the image. Supervised by auxiliary L1 loss on GT head_R.
                No explicit head pose input needed at inference.

Input:    (3 x 224 x 224) GazeGene face crop
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from RayNet.panet import PANet
from RayNet.coordatt import CoordinateAttention
from RayNet.heads import IrisPupilLandmarkHead, OpticalAxisHead
from backbone.repnext_utils import load_pretrained_repnext

device = "cuda" if torch.cuda.is_available() else "cpu"

# Channel configuration for RepNeXt variants
BACKBONE_CHANNELS = {
    'repnext_m0': [40, 80, 160, 320],
    'repnext_m1': [48, 96, 192, 384],
    'repnext_m2': [56, 112, 224, 448],
    'repnext_m3': [64, 128, 256, 512],
    'repnext_m4': [64, 128, 256, 512],
    'repnext_m5': [80, 160, 320, 640],
}


class CrossViewAttention(nn.Module):
    """
    Pre-norm Transformer Encoder for cross-view gaze feature fusion.

    Operates on pooled P5 feature vectors (B, d_model) reshaped into
    (G, V, d_model) groups. Self-attention across V views allows the
    model to learn multi-view consistency through attention, not just
    post-hoc loss penalties.

    Single-view (n_views=1) bypasses the encoder (identity).

    ~1.05M params with default config (2 layers, d_model=256, d_ff=512).
    AMP-safe, torch.compile compatible.
    """

    def __init__(self, d_model=256, n_heads=4, d_ff=512, dropout=0.1,
                 n_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

    def forward(self, x, n_views, cam_embed=None):
        """
        Args:
            x: (B, d_model) pooled gaze features
            n_views: number of camera views per group
            cam_embed: (B, d_model) camera geometry embedding, or None

        Returns:
            (B, d_model) cross-view enhanced features
        """
        if n_views <= 1:
            return x  # single-view bypass

        # Inject camera pose into feature space (additive fusion)
        if cam_embed is not None:
            x = x + cam_embed

        B, D = x.shape
        G = B // n_views
        x = x.view(G, n_views, D)       # (G, V, D)
        x = self.encoder(x)             # self-attention across views
        return x.view(G * n_views, D)   # (B, D)


class CameraEmbedding(nn.Module):
    """
    Encode camera extrinsics into a d_model-dim vector.

    Input: R_cam (3×3=9) + T_cam (3) = 12 dims.

    Camera extrinsics tell the transformer WHERE the camera is.
    Head pose information is learned implicitly by PoseEncoder
    (MAGE-style), avoiding the need for explicit head pose at inference.

    ~0.02M params with d_model=256.
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )

    def forward(self, R_cam, T_cam):
        """
        Args:
            R_cam: (B, 3, 3) camera extrinsic rotation
            T_cam: (B, 3) camera extrinsic translation

        Returns:
            (B, d_model) camera embedding
        """
        x = torch.cat([R_cam.flatten(1), T_cam], dim=-1)  # (B, 12)
        return self.mlp(x)


class PoseEncoder(nn.Module):
    """
    Implicit head pose encoder with separate backbone (MAGE-inspired).

    Uses an independent lightweight RepNeXt backbone to extract head pose
    features directly from the input image. Gradient-isolated from the
    main backbone — pose gradients don't interfere with landmark/gaze
    feature learning, avoiding the adversarial optimization seen when
    all tasks share one backbone.

    During training, an auxiliary head predicts GT head_R (L1 loss).
    At inference, the learned features capture head pose without any
    explicit input — no external head pose estimator needed.

    Architecture:
        Image → RepNeXt-M1 backbone → GlobalAvgPool → Linear(384→d_model)
                                                     → pose_head(d_model→9)

    ~4.8M + 0.1M params (backbone + projection + aux head).
    """

    def __init__(self, pose_backbone, pose_feat_dim, d_model=256):
        """
        Args:
            pose_backbone: RepNeXt model (e.g., M1) used as pose feature extractor
            pose_feat_dim: output channel dim of the backbone's last stage
            d_model: output dimension, must match PANet out_channels
        """
        super().__init__()
        self.backbone = pose_backbone
        # CoordinateAttention on last stage: captures directional spatial
        # cues (face asymmetry → yaw, vertical tilt → pitch)
        self.coord_att = CoordinateAttention(pose_feat_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(pose_feat_dim, d_model)
        # Auxiliary head: predict head rotation matrix for supervision
        self.pose_head = nn.Linear(d_model, 9)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) input image (same image as main backbone)

        Returns:
            pose_feat: (B, d_model) implicit head pose embedding
            pred_head_R: (B, 9) predicted head rotation (for aux loss)
        """
        # Run through pose backbone with gradient checkpointing
        c0 = checkpoint(self.backbone.stem, x, use_reentrant=False)
        c1 = checkpoint(self.backbone.stages[0], c0, use_reentrant=False)
        c2 = checkpoint(self.backbone.stages[1], c1, use_reentrant=False)
        c3 = checkpoint(self.backbone.stages[2], c2, use_reentrant=False)
        c4 = checkpoint(self.backbone.stages[3], c3, use_reentrant=False)

        # Coordinate Attention → directional spatial weighting for pose
        c4 = self.coord_att(c4)                      # (B, pose_feat_dim, 7, 7)

        # Global average pool → project to d_model
        pooled = self.pool(c4).flatten(1)            # (B, pose_feat_dim)
        pose_feat = self.proj(pooled)                # (B, d_model)
        pred_head_R = self.pose_head(pose_feat)      # (B, 9)
        return pose_feat, pred_head_R


class LandmarkGazeBridge(nn.Module):
    """
    Cross-attention bridge: P5 gaze features (query) attend to P2
    landmark features (key/value).

    Ties the landmark and gaze tasks in a shared latent space so the
    gaze head can leverage spatial landmark geometry.

    P2 is spatially large (56×56 at 224 input), so it is downsampled
    to 7×7 before attention to keep compute manageable.

    ~0.4M params with d_model=256, 4 heads.
    """

    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.downsample = nn.AdaptiveAvgPool2d(7)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

    def forward(self, p5_pooled, p2_feat):
        """
        Args:
            p5_pooled: (B, D) pooled gaze features
            p2_feat: (B, D, H, W) landmark feature map (CoordAtt-enhanced P2)

        Returns:
            (B, D) gaze features enriched with landmark spatial context
        """
        # Downsample P2 and reshape to sequence
        p2_down = self.downsample(p2_feat)            # (B, D, 7, 7)
        B, D, H, W = p2_down.shape
        kv = p2_down.flatten(2).permute(0, 2, 1)     # (B, 49, D)

        # Query: pooled gaze feature as single-token sequence
        q = p5_pooled.unsqueeze(1)                    # (B, 1, D)

        # Pre-norm cross-attention with residual
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)  # (B, 1, D)

        return p5_pooled + out.squeeze(1)             # (B, D)


class RayNet(nn.Module):
    """
    RayNet v4.1: Multi-task gaze estimation, landmark detection, and pose.

    Architecture (two independent backbones):
        Image → RepNeXt-M3 → PANet → {CoordAtt(P2) → LandmarkHead,
                                       CoordAtt(P5) → Pool → LmGazeBridge(P2) →
                                       CamEmbed + PoseFeat + CrossViewAttn → GazeFC}
        Image → RepNeXt-M1 → GlobalPool → PoseEncoder → pose_feat + pred_head_R

    The pose backbone is gradient-isolated from the main backbone to prevent
    adversarial optimization between landmark, gaze, and pose objectives.
    """

    def __init__(self, backbone, in_channels_list, panet_out_channels=256,
                 n_landmarks=14, cross_view_cfg=None,
                 pose_backbone=None, pose_backbone_channels=None):
        super().__init__()

        self.backbone = backbone
        self.panet = PANet(channels_list=in_channels_list,
                           out_channels=panet_out_channels)

        # Coordinate Attention on P2 (landmarks) and P5 (gaze)
        self.coord_att_p2 = CoordinateAttention(panet_out_channels)
        self.coord_att_p5 = CoordinateAttention(panet_out_channels)

        # v4: Landmark-Gaze bridge (P5 attends to P2)
        self.landmark_gaze_bridge = LandmarkGazeBridge(
            d_model=panet_out_channels)

        # v4: Camera extrinsics embedding for cross-view attention
        self.camera_embedding = CameraEmbedding(d_model=panet_out_channels)

        # v4.1: Separate pose backbone (MAGE-style, gradient-isolated)
        # Uses lightweight RepNeXt to learn implicit head pose from image
        if pose_backbone is not None:
            pose_feat_dim = pose_backbone_channels[-1]  # last stage channels
            self.pose_encoder = PoseEncoder(
                pose_backbone, pose_feat_dim, d_model=panet_out_channels)
        else:
            self.pose_encoder = None

        # Cross-View Attention on pooled P5 features
        cv_cfg = cross_view_cfg or {}
        cv_cfg.setdefault('d_model', panet_out_channels)
        self.cross_view_attn = CrossViewAttention(**cv_cfg)

        # Task heads
        self.landmark_head = IrisPupilLandmarkHead(
            in_ch=panet_out_channels, n_landmarks=n_landmarks)
        self.gaze_head = OpticalAxisHead(
            in_ch=panet_out_channels, hidden_dim=128)

    def forward(self, x, n_views=1, R_cam=None, T_cam=None):
        """
        Args:
            x: (B, 3, 224, 224) face crop (or any resolution; feature maps scale accordingly)
            n_views: number of camera views per group (1=single-view, 9=multi-view)
            R_cam: (B, 3, 3) camera extrinsic rotation matrices, or None
            T_cam: (B, 3) camera extrinsic translation vectors, or None

        Returns:
            dict with:
                'landmark_coords': (B, 14, 2) pixel coords in P2 feature space
                'landmark_heatmaps': (B, 14, H, W) raw logit heatmaps
                'gaze_vector': (B, 3) optical axis unit vector (CCS)
                'gaze_angles': (B, 2) pitch/yaw in radians
                'pred_head_R': (B, 9) predicted head rotation (for aux loss)
        """
        # Backbone: 4-stage feature extraction
        c0 = checkpoint(self.backbone.stem, x, use_reentrant=False)
        c1 = checkpoint(self.backbone.stages[0], c0, use_reentrant=False)
        c2 = checkpoint(self.backbone.stages[1], c1, use_reentrant=False)
        c3 = checkpoint(self.backbone.stages[2], c2, use_reentrant=False)
        c4 = checkpoint(self.backbone.stages[3], c3, use_reentrant=False)

        features = [c1, c2, c3, c4]

        # PANet multi-scale fusion
        panet_out = self.panet(features)  # [P2, P3, P4, P5]
        p2 = panet_out[0]   # (B, 256, 56, 56) stride=4 for 224 input
        p5 = panet_out[-1]  # (B, 256, 7, 7)  stride=32

        # Task A: Landmark detection on P2 (no cross-view)
        p2_att = self.coord_att_p2(p2)
        landmark_coords, landmark_heatmaps = self.landmark_head(p2_att)

        # Task B: Optical axis regression on P5 with cross-view attention
        p5_att = self.coord_att_p5(p5)
        pooled = self.gaze_head.pool_features(p5_att)          # (B, 256)

        # v4: Landmark-Gaze bridge — gaze attends to landmark features
        pooled = self.landmark_gaze_bridge(pooled, p2_att)     # (B, 256)

        # v4.1: Implicit pose from separate backbone (gradient-isolated)
        pred_head_R = None
        pose_feat = None
        if self.pose_encoder is not None:
            pose_feat, pred_head_R = self.pose_encoder(x)      # (B, 256), (B, 9)

        # v4: Camera extrinsics embedding for cross-view attention
        cam_embed = None
        if R_cam is not None and T_cam is not None:
            cam_embed = self.camera_embedding(R_cam, T_cam)    # (B, 256)
            if pose_feat is not None:
                cam_embed = cam_embed + pose_feat  # fuse camera + implicit pose

        pooled = self.cross_view_attn(pooled, n_views, cam_embed)
        gaze_vector, gaze_angles = self.gaze_head.predict_from_pooled(pooled)

        return {
            'landmark_coords': landmark_coords,
            'landmark_heatmaps': landmark_heatmaps,
            'gaze_vector': gaze_vector,
            'gaze_angles': gaze_angles,
            'pred_head_R': pred_head_R,
        }


def create_raynet(core_backbone_name="repnext_m3", core_backbone_weight_path=None,pose_backbone_weight_path=None, n_landmarks=14,
                  cross_view_cfg=None, pose_backbone_name="repnext_m1"):
    """
    Factory function to create RayNet v4.1.

    Args:
        backbone_name: RepNeXt variant for main backbone (landmarks + gaze)
        weight_path: path to pretrained main backbone weights (JIT format)
        n_landmarks: number of landmarks (default 14: 10 iris + 4 pupil)
        cross_view_cfg: dict of kwargs for CrossViewAttention
            (d_model, n_heads, d_ff, dropout, n_layers). None = defaults.
        pose_backbone_name: RepNeXt variant for pose encoder backbone.
            Lightweight variant recommended (M0/M1/M2). None = no pose encoder.

    Returns:
        RayNet model on device
    """

    # Core backbone: RepNeXt-M3 distilled not fused (training mode) repnext_m3_distill_300e.pth
    # backbone = create_repnext(model_name=backbone_name, pretrained=True,weight_path=core_backbone_weight_path)
    core_backbone = load_pretrained_repnext(core_backbone_name,core_backbone_weight_path)
    core_backbone = core_backbone.to(device)
    in_channels_list = BACKBONE_CHANNELS[core_backbone_name]

    # Head Pose backbone RepNeXt-M1 distilled and not fused (training mode) repnext_m1_distill_300e.pth
    pose_backbone = load_pretrained_repnext(pose_backbone_name,pose_backbone_weight_path)
    pose_backbone = pose_backbone.to(device)
    pose_channels = BACKBONE_CHANNELS[pose_backbone_name]

    model = RayNet(core_backbone, in_channels_list, n_landmarks=n_landmarks,
                   cross_view_cfg=cross_view_cfg,
                   pose_backbone=pose_backbone,
                   pose_backbone_channels=pose_channels)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    cv_params = sum(p.numel() for p in model.cross_view_attn.parameters()) / 1e6
    bridge_params = sum(p.numel() for p in model.landmark_gaze_bridge.parameters()) / 1e6
    cam_params = sum(p.numel() for p in model.camera_embedding.parameters()) / 1e6

    print(f"RayNet v4.1 created:")
    print(f"  Main backbone: {core_backbone_name} ({BACKBONE_CHANNELS[core_backbone_name]})")
    print(f"  Landmarks: {n_landmarks}")
    print(f"  CrossViewAttention: {cv_params:.2f}M params")
    print(f"  LandmarkGazeBridge: {bridge_params:.2f}M params")
    print(f"  CameraEmbedding: {cam_params:.2f}M params")
    if model.pose_encoder is not None:
        pose_params = sum(p.numel() for p in model.pose_encoder.parameters()) / 1e6
        print(f"  PoseEncoder: {pose_backbone_name} → {pose_params:.2f}M params")
    print(f"  Total params: {total_params:.1f}M")
    print(f"  Trainable params: {trainable_params:.1f}M")
    print(f"  Device: {device}")

    return model
