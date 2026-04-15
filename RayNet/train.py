"""
RayNet v5 Training Script (Triple-M1 architecture).

Supports both v4.1 and v5 architectures via --architecture flag.

v5 Staged training strategy:

  Stage 1 — Landmark + Pose baseline (no gaze, no bridges):
    Phase 1 (epochs 1-8):   λ_lm=1.0, λ_pose=0.5, λ_trans=0.5
    Phase 2 (epochs 9-15):  λ_lm=1.0, λ_pose=1.0, λ_trans=1.0

  Stage 2 — Add gaze with GazeGene 3D eyeball losses (bridges zero-init):
    Phase 1 (epochs 1-5):   λ_gaze=0.1, λ_eyeball=0.1, λ_pupil=0.1
    Phase 2 (epochs 6-15):  λ_gaze=0.5, λ_eyeball=0.3, λ_pupil=0.3, λ_ray=0.1
    Phase 3 (epochs 16-25): λ_gaze=1.0, λ_eyeball=0.5, λ_pupil=0.5, λ_geom=0.2, λ_ray=0.3

  Stage 3 — Full pipeline with bridges + MAGE box encoder:
    Phase 1 (epochs 1-5):   Bridge warmup, geometric angular active
    Phase 2 (epochs 6-15):  Full losses + multi-view consistency
    Phase 3 (epochs 16-25): Gaze-focused fine-tuning

  Gradient clipping (max_norm) varies by phase:
    Phase 1: max_norm=5.0 (aggressive, allows large multi-task gradients)
    Phase 2+: max_norm=2.0 (conservative, prevents gaze/pose interference)

Usage:
    # v5 Stage 1: Landmark + Pose baseline
    python -m RayNet.train --architecture v5 --stage 1 --profile t4 ...

    # v5 Stage 2: Add gaze with GazeGene losses
    python -m RayNet.train --architecture v5 --stage 2 --profile t4 ...

    # v5 Stage 3: Full pipeline
    python -m RayNet.train --architecture v5 --stage 3 --profile t4 ...

    # Legacy v4.1 (unchanged)
    python -m RayNet.train --architecture v4 --stage 3 --profile t4 ...
"""

import argparse
import os
import csv
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
import numpy as np
from datetime import datetime

from RayNet.raynet import create_raynet
from RayNet.dataset import create_dataloaders
from RayNet.losses import total_loss, total_loss_v5
from RayNet.multiview_loss import multiview_consistency_loss

log = logging.getLogger(__name__)


# ============== Hardware Profiles ==============

HARDWARE_PROFILES = {
    # ---- CPU / low-end GPU (testing, debugging) ----
    # v4: 224×224 input. Batch sizes ~4× larger than v3 (448×448).
    'default': {
        'batch_size': 504,          # 56 mv_groups × 9 views
        'mv_groups': 56,
        'num_workers': 4,
        'pin_memory': True,
        'amp': False,
        'amp_dtype': 'float32',
        'grad_accum_steps': 1,
        'compile_model': False,
        'tf32': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
    },
    # ---- NVIDIA T4  (16 GB, Colab free / GCP n1-standard) ----
    # FP16 is critical — T4 has weak FP32 but decent FP16 (65 TFLOPS).
    # 16 GB VRAM comfortable at 224×224: ~16 mv_groups (144 samples).
    't4': {
        'batch_size': 144,          # 16 mv_groups × 9 views
        'mv_groups': 16,
        'num_workers': 2,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'float16',
        'grad_accum_steps': 2,      # effective batch = 288
        'compile_model': False,     # T4 doesn't benefit much from compile
        'tf32': False,              # T4 doesn't support TF32
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
    # ---- NVIDIA L4  (24 GB, GCP g2-standard) ----
    # Ada Lovelace arch: good FP16/BF16 (121 TFLOPS FP16).
    # 24 GB comfortable at 224×224: ~32 mv_groups.
    'l4': {
        'batch_size': 288,          # 32 mv_groups × 9 views
        'mv_groups': 32,
        'num_workers': 4,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'bfloat16',    # BF16: same range as FP32 → no exp/log overflow
        'grad_accum_steps': 1,      # effective batch = 288
        'compile_model': False,     # disabled: interacts badly with grad checkpointing
        'tf32': True,               # Ada supports TF32
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
    # ---- NVIDIA A10G  (24 GB, AWS g5) ----
    # Ampere arch, similar to L4 in VRAM but different compute profile.
    'a10g': {
        'batch_size': 288,          # 32 mv_groups × 9 views
        'mv_groups': 32,
        'num_workers': 4,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'bfloat16',    # BF16: same range as FP32 → no exp/log overflow
        'grad_accum_steps': 1,      # effective batch = 288
        'compile_model': False,     # disabled: interacts badly with grad checkpointing
        'tf32': True,
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
    # ---- NVIDIA V100  (16 GB / 32 GB, GCP / AWS p3) ----
    # Volta: no TF32, no torch.compile benefit. Good FP16 via Tensor Cores.
    'v100': {
        'batch_size': 144,          # 16 mv_groups × 9 views
        'mv_groups': 16,
        'num_workers': 4,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'float16',
        'grad_accum_steps': 2,      # effective batch = 288
        'compile_model': False,     # Volta doesn't benefit from compile
        'tf32': False,              # Volta doesn't support TF32
        'prefetch_factor': 2,
        'persistent_workers': True,
    },
    # ---- NVIDIA A100  (40 GB / 80 GB, GCP a2, Colab Pro+) ----
    # Ampere flagship: TF32, BF16, huge memory bandwidth (2 TB/s).
    # At 224×224: fits large batches comfortably.
    'a100': {
        'batch_size': 1152,         # 128 mv_groups × 9 views
        'mv_groups': 128,
        'num_workers': 8,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'bfloat16',
        'grad_accum_steps': 1,      # effective batch = 1152
        'compile_model': True,
        'tf32': True,
        'prefetch_factor': 10,
        'persistent_workers': True,
    },
    # ---- NVIDIA H100  (80 GB, GCP a3, Lambda Labs) ----
    # Hopper: FP8 support, Transformer Engine, 3.4 TB/s bandwidth.
    # BF16 preferred (less overflow risk than FP16 at similar speed).
    'h100': {
        'batch_size': 2304,         # 256 mv_groups × 9 views
        'mv_groups': 256,
        'num_workers': 8,
        'pin_memory': True,
        'amp': True,
        'amp_dtype': 'bfloat16',
        'grad_accum_steps': 1,      # effective batch = 2304
        'compile_model': True,
        'tf32': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
    },
}


# ============== Staged Training Configuration ==============
#
# Stage 1: Landmark + Pose baseline (no gaze, no bridge, no cross-view)
#   Purpose: Validate both backbones learn useful features independently.
#   Expect:  Landmark px error < 5px by epoch 10, pose geodesic < 10° by epoch 15.
#   Anomaly: Pose loss stuck > 30° = backbone not learning face geometry.
#            Landmark loss diverging = PANet/CoordAtt issue (check BN float32).
#
# Stage 2: Add gaze, no bridge
#   Purpose: Test gaze learning without crop-poisoned LandmarkGazeBridge.
#   Expect:  Gaze angular error improving on BOTH train and val (no divergence).
#            Val gaze < 20° by epoch 15 = gaze learns from appearance alone.
#   Anomaly: Train gaze improving but val worsening = still adversarial.
#            If this happens WITHOUT bridge, problem is in shared backbone,
#            not the bridge — would need backbone freezing strategy.
#
# Stage 3: Full pipeline with bridge
#   Purpose: Test if bridge helps or hurts. Compare val gaze to Stage 2.
#   Expect:  If bridge helps: val gaze improves over Stage 2 baseline.
#            If bridge hurts: val gaze worse than Stage 2 → remove bridge.
#   Anomaly: Bridge loss oscillating wildly = crop augmentation poisoning.

#
# -----------------------------------------------------------------------------
# IMPORTANT — meaning of the `multiview` flag in phase configs below:
#
#   `multiview` does NOT control whether CrossViewAttention runs. It ONLY
#   gates the auxiliary `multiview_consistency_loss` (which adds
#   `lam_reproj` gaze-consistency and `lam_mask` shape terms).
#
#   CrossViewAttention itself is *always active during training* when
#   --mv_groups > 1, regardless of the phase's `multiview` setting. This is
#   because the MDS dataloader is constructed once before the phase loop,
#   producing 9-grouped multi-view batches, and `active_n_views` is
#   hardcoded to 9 at train.py:active_n_views (only suppressed by the
#   --no_multiview CLI ablation flag).
#
#   Validation uses n_views=1, so CrossViewAttention, camera embeddings,
#   and the PoseEncoder features ALL bypass during validation. This creates
#   a deliberate train/val asymmetry: training sees fused 9-view geometry,
#   val sees single-view — val metrics are therefore a strict lower bound
#   on what the model is actually capable of at inference with multi-view.
#
#   If you truly want a "no multi-view fusion" phase, pass --no_multiview
#   on the CLI (not the `multiview: False` config flag).
# -----------------------------------------------------------------------------

STAGE_CONFIGS = {
    # ---- Stage 1: Landmark + Pose baseline ----
    1: {
        1: {
            'epochs': (1, 10),
            'lam_lm': 1.0,
            'lam_gaze': 0.0,
            'lam_reproj': 0.0,
            'lam_mask': 0.0,
            'lam_ray': 0.0,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 1e-3,
            'sigma': 2.0,
            'multiview': False,
            'no_bridge': True,
            'description': 'S1P1: Landmark warmup + pose learning',
        },
        2: {
            'epochs': (11, 20),
            'lam_lm': 1.0,
            'lam_gaze': 0.0,
            'lam_reproj': 0.0,
            'lam_mask': 0.0,
            'lam_ray': 0.0,
            'lam_pose': 1.0,
            'lam_trans': 1.0,
            'lr': 3e-4,
            'sigma': 1.5,
            'multiview': False,
            'no_bridge': True,
            'description': 'S1P2: Landmark refinement + pose emphasis',
        },
    },

    # ---- Stage 2: Add gaze, NO bridge ----
    # Revised to address warmstart shock from Stage 1:
    #  - S2P1 LR is now 3e-4 (not 1e-3) to avoid disrupting converged weights.
    #  - Sigma is monotonically tightened 1.5 → 1.3 → 1.0 (no 1.5 → 2.0 reset).
    #  - S2P3 keeps lam_lm=1.0 instead of halving it (don't weaken landmark
    #    supervision at the same time the target sharpens).
    2: {
        1: {
            'epochs': (1, 5),
            'lam_lm': 1.0,
            'lam_gaze': 0.1,
            'lam_reproj': 0.0,
            'lam_mask': 0.0,
            'lam_ray': 0.0,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 3e-4,                 # continuation LR, not cold-start 1e-3
            'sigma': 1.5,               # continue from Stage-1 endpoint
            'multiview': False,         # aux MV losses off (CVA still active)
            'no_bridge': True,
            'description': 'S2P1: Gaze warmup from Stage-1 weights (no bridge)',
        },
        2: {
            'epochs': (6, 15),
            'lam_lm': 1.0,
            'lam_gaze': 0.5,
            'lam_reproj': 0.05,
            'lam_mask': 0.02,
            'lam_ray': 0.1,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 3e-4,                 # smoother continuation from P1
            'sigma': 1.3,               # monotonic tightening
            'multiview': True,          # aux MV consistency losses on
            'no_bridge': True,
            'description': 'S2P2: Balanced + ray + MV aux losses (no bridge)',
        },
        3: {
            'epochs': (16, 25),
            'lam_lm': 1.0,              # keep landmark strong (was 0.5)
            'lam_gaze': 1.0,
            'lam_reproj': 0.1,
            'lam_mask': 0.05,
            'lam_ray': 0.3,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 1e-4,
            'sigma': 1.0,               # tightest landmark target
            'multiview': True,
            'no_bridge': True,
            'description': 'S2P3: Gaze fine-tuning, landmark weight preserved',
        },
    },

    # ---- Stage 3: Full pipeline WITH bridge ----
    3: {
        1: {
            'epochs': (1, 5),
            'lam_lm': 1.0,
            'lam_gaze': 0.3,           # was 0.1 — too weak, couldn't protect gaze
            'lam_reproj': 0.0,
            'lam_mask': 0.0,
            'lam_ray': 0.0,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 3e-4,                # was 1e-3 — too aggressive for warmed weights
            'sigma': 2.0,
            'multiview': False,
            'no_bridge': False,
            'description': 'S3P1: Bridge warmup (zero-init out_proj, gentle LR)',
        },
        2: {
            'epochs': (6, 15),
            'lam_lm': 1.0,
            'lam_gaze': 0.5,
            'lam_reproj': 0.05,
            'lam_mask': 0.02,
            'lam_ray': 0.1,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 3e-4,                # was 5e-4 — aligned with S2 proven LR
            'sigma': 1.5,
            'multiview': True,
            'no_bridge': False,
            'description': 'S3P2: Balanced + multi-view + ray + bridge',
        },
        3: {
            'epochs': (16, 25),
            'lam_lm': 0.5,
            'lam_gaze': 1.0,
            'lam_reproj': 0.1,
            'lam_mask': 0.05,
            'lam_ray': 0.3,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 1e-4,
            'sigma': 1.0,
            'multiview': True,
            'no_bridge': False,
            'description': 'S3P3: Gaze-focused fine-tuning (with bridge)',
        },
    },
}

# ============== V5 Triple-M1 Staged Training Configuration ==============
#
# V5 has three branches (landmark, gaze, pose) with shared stem.
# GazeGene 3D eyeball structure losses (eyeball L1, pupil L1, geometric angular)
# are added progressively. MAGE-style BoxEncoder fuses with pose in Stage 3.
#
# Key differences from v4.1 config:
#   - No separate `no_bridge` flag — V5 bridges are always present (zero-init)
#     controlled by `use_landmark_bridge` and `use_pose_bridge`
#   - GazeGene losses: lam_eyeball, lam_pupil, lam_geom_angular
#   - lam_reproj / lam_mask (multi-view aux) still supported
#

STAGE_CONFIGS_V5 = {
    # ---- Stage 1: Landmark + Pose baseline (no gaze) ----
    # Purpose: Validate shared stem + branch encoders learn useful features.
    # Expect: Landmark px error < 4px by epoch 8, pose geodesic < 10° by epoch 12.
    1: {
        1: {
            'epochs': (1, 8),
            'lam_lm': 1.0,
            'lam_gaze': 0.0,
            'lam_eyeball': 0.0,
            'lam_pupil': 0.0,
            'lam_geom_angular': 0.0,
            'lam_ray': 0.0,
            'lam_reproj': 0.0,
            'lam_mask': 0.0,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 1e-3,
            'sigma': 2.0,
            'multiview': False,
            'use_landmark_bridge': False,
            'use_pose_bridge': False,
            'description': 'V5-S1P1: Landmark + pose warmup (shared stem + branch encoders)',
        },
        2: {
            'epochs': (9, 15),
            'lam_lm': 1.0,
            'lam_gaze': 0.0,
            'lam_eyeball': 0.0,
            'lam_pupil': 0.0,
            'lam_geom_angular': 0.0,
            'lam_ray': 0.0,
            'lam_reproj': 0.0,
            'lam_mask': 0.0,
            'lam_pose': 1.0,
            'lam_trans': 1.0,
            'lr': 3e-4,
            'sigma': 1.5,
            'multiview': False,
            'use_landmark_bridge': False,
            'use_pose_bridge': False,
            'description': 'V5-S1P2: Landmark refinement + pose emphasis',
        },
    },

    # ---- Stage 2: Add gaze + GazeGene 3D eyeball losses (bridges zero-init) ----
    # Purpose: Test gaze learning with explicit 3D eyeball structure.
    # Bridges are present but zero-init — they start as identity (skip).
    # Expect: Angular error improving steadily; eyeball/pupil L1 converging.
    2: {
        1: {
            'epochs': (1, 5),
            'lam_lm': 1.0,
            'lam_gaze': 0.1,
            'lam_eyeball': 0.1,
            'lam_pupil': 0.1,
            'lam_geom_angular': 0.0,  # too early — geometry not converged
            'lam_ray': 0.0,
            'lam_reproj': 0.0,
            'lam_mask': 0.0,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 3e-4,
            'sigma': 1.5,
            'multiview': False,
            'use_landmark_bridge': False,
            'use_pose_bridge': False,
            'description': 'V5-S2P1: Gaze warmup + 3D eyeball structure learning',
        },
        2: {
            'epochs': (6, 15),
            'lam_lm': 1.0,
            'lam_gaze': 0.5,
            'lam_eyeball': 0.3,
            'lam_pupil': 0.3,
            'lam_geom_angular': 0.0,  # still off — let eyeball/pupil stabilize
            'lam_ray': 0.1,
            'lam_reproj': 0.05,
            'lam_mask': 0.02,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 3e-4,
            'sigma': 1.3,
            'multiview': True,
            'use_landmark_bridge': False,
            'use_pose_bridge': False,
            'description': 'V5-S2P2: Balanced gaze + eyeball/pupil + ray + MV',
        },
        3: {
            'epochs': (16, 25),
            'lam_lm': 1.0,
            'lam_gaze': 1.0,
            'lam_eyeball': 0.5,
            'lam_pupil': 0.5,
            'lam_geom_angular': 0.2,  # NOW turn on — geometry should be close
            'lam_ray': 0.3,
            'lam_reproj': 0.1,
            'lam_mask': 0.05,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 1e-4,
            'sigma': 1.0,
            'multiview': True,
            'use_landmark_bridge': False,
            'use_pose_bridge': False,
            'description': 'V5-S2P3: Gaze fine-tuning + geometric angular loss',
        },
    },

    # ---- Stage 3: Full pipeline WITH bridges + MAGE box encoder ----
    # Purpose: Activate inter-branch bridges and BoxEncoder fusion.
    # Bridges are zero-init from checkpoint (never trained in S1/S2) so they
    # start as identity and learn gradually.
    3: {
        1: {
            'epochs': (1, 5),
            'lam_lm': 1.0,
            'lam_gaze': 0.3,
            'lam_eyeball': 0.3,
            'lam_pupil': 0.3,
            'lam_geom_angular': 0.1,
            'lam_ray': 0.0,
            'lam_reproj': 0.0,
            'lam_mask': 0.0,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 3e-4,
            'sigma': 2.0,
            'multiview': False,
            'use_landmark_bridge': True,
            'use_pose_bridge': True,
            'description': 'V5-S3P1: Bridge warmup (zero-init, gentle LR)',
        },
        2: {
            'epochs': (6, 15),
            'lam_lm': 1.0,
            'lam_gaze': 0.5,
            'lam_eyeball': 0.5,
            'lam_pupil': 0.5,
            'lam_geom_angular': 0.2,
            'lam_ray': 0.1,
            'lam_reproj': 0.05,
            'lam_mask': 0.02,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 3e-4,
            'sigma': 1.5,
            'multiview': True,
            'use_landmark_bridge': True,
            'use_pose_bridge': True,
            'description': 'V5-S3P2: Full losses + bridges + multi-view',
        },
        3: {
            'epochs': (16, 25),
            'lam_lm': 0.5,
            'lam_gaze': 1.0,
            'lam_eyeball': 0.5,
            'lam_pupil': 0.5,
            'lam_geom_angular': 0.3,
            'lam_ray': 0.3,
            'lam_reproj': 0.1,
            'lam_mask': 0.05,
            'lam_pose': 0.5,
            'lam_trans': 0.5,
            'lr': 1e-4,
            'sigma': 1.0,
            'multiview': True,
            'use_landmark_bridge': True,
            'use_pose_bridge': True,
            'description': 'V5-S3P3: Gaze-focused fine-tuning (full bridges)',
        },
    },
}


# Default: Stage 3 (full pipeline) for backward compatibility
PHASE_CONFIG = STAGE_CONFIGS[3]


def get_phase(epoch):
    """Return the training phase for a given epoch."""
    for phase, cfg in PHASE_CONFIG.items():
        start, end = cfg['epochs']
        if start <= epoch <= end:
            return phase
    return 3


def get_phase_config(epoch):
    """Get loss weights and hyperparams for the current epoch (returns a copy)."""
    phase = get_phase(epoch)
    return dict(PHASE_CONFIG[phase])


AMP_DTYPE_MAP = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
}


def apply_hardware_profile(args):
    """Apply hardware profile settings, allowing CLI overrides."""
    hw = HARDWARE_PROFILES[args.profile].copy()

    # CLI flags override profile defaults
    if args.batch_size is not None:
        hw['batch_size'] = args.batch_size
    if args.mv_groups is not None:
        hw['mv_groups'] = args.mv_groups
    if args.num_workers is not None:
        hw['num_workers'] = args.num_workers
    if args.grad_accum_steps is not None:
        hw['grad_accum_steps'] = args.grad_accum_steps
    if args.no_compile:
        hw['compile_model'] = False

    return hw


def setup_hardware(hw, device):
    """Configure hardware-specific optimizations."""
    if hw['tf32'] and device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        print("  TF32 enabled for matmul and cuDNN")

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")


# ============== Training Loop ==============

def train_one_epoch(model, train_loader, optimizer, device, epoch, cfg,
                    scaler=None, grad_accum_steps=1, amp_enabled=False,
                    amp_dtype=torch.float16, batch_csv_writer=None,
                    n_views=1, is_v5=False):
    """Run one training epoch with AMP and gradient accumulation support."""
    model.train()
    use_multiview = cfg.get('multiview', False)
    # v4.1: no_bridge flag; v5: always has bridges, controlled by use_*_bridge
    use_bridge = not cfg.get('no_bridge', False)
    use_landmark_bridge = cfg.get('use_landmark_bridge', True)
    use_pose_bridge = cfg.get('use_pose_bridge', True)
    if not amp_enabled:
        amp_dtype = torch.float32

    running_losses = {
        'total': 0.0, 'landmark': 0.0, 'angular': 0.0, 'angular_deg': 0.0,
        'reproj': 0.0, 'mask': 0.0, 'ray_target': 0.0, 'pose': 0.0,
        'translation': 0.0,
        # V5-specific
        'eyeball_center': 0.0, 'pupil_center': 0.0, 'geom_angular': 0.0,
    }
    n_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        # We don't process the samples that are skipped when samples_per_subject is used

        # To prevent accumulation on missing values we use this flag to not process the filtered samples
        has_accumulated = False
        if not batch or len(batch) == 0:
            continue

        # UINT 8 to tensor of floats normalized from 0.0 to 1.0
        images = batch['image'].to(device, non_blocking=True).float().div_(255.0)
        gt_landmarks = batch['landmark_coords'].to(device, non_blocking=True)
        gt_optical_axis = batch['optical_axis'].to(device, non_blocking=True)

        # Camera extrinsics for geometry-conditioned attention
        R_cam = batch['R_cam'].to(device, non_blocking=True)
        T_cam = batch['T_cam'].to(device, non_blocking=True)

        # GT head pose for auxiliary pose loss (not passed to model)
        gt_head_R = batch.get('head_R')
        if gt_head_R is not None:
            gt_head_R = gt_head_R.to(device, non_blocking=True)

        gt_head_t = batch.get('head_t')
        if gt_head_t is not None:
            gt_head_t = gt_head_t.to(device, non_blocking=True)

        mv_components = None
        # Forward pass with AMP autocast
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            if is_v5:
                predictions = model(images, n_views=n_views,
                                    R_cam=R_cam, T_cam=T_cam,
                                    use_landmark_bridge=use_landmark_bridge,
                                    use_pose_bridge=use_pose_bridge)
            else:
                predictions = model(images, n_views=n_views,
                                    R_cam=R_cam, T_cam=T_cam,
                                    use_bridge=use_bridge)

            pred_hm = predictions['landmark_heatmaps']
            pred_coords = predictions['landmark_coords']
            pred_gaze = predictions['gaze_vector']

            feat_H, feat_W = pred_hm.shape[2], pred_hm.shape[3]

            if is_v5:
                # V5: use total_loss_v5 with GazeGene 3D eyeball losses
                gt_eyeball = batch['eyeball_center_3d'].to(device, non_blocking=True)
                gt_pupil = batch.get('pupil_center_3d')
                if gt_pupil is not None:
                    gt_pupil = gt_pupil.to(device, non_blocking=True)

                loss, components = total_loss_v5(
                    pred_hm, pred_coords, pred_gaze,
                    gt_landmarks, gt_optical_axis,
                    feat_H, feat_W,
                    lam_lm=cfg['lam_lm'],
                    lam_gaze=cfg['lam_gaze'],
                    sigma=cfg['sigma'],
                    # GazeGene 3D eyeball structure losses
                    lam_eyeball=cfg.get('lam_eyeball', 0.0),
                    pred_eyeball=predictions.get('eyeball_center'),
                    gt_eyeball=gt_eyeball,
                    lam_pupil=cfg.get('lam_pupil', 0.0),
                    pred_pupil=predictions.get('pupil_center'),
                    gt_pupil=gt_pupil,
                    lam_geom_angular=cfg.get('lam_geom_angular', 0.0),
                    # Ray-to-target
                    lam_ray=cfg.get('lam_ray', 0.0),
                    eyeball_center=gt_eyeball,
                    gaze_target=batch['gaze_target'].to(device, non_blocking=True),
                    gaze_depth=batch['gaze_depth'].to(device, non_blocking=True),
                    # Pose
                    lam_pose=cfg.get('lam_pose', 0.0),
                    pred_pose_6d=predictions.get('pred_pose_6d'),
                    gt_head_R=gt_head_R,
                    lam_trans=cfg.get('lam_trans', 0.0),
                    pred_pose_t=predictions.get('pred_pose_t'),
                    gt_head_t=gt_head_t,
                )
            else:
                loss, components = total_loss(
                    pred_hm, pred_coords, pred_gaze,
                    gt_landmarks, gt_optical_axis,
                    feat_H, feat_W,
                    lam_lm=cfg['lam_lm'],
                    lam_gaze=cfg['lam_gaze'],
                    sigma=cfg['sigma'],
                    lam_ray=cfg.get('lam_ray', 0.0),
                    eyeball_center=batch['eyeball_center_3d'].to(
                        device, non_blocking=True),
                    gaze_target=batch['gaze_target'].to(
                        device, non_blocking=True),
                    gaze_depth=batch['gaze_depth'].to(
                        device, non_blocking=True),
                    lam_pose=cfg.get('lam_pose', 0.0),
                    pred_pose_6d=predictions.get('pred_pose_6d'),
                    gt_head_R=gt_head_R,
                    lam_trans=cfg.get('lam_trans', 0.0),
                    pred_pose_t=predictions.get('pred_pose_t'),
                    gt_head_t=gt_head_t,
                )

            # Multi-view consistency loss (Phase 2+)
            # Ray-based: works with unit gaze vectors, numerically stable.
            if use_multiview:
                mv_loss, mv_components = multiview_consistency_loss(
                    pred_gaze,
                    pred_coords,
                    R_cam,
                    lam_gaze_consist=cfg['lam_reproj'],
                    lam_shape=cfg['lam_mask'],
                )
                if torch.isfinite(mv_loss):
                    mv_weight = min(1.0, epoch / 10.0)  # smooth ramp
                    loss = loss + mv_weight * mv_loss

            # Scale loss by accumulation steps
            loss = loss / grad_accum_steps

        # Backward pass with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # using this flag only when a batch is valid to process
        has_accumulated = True

        # max_norm for multi-task learning scenario like ours requires a value>1
        # therefore we set max_norm=5.0 for more aggressive training strategy
        current_phase_num = get_phase(epoch)
        if current_phase_num >= 2:
            max_norm = 2.0
        else:
            max_norm = 5.0
        # Optimizer step at accumulation boundary
        if (step + 1) % grad_accum_steps == 0:
            if has_accumulated:
                if scaler is not None:
                    # FP16 path: scaler handles inf/nan detection internally
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # FP32 / BF16 path: manually check grad norm and skip step
                    # if non-finite (BF16 has no scaler to catch overflow/nan)
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=max_norm)
                    if torch.isfinite(total_norm):
                        optimizer.step()
                    else:
                        log.warning("Epoch %d batch %d: non-finite grad norm "
                                    "(%s), skipping optimizer step",
                                    epoch, step + 1, total_norm.item())
                optimizer.zero_grad()
                has_accumulated = False
            else:
                optimizer.zero_grad()

        # Accumulate metrics (use unscaled loss); skip nan to prevent poisoning epoch avg
        batch_total = components['total_loss'].item()
        if batch_total != batch_total:  # nan check
            log.warning("Epoch %d batch %d: nan loss detected, skipping metrics", epoch, step + 1)
            continue
        running_losses['total'] += batch_total
        running_losses['landmark'] += components['landmark_loss'].item()
        running_losses['angular'] += components['angular_loss'].item()
        running_losses['angular_deg'] += components['angular_loss_deg'].item()
        if mv_components is not None:
            running_losses['reproj'] += mv_components['gaze_consist_loss'].item()
            running_losses['mask'] += mv_components['shape_loss'].item()
        if 'ray_target_loss' in components:
            running_losses['ray_target'] += components['ray_target_loss'].item()
        if 'pose_loss' in components:
            running_losses['pose'] += components['pose_loss'].item()
        if 'translation_loss' in components:
            running_losses['translation'] += components['translation_loss'].item()
        n_batches += 1

        # Per-batch CSV logging (high granularity)
        if batch_csv_writer is not None:
            ray_val = components.get('ray_target_loss',
                                     torch.tensor(0.0)).item()
            pose_val = components.get('pose_loss',
                                      torch.tensor(0.0)).item()
            trans_val = components.get('translation_loss',
                                       torch.tensor(0.0)).item()
            batch_csv_writer.writerow([
                epoch, step + 1,
                f"{components['total_loss'].item():.6f}",
                f"{components['landmark_loss'].item():.6f}",
                f"{components['angular_loss_deg'].item():.4f}",
                f"{running_losses['reproj'] / max(n_batches, 1):.6f}",
                f"{running_losses['mask'] / max(n_batches, 1):.6f}",
                f"{ray_val:.6f}",
                f"{pose_val:.6f}",
                f"{trans_val:.6f}",
                f"{optimizer.param_groups[0]['lr']:.8f}",
            ])

        # Progress logging
        total_batches = len(train_loader)
        if step == 0 or (step + 1) % max(1, total_batches // 10) == 0 or step + 1 == total_batches:
            avg_loss = running_losses['total'] / max(n_batches, 1)
            print(f"  [Epoch {epoch}] batch {step+1}/{total_batches} "
                  f"loss={avg_loss:.4f}", flush=True)

    # Flush any remaining gradients from incomplete accumulation
    if n_batches % grad_accum_steps != 0:
        if has_accumulated:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_norm)
                if torch.isfinite(total_norm):
                    optimizer.step()
            optimizer.zero_grad()

    for k in running_losses:
        running_losses[k] /= max(n_batches, 1)

    return running_losses


@torch.no_grad()
def validate(model, val_loader, device, epoch, cfg, amp_enabled=False,
             amp_dtype=torch.float16, n_views=1):
    """Run validation."""
    model.eval()
    use_bridge = not cfg.get('no_bridge', False)
    if not amp_enabled:
        amp_dtype = torch.float32

    running_losses = {
        'total': 0.0, 'landmark': 0.0, 'angular': 0.0, 'angular_deg': 0.0,
        'landmark_px': 0.0, 'pose': 0.0, 'ray': 0.0, 'translation': 0.0,
    }
    n_batches = 0

    for batch in val_loader:

        #When using samples_per_subject in MDS shard loading we'll skip empty batches
        if not batch:
            continue

        images = batch['image'].to(device, non_blocking=True).float().div_(255.0)
        gt_landmarks = batch['landmark_coords'].to(device, non_blocking=True)
        gt_optical_axis = batch['optical_axis'].to(device, non_blocking=True)
        gt_landmarks_px = batch['landmark_coords_px'].to(device, non_blocking=True)

        R_cam = batch['R_cam'].to(device, non_blocking=True)
        T_cam = batch['T_cam'].to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            predictions = model(images, n_views=n_views,
                                R_cam=R_cam, T_cam=T_cam,
                                use_bridge=use_bridge)

            pred_hm = predictions['landmark_heatmaps']
            pred_coords = predictions['landmark_coords']
            pred_gaze = predictions['gaze_vector']

            feat_H, feat_W = pred_hm.shape[2], pred_hm.shape[3]

            loss, components = total_loss(
                pred_hm, pred_coords, pred_gaze,
                gt_landmarks, gt_optical_axis,
                feat_H, feat_W,
                lam_lm=cfg['lam_lm'],
                lam_gaze=cfg['lam_gaze'],
                sigma=cfg['sigma'],
            )

        img_size = images.shape[-1]
        feat_size = feat_H
        pred_coords_px = pred_coords * (img_size / feat_size)
        lm_px_error = torch.mean(torch.norm(pred_coords_px - gt_landmarks_px, dim=-1))

        batch_total = components['total_loss'].item()
        if batch_total != batch_total:  # nan check
            continue
        running_losses['total'] += batch_total
        running_losses['landmark'] += components['landmark_loss'].item()
        running_losses['angular'] += components['angular_loss'].item()
        running_losses['angular_deg'] += components['angular_loss_deg'].item()
        running_losses['landmark_px'] += lm_px_error.item()
        pose_val = components.get('pose_loss')
        if pose_val is not None:
            running_losses['pose'] += pose_val.item()
        ray_val = components.get('ray_target_loss')
        if ray_val is not None:
            running_losses['ray'] += ray_val.item()
        trans_val = components.get('translation_loss')
        if trans_val is not None:
            running_losses['translation'] += trans_val.item()
        n_batches += 1

    for k in running_losses:
        running_losses[k] /= max(n_batches, 1)

    return running_losses


# ============== Main Training ==============

def _build_run_config(args, hw):
    """Collect all training configuration into a single dict for metadata."""
    return {
        'stage': args.stage,
        'profile': args.profile,
        'core_backbone': args.core_backbone,
        'pose_backbone': args.pose_backbone,
        'epochs': args.epochs,
        'eye': args.eye,
        'hardware': hw,
        'phase_config': {
            str(k): {kk: vv for kk, vv in v.items() if kk != 'description'}
            for k, v in PHASE_CONFIG.items()
        },
        'streaming': args.streaming,
        'dataset_url': getattr(args, 'dataset_url', None),
        'data_dir': getattr(args, 'data_dir', None),
        'core_backbone_weight_path': args.core_backbone_weight_path,
        'pose_backbone_weight_path': args.pose_backbone_weight_path,
        'started_at': datetime.now().isoformat(),
    }


def _init_checkpoint_manager(args):
    """Create a CheckpointManager if MinIO checkpointing is enabled."""
    if not args.ckpt_bucket:
        return None
    from RayNet.streaming.checkpoint import CheckpointManager
    return CheckpointManager(
        bucket=args.ckpt_bucket,
        prefix=args.ckpt_prefix,
        run_id=args.run_id,
        endpoint=args.minio_endpoint,
        local_cache=os.path.join(args.output_dir, 'ckpt_cache'),
        save_local_copy=True,
    )


def train(args):
    """Main training function."""
    # Select stage config — updates module-level PHASE_CONFIG for get_phase/get_phase_config
    global PHASE_CONFIG
    is_v5 = getattr(args, 'architecture', 'v4') == 'v5'
    if is_v5:
        PHASE_CONFIG = STAGE_CONFIGS_V5[args.stage]
    else:
        PHASE_CONFIG = STAGE_CONFIGS[args.stage]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Training stage: {args.stage}")

    # Apply hardware profile
    hw = apply_hardware_profile(args)
    setup_hardware(hw, device)

    # Resolve AMP dtype from profile string
    amp_dtype = AMP_DTYPE_MAP.get(hw['amp_dtype'], torch.float16)

    print(f"Profile: {args.profile}")
    print(f"  Batch size: {hw['batch_size']} (effective: "
          f"{hw['batch_size'] * hw['grad_accum_steps']})")
    print(f"  AMP: {hw['amp']} ({hw['amp_dtype']})")
    print(f"  Gradient accumulation: {hw['grad_accum_steps']} steps")
    print(f"  torch.compile: {hw['compile_model']}")

    # Checkpoint manager (MinIO)
    ckpt_mgr = _init_checkpoint_manager(args)
    if ckpt_mgr is not None:
        print(f"  MinIO checkpoints: s3://{args.ckpt_bucket}/{args.ckpt_prefix}/{ckpt_mgr.run_id}/")

    # Create output directory
    if ckpt_mgr is not None:
        output_dir = os.path.join(args.output_dir, ckpt_mgr.run_id)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Create model
    if is_v5:
        from RayNet.raynet_v5 import create_raynet_v5
        model = create_raynet_v5(
            backbone_weight_path=args.core_backbone_weight_path,
            n_landmarks=14,
        )
    else:
        model = create_raynet(
            core_backbone_name=args.core_backbone,
            core_backbone_weight_path=args.core_backbone_weight_path,
            pose_backbone_name=args.pose_backbone,
            pose_backbone_weight_path=args.pose_backbone_weight_path,
            n_landmarks=14,
        )

    if hw['compile_model'] and hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # AMP scaler — ONLY for float16. BF16 has FP32 range, so loss scaling is
    # unnecessary and actively harmful (the scaler misdetects inf gradients
    # and silently skips every optimizer step, freezing training at init).
    use_scaler = hw['amp'] and hw.get('amp_dtype', 'float16') == 'float16'
    scaler = GradScaler('cuda', enabled=True) if use_scaler else None

    # --- Data loading ---
    # v3: multi-view is always active, so we create MV loader upfront.
    # Three modes: local disk, MDS streaming (MosaicML + MinIO), WebDataset streaming
    if args.mds_streaming:
        # Build MV loaders for BOTH train and val. The single-view val loader
        # from _create_mds_loaders was previously used here, which forced
        # validation to run with n_views=1 — bypassing CrossViewAttention,
        # cam_embed, and PoseEncoder fusion. That made val gaze metrics
        # uninformative. We now validate under the same multi-view topology
        # the model is trained with.
        train_loader_mv, val_loader = _create_mds_mv_loader(args, hw)
        streaming_mode = False
    elif args.streaming:
        _create_streaming_loaders(args, hw)
        train_loader_mv = None
        val_loader = None
        streaming_mode = True
    else:
        streaming_mode = False
        _, train_loader_mv, val_loader = _create_local_loaders(args, hw)

    # Record config in checkpoint metadata
    run_config = _build_run_config(args, hw)
    if ckpt_mgr is not None:
        ckpt_mgr.set_config(run_config)

    # CSV logger (epoch-level)
    csv_path = os.path.join(output_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'epoch', 'stage', 'phase', 'lr',
        'train_total', 'train_landmark', 'train_angular_deg',
        'train_reproj', 'train_mask', 'train_ray_target', 'train_pose',
        'train_translation',
        'val_total', 'val_landmark', 'val_angular_deg', 'val_landmark_px',
    ])

    # Batch-level CSV logger (high granularity)
    batch_csv_path = os.path.join(output_dir, 'batch_log.csv')
    batch_csv_file = open(batch_csv_path, 'w', newline='')
    batch_csv_writer = csv.writer(batch_csv_file)
    batch_csv_writer.writerow([
        'epoch', 'batch', 'loss', 'landmark', 'angular_deg',
        'gaze_consist', 'shape', 'ray_target', 'pose', 'translation', 'lr',
    ])

    # Training loop with phase transitions
    best_val_loss = float('inf')
    current_phase = 0
    start_epoch = 1

    # --- Resume from checkpoint ---
    # We need a dummy optimizer/scheduler for resume, so we create them
    # for the first phase first, then resume overwrites.
    optimizer = None
    scheduler = None

    if args.resume and ckpt_mgr is not None:
        print(f"Resuming from run {ckpt_mgr.run_id} ...")
        # We need optimizer & scheduler to exist before resume_state.
        # Create them for the starting phase; resume will overwrite state.
        resume_phase_cfg = get_phase_config(1)
        optimizer = optim.AdamW(model.parameters(), lr=resume_phase_cfg['lr'],
                                betas=(0.5, 0.95), weight_decay=1e-4)
        phase_start, phase_end = resume_phase_cfg['epochs']
        scheduler = CosineAnnealingLR(optimizer, T_max=phase_end - phase_start + 1)

        resume_file = getattr(args, 'resume_from', None)
        start_epoch, resume_ckpt = ckpt_mgr.resume_state(
            model, optimizer, scheduler=scheduler, scaler=scaler,
            map_location=device, filename=resume_file,
        )
        current_phase = get_phase(resume_ckpt['epoch'])
        best_val_loss = resume_ckpt.get('val_metrics', {}).get('total', best_val_loss)
        print(f"  Resumed at epoch {start_epoch} (phase {current_phase}), "
              f"best_val_loss={best_val_loss:.4f}")

        # Guard: empty training range. Happens when resuming a completed run
        # without extending --epochs. Fail loudly instead of silently exiting.
        if start_epoch > args.epochs:
            raise RuntimeError(
                f"Cannot resume: checkpoint is already at epoch "
                f"{resume_ckpt['epoch']}, so training would start at epoch "
                f"{start_epoch}, but --epochs={args.epochs}. Increase --epochs "
                f"to a value > {resume_ckpt['epoch']}, or use --warmstart_from "
                f"{ckpt_mgr.run_id} to start a new stage from these weights."
            )

    # --- Warmstart from a different run (e.g. Stage 1 -> Stage 2) ---
    # Loads ONLY model weights. Optimizer, scheduler, epoch counter, and
    # scaler stay fresh. A new run_id has already been generated by the
    # checkpoint manager (since --run_id is forbidden with --warmstart_from).
    if args.warmstart_from and ckpt_mgr is not None:
        print(f"Warmstarting from run {args.warmstart_from} "
              f"({args.warmstart_checkpoint}) into new run {ckpt_mgr.run_id} ...")
        ws_state = ckpt_mgr.load_from_run(
            source_run_id=args.warmstart_from,
            filename=args.warmstart_checkpoint,
            map_location=device,
        )
        target = model._orig_mod if hasattr(model, '_orig_mod') else model
        missing, unexpected = target.load_state_dict(
            ws_state['model_state_dict'], strict=False)
        if missing:
            print(f"  [warmstart] missing keys: {len(missing)} "
                  f"(first: {missing[:3]})")
        if unexpected:
            print(f"  [warmstart] unexpected keys: {len(unexpected)} "
                  f"(first: {unexpected[:3]})")
        src_stage = ws_state.get('config', {}).get('stage', '?')
        src_epoch = ws_state.get('epoch', '?')
        print(f"  Loaded weights from stage {src_stage} epoch {src_epoch}. "
              f"Starting fresh optimizer at epoch 1 of stage {args.stage}.")

        # Translation loss formulation changed (tanh/exp → direct linear
        # meters). The old pose_head rows 6:9 were trained for the prior
        # interpretation and would output garbage under the new loss.
        # Zero-init those rows while keeping rotation rows 0:6 (which
        # converged cleanly to ~0.015 rad geodesic). With zeroed
        # weight+bias, initial translation prediction is (0,0,0) m and
        # gradient flow resumes normally from the first batch.
        if (args.reset_pose_translation and hasattr(target, 'pose_encoder')
                and hasattr(target.pose_encoder, 'pose_head')):
            with torch.no_grad():
                target.pose_encoder.pose_head.weight[6:].zero_()
                target.pose_encoder.pose_head.bias[6:].zero_()
            print("  [warmstart] Reset pose_head translation rows (6:9) "
                  "to zero — translation loss reformulated to direct "
                  "cm→m SmoothL1. Rotation rows (0:6) preserved.")
        # Zero-init LandmarkGazeBridge out_proj so it starts as an identity
        # (skip connection). The bridge was disabled in earlier stages, so its
        # weights in the checkpoint are untrained random init. Loading them
        # overwrites the zero-init from __init__ — we must re-apply it here.
        if (args.reset_bridge and hasattr(target, 'landmark_gaze_bridge')
                and hasattr(target.landmark_gaze_bridge, 'cross_attn')):
            with torch.no_grad():
                target.landmark_gaze_bridge.cross_attn.out_proj.weight.zero_()
                target.landmark_gaze_bridge.cross_attn.out_proj.bias.zero_()
            print("  [warmstart] Zero-init bridge out_proj — bridge starts "
                  "as identity (skip connection), learns gradually.")

        # start_epoch stays at 1, optimizer/scheduler stay None — they will
        # be created in the phase-transition block below exactly like a
        # from-scratch run.

    for epoch in range(start_epoch, args.epochs + 1):
        phase = get_phase(epoch)
        cfg = get_phase_config(epoch)

        # Ablation overrides
        if args.no_multiview:
            cfg['multiview'] = False
            cfg['lam_reproj'] = 0.0
            cfg['lam_mask'] = 0.0
        if args.gaze_only:
            cfg['lam_lm'] = 0.0

        # Phase transition: update LR schedule, keep optimizer state
        if phase != current_phase:
            current_phase = phase
            print(f"\n{'='*60}")
            print(f"Phase {phase}: {cfg['description']}")
            print(f"  lr={cfg['lr']}, lam_lm={cfg['lam_lm']}, "
                  f"lam_gaze={cfg['lam_gaze']}, lam_ray={cfg.get('lam_ray', 0)}, "
                  f"lam_pose={cfg.get('lam_pose', 0)}, lam_trans={cfg.get('lam_trans', 0)}, "
                  f"sigma={cfg['sigma']}")
            if cfg.get('multiview'):
                print(f"  Multi-view: lam_gaze_consist={cfg['lam_reproj']}, "
                      f"lam_shape={cfg['lam_mask']}")
            print(f"{'='*60}")

            phase_start, phase_end = cfg['epochs']

            if optimizer is None:
                # First phase of the stage: create fresh optimizer
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=cfg['lr'],
                    betas=(0.5, 0.95),
                    weight_decay=1e-4,
                )
                print(f"  Created new AdamW optimizer (lr={cfg['lr']})")
            else:
                # Subsequent phases: carry over optimizer state (momentum +
                # adaptive second-moment estimates), only update the LR.
                # Recreating AdamW here would zero out m and v, causing a
                # destructive transient: raw-gradient steps at peak LR with
                # no per-parameter scaling — the "phase shock" that wastes
                # ~4 epochs recovering each time.
                for pg in optimizer.param_groups:
                    pg['lr'] = cfg['lr']
                    pg['initial_lr'] = cfg['lr']  # CosineAnnealingLR reads this
                print(f"  Carried over optimizer state, updated LR → {cfg['lr']}")

            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=phase_end - phase_start + 1,
            )

        # Select dataloader — v3 always uses multi-view for training
        if streaming_mode:
            active_train_loader, active_val_loader = _get_streaming_loaders(
                args, hw, cfg)
        else:
            active_train_loader = train_loader_mv
            active_val_loader = val_loader

        # n_views for CrossViewAttention during TRAINING.
        # NOTE: this is independent of cfg['multiview']. CrossViewAttention
        # runs whenever mv_groups > 1 (which produces 9-grouped batches).
        # cfg['multiview'] only gates the auxiliary multiview_consistency_loss
        # inside train_one_epoch — see header comment on STAGE_CONFIGS above.
        # Pass --no_multiview on the CLI to truly ablate cross-view fusion.
        active_n_views = 1 if args.no_multiview else 9

        # Train
        train_losses = train_one_epoch(
            model, active_train_loader, optimizer, device, epoch, cfg,
            scaler=scaler,
            grad_accum_steps=hw['grad_accum_steps'],
            amp_enabled=hw['amp'],
            amp_dtype=amp_dtype,
            batch_csv_writer=batch_csv_writer,
            n_views=active_n_views,
        )
        batch_csv_file.flush()

        # Validate with the SAME multi-view topology used in training.
        # Previously hardcoded to n_views=1, which bypassed CrossViewAttention,
        # camera embeddings, and the PoseEncoder fusion path — producing
        # uninformative val gaze metrics. The val loader now delivers
        # 9-grouped batches (see _create_mds_mv_loader), so the reshape
        # inside CrossViewAttention is well-defined.
        val_losses = validate(
            model, active_val_loader, device, epoch, cfg,
            amp_enabled=hw['amp'],
            amp_dtype=amp_dtype,
            n_views=active_n_views,
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        mv_str = ""
        if cfg.get('multiview'):
            mv_str = (f" gaze_mv={train_losses['reproj']:.4f}"
                      f" shape={train_losses['mask']:.4f}")
        if train_losses.get('ray_target', 0) > 0:
            mv_str += f" ray={train_losses['ray_target']:.4f}"
        if train_losses.get('pose', 0) > 0:
            mv_str += f" pose={train_losses['pose']:.4f}"
        if train_losses.get('translation', 0) > 0:
            mv_str += f" trans={train_losses['translation']:.4f}"
        print(f"Epoch {epoch:3d} | Phase {phase} | lr {current_lr:.2e} | "
              f"Train: loss={train_losses['total']:.4f} lm={train_losses['landmark']:.4f} "
              f"ang={train_losses['angular_deg']:.2f}deg{mv_str} | "
              f"Val: loss={val_losses['total']:.4f} lm_px={val_losses['landmark_px']:.2f}px "
              f"ang={val_losses['angular_deg']:.2f}deg")

        if hw['amp'] and device.type == 'cuda':
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"  GPU memory peak: {mem_gb:.1f} GB")

        csv_writer.writerow([
            epoch, args.stage, phase, f"{current_lr:.2e}",
            f"{train_losses['total']:.6f}",
            f"{train_losses['landmark']:.6f}",
            f"{train_losses['angular_deg']:.4f}",
            f"{train_losses.get('reproj', 0):.6f}",
            f"{train_losses.get('mask', 0):.6f}",
            f"{train_losses.get('ray_target', 0):.6f}",
            f"{train_losses.get('pose', 0):.6f}",
            f"{train_losses.get('translation', 0):.6f}",
            f"{val_losses['total']:.6f}",
            f"{val_losses['landmark']:.6f}",
            f"{val_losses['angular_deg']:.4f}",
            f"{val_losses['landmark_px']:.4f}",
        ])
        csv_file.flush()

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']

            if ckpt_mgr is not None:
                ckpt_mgr.save_best(
                    epoch=epoch, model=model, val_loss=best_val_loss,
                    val_metrics=val_losses, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler,
                    extra={'profile': args.profile},
                )
            else:
                save_dict = {
                    'epoch': epoch,
                    'phase': phase,
                    'model_state_dict': (model._orig_mod.state_dict()
                                         if hasattr(model, '_orig_mod')
                                         else model.state_dict()),
                    'val_loss': best_val_loss,
                    'val_landmark_px': val_losses['landmark_px'],
                    'val_angular_deg': val_losses['angular_deg'],
                    'profile': args.profile,
                }
                torch.save(save_dict, os.path.join(output_dir, 'best_model.pt'))
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

        # Periodic + latest checkpoint
        if ckpt_mgr is not None:
            # Always save latest (for resume)
            ckpt_mgr.save(
                epoch=epoch, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, phase=phase,
                train_metrics=train_losses, val_metrics=val_losses,
                tag='latest',
            )
            # Periodic named checkpoint
            if epoch % args.ckpt_every == 0:
                ckpt_mgr.save(
                    epoch=epoch, model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, phase=phase,
                    train_metrics=train_losses, val_metrics=val_losses,
                )
            # Upload logs after each epoch (survives interruption/crash)
            try:
                batch_csv_file.flush()
                csv_file.flush()
                for local_path in (batch_csv_path, csv_path):
                    log_key = f"{args.ckpt_prefix}/{ckpt_mgr.run_id}/{os.path.basename(local_path)}"
                    ckpt_mgr._client.fput_object(args.ckpt_bucket, log_key, local_path)
            except Exception as e:
                log.warning("MinIO log upload failed: %s", e)
        else:
            if epoch % args.ckpt_every == 0:
                save_dict = {
                    'epoch': epoch,
                    'phase': phase,
                    'model_state_dict': (model._orig_mod.state_dict()
                                         if hasattr(model, '_orig_mod')
                                         else model.state_dict()),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                if scaler is not None:
                    save_dict['scaler_state_dict'] = scaler.state_dict()
                torch.save(save_dict, os.path.join(output_dir, f'checkpoint_epoch{epoch}.pt'))

    csv_file.close()
    batch_csv_file.close()

    # Upload training logs to MinIO
    if ckpt_mgr is not None:
        for log_file in [csv_path, batch_csv_path]:
            if os.path.exists(log_file):
                log_key = f"{args.ckpt_prefix}/{ckpt_mgr.run_id}/{os.path.basename(log_file)}"
                try:
                    ckpt_mgr._client.fput_object(args.ckpt_bucket, log_key, log_file)
                    print(f"  Uploaded {os.path.basename(log_file)} to s3://{args.ckpt_bucket}/{log_key}")
                except Exception as e:
                    print(f"  Warning: failed to upload {os.path.basename(log_file)}: {e}")

    print(f"\nTraining complete. Results saved to {output_dir}")
    print(f"Best val loss: {best_val_loss:.4f}")
    if ckpt_mgr is not None:
        print(f"Checkpoints: s3://{args.ckpt_bucket}/{args.ckpt_prefix}/{ckpt_mgr.run_id}/")


# ============== Data Loader Helpers ==============

def _create_local_loaders(args, hw):
    """Create local disk-based dataloaders."""
    train_subjects = list(range(1, 47))
    val_subjects = list(range(47, 57))

    common_kwargs = dict(
        num_workers=hw['num_workers'],
        samples_per_subject=args.samples_per_subject,
        eye=args.eye,
    )

    train_loader_standard, val_loader = create_dataloaders(
        base_dir=args.data_dir,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        batch_size=hw['batch_size'],
        ensure_multiview=False,
        **common_kwargs,
    )

    train_loader_mv, _ = create_dataloaders(
        base_dir=args.data_dir,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        batch_size=hw['mv_groups'],
        ensure_multiview=True,
        **common_kwargs,
    )

    return train_loader_standard, train_loader_mv, val_loader


def _create_mds_loaders(args, hw):
    """Create MosaicML MDS streaming dataloaders from MinIO / S3."""
    from RayNet.streaming import create_streaming_dataloaders
    from RayNet.streaming.minio_utils import configure_minio_env

    # Configure MinIO env vars if endpoint is provided
    if args.minio_endpoint:
        configure_minio_env(
            args.minio_endpoint,
            os.environ.get('AWS_ACCESS_KEY_ID', 'minioadmin'),
            os.environ.get('AWS_SECRET_ACCESS_KEY', 'minioadmin'),
        )

    print(f"MDS streaming: train={args.mds_train}, val={args.mds_val}")

    # Data augmentation matching GazeGene paper (Sec 4.1.3):
    # random translation + color jitter
    from torchvision import transforms as T
    train_transform = T.Compose([
        T.ColorJitter(brightness=0.4, contrast=0.4,
                      saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ])

    train_loader, val_loader = create_streaming_dataloaders(
        remote_train=args.mds_train,
        remote_val=args.mds_val,
        local_cache=os.path.join(args.output_dir, 'mds_cache'),
        batch_size=hw['batch_size'],
        num_workers=hw['num_workers'],
        transform=train_transform,
        pin_memory=hw['pin_memory'],
        prefetch_factor=hw['prefetch_factor'],
        persistent_workers=hw['persistent_workers'],
        download_timeout=120,
        samples_per_subject=args.samples_per_subject
    )

    print(f"  Samples per subject: {args.samples_per_subject} samples")
    print(f"  Train dataset: {len(train_loader.dataset)} samples, "
          f"{len(train_loader)} batches")
    print(f"  Val dataset:   {len(val_loader.dataset)} samples, "
          f"{len(val_loader)} batches")

    return train_loader, val_loader


def _create_mds_mv_loader(args, hw):
    """Create multi-view MDS loaders (both train and val).

    Returns (train_loader_mv, val_loader_mv). The MV val loader delivers
    batches of mv_groups * 9 samples in (subject, frame, cam) order so
    that CrossViewAttention, cam_embed, and PoseEncoder features are
    exercised during validation — matching the training forward pass.
    Without this, validation bypasses all multi-view fusion (n_views=1)
    and val gaze metrics are a pessimistic lower bound that never moves.
    """
    from RayNet.streaming.dataset import create_multiview_streaming_dataloaders

    # Same augmentation as single-view loader (train only)
    from torchvision import transforms as T
    train_transform = T.Compose([
        T.ColorJitter(brightness=0.4, contrast=0.4,
                      saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ])

    print("Creating multi-view MDS streaming loaders (train + val)...")
    train_loader_mv, val_loader_mv = create_multiview_streaming_dataloaders(
        remote_train=args.mds_train,
        remote_val=args.mds_val,
        local_cache=os.path.join(args.output_dir, 'mds_cache_mv'),
        mv_groups=hw['mv_groups'],
        num_workers=hw['num_workers'],
        transform=train_transform,
        pin_memory=hw['pin_memory'],
        prefetch_factor=hw['prefetch_factor'],
        persistent_workers=hw['persistent_workers'],
        samples_per_subject=args.samples_per_subject,
    )
    return train_loader_mv, val_loader_mv


def _create_streaming_loaders(args, hw):
    """Validate streaming args at startup (WebDataset mode)."""
    if not args.dataset_url:
        raise ValueError("--dataset_url required when --streaming is set")
    print(f"WebDataset streaming mode: {args.dataset_url}")


def _get_streaming_loaders(args, hw, cfg):
    """Create streaming dataloaders for the current phase."""
    from RayNet.webdataset_utils import (
        create_streaming_dataloader,
        create_multiview_streaming_dataloader,
    )

    if cfg.get('multiview'):
        train_loader = create_multiview_streaming_dataloader(
            urls=args.dataset_url,
            mv_groups=hw['mv_groups'],
            num_workers=hw['num_workers'],
            shuffle=True,
        )
    else:
        train_loader = create_streaming_dataloader(
            urls=args.dataset_url,
            batch_size=hw['batch_size'],
            num_workers=hw['num_workers'],
            shuffle=True,
        )

    val_url = args.val_dataset_url or args.dataset_url
    val_loader = create_streaming_dataloader(
        urls=val_url,
        batch_size=hw['batch_size'],
        num_workers=hw['num_workers'],
        shuffle=False,
    )

    return train_loader, val_loader


# ============== CLI ==============

def parse_args():
    parser = argparse.ArgumentParser(description='RayNet v3 Training')

    # Data
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to GazeGene dataset (local mode)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--samples_per_subject', type=int, default=200)
    parser.add_argument('--eye', type=str, default='L', choices=['L', 'R'])

    # MDS streaming (MosaicML + MinIO)
    parser.add_argument('--mds_streaming', action='store_true',
                        help='Stream training data from MDS shards on MinIO/S3')
    parser.add_argument('--mds_train', type=str, default=None,
                        help='MDS shard URL for training (e.g. s3://gazegene/train)')
    parser.add_argument('--mds_val', type=str, default=None,
                        help='MDS shard URL for validation (e.g. s3://gazegene/val)')

    # WebDataset streaming (legacy)
    parser.add_argument('--streaming', action='store_true',
                        help='Use WebDataset streaming instead of local disk')
    parser.add_argument('--dataset_url', type=str, default=None,
                        help='WebDataset shard URL pattern for training')
    parser.add_argument('--val_dataset_url', type=str, default=None,
                        help='WebDataset shard URL pattern for validation')

    # Model
    parser.add_argument('--core_backbone', type=str, default='repnext_m3',
                        choices=['repnext_m0', 'repnext_m1', 'repnext_m2',
                                 'repnext_m3', 'repnext_m4', 'repnext_m5'])
    parser.add_argument('--pose_backbone', type=str, default='repnext_m1',
                        choices=['repnext_m0', 'repnext_m1', 'repnext_m2',
                                 'none'],
                        help='Pose encoder backbone (separate from main). '
                             '"none" disables pose encoder.')
    parser.add_argument('--core_backbone_weight_path', type=str, default=None,
                        help='Path to pretrained core backbone  weights(not fused)')
    parser.add_argument('--pose_backbone_weight_path', type=str, default=None,
                        help='Path to pretrained head pose  weights(not fused)')

    # Hardware profile
    parser.add_argument('--profile', type=str, default='default',
                        choices=list(HARDWARE_PROFILES.keys()),
                        help=f'Hardware profile ({", ".join(HARDWARE_PROFILES.keys())})')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile even if profile enables it')
    parser.add_argument('--stage', type=int, default=3, choices=[1, 2, 3],
                        help='Training stage: 1=landmark+pose baseline, '
                             '2=add gaze (no bridge), 3=full pipeline (with bridge)')
    parser.add_argument('--no_multiview', action='store_true',
                        help='Disable multi-view losses and CrossViewAttention '
                             '(n_views=1 throughout, for ablation)')
    parser.add_argument('--gaze_only', action='store_true',
                        help='Disable landmark loss (lam_lm=0) for gaze-only '
                             'training, matching GazeGene paper baseline')

    # Overrides (None = use profile default)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--mv_groups', type=int, default=None,
                        help='Multi-view batch groups (actual batch = mv_groups * 9)')
    parser.add_argument('--grad_accum_steps', type=int, default=None,
                        help='Gradient accumulation steps')

    # MinIO checkpoint storage
    parser.add_argument('--ckpt_bucket', type=str, default=None,
                        help='MinIO bucket for checkpoints (enables MinIO storage)')
    parser.add_argument('--ckpt_prefix', type=str, default='checkpoints',
                        help='Key prefix under the bucket (default: checkpoints)')
    parser.add_argument('--minio_endpoint', type=str, default=None,
                        help='MinIO endpoint URL (default: S3_ENDPOINT_URL env var)')
    parser.add_argument('--ckpt_every', type=int, default=5,
                        help='Save a named checkpoint every N epochs (default: 5)')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Run ID for checkpoint grouping (auto-generated if omitted)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint of --run_id '
                             '(same run, same stage, continues epoch counter)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from a specific checkpoint file '
                             '(e.g. checkpoint_epoch5.pt, best_model.pt)')
    parser.add_argument('--warmstart_from', type=str, default=None,
                        help='Warmstart from a DIFFERENT run (e.g. start Stage 2 '
                             'with weights from a completed Stage 1 run). Loads '
                             'ONLY model weights — optimizer/scheduler/epoch are '
                             'reset. A new run_id is auto-generated.')
    parser.add_argument('--warmstart_checkpoint', type=str, default='best_model.pt',
                        help='Checkpoint file to pull from --warmstart_from run '
                             '(default: best_model.pt)')
    parser.add_argument('--reset_pose_translation', action='store_true',
                        help='After warmstart, zero-init pose_head rows 6:9 '
                             '(translation). Use when the translation loss '
                             'formulation changed between the source run and '
                             'the current code (e.g. tanh/exp → direct '
                             'cm→m SmoothL1). Rotation rows 0:6 are preserved.')
    parser.add_argument('--reset_bridge', action='store_true',
                        help='After warmstart, zero-init LandmarkGazeBridge '
                             'out_proj so the bridge starts as a pure identity '
                             '(skip connection). Required when transitioning '
                             'to a stage with bridge enabled from a stage '
                             'where bridge was disabled (e.g. Stage 2 → 3).')

    args = parser.parse_args()

    if not args.streaming and not args.mds_streaming and args.data_dir is None:
        parser.error("--data_dir is required when not using --streaming or --mds_streaming")

    if args.mds_streaming and (not args.mds_train or not args.mds_val):
        parser.error("--mds_streaming requires both --mds_train and --mds_val")

    # --resume_from implies --resume
    if args.resume_from:
        args.resume = True

    if args.resume and not args.run_id:
        parser.error("--resume requires --run_id to identify which run to resume")
    if args.resume and not args.ckpt_bucket:
        parser.error("--resume requires --ckpt_bucket")
    if args.warmstart_from and args.resume:
        parser.error("--warmstart_from and --resume are mutually exclusive. "
                     "Use --resume to continue an interrupted run; use "
                     "--warmstart_from to start a new stage with weights "
                     "from another run.")
    if args.warmstart_from and not args.ckpt_bucket:
        parser.error("--warmstart_from requires --ckpt_bucket")
    if args.warmstart_from and args.run_id:
        parser.error("--warmstart_from creates a new run — do not pass --run_id "
                     "(a fresh timestamped run_id will be generated).")

    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)
