"""
RayNet v5 Training Script (Triple-M1 + AERI/HRFH-α, branch-staged).

Training strategy — one stage, three branch-staged phases:

  Phase 1 (epochs 1-8) — landmark + AERI seg + headpose. Gaze branch
    (encoder + fusion + GeometricGazeHead) is FROZEN. AERI head stays
    trainable so iris/eyeball masks converge before HRFH-α consumes
    them. lam_gaze = 0; seg weights raised to 1.0.

  Phase 2 (epochs 9-18) — monocular gaze fine-tune. Shared stem +
    landmark branch + pose branch are FROZEN (.eval() so BN stats
    don't drift). Multi-view consistency OFF, n_views forced to 1.
    α ramped from 0.4 to 0.7 over the first 3 epochs of P2 (epochs
    9-11), held at 0.7 for the rest. Mask supervision stays at 0.5.

  Phase 3 (epochs 19-35) — full unfreeze + multi-view fusion at
    5×–10× lower LR than P2. α held constant at 0.7 (no ramp during
    fine-tune — the previously-shipped 0.4→0.9 ramp during the cosine
    LR decay caused validation drift in
    `triple_m1_aeri_iris_eyeball_500spc_run_20260423_101115`).

  Gradient clipping (max_norm) varies by phase:
    Phase 1: max_norm=5.0 (aggressive, allows large multi-task gradients)
    Phase 2+: max_norm=2.0 (conservative, prevents gaze/pose interference)

Usage:
    # Single GPU
    python -m RayNet.train --profile t4 --mds_streaming \
        --mds_train s3://gazegene/train --mds_val s3://gazegene/val ...

    # Kaggle 2× Tesla T4 (single node, multi-GPU)
    accelerate launch --multi_gpu --num_processes 2 \
        -m RayNet.train --profile kaggle_t4x2 --mds_streaming ...

    # Two machines on the same network (see RayNet/hardware_profiles.py)
    accelerate launch --multi_gpu --num_machines 2 --num_processes 2 \
        --machine_rank <0|1> --main_process_ip $MAIN_IP --main_process_port 29500 \
        -m RayNet.train --profile multi_node_t4 --mds_streaming ...
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

from RayNet.dataset import create_dataloaders
from RayNet.hardware_profiles import (
    HARDWARE_PROFILES,
    AMP_DTYPE_MAP,
    apply_hardware_profile,
    setup_hardware,
    build_accelerator,
)
from RayNet.losses import total_loss
from RayNet.multiview_loss import multiview_consistency_loss
from RayNet.raynet_v5 import create_raynet_v5

log = logging.getLogger(__name__)


# ============== TriCam camera selection ==============
#
# GazeGene ships 9 cameras. Empirical 3D-triangulation analysis of the
# rig (see docs/camera_info.pkl + the analysis in run_20260430 review)
# shows the historic horizontal triplet {3, 4, 5} is geometrically
# near-degenerate for resolving eyeball-centre depth: those three cams
# share z = -162 cm, so the cross-axis baseline collapses and the
# triangulated 3D anchor drifts under sub-pixel detection error.
#
# The cam triplet {1, 6, 8} maximises 3D triangulation area (~85 k cm²
# vs 16.7 k for {3,4,5}) while keeping cam 1 in the front ring (best
# landmark visibility) and cams 6, 8 in the back ring (large z-baseline
# for depth recovery). See the unit-grounded analysis in conversation
# logs for the full ranking.
TRICAM_IDS = (1, 6, 8)
TRICAM_N_VIEWS = len(TRICAM_IDS)


# ============== Staged Training Configuration ==============
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
#   producing TRICAM_N_VIEWS-grouped multi-view batches, and
#   `active_n_views` is set to TRICAM_N_VIEWS (only suppressed by the
#   --no_multiview CLI ablation flag).
#
#   Validation uses n_views=1, so CrossViewAttention, camera embeddings,
#   and the PoseEncoder features ALL bypass during validation. This creates
#   a deliberate train/val asymmetry: training sees fused multi-view
#   geometry, val sees single-view — val metrics are therefore a strict
#   lower bound on what the model is actually capable of at inference
#   with multi-view.
#
#   If you truly want a "no multi-view fusion" phase, pass --no_multiview
#   on the CLI (not the `multiview: False` config flag).
# -----------------------------------------------------------------------------

PHASE_CONFIG = {
    # ---- Phase 1 — landmark + headpose only (skeleton geometry). ----
    # Texture-decoupling: GazeGene supervises ONLY skeleton/anatomical
    # features (sparse landmarks + 3D head pose). The dense AERI iris/
    # eyeball masks have been removed from the GazeGene loss path —
    # those will train exclusively on real OpenEDS IR masks in a future
    # OpenEDS-only stage to avoid letting the model overfit MetaHuman
    # foveal renderings. The AERI head is still constructed and produces
    # masks for the saliency gate, but with both seg weights = 0 it
    # receives no gradient on synthetic data and behaves as a frozen
    # uniform-ish prior. The shared stem is landmark-owned; pose trains
    # alongside. The gaze branch (encoder, fusion, GeometricGazeHead)
    # is frozen (.eval() + requires_grad_(False)) by `apply_phase_freeze`.
    1: {
        'epochs': (1, 15),
        'lam_lm': 1.0,
        'lam_gaze': 0.0,        # gaze branch frozen
        'lam_gaze_sv': 0.0,
        'lam_eyeball': 0.0,
        'lam_pupil': 0.0,
        'lam_geom_angular': 0.0,
        'lam_ray': 0.0,
        'lam_reproj': 0.0,
        'lam_mask': 0.0,
        'lam_pose': 0.0,         # pose deferred — strict landmark warmup
        'lam_trans': 0.0,        # v6.2 — pose translation head removed
        'lam_iris_seg': 0.0,    # foveal texture → OpenEDS-only stage
        'lam_eyeball_seg': 0.0, # foveal texture → OpenEDS-only stage
        # 3DGazeNet M-target — iris contour mesh in CCS. Off in P1
        # because the gaze branch (which hosts the iris-mesh head) is
        # frozen here; the head is constructed but receives no gradient.
        'lam_iris_mesh': 0.0,
        'lam_iris_edge': 0.0,
        # Mean-of-two and visual-axis loss only fire when gaze unfreezes
        'lam_gaze_geom': 0.0,
        'lam_gaze_direct': 0.0,
        'lam_gaze_visual': 0.0,
        # Macro (head) gaze (`gaze_C`) is owned by the gaze branch, so
        # its weight must follow the gaze freeze schedule. Off in P1.
        'lam_gaze_macro': 0.0,
        # Per-subject eyeball radius head — same gaze-branch ownership.
        'lam_eyeball_radius': 0.0,
        # v6.2.2 (LR Rework): constant LR for the entire phase.
        # Cosine decay to ~0 within the 15-epoch P1 budget starved
        # geometry bootstrap of optimization energy and was the
        # dominant cause of the landmark-loss plateau visible in
        # run_20260504_230121.
        'lr': 1e-3,
        'sigma': 2.0,
        'multiview': False,
        'freeze_set': 'landmark_only',  # strict isolation — pose+gaze frozen
        # April reference (run_20260405_025128) trained the entire model
        # at the full 1e-3 in Phase 1 — no per-module attenuation. The
        # default LR_MULTIPLIERS bucket the M3 backbone at 0.1× and the
        # landmark heads at 0.2×, dropping their effective LRs to 1e-4
        # and 2e-4 respectively. That's a 5–10× starvation on exactly
        # the params that need to fit the heatmap. Override to 1.0×
        # uniformly for Phase 1 only; Phases 2/3 omit this key and
        # inherit the tiered defaults.
        'lr_multipliers': 'uniform',
        'description': 'V6-P1: landmark-only warmup '
                       '(pose + gaze branches frozen, BN stats stable).',
    },
    # ---- Phase 2 — monocular gaze fine-tune. Stem/lm/pose frozen. ----
    # Single-view only. Skeleton geometry only: gaze direction +
    # 3D eyeball / pupil anchors + optical-axis consistency. AERI seg
    # losses are 0 — foveal texture is owned by the OpenEDS stage.
    # lam_gaze_sv = 1.0 (== lam_gaze pathway) because n_views == 1 means
    # the single-view pathway IS the supervision pathway — there is no
    # CrossViewAttention to bypass.
    2: {
        'epochs': (16, 30),
        'lam_lm': 0.0,          # landmark frozen
        'lam_gaze': 2.0,        # supervises the FUSED gaze (mean-of-two)
        'lam_gaze_sv': 0.0,     # n_views==1 → pooled_sv == pooled, no SV side path
        'lam_eyeball': 0.3,
        'lam_pupil': 0.3,
        'lam_geom_angular': 0.5,
        'lam_ray': 0.0,
        'lam_reproj': 0.0,
        'lam_mask': 0.0,
        'lam_pose': 0.0,        # pose frozen
        'lam_trans': 0.0,
        'lam_iris_seg': 0.0,    # foveal texture → OpenEDS-only stage
        'lam_eyeball_seg': 0.0, # foveal texture → OpenEDS-only stage
        # 3DGazeNet M-target — vertex L1 + edge length L2 (paper ratios).
        'lam_iris_mesh': 0.1,   # 3DGazeNet λ_v
        'lam_iris_edge': 0.01,  # 3DGazeNet λ_e
        # Mean-of-two: supervise both sub-heads so neither collapses.
        # Half the fused-gaze weight each, so the combined effective
        # weight on direction (geom + direct + fused) is 2.0+0.5+0.5 = 3.0.
        'lam_gaze_geom': 0.5,
        'lam_gaze_direct': 0.5,
        # Visual-axis (kappa-corrected) supervision on the GEOMETRIC
        # branch only — applies kappa to optical (= gaze_geom) and
        # supervises against gt_visual_axis. Off until P2 because
        # gaze_geom only has signal once eyeball + pupil heads are
        # warmed up.
        'lam_gaze_visual': 0.5,
        # Macro (head) gaze — GazeGene gaze_C. Trained alongside the
        # micro signals; the MacroGazeHead consumes pose_feat + the
        # predicted eyeball anchor (detached) and is otherwise
        # independent of the optical/visual branches.
        'lam_gaze_macro': 1.0,
        'lam_eyeball_radius': 0.2,        # cm-scale L1; small weight is plenty
        # v6.2.2 (LR Rework): constant LR for the entire phase. P2 is
        # the high-plasticity adaptation phase — newly unfrozen gaze
        # branch + iris-mesh + macro-gaze + visual-axis heads all need
        # sustained optimization energy, which cosine decay starved.
        # 5e-4 matches the rework doc's OneCycleLR max_lr but is held
        # constant throughout the phase per the user's note that
        # constant LR is preferred for our learning process.
        'lr': 5e-4,
        'sigma': 1.5,
        'multiview': False,
        'freeze_set': 'face_only',  # freeze stem + landmark + pose
        'description': 'V6-P2: monocular gaze on skeleton anchors '
                       '+ iris mesh + visual axis + macro gaze.',
    },
    # ---- Phase 3 — multi-view fusion at 5×–10× lower LR. Face frozen. ----
    # The cosine-decay-from-3e-4 schedule used in run_20260423_101115
    # over-shook the basin discovered in P2 (val_angular_deg climbed
    # from 12.43 at epoch 28 to 15.30 at epoch 35). We start from a
    # much lower LR; α stays at 0.7 — no ramp during fine-tune.
    # Multi-view consistency ramps in via
    #     mv_weight = min(1, max(0, (epoch - 30) / 5))
    # in train_one_epoch — 5-epoch ramp anchored to the P3 start.
    #
    # Landmark fine-tune is intentionally OFF in P3:
    #   - lam_lm = 0.0
    #   - shared_stem + landmark_branch frozen via 'face_kept'
    # Rationale: P1 already drove val_landmark_px to ≤2.2 px. Continuing
    # to update the landmark branch in P3 risks pulling the shared stem
    # in a direction that improves landmark sub-pixel error at the cost
    # of the gaze representation. Pose still trains (it has no shared
    # parameters with gaze beyond the detached s1 input).
    3: {
        'epochs': (31, 50),
        'lam_lm': 0.0,          # landmark frozen — see freeze_set
        'lam_gaze': 2.0,        # fused gaze
        'lam_gaze_sv': 1.0,     # close train/val gap when CrossViewAttn is on
        'lam_eyeball': 0.4,
        'lam_pupil': 0.4,
        'lam_geom_angular': 0.5,
        'lam_ray': 0.0,
        'lam_reproj': 0.1,
        'lam_mask': 0.05,
        'lam_pose': 0.3,
        'lam_trans': 0.0,        # v6.2 — pose translation head removed
        'lam_iris_seg': 0.0,    # foveal texture → OpenEDS-only stage
        'lam_eyeball_seg': 0.0, # foveal texture → OpenEDS-only stage
        # 3DGazeNet M-target — slightly lower than P2 to let multi-view
        # consistency dominate.
        'lam_iris_mesh': 0.05,
        'lam_iris_edge': 0.005,
        # Mean-of-two sub-supervisions held at P2 levels.
        'lam_gaze_geom': 0.5,
        'lam_gaze_direct': 0.5,
        'lam_gaze_visual': 0.5,
        'lam_gaze_macro': 1.0,
        'lam_eyeball_radius': 0.2,
        # v6.2.2 (LR Rework): constant LR for P3. The previous cosine
        # decay on top of the already-low 5e-5 baseline produced
        # effective LRs in [1e-6, 1e-7] — well below useful, the
        # "double attenuation" the rework doc calls out.
        'lr': 5e-5,
        'sigma': 1.0,
        'multiview': True,
        'freeze_set': 'face_kept',  # freeze stem + landmark, keep pose trainable
        'description': 'V6-P3: TriCam multi-view + iris mesh + visual axis, '
                       'skeleton-only supervision.',
    },
}


# ============== Branch-Staged Freeze Helpers ==============

# Three named freeze sets cover the curriculum + fully-unfrozen ablation.
# `apply_phase_freeze` is idempotent and called once per phase transition.
FREEZE_SETS = {
    # P1 — gaze branch frozen except AERI head (which is supervised).
    'gaze_only': {
        'freeze_modules': [
            'gaze_branch.encoder',
            'gaze_branch.coord_att',
            'gaze_branch.foveal_proj',
            'gaze_branch.global_norm',
            'gaze_branch.foveal_norm',
            'gaze_branch.proj',
            'gaze_branch.fusion_block',
            'gaze_branch.head',
        ],
        # AERI head + cross_view_attn + camera_embedding intentionally
        # NOT frozen — AERI is supervised by seg loss; cross-view modules
        # see no gradient when n_views==1 anyway.
    },
    # P1-strict — landmark-only warmup. Mirrors ablation/4th_april where
    # Phase 1 supervised landmarks alone and reached 0.60 px val by
    # epoch 5. Both pose_branch and gaze_branch are frozen so their BN
    # running stats stop drifting, no contention reaches the optimizer
    # step, and weight decay can't shrink un-supervised pose params.
    # Only shared_stem + landmark_branch (encoder + FPN + heads) train.
    'landmark_only': {
        'freeze_modules': [
            'pose_branch',
            'gaze_branch',
            'cross_view_attn',
            'camera_embedding',
        ],
    },
    # P2 — face path frozen (stem + landmark + pose). Gaze + AERI train.
    'face_only': {
        'freeze_modules': [
            'shared_stem',
            'landmark_branch',
            'pose_branch',
        ],
    },
    # P3 — landmark + shared stem frozen, pose stays trainable.
    # Used when lam_lm == 0: keeping landmark/stem in train mode would
    # let BN buffers drift on every forward pass without supervision,
    # eroding the val_landmark_px earned in P1. Pose has no shared params
    # with the gaze branch (s1 is detached on entry to both pose and
    # gaze), so leaving it trainable does not destabilise gaze.
    'face_kept': {
        'freeze_modules': [
            'shared_stem',
            'landmark_branch',
        ],
    },
    # Fully-unfrozen ablation set (kept for completeness / experiments).
    'none': {
        'freeze_modules': [],
    },
}


def _resolve_module(root, dotted_name):
    """Walk dotted name down to a child module."""
    mod = root
    for part in dotted_name.split('.'):
        mod = getattr(mod, part)
    return mod


def apply_phase_freeze(model, freeze_set_name, log_fn=print):
    """
    Apply a named freeze pattern to the unwrapped RayNetV5 instance.

    For each frozen submodule:
      - requires_grad_(False)  — no parameter updates
      - .eval()                — BN running stats stop drifting

    Trainable submodules are explicitly set back to .train() and have
    requires_grad_(True) restored, so a phase that goes "narrow → wider"
    correctly re-enables previously frozen branches.

    The function operates on the unwrapped RayNetV5 (call _unwrap_raynet
    first when handed a DDP / torch.compile wrapper).
    """
    spec = FREEZE_SETS.get(freeze_set_name, FREEZE_SETS['none'])
    frozen = set(spec['freeze_modules'])

    # First pass: re-enable everything (idempotent baseline). We touch
    # only top-level branches + their children to avoid clobbering
    # buffers or modules outside the named set.
    top_level = ['shared_stem', 'landmark_branch', 'pose_branch',
                 'gaze_branch', 'cross_view_attn', 'camera_embedding']
    for name in top_level:
        if not hasattr(model, name):
            continue
        mod = getattr(model, name)
        mod.requires_grad_(True)
        mod.train()

    # Second pass: freeze the named submodules.
    for name in frozen:
        try:
            mod = _resolve_module(model, name)
        except AttributeError:
            log_fn(f"  [freeze] WARN: '{name}' not found, skipping")
            continue
        mod.requires_grad_(False)
        mod.eval()

    log_fn(f"  [freeze] applied set='{freeze_set_name}' "
           f"(frozen={sorted(frozen) or 'none'})")


def get_phase(epoch):
    """Return the training phase for a given epoch."""
    for phase, cfg in PHASE_CONFIG.items():
        start, end = cfg['epochs']
        if start <= epoch <= end:
            return phase
    # Fallback: return the last defined phase (not hardcoded 3)
    return max(PHASE_CONFIG.keys())

def get_scheduled_alpha(epoch):
    """
    Three-region AERI-α schedule aligned with the branch-staged curriculum.

      - Phase 1 (epochs 1-15):  α = 0.4 — gaze branch is frozen, value is
        moot but kept low to keep the (training-only) AERI losses from
        being gated. Masks are still being learned.
      - Phase 2 ramp (epochs 16-18): α linearly interpolated 0.4 → 0.7
        over the first 3 epochs of gaze training — small, deliberate ARI
        window BEFORE LR cosine decay, not during fine-tune.
      - Phase 2 fine-tune + Phase 3 (epochs 19+): α = 0.7 held constant.
        The 0.4→0.9 ramp during cosine LR decay caused validation drift in
        triple_m1_aeri_iris_eyeball_500spc_run_20260423_101115 (val_angular
        12.4° → 15.3° from epoch 28 to 35). Hold α flat through fine-tune.
    """
    alpha_p1 = 0.4
    alpha_p2 = 0.7
    ramp_start_epoch = 16  # first epoch of Phase 2
    ramp_end_epoch = 19    # first epoch of constant α = alpha_p2

    if epoch < ramp_start_epoch:
        return alpha_p1
    if epoch < ramp_end_epoch:
        ratio = (epoch - ramp_start_epoch) / (ramp_end_epoch - ramp_start_epoch)
        return alpha_p1 + ratio * (alpha_p2 - alpha_p1)
    return alpha_p2


def get_phase_config(epoch):
    """Get loss weights and hyperparams for the current epoch (returns a copy)."""
    phase = get_phase(epoch)
    return dict(PHASE_CONFIG[phase])


def _unwrap_raynet(model):
    """Unwrap DDP / torch.compile to reach the underlying RayNetV5."""
    m = model
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def _rebuild_scheduler_for_resume(optimizer, start_epoch, log_fn=print):
    """
    v6.2.2: constant-LR mode. Resets each param group's LR to the
    target value derived from PHASE_CONFIG (re-applying the per-group
    multiplier set by ``build_param_groups``) and returns ``None``.

    Per the LR Management Rework, every phase now uses a constant
    learning rate with no scheduler. Cosine annealing's near-zero
    decay was the dominant source of the landmark-loss plateau and
    of the optimization-energy starvation observed when curriculum
    phases unfreeze new modules.

    The function is preserved (and still called by the resume + fork
    paths) only to rebroadcast the phase LR onto the optimizer —
    never to instantiate a scheduler.
    """
    cfg = get_phase_config(start_epoch)
    base_lr = cfg['lr']

    # Re-apply the per-group multiplier captured by build_param_groups.
    for pg in optimizer.param_groups:
        mult = pg.get('_lr_multiplier', 1.0)
        pg['lr'] = base_lr * mult
        pg['initial_lr'] = pg['lr']

    log_fn(f"  [scheduler] constant-LR mode (rework spec): "
           f"start_epoch={start_epoch}, base_lr={base_lr:.2e}, "
           f"groups={len(optimizer.param_groups)} (no scheduler)")
    return None


# ── Parameter-group LR multipliers (LR Rework, Phase 2 of doc) ──────
#
# Each module gets a base-LR multiplier matched to its optimization
# regime. Pretrained backbone weights need conservative updates;
# newly initialised heads need aggressive updates; cross-view
# attention is randomly initialised and must catch up to the
# already-warm gaze geometry head.
LR_MULTIPLIERS = {
    'backbone': 0.1,        # shared_stem + every *_branch.encoder
    'landmark': 0.2,        # FPN + heatmap/offset heads
    'pose':     0.5,        # interpolated between landmark and gaze
    'gaze':     1.0,        # GeometricGazeHead, IrisMeshHead, MacroGazeHead, ...
    'crossview': 1.5,       # CrossViewAttention + CameraEmbedding
}


def _classify_param(name: str) -> str:
    """Map a parameter's dotted name to one of the LR_MULTIPLIERS keys."""
    # Backbone weights live in shared_stem AND each *_branch.encoder
    # (the M3 s2+s3 stages); both share the pretrained-RepNeXt regime.
    if name.startswith('shared_stem'):
        return 'backbone'
    if name.startswith('landmark_branch.encoder'):
        return 'backbone'
    if name.startswith('pose_branch.encoder'):
        return 'backbone'
    if name.startswith('gaze_branch.encoder'):
        return 'backbone'
    # Branch-specific heads (everything in the branch except the encoder)
    if name.startswith('landmark_branch.'):
        return 'landmark'
    if name.startswith('pose_branch.'):
        return 'pose'
    if name.startswith('gaze_branch.'):
        return 'gaze'
    if name.startswith('cross_view_attn') or name.startswith('camera_embedding'):
        return 'crossview'
    # Fall back to gaze (1.0×) for any unclassified parameter — better
    # than silently dropping it.
    return 'gaze'


def build_param_groups(model, base_lr, multipliers=None):
    """
    Group ``model.named_parameters()`` by optimization regime and
    attach an ``_lr_multiplier`` to each group so resume/fork paths
    can rebroadcast the per-group LR after a checkpoint load.

    Only ``requires_grad=True`` parameters are included — frozen
    modules MUST NOT carry stale Adam moments across phase
    transitions (rework doc, "Optimizer Reinitialization Across
    Phases").

    Args:
        multipliers: optional dict overriding ``LR_MULTIPLIERS`` for
            this phase. Phase 1 (landmark warmup) passes an all-ones
            map to match the April reference run, where the entire
            model trained at the full ``cfg['lr']``. Later phases
            omit this and inherit the tiered defaults.

    Returns a list compatible with ``torch.optim.AdamW(param_groups)``.
    """
    mult_map = multipliers if multipliers is not None else LR_MULTIPLIERS
    buckets: dict[str, list] = {k: [] for k in mult_map}
    unwrapped = _unwrap_raynet(model)
    for name, p in unwrapped.named_parameters():
        if not p.requires_grad:
            continue
        cls = _classify_param(name)
        buckets[cls].append(p)
    groups = []
    for name, params in buckets.items():
        if not params:
            continue
        mult = mult_map[name]
        groups.append({
            'params': params,
            'lr': base_lr * mult,
            'initial_lr': base_lr * mult,
            '_lr_multiplier': mult,
            '_group_name': name,
        })
    return groups


def get_base_lr(optimizer) -> float:
    """Return the optimizer's *base* LR (the 1.0× / 'gaze' group).

    With v6.2.2 param-group multipliers, ``optimizer.param_groups[0]``
    is the *backbone* group (0.1× multiplier), so logging it as the
    canonical run LR was misleading — at base 1e-3 the CSV reported
    1e-4 because backbone is group 0. This helper resolves to the
    group whose multiplier is exactly 1.0 (the 'gaze' bucket); when no
    such group exists (e.g. a freeze pattern that excludes gaze) it
    falls back to the maximum LR across groups, which always equals
    the configured ``cfg['lr']`` because no multiplier exceeds the
    crossview 1.5× scaling for the same base LR.
    """
    base = None
    max_lr = 0.0
    for pg in optimizer.param_groups:
        mult = pg.get('_lr_multiplier', 1.0)
        max_lr = max(max_lr, pg['lr'] / max(mult, 1e-12))
        if abs(mult - 1.0) < 1e-9 and base is None:
            base = pg['lr']
    return base if base is not None else max_lr


def build_phase_optimizer(model, cfg, log_fn=print):
    """
    Build a fresh AdamW for the current phase from ``trainable_only``
    parameter groups with module-specific LR multipliers.

    Per the rework doc: rebuild on every phase transition so frozen
    modules don't carry stale Adam first/second moments and newly
    unfrozen modules don't inherit decayed LRs.

    Honors ``cfg['lr_multipliers']`` if present:
      - 'uniform' (string sentinel): all groups → 1.0× (April reference,
        Phase 1 landmark warmup runs the whole model at full base LR).
      - dict: explicit per-bucket overrides.
      - None / missing: inherit the tiered ``LR_MULTIPLIERS`` defaults.
    """
    raw_mult = cfg.get('lr_multipliers', None)
    if raw_mult == 'uniform':
        multipliers = {k: 1.0 for k in LR_MULTIPLIERS}
    else:
        multipliers = raw_mult  # dict or None
    groups = build_param_groups(model, base_lr=cfg['lr'],
                                multipliers=multipliers)
    opt = optim.AdamW(groups, betas=(0.5, 0.95), weight_decay=1e-4)
    if log_fn is not None:
        summary = ', '.join(
            f"{g['_group_name']}={g['lr']:.2e}({len(g['params'])}p)"
            for g in groups
        )
        log_fn(f"  [optimizer] rebuilt for phase (constant LR): "
               f"base_lr={cfg['lr']:.2e}; {summary}")
    return opt


def _filter_compatible_state(src_sd, target_sd):
    """
    Keep only (key, tensor) pairs from src_sd that have matching shape
    in target_sd. Returns (filtered_sd, dropped_keys). Used to bridge
    cross-stage forks where the model architecture changed between
    runs (e.g. old Stage 1 checkpoint into new eye-crop Stage 2 code).
    """
    filtered = {}
    dropped = []
    for k, v in src_sd.items():
        tgt = target_sd.get(k)
        if tgt is not None and hasattr(tgt, 'shape') and tgt.shape == v.shape:
            filtered[k] = v
        else:
            dropped.append(k)
    return filtered, dropped


def _optimizer_state_compatible(saved_state, new_optimizer):
    """
    Check whether a saved optimizer state_dict can be loaded into
    `new_optimizer`. PyTorch validates that every param_group has the
    same number of parameters as the saved group; if the model changed
    shape (e.g. new branches added) that assertion fails.

    Returns True iff groups match in count AND in per-group param count.
    """
    saved_groups = saved_state.get('param_groups', [])
    new_groups = new_optimizer.state_dict()['param_groups']
    if len(saved_groups) != len(new_groups):
        return False
    for sg, ng in zip(saved_groups, new_groups):
        if len(sg.get('params', [])) != len(ng.get('params', [])):
            return False
    return True


# ============== Training Loop ==============

def train_one_epoch(model, train_loader, optimizer, device, epoch, cfg,
                    scaler=None, grad_accum_steps=1, amp_enabled=False,
                    amp_dtype=torch.float16, batch_csv_writer=None,
                    n_views=1):
    """Run one training epoch with AMP and gradient accumulation support."""
    model.train()
    use_multiview = cfg.get('multiview', False)
    if not amp_enabled:
        amp_dtype = torch.float32

    running_losses = {
        'total': 0.0, 'landmark': 0.0, 'angular': 0.0, 'angular_deg': 0.0,
        'reproj': 0.0, 'mask': 0.0, 'ray_target': 0.0, 'pose': 0.0,
        'translation': 0.0,
        # V5-specific
        'eyeball_center': 0.0, 'pupil_center': 0.0, 'geom_angular': 0.0,
        'iris_seg': 0.0, 'eyeball_seg': 0.0,
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
        images = batch['image'].to(device, non_blocking=True)
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

        # Intrinsic-Delta face bbox GT for the MAGE BoxEncoder (v5 only).
        # Shape (B, 3) = [x_p ∈ [-1,1], y_p ∈ [-1,1], L_x > 0].
        face_bbox_gt = batch.get('face_bbox_gt')
        if face_bbox_gt is not None:
            face_bbox_gt = face_bbox_gt.to(device, non_blocking=True)

        # AERI ground-truth masks for segmentation supervision (uint8
        # {0, 255} @ 56x56).
        gt_iris_mask = batch.get('iris_mask')
        if gt_iris_mask is not None:
            gt_iris_mask = gt_iris_mask.to(device, non_blocking=True)
        gt_eyeball_mask = batch.get('eyeball_mask')
        if gt_eyeball_mask is not None:
            gt_eyeball_mask = gt_eyeball_mask.to(device, non_blocking=True)

        # 3DGazeNet M-target ground truth — iris contour mesh in CCS.
        # Falls back to None on shards that pre-date the schema bump
        # (the iris mesh + visual axis loss heads no-op when GT absent).
        gt_iris_mesh_3d = batch.get('iris_mesh_3d')
        if gt_iris_mesh_3d is not None:
            gt_iris_mesh_3d = gt_iris_mesh_3d.to(device, non_blocking=True)
        # Per-subject kappa rotation and GT visual axis for the kappa-
        # corrected visual-axis loss.
        R_kappa = batch.get('R_kappa')
        if R_kappa is not None:
            R_kappa = R_kappa.to(device, non_blocking=True)
        gt_visual_axis = batch.get('visual_axis')
        if gt_visual_axis is not None:
            gt_visual_axis = gt_visual_axis.to(device, non_blocking=True)

        mv_components = None
        # Forward pass with AMP autocast
        aeri_alpha = get_scheduled_alpha(epoch)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            predictions = model(images, n_views=n_views,
                                R_cam=R_cam, T_cam=T_cam,
                                face_bbox=face_bbox_gt,
                                aeri_alpha=aeri_alpha #AERI ALPHA
                                )

            pred_hm = predictions['landmark_heatmaps']
            pred_coords = predictions['landmark_coords']
            pred_gaze = predictions['gaze_vector']

            feat_H, feat_W = pred_hm.shape[2], pred_hm.shape[3]

            gt_eyeball = batch['eyeball_center_3d'].to(device, non_blocking=True)
            gt_pupil = batch.get('pupil_center_3d')
            if gt_pupil is not None:
                gt_pupil = gt_pupil.to(device, non_blocking=True)

            # Adding missing head gaze (gaze_c)
            gt_gaze_c = batch.get('gaze_c')
            if gt_gaze_c is not None:
                gt_gaze_c = gt_gaze_c.to(device, non_blocking=True)
                if torch.isnan(gt_gaze_c).any():
                    gt_gaze_c = None
            # Added missing eyeball gt
            gt_eyeball_radius = batch.get('eyeball_radius')
            if gt_eyeball_radius is not None:
                gt_eyeball_radius = gt_eyeball_radius.to(device, non_blocking=True)
                if torch.isnan(gt_eyeball_radius).any():
                    gt_eyeball_radius = None
            loss, components = total_loss(
                pred_hm, pred_coords, pred_gaze,
                gt_landmarks, gt_optical_axis,
                feat_H, feat_W,
                lam_lm=cfg['lam_lm'],
                lam_gaze=cfg['lam_gaze'],
                sigma=cfg['sigma'],
                lam_eyeball=cfg.get('lam_eyeball', 0.0),
                pred_eyeball=predictions.get('eyeball_center'),
                gt_eyeball=gt_eyeball,
                lam_pupil=cfg.get('lam_pupil', 0.0),
                pred_pupil=predictions.get('pupil_center'),
                gt_pupil=gt_pupil,
                lam_geom_angular=cfg.get('lam_geom_angular', 0.0),
                lam_ray=cfg.get('lam_ray', 0.0),
                eyeball_center=gt_eyeball,
                gaze_target=batch['gaze_target'].to(device, non_blocking=True),
                gaze_depth=batch['gaze_depth'].to(device, non_blocking=True),
                lam_pose=cfg.get('lam_pose', 0.0),
                pred_pose_6d=predictions.get('pred_pose_6d'),
                gt_head_R=gt_head_R,
                lam_trans=cfg.get('lam_trans', 0.0),
                pred_pose_t=predictions.get('pred_pose_t'),
                gt_head_t=gt_head_t,
                lam_iris_seg=cfg.get('lam_iris_seg', 0.0),
                pred_iris_mask_logits=predictions.get('iris_mask_logits'),
                gt_iris_mask=gt_iris_mask,
                lam_eyeball_seg=cfg.get('lam_eyeball_seg', 0.0),
                pred_eyeball_mask_logits=predictions.get('eyeball_mask_logits'),
                gt_eyeball_mask=gt_eyeball_mask,
                # 3DGazeNet M-target — iris contour mesh
                lam_iris_mesh=cfg.get('lam_iris_mesh', 0.0),
                pred_iris_mesh_3d=predictions.get('iris_mesh_3d'),
                gt_iris_mesh_3d=gt_iris_mesh_3d,
                lam_iris_edge=cfg.get('lam_iris_edge', 0.0),
                # Mean-of-two gaze fusion sub-supervisions
                lam_gaze_geom=cfg.get('lam_gaze_geom', 0.0),
                pred_gaze_geom=predictions.get('gaze_geom'),
                lam_gaze_direct=cfg.get('lam_gaze_direct', 0.0),
                pred_gaze_direct=predictions.get('gaze_direct'),
                # Visual-axis (kappa-corrected) supervision
                lam_gaze_visual=cfg.get('lam_gaze_visual', 0.0),
                pred_optical_axis=predictions.get('gaze_geom'),
                R_kappa=R_kappa,
                gt_visual_axis=gt_visual_axis,
                # Macro (head) gaze — GazeGene gaze_C paradigm
                lam_gaze_macro=cfg.get('lam_gaze_macro', 0.0),
                pred_gaze_macro=predictions.get('gaze_macro'),
                gt_gaze_c=gt_gaze_c,
                # Per-subject eyeball radius (R_s, cm)
                lam_eyeball_radius=cfg.get('lam_eyeball_radius', 0.0),
                pred_eyeball_radius=predictions.get('eyeball_radius'),
                gt_eyeball_radius=gt_eyeball_radius,
            )

            # Single-view pathway supervision: direct gaze L1 on pre-CrossViewAttn
            # features so the GeometricGazeHead is also optimized for val
            # (which bypasses CrossViewAttention entirely).
            lam_gaze_sv = cfg.get('lam_gaze_sv', 0.0)
            if lam_gaze_sv > 0:
                gaze_sv = predictions.get('gaze_vector_sv')
                if gaze_sv is not None:
                    from RayNet.losses import gaze_loss as _gaze_loss
                    sv_loss = _gaze_loss(gaze_sv, gt_optical_axis)
                    if torch.isfinite(sv_loss):
                        loss = loss + lam_gaze_sv * sv_loss

            # Multi-view consistency loss (Phase 2+)
            # Ray-based: works with unit gaze vectors, numerically stable.
            if use_multiview:
                mv_loss, mv_components = multiview_consistency_loss(
                    pred_gaze,
                    pred_coords,
                    R_cam,
                    lam_gaze_consist=cfg['lam_reproj'],
                    lam_shape=cfg['lam_mask'],
                    n_views=n_views,
                )
                if torch.isfinite(mv_loss):
                    # Smooth ramp from 0 → 1 over the first 5 epochs of P3
                    # (epochs 31-35). Multi-view consistency only fires when
                    # cfg['multiview'] is True, which is P3-only — so the
                    # ramp window is anchored to the P3 start (epoch 30).
                    mv_weight = min(1.0, max(0.0, (epoch - 30) / 5.0))
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
        if 'eyeball_center_loss' in components:
            running_losses['eyeball_center'] += components['eyeball_center_loss'].item()
        if 'pupil_center_loss' in components:
            running_losses['pupil_center'] += components['pupil_center_loss'].item()
        if 'geometric_angular_loss' in components:
            running_losses['geom_angular'] += components['geometric_angular_loss'].item()
        if 'iris_seg_loss' in components:
            running_losses['iris_seg'] += components['iris_seg_loss'].item()
        if 'eyeball_seg_loss' in components:
            running_losses['eyeball_seg'] += components['eyeball_seg_loss'].item()
        n_batches += 1

        # Per-batch CSV logging (high granularity)
        if batch_csv_writer is not None:
            ray_val = components.get('ray_target_loss',
                                     torch.tensor(0.0)).item()
            pose_val = components.get('pose_loss',
                                      torch.tensor(0.0)).item()
            trans_val = components.get('translation_loss',
                                       torch.tensor(0.0)).item()
            iris_seg_val = components.get('iris_seg_loss',
                                          torch.tensor(0.0)).item()
            eye_seg_val = components.get('eyeball_seg_loss',
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
                f"{iris_seg_val:.6f}",
                f"{eye_seg_val:.6f}",
                f"{get_base_lr(optimizer):.8f}",
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
    if not amp_enabled:
        amp_dtype = torch.float32

    running_losses = {
        'total': 0.0, 'landmark': 0.0, 'angular': 0.0, 'angular_deg': 0.0,
        'landmark_px': 0.0, 'pose': 0.0, 'ray': 0.0, 'translation': 0.0,
        'iris_seg': 0.0, 'eyeball_seg': 0.0,
    }
    n_batches = 0

    for batch in val_loader:

        #When using samples_per_subject in MDS shard loading we'll skip empty batches
        if not batch:
            continue

        images = batch['image'].to(device, non_blocking=True)
        gt_landmarks = batch['landmark_coords'].to(device, non_blocking=True)
        gt_optical_axis = batch['optical_axis'].to(device, non_blocking=True)
        gt_landmarks_px = batch['landmark_coords_px'].to(device, non_blocking=True)

        R_cam = batch['R_cam'].to(device, non_blocking=True)
        T_cam = batch['T_cam'].to(device, non_blocking=True)

        face_bbox_gt = batch.get('face_bbox_gt')
        if face_bbox_gt is not None:
            face_bbox_gt = face_bbox_gt.to(device, non_blocking=True)

        gt_iris_mask = batch.get('iris_mask')
        if gt_iris_mask is not None:
            gt_iris_mask = gt_iris_mask.to(device, non_blocking=True)
        gt_eyeball_mask = batch.get('eyeball_mask')
        if gt_eyeball_mask is not None:
            gt_eyeball_mask = gt_eyeball_mask.to(device, non_blocking=True)

        # New (v6) GT signals — gracefully None on legacy shards.
        # The streaming reader emits NaN-tensors when the new columns
        # (iris_mesh_3d, visual_axis) are missing from the shard; we
        # promote those to None here so total_loss skips them entirely.
        gt_iris_mesh_3d = batch.get('iris_mesh_3d')
        if gt_iris_mesh_3d is not None:
            gt_iris_mesh_3d = gt_iris_mesh_3d.to(device, non_blocking=True)
            if torch.isnan(gt_iris_mesh_3d).any():
                gt_iris_mesh_3d = None
        R_kappa = batch.get('R_kappa')
        if R_kappa is not None:
            R_kappa = R_kappa.to(device, non_blocking=True)
        gt_visual_axis = batch.get('visual_axis')
        if gt_visual_axis is not None:
            gt_visual_axis = gt_visual_axis.to(device, non_blocking=True)
            if torch.isnan(gt_visual_axis).any():
                gt_visual_axis = None
        gt_gaze_c = batch.get('gaze_c')
        if gt_gaze_c is not None:
            gt_gaze_c = gt_gaze_c.to(device, non_blocking=True)
            if torch.isnan(gt_gaze_c).any():
                gt_gaze_c = None
        gt_eyeball_radius = batch.get('eyeball_radius')
        if gt_eyeball_radius is not None:
            gt_eyeball_radius = gt_eyeball_radius.to(device, non_blocking=True)
            if torch.isnan(gt_eyeball_radius).any():
                gt_eyeball_radius = None

        aeri_alpha = get_scheduled_alpha(epoch)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            predictions = model(images, n_views=n_views,
                                R_cam=R_cam, T_cam=T_cam,
                                face_bbox=face_bbox_gt,
                                aeri_alpha=aeri_alpha,
                                )

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
                lam_iris_seg=cfg.get('lam_iris_seg', 0.0),
                pred_iris_mask_logits=predictions.get('iris_mask_logits'),
                gt_iris_mask=gt_iris_mask,
                lam_eyeball_seg=cfg.get('lam_eyeball_seg', 0.0),
                pred_eyeball_mask_logits=predictions.get('eyeball_mask_logits'),
                gt_eyeball_mask=gt_eyeball_mask,
                # 3DGazeNet M-target — iris contour mesh
                lam_iris_mesh=cfg.get('lam_iris_mesh', 0.0),
                pred_iris_mesh_3d=predictions.get('iris_mesh_3d'),
                gt_iris_mesh_3d=gt_iris_mesh_3d,
                lam_iris_edge=cfg.get('lam_iris_edge', 0.0),
                # Mean-of-two gaze fusion sub-supervisions
                lam_gaze_geom=cfg.get('lam_gaze_geom', 0.0),
                pred_gaze_geom=predictions.get('gaze_geom'),
                lam_gaze_direct=cfg.get('lam_gaze_direct', 0.0),
                pred_gaze_direct=predictions.get('gaze_direct'),
                # Visual-axis (kappa-corrected) supervision
                lam_gaze_visual=cfg.get('lam_gaze_visual', 0.0),
                pred_optical_axis=predictions.get('gaze_geom'),
                R_kappa=R_kappa,
                gt_visual_axis=gt_visual_axis,
                # Macro (head) gaze — GazeGene gaze_C paradigm
                lam_gaze_macro=cfg.get('lam_gaze_macro', 0.0),
                pred_gaze_macro=predictions.get('gaze_macro'),
                gt_gaze_c=gt_gaze_c,
                # Per-subject eyeball radius (R_s, cm)
                lam_eyeball_radius=cfg.get('lam_eyeball_radius', 0.0),
                pred_eyeball_radius=predictions.get('eyeball_radius'),
                gt_eyeball_radius=gt_eyeball_radius,
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
        iris_seg_val = components.get('iris_seg_loss')
        if iris_seg_val is not None:
            running_losses['iris_seg'] += iris_seg_val.item()
        eye_seg_val = components.get('eyeball_seg_loss')
        if eye_seg_val is not None:
            running_losses['eyeball_seg'] += eye_seg_val.item()
        n_batches += 1

    for k in running_losses:
        running_losses[k] /= max(n_batches, 1)

    return running_losses


# ============== Main Training ==============

def _build_run_config(args, hw):
    """Collect all training configuration into a single dict for metadata."""
    return {
        'profile': args.profile,
        'epochs': args.epochs,
        'eye': args.eye,
        'hardware': hw,
        'phase_config': {
            str(k): {kk: vv for kk, vv in v.items() if kk != 'description'}
            for k, v in PHASE_CONFIG.items()
        },
        'data_dir': getattr(args, 'data_dir', None),
        'core_backbone_weight_path': args.core_backbone_weight_path,
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
    # Auto-cap --epochs to the last epoch defined in PHASE_CONFIG
    stage_max_epoch = max(cfg['epochs'][1] for cfg in PHASE_CONFIG.values())
    if args.epochs > stage_max_epoch:
        print(f"[auto-cap] --epochs={args.epochs} exceeds the phase "
              f"schedule max epoch ({stage_max_epoch}). Capping to "
              f"{stage_max_epoch}.")
        args.epochs = stage_max_epoch

    # HuggingFace Accelerate — handles DDP wrapping, rank-aware dataloaders,
    # and multi-node orchestration. AMP (autocast + GradScaler) is still
    # managed manually below; Accelerator is configured with
    # mixed_precision='no' so it does not interfere.
    accelerator = build_accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"Device: {device}")
        print(f"Distributed: num_processes={accelerator.num_processes}, "
              f"process_index={accelerator.process_index}")

    # Apply hardware profile
    hw = apply_hardware_profile(args)
    if is_main:
        setup_hardware(hw, device)
    else:
        # Non-main processes still need TF32 flags on their own CUDA context,
        # but skip the GPU-info print.
        if hw['tf32'] and device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')

    # Resolve AMP dtype from profile string
    amp_dtype = AMP_DTYPE_MAP.get(hw['amp_dtype'], torch.float16)

    if is_main:
        print(f"Profile: {args.profile}")
        print(f"  Batch size (per-process): {hw['batch_size']} "
              f"(effective/proc: {hw['batch_size'] * hw['grad_accum_steps']}, "
              f"global: {hw['batch_size'] * hw['grad_accum_steps'] * accelerator.num_processes})")
        print(f"  AMP: {hw['amp']} ({hw['amp_dtype']})")
        print(f"  Gradient accumulation: {hw['grad_accum_steps']} steps")
        print(f"  torch.compile: {hw['compile_model']}")

    # In distributed mode, each rank would otherwise auto-generate its own
    # timestamp inside CheckpointManager and the paths would diverge.
    # Generate on rank 0 and broadcast so every rank writes to the same
    # checkpoint directory.
    if (accelerator.num_processes > 1 and args.ckpt_bucket
            and not args.run_id):
        from accelerate.utils import broadcast_object_list
        run_id_holder = [
            datetime.now().strftime('run_%Y%m%d_%H%M%S') if is_main else None
        ]
        broadcast_object_list(run_id_holder, from_process=0)
        args.run_id = run_id_holder[0]

    # Checkpoint manager (MinIO) — created on every rank so that resume
    # loads identical optimizer state across ranks. Save/upload calls are
    # gated on is_main_process further down.
    ckpt_mgr = _init_checkpoint_manager(args)
    if ckpt_mgr is not None and is_main:
        print(f"  MinIO checkpoints: s3://{args.ckpt_bucket}/{args.ckpt_prefix}/{ckpt_mgr.run_id}/")

    # Create output directory — main process only. All log and local
    # checkpoint writes are gated on is_main, so ranks without an
    # output_dir simply skip those writes.
    if is_main:
        if ckpt_mgr is not None:
            output_dir = os.path.join(args.output_dir, ckpt_mgr.run_id)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    # Create model — backbone is selectable (m1 / m3). v6.2 default is m3
    # for medical-grade gaze; pass --backbone m1 for ablation A/Bs.
    model = create_raynet_v5(
        backbone_weight_path=args.core_backbone_weight_path,
        n_landmarks=14,
        backbone=args.backbone,
    )

    if hw['compile_model'] and hasattr(torch, 'compile'):
        if is_main:
            print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # AMP scaler — ONLY for float16. BF16 has FP32 range, so loss scaling is
    # unnecessary and actively harmful (the scaler misdetects inf gradients
    # and silently skips every optimizer step, freezing training at init).
    use_scaler = hw['amp'] and hw.get('amp_dtype', 'float16') == 'float16'
    scaler = GradScaler('cuda', enabled=True) if use_scaler else None

    # Wrap model in DDP (no-op on single process). accelerator.prepare
    # also broadcasts rank-0 weights to all ranks, so after this call all
    # ranks start from identical parameters.
    model = accelerator.prepare(model)

    # --- Data loading ---
    # Multi-view is always active. Train + val both use 9-grouped batches so
    # CrossViewAttention, cam_embed, and PoseEncoder fusion are exercised at
    # validation (without this, val gaze metrics are an uninformative
    # single-view lower bound).
    #
    # For MDS streaming: StreamingDataset shards by RANK/WORLD_SIZE env
    # vars that `accelerate launch` sets, so we DON'T wrap it.
    # For local loaders: accelerator.prepare injects a DistributedSampler.
    if args.mds_streaming:
        train_loader_mv, val_loader = _create_mds_mv_loader(args, hw)
    else:
        _, train_loader_mv, val_loader = _create_local_loaders(args, hw)
        train_loader_mv, val_loader = accelerator.prepare(
            train_loader_mv, val_loader)

    # Record config in checkpoint metadata (main only — no rank divergence).
    if is_main:
        run_config = _build_run_config(args, hw)
        if ckpt_mgr is not None:
            ckpt_mgr.set_config(run_config)

    # CSV loggers — main process only. Non-main ranks keep these as None
    # and all write sites check `is not None` before writing.
    #
    # Append mode when resuming so the previous run's history survives
    # across restarts. The header is only written if the file is new
    # (or empty) — otherwise we'd inject a header row mid-file which
    # would break downstream parsers. `is_resuming` covers --resume,
    # --fork_from (continuation), and --warmstart_from cases where a
    # prior log may already exist under the same output_dir.
    csv_path = batch_csv_path = None
    csv_file = batch_csv_file = None
    csv_writer = batch_csv_writer = None
    is_resuming = bool(args.resume or args.fork_from or args.warmstart_from)
    if is_main:
        csv_path = os.path.join(output_dir, 'training_log.csv')
        batch_csv_path = os.path.join(output_dir, 'batch_log.csv')

        # On --resume, hydrate local log files from MinIO before opening
        # in append mode. Without this, resuming on a fresh machine
        # would write a single-row "new" log and the next upload would
        # overwrite the run's full history. Only --resume continues
        # under the same run_id; --fork_from / --warmstart_from create
        # a new run_id with no prior log to preserve.
        if args.resume and ckpt_mgr is not None:
            for local_path in (csv_path, batch_csv_path):
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    continue
                log_key = (f"{args.ckpt_prefix}/{ckpt_mgr.run_id}/"
                           f"{os.path.basename(local_path)}")
                try:
                    ckpt_mgr._client.fget_object(
                        args.ckpt_bucket, log_key, local_path)
                    print(f"  Hydrated {os.path.basename(local_path)} "
                          f"from MinIO ({os.path.getsize(local_path)} bytes)")
                except Exception as e:
                    print(f"  No prior {os.path.basename(local_path)} "
                          f"in MinIO ({e}); starting fresh log")

        write_header = not (is_resuming
                            and os.path.exists(csv_path)
                            and os.path.getsize(csv_path) > 0)
        csv_file = open(csv_path, 'a' if is_resuming else 'w', newline='')
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow([
                'epoch', 'phase', 'lr',
                'train_total', 'train_landmark', 'train_angular_deg',
                'train_reproj', 'train_mask', 'train_ray_target', 'train_pose',
                'train_translation', 'train_iris_seg', 'train_eyeball_seg',
                'val_total', 'val_landmark', 'val_angular_deg', 'val_landmark_px',
                'val_iris_seg', 'val_eyeball_seg',
            ])

        write_batch_header = not (is_resuming
                                   and os.path.exists(batch_csv_path)
                                   and os.path.getsize(batch_csv_path) > 0)
        batch_csv_file = open(batch_csv_path,
                              'a' if is_resuming else 'w', newline='')
        batch_csv_writer = csv.writer(batch_csv_file)
        if write_batch_header:
            batch_csv_writer.writerow([
                'epoch', 'batch', 'loss', 'landmark', 'angular_deg',
                'gaze_consist', 'shape', 'ray_target', 'pose', 'translation',
                'iris_seg', 'eyeball_seg', 'lr',
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
        if is_main:
            print(f"Resuming from run {ckpt_mgr.run_id} ...")
        # We need an optimizer to exist before resume_state. Build it
        # with the v6.2.2 param-group multipliers; the per-group LR is
        # rebroadcast from current PHASE_CONFIG by
        # `_rebuild_scheduler_for_resume` below.
        optimizer = build_phase_optimizer(
            model, get_phase_config(1),
            log_fn=(print if is_main else (lambda *_: None)),
        )

        resume_file = getattr(args, 'resume_from', None)
        # resume_state loads into the unwrapped model so DDP wrapping
        # doesn't interfere with state_dict key names.
        unwrapped = accelerator.unwrap_model(model)
        # Serialize the MinIO download: rank 0 downloads the checkpoint
        # to the local cache first, then non-main ranks cache-hit. Doing
        # this concurrently races on MinIO's .part.minio temp file.
        with accelerator.main_process_first():
            start_epoch, resume_ckpt = ckpt_mgr.resume_state(
                unwrapped, optimizer, scheduler=None, scaler=scaler,
                map_location=device, filename=resume_file,
            )
        # Always rebuild the scheduler from the *current* PHASE_CONFIG
        # so changes to phase epoch budgets between runs take effect.
        scheduler = _rebuild_scheduler_for_resume(
            optimizer, start_epoch,
            log_fn=(print if is_main else (lambda *_: None)),
        )
        current_phase = get_phase(resume_ckpt['epoch'])
        best_val_loss = resume_ckpt.get('val_metrics', {}).get('total', best_val_loss)
        if is_main:
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

    # --- Fork from a different run (new run_id, FULL training state) ---
    # Like --resume, but writes under a fresh run_id so the source run is
    # never overwritten. Loads model + optimizer + scheduler + scaler +
    # epoch counter + best_val_loss, then continues training under
    # ckpt_mgr.run_id (auto-generated unless --run_id was passed).
    # Use this to extend training past the original epoch budget or to
    # branch off for hyperparameter variations while keeping the baseline
    # intact.
    if args.fork_from and ckpt_mgr is not None:
        if is_main:
            print(f"Forking from run {args.fork_from} "
                  f"({args.fork_checkpoint}) into new run {ckpt_mgr.run_id} ...")
        # Serialize MinIO download across ranks — see note at the resume
        # site above for the race condition this avoids.
        with accelerator.main_process_first():
            fork_state = ckpt_mgr.load_from_run(
                source_run_id=args.fork_from,
                filename=args.fork_checkpoint,
                map_location=device,
            )

        if 'optimizer_state_dict' not in fork_state:
            raise RuntimeError(
                f"--fork_from: checkpoint '{args.fork_checkpoint}' from run "
                f"{args.fork_from} has no optimizer_state_dict. Pick a "
                f"checkpoint that carries full state (e.g. latest.pt or a "
                f"periodic checkpoint_epoch*.pt). If you only want model "
                f"weights, use --warmstart_from instead."
            )

        target = accelerator.unwrap_model(model)
        # Drop any tensors whose shape changed between the source run
        # and the current architecture (e.g. an old Stage 1 checkpoint
        # carries a 2-input FusionBlock — the new 3-input GazeFusionBlock
        # has a larger first linear layer). strict=False alone doesn't
        # handle shape mismatches; we filter by shape here so the fork
        # survives architecture migrations.
        filtered_sd, shape_dropped = _filter_compatible_state(
            fork_state['model_state_dict'], target.state_dict())
        missing, unexpected = target.load_state_dict(
            filtered_sd, strict=False)
        if is_main:
            if missing:
                print(f"  [fork] missing keys: {len(missing)} "
                      f"(first: {missing[:3]})")
            if unexpected:
                print(f"  [fork] unexpected keys: {len(unexpected)} "
                      f"(first: {unexpected[:3]})")
            if shape_dropped:
                print(f"  [fork] shape-mismatch keys dropped: "
                      f"{len(shape_dropped)} (first: {shape_dropped[:3]})")

        src_epoch = fork_state['epoch']

        if args.fork_reset_epoch:
            # Explicit reset: restart from epoch 1 under the current
            # phase schedule. Optimizer m/v state is preserved (the
            # whole point of fork vs warmstart); scheduler is rebuilt
            # fresh by the phase-transition block on the first loop
            # iteration (current_phase=0 forces that transition to fire).
            start_epoch = 1
            current_phase = 0
            phase1_cfg = get_phase_config(1)
            optimizer = build_phase_optimizer(
                model, phase1_cfg,
                log_fn=(print if is_main else (lambda *_: None)),
            )
            # The param count may not match after an architecture change.
            # PyTorch's optimizer.load_state_dict asserts per-group param
            # count equality, so we check first and fall back to a fresh
            # optimizer when the shapes diverge.
            optimizer_preserved = _optimizer_state_compatible(
                fork_state['optimizer_state_dict'], optimizer)
            if optimizer_preserved:
                optimizer.load_state_dict(
                    fork_state['optimizer_state_dict'])
            # Scheduler intentionally left None; rebuilt in-loop.
            scheduler = None
            if scaler is not None and 'scaler_state_dict' in fork_state:
                scaler.load_state_dict(fork_state['scaler_state_dict'])
            # Don't inherit src best_val_loss — it was measured under a
            # different loss composition and isn't comparable.
            if is_main:
                opt_status = ("preserved"
                              if optimizer_preserved
                              else "RESET (param count mismatch — "
                                   "architecture changed)")
                print(f"  Epoch counter reset to 1 (--fork_reset_epoch). "
                      f"Optimizer momentum: {opt_status}; scheduler "
                      f"and best_val_loss rebuilt for phase 1.")
        else:
            # Continuation fork: resume from src_epoch under the same
            # phase map, restoring full state.
            start_epoch = src_epoch + 1
            current_phase = get_phase(src_epoch)
            fork_phase_cfg = get_phase_config(src_epoch)

            optimizer = build_phase_optimizer(
                model, fork_phase_cfg,
                log_fn=(print if is_main else (lambda *_: None)),
            )

            # Same guard as the cross-stage path — if the model changed
            # shape the saved optimizer state can't load.
            if _optimizer_state_compatible(
                    fork_state['optimizer_state_dict'], optimizer):
                optimizer.load_state_dict(
                    fork_state['optimizer_state_dict'])
            elif is_main:
                print("  [fork] optimizer param count mismatch — "
                      "starting with fresh AdamW state.")
            # Scheduler is rebuilt from current PHASE_CONFIG below, NOT
            # loaded from the fork checkpoint — same stale-T_max bug as
            # the resume path. See _rebuild_scheduler_for_resume.
            scheduler = _rebuild_scheduler_for_resume(
                optimizer, start_epoch,
                log_fn=(print if is_main else (lambda *_: None)),
            )
            if scaler is not None and 'scaler_state_dict' in fork_state:
                scaler.load_state_dict(fork_state['scaler_state_dict'])

            best_val_loss = fork_state.get(
                'val_metrics', {}).get('total', best_val_loss)
            if is_main:
                print(f"  Fork: loaded full state from epoch "
                      f"{src_epoch} (phase {current_phase}). New run "
                      f"continues at epoch {start_epoch}, "
                      f"best_val_loss={best_val_loss:.4f}")

            if start_epoch > args.epochs:
                raise RuntimeError(
                    f"Cannot fork: source checkpoint is at epoch "
                    f"{src_epoch}, so training would start at epoch "
                    f"{start_epoch}, but --epochs={args.epochs}. Increase "
                    f"--epochs to extend training past the fork point."
                )

    # --- Warmstart from a different run (e.g. Stage 1 -> Stage 2) ---
    # Loads ONLY model weights. Optimizer, scheduler, epoch counter, and
    # scaler stay fresh. A new run_id has already been generated by the
    # checkpoint manager (since --run_id is forbidden with --warmstart_from).
    if args.warmstart_from and ckpt_mgr is not None:
        if is_main:
            print(f"Warmstarting from run {args.warmstart_from} "
                  f"({args.warmstart_checkpoint}) into new run {ckpt_mgr.run_id} ...")
        # Serialize MinIO download across ranks — see note at the resume
        # site above for the race condition this avoids.
        with accelerator.main_process_first():
            ws_state = ckpt_mgr.load_from_run(
                source_run_id=args.warmstart_from,
                filename=args.warmstart_checkpoint,
                map_location=device,
            )
        target = accelerator.unwrap_model(model)
        # Filter by shape to survive architecture migrations (see fork block).
        filtered_sd, shape_dropped = _filter_compatible_state(
            ws_state['model_state_dict'], target.state_dict())
        missing, unexpected = target.load_state_dict(
            filtered_sd, strict=False)
        if is_main:
            if missing:
                print(f"  [warmstart] missing keys: {len(missing)} "
                      f"(first: {missing[:3]})")
            if unexpected:
                print(f"  [warmstart] unexpected keys: {len(unexpected)} "
                      f"(first: {unexpected[:3]})")
            if shape_dropped:
                print(f"  [warmstart] shape-mismatch keys dropped: "
                      f"{len(shape_dropped)} (first: {shape_dropped[:3]})")
            src_epoch = ws_state.get('epoch', '?')
            print(f"  Loaded weights from epoch {src_epoch}. "
                  f"Starting fresh optimizer at epoch 1.")
        # start_epoch stays at 1, optimizer/scheduler stay None — they will
        # be created in the phase-transition block below exactly like a
        # from-scratch run.

        # Default behavior : start from epoch 1
        start_epoch = 1
        current_phase = 0

        # override with warmstart_phase if provided
        if args.warmstart_phase is not None:
            if args.warmstart_phase not in PHASE_CONFIG:
                raise ValueError(
                    f"--warmstart_phase={args.warmstart_phase} is invalid. "
                    f"Valid phases: {list(PHASE_CONFIG.keys())}"
                )

            phase_cfg = PHASE_CONFIG[args.warmstart_phase]
            phase_start, _ = phase_cfg['epochs']

            start_epoch = phase_start
            current_phase = 0  # force phase transition logic to trigger

            if is_main:
                print(f"  Warmstart phase override: starting from "
                      f"phase {args.warmstart_phase} (epoch {start_epoch})")


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
            if is_main:
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

            # Branch-staged freeze: requires_grad_(False) + .eval() the
            # submodules named by cfg['freeze_set']; re-enable everything
            # else. Operates on the unwrapped RayNetV5 so DDP / compile
            # wrappers don't intercept the call.
            freeze_set = cfg.get('freeze_set', 'none')
            apply_phase_freeze(
                _unwrap_raynet(model),
                freeze_set,
                log_fn=(print if is_main else (lambda *_: None)),
            )

            phase_start, phase_end = cfg['epochs']

            # v6.2.2 — LR Management Rework: rebuild a *fresh* AdamW on
            # every phase transition over only the trainable params,
            # using module-specific LR multipliers. This eliminates two
            # historic failure modes:
            #   1. Stale Adam moments on previously-frozen modules
            #      driving destabilised gradients on unfreeze.
            #   2. Newly-unfrozen modules inheriting cosine-decayed LRs
            #      from the previous phase.
            # The replaced "carry-over" path optimised for momentum
            # continuity but routinely hurt the very modules a phase
            # transition is meant to adapt — see the rework doc for
            # the diagnosis.
            optimizer = build_phase_optimizer(
                model, cfg,
                log_fn=(print if is_main else (lambda *_: None)),
            )

            # Constant-LR mode (rework Preferred Option for P1/P3, plus
            # the user's override extending it to P2). No scheduler is
            # constructed; per-epoch scheduler.step() is gated below.
            scheduler = None

        active_train_loader = train_loader_mv
        active_val_loader = val_loader

        # n_views for CrossViewAttention during TRAINING.
        # NOTE: this is independent of cfg['multiview']. CrossViewAttention
        # runs whenever mv_groups > 1 (which produces 9-grouped batches).
        # cfg['multiview'] only gates the auxiliary multiview_consistency_loss
        # inside train_one_epoch — see header comment on PHASE_CONFIG above.
        # Pass --no_multiview on the CLI to truly ablate cross-view fusion.
        #
        # Phase 2 is monocular by design: the gaze branch trains on a
        # single-view objective and CrossViewAttention is short-circuited
        # so its parameters do not drift on identity-only batches. Force
        # n_views=1 there regardless of cfg['multiview'] (defense in depth).
        if phase == 2 or args.no_multiview:
            active_n_views = 1
        else:
            active_n_views = TRICAM_N_VIEWS

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
        if batch_csv_file is not None:
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

        # Step scheduler (constant-LR mode: scheduler is None; no-op).
        if scheduler is not None:
            scheduler.step()
        # v6.2.2: report the BASE LR (gaze 1.0× group), not group 0
        # which is the backbone bucket at 0.1× the configured cfg['lr'].
        current_lr = get_base_lr(optimizer)

        # All ranks finish the epoch before the main process saves / uploads.
        accelerator.wait_for_everyone()

        if is_main:
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

            if device.type == 'cuda':
                gpu_idx = device.index if device.index is not None else 0
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                total_gb = torch.cuda.get_device_properties(
                    gpu_idx).total_memory / 1e9
                headroom_gb = total_gb - peak_gb
                print(f"  GPU mem (rank 0): peak {peak_gb:.1f} / "
                      f"total {total_gb:.1f} GB  "
                      f"(headroom {headroom_gb:.1f} GB)")
                # Reset so each epoch reports its own peak; the worst
                # epoch over the run is what determines the safe batch.
                torch.cuda.reset_peak_memory_stats()

            csv_writer.writerow([
                epoch, phase, f"{current_lr:.2e}",
                f"{train_losses['total']:.6f}",
                f"{train_losses['landmark']:.6f}",
                f"{train_losses['angular_deg']:.4f}",
                f"{train_losses.get('reproj', 0):.6f}",
                f"{train_losses.get('mask', 0):.6f}",
                f"{train_losses.get('ray_target', 0):.6f}",
                f"{train_losses.get('pose', 0):.6f}",
                f"{train_losses.get('translation', 0):.6f}",
                f"{train_losses.get('iris_seg', 0):.6f}",
                f"{train_losses.get('eyeball_seg', 0):.6f}",
                f"{val_losses['total']:.6f}",
                f"{val_losses['landmark']:.6f}",
                f"{val_losses['angular_deg']:.4f}",
                f"{val_losses['landmark_px']:.4f}",
                f"{val_losses.get('iris_seg', 0):.6f}",
                f"{val_losses.get('eyeball_seg', 0):.6f}",
            ])
            csv_file.flush()

            # Unwrap DDP (and torch.compile) before handing the raw module to
            # the checkpoint code; its state_dict is what's serialised.
            save_model = accelerator.unwrap_model(model)

            # Build the extras blob once per epoch — picked up by both
            # save_best and the periodic save below.
            ckpt_extras = {'profile': args.profile}

            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']

                if ckpt_mgr is not None:
                    ckpt_mgr.save_best(
                        epoch=epoch, model=save_model, val_loss=best_val_loss,
                        val_metrics=val_losses, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler,
                        extra=ckpt_extras,
                    )
                else:
                    save_dict = {
                        'epoch': epoch,
                        'phase': phase,
                        'model_state_dict': save_model.state_dict(),
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
                    epoch=epoch, model=save_model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, phase=phase,
                    train_metrics=train_losses, val_metrics=val_losses,
                    tag='latest', extra=ckpt_extras,
                )
                # Periodic named checkpoint
                if epoch % args.ckpt_every == 0:
                    ckpt_mgr.save(
                        epoch=epoch, model=save_model, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, phase=phase,
                        train_metrics=train_losses, val_metrics=val_losses,
                        extra=ckpt_extras,
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
                        'model_state_dict': save_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    if scaler is not None:
                        save_dict['scaler_state_dict'] = scaler.state_dict()
                    torch.save(save_dict, os.path.join(output_dir, f'checkpoint_epoch{epoch}.pt'))

        # Broadcast best_val_loss from main so every rank stays in sync on
        # the "was this a new best?" question. For DDP this doesn't affect
        # training state (only main saves), but keeping the value consistent
        # makes the resume path simpler.
        accelerator.wait_for_everyone()

    if is_main:
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
        camera_ids=list(TRICAM_IDS),
        img_size=args.img_size,
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

    from torchvision import transforms as T

    # Normalize to ImageNet stats (applied to BOTH train and val so that
    # the shared stem's BN running_mean/var — calibrated on ColorJitter'd
    # training images — match the activation scale seen at val time.
    # Without this, the jitter ±40% brightness/contrast widens the training
    # pixel variance relative to clean val images; the stem's BN running_var
    # grows to absorb that extra variance, then attenuates clean val
    # activations by the corresponding factor — degrading both landmark and
    # gaze metrics progressively with each epoch.
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    train_transform = T.Compose([
        T.ColorJitter(brightness=0.4, contrast=0.4,
                      saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        normalize,
    ])
    val_transform = normalize

    # TriCam: keep only cams in TRICAM_IDS. The streaming loader fetches
    # raw 9-per-group windows from shards (preserved order), then drops
    # samples whose cam_id is outside the subset — so the effective
    # batch carries TRICAM_N_VIEWS cams per group, matching active_n_views.
    print(f"Creating multi-view MDS streaming loaders (train + val), "
          f"TriCam={TRICAM_IDS}...")
    train_loader_mv, val_loader_mv = create_multiview_streaming_dataloaders(
        remote_train=args.mds_train,
        remote_val=args.mds_val,
        local_cache=os.path.join(args.output_dir, 'mds_cache_mv'),
        mv_groups=hw['mv_groups'],
        num_workers=hw['num_workers'],
        transform=train_transform,
        val_transform=val_transform,
        pin_memory=hw['pin_memory'],
        prefetch_factor=hw['prefetch_factor'],
        persistent_workers=hw['persistent_workers'],
        samples_per_subject=args.samples_per_subject,
        eyelid_occlusion_p=args.eyelid_occlusion_p,
        tricam_ids=TRICAM_IDS,
    )
    return train_loader_mv, val_loader_mv


# ============== CLI ==============

def parse_args():
    parser = argparse.ArgumentParser(description='RayNet v5 Training')

    # Data
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to GazeGene dataset (local mode)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--samples_per_subject', type=int, default=200)
    parser.add_argument('--eye', type=str, default='L', choices=['L', 'R'])
    parser.add_argument('--img_size', type=int, default=224,
                        choices=[224, 448],
                        help='Face-crop side length. v6.2 target is 448 — '
                             'aligns with native GazeGene scale and unblocks '
                             'sub-pixel iris precision. NOTE: requires a '
                             'reshard at 448; existing 224 shards keep working '
                             'with --img_size 224.')
    parser.add_argument('--eyelid_occlusion_p', type=float, default=0.30,
                        help='Per-sample probability of synthetic eyelid '
                             'occlusion in the train transform. The GT '
                             'eyeball/iris masks are NOT modified — the '
                             'AERI head is supervised to predict the full '
                             'silhouette through the occluder, which is '
                             'what gives OpenFace-style robustness to '
                             'partial blinks at inference. 0.0 disables.')

    # MDS streaming (MosaicML + MinIO)
    parser.add_argument('--mds_streaming', action='store_true',
                        help='Stream training data from MDS shards on MinIO/S3')
    parser.add_argument('--mds_train', type=str, default=None,
                        help='MDS shard URL for training (e.g. s3://gazegene/train)')
    parser.add_argument('--mds_val', type=str, default=None,
                        help='MDS shard URL for validation (e.g. s3://gazegene/val)')

    # Model — v6.2 uses RepNeXt-M3 for all branches by default
    parser.add_argument('--backbone', type=str, default='m3',
                        choices=['m1', 'm3'],
                        help='RepNeXt variant for the four backbone instances '
                             '(shared + landmark + gaze + pose). v6.2 default is '
                             'M3 (embed_dim=64,128,256,512) — needed for the '
                             'GLOBAL/FOVEAL_FLOOR=0.5 occlusion robustness and '
                             '448-resolution sub-pixel iris precision. M1 '
                             '(embed_dim=48,96,192,384) is preserved for ablation.')
    parser.add_argument('--core_backbone_weight_path', type=str, default=None,
                        help='Path to pretrained RepNeXt weights '
                             '(loaded into all 4 backbone instances). '
                             'Pass None / empty to train from random init.')

    # Hardware profile
    parser.add_argument('--profile', type=str, default='default',
                        choices=list(HARDWARE_PROFILES.keys()),
                        help=f'Hardware profile ({", ".join(HARDWARE_PROFILES.keys())})')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile even if profile enables it')
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
                        help='Multi-view batch groups. Raw fetch = mv_groups * 9 '
                             '(shards are stored 9-per-group); after the TriCam '
                             f'filter ({TRICAM_IDS}) the effective batch size is '
                             f'mv_groups * {TRICAM_N_VIEWS}.')
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
                             '(same run, continues epoch counter)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from a specific checkpoint file '
                             '(e.g. checkpoint_epoch5.pt, best_model.pt)')
    parser.add_argument('--warmstart_from', type=str, default=None,
                        help='Warmstart from a DIFFERENT run. Loads ONLY model '
                             'weights — optimizer/scheduler/epoch are reset. '
                             'A new run_id is auto-generated.')
    parser.add_argument('--warmstart_checkpoint', type=str, default='best_model.pt',
                        help='Checkpoint file to pull from --warmstart_from run '
                             '(default: best_model.pt)')
    parser.add_argument('--warmstart_phase', type=int, default=None,
                        help='Start warmstart from a specific phase (overrides epoch=1). '
                             'Optimizer is NOT loaded (unlike fork).')
    parser.add_argument('--fork_from', type=str, default=None,
                        help='Fork from a previous run: load FULL training '
                             'state (model + optimizer + scheduler + scaler '
                             '+ epoch + best_val_loss) from the source run, '
                             'but write new checkpoints under a NEW run_id '
                             'so the source run is not overwritten. Use to '
                             'extend training past the original epoch budget '
                             'or branch off for hyperparameter variations '
                             'without corrupting the baseline. A new run_id '
                             'is auto-generated unless --run_id is passed.')
    parser.add_argument('--fork_checkpoint', type=str, default='latest.pt',
                        help='Checkpoint file to pull from --fork_from run '
                             '(default: latest.pt — has full optimizer/'
                             'scheduler state). best_model.pt also works if '
                             'it was saved with optimizer state.')
    parser.add_argument('--fork_reset_epoch', action='store_true',
                        help='Force fork to reset epoch counter to 1 under '
                             'the current phase schedule. Optimizer momentum '
                             'and scaler state are still preserved.')
    args = parser.parse_args()

    if not args.mds_streaming and args.data_dir is None:
        parser.error("--data_dir is required when not using --mds_streaming")

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
                     "--warmstart_from to start a new run with weights "
                     "from another run.")
    if args.warmstart_from and not args.ckpt_bucket:
        parser.error("--warmstart_from requires --ckpt_bucket")
    if args.warmstart_from and args.run_id:
        parser.error("--warmstart_from creates a new run — do not pass --run_id "
                     "(a fresh timestamped run_id will be generated).")
    if args.fork_from:
        if args.resume or args.warmstart_from:
            parser.error("--fork_from is mutually exclusive with --resume "
                         "and --warmstart_from. Use --resume to continue the "
                         "same run in place; --warmstart_from to start fresh "
                         "with weights only; --fork_from to branch the "
                         "full training state into a new run_id.")
        if not args.ckpt_bucket:
            parser.error("--fork_from requires --ckpt_bucket")
        if args.run_id and args.run_id == args.fork_from:
            parser.error("--run_id cannot equal --fork_from — that would "
                         "overwrite the source run. Pick a different "
                         "--run_id or omit it to auto-generate.")

    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)
