"""
OpenEDS-only training entry point (segmenter + TCN).

Decoupled from the GazeGene `RayNet.train` script so the two paths
share no optimiser state or freeze logic. The GazeGene shared stem
remains untouched by anything in this module — gradients from the
OpenEDS path can never reach the synthetic-trained skeleton.

Two staged objectives:

    Stage S1 (segmenter):
        4-class semantic segmentation on ~62k labelled OpenEDS frames.
        Outputs feed the torsion estimator and the TCN's per-frame
        feature vector.

    Stage S2 (TCN):
        Per-frame feature vectors → TCN → smoothed gaze, torsion
        residual, blink logit, movement class. Trained on per-subject
        sequence windows; pseudo-labels for blink and saccade derived
        from segmenter output (no OpenEDS GT for either).

This script is the high-level harness; intentionally skinny because
the OpenEDS dataset path is *not* present on the host where the
GazeGene shards live, so the harness must run elsewhere.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from RayNet.openeds.dataset import (
    OpenEDSSegDataset,
    OpenEDSSequenceDataset,
)
from RayNet.openeds.segmenter import RITnetStyleSegmenter, combined_loss
from RayNet.openeds.temporal import (
    TCNTemporalBlock,
    derive_pseudo_blink_labels,
    derive_pseudo_saccade_labels,
)

log = logging.getLogger(__name__)


# ── Stage S1 — segmenter ─────────────────────────────────────────────

def train_segmenter(
    train_loader: Iterable, val_loader: Iterable,
    device: torch.device, epochs: int = 30,
    base_channels: int = 32, growth_rate: int = 16,
    lr: float = 5e-4,
    class_weights: torch.Tensor | None = None,
    log_every: int = 50,
) -> RITnetStyleSegmenter:
    """Train the RITnet-style 4-class segmenter on OpenEDS."""
    model = RITnetStyleSegmenter(
        in_channels=1, num_classes=4,
        base_channels=base_channels, growth_rate=growth_rate,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for step, batch in enumerate(train_loader):
            imgs = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            logits = model(imgs)
            loss = combined_loss(
                logits, masks, class_weights=class_weights, dice_weight=1.0
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += loss.item()
            n += 1
            if step % log_every == 0:
                log.info('epoch=%d step=%d loss=%.4f', epoch, step, loss.item())
        scheduler.step()

        val_iou = _validate_segmenter(model, val_loader, device)
        log.info('epoch=%d train_loss=%.4f val_meanIoU=%.4f',
                 epoch, running / max(n, 1), val_iou)
    return model


@torch.no_grad()
def _validate_segmenter(model, val_loader, device) -> float:
    model.eval()
    confusion = torch.zeros(4, 4, dtype=torch.int64, device=device)
    for batch in val_loader:
        imgs = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        pred = model(imgs).argmax(dim=1)
        for c in range(4):
            for k in range(4):
                confusion[c, k] += int(((masks == c) & (pred == k)).sum())
    iou_per_class = []
    for c in range(4):
        tp = confusion[c, c].item()
        fp = (confusion[:, c].sum() - confusion[c, c]).item()
        fn = (confusion[c, :].sum() - confusion[c, c]).item()
        denom = tp + fp + fn
        iou_per_class.append(tp / denom if denom else 0.0)
    return sum(iou_per_class[1:]) / 3.0  # exclude background


# ── Stage S2 — TCN ───────────────────────────────────────────────────

def per_frame_feature_packer(
    pred_masks: torch.Tensor,
    raw_torsion_deg: torch.Tensor,
    pred_gaze_3d: torch.Tensor,
) -> torch.Tensor:
    """Pack a (B, T, in_dim) feature tensor for the TCN.

    The packing is intentionally simple — refine as the OpenEDS
    pipeline matures. ``in_dim = 15`` matches the TCN default.

    Args:
        pred_masks: (B, T, H, W) int64 segmentation outputs.
        raw_torsion_deg: (B, T) classical torsion estimate per frame.
        pred_gaze_3d: (B, T, 3) per-frame gaze unit vector (from the
            GazeGene-trained head, applied to the eye crop's intrinsics).

    Returns:
        (B, T, 15) feature tensor.
    """
    B, T, H, W = pred_masks.shape
    pupil = (pred_masks == 3)
    iris = (pred_masks == 2)

    pupil_area = pupil.float().sum(dim=(2, 3))                    # (B, T)
    iris_area = iris.float().sum(dim=(2, 3))                      # (B, T)
    # Median-normalise areas so the TCN sees scale-invariant numbers.
    pupil_area_norm = pupil_area / (pupil_area.median(dim=1, keepdim=True).values + 1.0)
    iris_area_norm = iris_area / (iris_area.median(dim=1, keepdim=True).values + 1.0)

    # Centroid of pupil/iris in normalised image space (cy, cx).
    grid_y = torch.arange(H, device=pred_masks.device, dtype=torch.float32)[None, None, :, None]
    grid_x = torch.arange(W, device=pred_masks.device, dtype=torch.float32)[None, None, None, :]
    pupil_cy = (pupil.float() * grid_y).sum(dim=(2, 3)) / (pupil_area + 1.0)
    pupil_cx = (pupil.float() * grid_x).sum(dim=(2, 3)) / (pupil_area + 1.0)
    iris_cy = (iris.float() * grid_y).sum(dim=(2, 3)) / (iris_area + 1.0)
    iris_cx = (iris.float() * grid_x).sum(dim=(2, 3)) / (iris_area + 1.0)

    pupil_cy_n = pupil_cy / H
    pupil_cx_n = pupil_cx / W
    iris_cy_n = iris_cy / H
    iris_cx_n = iris_cx / W

    feats = torch.stack([
        pred_gaze_3d[..., 0], pred_gaze_3d[..., 1], pred_gaze_3d[..., 2],
        pupil_cy_n, pupil_cx_n, pupil_area_norm,
        iris_cy_n, iris_cx_n, iris_area_norm,
        # Pad to 15 dims with diff/velocity proxies (cheap structural cues).
        pupil_cy_n - iris_cy_n,
        pupil_cx_n - iris_cx_n,
        torch.zeros_like(pupil_cy_n),  # reserved
        torch.zeros_like(pupil_cy_n),  # reserved
        torch.zeros_like(pupil_cy_n),  # reserved
        raw_torsion_deg,
    ], dim=-1)
    return feats   # (B, T, 15)


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='OpenEDS segmenter + TCN training')
    parser.add_argument('--root', type=str, required=True,
                        help='openEDS/openEDS root with S_<id> subjects.')
    parser.add_argument('--stage', choices=['seg', 'tcn'], default='seg')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--crop_h', type=int, default=416)
    parser.add_argument('--crop_w', type=int, default=640)
    parser.add_argument('--window', type=int, default=64)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    device = torch.device(args.device)
    crop = (args.crop_h, args.crop_w)

    if args.stage == 'seg':
        train_ds = OpenEDSSegDataset(args.root, crop=crop)
        # Split subject-wise: 80/20 by sorted subject id for now.
        n = len(train_ds.subjects)
        train_subjects = train_ds.subjects[: max(1, int(n * 0.8))]
        val_subjects = train_ds.subjects[max(1, int(n * 0.8)):]
        log.info('Segmenter train subjects=%d val=%d',
                 len(train_subjects), len(val_subjects))
        train_loader = DataLoader(
            OpenEDSSegDataset(args.root, subjects=train_subjects, crop=crop),
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=True, pin_memory=True,
        )
        val_loader = DataLoader(
            OpenEDSSegDataset(args.root, subjects=val_subjects, crop=crop),
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False, pin_memory=True,
        )
        train_segmenter(train_loader, val_loader, device,
                        epochs=args.epochs, lr=args.lr)
    else:
        # The TCN training harness depends on a frozen segmenter, the
        # IrisPolarTorsion estimator, and a GazeGene-trained gaze head
        # applied to OpenEDS crops via known intrinsics. The full glue
        # is left to a follow-up — this stub keeps the CLI complete
        # and the seq dataloader exercised.
        seq_loader = DataLoader(
            OpenEDSSequenceDataset(args.root, window=args.window, crop=crop),
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=True, pin_memory=True,
        )
        tcn = TCNTemporalBlock(in_dim=15).to(device)
        log.info('TCN initialised. Receptive field=%d frames',
                 tcn.receptive_field())
        log.info('Sequence dataset windows=%d', len(seq_loader.dataset))
        log.warning('TCN supervised fine-tuning needs the segmenter '
                    'checkpoint + per-frame gaze pipeline; this script '
                    'currently exercises only the sequence loader.')


if __name__ == '__main__':
    main()
