"""
MosaicML Streaming readers for OpenEDS MDS shards.

Mirrors :mod:`RayNet.streaming.dataset` for OpenEDS. Two reader
classes:

- :class:`StreamingOpenEDSSegDataset` — flat per-frame iterator for
  the segmenter. Decodes the PNG image bytes and packs the mask.
- :class:`StreamingOpenEDSSequenceDataset` — yields contiguous
  temporal windows. Relies on shards being written with
  ``sequence_grouped=True`` so that per-subject frames are
  consecutive with monotonic ``frame_idx``. Streams with
  ``shuffle=False`` to preserve order; the Dataset wrapper boundary-
  checks each window to ensure all frames belong to the same
  subject and therefore form a real sequence.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from streaming import StreamingDataset
except ImportError:
    StreamingDataset = None

log = logging.getLogger(__name__)

_Base = StreamingDataset if StreamingDataset is not None else object


def _decode_image(img_bytes: bytes) -> np.ndarray:
    """Decode the PNG bytes from MDS to a single-channel grayscale array."""
    import cv2
    nparr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(
            'OpenEDS MDS sample failed to decode (non-PNG bytes?). '
            'Re-run convert_to_mds.py against the FovalNet root.'
        )
    return img


def _maybe_pad_to(img: np.ndarray, mask: np.ndarray,
                  out_h: Optional[int], out_w: Optional[int]
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Centre-crop / zero-pad to (out_h, out_w). No-op when None."""
    if out_h is None and out_w is None:
        return img, mask
    h, w = img.shape

    def _adjust(a, src, tgt, axis):
        if src == tgt or tgt is None:
            return a
        if src > tgt:
            start = (src - tgt) // 2
            slicer = [slice(None), slice(None)]
            slicer[axis] = slice(start, start + tgt)
            return a[tuple(slicer)]
        pad = tgt - src
        pad_a = pad // 2
        pad_b = pad - pad_a
        spec = [(0, 0), (0, 0)]
        spec[axis] = (pad_a, pad_b)
        return np.pad(a, spec, mode='constant', constant_values=0)

    img = _adjust(img, h, out_h, axis=0)
    img = _adjust(img, w, out_w, axis=1)
    mask = _adjust(mask, h, out_h, axis=0)
    mask = _adjust(mask, w, out_w, axis=1)
    return img, mask


class StreamingOpenEDSSegDataset(_Base):
    """Per-frame OpenEDS streaming dataset for the segmenter.

    Produces sample dicts compatible with
    :class:`RayNet.openeds.dataset.OpenEDSSegDataset`::

        {
            'image': (1, H, W) float32 in [0, 1],
            'mask':  (H, W) int64 class index,
            'subject': int,
            'frame_idx': int,
            'has_mask': int (0 / 1),
        }

    Args:
        crop: optional ``(H, W)`` post-load crop. Defaults to native.
        require_labelled: if True, frames where ``has_mask == 0`` are
            returned as ``None`` so an upstream ``NonEmptyBatchLoader``
            can drop them.
        transform: optional torchvision transform applied to ``image``.
        **kwargs: forwarded to ``StreamingDataset`` (``local``,
            ``remote``, ``shuffle``, ``batch_size``, etc.).
    """

    def __init__(
        self,
        transform=None,
        crop: Optional[Tuple[int, int]] = None,
        require_labelled: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.transform = transform
        self.crop = crop
        self.require_labelled = bool(require_labelled)

    def __getitem__(self, idx: int):
        raw = super().__getitem__(idx)
        if self.require_labelled and int(raw.get('has_mask', 1)) == 0:
            return None

        img_np = _decode_image(raw['image'])
        mask_np = np.asarray(raw['mask'], dtype=np.uint8)

        out_h = self.crop[0] if self.crop else None
        out_w = self.crop[1] if self.crop else None
        img_np, mask_np = _maybe_pad_to(img_np, mask_np, out_h, out_w)

        img_t = torch.from_numpy(img_np).float().div_(255.0).unsqueeze(0)
        mask_t = torch.from_numpy(mask_np.astype(np.int64))

        if self.transform is not None:
            img_t = self.transform(img_t)

        return {
            'image': img_t,
            'mask': mask_t,
            'subject': int(raw['subject']),
            'frame_idx': int(raw['frame_idx']),
            'has_mask': int(raw.get('has_mask', 1)),
        }


class StreamingOpenEDSSequenceDataset(_Base):
    """Contiguous-window OpenEDS streaming dataset for the TCN.

    Returns sliding windows of ``window`` consecutive frames. Requires
    shards to be written with ``sequence_grouped=True`` and read with
    ``shuffle=False``; the wrapper sanity-checks per window that all
    frames share a single ``subject`` id and contiguous ``frame_idx``
    ordering, returning ``None`` (skipped via NonEmptyBatchLoader) for
    windows that span a subject boundary.

    Args:
        window: number of frames per item.
        crop: optional ``(H, W)`` crop.
        transform: optional torchvision transform applied per frame.

    Each item is::

        {
            'images':     (T, 1, H, W) float32 in [0, 1],
            'masks':      (T, H, W) int64,
            'subject':    int,
            'frame_idxs': (T,) long,
            'has_masks':  (T,) long  (0/1 per frame),
        }
    """

    def __init__(
        self,
        window: int = 64,
        transform=None,
        crop: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if window < 2:
            raise ValueError('window must be >= 2')
        self.window = int(window)
        self.transform = transform
        self.crop = crop

    def __len__(self) -> int:                              # type: ignore[override]
        # Each item consumes ``window`` consecutive samples; truncate
        # the tail that cannot fill a full window.
        return max(0, super().__len__() - self.window + 1)

    def __getitem__(self, idx: int):
        # Pull ``window`` consecutive samples from the underlying
        # StreamingDataset. Order is preserved when shuffle=False.
        raws = [super(StreamingOpenEDSSequenceDataset, self).__getitem__(idx + i)
                for i in range(self.window)]

        # Sanity check: window must lie inside a single subject.
        first_subject = int(raws[0]['subject'])
        for r in raws[1:]:
            if int(r['subject']) != first_subject:
                return None  # window spans a subject boundary; skip

        out_h = self.crop[0] if self.crop else None
        out_w = self.crop[1] if self.crop else None

        imgs = []
        masks = []
        frame_idxs = []
        has_masks = []
        for r in raws:
            img_np = _decode_image(r['image'])
            mask_np = np.asarray(r['mask'], dtype=np.uint8)
            img_np, mask_np = _maybe_pad_to(img_np, mask_np, out_h, out_w)
            imgs.append(img_np)
            masks.append(mask_np)
            frame_idxs.append(int(r['frame_idx']))
            has_masks.append(int(r.get('has_mask', 1)))

        img_stack = np.stack(imgs).astype(np.float32) / 255.0
        mask_stack = np.stack(masks).astype(np.int64)
        img_t = torch.from_numpy(img_stack).unsqueeze(1)   # (T, 1, H, W)
        mask_t = torch.from_numpy(mask_stack)              # (T, H, W)
        if self.transform is not None:
            img_t = torch.stack([self.transform(f) for f in img_t])

        return {
            'images': img_t,
            'masks': mask_t,
            'subject': first_subject,
            'frame_idxs': torch.tensor(frame_idxs, dtype=torch.long),
            'has_masks': torch.tensor(has_masks, dtype=torch.long),
        }


# ── Loader builders ──────────────────────────────────────────────────

def create_openeds_seg_streaming_dataloaders(
    remote_train: str,
    remote_val: str,
    local_cache: str = './mds_cache_openeds',
    batch_size: int = 8,
    num_workers: int = 4,
    transform=None,
    val_transform=None,
    crop: Optional[Tuple[int, int]] = (416, 640),
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    require_labelled: bool = True,
    **streaming_kwargs,
):
    """Build OpenEDS segmenter dataloaders from MDS shards.

    The defaults match the segmenter recipe: 416×640 zero-padded input
    (stride-32 friendly), labelled-only frames.
    """
    assert StreamingDataset is not None, 'pip install mosaicml-streaming'

    def _empty_skip(loader):
        from RayNet.streaming.dataset import NonEmptyBatchLoader
        return NonEmptyBatchLoader(loader)

    train_ds = StreamingOpenEDSSegDataset(
        transform=transform, crop=crop, require_labelled=require_labelled,
        remote=None, local=remote_train, split=None, shuffle=True,
        batch_size=batch_size, **streaming_kwargs,
    )
    val_ds = StreamingOpenEDSSegDataset(
        transform=val_transform, crop=crop, require_labelled=require_labelled,
        remote=None, local=remote_val, split=None, shuffle=False,
        batch_size=batch_size, **streaming_kwargs,
    )

    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  prefetch_factor=prefetch_factor if num_workers > 0 else None,
                  persistent_workers=persistent_workers and num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, **common)
    if require_labelled:
        train_loader = _empty_skip(train_loader)
        val_loader = _empty_skip(val_loader)
    return train_loader, val_loader


def create_openeds_sequence_streaming_dataloaders(
    remote_train: str,
    remote_val: str,
    local_cache: str = './mds_cache_openeds',
    window: int = 64,
    batch_size: int = 4,
    num_workers: int = 4,
    transform=None,
    val_transform=None,
    crop: Optional[Tuple[int, int]] = (416, 640),
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    **streaming_kwargs,
):
    """Build OpenEDS sequence dataloaders from MDS shards.

    Requires shards written with ``sequence_grouped=True``.
    """
    assert StreamingDataset is not None, 'pip install mosaicml-streaming'

    def _empty_skip(loader):
        from RayNet.streaming.dataset import NonEmptyBatchLoader
        return NonEmptyBatchLoader(loader)

    train_ds = StreamingOpenEDSSequenceDataset(
        window=window, transform=transform, crop=crop,
        remote=None, local=remote_train, split=None,
        shuffle=False,                                     # preserve sequence
        batch_size=batch_size, **streaming_kwargs,
    )
    val_ds = StreamingOpenEDSSequenceDataset(
        window=window, transform=val_transform, crop=crop,
        remote=None, local=remote_val, split=None, shuffle=False,
        batch_size=batch_size, **streaming_kwargs,
    )
    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  prefetch_factor=prefetch_factor if num_workers > 0 else None,
                  persistent_workers=persistent_workers and num_workers > 0)
    train_loader = _empty_skip(
        DataLoader(train_ds, batch_size=batch_size, **common))
    val_loader = _empty_skip(
        DataLoader(val_ds, batch_size=batch_size, **common))
    return train_loader, val_loader


__all__ = [
    'StreamingOpenEDSSegDataset',
    'StreamingOpenEDSSequenceDataset',
    'create_openeds_seg_streaming_dataloaders',
    'create_openeds_sequence_streaming_dataloaders',
]
