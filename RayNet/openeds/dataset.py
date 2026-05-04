"""
OpenEDS preprocessed (FovalNet variant) dataset loaders.

The on-disk layout produced by the FovalNet preprocessing notebook
(`docs/fovalnet.ipynb`) is::

    openEDS/openEDS/
        S_<subject_id>/
            0.png      # 400x640 uint8 grayscale IR eye crop
            0.npy      # 400x640 uint8 segmentation mask
            1.png
            1.npy
            ...

There are 191 subject directories and roughly 93 k images / 62 k masks
(some images are unlabelled). Per-pixel mask values use the 4-class
convention :data:`OPENEDS_CLASS_MAP`.

Two dataset classes are provided:

- :class:`OpenEDSSegDataset` — flat per-frame iterator for training the
  semantic segmenter (one (image, mask) tuple per ``__getitem__``).
- :class:`OpenEDSSequenceDataset` — per-subject sequence iterator that
  yields fixed-length windows of frames for the temporal block. Frames
  within a sequence are sorted by integer index extracted from the
  filename, which matches the original OpenEDS capture order.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# OpenEDS preprocessed semantic class map. Background is implicit
# class 0; sclera, iris, and pupil follow in order. The FovalNet
# notebook documents this mapping in cell-12 (`class_names = {0:
# 'Background', 1: 'Sclera', 2: 'Iris', 3: 'Pupil'}`).
OPENEDS_CLASS_MAP = {
    0: 'background',
    1: 'sclera',
    2: 'iris',
    3: 'pupil',
}
NUM_CLASSES = len(OPENEDS_CLASS_MAP)
NATIVE_H = 400
NATIVE_W = 640

# Image filename pattern: "<idx>.png" with arbitrary integer index.
_FRAME_RE = re.compile(r'^(\d+)\.png$')


def _list_subjects(root: str) -> List[str]:
    """Return sorted ``S_<id>`` subdirectories under *root*."""
    p = Path(root)
    if not p.is_dir():
        raise FileNotFoundError(f'OpenEDS root not found: {root}')
    return sorted(d.name for d in p.iterdir()
                  if d.is_dir() and d.name.startswith('S_'))


def _list_frames(subject_dir: Path) -> List[Tuple[int, Path, Path]]:
    """List ``(frame_idx, image_path, mask_path)`` for a subject.

    Includes only frames that have *both* a PNG and a paired NPY label
    file; unlabelled frames are skipped (about a third of the corpus).
    Frames are returned sorted by integer index, not lexical order, so
    sequence loaders can rely on temporal ordering.
    """
    frames: List[Tuple[int, Path, Path]] = []
    for entry in subject_dir.iterdir():
        m = _FRAME_RE.match(entry.name)
        if not m:
            continue
        idx = int(m.group(1))
        mask = subject_dir / f'{idx}.npy'
        if mask.is_file():
            frames.append((idx, entry, mask))
    frames.sort(key=lambda t: t[0])
    return frames


class OpenEDSSegDataset(Dataset):
    """Per-frame loader for OpenEDS semantic segmentation.

    Each item is a dict with:
        - ``image`` : ``(1, H, W) float32`` in ``[0, 1]`` (grayscale IR).
        - ``mask``  : ``(H, W) int64`` with class indices in ``[0, 3]``.
        - ``subject``: subject id string (e.g. ``"S_42"``).
        - ``frame_idx`` : original integer frame index.

    Args:
        root: path to the ``openEDS/openEDS`` directory containing
            the per-subject ``S_<id>`` folders.
        subjects: optional list of subject ids to include
            (e.g. ``["S_0", "S_1"]``). ``None`` means *all*.
        crop: ``(out_h, out_w)`` post-load resize. ``None`` keeps the
            native 400×640 resolution. Common choice for stride-32-
            compatible inputs is ``(384, 640)`` (centre-crop) or
            ``(416, 640)`` (zero-pad). Defaults to native.
        transform: optional callable applied to ``image`` after the
            tensor conversion. Mask is *not* transformed.

    The native 400×640 aspect ratio is 1:1.6. Squashing to a square
    warps eyelid geometry and degrades sub-pixel pupil boundaries;
    callers that need stride-32-compatibility should pad to 416×640
    rather than resize.
    """

    def __init__(
        self,
        root: str,
        subjects: Optional[Sequence[str]] = None,
        crop: Optional[Tuple[int, int]] = None,
        transform=None,
    ) -> None:
        super().__init__()
        all_subjects = _list_subjects(root)
        if subjects is not None:
            wanted = set(subjects)
            keep = [s for s in all_subjects if s in wanted]
            missing = wanted - set(keep)
            if missing:
                raise ValueError(
                    f'OpenEDS subjects not found under {root}: {sorted(missing)}'
                )
            all_subjects = keep
        self.root = Path(root)
        self.subjects = all_subjects
        self.crop = crop
        self.transform = transform

        # Flat list of (subject, idx, image_path, mask_path) for O(1)
        # __getitem__ lookup. Cheap to build because we only stat
        # filenames, not the contents.
        self._items: List[Tuple[str, int, Path, Path]] = []
        for subj in self.subjects:
            for idx, png, npy in _list_frames(self.root / subj):
                self._items.append((subj, idx, png, npy))

    def __len__(self) -> int:
        return len(self._items)

    def _load_pair(self, png: Path, npy: Path) -> Tuple[np.ndarray, np.ndarray]:
        # cv2 import is local so the dataset module stays importable
        # in environments without OpenCV (e.g. headless CI).
        import cv2
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f'Failed to read OpenEDS frame: {png}')
        mask = np.load(str(npy))
        if img.shape != mask.shape:
            raise ValueError(
                f'OpenEDS image/mask shape mismatch '
                f'({img.shape} vs {mask.shape}) at {png.parent}'
            )
        return img, mask

    def _maybe_crop(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.crop is None:
            return img, mask
        out_h, out_w = self.crop
        h, w = img.shape
        # Centre-crop or zero-pad as needed (one axis at a time).
        # Crop axis where target is smaller, pad where target is larger.
        # Pupil is roughly centred in OpenEDS frames so centre-crop
        # is a safe default.
        def _adjust(a: np.ndarray, src: int, tgt: int, axis: int):
            if src == tgt:
                return a
            if src > tgt:
                start = (src - tgt) // 2
                slicer = [slice(None), slice(None)]
                slicer[axis] = slice(start, start + tgt)
                return a[tuple(slicer)]
            pad = tgt - src
            pad_a = pad // 2
            pad_b = pad - pad_a
            pad_spec = [(0, 0), (0, 0)]
            pad_spec[axis] = (pad_a, pad_b)
            return np.pad(a, pad_spec, mode='constant', constant_values=0)

        img = _adjust(img, h, out_h, axis=0)
        img = _adjust(img, w, out_w, axis=1)
        mask = _adjust(mask, h, out_h, axis=0)
        mask = _adjust(mask, w, out_w, axis=1)
        return img, mask

    def __getitem__(self, idx: int) -> dict:
        subj, frame_idx, png, npy = self._items[idx]
        img, mask = self._load_pair(png, npy)
        img, mask = self._maybe_crop(img, mask)

        img_t = torch.from_numpy(img).float().div_(255.0).unsqueeze(0)
        mask_t = torch.from_numpy(mask).long()

        if self.transform is not None:
            img_t = self.transform(img_t)

        return {
            'image': img_t,         # (1, H, W) float32
            'mask': mask_t,         # (H, W) int64
            'subject': subj,
            'frame_idx': frame_idx,
        }


class OpenEDSSequenceDataset(Dataset):
    """Sequence loader for the temporal block.

    Each item is a contiguous window of ``window`` frames drawn from a
    single subject. The window slides by ``stride`` frames; with
    ``stride == window`` windows are non-overlapping. Returned tensor
    shapes are ``(window, 1, H, W)`` for the image stack and
    ``(window, H, W)`` for the mask stack.

    Sequences shorter than ``window`` are dropped entirely (they
    cannot fill a single window). For typical OpenEDS subjects with
    150-330 labelled frames and ``window=64``, this drops at most a
    handful of subjects.

    Args:
        root: path to the ``openEDS/openEDS`` directory.
        window: number of consecutive frames per item.
        stride: frame offset between consecutive windows. Defaults
            to ``window`` (non-overlapping). Smaller stride yields
            more overlapping windows for richer training.
        subjects: optional subset of subjects.
        crop: optional spatial crop (passed to image/mask).
    """

    def __init__(
        self,
        root: str,
        window: int = 64,
        stride: Optional[int] = None,
        subjects: Optional[Sequence[str]] = None,
        crop: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        if window < 2:
            raise ValueError(f'window must be >= 2, got {window}')
        self.root = Path(root)
        self.window = int(window)
        self.stride = int(stride) if stride is not None else self.window
        self.crop = crop

        all_subjects = _list_subjects(root)
        if subjects is not None:
            wanted = set(subjects)
            all_subjects = [s for s in all_subjects if s in wanted]
        self.subjects = all_subjects

        # Pre-compute the per-window (subject, list-of-frame-tuples).
        self._windows: List[Tuple[str, List[Tuple[int, Path, Path]]]] = []
        for subj in self.subjects:
            frames = _list_frames(self.root / subj)
            if len(frames) < self.window:
                continue
            for start in range(0, len(frames) - self.window + 1, self.stride):
                self._windows.append((subj, frames[start:start + self.window]))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        import cv2
        subj, frames = self._windows[idx]
        imgs = []
        masks = []
        frame_idxs = []
        for f_idx, png, npy in frames:
            img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            mask = np.load(str(npy))
            if self.crop is not None:
                # Reuse the SegDataset helper without instantiating it —
                # tiny inline duplication keeps the sequence loader free
                # of state in the hot path.
                out_h, out_w = self.crop
                h, w = img.shape
                if h > out_h:
                    s = (h - out_h) // 2
                    img = img[s:s + out_h]
                    mask = mask[s:s + out_h]
                elif h < out_h:
                    pad = out_h - h
                    img = np.pad(img, ((pad // 2, pad - pad // 2), (0, 0)))
                    mask = np.pad(mask, ((pad // 2, pad - pad // 2), (0, 0)))
                if w > out_w:
                    s = (w - out_w) // 2
                    img = img[:, s:s + out_w]
                    mask = mask[:, s:s + out_w]
                elif w < out_w:
                    pad = out_w - w
                    img = np.pad(img, ((0, 0), (pad // 2, pad - pad // 2)))
                    mask = np.pad(mask, ((0, 0), (pad // 2, pad - pad // 2)))
            imgs.append(img)
            masks.append(mask)
            frame_idxs.append(f_idx)

        img_stack = np.stack(imgs).astype(np.float32) / 255.0  # (W, H, W)
        mask_stack = np.stack(masks).astype(np.int64)

        return {
            'images': torch.from_numpy(img_stack).unsqueeze(1),  # (T, 1, H, W)
            'masks': torch.from_numpy(mask_stack),               # (T, H, W)
            'subject': subj,
            'frame_idxs': torch.tensor(frame_idxs, dtype=torch.long),
        }
