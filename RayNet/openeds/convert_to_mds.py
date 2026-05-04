"""
Convert FovalNet-preprocessed OpenEDS to MDS shards.

Mirrors the GazeGene MDS converter (`RayNet/streaming/convert_to_mds.py`)
so the OpenEDS pipeline shares the same operational ergonomics: shard
once, stream from local cache or S3/MinIO/GCS during training.

Usage::

    python -m RayNet.openeds.convert_to_mds \\
        --data_dir /path/to/openEDS/openEDS \\
        --output_dir /path/to/mds_openeds/train \\
        --split train \\
        --sequence_grouped

The default split keeps subjects S_0 ... S_<80% threshold> in train
and the rest in val. For deterministic splits across machines pass
``--train_subjects "S_0,S_1,..."`` explicitly.

Sequence grouping
-----------------
``--sequence_grouped`` emits samples per subject in monotonic
``frame_idx`` order so a streaming reader with ``shuffle=False`` can
produce contiguous temporal windows for the TCN. Without this flag
samples are shuffled, which is fine for training the per-frame
segmenter but breaks the temporal block.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2

try:
    from streaming import MDSWriter
except ImportError:                    # pragma: no cover
    MDSWriter = None

from RayNet.openeds.dataset import (
    NATIVE_H,
    NATIVE_W,
    OPENEDS_CLASS_MAP,
    _list_frames,
    _list_subjects,
)

log = logging.getLogger(__name__)

# OpenEDS MDS schema. Mirrors the GazeGene v6 schema's column-encoding
# style (jpeg-encoded image + ndarray labels) so the streaming reader
# can share decode logic.
OPENEDS_MDS_COLUMNS = {
    'image': 'bytes',           # PNG-encoded grayscale (1 channel)
    'mask': 'ndarray',          # uint8 (H, W) class indices in [0, 3]
    'subject': 'int',           # parsed from S_<id>
    'frame_idx': 'int',
    'has_mask': 'int',          # 1 = paired .npy present, 0 = unlabelled frame
}


def _parse_subject_id(s_dir: str) -> int:
    """Parse the integer id from an ``S_<int>`` directory name."""
    if not s_dir.startswith('S_'):
        raise ValueError(f'Unexpected OpenEDS subject directory: {s_dir}')
    return int(s_dir[2:])


def _png_bytes(img_path: Path) -> bytes:
    """Read raw PNG bytes off disk (no re-encoding)."""
    with open(img_path, 'rb') as f:
        return f.read()


def _walk_samples(
    data_dir: str, subjects: Sequence[str],
    sequence_grouped: bool, include_unlabelled: bool,
) -> Iterable[dict]:
    """Yield MDS-ready records.

    Order rules:
      - With ``sequence_grouped``: subjects in sorted order, within each
        subject frames in sorted ``frame_idx`` order. The streaming
        reader must use ``shuffle=False`` to preserve this ordering.
      - Without ``sequence_grouped``: subjects in sorted order but
        frames shuffled within each subject (so per-shard variety
        helps random-access dataloaders even with shuffle=False at
        read time).
    """
    rng = np.random.default_rng(42)
    root = Path(data_dir)
    for subj in subjects:
        subj_dir = root / subj
        triples: List[Tuple[int, Path, Path | None]] = []
        # Discover every PNG (labelled or unlabelled) when include_unlabelled.
        # _list_frames only returns paired frames, so re-walk for the
        # unlabelled case.
        if include_unlabelled:
            import re
            frame_re = re.compile(r'^(\d+)\.png$')
            for entry in subj_dir.iterdir():
                m = frame_re.match(entry.name)
                if not m:
                    continue
                idx = int(m.group(1))
                npy = subj_dir / f'{idx}.npy'
                triples.append((idx, entry, npy if npy.is_file() else None))
        else:
            for idx, png, npy in _list_frames(subj_dir):
                triples.append((idx, png, npy))

        if not triples:
            continue
        if sequence_grouped:
            triples.sort(key=lambda t: t[0])
        else:
            rng.shuffle(triples)

        subj_id = _parse_subject_id(subj)
        for idx, png, npy in triples:
            mask: Optional[np.ndarray]
            has_mask = 1 if (npy is not None and npy.is_file()) else 0
            if has_mask:
                mask = np.load(str(npy)).astype(np.uint8)
                if mask.shape != (NATIVE_H, NATIVE_W):
                    log.warning(
                        'Mask shape %s != native (%d, %d) at %s; resizing nearest.',
                        mask.shape, NATIVE_H, NATIVE_W, npy)
                    mask = cv2.resize(
                        mask, (NATIVE_W, NATIVE_H),
                        interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros((NATIVE_H, NATIVE_W), dtype=np.uint8)

            yield {
                'image': _png_bytes(png),
                'mask': mask,
                'subject': subj_id,
                'frame_idx': idx,
                'has_mask': has_mask,
            }


def convert_openeds_to_mds(
    data_dir: str,
    output_dir: str,
    split: str = 'train',
    subjects: Optional[Sequence[str]] = None,
    sequence_grouped: bool = True,
    include_unlabelled: bool = False,
    size_limit: int = 1 << 27,         # 128 MB per shard
    compression: str = 'zstd',
    hashes: Sequence[str] = ('sha1',),
) -> int:
    """Walk the FovalNet preprocessed OpenEDS tree and emit MDS shards.

    Args:
        data_dir: path containing ``S_<id>`` subject sub-folders. This
            is typically the inner ``openEDS/openEDS`` directory.
        output_dir: target directory for shard files (created if needed).
        split: free-form label written into ``split_meta.json``; the
            convention is ``train`` / ``val`` / ``test``.
        subjects: optional explicit subject list (e.g. ``["S_0", ...]``).
            ``None`` means *all* subjects under *data_dir*.
        sequence_grouped: when True, samples are emitted in monotonic
            frame order per subject so a streaming reader with
            ``shuffle=False`` produces contiguous temporal windows.
        include_unlabelled: include images without paired ``.npy``
            masks. Mask is then filled with zeros and ``has_mask=0``;
            train scripts can drop these via the ``has_mask`` flag.
        size_limit: target shard byte budget (default 128 MB).
        compression: shard compression — 'zstd' is the default.
        hashes: hash algorithms to record per shard.

    Returns:
        n_samples: number of records written.
    """
    assert MDSWriter is not None, (
        'mosaicml-streaming is required: pip install mosaicml-streaming'
    )
    os.makedirs(output_dir, exist_ok=True)
    if subjects is None:
        subjects = _list_subjects(data_dir)
    log.info(
        'OpenEDS → MDS: split=%s, subjects=%d, sequence_grouped=%s, '
        'include_unlabelled=%s',
        split, len(subjects), sequence_grouped, include_unlabelled,
    )

    n = 0
    with MDSWriter(
        out=output_dir, columns=OPENEDS_MDS_COLUMNS,
        compression=compression, hashes=list(hashes),
        size_limit=size_limit,
    ) as writer:
        for record in _walk_samples(
            data_dir, subjects,
            sequence_grouped=sequence_grouped,
            include_unlabelled=include_unlabelled,
        ):
            writer.write(record)
            n += 1
            if n % 5000 == 0:
                log.info('  ... %d samples written', n)
    log.info('OpenEDS MDS done: %d samples → %s', n, output_dir)
    return n


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert OpenEDS (FovalNet preprocessed) to MDS shards.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='openEDS/openEDS root with S_<id> subdirs.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Target directory for MDS shards.')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of subjects assigned to train when '
                             '--train_subjects / --val_subjects are absent.')
    parser.add_argument('--train_subjects', type=str, default=None,
                        help='Comma-separated explicit subject list (e.g. '
                             '"S_0,S_1,..."). Overrides --train_split.')
    parser.add_argument('--val_subjects', type=str, default=None,
                        help='Same as --train_subjects but for the val shard.')
    parser.add_argument('--sequence_grouped', action='store_true',
                        default=True,
                        help='Emit per-subject sequential frame order. ON by '
                             'default; pass --no_sequence_grouped to shuffle.')
    parser.add_argument('--no_sequence_grouped', dest='sequence_grouped',
                        action='store_false')
    parser.add_argument('--include_unlabelled', action='store_true',
                        help='Include frames without paired .npy masks (for '
                             'TCN sequence training that needs dense frames).')
    parser.add_argument('--size_limit_mb', type=int, default=128)
    return parser.parse_args(argv)


def _resolve_subject_split(
    data_dir: str, args: argparse.Namespace,
) -> Tuple[List[str], List[str]]:
    all_subjects = _list_subjects(data_dir)
    if args.train_subjects is not None and args.val_subjects is not None:
        train = [s.strip() for s in args.train_subjects.split(',') if s.strip()]
        val = [s.strip() for s in args.val_subjects.split(',') if s.strip()]
        return train, val
    n = len(all_subjects)
    cutoff = max(1, int(round(n * args.train_split)))
    return all_subjects[:cutoff], all_subjects[cutoff:]


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    train_subjects, val_subjects = _resolve_subject_split(args.data_dir, args)
    if args.split == 'train':
        subjects = train_subjects
    elif args.split == 'val':
        subjects = val_subjects
    else:                                          # test
        subjects = val_subjects                    # mirror val by default

    convert_openeds_to_mds(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        subjects=subjects,
        sequence_grouped=args.sequence_grouped,
        include_unlabelled=args.include_unlabelled,
        size_limit=args.size_limit_mb << 20,
    )
    log.info('Class map: %s', OPENEDS_CLASS_MAP)


if __name__ == '__main__':
    main()
