"""
Convert GazeGeneDataset to MosaicML MDS (Mosaic Dataset Shard) format.

MDS is optimized for deterministic, resumable streaming with:
  - Efficient binary serialization (no JPEG decode overhead)
  - Deterministic shuffle across workers and nodes
  - Built-in S3/MinIO/GCS streaming support

Usage:
    python -m RayNet.streaming.convert_to_mds \
        --data_dir /path/to/GazeGene_FaceCrops \
        --output_dir ./mds_shards/train \
        --split train
"""

import os
import json
import numpy as np
from tqdm import tqdm

try:
    from streaming import MDSWriter
except ImportError:
    MDSWriter = None


# MDS column schema — maps field names to MDS types
MDS_COLUMNS = {
    'image': 'jpeg',
    'landmark_coords': 'ndarray',
    'landmark_coords_px': 'ndarray',
    'optical_axis': 'ndarray',
    'R_norm': 'ndarray',
    'R_kappa': 'ndarray',
    'K': 'ndarray',
    'R_cam': 'ndarray',
    'T_cam': 'ndarray',
    'M_norm_inv': 'ndarray',
    'eyeball_center_3d': 'ndarray',
    'subject': 'int',
    'cam_id': 'int',
    'frame_idx': 'int',
}


def _tensor_image_to_pil(img_tensor):
    """Convert (3, H, W) float [0,1] tensor to PIL Image."""
    from PIL import Image
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def convert_to_mds(dataset, output_dir, split='train',
                    multiview_grouped=True):
    """
    Convert a GazeGeneDataset to MDS shards.

    When multiview_grouped=True, samples are written in groups of 9
    (all cameras for the same subject+frame consecutive) to enable
    multi-view batch construction during streaming.

    Args:
        dataset: GazeGeneDataset instance
        output_dir: directory for MDS shards (e.g. ./mds_shards/train)
        split: 'train' or 'val' (metadata only)
        multiview_grouped: sort samples so 9-camera groups are consecutive

    Returns:
        n_samples: number of samples written
    """
    assert MDSWriter is not None, (
        "pip install mosaicml-streaming"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Build ordered index for multi-view grouping
    if multiview_grouped:
        indices = _build_multiview_order(dataset)
    else:
        indices = list(range(len(dataset)))

    print(f"Converting {len(indices)} samples to MDS in {output_dir}")

    with MDSWriter(
        out=output_dir,
        columns=MDS_COLUMNS,
        compression='zstd',
        hashes=['sha1'],
        size_limit=1 << 27,  # 128 MB per shard
    ) as writer:
        for count, idx in enumerate(tqdm(indices, desc=f'MDS {split}')):
            sample = dataset[idx]

            mds_sample = {
                'image': _tensor_image_to_pil(sample['image']),
                'landmark_coords': sample['landmark_coords'].numpy(),
                'landmark_coords_px': sample['landmark_coords_px'].numpy(),
                'optical_axis': sample['optical_axis'].numpy(),
                'R_norm': sample['R_norm'].numpy(),
                'R_kappa': sample['R_kappa'].numpy(),
                'K': sample['K'].numpy(),
                'R_cam': sample['R_cam'].numpy(),
                'T_cam': sample['T_cam'].numpy(),
                'M_norm_inv': sample['M_norm_inv'].numpy(),
                'eyeball_center_3d': sample['eyeball_center_3d'].numpy(),
                'subject': sample['subject'],
                'cam_id': sample['cam_id'],
                'frame_idx': sample['frame_idx'],
            }

            writer.write(mds_sample)

    print(f"Done. {len(indices)} samples written to {output_dir}")
    return len(indices)


def _build_multiview_order(dataset):
    """Return indices ordered so 9-view groups are consecutive."""
    from collections import defaultdict
    groups = defaultdict(list)
    for idx in range(len(dataset)):
        s = dataset.samples[idx]
        key = (s['subject'], s['frame_idx'])
        groups[key].append(idx)

    ordered = []
    incomplete = []
    for key in sorted(groups.keys()):
        idxs = groups[key]
        if len(idxs) == 9:
            idxs.sort(key=lambda i: dataset.samples[i]['cam_id'])
            ordered.extend(idxs)
        else:
            incomplete.extend(idxs)

    ordered.extend(incomplete)
    return ordered


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert GazeGene dataset to MDS format')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'])
    parser.add_argument('--subject_start', type=int, default=1)
    parser.add_argument('--subject_end', type=int, default=46)
    parser.add_argument('--samples_per_subject', type=int, default=None)
    parser.add_argument('--eye', type=str, default='L', choices=['L', 'R'])
    parser.add_argument('--no_multiview_group', action='store_true')
    args = parser.parse_args()

    from RayNet.dataset import GazeGeneDataset
    ds = GazeGeneDataset(
        base_dir=args.data_dir,
        subject_ids=list(range(args.subject_start, args.subject_end + 1)),
        samples_per_subject=args.samples_per_subject,
        eye=args.eye,
        augment=False,
    )
    convert_to_mds(ds, args.output_dir, split=args.split,
                   multiview_grouped=not args.no_multiview_group)


if __name__ == '__main__':
    main()
