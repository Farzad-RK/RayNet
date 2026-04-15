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
import cv2


try:
    from streaming import MDSWriter
except ImportError:
    MDSWriter = None


# MDS column schema — maps field names to MDS types
MDS_COLUMNS = {
    'image': 'bytes',
    'landmark_coords': 'ndarray',
    'landmark_coords_px': 'ndarray',
    'optical_axis': 'ndarray',
    'R_kappa': 'ndarray',
    'K': 'ndarray',                   # K_cropped rescaled to 224
    'intrinsic_original': 'ndarray',  # K_orig (full camera) — for BoxEncoder GT / rederivation
    'face_bbox_gt': 'ndarray',        # (3,) [x_p, y_p, L_x] from Intrinsic Delta
    'R_cam': 'ndarray',
    'T_cam': 'ndarray',
    'eyeball_center_3d': 'ndarray',
    'pupil_center_3d': 'ndarray',
    'head_R': 'ndarray',
    'head_t': 'ndarray',
    'gaze_target': 'ndarray',
    'gaze_depth': 'float32',
    'subject': 'int',
    'cam_id': 'int',
    'frame_idx': 'int',
}


def _sample_to_mds(sample):
    """Assemble the MDS record for one GazeGeneDataset sample."""
    return {
        'image': image_to_jpeg_bytes(sample['image']),
        'landmark_coords': sample['landmark_coords'].numpy(),
        'landmark_coords_px': sample['landmark_coords_px'].numpy(),
        'optical_axis': sample['optical_axis'].numpy(),
        'R_kappa': sample['R_kappa'].numpy(),
        'K': sample['K'].numpy(),
        'intrinsic_original': sample['intrinsic_original'].numpy(),
        'face_bbox_gt': sample['face_bbox_gt'].numpy().astype(np.float32),
        'R_cam': sample['R_cam'].numpy(),
        'T_cam': sample['T_cam'].numpy(),
        'eyeball_center_3d': sample['eyeball_center_3d'].numpy(),
        'pupil_center_3d': sample['pupil_center_3d'].numpy(),
        'head_R': sample['head_R'].numpy(),
        'head_t': sample['head_t'].numpy(),
        'gaze_target': sample['gaze_target'].numpy(),
        'gaze_depth': float(sample['gaze_depth']),
        'subject': sample['subject'],
        'cam_id': sample['cam_id'],
        'frame_idx': sample['frame_idx'],
    }


def image_to_jpeg_bytes(img_tensor, quality=90):
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(".jpg", img, encode_param)

    return buf.tobytes()


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
        compression=None,
        hashes=['sha1'],
        size_limit=1 << 28,  # 256 mb
    ) as writer:
        for count, idx in enumerate(tqdm(indices, desc=f'MDS {split}')):
            writer.write(_sample_to_mds(dataset[idx]))

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


def convert_to_mds_chunked(data_dir, output_dir, subject_ids,
                           split='train', multiview_grouped=True,
                           samples_per_subject=None, eye='L',
                           chunk_size=3):
    """
    Convert GazeGene dataset to MDS shards, loading subjects in chunks
    to avoid exceeding RAM.

    Args:
        data_dir: path to GazeGene_FaceCrops root
        output_dir: directory for MDS shards
        subject_ids: list of subject IDs to convert
        split: 'train' or 'val' (metadata)
        multiview_grouped: sort so 9-camera groups are consecutive
        samples_per_subject: limit frames per subject (None = all)
        eye: which eye ('L' or 'R')
        chunk_size: number of subjects to load at a time

    Returns:
        total number of samples written
    """
    assert MDSWriter is not None, "pip install mosaicml-streaming"
    os.makedirs(output_dir, exist_ok=True)

    from RayNet.dataset import GazeGeneDataset
    import gc

    total_written = 0
    chunks = [subject_ids[i:i + chunk_size]
              for i in range(0, len(subject_ids), chunk_size)]

    print(f"Converting {len(subject_ids)} subjects in "
          f"{len(chunks)} chunks of ≤{chunk_size}")

    with MDSWriter(
        out=output_dir,
        columns=MDS_COLUMNS,
        compression=None,
        hashes=['sha1'],
        size_limit=1 << 28,  # 256 mb
    ) as writer:
        for ci, chunk_subjs in enumerate(chunks):
            print(f"\nChunk {ci + 1}/{len(chunks)}: "
                  f"subjects {chunk_subjs}")

            ds = GazeGeneDataset(
                base_dir=data_dir,
                subject_ids=chunk_subjs,
                samples_per_subject=samples_per_subject,
                eye=eye,
                augment=False,
            )

            if multiview_grouped:
                indices = _build_multiview_order(ds)
            else:
                indices = list(range(len(ds)))

            for idx in tqdm(indices,
                            desc=f'MDS {split} chunk {ci + 1}'):
                writer.write(_sample_to_mds(ds[idx]))

            total_written += len(indices)

            # Free memory before loading next chunk
            del ds, indices
            gc.collect()

    print(f"\nDone. {total_written} samples written to {output_dir}")
    return total_written


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
    parser.add_argument('--chunk_size', type=int, default=3,
                        help='Subjects to load at a time (default: 3)')
    args = parser.parse_args()

    subject_ids = list(range(args.subject_start, args.subject_end + 1))

    convert_to_mds_chunked(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        subject_ids=subject_ids,
        split=args.split,
        multiview_grouped=not args.no_multiview_group,
        samples_per_subject=args.samples_per_subject,
        eye=args.eye,
        chunk_size=args.chunk_size,
    )


if __name__ == '__main__':
    main()
