"""
MosaicML Streaming dataset for RayNet.

Streams pre-processed GazeGene samples from MDS shards stored locally,
on S3-compatible storage (MinIO), or any cloud path supported by
mosaicml-streaming.

Usage:
    from RayNet.streaming import StreamingGazeGeneDataset, create_streaming_dataloaders

    # From MinIO
    train_loader, val_loader = create_streaming_dataloaders(
        remote_train='s3://gazegene/train',
        remote_val='s3://gazegene/val',
        local_cache='/tmp/mds_cache',
        batch_size=2048,
    )
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from streaming import StreamingDataset
except ImportError:
    StreamingDataset = None


class StreamingGazeGeneDataset(StreamingDataset):
    """
    Streaming dataset that reads MDS shards and returns sample dicts
    matching GazeGeneDataset.__getitem__ format.

    Supports local, S3, MinIO, GCS, and OCI backends transparently
    via mosaicml-streaming.
    """

    def __getitem__(self, idx):
        raw = super().__getitem__(idx)

        # Image: PIL -> (3, 224, 224) float tensor
        img = raw['image']
        if not isinstance(img, torch.Tensor):
            img_np = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(img_np.transpose(2, 0, 1))

        return {
            'image': img,
            'landmark_coords': torch.from_numpy(
                raw['landmark_coords'].astype(np.float32)),
            'landmark_coords_px': torch.from_numpy(
                raw['landmark_coords_px'].astype(np.float32)),
            'optical_axis': torch.from_numpy(
                raw['optical_axis'].astype(np.float32)),
            'R_norm': torch.from_numpy(
                raw['R_norm'].astype(np.float32)),
            'R_kappa': torch.from_numpy(
                raw['R_kappa'].astype(np.float32)),
            'K': torch.from_numpy(
                raw['K'].astype(np.float32)),
            'R_cam': torch.from_numpy(
                raw['R_cam'].astype(np.float32)),
            'T_cam': torch.from_numpy(
                raw['T_cam'].astype(np.float32)),
            'M_norm_inv': torch.from_numpy(
                raw['M_norm_inv'].astype(np.float32)),
            'eyeball_center_3d': torch.from_numpy(
                raw['eyeball_center_3d'].astype(np.float32)),
            'subject': int(raw['subject']),
            'cam_id': int(raw['cam_id']),
            'frame_idx': int(raw['frame_idx']),
        }


def _collate_fn(batch):
    """Collate matching gazegene_collate_fn format."""
    if not batch:
        return {}

    collated = {}
    tensor_keys = [
        'image', 'landmark_coords', 'landmark_coords_px',
        'optical_axis', 'R_norm', 'R_kappa',
        'K', 'R_cam', 'T_cam', 'M_norm_inv', 'eyeball_center_3d',
    ]
    scalar_keys = ['subject', 'cam_id', 'frame_idx']

    for key in tensor_keys:
        if key in batch[0]:
            collated[key] = torch.stack([s[key] for s in batch])

    for key in scalar_keys:
        if key in batch[0]:
            collated[key] = [s[key] for s in batch]

    return collated


def create_streaming_dataloaders(
    remote_train,
    remote_val,
    local_cache='./mds_cache',
    batch_size=512,
    num_workers=4,
    shuffle_train=True,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=False,
    **streaming_kwargs,
):
    """
    Create train and val dataloaders from MDS shards.

    Args:
        remote_train: Remote path for training shards.
            - MinIO: 's3://bucket/train'
            - Local: '/path/to/mds_shards/train'
            - GCS: 'gs://bucket/train'
        remote_val: Remote path for validation shards.
        local_cache: Local directory for shard caching.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        shuffle_train: Shuffle training data.
        pin_memory: Pin memory for GPU transfer.
        prefetch_factor: Prefetch factor per worker.
        persistent_workers: Keep workers alive between epochs.
        **streaming_kwargs: Extra kwargs for StreamingDataset
            (e.g., download_retry, download_timeout).

    Returns:
        (train_loader, val_loader)

    Example with MinIO:
        # Requires env vars: S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
        train_loader, val_loader = create_streaming_dataloaders(
            remote_train='s3://gazegene/train',
            remote_val='s3://gazegene/val',
        )
    """
    assert StreamingDataset is not None, (
        "pip install mosaicml-streaming"
    )

    train_local = os.path.join(local_cache, 'train')
    val_local = os.path.join(local_cache, 'val')

    train_dataset = StreamingGazeGeneDataset(
        remote=remote_train,
        local=train_local,
        split=None,
        shuffle=shuffle_train,
        batch_size=batch_size,
        **streaming_kwargs,
    )

    val_dataset = StreamingGazeGeneDataset(
        remote=remote_val,
        local=val_local,
        split=None,
        shuffle=False,
        batch_size=batch_size,
        **streaming_kwargs,
    )

    loader_kwargs = dict(
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, **loader_kwargs)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, **loader_kwargs)

    return train_loader, val_loader


def create_multiview_streaming_dataloaders(
    remote_train,
    remote_val,
    local_cache='./mds_cache',
    mv_groups=2,
    num_workers=4,
    **streaming_kwargs,
):
    """
    Create multi-view dataloaders from MDS shards.

    Shards must have been created with multiview_grouped=True so that
    9 consecutive samples form one (subject, frame) group.

    Batch size = mv_groups * 9.

    Args:
        remote_train: Remote path for training MDS shards.
        remote_val: Remote path for validation MDS shards.
        local_cache: Local cache directory.
        mv_groups: Number of multi-view groups per batch.
        num_workers: DataLoader workers.
        **streaming_kwargs: Extra kwargs for StreamingDataset.

    Returns:
        (train_loader, val_loader)
    """
    actual_batch = mv_groups * 9

    return create_streaming_dataloaders(
        remote_train=remote_train,
        remote_val=remote_val,
        local_cache=local_cache,
        batch_size=actual_batch,
        num_workers=num_workers,
        shuffle_train=False,  # preserve multi-view grouping order
        **streaming_kwargs,
    )


import os  # noqa: E402 — needed by create_streaming_dataloaders
