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

import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2

try:
    from streaming import StreamingDataset
except ImportError:
    StreamingDataset = None

log = logging.getLogger(__name__)

# Defer class definition so the module can be imported even when
# mosaicml-streaming is not installed.  Functions that actually
# *use* the class guard with an assert at runtime.
_Base = StreamingDataset if StreamingDataset is not None else object


class StreamingGazeGeneDataset(_Base):

    """
    Streaming dataset that reads MDS shards and returns sample dicts
    matching GazeGeneDataset.__getitem__ format.

    Supports local, S3, MinIO, GCS, and OCI backends transparently
    via mosaicml-streaming.
    """

    def __init__(self, transform=None, samples_per_subject=None,
                 eyelid_occlusion_p=0.0, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform
        self.samples_per_subject = samples_per_subject
        # Eyelid-occlusion augmentation probability (per-sample, train only).
        # 0.0 disables it entirely; values in [0.2, 0.4] are a reasonable
        # range for AERI robustness tuning. See streaming/occlusion_aug.py.
        self.eyelid_occlusion_p = float(eyelid_occlusion_p)

    def __getitem__(self, idx):
        raw = super().__getitem__(idx)

        # Stateless and deterministic subsetting using frame_idx.
        if self.samples_per_subject is not None:
            if int(raw['frame_idx']) >= self.samples_per_subject:
                return None

        # 1. Get the JPEG bytes from MDS
        img_bytes = raw['image']

        # 2. Decode bytes to numpy (BGR)
        # np.frombuffer is O(1) memory view, very fast
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 3. Convert BGR to RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # 4. Convert to Torch Tensor (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().float().div(255.0)

        # 4a. Eyelid-occlusion augmentation (BEFORE Normalize). Uses the
        # GT eyeball_mask to find the eye bbox; the GT mask itself is
        # NOT modified — the seg head must learn to predict the full
        # silhouette through the synthetic occluder. See
        # streaming/occlusion_aug.py for the rationale.
        if self.eyelid_occlusion_p > 0 and 'eyeball_mask' in raw:
            from RayNet.streaming.occlusion_aug import random_eyelid_occlusion
            eb_tensor = torch.from_numpy(np.ascontiguousarray(raw['eyeball_mask']))
            img = random_eyelid_occlusion(img, eb_tensor,
                                          p=self.eyelid_occlusion_p)

        if self.transform is not None:
            img = self.transform(img)

        return {
            'image': img,
            'landmark_coords': torch.from_numpy(
                raw['landmark_coords'].astype(np.float32)),
            'landmark_coords_px': torch.from_numpy(
                raw['landmark_coords_px'].astype(np.float32)),
            'optical_axis': torch.from_numpy(
                raw['optical_axis'].astype(np.float32)),
            'R_kappa': torch.from_numpy(
                raw['R_kappa'].astype(np.float32)),
            'K': torch.from_numpy(
                raw['K'].astype(np.float32)),
            'intrinsic_original': torch.from_numpy(
                raw['intrinsic_original'].astype(np.float32))
                if 'intrinsic_original' in raw else torch.zeros(3, 3),
            'face_bbox_gt': torch.from_numpy(
                raw['face_bbox_gt'].astype(np.float32))
                if 'face_bbox_gt' in raw else torch.zeros(3),
            'R_cam': torch.from_numpy(
                raw['R_cam'].astype(np.float32)),
            'T_cam': torch.from_numpy(
                raw['T_cam'].astype(np.float32)),
            'eyeball_center_3d': torch.from_numpy(
                raw['eyeball_center_3d'].astype(np.float32)),
            'pupil_center_3d': torch.from_numpy(
                raw['pupil_center_3d'].astype(np.float32))
                if 'pupil_center_3d' in raw else torch.zeros(3),
            'head_R': torch.from_numpy(
                raw['head_R'].astype(np.float32))
                if 'head_R' in raw else torch.eye(3),
            'head_t': torch.from_numpy(
                raw['head_t'].astype(np.float32))
                if 'head_t' in raw else torch.zeros(3),
            'gaze_target': torch.from_numpy(
                raw['gaze_target'].astype(np.float32))
                if 'gaze_target' in raw else torch.zeros(3),
            'gaze_depth': torch.tensor(
                float(raw['gaze_depth']), dtype=torch.float32)
                if 'gaze_depth' in raw else torch.tensor(0.0),
            'iris_mask': torch.from_numpy(
                np.ascontiguousarray(raw['iris_mask']))
                if 'iris_mask' in raw else torch.zeros(56, 56, dtype=torch.uint8),
            'eyeball_mask': torch.from_numpy(
                np.ascontiguousarray(raw['eyeball_mask']))
                if 'eyeball_mask' in raw else torch.zeros(56, 56, dtype=torch.uint8),
            'subject': int(raw['subject']),
            'cam_id': int(raw['cam_id']),
            'frame_idx': int(raw['frame_idx']),
        }


class NonEmptyBatchLoader:
    """Wraps a DataLoader, transparently skipping empty batches.

    When ``samples_per_subject`` filters out all samples in a micro-batch,
    the collate_fn returns an empty dict ``{}``. This wrapper silently
    drops those so the training loop sees a clean sequence of real batches:
    batch counters and gradient-accumulation boundaries stay consistent.

    ``__len__`` is an UPPER BOUND (the underlying loader's length) because
    the actual number of non-empty batches per epoch depends on filtering
    and cannot be known ahead of iteration.
    """

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for batch in self._loader:
            if batch and len(batch) > 0:
                yield batch

    def __len__(self):
        return len(self._loader)

    @property
    def dataset(self):
        return self._loader.dataset


def _collate_fn(batch):
    """Collate matching gazegene_collate_fn format."""

    # This for the implementation of samples_per_subject
    # Filter out skipped samples (None)
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}

    collated = {}
    tensor_keys = [
        'image', 'landmark_coords', 'landmark_coords_px',
        'optical_axis', 'R_kappa',
        'K', 'intrinsic_original', 'face_bbox_gt',
        'R_cam', 'T_cam', 'eyeball_center_3d', 'pupil_center_3d',
        'head_R', 'head_t', 'gaze_target', 'gaze_depth',
        'iris_mask', 'eyeball_mask',
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
    transform=None,
    val_transform=None,
    shuffle_train=True,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=False,
    samples_per_subject=None,
    eyelid_occlusion_p=0.0,
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
        transform: Optional torchvision transform to apply to image tensors.main
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
        transform=transform,
        remote=None,
        local=remote_train,
        split=None,
        shuffle=shuffle_train,
        batch_size=batch_size,
        samples_per_subject=samples_per_subject,
        eyelid_occlusion_p=eyelid_occlusion_p,
        **streaming_kwargs,
    )

    val_dataset = StreamingGazeGeneDataset(
        transform=val_transform,
        remote=None,
        local=remote_val,
        split=None,
        shuffle=False,
        batch_size=batch_size,
        samples_per_subject=samples_per_subject,
        eyelid_occlusion_p=0.0,            # never augment val
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

    # Wrap with empty-batch skipper when samples_per_subject filtering is on.
    # Without this, fully-filtered batches reach the training loop and pollute
    # step counters, batch_log.csv row numbering, and gradient-accumulation
    # boundaries (they are handled via `continue`, but at the cost of wasted
    # iterations and confusing logs).
    if samples_per_subject is not None:
        train_loader = NonEmptyBatchLoader(train_loader)
        val_loader = NonEmptyBatchLoader(val_loader)

    return train_loader, val_loader


def create_multiview_streaming_dataloaders(
    remote_train,
    remote_val,
    local_cache='./mds_cache',
    mv_groups=2,
    num_workers=4,
    transform=None,
    val_transform=None,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=False,
    samples_per_subject=None,
    eyelid_occlusion_p=0.0,
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
        transform: Train-only transform (ColorJitter, RandomAffine, etc.).
        val_transform: Val-only transform. Should at minimum apply the same
            Normalize step as the end of `transform` so that the shared
            stem's BN running_mean/var — calibrated on train images —
            produce correctly-scaled activations for val images too.
        pin_memory: Pin memory for GPU transfer.
        prefetch_factor: Prefetch factor per worker.
        persistent_workers: Keep workers alive between epochs.
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
        transform=transform,
        val_transform=val_transform,
        shuffle_train=False,  # preserve multi-view grouping order
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        samples_per_subject=samples_per_subject,
        eyelid_occlusion_p=eyelid_occlusion_p,
        **streaming_kwargs,
    )


