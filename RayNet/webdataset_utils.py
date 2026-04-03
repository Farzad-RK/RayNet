"""
WebDataset utilities for RayNet.

Provides tools to:
  1. Convert GazeGeneDataset to WebDataset .tar shards (pre-processed)
  2. Push shards to Hugging Face Hub
  3. Stream shards back as a PyTorch-compatible DataLoader
  4. Multi-view streaming with grouped (subject, frame) batches

Usage:
    # Create shards from local dataset
    python -m RayNet.webdataset_utils create_shards \
        --data_dir /path/to/gazegene --output_dir ./shards --split train

    # Push to HF Hub
    python -m RayNet.webdataset_utils push \
        --shard_dir ./shards/train --repo_id user/gazegene-wds --split train
"""

import os
import io
import json
import math
import numpy as np
import torch
from torch.utils.data import IterableDataset
from collections import defaultdict

try:
    import webdataset as wds
except ImportError:
    wds = None

try:
    from huggingface_hub import HfApi, hf_hub_url
except ImportError:
    HfApi = None
    hf_hub_url = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All tensor fields produced by GazeGeneDataset.__getitem__
TENSOR_FIELDS = [
    'landmark_coords', 'landmark_coords_px', 'optical_axis',
    'R_norm', 'R_kappa', 'K', 'R_cam', 'T_cam',
    'M_norm_inv', 'eyeball_center_3d',
]

METADATA_FIELDS = ['subject', 'cam_id', 'frame_idx']

N_CAMERAS = 9  # GazeGene synchronized camera count


# ---------------------------------------------------------------------------
# Shard Creation
# ---------------------------------------------------------------------------

def _tensor_to_npy_bytes(tensor):
    """Serialize a torch tensor to .npy bytes."""
    buf = io.BytesIO()
    np.save(buf, tensor.numpy())
    return buf.getvalue()


def _image_to_jpeg_bytes(img_tensor, quality=95):
    """Convert (3, H, W) float [0,1] tensor to JPEG bytes."""
    import cv2
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def create_webdataset_shards(dataset, output_dir, samples_per_shard=1000,
                              split='train', multiview_grouped=True):
    """
    Convert a GazeGeneDataset into WebDataset .tar shards.

    When multiview_grouped=True, samples are sorted so all 9 views of the
    same (subject, frame) are consecutive within a shard. This enables
    multi-view streaming without cross-shard reads.

    Args:
        dataset: GazeGeneDataset instance
        output_dir: directory to write shards into (e.g. ./shards/train)
        samples_per_shard: max samples per .tar file
        split: 'train' or 'val' (used in shard naming)
        multiview_grouped: if True, group all 9 views together
    """
    assert wds is not None, "pip install webdataset"
    os.makedirs(output_dir, exist_ok=True)

    # Build ordered index for multi-view grouping
    if multiview_grouped:
        indices = _build_multiview_order(dataset)
    else:
        indices = list(range(len(dataset)))

    shard_pattern = os.path.join(output_dir, f"gazegene-{split}-%06d.tar")
    n_shards = math.ceil(len(indices) / samples_per_shard)

    print(f"Creating {n_shards} shards ({len(indices)} samples, "
          f"{samples_per_shard}/shard) in {output_dir}")

    with wds.ShardWriter(shard_pattern, maxcount=samples_per_shard) as sink:
        for count, idx in enumerate(indices):
            sample = dataset[idx]

            # Build unique key
            subj = sample['subject']
            cam = sample['cam_id']
            frame = sample['frame_idx']
            key = f"{subj:04d}_{cam:01d}_{frame:06d}"

            wds_sample = {"__key__": key}

            # Image as JPEG
            wds_sample["image.jpg"] = _image_to_jpeg_bytes(sample['image'])

            # Tensor fields as .npy
            for field in TENSOR_FIELDS:
                wds_sample[f"{field}.npy"] = _tensor_to_npy_bytes(sample[field])

            # Metadata as JSON
            meta = {f: sample[f] for f in METADATA_FIELDS}
            wds_sample["metadata.json"] = json.dumps(meta).encode('utf-8')

            sink.write(wds_sample)

            if (count + 1) % 5000 == 0:
                print(f"  Written {count + 1}/{len(indices)} samples")

    print(f"Done. Shards written to {output_dir}")
    return n_shards


def _build_multiview_order(dataset):
    """
    Return sample indices ordered so all 9 views of the same (subject, frame)
    are consecutive. Groups with fewer than 9 views are placed at the end.
    """
    groups = defaultdict(list)
    for idx in range(len(dataset)):
        s = dataset.samples[idx]
        key = (s['subject'], s['frame_idx'])
        groups[key].append(idx)

    ordered = []
    incomplete = []

    for key in sorted(groups.keys()):
        idxs = groups[key]
        if len(idxs) == N_CAMERAS:
            # Sort by cam_id for deterministic ordering
            idxs.sort(key=lambda i: dataset.samples[i]['cam_id'])
            ordered.extend(idxs)
        else:
            incomplete.extend(idxs)

    ordered.extend(incomplete)
    return ordered


# ---------------------------------------------------------------------------
# Hugging Face Hub Push
# ---------------------------------------------------------------------------

def push_shards_to_hub(shard_dir, repo_id, split='train', private=True):
    """
    Upload WebDataset shards to a Hugging Face Hub dataset repository.

    Args:
        shard_dir: local directory containing .tar shard files
        repo_id: HF repo like "username/gazegene-webdataset"
        split: subfolder on hub ("train" or "val")
        private: whether the repo should be private
    """
    assert HfApi is not None, "pip install huggingface_hub"
    api = HfApi()

    # Create repo if it doesn't exist
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    # Upload all tar files
    tar_files = sorted(f for f in os.listdir(shard_dir) if f.endswith('.tar'))
    print(f"Uploading {len(tar_files)} shards to {repo_id}/{split}/")

    for i, fname in enumerate(tar_files):
        local_path = os.path.join(shard_dir, fname)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{split}/{fname}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"  [{i+1}/{len(tar_files)}] {fname}")

    print(f"Done. Dataset available at: https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Streaming Dataset
# ---------------------------------------------------------------------------

def _deserialize_sample(sample):
    """
    Convert a raw WebDataset sample dict back into the tensor dict
    matching GazeGeneDataset.__getitem__ output.
    """
    import cv2

    result = {}

    # Image: JPEG bytes -> (3, 224, 224) float tensor
    img_bytes = sample['image.jpg']
    img_np = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    result['image'] = torch.from_numpy(
        img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)

    # Tensor fields: .npy bytes -> torch tensor
    for field in TENSOR_FIELDS:
        npy_key = f'{field}.npy'
        arr = np.load(io.BytesIO(sample[npy_key]))
        result[field] = torch.from_numpy(arr.astype(np.float32))

    # Metadata
    meta = json.loads(sample['metadata.json'])
    result['subject'] = meta['subject']
    result['cam_id'] = meta['cam_id']
    result['frame_idx'] = meta['frame_idx']

    return result


def _streaming_collate_fn(batch):
    """Collate for streaming dataloader — same as gazegene_collate_fn."""
    if not batch:
        return {}

    collated = {}
    tensor_keys = ['image', 'landmark_coords', 'landmark_coords_px',
                   'optical_axis', 'R_norm', 'R_kappa',
                   'K', 'R_cam', 'T_cam', 'M_norm_inv', 'eyeball_center_3d']
    scalar_keys = ['subject', 'cam_id', 'frame_idx']

    for key in tensor_keys:
        if key in batch[0]:
            collated[key] = torch.stack([s[key] for s in batch])

    for key in scalar_keys:
        if key in batch[0]:
            collated[key] = [s[key] for s in batch]

    return collated


def create_streaming_dataloader(urls, batch_size=512, num_workers=4,
                                 shuffle=True, epoch_length=None):
    """
    Create a streaming DataLoader from WebDataset tar URLs.

    Args:
        urls: shard URLs or paths. Supports:
            - Local glob: "./shards/train/gazegene-train-{000000..000099}.tar"
            - HF Hub: "pipe:curl -sL https://huggingface.co/.../resolve/main/train/gazegene-train-{000000..000099}.tar"
            - List of URLs
        batch_size: samples per batch
        num_workers: dataloader workers
        shuffle: whether to shuffle shards and samples
        epoch_length: if set, limits the number of batches per epoch
            (useful for streaming datasets with unknown length)

    Returns:
        DataLoader-compatible iterable
    """
    assert wds is not None, "pip install webdataset"

    dataset = wds.WebDataset(urls, shardshuffle=shuffle)

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.map(_deserialize_sample)

    # Use wds.WebLoader for proper multi-worker streaming
    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_streaming_collate_fn,
        pin_memory=True,
    )

    if shuffle:
        loader = loader.shuffle(4)  # shuffle across batches

    if epoch_length is not None:
        loader = loader.with_epoch(epoch_length)

    return loader


# ---------------------------------------------------------------------------
# Multi-View Streaming
# ---------------------------------------------------------------------------

class MultiViewStreamingDataset(IterableDataset):
    """
    Streaming dataset that yields multi-view groups of 9 consecutive samples.

    Expects shards created with multiview_grouped=True, where all 9 views
    of the same (subject, frame) are consecutive.

    Each iteration yields a dict with tensors of shape (9, ...) for one group.
    """

    def __init__(self, urls, shuffle=True):
        assert wds is not None, "pip install webdataset"
        self.urls = urls
        self.shuffle = shuffle

    def __iter__(self):
        dataset = wds.WebDataset(self.urls, shardshuffle=self.shuffle)
        if self.shuffle:
            dataset = dataset.shuffle(100)
        dataset = dataset.map(_deserialize_sample)

        buffer = []
        current_key = None

        for sample in dataset:
            key = (sample['subject'], sample['frame_idx'])

            if current_key is not None and key != current_key:
                # Emit the completed group
                if len(buffer) == N_CAMERAS:
                    yield self._stack_group(buffer)
                buffer = []

            current_key = key
            buffer.append(sample)

            if len(buffer) == N_CAMERAS:
                yield self._stack_group(buffer)
                buffer = []
                current_key = None

        # Don't yield incomplete groups

    @staticmethod
    def _stack_group(samples):
        """Stack 9 samples into a multi-view group dict."""
        result = {}
        tensor_keys = ['image', 'landmark_coords', 'landmark_coords_px',
                       'optical_axis', 'R_norm', 'R_kappa',
                       'K', 'R_cam', 'T_cam', 'M_norm_inv', 'eyeball_center_3d']

        for key in tensor_keys:
            result[key] = torch.stack([s[key] for s in samples])  # (9, ...)

        result['subject'] = samples[0]['subject']
        result['cam_id'] = [s['cam_id'] for s in samples]
        result['frame_idx'] = samples[0]['frame_idx']

        return result


def _multiview_collate_fn(groups):
    """
    Collate multi-view groups into flat batch matching train.py expectations.

    Input: list of dicts where tensors have shape (9, ...)
    Output: dict where tensors have shape (G*9, ...) — flat batch
    """
    if not groups:
        return {}

    collated = {}
    tensor_keys = ['image', 'landmark_coords', 'landmark_coords_px',
                   'optical_axis', 'R_norm', 'R_kappa',
                   'K', 'R_cam', 'T_cam', 'M_norm_inv', 'eyeball_center_3d']

    for key in tensor_keys:
        stacked = torch.stack([g[key] for g in groups])  # (G, 9, ...)
        G, V = stacked.shape[:2]
        collated[key] = stacked.reshape(G * V, *stacked.shape[2:])

    collated['subject'] = []
    collated['cam_id'] = []
    collated['frame_idx'] = []
    for g in groups:
        collated['subject'].extend([g['subject']] * N_CAMERAS)
        collated['cam_id'].extend(g['cam_id'])
        collated['frame_idx'].extend([g['frame_idx']] * N_CAMERAS)

    return collated


def create_multiview_streaming_dataloader(urls, mv_groups=2, num_workers=4,
                                           shuffle=True, epoch_length=None):
    """
    Create a streaming DataLoader that yields multi-view grouped batches.

    Each batch contains mv_groups * 9 samples, with consecutive groups of 9
    being from the same (subject, frame).

    Args:
        urls: shard URLs (must be created with multiview_grouped=True)
        mv_groups: number of (subject, frame) groups per batch
        num_workers: dataloader workers
        shuffle: whether to shuffle
        epoch_length: batches per epoch

    Returns:
        DataLoader-compatible iterable
    """
    assert wds is not None, "pip install webdataset"

    mv_dataset = MultiViewStreamingDataset(urls, shuffle=shuffle)

    loader = torch.utils.data.DataLoader(
        mv_dataset,
        batch_size=mv_groups,
        num_workers=num_workers,
        collate_fn=_multiview_collate_fn,
        pin_memory=True,
    )

    return loader


# ---------------------------------------------------------------------------
# HF Hub URL Builder
# ---------------------------------------------------------------------------

def hf_hub_shard_urls(repo_id, split='train', n_shards=None, shard_pattern=None):
    """
    Build WebDataset-compatible URLs for shards hosted on HF Hub.

    Args:
        repo_id: e.g. "username/gazegene-webdataset"
        split: "train" or "val"
        n_shards: number of shards (if using brace expansion)
        shard_pattern: override pattern (e.g. "gazegene-train-{000000..000099}.tar")

    Returns:
        URL string or list suitable for wds.WebDataset()
    """
    base = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{split}"

    if shard_pattern is not None:
        return f"pipe:curl -sL {base}/{shard_pattern}"
    elif n_shards is not None:
        last = n_shards - 1
        return f"pipe:curl -sL {base}/gazegene-{split}-{{000000..{last:06d}}}.tar"
    else:
        raise ValueError("Provide either n_shards or shard_pattern")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='RayNet WebDataset utilities')
    subparsers = parser.add_subparsers(dest='command')

    # create_shards
    sp = subparsers.add_parser('create_shards', help='Convert dataset to shards')
    sp.add_argument('--data_dir', type=str, required=True)
    sp.add_argument('--output_dir', type=str, required=True)
    sp.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    sp.add_argument('--samples_per_shard', type=int, default=1000)
    sp.add_argument('--samples_per_subject', type=int, default=None)
    sp.add_argument('--eye', type=str, default='L', choices=['L', 'R'])
    sp.add_argument('--subject_start', type=int, default=1)
    sp.add_argument('--subject_end', type=int, default=47)

    # push
    sp2 = subparsers.add_parser('push', help='Push shards to HF Hub')
    sp2.add_argument('--shard_dir', type=str, required=True)
    sp2.add_argument('--repo_id', type=str, required=True)
    sp2.add_argument('--split', type=str, default='train')
    sp2.add_argument('--private', action='store_true', default=True)

    args = parser.parse_args()

    if args.command == 'create_shards':
        from RayNet.dataset import GazeGeneDataset
        subject_ids = list(range(args.subject_start, args.subject_end + 1))
        dataset = GazeGeneDataset(
            base_dir=args.data_dir,
            subject_ids=subject_ids,
            samples_per_subject=args.samples_per_subject,
            eye=args.eye,
            augment=False,
        )
        create_webdataset_shards(
            dataset, args.output_dir,
            samples_per_shard=args.samples_per_shard,
            split=args.split,
        )

    elif args.command == 'push':
        push_shards_to_hub(args.shard_dir, args.repo_id, args.split)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
