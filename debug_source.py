"""Check if source GazeGeneDataset produces all-zero images BEFORE MDS conversion."""
import sys
import numpy as np
import torch

from RayNet.dataset import GazeGeneDataset
from RayNet.streaming.convert_to_mds import _tensor_image_to_pil

# Adjust these to match your conversion command
data_dir = sys.argv[1] if len(sys.argv) > 1 else '/path/to/GazeGene_FaceCrops'
subject_ids = [1, 2, 3]  # first chunk

ds = GazeGeneDataset(
    base_dir=data_dir,
    subject_ids=subject_ids,
    eye='L',
    augment=False,
)
print(f"Source dataset: {len(ds)} samples")

for i in range(min(10, len(ds))):
    sample = ds[i]
    img_t = sample['image']  # (3, 224, 224) float [0,1]

    print(f"\n--- Sample {i} (subj={sample['subject']}, cam={sample['cam_id']}) ---")
    print(f"  tensor shape={img_t.shape}, dtype={img_t.dtype}")
    print(f"  min={img_t.min():.4f}, max={img_t.max():.4f}, mean={img_t.mean():.4f}")
    print(f"  nonzero={torch.count_nonzero(img_t)}/{img_t.numel()}")

    # Also test the PIL conversion used by the MDS writer
    pil_img = _tensor_image_to_pil(img_t)
    arr = np.array(pil_img)
    print(f"  after _tensor_image_to_pil: min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}")
