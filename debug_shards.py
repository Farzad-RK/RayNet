"""Quick diagnostic: check if images in MDS shards are actually zero."""
import sys
import numpy as np

from streaming import StreamingDataset
from PIL import Image

shard_path = sys.argv[1] if len(sys.argv) > 1 else './mosaic_shards_test'

ds = StreamingDataset(local=shard_path, shuffle=False)
print(f"Total samples: {len(ds)}")

n_zero = 0
n_checked = min(20, len(ds))

for i in range(n_checked):
    raw = ds[i]
    img = raw['image']

    print(f"\n--- Sample {i} ---")
    print(f"  raw type: {type(img)}")

    if isinstance(img, Image.Image):
        print(f"  PIL mode={img.mode}, size={img.size}")
        arr = np.array(img)
    elif isinstance(img, bytes):
        print(f"  bytes len={len(img)}, header={img[:4]}")
        img_pil = Image.open(__import__('io').BytesIO(img))
        arr = np.array(img_pil)
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        print(f"  unexpected type!")
        continue

    print(f"  shape={arr.shape}, dtype={arr.dtype}")
    print(f"  min={arr.min()}, max={arr.max()}, mean={arr.mean():.4f}")
    print(f"  nonzero={np.count_nonzero(arr)}/{arr.size}")

    if arr.max() == 0:
        n_zero += 1

print(f"\n=== {n_zero}/{n_checked} samples have all-zero images ===")
