"""
Eyelid-occlusion augmentation for AERI robustness.

Trains the AERI segmentation head and downstream HRFH gaze pipeline to
predict the FULL theoretical eyeball silhouette (and the full iris
contour) even when an eyelid covers part of the sclera at inference
time — i.e. OpenFace-style robustness to drowsy / partial-blink frames.

How it works
------------
The GT iris and eyeball masks baked into the MDS shards are the
THEORETICAL silhouettes (no eyelid clip — see streaming/eye_masks.py).
We exploit that by darkening / replacing pixels in the upper portion of
the visible eye region while leaving the GT mask untouched. The seg
head therefore receives:

  image  : eye partially covered by a synthetic eyelid-like patch
  GT     : the full un-occluded silhouette

so it must learn to extrapolate the silhouette from anatomical context
(eyebrow, lashes, sclera-skin boundary) rather than relying on a crisp
iris/sclera edge.  This breaks the shortcut that the prior runs were
exploiting and is the same inductive bias that makes part-based models
like CLNF (OpenFace) robust to partial occlusion despite using older
backbones.

The augmentation is keyed off the (downsampled, 56x56) eyeball_mask
already in the shard — no extra GT plumbing is required.

Designed to be cheap (no GPU, no CV2 polygon ops) so the dataloader
worker stays I/O bound.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def random_eyelid_occlusion(
    image: torch.Tensor,
    eyeball_mask_56: torch.Tensor,
    p: float = 0.30,
    cover_frac_range: tuple = (0.20, 0.55),
    feather_px: int = 4,
    skin_tone_jitter: float = 0.10,
) -> torch.Tensor:
    """
    With probability `p`, paint a downward-facing eyelid-like band over
    the top portion of the eye region inside `image` and return the
    modified image. With probability 1-p the image is returned
    unchanged.

    The augmentation is mask-anchored: it uses the eyeball_mask bounding
    box (upsampled from 56x56 to the image resolution) to locate the
    eye, then covers a `cover_frac` fraction of the box height starting
    from the top. The cover colour is sampled from skin tones nearby
    so the band looks plausible.

    Args:
        image           : (3, H, W) float tensor in [0, 1] or normalised
                          floats (the augmentation does not care about
                          the absolute scale, it samples local pixel
                          values for its skin tone).
        eyeball_mask_56 : (56, 56) uint8 mask in {0, 255} from the shard.
        p               : per-sample probability of applying the augment.
        cover_frac_range: (min, max) fraction of the eye-bbox HEIGHT to
                          cover with the synthetic eyelid.
        feather_px      : soft edge size at the eyelid bottom edge so
                          the boundary is not razor-sharp (more realistic).
        skin_tone_jitter: ± per-channel multiplicative noise on the
                          sampled skin colour, so the model can't rote
                          memorise the augmenter's colour palette.

    Returns:
        (3, H, W) tensor — same dtype/device as the input.
    """
    if p <= 0 or random.random() >= p:
        return image
    if image.dim() != 3 or image.shape[0] != 3:
        return image
    if eyeball_mask_56 is None or eyeball_mask_56.numel() == 0:
        return image

    _, H, W = image.shape

    # Upsample the 56x56 eyeball mask to image resolution.  Use nearest:
    # the bbox edges only need to be approximate.
    mask = eyeball_mask_56.to(torch.float32)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    mask_up = F.interpolate(
        mask, size=(H, W), mode='nearest'
    ).squeeze(0).squeeze(0)

    ys, xs = torch.where(mask_up > 0)
    if ys.numel() < 16:                     # Too small: bail out cleanly.
        return image

    y_min = int(ys.min().item())
    y_max = int(ys.max().item())
    x_min = int(xs.min().item())
    x_max = int(xs.max().item())

    box_h = y_max - y_min + 1
    box_w = x_max - x_min + 1
    if box_h < 6 or box_w < 6:
        return image

    cover_frac = random.uniform(*cover_frac_range)
    cover_h = max(2, int(round(box_h * cover_frac)))
    cover_y_end = min(H - 1, y_min + cover_h)

    # Sample a skin tone from the strip just BELOW the eye box (cheek).
    cheek_y = min(H - 4, y_max + 4)
    cheek_y_lo = max(0, cheek_y - 2)
    cheek_y_hi = min(H, cheek_y + 3)
    cheek_x_lo = max(0, x_min)
    cheek_x_hi = min(W, x_max + 1)
    skin = image[:, cheek_y_lo:cheek_y_hi, cheek_x_lo:cheek_x_hi]
    if skin.numel() == 0:
        return image
    skin_color = skin.mean(dim=(1, 2))      # (3,)

    # Multiplicative tone jitter so the augmenter doesn't paint the same
    # colour into every sample.
    if skin_tone_jitter > 0:
        jitter = 1.0 + (torch.rand(3, device=image.device, dtype=image.dtype)
                        * 2 - 1) * skin_tone_jitter
        skin_color = skin_color * jitter

    # Hard fill the top region.
    out = image.clone()
    out[:, y_min:cover_y_end + 1, x_min:x_max + 1] = skin_color[:, None, None]

    # Feather the bottom edge so the boundary isn't a hard step (which
    # the model can otherwise memorise as "eyelid".)
    if feather_px > 0:
        feather_start = max(y_min, cover_y_end - feather_px)
        feather_end = min(H, cover_y_end + feather_px + 1)
        feather_h = feather_end - feather_start
        if feather_h > 1:
            # Linear blend from 1.0 (occluder) at top to 0.0 at bottom.
            alpha = torch.linspace(
                1.0, 0.0, feather_h, device=image.device, dtype=image.dtype
            )[:, None]                         # (feather_h, 1)
            band = out[:, feather_start:feather_end, x_min:x_max + 1]
            orig = image[:, feather_start:feather_end, x_min:x_max + 1]
            blend = alpha[None] * skin_color[:, None, None] \
                  + (1.0 - alpha[None]) * orig
            out[:, feather_start:feather_end, x_min:x_max + 1] = blend
            del band

    return out
