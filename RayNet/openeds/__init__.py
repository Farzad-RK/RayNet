"""
OpenEDS pipeline (RayNet v6).

Decoupled foveal-segmentation + torsion + temporal modules trained on
the FovalNet-preprocessed OpenEDS dataset (400×640 grayscale IR with
per-pixel masks for {background, sclera, iris, pupil}). These modules
are intentionally separate from the GazeGene `RayNet.RayNet` package
so that the synthetic-skeleton path and the real-IR-foveal path can
train on disjoint optimisers and never share gradients.

Modules
-------
- ``dataset``: Kaggle/FovalNet preprocessed loader (per-frame and
  per-sequence variants) for the 4-class semantic-segmentation track.
- ``segmenter``: RITnet-style 1-channel encoder/decoder, ~250k params,
  4-class softmax output.
- ``torsion``: classical 3DeepVOG-style cyclotorsion estimator using
  ellipse fitting + polar rubbersheet + NCC patch matching against a
  rolling reference template. No learnable parameters by default.
- ``temporal``: causal dilated TCN that smooths per-frame features
  (gaze, torsion, ellipse params) and emits blink / saccade / fixation
  classifications. Trained on OpenEDS sequences only.
"""

from RayNet.openeds.dataset import (
    OpenEDSSegDataset,
    OpenEDSSequenceDataset,
    OPENEDS_CLASS_MAP,
)
from RayNet.openeds.segmenter import (
    RITnetStyleSegmenter,
    build_ritnet_full,
    build_ritnet_tiny,
    make_geometric_prior_channel,
)
from RayNet.openeds.torsion import IrisPolarTorsion
from RayNet.openeds.temporal import TCNTemporalBlock

# Streaming readers (lazy-imported because mosaicml-streaming may not
# be installed in every environment).
try:
    from RayNet.openeds.streaming import (
        StreamingOpenEDSSegDataset,
        StreamingOpenEDSSequenceDataset,
        create_openeds_seg_streaming_dataloaders,
        create_openeds_sequence_streaming_dataloaders,
    )
except ImportError:    # pragma: no cover
    StreamingOpenEDSSegDataset = None
    StreamingOpenEDSSequenceDataset = None
    create_openeds_seg_streaming_dataloaders = None
    create_openeds_sequence_streaming_dataloaders = None

__all__ = [
    'OpenEDSSegDataset',
    'OpenEDSSequenceDataset',
    'OPENEDS_CLASS_MAP',
    'RITnetStyleSegmenter',
    'build_ritnet_full',
    'build_ritnet_tiny',
    'make_geometric_prior_channel',
    'IrisPolarTorsion',
    'TCNTemporalBlock',
    'StreamingOpenEDSSegDataset',
    'StreamingOpenEDSSequenceDataset',
    'create_openeds_seg_streaming_dataloaders',
    'create_openeds_sequence_streaming_dataloaders',
]
