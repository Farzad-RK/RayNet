## RayNet: GazeGene Dataset Loader & Multi-View Sampler

This module provides PyTorch Dataset and Sampler classes for the GazeGene synthetic gaze estimation dataset, 
as used in RayNet. It is designed for efficient, balanced, multi-camera training of deep gaze models.

Main Features:
--------------
- Loads images, mesh, gaze, headpose, and camera info from GazeGene directory structure.
- Returns samples as PyTorch tensors for direct use with RepNeXt and BiFPN backbones.
- Supports selecting a subset of frames per subject for faster experiments.
- Provides balanced sampling over subject attributes (e.g., ethnicity, gender, eye color).
- Provides multi-view batches: each batch contains all 9 camera views for a single (subject, frame).

-----------------------------------------------------------------------------------
GazeGeneDataset
-----------------------------------------------------------------------------------

class GazeGeneDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for GazeGene synthetic gaze estimation dataset.

    Args:
        base_dir (str): Path to GazeGene_FaceCrops directory.
        subject_ids (list, optional): List of subject IDs to include (e.g., ['subject1', ...]).
        camera_ids (list, optional): List of camera indices (0-8) to include. Default: all 9.
        samples_per_subject (int, optional): Number of unique frames per subject to load. 
                                             If None, loads all frames.
        transform (callable, optional): Optional transform to be applied on images.
        balance_attributes (list, optional): List of attribute names (e.g., ['ethicity', 'gender'])
                                             for balanced sampling.
        seed (int): Random seed for reproducible sampling.

    Returns:
        Each __getitem__ returns a dict with:
            'img'        : Image tensor [3, H, W]
            'subject'    : Subject numeric ID
            'camera'     : Camera index (0-8)
            'frame_idx'  : Frame index within subject
            'mesh'       : Dict of mesh tensors (eyeball_center_3D, pupil_center_3D, iris_mesh_3D)
            'gaze'       : Dict of gaze vectors (gaze_C, visual_axis_L/R, optic_axis_L/R)
            'gaze_point' : 3D gaze point (if available)
            'head_pose'  : Dict of rotation matrix 'R' and translation 't'
            'intrinsic'  : Intrinsic camera matrix for cropped image
            'attributes' : Subject-level attributes (ethnicity, gender, etc.)
    """

-----------------------------------------------------------------------------------
MultiViewBatchSampler
-----------------------------------------------------------------------------------

class MultiViewBatchSampler(torch.utils.data.Sampler):
    """
    Custom batch sampler for multi-camera training.

    Each batch consists of all 9 camera views for a given (subject, frame_idx).
    Optionally balances sampling over subject attributes, for robust, unbiased training.

    Args:
        dataset (GazeGeneDataset): The dataset instance.
        balance_attributes (list, optional): List of attribute names for balanced grouping.
        shuffle (bool): Whether to shuffle batches each epoch.

    Yields:
        List of 9 indices (one per camera) for a single (subject, frame_idx).
    """

-----------------------------------------------------------------------------------
Example Usage
-----------------------------------------------------------------------------------

```python
from torch.utils.data import DataLoader

# Initialize dataset (50 random frames per subject, balance by ethnicity)
dataset = GazeGeneDataset(
    base_dir='./GazeGene_FaceCrops',
    samples_per_subject=50,
    transform=None,               # or torchvision transforms
    balance_attributes=['ethicity']
)

# Initialize multi-view sampler and loader
batch_sampler = MultiViewBatchSampler(dataset, balance_attributes=['ethicity'], shuffle=True)
loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

# Iterate over batches
for batch in loader:
    # batch['img'] is a list of 9 tensors [3, H, W] (all camera views)
    # batch['gaze']['gaze_C'] is a list of 9 vectors
    print(batch['img'][0].shape, batch['gaze']['gaze_C'][0])
    break
