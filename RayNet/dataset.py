import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import cv2
from collections import defaultdict
import random


class GazeGeneDataset(Dataset):
    def __init__(
            self,
            base_dir,
            subject_ids=None,
            camera_ids=None,
            samples_per_subject=None,
            transform=None,
            balance_attributes=None,
            include_2d_landmarks=True,
            include_camera_params=True,
            seed=42
    ):
        """
        Enhanced GazeGene Dataset with full annotation support.

        Args:
            base_dir: Path to GazeGene dataset
            subject_ids: List of subject IDs to include (default: all)
            camera_ids: List of camera IDs to include (default: all 0-8)
            samples_per_subject: Number of frames per subject (default: all)
            transform: Image transformations
            balance_attributes: Attributes for balanced sampling ['ethicity', 'gender']
            include_2d_landmarks: Whether to include 2D iris landmarks
            include_camera_params: Whether to include camera parameters
            seed: Random seed
        """
        self.samples = []
        self.index_by_key = defaultdict(list)
        self.attr_dict = {}
        self.camera_params = {}
        self.transform = transform
        self.balance_attributes = balance_attributes
        self.include_2d_landmarks = include_2d_landmarks
        self.include_camera_params = include_camera_params
        random.seed(seed)

        # Load subject attributes and camera parameters
        self._load_metadata(base_dir)

        # Load samples
        self._load_samples(base_dir, subject_ids, camera_ids, samples_per_subject)

    def _load_metadata(self, base_dir):
        """Load subject attributes and camera parameters."""
        subjects = [d for d in os.listdir(base_dir) if d.startswith('subject')]

        for subject in subjects:
            subj_num = int(subject.replace('subject', ''))

            # Load subject attributes
            attr_path = os.path.join(base_dir, subject, 'subject_label.pkl')
            if os.path.exists(attr_path):
                with open(attr_path, 'rb') as f:
                    attrs = pickle.load(f)
                self.attr_dict[subj_num] = attrs

            # Load camera parameters (only need to load once per subject)
            if self.include_camera_params and subj_num not in self.camera_params:
                camera_path = os.path.join(base_dir, subject, 'camera_info.pkl')
                if os.path.exists(camera_path):
                    with open(camera_path, 'rb') as f:
                        camera_info = pickle.load(f)

                    # Convert list to dictionary indexed by camera ID
                    if isinstance(camera_info, list):
                        camera_dict = {}
                        for cam_data in camera_info:
                            if isinstance(cam_data, dict) and 'cam_id' in cam_data:
                                camera_dict[cam_data['cam_id']] = cam_data
                        self.camera_params[subj_num] = camera_dict
                    else:
                        # Assume it's already a dictionary
                        self.camera_params[subj_num] = camera_info

    def _load_samples(self, base_dir, subject_ids, camera_ids, samples_per_subject):
        """Load all samples with full annotations."""
        print(base_dir)
        if subject_ids is None:
            subjects = sorted([d for d in os.listdir(base_dir) if d.startswith('subject')])
        else:
            # Handle both string and integer subject IDs
            if isinstance(subject_ids[0], int):
                subjects = [f"subject{i}" for i in subject_ids]
            else:
                subjects = subject_ids

        print(
            f"Processing {len(subjects)} subjects: {subjects[:3]}...{subjects[-3:] if len(subjects) > 3 else subjects}")

        for subject in subjects:
            if not os.path.exists(os.path.join(base_dir, subject)):
                print(f"Warning: Subject directory {subject} not found, skipping...")
                continue

            subj_num = int(subject.replace('subject', ''))
            label_dir = os.path.join(base_dir, subject, 'labels')

            if not os.path.exists(label_dir):
                print(f"Warning: Labels directory not found for {subject}, skipping...")
                continue

            # Load labels for all cameras
            complex_labels = {}
            gaze_labels = {}

            camera_list = camera_ids if camera_ids is not None else range(9)
            labels_loaded = 0

            for cam_id in camera_list:
                # Load complex labels (3D eyeball annotations)
                complex_path = os.path.join(label_dir, f'complex_label_camera{cam_id}.pkl')
                if os.path.exists(complex_path):
                    try:
                        with open(complex_path, 'rb') as f:
                            complex_labels[cam_id] = pickle.load(f)
                        labels_loaded += 1
                    except Exception as e:
                        print(f"Error loading {complex_path}: {e}")
                        continue
                else:
                    print(f"Warning: {complex_path} not found")

                # Load gaze labels (gaze directions and head poses)
                gaze_path = os.path.join(label_dir, f'gaze_label_camera{cam_id}.pkl')
                if os.path.exists(gaze_path):
                    try:
                        with open(gaze_path, 'rb') as f:
                            gaze_labels[cam_id] = pickle.load(f)
                    except Exception as e:
                        print(f"Error loading {gaze_path}: {e}")
                        continue
                else:
                    print(f"Warning: {gaze_path} not found")

            if not complex_labels or not gaze_labels:
                print(f"Warning: No valid labels found for {subject}, skipping...")
                continue

            print(f"Loaded labels for {subject}: {labels_loaded} cameras")

            # Determine number of frames
            num_frames = len(complex_labels[list(complex_labels.keys())[0]]['img_path'])
            print(f"  {num_frames} frames available")

            # Sample frames if specified
            if samples_per_subject is not None:
                frame_idxs = random.sample(range(num_frames), min(samples_per_subject, num_frames))
                print(f"  Sampling {len(frame_idxs)} frames")
            else:
                frame_idxs = range(num_frames)

            # Create samples for each frame and camera
            samples_created = 0
            for idx in frame_idxs:
                for cam_id in camera_list:
                    if cam_id not in complex_labels or cam_id not in gaze_labels:
                        continue

                    complex_label = complex_labels[cam_id]
                    gaze_label = gaze_labels[cam_id]

                    # Check if image path exists
                    img_path = os.path.join(base_dir, subject, complex_label['img_path'][idx])
                    if not os.path.exists(img_path):
                        continue

                    sample = self._create_sample(
                        base_dir, subject, subj_num, cam_id, idx,
                        complex_label, gaze_label
                    )

                    self.samples.append(sample)
                    self.index_by_key[(subj_num, idx)].append(len(self.samples) - 1)
                    samples_created += 1

            print(f"  Created {samples_created} samples for {subject}")

        print(f"Total samples created: {len(self.samples)}")

    def _create_sample(self, base_dir, subject, subj_num, cam_id, idx, complex_label, gaze_label):
        """Create a single sample with all annotations."""
        sample = {
            'img_path': os.path.join(base_dir, subject, complex_label['img_path'][idx]),
            'subject': subj_num,
            'camera': cam_id,
            'frame_idx': idx,

            # 3D Mesh Data
            'mesh_3d': {
                'eyeball_center_3D': complex_label['eyeball_center_3D'][idx],  # [2, 3]
                'pupil_center_3D': complex_label['pupil_center_3D'][idx],  # [2, 3]
                'iris_mesh_3D': complex_label['iris_mesh_3D'][idx],  # [2, 100, 3]
            },

            # 2D Pixel Data (if requested)
            'mesh_2d': {
                'eyeball_center_2D': complex_label['eyeball_center_2D'][idx],  # [2, 2]
                'pupil_center_2D': complex_label['pupil_center_2D'][idx],  # [2, 2]
                'iris_mesh_2D': complex_label['iris_mesh_2D'][idx],  # [2, 100, 2]
            } if self.include_2d_landmarks else None,

            # Camera Parameters
            'camera': {
                'intrinsic_cropped': complex_label['intrinsic_matrix_cropped'][idx],  # [3, 3]
                'cam_id': cam_id,
            },

            # Gaze Information
            'gaze': {
                'gaze_vector_C': gaze_label['gaze_C'][idx],  # [3] - Unit gaze vector in CCS
                'visual_axis_L': gaze_label['visual_axis_L'][idx],  # [3] - Left eye visual axis
                'visual_axis_R': gaze_label['visual_axis_R'][idx],  # [3] - Right eye visual axis
                'optic_axis_L': gaze_label['optic_axis_L'][idx],  # [3] - Left eye optic axis
                'optic_axis_R': gaze_label['optic_axis_R'][idx],  # [3] - Right eye optic axis
                'gaze_target': gaze_label['gaze_target'][idx],  # [3] - 3D gaze target in CCS
                'gaze_depth': gaze_label['gaze_depth'][idx],  # scalar - Gaze depth
            },

            # Head Pose
            'head_pose': {
                'R': gaze_label['head_R_mat'][idx],  # [3, 3] - Rotation matrix in CCS
                't': gaze_label['head_T_vec'][idx],  # [3] - Translation vector in CCS
            },

            # Subject Attributes
            'subject_attributes': self.attr_dict.get(subj_num, None),

            # Camera Parameters (full camera info)
            'camera_params': self.camera_params.get(subj_num, {}).get(cam_id,
                                                                      None) if self.include_camera_params else None,
        }

        return sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single sample with all annotations converted to tensors."""
        s = self.samples[idx]

        # Load and process image
        img = cv2.imread(s['img_path'])
        if img is None:
            raise FileNotFoundError(f"Image not found: {s['img_path']}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        def to_tensor(x):
            """Convert numpy array to tensor safely."""
            if x is None:
                return None
            return torch.from_numpy(np.asarray(x)).float()

        # Prepare output dictionary
        output = {
            'img': img,
            'subject': s['subject'],
            'camera': s['camera'],
            'frame_idx': s['frame_idx'],

            # 3D Mesh annotations
            'mesh': {k: to_tensor(v) for k, v in s['mesh_3d'].items()},

            # Camera intrinsics
            'intrinsic': to_tensor(s['camera']['intrinsic_cropped']),

            # Gaze annotations
            'gaze': {k: to_tensor(v) for k, v in s['gaze'].items()},

            # Head pose
            'head_pose': {
                'R': to_tensor(s['head_pose']['R']),
                't': to_tensor(s['head_pose']['t'])
            },

            # Subject metadata
            'subject_attributes': s['subject_attributes'],
        }

        # Add 2D landmarks if requested
        if self.include_2d_landmarks and s['mesh_2d'] is not None:
            output['mesh_2d'] = {k: to_tensor(v) for k, v in s['mesh_2d'].items()}

        # Add camera parameters if requested
        if self.include_camera_params and s['camera_params'] is not None:
            output['camera_params'] = {
                'cam_id': s['camera_params']['cam_id'],
                'intrinsic_matrix': to_tensor(s['camera_params']['intrinsic_matrix']),
                'R_mat': to_tensor(s['camera_params']['R_mat']),
                'T_vec': to_tensor(s['camera_params']['T_vec']),
            }

        return output

    def get_subject_info(self, subject_id):
        """Get detailed subject information."""
        return self.attr_dict.get(subject_id, {})

    def get_camera_info(self, subject_id, camera_id):
        """Get camera parameters for specific subject and camera."""
        subject_cameras = self.camera_params.get(subject_id, {})
        return subject_cameras.get(camera_id, {})

    def get_dataset_statistics(self):
        """Get dataset statistics."""
        stats = {
            'num_samples': len(self.samples),
            'num_subjects': len(set(s['subject'] for s in self.samples)),
            'num_cameras': len(set(s['camera'] for s in self.samples)),
            'subjects': sorted(list(set(s['subject'] for s in self.samples))),
            'cameras': sorted(list(set(s['camera'] for s in self.samples))),
        }

        # Attribute statistics
        if self.attr_dict:
            ethnicities = [attrs.get('ethicity', 'Unknown') for attrs in self.attr_dict.values()]
            genders = [attrs.get('gender', 'Unknown') for attrs in self.attr_dict.values()]

            stats['ethnicity_distribution'] = {
                ethnicity: ethnicities.count(ethnicity) for ethnicity in set(ethnicities)
            }
            stats['gender_distribution'] = {
                gender: genders.count(gender) for gender in set(genders)
            }

        return stats


class EnhancedMultiViewBatchSampler(Sampler):
    """
    Enhanced batch sampler with support for multi-view consistency and attribute balancing.
    """

    def __init__(self, dataset, batch_size=1, balance_attributes=None,
                 ensure_multiview=True, shuffle=True):
        """
        Args:
            dataset: GazeGeneDataset instance
            batch_size: Number of (subject, frame) pairs per batch
            balance_attributes: List of attributes to balance ['ethicity', 'gender']
            ensure_multiview: If True, only include samples with all 9 cameras
            shuffle: Whether to shuffle the data
        """
        self.index_by_key = dataset.index_by_key
        self.keys = list(self.index_by_key.keys())
        self.balance_attributes = balance_attributes
        self.ensure_multiview = ensure_multiview
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dataset = dataset

        # Filter keys for multi-view if requested
        if self.ensure_multiview:
            self.keys = [k for k in self.keys if len(self.index_by_key[k]) == 9]

        # Group by attributes for balanced sampling
        if balance_attributes:
            self.grouped_keys = defaultdict(list)
            for k in self.keys:
                subj, _ = k
                attrs = dataset.attr_dict.get(subj, {})
                attr_vals = tuple([attrs.get(a, 'Unknown') for a in balance_attributes])
                self.grouped_keys[attr_vals].append(k)
            self.groups = list(self.grouped_keys.keys())
        else:
            self.grouped_keys = None
            self.groups = None

    def __iter__(self):
        if self.grouped_keys:
            # Balanced sampling across attributes
            group_keys = self.groups.copy()
            if self.shuffle:
                random.shuffle(group_keys)

            all_keys = []
            for g in group_keys:
                klist = self.grouped_keys[g].copy()
                if self.shuffle:
                    random.shuffle(klist)
                all_keys.extend(klist)

            if self.shuffle:
                random.shuffle(all_keys)
        else:
            # Regular sampling
            all_keys = self.keys.copy()
            if self.shuffle:
                random.shuffle(all_keys)

        # Yield batches
        batch = []
        for k in all_keys:
            indices = self.index_by_key[k]

            if self.ensure_multiview and len(indices) != 9:
                continue

            batch.extend(indices)

            if len(batch) >= (9 if self.ensure_multiview else 1) * self.batch_size:
                yield batch
                batch = []

        if batch:  # Yield remaining samples
            yield batch

    def __len__(self):
        valid_keys = self.keys
        if self.ensure_multiview:
            valid_keys = [k for k in self.keys if len(self.index_by_key[k]) == 9]
        return len(valid_keys) // self.batch_size


# Utility functions for working with the enhanced dataset
def project_3d_to_2d(points_3d, intrinsic_matrix):
    """
    Project 3D points to 2D pixel coordinates.

    Args:
        points_3d: [B, N, 3] or [N, 3] - 3D points
        intrinsic_matrix: [3, 3] or [B, 3, 3] - Camera intrinsics

    Returns:
        points_2d: [B, N, 2] or [N, 2] - 2D pixel coordinates
    """
    if len(points_3d.shape) == 2:  # [N, 3]
        points_3d = points_3d.unsqueeze(0)  # [1, N, 3]

    if len(intrinsic_matrix.shape) == 2:  # [3, 3]
        intrinsic_matrix = intrinsic_matrix.unsqueeze(0)  # [1, 3, 3]

    # Homogeneous coordinates
    ones = torch.ones(points_3d.shape[:-1] + (1,), device=points_3d.device)
    points_homogeneous = torch.cat([points_3d, ones], dim=-1)  # [B, N, 4]

    # Project to 2D
    projected = torch.matmul(intrinsic_matrix, points_3d.transpose(-1, -2))  # [B, 3, N]
    projected = projected.transpose(-1, -2)  # [B, N, 3]

    # Normalize by depth
    points_2d = projected[..., :2] / (projected[..., 2:3] + 1e-8)  # [B, N, 2]

    return points_2d.squeeze(0) if points_2d.shape[0] == 1 else points_2d


def create_gazegene_dataloaders(base_dir, train_subjects, val_subjects,
                                batch_size=4, num_workers=4, **kwargs):
    """
    Create train and validation dataloaders for GazeGene.

    Args:
        base_dir: Path to GazeGene dataset
        train_subjects: List of subject IDs for training
        val_subjects: List of subject IDs for validation
        batch_size: Batch size
        num_workers: Number of worker processes
        **kwargs: Additional arguments for GazeGeneDataset

    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = GazeGeneDataset(
        base_dir=base_dir,
        subject_ids=train_subjects,
        **kwargs
    )

    val_dataset = GazeGeneDataset(
        base_dir=base_dir,
        subject_ids=val_subjects,
        **kwargs
    )

    # Create batch samplers
    train_sampler = EnhancedMultiViewBatchSampler(
        train_dataset,
        batch_size=batch_size,
        balance_attributes=kwargs.get('balance_attributes', None),
        shuffle=True
    )

    val_sampler = EnhancedMultiViewBatchSampler(
        val_dataset,
        batch_size=batch_size,
        ensure_multiview=True,
        shuffle=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader



def gazegene_collate_fn(batch):
    """
    Custom collate function for GazeGene dataset that handles:
    - Multi-view samples (9 cameras per subject)
    - Complex nested dictionaries
    - Mixed data types (tensors, scalars, None values)

    Args:
        batch: List of samples from dataset.__getitem__()

    Returns:
        Collated batch dictionary
    """
    if len(batch) == 0:
        return {}

    # Handle case where some samples might be None or invalid
    valid_batch = [item for item in batch if item is not None]
    if len(valid_batch) == 0:
        return {}

    # Initialize collated batch
    collated = {}

    # Collate simple tensor fields
    tensor_fields = ['img', 'intrinsic']
    for field in tensor_fields:
        if field in valid_batch[0]:
            try:
                collated[field] = torch.stack([item[field] for item in valid_batch])
            except Exception as e:
                print(f"Error collating {field}: {e}")
                # If stacking fails, keep as list
                collated[field] = [item[field] for item in valid_batch]

    # Collate simple scalar fields
    scalar_fields = ['subject', 'camera', 'frame_idx']
    for field in scalar_fields:
        if field in valid_batch[0]:
            collated[field] = [item[field] for item in valid_batch]

    # Collate nested dictionary fields
    nested_dict_fields = ['mesh', 'gaze', 'head_pose']
    for field in nested_dict_fields:
        if field in valid_batch[0] and valid_batch[0][field] is not None:
            collated[field] = {}

            # Get all keys from the nested dictionary
            all_keys = set()
            for item in valid_batch:
                if item[field] is not None:
                    all_keys.update(item[field].keys())

            # Collate each key in the nested dictionary
            for key in all_keys:
                try:
                    values = []
                    for item in valid_batch:
                        if item[field] is not None and key in item[field]:
                            values.append(item[field][key])
                        else:
                            # Handle missing values by using None or zero tensor
                            values.append(None)

                    # Filter out None values
                    non_none_values = [v for v in values if v is not None]
                    if non_none_values:
                        if torch.is_tensor(non_none_values[0]):
                            # Try to stack tensors
                            try:
                                collated[field][key] = torch.stack(non_none_values)
                            except Exception as e:
                                print(f"Error stacking {field}.{key}: {e}")
                                collated[field][key] = non_none_values
                        else:
                            # Keep as list for non-tensor values
                            collated[field][key] = non_none_values
                    else:
                        collated[field][key] = None

                except Exception as e:
                    print(f"Error collating {field}.{key}: {e}")
                    collated[field][key] = [item[field][key] if item[field] else None for item in valid_batch]

    # Handle optional nested fields that might be None
    optional_nested_fields = ['mesh_2d', 'camera_params']
    for field in optional_nested_fields:
        if field in valid_batch[0]:
            # Check if any item has non-None values for this field
            has_valid_data = any(item[field] is not None for item in valid_batch)

            if has_valid_data:
                collated[field] = {}

                # Get all possible keys
                all_keys = set()
                for item in valid_batch:
                    if item[field] is not None:
                        all_keys.update(item[field].keys())

                # Collate each key
                for key in all_keys:
                    values = []
                    for item in valid_batch:
                        if item[field] is not None and key in item[field]:
                            values.append(item[field][key])
                        else:
                            values.append(None)

                    # Filter and collate non-None values
                    non_none_values = [v for v in values if v is not None]
                    if non_none_values:
                        if torch.is_tensor(non_none_values[0]):
                            try:
                                collated[field][key] = torch.stack(non_none_values)
                            except:
                                collated[field][key] = non_none_values
                        else:
                            collated[field][key] = non_none_values
                    else:
                        collated[field][key] = None
            else:
                collated[field] = None

    # Handle subject_attributes specially (list of dictionaries)
    if 'subject_attributes' in valid_batch[0]:
        collated['subject_attributes'] = [item['subject_attributes'] for item in valid_batch]

    return collated


def convert_dataset_to_model_format(batch, device):
    """
    Convert dataset batch for direct CCS prediction model
    No coordinate transformation needed - ground truth already in CCS
    """
    batch_size = batch['img'].shape[0]

    # Extract images
    images = batch['img'].to(device)  # [B, 3, H, W]

    # Extract subject parameters (unchanged)
    gazegene_subject_params = {}
    subject_attrs_list = batch.get('subject_attributes', [])

    reference_attrs = None
    for attrs in subject_attrs_list:
        if attrs is not None and isinstance(attrs, dict):
            reference_attrs = attrs
            break

    if reference_attrs:
        for param_name in ['eyecenter_L', 'eyecenter_R', 'eyeball_radius', 'iris_radius',
                           'cornea_radius', 'cornea2center', 'UVRadius', 'L_kappa', 'R_kappa']:
            if param_name in reference_attrs:
                raw_value = reference_attrs[param_name]

                # Convert to tensor
                if isinstance(raw_value, (int, float)):
                    param_tensor = torch.tensor([raw_value], dtype=torch.float32)
                else:
                    param_tensor = torch.tensor(raw_value, dtype=torch.float32)

                # Fix kappa angles if needed (2D -> 3D)
                if param_name in ['L_kappa', 'R_kappa'] and param_tensor.shape[-1] == 2:
                    zero_roll = torch.tensor([0.0], dtype=param_tensor.dtype)
                    param_tensor = torch.cat([param_tensor, zero_roll])

                # Expand to batch size
                param_tensor = param_tensor.unsqueeze(0).expand(batch_size, -1)
                gazegene_subject_params[param_name] = param_tensor.to(device)

    # Camera parameters
    camera_params = {}
    if 'intrinsic' in batch and batch['intrinsic'] is not None:
        camera_params['intrinsic_matrix'] = batch['intrinsic'].to(device)

    # Ground truth - ALREADY IN CCS, NO TRANSFORMATION NEEDED
    ground_truth = {}

    if 'mesh' in batch:
        mesh = batch['mesh']
        # Direct use of CCS coordinates
        ground_truth['eyeball_center_3D'] = mesh['eyeball_center_3D'].to(device)  # [B, 2, 3] in CCS
        ground_truth['pupil_center_3D'] = mesh['pupil_center_3D'].to(device)  # [B, 2, 3] in CCS

        # Fix iris mesh shape
        iris_mesh_3d = mesh['iris_mesh_3D'].to(device)
        if iris_mesh_3d.dim() == 4 and iris_mesh_3d.shape[1] == 2:
            ground_truth['iris_mesh_3D'] = iris_mesh_3d.reshape(batch_size, -1, 3)  # [B, 200, 3] in CCS
        else:
            ground_truth['iris_mesh_3D'] = iris_mesh_3d

    if 'gaze' in batch:
        gaze = batch['gaze']
        ground_truth['gaze_C'] = gaze['gaze_vector_C'].to(device)  # Already in CCS
        ground_truth['optic_axis_L'] = gaze['optic_axis_L'].to(device)  # Already in CCS
        ground_truth['optic_axis_R'] = gaze['optic_axis_R'].to(device)  # Already in CCS
        ground_truth['visual_axis_L'] = gaze['visual_axis_L'].to(device)  # Already in CCS
        ground_truth['visual_axis_R'] = gaze['visual_axis_R'].to(device)  # Already in CCS

    # 2D data (if available)
    if 'mesh_2d' in batch and batch['mesh_2d'] is not None:
        mesh_2d = batch['mesh_2d']
        if 'eyeball_center_2D' in mesh_2d and mesh_2d['eyeball_center_2D'] is not None:
            ground_truth['eyeball_center_2D'] = mesh_2d['eyeball_center_2D'].to(device)
            ground_truth['pupil_center_2D'] = mesh_2d['pupil_center_2D'].to(device)

            iris_mesh_2d = mesh_2d['iris_mesh_2D'].to(device)
            if iris_mesh_2d.dim() == 4 and iris_mesh_2d.shape[1] == 2:
                ground_truth['iris_mesh_2D'] = iris_mesh_2d.reshape(batch_size, -1, 2)
            else:
                ground_truth['iris_mesh_2D'] = iris_mesh_2d

    return {
        'images': images,
        'gazegene_subject_params': gazegene_subject_params,
        'camera_params': camera_params,
        'ground_truth': ground_truth
    }


# Example usage
if __name__ == "__main__":
    # Example usage with full annotations
    dataset = GazeGeneDataset(
        base_dir="/path/to/gazegene",
        subject_ids=[1, 2, 3, 4, 5],  # First 5 subjects
        camera_ids=[0, 1, 2],  # First 3 cameras
        include_2d_landmarks=True,  # Include 2D landmarks
        include_camera_params=True,  # Include camera parameters
        balance_attributes=['ethicity', 'gender']
    )

    # Get sample
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("3D iris mesh shape:", sample['mesh']['iris_mesh_3D'].shape)
    print("2D iris mesh shape:", sample['mesh_2d']['iris_mesh_2D'].shape)
    print("Camera intrinsics shape:", sample['intrinsic'].shape)

    # Get dataset statistics
    stats = dataset.get_dataset_statistics()
    print("Dataset statistics:", stats)