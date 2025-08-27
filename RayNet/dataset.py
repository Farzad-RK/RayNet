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


def convert_dataset_to_model_format(batch, device):
    """
    Convert collated dataset batch to format expected by EyeFLAME model

    Args:
        batch: Collated batch from DataLoader (dictionary with stacked tensors)
        device: Device to move tensors to

    Returns:
        Dictionary with model inputs and ground truth
    """
    # Validate batch structure
    if not isinstance(batch, dict):
        raise ValueError(f"Expected batch to be dict, got {type(batch)}")

    if 'img' not in batch:
        raise ValueError("Batch missing 'img' key")

    batch_size = batch['img'].shape[0]
    print(f"Converting batch with size: {batch_size}")

    # Extract images
    images = batch['img'].to(device)  # [B, 3, H, W]

    # Extract subject-specific parameters
    gazegene_subject_params = {}
    subject_attrs_list = batch.get('subject_attributes', [])

    print(f"Subject attributes list length: {len(subject_attrs_list)}")

    # Get the first valid subject attributes (should be same for all camera views)
    reference_attrs = None
    for attrs in subject_attrs_list:
        if attrs is not None and isinstance(attrs, dict):
            reference_attrs = attrs
            break

    if reference_attrs is None:
        print("WARNING: No valid subject attributes found, using defaults")
        # Create default parameters
        gazegene_subject_params = {
            'eyecenter_L': torch.zeros(batch_size, 3, dtype=torch.float32).to(device),
            'eyecenter_R': torch.zeros(batch_size, 3, dtype=torch.float32).to(device),
            'eyeball_radius': torch.ones(batch_size, 1, dtype=torch.float32).to(device) * 1.2,
            'iris_radius': torch.ones(batch_size, 1, dtype=torch.float32).to(device) * 0.6,
            'cornea_radius': torch.ones(batch_size, 1, dtype=torch.float32).to(device) * 0.78,
            'cornea2center': torch.ones(batch_size, 1, dtype=torch.float32).to(device) * 0.5,
            'UVRadius': torch.ones(batch_size, 1, dtype=torch.float32).to(device) * 0.15,
            'L_kappa': torch.zeros(batch_size, 3, dtype=torch.float32).to(device),  # [0, 0, 0]
            'R_kappa': torch.zeros(batch_size, 3, dtype=torch.float32).to(device)  # [0, 0, 0]
        }
    else:
        print(f"Using reference subject attributes: {list(reference_attrs.keys())}")

        # Process each parameter
        for param_name in ['eyecenter_L', 'eyecenter_R', 'eyeball_radius', 'iris_radius',
                           'cornea_radius', 'cornea2center', 'UVRadius', 'L_kappa', 'R_kappa']:

            if param_name in reference_attrs:
                raw_value = reference_attrs[param_name]
                print(f"\nProcessing {param_name}:")
                print(f"  Raw value: {raw_value} (type: {type(raw_value)})")

                # Convert to tensor
                if isinstance(raw_value, (int, float)):
                    param_tensor = torch.tensor([raw_value], dtype=torch.float32)
                else:
                    param_tensor = torch.tensor(raw_value, dtype=torch.float32)

                print(f"  After tensor conversion: shape={param_tensor.shape}, values={param_tensor}")

                # CRITICAL: Handle kappa angles - convert 2D to 3D BEFORE batch expansion
                if param_name in ['L_kappa', 'R_kappa']:
                    if param_tensor.shape[-1] == 2:
                        print(f"  🔧 Converting 2D kappa to 3D: {param_tensor} -> ", end="")
                        # Add zero roll component: [horizontal, vertical] -> [horizontal, vertical, 0]
                        zero_roll = torch.tensor([0.0], dtype=param_tensor.dtype)
                        param_tensor = torch.cat([param_tensor, zero_roll])  # [2] -> [3]
                        print(f"{param_tensor}")
                        print(f"  ✅ Kappa shape after 2D->3D conversion: {param_tensor.shape}")
                    elif param_tensor.shape[-1] == 3:
                        print(f"  ✅ Kappa already 3D: {param_tensor}")
                    else:
                        print(f"  ❌ Unexpected kappa shape: {param_tensor.shape}, creating default")
                        param_tensor = torch.tensor([0.0698, 0.0175, 0.0], dtype=torch.float32)  # ~4°, ~1°, 0°

                # Expand to batch size (replicate for all camera views)
                param_tensor = param_tensor.unsqueeze(0).expand(batch_size, -1)
                print(f"  Final shape after batch expansion: {param_tensor.shape}")

                gazegene_subject_params[param_name] = param_tensor.to(device)

            else:
                print(f"\n{param_name}: Missing from attributes, using default")
                # Create defaults
                if param_name in ['eyecenter_L', 'eyecenter_R']:
                    gazegene_subject_params[param_name] = torch.zeros(batch_size, 3, dtype=torch.float32).to(device)
                elif param_name in ['L_kappa', 'R_kappa']:
                    # Default kappa: [horizontal≈4°, vertical≈1°, roll=0°] in radians
                    default_kappa = torch.tensor([0.0698, 0.0175, 0.0], dtype=torch.float32)
                    gazegene_subject_params[param_name] = default_kappa.unsqueeze(0).expand(batch_size, -1).to(device)
                    print(f"  Default kappa shape: {gazegene_subject_params[param_name].shape}")
                else:
                    # Scalar defaults
                    default_val = 0.6 if 'radius' in param_name else 0.5
                    gazegene_subject_params[param_name] = torch.ones(batch_size, 1, dtype=torch.float32).to(
                        device) * default_val

    # VERIFICATION: Double-check kappa shapes
    print(f"\n=== KAPPA VERIFICATION ===")
    for kappa_name in ['L_kappa', 'R_kappa']:
        if kappa_name in gazegene_subject_params:
            kappa_tensor = gazegene_subject_params[kappa_name]
            print(f"{kappa_name}: shape={kappa_tensor.shape}, values[0]={kappa_tensor[0]}")
            if kappa_tensor.shape[-1] != 3:
                print(f"❌ CRITICAL ERROR: {kappa_name} still has wrong shape!")
                # Force fix
                if kappa_tensor.shape[-1] == 2:
                    print(f"   Force-fixing by adding zero roll...")
                    zeros = torch.zeros(kappa_tensor.shape[0], 1, device=kappa_tensor.device, dtype=kappa_tensor.dtype)
                    gazegene_subject_params[kappa_name] = torch.cat([kappa_tensor, zeros], dim=-1)
                    print(f"   After force-fix: {gazegene_subject_params[kappa_name].shape}")
                else:
                    print(f"   Creating completely new default kappa...")
                    default_kappa = torch.tensor([0.0698, 0.0175, 0.0], dtype=torch.float32, device=device)
                    gazegene_subject_params[kappa_name] = default_kappa.unsqueeze(0).expand(batch_size, -1)
            else:
                print(f"✅ {kappa_name} has correct 3D shape")

    # Camera parameters
    camera_params = {}
    if 'intrinsic' in batch and batch['intrinsic'] is not None:
        camera_params['intrinsic_matrix'] = batch['intrinsic'].to(device)

    # Ground truth data
    ground_truth = {}

    if 'mesh' in batch:
        mesh = batch['mesh']
        ground_truth['eyeball_center_3D'] = mesh['eyeball_center_3D'].to(device)
        ground_truth['pupil_center_3D'] = mesh['pupil_center_3D'].to(device)

        # Fix iris mesh shape mismatch: [B, 2, 100, 3] -> [B, 200, 3]
        iris_mesh_3d = mesh['iris_mesh_3D'].to(device)  # [B, 2, 100, 3]
        print(f"Original iris mesh shape: {iris_mesh_3d.shape}")

        if iris_mesh_3d.dim() == 4 and iris_mesh_3d.shape[1] == 2:  # [B, 2, 100, 3]
            # Reshape: [B, 2, 100, 3] -> [B, 200, 3] (concatenate left and right eye)
            ground_truth['iris_mesh_3D'] = iris_mesh_3d.reshape(batch_size, -1, 3)  # [B, 200, 3]
            print(f"Reshaped iris mesh to: {ground_truth['iris_mesh_3D'].shape}")
        else:
            # Already in the correct format or different structure
            ground_truth['iris_mesh_3D'] = iris_mesh_3d
            print(f"Keeping iris mesh shape as: {ground_truth['iris_mesh_3D'].shape}")

    if 'gaze' in batch:
        gaze = batch['gaze']
        ground_truth['gaze_C'] = gaze['gaze_vector_C'].to(device)
        ground_truth['optic_axis_L'] = gaze['optic_axis_L'].to(device)
        ground_truth['optic_axis_R'] = gaze['optic_axis_R'].to(device)
        ground_truth['visual_axis_L'] = gaze['visual_axis_L'].to(device)
        ground_truth['visual_axis_R'] = gaze['visual_axis_R'].to(device)

    # Optional 2D data
    if 'mesh_2d' in batch and batch['mesh_2d'] is not None:
        mesh_2d = batch['mesh_2d']
        if 'eyeball_center_2D' in mesh_2d and mesh_2d['eyeball_center_2D'] is not None:
            ground_truth['eyeball_center_2D'] = mesh_2d['eyeball_center_2D'].to(device)
            ground_truth['pupil_center_2D'] = mesh_2d['pupil_center_2D'].to(device)

            # Fix 2D iris mesh shape if needed: [B, 2, 100, 2] -> [B, 200, 2]
            iris_mesh_2d = mesh_2d['iris_mesh_2D'].to(device)
            if iris_mesh_2d.dim() == 4 and iris_mesh_2d.shape[1] == 2:  # [B, 2, 100, 2]
                ground_truth['iris_mesh_2D'] = iris_mesh_2d.reshape(batch_size, -1, 2)  # [B, 200, 2]
                print(f"Reshaped 2D iris mesh from {iris_mesh_2d.shape} to {ground_truth['iris_mesh_2D'].shape}")
            else:
                ground_truth['iris_mesh_2D'] = iris_mesh_2d

    # # Final verification
    # print(f"\n=== FINAL VERIFICATION ===")
    # for param_name, tensor in gazegene_subject_params.items():
    #     print(f"{param_name}: {tensor.shape}")
    #     if param_name in ['L_kappa', 'R_kappa'] and tensor.shape[-1] != 3:
    #         raise ValueError(f"FATAL: {param_name} still has shape {tensor.shape}, expected [..., 3]")

    # print(f"\nConversion complete:")
    # print(f"  Images shape: {images.shape}")
    # print(f"  Eyeball centers shape: {ground_truth.get('eyeball_center_3D', torch.tensor([])).shape}")
    # print(f"  Iris mesh 3D shape: {ground_truth.get('iris_mesh_3D', torch.tensor([])).shape}")
    # print(f"  Subject params keys: {list(gazegene_subject_params.keys())}")
    # Add this to your conversion function
    # print("=== COORDINATE RANGE ANALYSIS ===")
    # print(f"Eyeball centers range: {ground_truth['eyeball_center_3D'].min()} to {ground_truth['eyeball_center_3D'].max()}")
    # print(f"Pupil centers range: {ground_truth['pupil_center_3D'].min()} to {ground_truth['pupil_center_3D'].max()}")
    # print(f"Iris mesh range: {ground_truth['iris_mesh_3D'].min()} to {ground_truth['iris_mesh_3D'].max()}")
    # print(f"Gaze vector range: {ground_truth['gaze_C'].min()} to {ground_truth['gaze_C'].max()}")
    print("=== GROUND TRUTH CORRUPTION ANALYSIS ===")
    eyeball_centers = ground_truth['eyeball_center_3D']
    pupil_centers = ground_truth['pupil_center_3D']
    iris_mesh = ground_truth['iris_mesh_3D']

    # Find samples with extreme values
    extreme_threshold = 100.0  # 100cm = 1 meter
    eyeball_extreme = torch.abs(eyeball_centers) > extreme_threshold
    pupil_extreme = torch.abs(pupil_centers) > extreme_threshold
    iris_extreme = torch.abs(iris_mesh) > extreme_threshold

    print(
        f"Samples with extreme eyeball centers: {eyeball_extreme.any(dim=-1).any(dim=-1).sum()}/{eyeball_centers.shape[0]}")
    print(f"Samples with extreme pupil centers: {pupil_extreme.any(dim=-1).any(dim=-1).sum()}/{pupil_centers.shape[0]}")
    print(f"Samples with extreme iris mesh: {iris_extreme.any(dim=-1).any(dim=-1).sum()}/{iris_mesh.shape[0]}")

    # Show the extreme values
    if eyeball_extreme.any():
        extreme_indices = torch.where(eyeball_extreme.any(dim=-1).any(dim=-1))[0]
        print(f"Extreme eyeball center samples at indices: {extreme_indices}")
        for idx in extreme_indices[:3]:  # Show first 3
            print(f"  Sample {idx}: {eyeball_centers[idx]}")

    return {
        'images': images,
        'gazegene_subject_params': gazegene_subject_params,
        'camera_params': camera_params if camera_params else None,
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