"""
GazeGene dataset loader for RayNet v2.

Key changes from v1:
  - Per-frame normalization (Zhang et al. 2018) removes depth ambiguity
  - Optical axis GT from geometry (eyeball -> pupil), NOT head gaze
  - Kappa roll zeroed out
  - Iris landmarks warped to normalized image space
  - 14 landmarks: 10 iris contour + 4 pupil boundary (subsampled from 100)
  - Output size: 224x224 normalized eye crops
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import cv2
from collections import defaultdict
import random

from RayNet.normalization import normalize_sample, normalize_gaze, compute_normalization_matrix, warp_points_2d
from RayNet.kappa import build_R_kappa, ground_truth_optical_axis

# Indices to subsample 100 iris points down to 10 evenly-spaced points
IRIS_SUBSAMPLE_IDX = list(range(0, 100, 10))  # [0, 10, 20, ..., 90]


class GazeGeneDataset(Dataset):
    """
    GazeGene dataset with per-frame normalization for RayNet v2.

    Each sample yields a normalized eye crop (224x224) and targets:
      - 14 landmarks in normalized pixel space (10 iris + 4 pupil)
      - optical axis in normalized space (unit vector)
      - R_norm for denormalization at inference
    """

    def __init__(
            self,
            base_dir,
            subject_ids=None,
            camera_ids=None,
            samples_per_subject=None,
            eye='L',
            d_norm=600,
            f_norm=960,
            img_size=224,
            augment=False,
            seed=42
    ):
        """
        Args:
            base_dir: path to GazeGene dataset root
            subject_ids: list of int subject IDs (default: all)
            camera_ids: list of int camera IDs (default: 0-8)
            samples_per_subject: max frames per subject (default: all)
            eye: which eye to use ('L' or 'R')
            d_norm: normalization canonical distance (mm)
            f_norm: normalization canonical focal length (px)
            img_size: output crop size
            augment: whether to apply data augmentation
            seed: random seed
        """
        self.samples = []
        self.index_by_key = defaultdict(list)
        self.attr_dict = {}
        self.camera_params = {}

        self.eye = eye
        self.d_norm = d_norm
        self.f_norm = f_norm
        self.img_size = img_size
        self.augment = augment

        random.seed(seed)
        np.random.seed(seed)

        self._load_metadata(base_dir)
        self._load_samples(base_dir, subject_ids, camera_ids, samples_per_subject)

    def _load_metadata(self, base_dir):
        """Load subject attributes and camera parameters."""
        subjects = [d for d in os.listdir(base_dir) if d.startswith('subject')]

        for subject in subjects:
            subj_num = int(subject.replace('subject', ''))

            attr_path = os.path.join(base_dir, subject, 'subject_label.pkl')
            if os.path.exists(attr_path):
                with open(attr_path, 'rb') as f:
                    self.attr_dict[subj_num] = pickle.load(f)

            camera_path = os.path.join(base_dir, subject, 'camera_info.pkl')
            if os.path.exists(camera_path):
                with open(camera_path, 'rb') as f:
                    camera_info = pickle.load(f)
                if isinstance(camera_info, list):
                    camera_dict = {}
                    for cam_data in camera_info:
                        if isinstance(cam_data, dict) and 'cam_id' in cam_data:
                            camera_dict[cam_data['cam_id']] = cam_data
                    self.camera_params[subj_num] = camera_dict
                else:
                    self.camera_params[subj_num] = camera_info

    def _load_samples(self, base_dir, subject_ids, camera_ids, samples_per_subject):
        """Load all samples."""
        if subject_ids is None:
            subjects = sorted([d for d in os.listdir(base_dir) if d.startswith('subject')])
        else:
            subjects = [f"subject{i}" for i in subject_ids]

        camera_list = camera_ids if camera_ids is not None else list(range(9))

        for subject in subjects:
            subj_dir = os.path.join(base_dir, subject)
            if not os.path.exists(subj_dir):
                continue

            subj_num = int(subject.replace('subject', ''))
            label_dir = os.path.join(subj_dir, 'labels')
            if not os.path.exists(label_dir):
                continue

            complex_labels = {}
            gaze_labels = {}

            for cam_id in camera_list:
                complex_path = os.path.join(label_dir, f'complex_label_camera{cam_id}.pkl')
                gaze_path = os.path.join(label_dir, f'gaze_label_camera{cam_id}.pkl')

                if os.path.exists(complex_path):
                    with open(complex_path, 'rb') as f:
                        complex_labels[cam_id] = pickle.load(f)
                if os.path.exists(gaze_path):
                    with open(gaze_path, 'rb') as f:
                        gaze_labels[cam_id] = pickle.load(f)

            if not complex_labels or not gaze_labels:
                continue

            first_cam = list(complex_labels.keys())[0]
            num_frames = len(complex_labels[first_cam]['img_path'])

            if samples_per_subject is not None:
                frame_idxs = random.sample(range(num_frames),
                                           min(samples_per_subject, num_frames))
            else:
                frame_idxs = list(range(num_frames))

            for idx in frame_idxs:
                for cam_id in camera_list:
                    if cam_id not in complex_labels or cam_id not in gaze_labels:
                        continue

                    cl = complex_labels[cam_id]
                    gl = gaze_labels[cam_id]

                    img_path = os.path.join(base_dir, subject, cl['img_path'][idx])
                    if not os.path.exists(img_path):
                        continue

                    sample = {
                        'img_path': img_path,
                        'subject': subj_num,
                        'cam_id': cam_id,
                        'frame_idx': idx,
                        # 3D data (camera coordinate system)
                        'eyeball_center_3D': cl['eyeball_center_3D'][idx],  # [2, 3]
                        'pupil_center_3D': cl['pupil_center_3D'][idx],      # [2, 3]
                        'iris_mesh_3D': cl['iris_mesh_3D'][idx],            # [2, 100, 3]
                        # 2D data (pixel space)
                        'iris_mesh_2D': cl['iris_mesh_2D'][idx],            # [2, 100, 2]
                        'pupil_center_2D': cl['pupil_center_2D'][idx],      # [2, 2]
                        # Head pose
                        'head_R': gl['head_R_mat'][idx],  # [3, 3]
                        'head_t': gl['head_T_vec'][idx],  # [3]
                        # Camera intrinsics (cropped)
                        'K_cropped': cl['intrinsic_matrix_cropped'][idx],    # [3, 3]
                    }

                    self.samples.append(sample)
                    self.index_by_key[(subj_num, idx)].append(len(self.samples) - 1)

        print(f"Loaded {len(self.samples)} samples "
              f"({len(set(s['subject'] for s in self.samples))} subjects)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a normalized sample for training.

        Output dict:
            image: (3, 224, 224) normalized eye crop
            landmark_coords: (14, 2) landmark pixel coords in normalized image
            optical_axis: (3,) GT optical axis unit vector in normalized space
            R_norm: (3, 3) normalization rotation (for denormalization)
            R_kappa: (3, 3) kappa rotation matrix
            subject: int subject ID
            cam_id: int camera ID
        """
        s = self.samples[idx]
        subj_num = s['subject']
        eye_idx = 0 if self.eye == 'L' else 1

        # Load image
        img = cv2.imread(s['img_path'])
        if img is None:
            raise FileNotFoundError(f"Image not found: {s['img_path']}")

        # Get camera parameters
        # Always use K_cropped for the warp — it matches the 448×448 face crop.
        # camera_info.pkl has full-resolution intrinsics (e.g. f=21549, cx=1280)
        # which do NOT match the cropped image coordinates.
        K = np.array(s['K_cropped'], dtype=np.float64)
        cam_info = self.camera_params.get(subj_num, {}).get(s['cam_id'], None)
        if cam_info is not None:
            R_cam = np.array(cam_info['R_mat'], dtype=np.float64)
            T_cam = np.array(cam_info['T_vec'], dtype=np.float64).flatten()
        else:
            R_cam = np.eye(3, dtype=np.float64)
            T_cam = np.zeros(3, dtype=np.float64)

        # Subject attributes
        subject_attrs = self.attr_dict.get(subj_num, {})

        # --- Eye center in camera coordinates ---
        # Use per-frame eyeball_center_3D which is already in camera coords
        # and consistent with the face crop.  The static subject_attrs
        # eyecenter_L/R is in world coords; transforming it via R_cam/T_cam
        # yields positions that project outside the 448×448 crop.
        t_eye = np.array(s['eyeball_center_3D'][eye_idx], dtype=np.float64)

        # --- Per-frame normalization ---
        R_head = np.array(s['head_R'], dtype=np.float64)
        img_norm, R_norm = normalize_sample(
            img, K, R_head, t_eye,
            d_norm=self.d_norm, f_norm=self.f_norm, size=self.img_size
        )

        # Compute the warp matrix for transforming 2D landmarks
        M, _ = compute_normalization_matrix(
            K, t_eye, d_norm=self.d_norm, f_norm=self.f_norm, size=self.img_size
        )

        # --- Ground truth optical axis ---
        eyeball_3d = np.array(s['eyeball_center_3D'][eye_idx], dtype=np.float64)
        pupil_3d = np.array(s['pupil_center_3D'][eye_idx], dtype=np.float64)
        optical_axis_ccs = ground_truth_optical_axis(eyeball_3d, pupil_3d)

        # Transform optical axis to normalized space
        optical_axis_norm = normalize_gaze(optical_axis_ccs, R_norm)

        # --- Iris landmarks: subsample 100 -> 10 + 4 pupil points ---
        iris_2d = np.array(s['iris_mesh_2D'][eye_idx], dtype=np.float64)  # (100, 2)
        iris_10 = iris_2d[IRIS_SUBSAMPLE_IDX]  # (10, 2)

        # Pupil boundary: approximate 4 cardinal points from pupil center
        # Use iris mesh points closest to pupil center as pupil boundary
        pupil_2d = np.array(s['pupil_center_2D'][eye_idx], dtype=np.float64)  # (2,)
        # Take 4 iris points closest to pupil center as pupil boundary approximation
        dists = np.linalg.norm(iris_2d - pupil_2d[None, :], axis=1)
        pupil_boundary_idx = np.argsort(dists)[:4]
        pupil_4 = iris_2d[pupil_boundary_idx]  # (4, 2)

        # Combine: 10 iris + 4 pupil = 14 landmarks
        landmarks_2d = np.concatenate([iris_10, pupil_4], axis=0)  # (14, 2)

        # Warp landmarks to normalized image space
        landmarks_norm = warp_points_2d(landmarks_2d, M)  # (14, 2)

        # --- Kappa rotation matrix ---
        kappa_key = f'{self.eye}_kappa'
        if kappa_key in subject_attrs:
            kappa_angles = np.array(subject_attrs[kappa_key], dtype=np.float64)
            R_kappa = build_R_kappa(kappa_angles)
        else:
            R_kappa = np.eye(3, dtype=np.float64)

        # --- Convert image to tensor ---
        img_tensor = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_tensor.transpose(2, 0, 1)).float() / 255.0

        # --- Data augmentation ---
        if self.augment:
            img_tensor = self._augment(img_tensor)

        # Scale landmark coordinates to feature map space
        # P2 has stride=4, so feature map is img_size/4
        feat_scale = self.img_size / 4.0  # 224/4 = 56
        pixel_to_feat = feat_scale / self.img_size  # 56/224 = 0.25
        landmarks_feat = landmarks_norm * pixel_to_feat

        # Inverse normalization warp for denormalizing predictions back to original pixel space
        M_inv = np.linalg.inv(M)

        return {
            'image': img_tensor,                                        # (3, 224, 224)
            'landmark_coords': torch.from_numpy(landmarks_feat).float(),  # (14, 2)
            'landmark_coords_px': torch.from_numpy(landmarks_norm).float(),  # (14, 2) pixel space
            'optical_axis': torch.from_numpy(optical_axis_norm).float(),  # (3,)
            'R_norm': torch.from_numpy(R_norm).float(),                   # (3, 3)
            'R_kappa': torch.from_numpy(R_kappa).float(),                 # (3, 3)
            # Camera parameters for multi-view consistency
            'K': torch.from_numpy(K).float(),                             # (3, 3) intrinsics
            'R_cam': torch.from_numpy(R_cam).float(),                     # (3, 3) extrinsic rotation
            'T_cam': torch.from_numpy(T_cam).float(),                     # (3,) extrinsic translation
            'M_norm_inv': torch.from_numpy(M_inv).float(),                # (3, 3) inverse warp
            'eyeball_center_3d': torch.from_numpy(t_eye).float(),         # (3,) eye center in CCS
            'subject': subj_num,
            'cam_id': s['cam_id'],
            'frame_idx': s['frame_idx'],
        }

    def _augment(self, img_tensor):
        """Lightweight augmentation: color jitter + random horizontal flip."""
        # Random brightness/contrast
        if random.random() > 0.5:
            brightness = 0.8 + random.random() * 0.4  # [0.8, 1.2]
            img_tensor = img_tensor * brightness
            img_tensor = img_tensor.clamp(0, 1)
        # Random Gaussian noise
        if random.random() > 0.5:
            noise = torch.randn_like(img_tensor) * 0.02
            img_tensor = (img_tensor + noise).clamp(0, 1)
        return img_tensor


class MultiViewBatchSampler(Sampler):
    """
    Batch sampler that groups all 9 camera views of the same (subject, frame).
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, ensure_multiview=True):
        """
        Args:
            dataset: GazeGeneDataset instance
            batch_size: number of (subject, frame) groups per batch
            shuffle: whether to shuffle
            ensure_multiview: only include groups with all 9 cameras
        """
        self.index_by_key = dataset.index_by_key
        self.keys = list(self.index_by_key.keys())
        self.shuffle = shuffle
        self.batch_size = batch_size

        if ensure_multiview:
            self.keys = [k for k in self.keys if len(self.index_by_key[k]) == 9]

    def __iter__(self):
        keys = self.keys.copy()
        if self.shuffle:
            random.shuffle(keys)

        batch = []
        for k in keys:
            batch.extend(self.index_by_key[k])
            if len(batch) >= 9 * self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        return len(self.keys) // self.batch_size


def gazegene_collate_fn(batch):
    """
    Collate function for GazeGene normalized samples.
    Stacks tensors and collects scalars into lists.
    """
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


def create_dataloaders(base_dir, train_subjects, val_subjects,
                       batch_size=4, num_workers=4,
                       samples_per_subject=None, eye='L',
                       ensure_multiview=False):
    """
    Create train and validation dataloaders.

    Args:
        base_dir: path to GazeGene dataset
        train_subjects: list of subject IDs for training (e.g., range(1, 47))
        val_subjects: list of subject IDs for validation (e.g., range(47, 57))
        batch_size: batch size
        num_workers: dataloader workers
        samples_per_subject: max frames per subject
        eye: which eye ('L' or 'R')
        ensure_multiview: require all 9 cameras per group

    Returns:
        train_loader, val_loader
    """
    train_dataset = GazeGeneDataset(
        base_dir=base_dir,
        subject_ids=train_subjects,
        samples_per_subject=samples_per_subject,
        eye=eye,
        augment=True,
    )

    val_dataset = GazeGeneDataset(
        base_dir=base_dir,
        subject_ids=val_subjects,
        samples_per_subject=samples_per_subject,
        eye=eye,
        augment=False,
    )

    if ensure_multiview:
        train_sampler = MultiViewBatchSampler(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_sampler = MultiViewBatchSampler(
            val_dataset, batch_size=batch_size, shuffle=False)

        train_loader = DataLoader(
            train_dataset, batch_sampler=train_sampler,
            num_workers=num_workers, collate_fn=gazegene_collate_fn,
            pin_memory=True)
        val_loader = DataLoader(
            val_dataset, batch_sampler=val_sampler,
            num_workers=num_workers, collate_fn=gazegene_collate_fn,
            pin_memory=True)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=gazegene_collate_fn,
            pin_memory=True, drop_last=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=gazegene_collate_fn,
            pin_memory=True)

    return train_loader, val_loader
