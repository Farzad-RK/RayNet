"""
GazeGene dataset loader for RayNet v4.

Input: GazeGene face crops resized to 224×224.
Output: 224×224 face image tensor (NO Zhang normalization warp).

Key design:
  - 224×224 resolution (matches MDS shard storage, reduces memory 4× vs 448)
  - Optical axis GT from GazeGene gaze_label (optic_axis_L/R) in CCS
  - R_cam (camera extrinsics) used for multi-view world-frame transform
  - 14 landmarks: 10 iris contour + 4 pupil boundary (subsampled from 100)
  - GazeGene units: centimeters (all 3D coordinates)

Feature map sizes at 224 input (RepNeXt stride pattern):
  P2 = 56×56 (stride 4), P3 = 28×28 (stride 8),
  P4 = 14×14 (stride 16), P5 = 7×7 (stride 32)
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import cv2
from collections import defaultdict
import random

from RayNet.kappa import build_R_kappa

# Indices to subsample 100 iris points down to 10 evenly-spaced points
IRIS_SUBSAMPLE_IDX = list(range(0, 100, 10))  # [0, 10, 20, ..., 90]


class GazeGeneDataset(Dataset):
    """
    GazeGene dataset for RayNet v4.

    Each sample yields a 224×224 face image and targets:
      - 14 landmarks in feature map space (10 iris + 4 pupil)
      - optical axis in CCS (unit vector, from gaze_label)
      - R_cam for multi-view world-frame consistency
    """

    def __init__(
            self,
            base_dir,
            subject_ids=None,
            camera_ids=None,
            samples_per_subject=None,
            eye='L',
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
            img_size: output image size (default 224, matching MDS shards)
            augment: whether to apply data augmentation
            seed: random seed
        """
        self.samples = []
        self.index_by_key = defaultdict(list)
        self.attr_dict = {}
        self.camera_params = {}

        self.eye = eye
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
                        # 3D data (camera coordinate system, units: cm)
                        'eyeball_center_3D': cl['eyeball_center_3D'][idx],  # [2, 3]
                        'pupil_center_3D': cl['pupil_center_3D'][idx],      # [2, 3]
                        'iris_mesh_3D': cl['iris_mesh_3D'][idx],            # [2, 100, 3]
                        # 2D data (pixel space, face crop)
                        'iris_mesh_2D': cl['iris_mesh_2D'][idx],            # [2, 100, 2]
                        'pupil_center_2D': cl['pupil_center_2D'][idx],      # [2, 2]
                        # Head pose
                        'head_R': gl['head_R_mat'][idx],  # [3, 3]
                        'head_t': gl['head_T_vec'][idx],  # [3]
                        # Camera intrinsics (cropped)
                        'K_cropped': cl['intrinsic_matrix_cropped'][idx],    # [3, 3]
                        # --- Gaze labels from gaze_label (pre-computed by GazeGene) ---
                        'optic_axis_L': gl['optic_axis_L'][idx],   # [3] unit vector, CCS
                        'optic_axis_R': gl['optic_axis_R'][idx],   # [3] unit vector, CCS
                        'visual_axis_L': gl['visual_axis_L'][idx], # [3] unit vector, CCS
                        'visual_axis_R': gl['visual_axis_R'][idx], # [3] unit vector, CCS
                        'gaze_C': gl['gaze_C'][idx],               # [3] unit head gaze, CCS
                        'gaze_target': gl['gaze_target'][idx],     # [3] 3D target position, CCS
                        'gaze_depth': gl['gaze_depth'][idx],       # scalar, vergence depth
                    }

                    self.samples.append(sample)
                    self.index_by_key[(subj_num, idx)].append(len(self.samples) - 1)

        print(f"Loaded {len(self.samples)} samples "
              f"({len(set(s['subject'] for s in self.samples))} subjects)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a sample for training.

        Output dict:
            image: (3, img_size, img_size) face crop tensor
            landmark_coords: (14, 2) landmark coords in feature map space
            landmark_coords_px: (14, 2) landmark coords in pixel space
            optical_axis: (3,) GT optical axis unit vector in CCS
            R_cam: (3, 3) camera extrinsic rotation (for multi-view)
            R_kappa: (3, 3) kappa rotation matrix
            subject: int subject ID
            cam_id: int camera ID
        """
        s = self.samples[idx]
        subj_num = s['subject']
        eye_idx = 0 if self.eye == 'L' else 1

        # Load image (face crop, resized to img_size)
        img = cv2.imread(s['img_path'])
        if img is None:
            raise FileNotFoundError(f"Image not found: {s['img_path']}")

        # Get the original image size for landmark scaling
        orig_h, orig_w = img.shape[:2]

        # --- Use native resolution or resize if needed ---
        if orig_h == self.img_size and orig_w == self.img_size:
            img_resized = img  # already at target size
        else:
            img_resized = cv2.resize(img, (self.img_size, self.img_size),
                                     interpolation=cv2.INTER_LINEAR)

        # Get camera extrinsics (for multi-view world-frame transform)
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

        # Eye center and pupil center in camera coordinates
        t_eye = np.array(s['eyeball_center_3D'][eye_idx], dtype=np.float64)
        t_pupil = np.array(s['pupil_center_3D'][eye_idx], dtype=np.float64)

        # Head pose rotation (per-frame, varies across samples)
        head_R = np.array(s['head_R'], dtype=np.float64)  # (3, 3)

        # --- Ground truth optical axis (from GazeGene gaze_label) ---
        # Used directly in CCS — no normalization rotation
        optic_key = f'optic_axis_{self.eye}'
        optical_axis_ccs = np.array(s[optic_key], dtype=np.float64)
        oa_norm = np.linalg.norm(optical_axis_ccs)
        if oa_norm > 1e-8:
            optical_axis_ccs = optical_axis_ccs / oa_norm

        # Gaze target and depth
        gaze_target = np.array(s['gaze_target'], dtype=np.float64)
        gaze_depth = float(s['gaze_depth'])

        # --- Iris landmarks: subsample 100 -> 10 + 4 pupil points ---
        iris_2d = np.array(s['iris_mesh_2D'][eye_idx], dtype=np.float64)  # (100, 2)
        iris_10 = iris_2d[IRIS_SUBSAMPLE_IDX]  # (10, 2)

        # Pupil boundary: 4 iris points closest to pupil center
        pupil_2d = np.array(s['pupil_center_2D'][eye_idx], dtype=np.float64)  # (2,)
        dists = np.linalg.norm(iris_2d - pupil_2d[None, :], axis=1)
        pupil_boundary_idx = np.argsort(dists)[:4]
        pupil_4 = iris_2d[pupil_boundary_idx]  # (4, 2)

        # Combine: 10 iris + 4 pupil = 14 landmarks in 448 pixel space
        landmarks_2d = np.concatenate([iris_10, pupil_4], axis=0)  # (14, 2)

        # Scale landmarks from original image space to target pixel space
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        landmarks_px = landmarks_2d * np.array([scale_x, scale_y])  # (14, 2)

        # Scale to feature map space (P2 stride=4, so 224/4=56)
        landmarks_feat = landmarks_px / 4.0  # (14, 2) in [0, img_size/4) range

        # --- Kappa rotation matrix ---
        kappa_key = f'{self.eye}_kappa'
        if kappa_key in subject_attrs:
            kappa_angles = np.array(subject_attrs[kappa_key], dtype=np.float64)
            R_kappa = build_R_kappa(kappa_angles)
        else:
            R_kappa = np.eye(3, dtype=np.float64)

        # --- Convert image to tensor (uint8, normalized in train.py) ---
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).contiguous()
        # NOTE: kept as uint8 [0,255] — train.py does .float().div_(255.0)
        # This matches StreamingGazeGeneDataset which also returns uint8.

        # --- Data augmentation ---
        if self.augment:
            img_tensor = self._augment(img_tensor.float() / 255.0)
            # Augmentation operates on [0,1] floats, convert back to uint8
            img_tensor = (img_tensor * 255.0).clamp(0, 255).byte()

        return {
            'image': img_tensor,                                            # (3, img_size, img_size)
            'landmark_coords': torch.from_numpy(landmarks_feat).float(),    # (14, 2) feature map
            'landmark_coords_px': torch.from_numpy(landmarks_px).float(),   # (14, 2) pixel space
            'optical_axis': torch.from_numpy(optical_axis_ccs).float(),     # (3,) CCS unit vector
            'R_kappa': torch.from_numpy(R_kappa).float(),                   # (3, 3)
            # Camera parameters for multi-view consistency
            'K': torch.from_numpy(K).float(),                               # (3, 3) intrinsics
            'R_cam': torch.from_numpy(R_cam).float(),                       # (3, 3) extrinsic rotation
            'T_cam': torch.from_numpy(T_cam).float(),                       # (3,) extrinsic translation
            'head_R': torch.from_numpy(head_R).float(),                     # (3, 3) head pose rotation
            'head_t': torch.from_numpy(np.array(s['head_t'], dtype=np.float64)).float(),  # (3,) head translation
            'eyeball_center_3d': torch.from_numpy(t_eye).float(),           # (3,) eye center in CCS
            'pupil_center_3d': torch.from_numpy(t_pupil).float(),           # (3,) pupil center in CCS
            # GazeGene gaze labels
            'gaze_target': torch.from_numpy(gaze_target).float(),           # (3,) 3D target, CCS
            'gaze_depth': torch.tensor(gaze_depth).float(),                 # scalar, vergence depth
            'subject': subj_num,
            'cam_id': s['cam_id'],
            'frame_idx': s['frame_idx'],
        }

    def _augment(self, img_tensor):
        """
        Data augmentation matching GazeGene paper methodology (Sec 4.1.3):
        random translation + color jitter.
        """
        from torchvision import transforms as T

        # Color jitter (matching paper's cross-domain protocol)
        jitter = T.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.2, hue=0.1)
        img_tensor = jitter(img_tensor)

        # Random translation (small shift, ~5% of image)
        if random.random() > 0.5:
            max_shift = int(self.img_size * 0.05)  # ~11 pixels for 224
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
            img_tensor = torch.roll(img_tensor, shifts=(dy, dx), dims=(1, 2))

        return img_tensor


class MultiViewBatchSampler(Sampler):
    """
    Batch sampler that groups all 9 camera views of the same (subject, frame).
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, ensure_multiview=True):
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
    """Collate: stack tensors, collect scalars into lists."""
    if not batch:
        return {}

    collated = {}
    tensor_keys = ['image', 'landmark_coords', 'landmark_coords_px',
                   'optical_axis', 'R_kappa',
                   'K', 'R_cam', 'T_cam', 'head_R', 'head_t', 'eyeball_center_3d',
                   'gaze_target', 'gaze_depth']
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
    """Create train and validation dataloaders."""
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
