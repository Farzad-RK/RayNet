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
        samples_per_subject=None,  # Number of unique frames per subject (not per camera)
        transform=None,
        balance_attributes=None,
        seed=42
    ):
        self.samples = []
        self.index_by_key = defaultdict(list)
        self.attr_dict = {}
        self.transform = transform
        self.balance_attributes = balance_attributes
        random.seed(seed)

        # Load subject attributes (for balancing, e.g. ethnicity, gender)
        if self.balance_attributes:
            for subject in os.listdir(base_dir):
                if not subject.startswith('subject'): continue
                attr_path = os.path.join(base_dir, subject, 'subject_label.pkl')
                if os.path.exists(attr_path):
                    with open(attr_path, 'rb') as f:
                        attrs = pickle.load(f)
                    self.attr_dict[int(subject.replace('subject', ''))] = attrs

        subjects = subject_ids if subject_ids else sorted([d for d in os.listdir(base_dir) if d.startswith('subject')])
        for subject in subjects:
            subj_num = int(subject.replace('subject',''))
            label_dir = os.path.join(base_dir, subject, 'labels')

            # --- Efficient: Load each .pkl for this subject+camera only ONCE ---
            complex_labels = {}
            gaze_labels = {}
            for cam_id in (camera_ids if camera_ids is not None else range(9)):
                cstr = f'camera{cam_id}'
                with open(os.path.join(label_dir, f'complex_label_{cstr}.pkl'), 'rb') as f:
                    complex_labels[cam_id] = pickle.load(f)
                with open(os.path.join(label_dir, f'gaze_label_{cstr}.pkl'), 'rb') as f:
                    gaze_labels[cam_id] = pickle.load(f)

            num_frames = len(complex_labels[0]['img_path'])
            if samples_per_subject is not None:
                frame_idxs = random.sample(range(num_frames), min(samples_per_subject, num_frames))
            else:
                frame_idxs = range(num_frames)

            for idx in frame_idxs:
                for cam_id in (camera_ids if camera_ids is not None else range(9)):
                    complex_label = complex_labels[cam_id]
                    gaze_label = gaze_labels[cam_id]
                    sample = {
                        'img_path': os.path.join(base_dir, subject, complex_label['img_path'][idx]),
                        'subject': subj_num,
                        'camera': cam_id,
                        'frame_idx': idx,
                        'mesh': {
                            'eyeball_center_3D': complex_label['eyeball_center_3D'][idx],
                            'pupil_center_3D': complex_label['pupil_center_3D'][idx],
                            'iris_mesh_3D': complex_label['iris_mesh_3D'][idx],
                        },
                        'intrinsic': complex_label['intrinsic_matrix_cropped'][idx],
                        'gaze': {
                            'gaze_C': gaze_label['gaze_C'][idx],
                            'visual_axis_L': gaze_label['visual_axis_L'][idx],
                            'visual_axis_R': gaze_label['visual_axis_R'][idx],
                            'optic_axis_L': gaze_label['optic_axis_L'][idx],
                            'optic_axis_R': gaze_label['optic_axis_R'][idx],
                        },
                        'gaze_point': gaze_label['gaze_target'][idx],
                        'head_pose': {
                            'R': gaze_label['head_R_mat'][idx],
                            't': gaze_label['head_T_vec'][idx],
                        },
                        'attributes': self.attr_dict.get(subj_num, None)
                    }
                    self.samples.append(sample)
                    self.index_by_key[(subj_num, idx)].append(len(self.samples) - 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
        def to_tensor(x): return torch.from_numpy(np.asarray(x)).float()
        out = {
            'img': img,
            'subject': s['subject'],
            'camera': s['camera'],
            'frame_idx': s['frame_idx'],
            'mesh': {k: to_tensor(v) for k,v in s['mesh'].items()},
            'gaze': {k: to_tensor(v) for k,v in s['gaze'].items()},
            'gaze_point': to_tensor(s['gaze_point']),
            'head_pose': {'R': to_tensor(s['head_pose']['R']), 't': to_tensor(s['head_pose']['t'])},
            'intrinsic': to_tensor(s['intrinsic']),
            'attributes': s['attributes'],
        }
        return out

class MultiViewBatchSampler(Sampler):
    """
    Each batch consists of all 9 camera views for multiple (subject, frame) samples.
    Optionally balances over attributes like skin color, eye color, etc.
    Supports dynamic batch sizes.
    """
    def __init__(self, dataset, batch_size=1, balance_attributes=None, shuffle=True):
        self.index_by_key = dataset.index_by_key  # (subject, frame_idx) -> [indices for all cameras]
        self.keys = list(self.index_by_key.keys())
        self.balance_attributes = balance_attributes
        self.shuffle = shuffle
        self.batch_size = batch_size

        if balance_attributes:
            self.grouped_keys = defaultdict(list)
            for k in self.keys:
                subj, _ = k
                attrs = dataset.attr_dict.get(subj, {})
                attr_vals = tuple([attrs[a] for a in balance_attributes if a in attrs])
                self.grouped_keys[attr_vals].append(k)
            self.groups = list(self.grouped_keys.keys())
        else:
            self.grouped_keys = None
            self.groups = None

    def __iter__(self):
        if self.grouped_keys:
            group_keys = self.groups.copy()
            if self.shuffle: random.shuffle(group_keys)
            all_keys = []
            for g in group_keys:
                klist = self.grouped_keys[g]
                if self.shuffle: random.shuffle(klist)
                all_keys.extend(klist)
            if self.shuffle: random.shuffle(all_keys)
        else:
            all_keys = self.keys.copy()
            if self.shuffle: random.shuffle(all_keys)

        batch = []
        for k in all_keys:
            indices = self.index_by_key[k]
            if len(indices) == 9:  # Ensure all cameras present
                batch.extend(indices)
                if len(batch) == 9 * self.batch_size:
                    yield batch
                    batch = []

        if batch:  # Yield remaining samples
            yield batch

    def __len__(self):
        total_complete_samples = len(self.keys)
        return total_complete_samples // self.batch_size


# --- USAGE EXAMPLE ---
# batch_sampler = MultiViewBatchSampler(
#     dataset,
#     batch_size=args.batch_size,
#     balance_attributes=['ethicity'],
#     shuffle=True
# )
#
# loader = DataLoader(
#     dataset,
#     batch_sampler=batch_sampler,
#     num_workers=args.num_workers,
#     pin_memory=True
# )
