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
    Each batch consists of all 9 camera views for a given (subject, frame).
    Optionally balances over attributes like skin color, eye color, etc.
    """
    def __init__(self, dataset, balance_attributes=None, shuffle=True):
        self.index_by_key = dataset.index_by_key  # (subject, frame_idx) -> [indices for all cameras]
        self.keys = list(self.index_by_key.keys())
        self.balance_attributes = balance_attributes
        self.shuffle = shuffle

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
            all_batches = []
            for g in group_keys:
                klist = self.grouped_keys[g]
                if self.shuffle: random.shuffle(klist)
                all_batches.extend(klist)
            if self.shuffle: random.shuffle(all_batches)
            for k in all_batches:
                indices = self.index_by_key[k]
                if len(indices) == 9:  # all cameras present
                    yield indices
        else:
            keys = self.keys.copy()
            if self.shuffle: random.shuffle(keys)
            for k in keys:
                indices = self.index_by_key[k]
                if len(indices) == 9:
                    yield indices

    def __len__(self):
        if self.grouped_keys:
            return sum(len(v) for v in self.grouped_keys.values())
        return len(self.keys)

# --- USAGE EXAMPLE ---
if __name__ == '__main__':
    base_dir = './GazeGene_FaceCrops'
    dataset = GazeGeneDataset(
        base_dir,
        samples_per_subject=50,       # Only 50 random frames per subject
        transform=None,               # or your torchvision transforms
        balance_attributes=['ethicity']  # or other attribute(s) from subject_label.pkl
    )

    batch_sampler = MultiViewBatchSampler(dataset, balance_attributes=['ethicity'], shuffle=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4)

    for batch in loader:
        # batch['img'] is a list of 9 images (all cameras) per sample
        print(batch['img'][0].shape, batch['gaze']['gaze_C'][0])
        break  # just demo
