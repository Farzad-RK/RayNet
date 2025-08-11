import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import cv2
from collections import defaultdict
import random


def _as_subject_folder_list(base_dir, subject_ids):
    """Accept ['subject3', ...] or [3, ...] and normalize to folder names."""
    if subject_ids is None:
        return sorted([d for d in os.listdir(base_dir) if d.startswith('subject')])
    out = []
    for s in subject_ids:
        if isinstance(s, str) and s.startswith('subject'):
            out.append(s)
        else:
            out.append(f"subject{int(s)}")
    return out


def _safe_load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_camera_info_for_subject(base_dir, subject):
    """
    Try per-subject first: <base_dir>/<subject>/camera_info.pkl
    Fallback to dataset-level: <base_dir>/camera_info.pkl
    Returns: dict cam_id -> {'K_full','R_wc','T_wc_cm'}   (all np arrays)
    """
    cand = [
        os.path.join(base_dir, subject, "camera_info.pkl"),
        os.path.join(base_dir, "camera_info.pkl"),
    ]
    cam_map = {}
    for p in cand:
        if os.path.exists(p):
            info_list = _safe_load_pickle(p)
            for entry in info_list:
                cid = int(entry["cam_id"])
                cam_map[cid] = {
                    "K_full": np.asarray(entry["intrinsic_matrix"], dtype=np.float32),
                    "R_wc":   np.asarray(entry["R_mat"], dtype=np.float32),     # camera orientation in WCS
                    "T_wc_cm":np.asarray(entry["T_vec"], dtype=np.float32),     # camera position (cm) in WCS
                }
            break
    return cam_map  # may be empty if file missing


class GazeGeneDataset(Dataset):
    """
    Loads all 9 camera views per (subject, frame_idx). Also builds an index to
    recover (subject, frame) groups for multi-view sampling.

    NOTE: We ALWAYS load subject_label.pkl to get iris_radius (used by geometry),
    independent of whether attribute-balancing is requested.

    Extra fields added (if available from camera_info.pkl):
      - 'cam_id' (int), 'intrinsic_full' (3x3)
      - 'extrinsic': {'R_wc','T_wc_cm','R_cw','t_cw_cm'}  (all camera/world transforms; cm)
    """
    def __init__(
        self,
        base_dir,
        subject_ids=None,
        camera_ids=None,
        samples_per_subject=None,   # number of unique frames per subject
        transform=None,
        balance_attributes=None,
        seed=42,
    ):
        self.samples = []
        self.index_by_key = defaultdict(list)   # (subject_id:int, frame_idx:int) -> [flat indices for 9 cameras]
        self.attr_dict = {}                     # subject_id -> attrs (includes 'iris_radius')
        self.transform = transform
        self.balance_attributes = balance_attributes
        self.base_dir = base_dir
        random.seed(seed)

        # ---- Load per-subject attributes (ALWAYS), then optional balancing uses them
        for subject in os.listdir(base_dir):
            if not subject.startswith('subject'):
                continue
            attr_path = os.path.join(base_dir, subject, 'subject_label.pkl')
            if os.path.exists(attr_path):
                with open(attr_path, 'rb') as f:
                    attrs = pickle.load(f)
                try:
                    subj_num = int(subject.replace('subject', ''))
                except Exception:
                    continue
                self.attr_dict[subj_num] = attrs

        # ---- Enumerate subjects / cameras
        subjects = _as_subject_folder_list(base_dir, subject_ids)
        cam_list = list(camera_ids) if camera_ids is not None else list(range(9))

        for subject in subjects:
            subj_num = int(subject.replace('subject', ''))
            label_dir = os.path.join(base_dir, subject, 'labels')

            # Sanity: iris radius must exist
            if subj_num not in self.attr_dict or 'iris_radius' not in self.attr_dict[subj_num]:
                raise FileNotFoundError(
                    f"Missing 'iris_radius' for {subject}. "
                    f"Expected {os.path.join(base_dir, subject, 'subject_label.pkl')} to contain it."
                )

            # Optional: camera_info (intrinsics/extrinsics per camera)
            cam_info_map = _load_camera_info_for_subject(base_dir, subject)  # may be {}

            # Load each .pkl once per camera
            complex_labels = {}
            gaze_labels = {}
            for cam_id in cam_list:
                cstr = f'camera{cam_id}'
                with open(os.path.join(label_dir, f'complex_label_{cstr}.pkl'), 'rb') as f:
                    complex_labels[cam_id] = pickle.load(f)
                with open(os.path.join(label_dir, f'gaze_label_{cstr}.pkl'), 'rb') as f:
                    gaze_labels[cam_id] = pickle.load(f)

            num_frames = len(complex_labels[cam_list[0]]['img_path'])
            if samples_per_subject is not None:
                frame_idxs = random.sample(range(num_frames), min(samples_per_subject, num_frames))
            else:
                frame_idxs = range(num_frames)

            for idx in frame_idxs:
                for cam_id in cam_list:
                    c_label = complex_labels[cam_id]
                    g_label = gaze_labels[cam_id]

                    # Base fields from labels
                    img_rel = c_label['img_path'][idx]
                    img_abs = os.path.join(base_dir, subject, img_rel)

                    sample = {
                        'img_path': img_abs,
                        'subject': subj_num,
                        'camera': cam_id,
                        'frame_idx': idx,

                        # Mesh / 3D geometry (in centimeters by dataset definition)
                        'mesh': {
                            'eyeball_center_3D': c_label['eyeball_center_3D'][idx],
                            'pupil_center_3D':  c_label['pupil_center_3D'][idx],
                            'iris_mesh_3D':     c_label['iris_mesh_3D'][idx],
                            'iris_mesh_2D':     c_label['iris_mesh_2D'][idx],
                        },

                        # Cropped, per-frame intrinsics aligned to face crop (preferred for training)
                        'intrinsic': c_label['intrinsic_matrix_cropped'][idx],

                        # Scalar iris radius [cm] (from subject-level labels)
                        'iris_radius_cm': self.attr_dict[subj_num]['iris_radius'],

                        # Gaze labels
                        'gaze': {
                            'gaze_C':        g_label['gaze_C'][idx],       # in camera coords
                            'visual_axis_L': g_label['visual_axis_L'][idx],
                            'visual_axis_R': g_label['visual_axis_R'][idx],
                            'optic_axis_L':  g_label['optic_axis_L'][idx],
                            'optic_axis_R':  g_label['optic_axis_R'][idx],
                            'gaze_depth':    g_label['gaze_depth'][idx],
                        },
                        'gaze_point': g_label['gaze_target'][idx],         # typically in WCS

                        # Head pose (camera coords)
                        'head_pose': {
                            'R': g_label['head_R_mat'][idx],
                            't': g_label['head_T_vec'][idx],
                        },
                    }

                    # --- Optional extras from camera_info.pkl (if present) ---
                    if cam_id in cam_info_map:
                        K_full = cam_info_map[cam_id]["K_full"]
                        R_wc   = cam_info_map[cam_id]["R_wc"]
                        T_wc   = cam_info_map[cam_id]["T_wc_cm"]  # centimeters

                        # Build world->camera transform: X_C = R_cw * (X_W - T_wc)
                        R_cw = R_wc.T
                        t_cw = (-R_cw @ T_wc.reshape(3, 1)).reshape(3)

                        sample.update({
                            'cam_id': cam_id,
                            'intrinsic_full': K_full,
                            'extrinsic': {
                                'R_wc': R_wc,       # camera orientation in world (3x3)
                                'T_wc_cm': T_wc,    # camera position in world [cm] (3,)
                                'R_cw': R_cw,       # world->camera rotation (3x3)
                                't_cw_cm': t_cw,    # world->camera translation [cm] (3,)
                            }
                        })

                    self.samples.append(sample)
                    self.index_by_key[(subj_num, idx)].append(len(self.samples) - 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = cv2.imread(s['img_path'])
        if img is None:
            raise FileNotFoundError(f"Could not read image at {s['img_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)  # user-provided transform must return CHW tensor
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().contiguous() / 255.0

        def to_tensor(x):
            return torch.from_numpy(np.asarray(x)).float()

        out = {
            'img': img,
            'subject': s['subject'],
            'camera': s['camera'],
            'frame_idx': s['frame_idx'],

            'mesh': {k: to_tensor(v) for k, v in s['mesh'].items()},
            'gaze': {k: to_tensor(v) for k, v in s['gaze'].items()},
            'gaze_point': to_tensor(s['gaze_point']),

            'head_pose': {'R': to_tensor(s['head_pose']['R']),
                          't': to_tensor(s['head_pose']['t'])},

            # Preferred crop-aware intrinsics for training
            'intrinsic': to_tensor(s['intrinsic']),

            # Radius in centimeters (scalar tensor)
            'iris_radius_cm': torch.tensor(s['iris_radius_cm'], dtype=torch.float32),
        }

        # Optional: pass through extra camera info as tensors if present
        if 'intrinsic_full' in s:
            out['intrinsic_full'] = to_tensor(s['intrinsic_full'])
        if 'extrinsic' in s:
            ext = s['extrinsic']
            out['extrinsic'] = {
                'R_wc': to_tensor(ext['R_wc']),
                'T_wc_cm': to_tensor(ext['T_wc_cm']),
                'R_cw': to_tensor(ext['R_cw']),
                't_cw_cm': to_tensor(ext['t_cw_cm']),
            }
        if 'cam_id' in s:
            out['cam_id'] = torch.tensor(s['cam_id'], dtype=torch.int64)

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
        self.dataset = dataset

        if balance_attributes:
            self.grouped_keys = defaultdict(list)
            for k in self.keys:
                subj, _ = k
                attrs = dataset.attr_dict.get(subj, {})
                # Only include requested attributes that exist
                attr_vals = tuple([attrs[a] for a in balance_attributes if a in attrs])
                self.grouped_keys[attr_vals].append(k)
            self.groups = list(self.grouped_keys.keys())
        else:
            self.grouped_keys = None
            self.groups = None

    def __iter__(self):
        if self.grouped_keys:
            group_keys = self.groups.copy()
            if self.shuffle:
                random.shuffle(group_keys)
            all_keys = []
            for g in group_keys:
                klist = self.grouped_keys[g]
                if self.shuffle:
                    random.shuffle(klist)
                all_keys.extend(klist)
            if self.shuffle:
                random.shuffle(all_keys)
        else:
            all_keys = self.keys.copy()
            if self.shuffle:
                random.shuffle(all_keys)

        batch = []
        for k in all_keys:
            indices = self.index_by_key[k]
            if len(indices) == 9:  # Ensure all cameras present
                batch.extend(indices)
                if len(batch) == 9 * self.batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch

    def __len__(self):
        complete = sum(1 for _, idxs in self.index_by_key.items() if len(idxs) == 9)
        return complete // self.batch_size
