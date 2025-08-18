import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import cv2
from collections import defaultdict
import random
from typing import Any, Dict, List, Optional, Tuple


def _safe_pickle_load(path: str):
    with open(path, "rb") as f:
        # Some datasets are saved with different pickling protocols; keep it simple
        return pickle.load(f)


class GazeGeneDataset(Dataset):
    """
    Loads per-subject, per-camera, per-frame samples from GazeGene.
    Returns tensors and includes:
      - RGB image (face crop)
      - Intrinsics for the crop (K)
      - Head pose (R, t) in Camera Coord. System (CCS)
      - 3D iris ring (2 eyes × 100 × 3), pupil & eyeball centers (2 × 3)
      - Optional 2D versions of the above for reprojection supervision
      - Gaze axes (visual/optic), gaze target and depth
      - Subject-level "β-like" geometric params: eye centers (HCS), radii, cornea depth, kappa, UVRadius
    """

    def __init__(
        self,
        base_dir: str,
        subject_ids: Optional[List[str]] = None,   # e.g., ["subject01", "subject02"]
        camera_ids: Optional[List[int]] = None,    # e.g., [0,1,2,...,8]
        samples_per_subject: Optional[int] = None, # number of unique frames per subject
        transform=None,
        balance_attributes: Optional[List[str]] = None,  # e.g., ['ethicity', 'gender']
        include_2d: bool = True,
        seed: int = 42,
    ):
        super().__init__()

        self.samples: List[Dict[str, Any]] = []
        self.index_by_key: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.attr_dict: Dict[int, Dict[str, Any]] = {}
        self.transform = transform
        self.balance_attributes = balance_attributes
        self.include_2d = include_2d
        random.seed(seed)

        # subjects to iterate
        subjects_fs = sorted([d for d in os.listdir(base_dir) if d.startswith("subject")])
        subjects = subject_ids if subject_ids else subjects_fs

        for subject in subjects:
            subj_num = int(subject.replace("subject", ""))

            subj_dir = os.path.join(base_dir, subject)
            label_dir = os.path.join(subj_dir, "labels")
            subj_label_path = os.path.join(subj_dir, "subject_label.pkl")

            if not os.path.exists(subj_label_path):
                raise FileNotFoundError(f"Missing subject_label.pkl for {subject}")

            # Load subject-level parameters (β-like)
            subj_params = _safe_pickle_load(subj_label_path)

            # Save attributes for optional balancing (safe even if not used)
            self.attr_dict[subj_num] = {
                "ID": subj_params.get("ID", subj_num),
                "gender": subj_params.get("gender", None),
                "ethicity": subj_params.get("ethicity", None),
            }

            # Per-camera labels (load once per subject)
            complex_labels: Dict[int, Dict[str, Any]] = {}
            gaze_labels: Dict[int, Dict[str, Any]] = {}
            cams = camera_ids if camera_ids is not None else list(range(9))
            for cam_id in cams:
                cstr = f"camera{cam_id}"
                complex_path = os.path.join(label_dir, f"complex_label_{cstr}.pkl")
                gaze_path = os.path.join(label_dir, f"gaze_label_{cstr}.pkl")
                if not (os.path.exists(complex_path) and os.path.exists(gaze_path)):
                    raise FileNotFoundError(f"Missing label pkl(s) for {subject} {cstr}")
                complex_labels[cam_id] = _safe_pickle_load(complex_path)
                gaze_labels[cam_id] = _safe_pickle_load(gaze_path)

            # Determine frame count from any camera (assumes aligned labels)
            num_frames = len(complex_labels[cams[0]]["img_path"])

            if samples_per_subject is not None:
                frame_idxs = random.sample(range(num_frames), min(samples_per_subject, num_frames))
            else:
                frame_idxs = range(num_frames)

            for idx in frame_idxs:
                for cam_id in cams:
                    complex_label = complex_labels[cam_id]
                    gaze_label = gaze_labels[cam_id]

                    # Build per-sample record
                    sample = {
                        "img_path": os.path.join(base_dir, subject, complex_label["img_path"][idx]),
                        "subject": subj_num,
                        "camera": cam_id,
                        "frame_idx": idx,

                        # 3D geometry (CCS)
                        "mesh": {
                            "eyeball_center_3D": complex_label["eyeball_center_3D"][idx],  # [2,3]
                            "pupil_center_3D":   complex_label["pupil_center_3D"][idx],     # [2,3]
                            "iris_mesh_3D":      complex_label["iris_mesh_3D"][idx],        # [2,100,3]
                        },

                        # Optional 2D geometry (pixels of cropped image)
                        "mesh2d": None,

                        # intrinsics for the cropped/scaled face image (3×3)
                        "intrinsic": complex_label["intrinsic_matrix_cropped"][idx],

                        # gaze & pose (CCS)
                        "gaze": {
                            "gaze_C":        gaze_label["gaze_C"][idx],         # [3]
                            "visual_axis_L": gaze_label["visual_axis_L"][idx],  # [3]
                            "visual_axis_R": gaze_label["visual_axis_R"][idx],  # [3]
                            "optic_axis_L":  gaze_label["optic_axis_L"][idx],   # [3]
                            "optic_axis_R":  gaze_label["optic_axis_R"][idx],   # [3]
                            "gaze_depth":    gaze_label["gaze_depth"][idx],     # scalar
                        },
                        "gaze_point": gaze_label["gaze_target"][idx],           # [3]

                        # head pose (CCS)
                        "head_pose": {
                            "R": gaze_label["head_R_mat"][idx],   # [3,3]
                            "t": gaze_label["head_T_vec"][idx],   # [3]
                        },

                        # subject-level parameters (HCS) needed for β supervision/priors
                        "subject_params": {
                            "eyecenter_L":   subj_params["eyecenter_L"],  # [3]
                            "eyecenter_R":   subj_params["eyecenter_R"],  # [3]
                            "eyeball_radius":subj_params["eyeball_radius"], # scalar
                            "iris_radius":   subj_params["iris_radius"],    # scalar
                            "cornea_radius": subj_params["cornea_radius"],  # scalar
                            "cornea2center": subj_params["cornea2center"],  # scalar
                            "UVRadius":      subj_params["UVRadius"],       # normalized pupil size (subject prior)
                            "L_kappa":       subj_params["L_kappa"],        # [3]
                            "R_kappa":       subj_params["R_kappa"],        # [3]
                        },

                        # demographics etc. (optional)
                        "attributes": self.attr_dict.get(subj_num, None),
                    }

                    # Attach 2D labels if requested
                    if self.include_2d:
                        sample["mesh2d"] = {
                            "eyeball_center_2D": complex_label["eyeball_center_2D"][idx],  # [2,2]
                            "pupil_center_2D":   complex_label["pupil_center_2D"][idx],    # [2,2]
                            "iris_mesh_2D":      complex_label["iris_mesh_2D"][idx],       # [2,100,2]
                        }

                    self.samples.append(sample)
                    self.index_by_key[(subj_num, idx)].append(len(self.samples) - 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        img = cv2.imread(s["img_path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {s['img_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            # Let user's transform handle HWC numpy or PIL; expect it returns CHW torch.FloatTensor [0..1]
            img_t = self.transform(img)
        else:
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        def to_tensor(x):
            return torch.from_numpy(np.asarray(x)).float()

        out: Dict[str, Any] = {
            "img": img_t,
            "subject": s["subject"],
            "camera": s["camera"],
            "frame_idx": s["frame_idx"],

            "mesh": {k: to_tensor(v) for k, v in s["mesh"].items()},
            "gaze": {k: to_tensor(v) for k, v in s["gaze"].items()},
            "gaze_point": to_tensor(s["gaze_point"]),
            "head_pose": {"R": to_tensor(s["head_pose"]["R"]),
                          "t": to_tensor(s["head_pose"]["t"])},
            "intrinsic": to_tensor(s["intrinsic"]),
            "attributes": s["attributes"],
            "subject_params": {
                "eyecenter_L":   to_tensor(s["subject_params"]["eyecenter_L"]),
                "eyecenter_R":   to_tensor(s["subject_params"]["eyecenter_R"]),
                "eyeball_radius":float(s["subject_params"]["eyeball_radius"]),
                "iris_radius":   float(s["subject_params"]["iris_radius"]),
                "cornea_radius": float(s["subject_params"]["cornea_radius"]),
                "cornea2center": float(s["subject_params"]["cornea2center"]),
                "UVRadius":      float(s["subject_params"]["UVRadius"]),
                "L_kappa":       to_tensor(s["subject_params"]["L_kappa"]),
                "R_kappa":       to_tensor(s["subject_params"]["R_kappa"]),
            },
        }

        if s["mesh2d"] is not None:
            out["mesh2d"] = {k: to_tensor(v) for k, v in s["mesh2d"].items()}

        return out


class MultiViewBatchSampler(Sampler):
    """
    Each batch = all 9 camera views for 'batch_size' distinct (subject, frame) groups.
    """
    def __init__(self, dataset: GazeGeneDataset, batch_size: int = 1,
                 balance_attributes: Optional[List[str]] = None, shuffle: bool = True):
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

        batch: List[int] = []
        for k in all_keys:
            indices = self.index_by_key[k]
            if len(indices) == 9:  # ensure complete multi-view set
                batch.extend(indices)
                if len(batch) == 9 * self.batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch

    def __len__(self):
        total_complete_samples = len(self.keys)
        return total_complete_samples // self.batch_size


def multiview_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Packs a flat list of length (B*9) into tensors of shape [B, 9, ...].
    Assumes MultiViewBatchSampler grouped the items such that every 9 entries form a (subject, frame) set.
    """
    assert len(batch) % 9 == 0, "Batch size must be a multiple of 9 (views)"
    grouped = [batch[i:i+9] for i in range(0, len(batch), 9)]
    # sort views by camera id to ensure deterministic order
    for g in grouped:
        g.sort(key=lambda s: s["camera"])
    B, V = len(grouped), 9

    def stack_field(getter):
        tensors = []
        for g in grouped:
            tensors.append(torch.stack([getter(s) for s in g], dim=0))  # [V,...]
        return torch.stack(tensors, dim=0)  # [B,V,...]

    out: Dict[str, Any] = {}
    out["img"]         = stack_field(lambda s: s["img"])                        # [B,V,3,H,W]
    out["K"]           = stack_field(lambda s: s["intrinsic"])                  # [B,V,3,3]
    out["R_head"]      = stack_field(lambda s: s["head_pose"]["R"])             # [B,V,3,3]
    out["t_head"]      = stack_field(lambda s: s["head_pose"]["t"])             # [B,V,3]
    out["iris3d"]      = stack_field(lambda s: s["mesh"]["iris_mesh_3D"])       # [B,V,2,100,3]
    out["pupil3d"]     = stack_field(lambda s: s["mesh"]["pupil_center_3D"])    # [B,V,2,3]
    out["eyecenter3d"] = stack_field(lambda s: s["mesh"]["eyeball_center_3D"])  # [B,V,2,3]

    # Optional 2D supervision
    if "mesh2d" in grouped[0][0] and grouped[0][0]["mesh2d"] is not None:
        out["iris2d"]      = stack_field(lambda s: s["mesh2d"]["iris_mesh_2D"])      # [B,V,2,100,2]
        out["pupil2d"]     = stack_field(lambda s: s["mesh2d"]["pupil_center_2D"])   # [B,V,2,2]
        out["eyecenter2d"] = stack_field(lambda s: s["mesh2d"]["eyeball_center_2D"]) # [B,V,2,2]

    out["optic_axis"] = stack_field(lambda s: torch.stack([s["gaze"]["optic_axis_L"],
                                                           s["gaze"]["optic_axis_R"]], dim=0))  # [B,V,2,3]
    out["visual_axis"] = stack_field(lambda s: torch.stack([s["gaze"]["visual_axis_L"],
                                                            s["gaze"]["visual_axis_R"]], dim=0)) # [B,V,2,3]
    out["gaze_point"]  = stack_field(lambda s: s["gaze_point"])                    # [B,V,3]
    out["gaze_depth"]  = stack_field(lambda s: s["gaze"]["gaze_depth"])            # [B,V]

    # Subject-level β (same across the 9 views in a group) — take view 0 per group
    def take_beta(key):
        vals = []
        for g in grouped:
            v = g[0]["subject_params"][key]
            if torch.is_tensor(v):
                vals.append(v)
            else:
                vals.append(torch.tensor(v, dtype=torch.float32))
        return torch.stack(vals, dim=0)  # [B,...]

    out["beta"] = {
        "c_L": take_beta("eyecenter_L"),   # [B,3]   (HCS)
        "c_R": take_beta("eyecenter_R"),   # [B,3]
        "r_eye":     take_beta("eyeball_radius").unsqueeze(-1),  # [B,1]
        "r_iris":    take_beta("iris_radius").unsqueeze(-1),     # [B,1]
        "r_cornea":  take_beta("cornea_radius").unsqueeze(-1),   # [B,1]
        "d_cornea":  take_beta("cornea2center").unsqueeze(-1),   # [B,1]
        "UV":        take_beta("UVRadius").unsqueeze(-1),        # [B,1]
        "kappa_L":   take_beta("L_kappa"),                       # [B,3]
        "kappa_R":   take_beta("R_kappa"),                       # [B,3]
    }

    out["subject"]   = torch.tensor([grouped[i][0]["subject"] for i in range(B)], dtype=torch.long)
    out["frame_idx"] = torch.tensor([grouped[i][0]["frame_idx"] for i in range(B)], dtype=torch.long)
    out["cameras"]   = torch.arange(V, dtype=torch.long)

    return out
