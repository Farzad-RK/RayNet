import glob
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ARGazeDataset(Dataset):
    def __init__(self, root_dir, subject_ids, transform=None, camera="C1", max_samples=None):
        self.data = []
        self.transform = transform
        self.camera = camera

        for sid in subject_ids:
            subj_path = os.path.join(root_dir, sid)
            if not os.path.isdir(subj_path): continue

            sessions = [d for d in os.listdir(subj_path) if d.startswith(f"{sid}_S")]
            for session in sessions:
                session_path = os.path.join(subj_path, session)
                image_folder = os.path.join(session_path, f"{session}_{camera}")
                if not os.path.isdir(image_folder):
                    print(f"⚠️ Missing folder: {image_folder}")
                    continue

                target_path = os.path.join(session_path, "target.npy")
                if not os.path.isfile(target_path):
                    print(f"⚠️ Missing target.npy: {target_path}")
                    continue

                image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))
                targets = np.load(target_path)

                if len(image_paths) != len(targets):
                    print(f"⚠️ Mismatch in {session_path}: {len(image_paths)} images vs {len(targets)} targets")
                    continue

                for img_path, gaze in zip(image_paths, targets):
                    self.data.append((img_path, gaze))

        if max_samples is not None and max_samples > 0:
            np.random.seed(0)
            np.random.shuffle(self.data)
            self.data = self.data[:max_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, gaze = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(gaze, dtype=torch.float32)