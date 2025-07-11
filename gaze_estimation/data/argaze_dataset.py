import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class ARGazeDataset(Dataset):
    """
    ARGaze Dataset for gaze estimation.
    
    Args:
        root_dir (str): Root directory of the dataset
        subject_ids (list): List of subject IDs to include in the dataset
        transform: Transformations to apply to the images
        camera (str): Camera view to use (e.g., 'C1')
        max_samples (int, optional): Maximum number of samples to load
    """
    def __init__(self, root_dir, subject_ids, transform=None, camera="C1", max_samples=None):
        self.data = []
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        self.camera = camera

        for sid in subject_ids:
            subj_path = os.path.join(root_dir, sid)
            if not os.path.isdir(subj_path):
                continue

            sessions = [d for d in os.listdir(subj_path) if d.startswith(f"{sid}_S")]
            for session in sessions:
                session_path = os.path.join(subj_path, session)
                image_folder = os.path.join(session_path, f"{session}_{camera}")
                if not os.path.isdir(image_folder):
                    continue

                target_path = os.path.join(session_path, "target.npy")
                if not os.path.isfile(target_path):
                    continue

                image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))
                targets = np.load(target_path)

                if len(image_paths) != len(targets):
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
            
        return image, gaze.astype(np.float32)

def get_train_transforms():
    """Get data augmentation transforms for training."""
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

def get_test_transforms():
    """Get transforms for validation/testing."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
