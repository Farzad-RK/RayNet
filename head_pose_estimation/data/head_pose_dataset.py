import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class HeadPoseDataset(Dataset):
    """
    Dataset class for head pose estimation.
    Supports both BIWI and 300W_LP datasets.
    """
    def __init__(self, data_dir, transform=None, dataset_type='300W_LP'):
        self.data_dir = data_dir
        self.transform = transform or self.get_default_transform()
        self.dataset_type = dataset_type
        self.data = []
        
        self.load_data()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, pose = self.data[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Convert pose to tensor
        pose = torch.FloatTensor(pose)
        
        return img, pose
    
    def load_data(self):
        """Load dataset based on the specified type."""
        if self.dataset_type == '300W_LP':
            self._load_300w_lp()
        elif self.dataset_type == 'BIWI':
            self._load_biwi()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
    
    def _load_300w_lp(self):
        """Load 300W-LP dataset."""
        # Implementation for 300W-LP dataset
        # This is a placeholder - you'll need to implement the actual loading logic
        pass
    
    def _load_biwi(self):
        """Load BIWI dataset."""
        # Implementation for BIWI dataset
        # This is a placeholder - you'll need to implement the actual loading logic
        pass
    
    @staticmethod
    def get_default_transform():
        """Get default transform for head pose estimation."""
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_augmentation_transform():
        """Get data augmentation transform for training."""
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop((224, 224)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
