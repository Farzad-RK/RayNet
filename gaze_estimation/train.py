import torch
import numpy as np
from tqdm import tqdm
from losses import gaze_loss, angular_error

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, gazes in tqdm(loader, leave=False):
        imgs, gazes = imgs.to(device), gazes.to(device)
        preds = model(imgs)
        loss = gaze_loss(preds, gazes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss

def validate(model, loader, device):
    model.eval()
    angles = []
    with torch.no_grad():
        for imgs, gazes in loader:
            imgs, gazes = imgs.to(device), gazes.to(device)
            preds = model(imgs)
            angles.extend(angular_error(preds, gazes).cpu().numpy())
    return np.mean(angles)
