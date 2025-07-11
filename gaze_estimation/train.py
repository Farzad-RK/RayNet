import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gaze_estimation.models.gaze_estimator import get_gaze_estimator
from gaze_estimation.data.argaze_dataset import ARGazeDataset, get_train_transforms, get_test_transforms
from gaze_estimation.utils.losses import gaze_loss, angular_error
from shared.backbone.repnext import RepNeXt

def parse_args():
    parser = argparse.ArgumentParser(description='Train Gaze Estimation Model')
    parser.add_argument('--data_root', type=str, required=True, help='Path to ARGaze dataset')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained_backbone', type=str, default=None, help='Path to pre-trained backbone weights')
    parser.add_argument('--few_shot', type=int, default=None, help='Number of samples per subject for few-shot training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for images, targets in tqdm(dataloader, desc='Training', leave=False):
        images, targets = images.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = gaze_loss(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_error = 0.0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Evaluating', leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            errors = angular_error(outputs, targets)
            total_error += errors.sum().item()
    
    return total_error / len(dataloader.dataset)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize device
    device = torch.device(args.device)
    
    # Initialize backbone
    backbone = RepNeXt()
    if args.pretrained_backbone:
        state_dict = torch.load(args.pretrained_backbone, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        backbone.load_state_dict(state_dict, strict=False)
    
    # Initialize model
    model = get_gaze_estimator(backbone=backbone, pretrained_weights=args.resume)
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume training if checkpoint provided
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        start_epoch = checkpoint.get('epoch', 0) + 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Prepare datasets and dataloaders
    all_subjects = [f'P{i}' for i in range(1, 26)]
    train_subjects = all_subjects[:-5]  # Use first 20 subjects for training
    val_subjects = all_subjects[-5:]    # Use last 5 subjects for validation
    
    train_dataset = ARGazeDataset(
        root_dir=args.data_root,
        subject_ids=train_subjects,
        transform=get_train_transforms(),
        max_samples=args.few_shot
    )
    
    val_dataset = ARGazeDataset(
        root_dir=args.data_root,
        subject_ids=val_subjects,
        transform=get_test_transforms(),
        max_samples=args.few_shot
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Training loop
    best_val_error = float('inf')
    log_file = os.path.join(args.output_dir, 'training_log.csv')
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate on validation set
        val_error = evaluate(model, val_loader, device)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Save checkpoint
        is_best = val_error < best_val_error
        if is_best:
            best_val_error = val_error
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_error': val_error,
            'best_val_error': best_val_error,
        }
        
        torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
        
        # Log progress
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Error: {val_error:.2f}°, Best Val Error: {best_val_error:.2f}°')
        
        with open(log_file, 'a') as f:
            if epoch == 0:
                f.write('epoch,train_loss,val_error\n')
            f.write(f'{epoch},{train_loss},{val_error}\n')

if __name__ == '__main__':
    main()
