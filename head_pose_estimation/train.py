import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from head_pose_estimation.models.head_pose_estimator import get_head_pose_estimator
from head_pose_estimation.data.head_pose_dataset import HeadPoseDataset
from head_pose_estimation.utils.losses import geodesic_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train Head Pose Estimation Model')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--backbone', type=str, default='repnext_m4', help='Backbone architecture')
    parser.add_argument('--dataset', type=str, default='300W_LP', choices=['300W_LP', 'BIWI'], help='Dataset to use')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for images, poses in tqdm(dataloader, desc='Training', leave=False):
        images, poses = images.to(device), poses.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, poses)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, poses in tqdm(dataloader, desc='Validating', leave=False):
            images, poses = images.to(device), poses.to(device)
            outputs = model(images)
            loss = criterion(outputs, poses)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize device
    device = torch.device(args.device)
    
    # Initialize model
    model = get_head_pose_estimator(backbone_name=args.backbone)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = geodesic_loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Prepare datasets and dataloaders
    train_dataset = HeadPoseDataset(
        data_dir=os.path.join(args.data_root, 'train'),
        transform=HeadPoseDataset.get_augmentation_transform(),
        dataset_type=args.dataset
    )
    
    val_dataset = HeadPoseDataset(
        data_dir=os.path.join(args.data_root, 'val'),
        transform=HeadPoseDataset.get_default_transform(),
        dataset_type=args.dataset
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
    best_val_loss = float('inf')
    log_file = os.path.join(args.output_dir, 'training_log.csv')
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))
        
        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
        
        # Log progress
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}')
        
        # Save to log file
        with open(log_file, 'a') as f:
            if epoch == 0:
                f.write('epoch,train_loss,val_loss\n')
            f.write(f'{epoch},{train_loss},{val_loss}\n')

if __name__ == '__main__':
    main()
