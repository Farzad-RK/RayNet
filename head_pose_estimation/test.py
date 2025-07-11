import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from head_pose_estimation.models.head_pose_estimator import get_head_pose_estimator
from head_pose_estimation.data.head_pose_dataset import HeadPoseDataset
from head_pose_estimation.utils.metrics import calculate_mae, calculate_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Test Head Pose Estimation Model')
    parser.add_argument('--data_root', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='repnext_m4', help='Backbone architecture')
    parser.add_argument('--dataset', type=str, default='300W_LP', choices=['300W_LP', 'BIWI'], help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--output_file', type=str, default='test_results.csv', help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize device
    device = torch.device(args.device)
    
    # Initialize model
    print(f"Loading model from {args.checkpoint}")
    model = get_head_pose_estimator(backbone_name=args.backbone, pretrained_weights=args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # Prepare dataset and dataloader
    test_dataset = HeadPoseDataset(
        data_dir=args.data_root,
        transform=HeadPoseDataset.get_default_transform(),
        dataset_type=args.dataset
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Run evaluation
    print("Running evaluation...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, poses in tqdm(test_loader, desc='Testing'):
            images, poses = images.to(device), poses.to(device)
            outputs = model(images)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(poses.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    mae = calculate_mae(all_preds, all_targets)
    acc_5 = calculate_accuracy(all_preds, all_targets, threshold=5.0)
    acc_10 = calculate_accuracy(all_preds, all_targets, threshold=10.0)
    
    # Print results
    print("\nTest Results:")
    print(f"  MAE: {mae:.2f}°")
    print(f"  Accuracy @5°: {acc_5:.2f}%")
    print(f"  Accuracy @10°: {acc_10:.2f}%")
    
    # Save results to file
    results = {
        'dataset': [args.dataset],
        'backbone': [args.backbone],
        'mae': [mae],
        'acc_5': [acc_5],
        'acc_10': [acc_10],
        'num_samples': [len(test_dataset)]
    }
    
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    print(f"\nResults saved to {args.output_file}")

if __name__ == '__main__':
    main()
