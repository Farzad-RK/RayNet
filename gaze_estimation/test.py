import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gaze_estimation.models.gaze_estimator import get_gaze_estimator
from gaze_estimation.data.argaze_dataset import ARGazeDataset, get_test_transforms
from gaze_estimation.utils.losses import angular_error

def parse_args():
    parser = argparse.ArgumentParser(description='Test Gaze Estimation Model')
    parser.add_argument('--data_root', type=str, required=True, help='Path to ARGaze dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--subject_id', type=str, default=None, help='Specific subject ID to test on')
    parser.add_argument('--output_file', type=str, default='test_results.csv', help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help='Device to use')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize device
    device = torch.device(args.device)
    
    # Load model from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model = get_gaze_estimator(pretrained_weights=args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # Determine subjects to test on
    if args.subject_id:
        test_subjects = [args.subject_id]
    else:
        test_subjects = [f'P{i}' for i in range(1, 26)]  # Test on all subjects by default
    
    results = []
    
    for subject_id in test_subjects:
        # Create test dataset and dataloader
        test_dataset = ARGazeDataset(
            root_dir=args.data_root,
            subject_ids=[subject_id],
            transform=get_test_transforms()
        )
        
        if len(test_dataset) == 0:
            print(f"No data found for subject {subject_id}, skipping...")
            continue
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # Run inference
        errors = []
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc=f'Testing {subject_id}', leave=False):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                batch_errors = angular_error(outputs, targets)
                errors.extend(batch_errors.cpu().numpy())
        
        # Calculate metrics
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        acc_5 = np.mean(np.array(errors) <= 5.0) * 100
        acc_10 = np.mean(np.array(errors) <= 10.0) * 100
        
        print(f'Subject {subject_id}:')
        print(f'  Mean Error: {mean_error:.2f}°')
        print(f'  Median Error: {median_error:.2f}°')
        print(f'  Std Dev: {std_error:.2f}°')
        print(f'  Accuracy @5°: {acc_5:.2f}%')
        print(f'  Accuracy @10°: {acc_10:.2f}%')
        
        results.append({
            'subject_id': subject_id,
            'num_samples': len(test_dataset),
            'mean_error': mean_error,
            'median_error': median_error,
            'std_error': std_error,
            'acc_5': acc_5,
            'acc_10': acc_10,
        })
    
    # Save results
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output_file, index=False)
        print(f'\nResults saved to {args.output_file}')
        
        # Print overall statistics if testing multiple subjects
        if len(results) > 1:
            print('\nOverall Statistics:')
            print(f'  Mean Error: {df["mean_error"].mean():.2f}°')
            print(f'  Median Error: {df["median_error"].median():.2f}°')
            print(f'  Accuracy @5°: {df["acc_5"].mean():.2f}%')
            print(f'  Accuracy @10°: {df["acc_10"].mean():.2f}%')

if __name__ == '__main__':
    main()
