import argparse
import os
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import pandas as pd

from config import device, data_path, all_subjects
from dataset import ARGazeDataset
from transforms import test_tf, train_tf
from losses import gaze_loss, angular_error
from backbone.repnext import create_repnext


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_samples', type=int, default=2000, help='Number of samples per training subject')
    parser.add_argument('--test_samples', type=int, default=1000, help='Number of samples per test subject')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='./ARGaze_logs', help='Directory to save logs and models')
    parser.add_argument('--ckpt_dir', type=str, default='./ARGaze_checkpoints',
                        help='Directory for periodic checkpoints')
    parser.add_argument('--model_type', type=str, default='repnext_m3',
                        choices=['repnext_m0', 'repnext_m1', 'repnext_m2', 'repnext_m3', 'repnext_m4', 'repnext_m5'],
                        help='RepNext model type')
    parser.add_argument('--pretrained_weights', type=str, default='', help='Path to pretrained weights')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for optimizer')
    return parser.parse_args()


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        else:
            replace_batchnorm(child)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    batch_count = 0
    for imgs, gazes in tqdm(loader, leave=False):
        imgs, gazes = imgs.to(device), gazes.to(device)
        preds = model(imgs)
        loss = gaze_loss(preds, gazes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"      Batch {batch_count}: Loss={loss.item():.4f}")
    return total_loss / max(1, batch_count)


def validate(model, loader, device):
    model.eval()
    angles = []
    with torch.no_grad():
        for i, (imgs, gazes) in enumerate(loader):
            if i % 20 == 0:
                print(f"    [Val batch {i + 1}/{len(loader)}]")
            imgs, gazes = imgs.to(device), gazes.to(device)
            preds = model(imgs)
            angles.extend(angular_error(preds, gazes).cpu().numpy())
    return np.mean(angles)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, "training_log.csv")

    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        completed_folds = set(log_df["fold"].unique())
    else:
        log_df = pd.DataFrame(columns=["fold", "epoch", "train_loss", "val_mae"])
        completed_folds = set()

    fold_errors = []
    full_log = log_df.to_dict("records")
    start_time = time.time()

    for fold, test_sub in enumerate(all_subjects):
        if test_sub in completed_folds:
            print(f"Skipping completed fold {test_sub}")
            continue

        print(f"\nLOSO Fold {fold + 1}/25: Testing on {test_sub}")
        train_subs = [s for s in all_subjects if s != test_sub]

        train_ds = ARGazeDataset(data_path, train_subs, transform=train_tf, max_samples=args.train_samples)
        test_ds = ARGazeDataset(data_path, [test_sub], transform=test_tf, max_samples=args.test_samples)
        print(f"   [INFO] Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

        if len(train_ds) == 0 or len(test_ds) == 0:
            print(f"Skipping fold {test_sub} due to empty dataset")
            continue

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=(device == "cuda"))
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=(device == "cuda"))

        # Model initialization
        model = create_repnext(args.model_type, pretrained=False, num_classes=6)
        if args.pretrained_weights:
            state_dict = torch.load(args.pretrained_weights, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        replace_batchnorm(model)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_mae = float('inf')
        best_model_path = os.path.join(args.save_dir, f"model_{test_sub}.pth")

        for epoch in range(args.epochs):
            print(f"\nStarting epoch {epoch + 1}...")
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_mae = validate(model, test_loader, device)
            scheduler.step()

            full_log.append({
                "fold": test_sub,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_mae": val_mae
            })

            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), best_model_path)

            print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val MAE={val_mae:.2f}°")

            # Save periodic checkpoints
            if (epoch + 1) % 5 == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"{test_sub}_epoch{epoch + 1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_mae
                }, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

        print(f"Best MAE for {test_sub}: {best_mae:.2f}°")
        fold_errors.append(best_mae)
        pd.DataFrame(full_log).to_csv(log_file, index=False)

        elapsed = time.time() - start_time
        avg_time = elapsed / (fold + 1 - len(completed_folds))
        remaining = avg_time * (25 - fold - 1)
        print(f"ETA: {remaining / 60:.1f} min")

    print(f"\nFinal LOSO MAE: {np.mean(fold_errors):.2f}°")
    print(f"Logs saved to: {log_file}")


if __name__ == '__main__':
    main()
