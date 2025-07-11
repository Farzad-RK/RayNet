import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import device, data_path, all_subjects  # import other config as needed
from dataset import ARGazeDataset
from transforms import test_tf
from losses import angular_error
from backbone.repnext import repnext_m3   # Or import your model appropriately

def load_model(checkpoint_path, device):
    model = repnext_m3(pretrained=False, num_classes=6)
    state_dict = torch.load(checkpoint_path, map_location=device)
    # If checkpoint is dict with 'model_state_dict', get the inner dict
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

def evaluate(model, dataloader, device):
    all_angles = []
    with torch.no_grad():
        for imgs, gazes in tqdm(dataloader, desc="Testing"):
            imgs, gazes = imgs.to(device), gazes.to(device)
            preds = model(imgs)
            angles = angular_error(preds, gazes)
            all_angles.extend(angles.cpu().numpy())
    return np.mean(all_angles), np.std(all_angles), all_angles

def main(args):
    # --- Select subjects to test ---
    if args.subjects == "all":
        test_subjects = all_subjects
    else:
        test_subjects = [s.strip() for s in args.subjects.split(",")]

    results = []
    for subject in test_subjects:
        # Path to checkpoint, adjust as needed!
        if args.checkpoint_dir:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_{subject}.pth")
        else:
            checkpoint_path = args.checkpoint_path

        assert os.path.exists(checkpoint_path), f"Checkpoint {checkpoint_path} not found"

        print(f"\n🧑‍🦱 Testing on subject: {subject}")
        print(f"    Loading model weights from: {checkpoint_path}")
        model = load_model(checkpoint_path, device)

        # Build test set for the subject
        test_ds = ARGazeDataset(
            root_dir=data_path,
            subject_ids=[subject],
            transform=test_tf,
            camera=args.camera,
            max_samples=args.max_samples
        )
        print(f"    Test samples: {len(test_ds)}")
        if len(test_ds) == 0:
            print(f"    ⚠️ No test data for subject {subject}, skipping.")
            continue

        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=(device=="cuda")
        )

        # Evaluate
        mae, std, all_angles = evaluate(model, test_loader, device)
        print(f"    MAE: {mae:.2f}°, Std: {std:.2f}°")

        results.append({"subject": subject, "mae": mae, "std": std})

        if args.save_predictions:
            np.save(os.path.join(args.save_predictions, f"{subject}_angles.npy"), np.array(all_angles))

    # Summary
    df = pd.DataFrame(results)
    print("\n========= Summary =========")
    print(df)
    if args.save_results:
        df.to_csv(args.save_results, index=False)
        print(f"Results saved to {args.save_results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ARGaze Model")
    parser.add_argument("--subjects", type=str, default="all",
                        help="Subjects to test (comma-separated or 'all')")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to a specific checkpoint to load. If testing multiple, use --checkpoint_dir.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing checkpoints (model_{subject}.pth)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--camera", type=str, default="C1")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max test samples (debug)")
    parser.add_argument("--save_results", type=str, default=None,
                        help="CSV file to save summary results")
    parser.add_argument("--save_predictions", type=str, default=None,
                        help="Directory to save per-subject angle predictions (as .npy)")
    args = parser.parse_args()
    main(args)
