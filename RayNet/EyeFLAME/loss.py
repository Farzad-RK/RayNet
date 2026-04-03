# loss.py

import torch
import torch.nn as nn



class WeakPerspectiveLoss(nn.Module):
    """
     Weak Perspective and 2D/3D Keypoint Losses for Eye and Gaze Estimation.
    """

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, ground_truth, subject_params=None, camera_params=None):
        losses = {}
        device = predictions['eyeball_centers'].device

        # Debug shapes
        print("\n=== Loss Computation Debug ===")

        # === 2D SUPERVISION (PRIMARY) ===
        if 'projections_2d' in predictions and predictions['projections_2d'] is not None:

            # Iris 2D loss
            if 'iris_mesh_2D' in ground_truth:
                pred_iris_2d = predictions['projections_2d']['iris_landmarks_2d']
                gt_iris_2d = ground_truth['iris_mesh_2D']

                print(f"  Iris 2D - Pred: {pred_iris_2d.shape}, GT: {gt_iris_2d.shape}")

                # Ensure shapes match
                if pred_iris_2d.shape != gt_iris_2d.shape:
                    print(f"  WARNING: Shape mismatch in iris 2D!")
                    # Try to fix common issues
                    if gt_iris_2d.dim() == 4 and gt_iris_2d.shape[1] == 2:
                        gt_iris_2d = gt_iris_2d.reshape(gt_iris_2d.shape[0], -1, 2)
                        print(f"  Fixed GT shape to: {gt_iris_2d.shape}")

                if pred_iris_2d.shape == gt_iris_2d.shape:
                    losses['iris_2d'] = self.l1_loss(pred_iris_2d, gt_iris_2d)
                else:
                    losses['iris_2d'] = torch.tensor(0.0, device=device)

            # Eyeball 2D loss
            if 'eyeball_center_2D' in ground_truth:
                pred_eyeball_2d = predictions['projections_2d']['eyeball_centers_2d']
                gt_eyeball_2d = ground_truth['eyeball_center_2D']

                print(f"  Eyeball 2D - Pred: {pred_eyeball_2d.shape}, GT: {gt_eyeball_2d.shape}")

                if pred_eyeball_2d.shape == gt_eyeball_2d.shape:
                    losses['eyeball_2d'] = self.l1_loss(pred_eyeball_2d, gt_eyeball_2d)
                else:
                    losses['eyeball_2d'] = torch.tensor(0.0, device=device)

            # Pupil 2D loss
            if 'pupil_center_2D' in ground_truth:
                pred_pupil_2d = predictions['projections_2d']['pupil_centers_2d']
                gt_pupil_2d = ground_truth['pupil_center_2D']

                print(f"  Pupil 2D - Pred: {pred_pupil_2d.shape}, GT: {gt_pupil_2d.shape}")

                if pred_pupil_2d.shape == gt_pupil_2d.shape:
                    losses['pupil_2d'] = self.l1_loss(pred_pupil_2d, gt_pupil_2d)
                else:
                    losses['pupil_2d'] = torch.tensor(0.0, device=device)

        # === 3D SUPERVISION ===

        # Eyeball 3D loss
        pred_eyeball_3d = predictions['eyeball_centers']
        gt_eyeball_3d = ground_truth['eyeball_center_3D']
        print(f"  Eyeball 3D - Pred: {pred_eyeball_3d.shape}, GT: {gt_eyeball_3d.shape}")

        if pred_eyeball_3d.shape == gt_eyeball_3d.shape:
            losses['eyeball_3d'] = self.l1_loss(pred_eyeball_3d, gt_eyeball_3d)
        else:
            losses['eyeball_3d'] = torch.tensor(0.0, device=device)

        # Pupil 3D loss
        pred_pupil_3d = predictions['pupil_centers']
        gt_pupil_3d = ground_truth['pupil_center_3D']
        print(f"  Pupil 3D - Pred: {pred_pupil_3d.shape}, GT: {gt_pupil_3d.shape}")

        if pred_pupil_3d.shape == gt_pupil_3d.shape:
            losses['pupil_3d'] = self.l1_loss(pred_pupil_3d, gt_pupil_3d)
        else:
            losses['pupil_3d'] = torch.tensor(0.0, device=device)

        # Iris 3D loss
        pred_iris_3d = predictions['iris_landmarks_100']
        gt_iris_3d = ground_truth['iris_mesh_3D']
        print(f"  Iris 3D - Pred: {pred_iris_3d.shape}, GT: {gt_iris_3d.shape}")

        if pred_iris_3d.shape == gt_iris_3d.shape:
            losses['iris_3d'] = self.l1_loss(pred_iris_3d, gt_iris_3d)
        else:
            losses['iris_3d'] = torch.tensor(0.0, device=device)

        # === ANGULAR LOSSES ===

        # Gaze direction
        if 'gaze_C' in ground_truth:
            pred_gaze = predictions['head_gaze_direction']
            gt_gaze = ground_truth['gaze_C']
            print(f"  Gaze - Pred: {pred_gaze.shape}, GT: {gt_gaze.shape}")

            if pred_gaze.shape == gt_gaze.shape:
                losses['gaze_direction'] = self.l1_loss(pred_gaze, gt_gaze)
            else:
                losses['gaze_direction'] = torch.tensor(0.0, device=device)

        # Optical axes
        if 'optic_axis_L' in ground_truth and 'optic_axis_R' in ground_truth:
            pred_optical = predictions['optical_axes']  # [B, 2, 3]
            gt_optical_L = ground_truth['optic_axis_L']  # [B, 3]
            gt_optical_R = ground_truth['optic_axis_R']  # [B, 3]

            print(f"  Optical - Pred: {pred_optical.shape}, GT_L: {gt_optical_L.shape}, GT_R: {gt_optical_R.shape}")

            # Compute cosine similarity for each eye
            if pred_optical.shape[0] == gt_optical_L.shape[0]:
                cos_sim_L = torch.sum(pred_optical[:, 0] * gt_optical_L, dim=-1)
                cos_sim_R = torch.sum(pred_optical[:, 1] * gt_optical_R, dim=-1)
                losses['optical_axes'] = 2.0 - (torch.mean(cos_sim_L) + torch.mean(cos_sim_R))
            else:
                losses['optical_axes'] = torch.tensor(0.0, device=device)

        # === REGULARIZATION ===
        if 'weak_perspective' in predictions:
            wp = predictions['weak_perspective']
            losses['scale_reg'] = torch.mean((wp['scale'] - 1.0) ** 2)
            losses['trans_reg'] = torch.mean(wp['translation_2d'] ** 2)
            losses['norm_centers_reg'] = torch.mean(torch.abs(wp['normalized_centers']))

        # Add default values for any missing losses
        default_loss_keys = ['iris_2d', 'eyeball_2d', 'pupil_2d', 'eyeball_3d',
                             'pupil_3d', 'iris_3d', 'gaze_direction', 'optical_axes']
        for key in default_loss_keys:
            if key not in losses:
                losses[key] = torch.tensor(0.0, device=device)

        total_loss = sum(losses.values())

        print(f"  Total loss: {total_loss.item():.4f}")
        print("=== End Loss Debug ===\n")

        return total_loss, losses