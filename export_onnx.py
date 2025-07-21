"""
ONNX Export Script for Head Pose Estimation Models

This script allows exporting both Gaze Estimation and SixDRepNet models to ONNX format.
It handles model loading, input/output configuration, and ONNX export with proper metadata.
"""

import argparse
import torch
import torch.nn as nn
from torch.onnx import export
import onnx
import onnxruntime as ort
import numpy as np
from typing import Tuple, Dict, Optional
from sixdrepnet.model import SixDRepNet, SixDRepNet_RepNeXt
from  backbone.repnext import create_repnext
from sixdrepnet.utils import compute_rotation_matrix_from_ortho6d,compute_euler_angles_from_rotation_matrices


class ModelExporter:
    """Base class for model exporters."""
    
    @staticmethod
    def export_onnx(
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...],
        input_names: list = ["input"],
        output_names: list = ["output"],
        dynamic_axes: Optional[Dict] = None,
        opset_version: int = 12,
        **kwargs
    ) -> str:
        """
        Export a PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            output_path: Path to save the ONNX model
            input_shape: Input tensor shape (C, H, W)
            input_names: List of input names
            output_names: List of output names
            dynamic_axes: Dictionary specifying dynamic axes
            opset_version: ONNX opset version
            **kwargs: Additional arguments for torch.onnx.export
            
        Returns:
            Path to the exported ONNX model
        """
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape, device=next(model.parameters()).device)
        
        # Default dynamic axes if not provided
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export the model
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            **kwargs
        )
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        return output_path


class SixDRepNetExporter(ModelExporter):
    """Exporter for SixDRepNet models."""
    
    @staticmethod
    def load_model(weights_path: str, backbone: str = "repnext_m4") -> nn.Module:
        """
        Load SixDRepNet model from weights.
        
        Args:
            weights_path: Path to the model weights
            backbone: Backbone architecture. Can be:
                    - 'repnext_m0' to 'repnext_m5' for RepNeXt variants (recommended, default: 'repnext_m4')
                    - 'repvgg_a0' for RepVGG-A0
        """
        if "repnext" in backbone.lower():
            # Validate RepNeXt version
            valid_versions = [f'repnext_m{i}' for i in range(6)]  # m0 to m5
            if backbone not in valid_versions:
                raise ValueError(f"Invalid RepNeXt version: {backbone}. Must be one of {valid_versions}")
                
            model = SixDRepNet_RepNeXt(backbone_fn=create_repnext(backbone), pretrained=False)
        else:
            if backbone != "repvgg_a0":
                print(f"Warning: Only 'repvgg_a0' is supported for RepVGG. Using 'repvgg_a0' instead of '{backbone}'")
            model = SixDRepNet("repvgg_a0", "", deploy=False)  # deploy=False for training state
            
        # Load weights
        checkpoint = torch.load(weights_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    @staticmethod
    def export(
        weights_path: str,
        output_path: str,
        backbone: str = "repnext_m4",
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        **kwargs
    ) -> str:
        """Export SixDRepNet model to ONNX."""
        model = SixDRepNetExporter.load_model(weights_path, backbone)
        
        # Define dynamic axes for batch processing
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'rotation_matrix': {0: 'batch_size'},
            'euler_angles': {0: 'batch_size'}
        }
        
        # Create a wrapper model to output both rotation matrix and euler angles
        class WrappedModel(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                # Get 6D representation
                pose_6d = self.model(x)
                # Convert to rotation matrix
                rotation_matrix = compute_rotation_matrix_from_ortho6d(pose_6d)
                # Convert to euler angles (pitch, yaw, roll)
                euler_angles = compute_euler_angles_from_rotation_matrices(rotation_matrix)
                return rotation_matrix, euler_angles
        
        wrapped_model = WrappedModel(model)
        wrapped_model.eval()
        
        # Export the model
        return ModelExporter.export_onnx(
            model=wrapped_model,
            output_path=output_path,
            input_shape=input_shape,
            input_names=["input"],
            output_names=["rotation_matrix", "euler_angles"],
            dynamic_axes=dynamic_axes,
            **kwargs
        )


class GazeEstimationExporter(ModelExporter):
    """Exporter for Gaze Estimation models."""
    
    @staticmethod
    def load_model(weights_path: str, model_type: str = "repnext_m3") -> nn.Module:
        model = create_repnext(model_type,pretrained=False)
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location='cpu',weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    @staticmethod
    def export(
        weights_path: str,
        output_path: str,
        model_type: str = "repnext_m3",
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        **kwargs
    ) -> str:
        """Export Gaze Estimation model to ONNX."""
        model = GazeEstimationExporter.load_model(weights_path, model_type)
        
        # Define dynamic axes for batch processing
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'gaze_vector': {0: 'batch_size'},
            'gaze_angles': {0: 'batch_size'}
        }
        
        # Create a wrapper model to output both gaze vector and angles
        class WrappedModel(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                # Get gaze vector
                gaze_vector = self.model(x)
                # Convert to angles (pitch, yaw)
                pitch = torch.asin(-gaze_vector[:, 1])
                yaw = torch.atan2(gaze_vector[:, 0], gaze_vector[:, 2])
                gaze_angles = torch.stack([pitch, yaw], dim=1)
                return gaze_vector, gaze_angles
        
        wrapped_model = WrappedModel(model)
        wrapped_model.eval()
        
        # Export the model
        return ModelExporter.export_onnx(
            model=wrapped_model,
            output_path=output_path,
            input_shape=input_shape,
            input_names=["input"],
            output_names=["gaze_vector", "gaze_angles"],
            dynamic_axes=dynamic_axes,
            **kwargs
        )



def parse_args():
    parser = argparse.ArgumentParser(description='Export models to ONNX format')
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--sixdrepnet', action='store_true', help='Export SixDRepNet model')
    model_group.add_argument('--gaze-estimation', action='store_true', help='Export Gaze Estimation model')
    
    # Model parameters
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--output', type=str, required=True, help='Output ONNX file path')
    parser.add_argument('--backbone', type=str, default='repnext_m4', 
                       help='''Backbone architecture:
                       - For SixDRepNet: 'repnext_m0' to 'repnext_m5' (default: 'repnext_m4') or 'repvgg_a0'
                       - For Gaze Estimation: 'repnext_m0' to 'repnext_m5' (default: 'repnext_m3')
                       ''')
    parser.add_argument('--input-shape', type=int, nargs=3, default=[3, 224, 224],
                       metavar=('CHANNELS', 'HEIGHT', 'WIDTH'),
                       help='Input shape (default: 3 224 224)')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version (default: 12)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        if args.sixdrepnet:
            print(f"Exporting SixDRepNet model with backbone: {args.backbone}")
            print(f"Note: Using {args.backbone} as backbone. Available options: 'repnext_m0' to 'repnext_m5' or 'repvgg_a0'")
            output_path = SixDRepNetExporter.export(
                weights_path=args.weights,
                output_path=args.output,
                backbone=args.backbone.lower(),  # Ensure lowercase
                input_shape=tuple(args.input_shape),
                opset_version=args.opset
            )
            print(f"SixDRepNet model exported to: {output_path}")
            
        elif args.gaze_estimation:
            print(f"Exporting Gaze Estimation model with backbone: {args.backbone}")
            output_path = GazeEstimationExporter.export(
                weights_path=args.weights,
                output_path=args.output,
                model_type=args.backbone,
                input_shape=tuple(args.input_shape),
                opset_version=args.opset
            )
            print(f"Gaze Estimation model exported to: {output_path}")
            
        print("ONNX export completed successfully!")
        
    except Exception as e:
        print(f"Error during ONNX export: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
