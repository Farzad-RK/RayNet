"""
ONNX Export Script for Head Pose Estimation Models

Exports the SixDRepNet (RepNeXt/RepVGG) head-pose model — and, separately, the
gaze model — to ONNX. Run from the repo root, e.g.:

    python export_onnx.py --sixdrepnet \
        --weights sixdrepnet/pretrained_models/repnext_m4/myexp_epoch_80.tar \
        --backbone repnext_m4 \
        --output head_pose_repnext_m4.onnx

SixDRepNet specifics (the path we ship to mobile):
  * Built in DEPLOY mode (reparameterized/fused convs) by default — matching how
    our checkpoints are saved and what runs efficiently on device. `--no-deploy`
    exports the unfused training graph instead.
  * Loading is non-strict; the only tolerated missing keys are the unused
    RepNeXt classifier head ('backbone.head.*'). Any other missing key is fatal.
  * Outputs: `rotation_matrix` (N,3,3) and `euler_angles` (N,3, radians). NOTE:
    the model's forward already returns the rotation matrix, so the export
    wrapper must NOT re-apply the 6D->matrix conversion.
  * Uses the legacy TorchScript exporter so `--opset` (default 12) is honored
    exactly; the Torch 2.x dynamo exporter emits opset 18 and fails to
    down-convert. Deprecated in Torch >= 2.9 but still functional.
  * After export, a PyTorch<->ONNX parity check runs automatically and FAILS the
    export if max|torch-onnx| > --parity-atol (default 1e-4). `--no-verify` skips.
  * `--mobile` emits a device-ready artifact for ONNX Runtime Mobile: static
    batch=1 (no dynamic axes), rotation-matrix-ONLY output (drops the euler
    ScatterND/Atan path), onnxsim folding (~1878->803 nodes), and a `.ort`
    flatbuffer + required-operators config. Parity is checked before conversion.

Requires: onnx, onnxruntime, onnxscript (+ onnxsim for --mobile).
"""

import argparse
import os
import subprocess
import sys

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from typing import Tuple, Dict, Optional

# model.py / utils.py / backbone.* all use top-level (non-packaged) imports
# (e.g. `import utils`, `from backbone.repnext import ...`), exactly as the demo
# does, so the sixdrepnet/ directory must be on sys.path before importing them.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'sixdrepnet'))

from model import SixDRepNet, SixDRepNet_RepNeXt
from backbone.repnext import create_repnext
from utils import (compute_rotation_matrix_from_ortho6d,
                   compute_euler_angles_from_rotation_matrices)


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
        
        # Export the model. Use the legacy TorchScript exporter (dynamo=False):
        # unlike the dynamo path on Torch 2.x — which emits opset 18 and silently
        # FAILS to down-convert to a lower requested opset (leaving 18) — the
        # TorchScript exporter honors `opset_version` exactly and pairs correctly
        # with `dynamic_axes`. Exact, predictable opset matters for the mobile
        # runtimes/converters downstream.
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            dynamo=False,
            **kwargs
        )

        # Verify structural validity and report the ACTUAL opset (so a silent
        # mismatch with what was requested can never slip through to mobile).
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        actual_opset = onnx_model.opset_import[0].version
        if actual_opset != opset_version:
            print(f"WARNING: requested opset {opset_version} but the exported "
                  f"model is opset {actual_opset}.")
        else:
            print(f"Exported at opset {actual_opset} (as requested).")

        return output_path


class SixDRepNetExporter(ModelExporter):
    """Exporter for SixDRepNet models."""
    
    @staticmethod
    def load_model(weights_path: str, backbone: str = "repnext_m4",
                   deploy: bool = True) -> nn.Module:
        """
        Load a SixDRepNet model from weights, matching how `demo.py` builds it.

        Args:
            weights_path: Path to the model weights (.tar/.pth).
            backbone: Backbone architecture. Can be:
                    - 'repnext_m0'..'repnext_m5' (recommended, default 'repnext_m4')
                    - 'repvgg_a0' for RepVGG-A0
            deploy: build the REPARAMETERIZED (fused-conv) structure. Our training
                    checkpoints are saved in deploy mode, and the fused graph is
                    what we want on-device (fewer ops, faster), so this defaults
                    to True. It MUST match the structure the checkpoint was saved
                    in or the weights will not map.
        """
        if "repnext" in backbone.lower():
            valid_versions = [f'repnext_m{i}' for i in range(6)]  # m0..m5
            if backbone not in valid_versions:
                raise ValueError(f"Invalid RepNeXt version: {backbone}. "
                                 f"Must be one of {valid_versions}")
            # Pass the backbone NAME (a str) + deploy, exactly like demo.py — the
            # constructor then builds the deploy backbone and fuses BatchNorm.
            # (The old code passed a non-deploy backbone instance, producing the
            # wrong structure for our fused checkpoints.)
            model = SixDRepNet_RepNeXt(backbone_fn=backbone, pretrained=False,
                                       deploy=deploy)
        else:
            if backbone != "repvgg_a0":
                print(f"Warning: only 'repvgg_a0' is supported for RepVGG; "
                      f"using 'repvgg_a0' instead of '{backbone}'.")
            model = SixDRepNet("repvgg_a0", "", deploy=deploy, pretrained=False)

        # Checkpoints store the full model state under 'model_state_dict' (demo
        # path) or 'state_dict', or may be a bare state dict.
        checkpoint = torch.load(weights_path, map_location='cpu',
                                weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Non-strict: the only expected MISSING keys are the unused RepNeXt
        # classifier head ('backbone.head.*'); the head-pose path uses
        # forward_features + our own linear_reg. Anything else missing means the
        # pose weights did not load -> hard fail rather than export garbage.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        critical = [k for k in missing if 'head' not in k]
        if critical:
            raise RuntimeError(
                "Head-pose weights failed to load (missing non-head keys: "
                f"{critical}). Check --backbone / --deploy match the checkpoint.")
        if unexpected:
            print(f"WARNING: unexpected keys ignored: {unexpected}")
        print(f"Loaded SixDRepNet weights (deploy={deploy}); "
              f"benign missing head keys: {missing}")
        model.eval()
        return model
    
    @staticmethod
    def export(
        weights_path: str,
        output_path: str,
        backbone: str = "repnext_m4",
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        deploy: bool = True,
        verify: bool = True,
        parity_atol: float = 1e-4,
        **kwargs
    ) -> str:
        """Export SixDRepNet model to ONNX (rotation matrix + Euler angles)."""
        model = SixDRepNetExporter.load_model(weights_path, backbone, deploy=deploy)

        # Define dynamic axes for batch processing
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'rotation_matrix': {0: 'batch_size'},
            'euler_angles': {0: 'batch_size'}
        }

        # Wrapper exposing both the rotation matrix and Euler angles.
        # NOTE: SixDRepNet(_RepNeXt).forward ALREADY returns the 3x3 rotation
        # matrix (it applies compute_rotation_matrix_from_ortho6d internally), so
        # we must NOT apply it again here — the old code fed a (B,3,3) matrix
        # into a function expecting a (B,6) vector, producing a garbage export.
        class WrappedModel(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                rotation_matrix = self.model(x)  # (B, 3, 3)
                euler_angles = compute_euler_angles_from_rotation_matrices(
                    rotation_matrix)  # (B, 3) = pitch, yaw, roll (radians)
                return rotation_matrix, euler_angles

        wrapped_model = WrappedModel(model)
        wrapped_model.eval()

        output_names = ["rotation_matrix", "euler_angles"]
        ModelExporter.export_onnx(
            model=wrapped_model,
            output_path=output_path,
            input_shape=input_shape,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            **kwargs
        )

        if verify:
            verify_parity(wrapped_model, output_path, input_shape,
                          output_names=output_names, atol=parity_atol)
        return output_path

    @staticmethod
    def export_mobile(
        weights_path: str,
        output_path: str,
        backbone: str = "repnext_m4",
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        opset_version: int = 12,
        parity_atol: float = 1e-4,
        simplify: bool = True,
    ) -> Tuple[str, str]:
        """
        Export a mobile-optimized SixDRepNet for ONNX Runtime Mobile.

        Differences from the desktop export, all to shrink the on-device graph:
          * deploy/fused backbone (as always);
          * STATIC batch=1 input, NO dynamic axes -> the shape machinery
            (Shape/Slice/Gather/Range from the dynamic batch) constant-folds away;
          * `rotation_matrix` ONLY -> drops the `euler_angles` output and with it
            the mobile-awkward ScatterND/Atan ops. The device pipeline never needs
            Euler: it smooths in quaternion space and warps with the matrix R;
          * onnxsim constant-folding;
          * conversion to the `.ort` flatbuffer with mobile-safe ('Fixed')
            optimizations (also emits a *.required_operators.config for building a
            reduced-operator ORT if you want the smallest binary).

        Returns (onnx_path, ort_path).
        """
        model = SixDRepNetExporter.load_model(weights_path, backbone, deploy=True)

        class RotationOnly(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x)  # (B, 3, 3) rotation matrix only

        wrapped = RotationOnly(model).eval()

        # Static shape (no dynamic_axes): batch is always 1 on device.
        torch.onnx.export(
            wrapped,
            torch.randn(1, *input_shape),
            output_path,
            input_names=["input"],
            output_names=["rotation_matrix"],
            opset_version=opset_version,
            dynamo=False,
        )
        m = onnx.load(output_path)
        onnx.checker.check_model(m)
        n_raw = len(m.graph.node)
        print(f"Exported static rot-only graph at opset "
              f"{m.opset_import[0].version}: {n_raw} nodes")

        if simplify:
            try:
                from onnxsim import simplify as onnxsim_simplify
                sm, ok = onnxsim_simplify(m)
                if ok:
                    onnx.save(sm, output_path)
                    print(f"onnxsim: {n_raw} -> {len(sm.graph.node)} nodes")
                else:
                    print("WARNING: onnxsim validation failed; keeping raw graph.")
            except ImportError:
                print("WARNING: onnxsim not installed (pip install onnxsim); "
                      "shipping the unsimplified graph.")

        # Parity against the (possibly simplified) ONNX before converting.
        verify_parity(wrapped, output_path, input_shape,
                      output_names=["rotation_matrix"], atol=parity_atol)

        # Convert to the .ort flatbuffer with mobile-safe optimizations.
        ort_path = os.path.splitext(output_path)[0] + ".ort"
        print("Converting to ORT format (.ort, optimization_style=Fixed) ...")
        subprocess.run(
            [sys.executable, "-m",
             "onnxruntime.tools.convert_onnx_models_to_ort",
             output_path, "--optimization_style", "Fixed"],
            check=True,
        )
        print(f"Mobile artifacts:\n  ONNX: {output_path}\n  ORT : {ort_path}")
        return output_path, ort_path


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



def verify_parity(torch_model: nn.Module, onnx_path: str,
                  input_shape: Tuple[int, ...], output_names: list,
                  n: int = 8, atol: float = 1e-4) -> Dict[str, float]:
    """
    Assert the exported ONNX graph matches the PyTorch model numerically.

    Runs both on the same random inputs and compares each output. Raises
    AssertionError if any output's max absolute difference exceeds `atol`.
    Returns the per-output worst-case max-abs-diff for logging.
    """
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    onnx_out_order = [o.name for o in sess.get_outputs()]

    worst = {name: 0.0 for name in output_names}
    torch_model.eval()
    with torch.no_grad():
        for _ in range(n):
            x = torch.randn(1, *input_shape)
            t_out = torch_model(x)
            if not isinstance(t_out, (list, tuple)):
                t_out = (t_out,)  # single-output models (e.g. mobile, rot-only)
            o_out = sess.run(None, {in_name: x.numpy()})
            o_by_name = dict(zip(onnx_out_order, o_out))
            for name, t in zip(output_names, t_out):
                d = float(np.abs(t.cpu().numpy() - o_by_name[name]).max())
                worst[name] = max(worst[name], d)

    print(f"Parity check over {n} random inputs (atol={atol:g}):")
    for name, d in worst.items():
        status = 'OK' if d <= atol else 'FAIL'
        print(f"  {name:<16} max|torch-onnx| = {d:.3e}  [{status}]")
    bad = {k: v for k, v in worst.items() if v > atol}
    if bad:
        raise AssertionError(
            f"ONNX parity check FAILED (atol={atol:g}): {bad}. The exported "
            "model does not match PyTorch — do not ship it.")
    print("Parity check PASSED.")
    return worst


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
    parser.add_argument('--no-deploy', dest='deploy', action='store_false',
                        help='Export the NON-reparameterized (training) structure. '
                             'Default is deploy=True (fused convs), which matches '
                             'our checkpoints and is what we ship to mobile.')
    parser.set_defaults(deploy=True)
    parser.add_argument('--no-verify', dest='verify', action='store_false',
                        help='Skip the PyTorch<->ONNX parity check after export.')
    parser.set_defaults(verify=True)
    parser.add_argument('--parity-atol', type=float, default=1e-4,
                        help='Max allowed |torch-onnx| in the parity check (default 1e-4).')
    parser.add_argument('--mobile', action='store_true',
                        help='SixDRepNet only: emit a mobile-optimized artifact for '
                             'ONNX Runtime Mobile (static batch=1, rotation-matrix-only '
                             'output, onnxsim, and a .ort flatbuffer).')
    parser.add_argument('--no-simplify', dest='simplify', action='store_false',
                        help='In --mobile mode, skip onnxsim constant-folding.')
    parser.set_defaults(simplify=True)

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        if args.sixdrepnet:
            print(f"Exporting SixDRepNet model with backbone: {args.backbone}")
            print(f"Note: Using {args.backbone} as backbone. Available options: 'repnext_m0' to 'repnext_m5' or 'repvgg_a0'")
            if args.mobile:
                onnx_path, ort_path = SixDRepNetExporter.export_mobile(
                    weights_path=args.weights,
                    output_path=args.output,
                    backbone=args.backbone.lower(),
                    input_shape=tuple(args.input_shape),
                    opset_version=args.opset,
                    parity_atol=args.parity_atol,
                    simplify=args.simplify,
                )
                print(f"SixDRepNet mobile model exported to: {ort_path}")
            else:
                output_path = SixDRepNetExporter.export(
                    weights_path=args.weights,
                    output_path=args.output,
                    backbone=args.backbone.lower(),  # Ensure lowercase
                    input_shape=tuple(args.input_shape),
                    deploy=args.deploy,
                    verify=args.verify,
                    parity_atol=args.parity_atol,
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
