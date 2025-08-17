"""
Test script to verify RayNet model can run without data
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from raynet import RayNet, EyeballModel, IrisMeshDecoder, GazeRayEstimator


def test_eyeball_model():
    """Test the eyeball model component"""
    print("Testing EyeballModel...")

    model = EyeballModel(input_dim=256, n_vertices=642)
    model.eval()

    # Test input
    batch_size = 2
    features = torch.randn(batch_size, 256)

    # Forward pass
    output = model(features)

    print(f"  Vertices shape: {output['vertices'].shape}")
    print(f"  Center shape: {output['center'].shape}")
    print(f"  Radius shape: {output['radius'].shape}")
    print(f"  Rotation shape: {output['rotation'].shape}")

    assert output['vertices'].shape == (batch_size, 642, 3)
    assert output['center'].shape == (batch_size, 3)
    assert output['radius'].shape == (batch_size,)
    assert output['rotation'].shape == (batch_size, 3, 3)

    print("  ✓ EyeballModel test passed!")
    return True


def test_iris_decoder():
    """Test the iris decoder"""
    print("Testing IrisMeshDecoder...")

    model = IrisMeshDecoder(input_dim=256, n_landmarks=100)
    model.eval()

    batch_size = 2
    features = torch.randn(batch_size, 256)

    # Test without eyeball info
    output = model(features)

    print(f"  Left iris shape: {output['iris_landmarks']['left'].shape}")
    print(f"  Right iris shape: {output['iris_landmarks']['right'].shape}")
    print(f"  Left pupil shape: {output['pupil_centers']['left'].shape}")
    print(f"  Right pupil shape: {output['pupil_centers']['right'].shape}")

    assert output['iris_landmarks']['left'].shape == (batch_size, 100, 3)
    assert output['iris_landmarks']['right'].shape == (batch_size, 100, 3)
    assert output['pupil_centers']['left'].shape == (batch_size, 3)
    assert output['pupil_centers']['right'].shape == (batch_size, 3)

    print("  ✓ IrisMeshDecoder test passed!")
    return True


def test_full_model():
    """Test the full RayNet model"""
    print("Testing full RayNet model...")

    # Mock backbone for testing
    class MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 64, 3, 1, 1)
            self.stages = nn.ModuleList([
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.Conv2d(256, 512, 3, 2, 1)
            ])

    backbone = MockBackbone()
    in_channels_list = [64, 128, 256, 512]

    model = RayNet(
        backbone=backbone,
        in_channels_list=in_channels_list,
        n_iris_landmarks=100,
        panet_out_channels=256
    )
    model.eval()

    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 448, 448)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print("Output keys:", output.keys())

    # Check key outputs
    expected_outputs = [
        'eyeball_left', 'eyeball_right', 'eyeball_centers',
        'iris_landmarks', 'pupil_centers', 'ray_origin',
        'ray_direction', 'gaze_vector', 'gaze_depth'
    ]

    for key in expected_outputs:
        assert key in output, f"Missing output: {key}"
        if isinstance(output[key], torch.Tensor):
            print(f"  {key}: {output[key].shape}")
        elif isinstance(output[key], dict):
            print(f"  {key}: dict with keys {output[key].keys()}")

    print("  ✓ Full RayNet test passed!")
    return True


def test_ray_casting():
    """Test ray casting functionality"""
    print("Testing ray casting...")

    # Mock model outputs
    ray_origin = torch.tensor([[0., 0., 0.]])
    ray_direction = torch.tensor([[0., 0., 1.]])  # Looking straight ahead

    # Screen at 500mm distance
    screen_point = torch.tensor([0., 0., 500.])
    screen_normal = torch.tensor([0., 0., -1.])  # Facing the user

    # Simple ray-plane intersection
    t = torch.sum(screen_normal * (screen_point - ray_origin), dim=1) / \
        (torch.sum(screen_normal * ray_direction, dim=1) + 1e-8)

    intersection = ray_origin + t.unsqueeze(-1) * ray_direction

    print(f"  Intersection point: {intersection}")
    print(f"  Distance: {t}")

    # Check intersection is on the screen plane
    assert torch.allclose(intersection[0, 2], torch.tensor(500.)), "Intersection not on screen plane"

    print("  ✓ Ray casting test passed!")
    return True


def main():
    """Run all tests"""
    print("=" * 50)
    print("RAYNET MODEL TESTS")
    print("=" * 50)

    tests = [
        test_eyeball_model,
        test_iris_decoder,
        test_full_model,
        test_ray_casting
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed with error: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("✓ All tests passed! Model is ready for training.")
    else:
        print("✗ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()