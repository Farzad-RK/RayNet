# utils.py

import torch
import torch.nn.functional as F
import numpy as np


def normalize_vector(vectors, eps=1e-8):
    """
    Normalize vectors to unit length

    Args:
        vectors: [..., 3] tensor of vectors
        eps: Small epsilon for numerical stability

    Returns:
        Normalized vectors of same shape
    """
    norm = torch.norm(vectors, dim=-1, keepdim=True)
    return vectors / (norm + eps)


def create_orthonormal_basis(normal):
    """
    Create orthonormal u, v axes perpendicular to normal vector

    Args:
        normal: [..., 3] normal vectors (should be normalized)

    Returns:
        u_axis, v_axis: [..., 3] orthonormal vectors
    """
    # Handle batch dimensions
    shape = normal.shape
    device = normal.device

    # Choose arbitrary vector not parallel to normal
    temp = torch.zeros_like(normal)

    # Use different temp vectors based on which component is smallest
    abs_normal = torch.abs(normal)
    min_component = torch.argmin(abs_normal, dim=-1, keepdim=True)

    # Set temp vector along the axis with smallest normal component
    temp.scatter_(-1, min_component, 1.0)

    # Create orthonormal basis using Gram-Schmidt
    u_axis = temp - torch.sum(temp * normal, dim=-1, keepdim=True) * normal
    u_axis = normalize_vector(u_axis)

    v_axis = torch.cross(normal, u_axis, dim=-1)
    v_axis = normalize_vector(v_axis)

    return u_axis, v_axis


def fit_plane_svd(points):
    """
    Fit plane through 3D points using SVD

    Args:
        points: [N, 3] points in 3D

    Returns:
        plane_normal: [3] unit normal vector
        plane_d: scalar distance parameter (plane equation: n·x + d = 0)
    """
    # Center the points
    centroid = torch.mean(points, dim=0)  # [3]
    centered_points = points - centroid  # [N, 3]

    # SVD to find plane normal (smallest singular vector)
    U, S, Vt = torch.svd(centered_points)
    normal = Vt[-1, :]  # Last row of V^T gives normal direction

    # Compute d parameter: n·centroid + d = 0 => d = -n·centroid
    d = -torch.dot(normal, centroid)

    return normal, d


def compute_plane_distances(points, plane_normal, plane_d):
    """
    Compute distances from points to plane

    Args:
        points: [N, 3] points
        plane_normal: [3] unit normal vector
        plane_d: scalar distance parameter

    Returns:
        distances: [N] signed distances to plane
    """
    distances = torch.abs(torch.sum(points * plane_normal, dim=-1) + plane_d)
    return distances


def euler_to_rotation_matrix(euler_angles):
    """
    Convert Euler angles to rotation matrix

    Args:
        euler_angles: [..., 3] Euler angles (roll, pitch, yaw) in radians

    Returns:
        rotation_matrices: [..., 3, 3] rotation matrices
    """
    # Handle batch dimensions
    original_shape = euler_angles.shape[:-1]
    euler_angles = euler_angles.view(-1, 3)  # [B, 3]
    batch_size = euler_angles.shape[0]

    # Extract individual angles
    roll = euler_angles[:, 0]  # [B]
    pitch = euler_angles[:, 1]  # [B]
    yaw = euler_angles[:, 2]  # [B]

    # Compute trigonometric values
    cos_roll = torch.cos(roll)
    sin_roll = torch.sin(roll)
    cos_pitch = torch.cos(pitch)
    sin_pitch = torch.sin(pitch)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # Build rotation matrix (ZYX convention)
    R = torch.zeros(batch_size, 3, 3, device=euler_angles.device)

    R[:, 0, 0] = cos_yaw * cos_pitch
    R[:, 0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    R[:, 0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll

    R[:, 1, 0] = sin_yaw * cos_pitch
    R[:, 1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    R[:, 1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll

    R[:, 2, 0] = -sin_pitch
    R[:, 2, 1] = cos_pitch * sin_roll
    R[:, 2, 2] = cos_pitch * cos_roll

    # Reshape back to original batch dimensions
    return R.view(*original_shape, 3, 3)


def transform_HCS_to_CCS(points_hcs, head_rotation):
    """
    Transform points from Head Coordinate System to Camera Coordinate System

    Args:
        points_hcs: [..., 3] points in HCS
        head_rotation: [..., 3, 3] head rotation matrix

    Returns:
        points_ccs: [..., 3] points in CCS
    """
    # Apply rotation: points_ccs = R @ points_hcs
    points_ccs = torch.matmul(head_rotation, points_hcs.unsqueeze(-1)).squeeze(-1)
    return points_ccs


def generate_iris_100_landmarks_in_plane(eyeball_center, iris_radius, cornea2center,
                                         batch_size, device, num_points=100):
    """
    Generate 100 iris landmarks in a perfect circle on iris plane

    Args:
        eyeball_center: [B, 3] eyeball center in HCS
        iris_radius: [B, 1] or [B] iris radius in cm
        cornea2center: [B, 1] or [B] distance from cornea center to eyeball center
        batch_size: Batch size
        device: Device
        num_points: Number of landmarks (100)

    Returns:
        iris_landmarks: [B, 100, 3] iris landmarks in HCS
    """
    # Ensure proper shapes
    if iris_radius.dim() == 2:
        iris_radius = iris_radius.squeeze(-1)  # [B]
    if cornea2center.dim() == 2:
        cornea2center = cornea2center.squeeze(-1)  # [B]

    # Default optical axis direction (looking forward in HCS)
    optical_axis = torch.tensor([0.0, 0.0, -1.0], device=device).unsqueeze(0).expand(batch_size, -1)  # [B, 3]

    # Iris plane center (along optical axis from eyeball center)
    iris_center = eyeball_center + cornea2center.unsqueeze(-1) * optical_axis  # [B, 3]

    # Create orthonormal basis for iris plane
    u_axis, v_axis = create_orthonormal_basis(optical_axis)  # [B, 3], [B, 3]

    # Generate angles for 100 points uniformly around circle
    angles = torch.linspace(0, 2 * np.pi, num_points, device=device)  # [100]
    angles = angles.unsqueeze(0).expand(batch_size, -1)  # [B, 100]

    # Generate points on circle
    cos_angles = torch.cos(angles)  # [B, 100]
    sin_angles = torch.sin(angles)  # [B, 100]

    # Scale by iris radius
    x_coords = iris_radius.unsqueeze(-1) * cos_angles  # [B, 100]
    y_coords = iris_radius.unsqueeze(-1) * sin_angles  # [B, 100]

    # Convert to 3D coordinates in iris plane
    iris_landmarks = (iris_center.unsqueeze(1) +  # [B, 1, 3]
                      x_coords.unsqueeze(-1) * u_axis.unsqueeze(1) +  # [B, 100, 3]
                      y_coords.unsqueeze(-1) * v_axis.unsqueeze(1))  # [B, 100, 3]

    return iris_landmarks


def linear_blend_skinning(vertices, joint_transforms, blend_weights):
    """
    FLAME-style Linear Blend Skinning for eye structures

    Args:
        vertices: [B, N, 3] template vertices
        joint_transforms: Dict with joint transformation matrices
        blend_weights: [N, K] blend weights for N vertices and K joints

    Returns:
        transformed_vertices: [B, N, 3] transformed vertices
    """
    batch_size, num_vertices, _ = vertices.shape
    num_joints = len(joint_transforms)
    device = vertices.device

    # Stack joint transforms
    if 'head_pose' in joint_transforms:
        # Create 4x4 transformation matrices from rotation matrices and gaze
        joint_matrices = []

        # Joint 0: Head pose (rotation only)
        head_pose_4x4 = create_4x4_transform(joint_transforms['head_pose'])  # [B, 4, 4]
        joint_matrices.append(head_pose_4x4)

        # Joint 1: Head gaze (treated as identity with translation)
        head_gaze_4x4 = create_4x4_identity_with_translation(
            joint_transforms['head_gaze'] * 0.01  # Small translation for gaze influence
        )  # [B, 4, 4]
        joint_matrices.append(head_gaze_4x4)

        # Joint 2: Left eyeball
        left_eyeball_4x4 = create_4x4_transform(joint_transforms['left_eyeball'])  # [B, 4, 4]
        joint_matrices.append(left_eyeball_4x4)

        # Joint 3: Right eyeball
        right_eyeball_4x4 = create_4x4_transform(joint_transforms['right_eyeball'])  # [B, 4, 4]
        joint_matrices.append(right_eyeball_4x4)

        joint_matrices = torch.stack(joint_matrices, dim=1)  # [B, K, 4, 4]
    else:
        # Fallback: create identity transforms
        joint_matrices = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_joints, -1, -1)

    # Convert vertices to homogeneous coordinates
    vertices_homo = torch.cat([vertices, torch.ones(batch_size, num_vertices, 1, device=device)], dim=-1)  # [B, N, 4]

    # Apply Linear Blend Skinning
    transformed_vertices = torch.zeros_like(vertices)  # [B, N, 3]

    for v_idx in range(num_vertices):
        vertex = vertices_homo[:, v_idx]  # [B, 4]
        weights = blend_weights[v_idx]  # [K]

        # Weighted combination of joint transformations
        weighted_transform = torch.zeros(batch_size, 4, 4, device=device)
        for j_idx in range(num_joints):
            weighted_transform += weights[j_idx] * joint_matrices[:, j_idx]  # [B, 4, 4]

        # Apply transformation to vertex
        transformed_vertex = torch.matmul(weighted_transform, vertex.unsqueeze(-1)).squeeze(-1)  # [B, 4]
        transformed_vertices[:, v_idx] = transformed_vertex[:, :3]  # Take only 3D coordinates

    return transformed_vertices


def create_4x4_transform(rotation_matrix, translation=None):
    """
    Create 4x4 transformation matrix from rotation and optional translation

    Args:
        rotation_matrix: [B, 3, 3] rotation matrices
        translation: [B, 3] translation vectors (optional)

    Returns:
        transform_4x4: [B, 4, 4] transformation matrices
    """
    batch_size = rotation_matrix.shape[0]
    device = rotation_matrix.device

    transform_4x4 = torch.zeros(batch_size, 4, 4, device=device)

    # Set rotation part
    transform_4x4[:, :3, :3] = rotation_matrix

    # Set translation part
    if translation is not None:
        transform_4x4[:, :3, 3] = translation

    # Set homogeneous coordinate
    transform_4x4[:, 3, 3] = 1.0

    return transform_4x4


def create_4x4_identity_with_translation(translation):
    """
    Create 4x4 identity matrix with translation

    Args:
        translation: [B, 3] translation vectors

    Returns:
        transform_4x4: [B, 4, 4] transformation matrices
    """
    batch_size = translation.shape[0]
    device = translation.device

    transform_4x4 = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()
    transform_4x4[:, :3, 3] = translation

    return transform_4x4


def geodesic_distance_rotmat(R1, R2, eps=1e-7):
    """
    Compute geodesic distance between rotation matrices

    Args:
        R1: [B, 3, 3] rotation matrices
        R2: [B, 3, 3] rotation matrices
        eps: Small epsilon for numerical stability

    Returns:
        geodesic distances: [B] or scalar (mean)
    """
    # Compute relative rotation
    R_rel = torch.bmm(R1, R2.transpose(1, 2))  # [B, 3, 3]

    # Extract rotation angle from trace
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]  # [B]
    cos_angle = (trace - 1) / 2

    # Clamp to valid range for acos
    cos_angle = torch.clamp(cos_angle, -1 + eps, 1 - eps)

    # Compute angle
    angles = torch.acos(cos_angle)  # [B]

    return torch.mean(angles)  # Return mean across batch


def ortho6d_to_rotmat(ortho6d):
    """
    Convert 6D orthogonal representation to rotation matrix
    Based on "On the Continuity of Rotation Representations in Neural Networks"

    Args:
        ortho6d: [..., 6] 6D representation

    Returns:
        rotation matrices: [..., 3, 3]
    """
    # Handle batch dimensions
    original_shape = ortho6d.shape[:-1]
    ortho6d = ortho6d.view(-1, 6)  # [B, 6]

    x_raw = ortho6d[:, 0:3]  # [B, 3]
    y_raw = ortho6d[:, 3:6]  # [B, 3]

    # Normalize first vector
    x = normalize_vector(x_raw)  # [B, 3]

    # Gram-Schmidt process
    z = torch.cross(x, y_raw, dim=1)  # [B, 3]
    z = normalize_vector(z)  # [B, 3]
    y = torch.cross(z, x, dim=1)  # [B, 3]

    # Stack to form rotation matrix
    rotmat = torch.stack([x, y, z], dim=2)  # [B, 3, 3]

    # Reshape back to original batch dimensions
    return rotmat.view(*original_shape, 3, 3)


def rodrigues_rotation_matrix(axis_angle):
    """
    Convert axis-angle representation to rotation matrix using Rodrigues formula

    Args:
        axis_angle: [..., 3] axis-angle vectors

    Returns:
        rotation matrices: [..., 3, 3]
    """
    # Handle batch dimensions
    original_shape = axis_angle.shape[:-1]
    axis_angle = axis_angle.view(-1, 3)  # [B, 3]
    batch_size = axis_angle.shape[0]
    device = axis_angle.device

    # Compute angle and axis
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)  # [B, 1]
    axis = axis_angle / (angle + 1e-8)  # [B, 3]

    # Rodrigues formula components
    cos_angle = torch.cos(angle)  # [B, 1]
    sin_angle = torch.sin(angle)  # [B, 1]

    # Cross product matrix
    K = torch.zeros(batch_size, 3, 3, device=device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    # Identity matrix
    I = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)

    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    R = I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle).unsqueeze(-1) * torch.bmm(K, K)

    # Reshape back to original batch dimensions
    return R.view(*original_shape, 3, 3)


def batch_eye_like(tensor, n):
    """
    Create batch of identity matrices with same batch dimensions as input tensor

    Args:
        tensor: Input tensor for batch dimensions
        n: Size of identity matrix

    Returns:
        batch_eye: [..., n, n] batch of identity matrices
    """
    batch_shape = tensor.shape[:-1] if tensor.dim() > 1 else tensor.shape
    device = tensor.device

    eye = torch.eye(n, device=device)
    batch_eye = eye.expand(*batch_shape, n, n).contiguous()

    return batch_eye


def apply_camera_projection(points_3d, intrinsic_matrix, distortion_coeffs=None):
    """
    Apply camera projection to 3D points

    Args:
        points_3d: [B, N, 3] 3D points in camera coordinates
        intrinsic_matrix: [B, 3, 3] or [3, 3] camera intrinsic matrix
        distortion_coeffs: Optional distortion coefficients

    Returns:
        points_2d: [B, N, 2] projected 2D points
    """
    batch_size, num_points, _ = points_3d.shape

    # Handle single intrinsic matrix
    if intrinsic_matrix.dim() == 2:
        intrinsic_matrix = intrinsic_matrix.unsqueeze(0).expand(batch_size, -1, -1)

    # Project points: [B, 3, 3] @ [B, N, 3].T = [B, 3, N]
    points_projected = torch.bmm(intrinsic_matrix, points_3d.transpose(1, 2))  # [B, 3, N]
    points_projected = points_projected.transpose(1, 2)  # [B, N, 3]

    # Convert to 2D by dividing by Z coordinate
    z_coords = points_projected[:, :, 2:3]  # [B, N, 1]
    points_2d = points_projected[:, :, :2] / (z_coords + 1e-8)  # [B, N, 2]

    # Apply distortion if provided (placeholder - implement based on distortion model)
    if distortion_coeffs is not None:
        # Apply radial/tangential distortion
        pass  # Implement based on your distortion model

    return points_2d


def compute_iris_contour_properties(iris_landmarks_100):
    """
    Compute geometric properties of iris contour from 100 landmarks

    Args:
        iris_landmarks_100: [B, 100, 3] iris landmarks for one eye

    Returns:
        dict: Geometric properties
    """
    batch_size = iris_landmarks_100.shape[0]

    # Compute centroid
    centroid = torch.mean(iris_landmarks_100, dim=1)  # [B, 3]

    # Compute distances from centroid
    distances = torch.norm(iris_landmarks_100 - centroid.unsqueeze(1), dim=-1)  # [B, 100]

    # Compute radius statistics
    mean_radius = torch.mean(distances, dim=1)  # [B]
    std_radius = torch.std(distances, dim=1)  # [B]

    # Fit plane through landmarks
    properties = []
    for b in range(batch_size):
        plane_normal, plane_d = fit_plane_svd(iris_landmarks_100[b])  # Single batch item
        plane_distances = compute_plane_distances(iris_landmarks_100[b], plane_normal, plane_d)
        planarity_error = torch.max(plane_distances)

        properties.append({
            'centroid': centroid[b],
            'mean_radius': mean_radius[b],
            'std_radius': std_radius[b],
            'plane_normal': plane_normal,
            'planarity_error': planarity_error
        })

    return properties


def sample_points_on_circle(center, normal, radius, num_points, device):
    """
    Sample points uniformly on a circle in 3D space

    Args:
        center: [3] circle center
        normal: [3] circle normal (unit vector)
        radius: scalar radius
        num_points: number of points to sample
        device: torch device

    Returns:
        points: [num_points, 3] points on circle
    """
    # Create orthonormal basis
    u_axis, v_axis = create_orthonormal_basis(normal.unsqueeze(0))
    u_axis = u_axis.squeeze(0)  # [3]
    v_axis = v_axis.squeeze(0)  # [3]

    # Generate angles
    angles = torch.linspace(0, 2 * np.pi, num_points, device=device)

    # Generate points
    cos_angles = torch.cos(angles)  # [num_points]
    sin_angles = torch.sin(angles)  # [num_points]

    points = (center.unsqueeze(0) +  # [1, 3]
              radius * cos_angles.unsqueeze(-1) * u_axis.unsqueeze(0) +  # [num_points, 3]
              radius * sin_angles.unsqueeze(-1) * v_axis.unsqueeze(0))  # [num_points, 3]

    return points


def verify_rotation_matrix(R, eps=1e-6):
    """
    Verify that a matrix is a valid rotation matrix

    Args:
        R: [..., 3, 3] rotation matrices
        eps: tolerance

    Returns:
        is_valid: [...] boolean tensor indicating validity
    """
    # Check if R^T @ R = I
    should_be_identity = torch.matmul(R.transpose(-2, -1), R)
    identity = torch.eye(3, device=R.device).expand_as(should_be_identity)
    orthogonal_error = torch.norm(should_be_identity - identity, dim=(-2, -1))

    # Check if det(R) = 1
    det_R = torch.det(R)
    det_error = torch.abs(det_R - 1.0)

    is_valid = (orthogonal_error < eps) & (det_error < eps)

    return is_valid