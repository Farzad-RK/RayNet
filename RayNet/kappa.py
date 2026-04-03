"""
Kappa angle handling for GazeGene dataset.

Kappa is the angular offset between the optical axis and the visual axis.
It has only yaw and pitch components -- NO roll (anatomically impossible).
GazeGene stores kappa as (3,) but the third element must be zeroed.
"""

import numpy as np


# Population mean kappa angles (radians) for zero-calibration inference
KAPPA_MEAN_YAW = np.deg2rad(4.0)    # horizontal, nasal direction
KAPPA_MEAN_PITCH = np.deg2rad(1.0)  # vertical, superior direction


def build_R_kappa(kappa_angles):
    """
    Build kappa rotation matrix from GazeGene kappa angles.
    ALWAYS zeros out roll (index 2).

    Args:
        kappa_angles: (3,) or (2,) array from GazeGene subject data.
                      [yaw, pitch, (roll)] in radians.

    Returns:
        R_kappa: (3, 3) rotation matrix (yaw @ pitch, no roll)
    """
    yaw = float(kappa_angles[0])
    pitch = float(kappa_angles[1])
    # roll is HARD-ZERO regardless of what the dataset says

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    Ry = np.array([
        [cy,  0, sy],
        [0,   1, 0],
        [-sy, 0, cy]
    ])
    Rx = np.array([
        [1, 0,   0],
        [0, cp, -sp],
        [0, sp,  cp]
    ])
    return Ry @ Rx


def build_R_kappa_population_mean():
    """Build R_kappa from population mean kappa values."""
    return build_R_kappa(np.array([KAPPA_MEAN_YAW, KAPPA_MEAN_PITCH]))


def ground_truth_optical_axis(eyeball_center, pupil_center):
    """
    Compute ground-truth optical axis geometrically.

    The optical axis is the unit vector from eyeball center to pupil center.
    This is the correct training target (NOT head gaze, NOT visual axis).

    Args:
        eyeball_center: (3,) eyeball center position (any coordinate system)
        pupil_center: (3,) pupil center position (same coordinate system)

    Returns:
        optical_axis: (3,) unit vector
    """
    optical_axis = pupil_center - eyeball_center
    norm = np.linalg.norm(optical_axis)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])  # fallback
    return optical_axis / norm


def optical_to_visual(optical_axis, R_kappa):
    """
    Convert predicted optical axis to visual axis at inference time.

    Args:
        optical_axis: (3,) predicted optical axis (unit vector)
        R_kappa: (3, 3) kappa rotation matrix (from calibration or population mean)

    Returns:
        visual_axis: (3,) visual axis (unit vector)
    """
    visual = R_kappa @ optical_axis
    return visual / np.linalg.norm(visual)


def visual_to_optical(visual_axis, R_kappa):
    """
    Convert visual axis to optical axis (inverse of kappa rotation).

    Args:
        visual_axis: (3,) visual axis (unit vector)
        R_kappa: (3, 3) kappa rotation matrix

    Returns:
        optical_axis: (3,) optical axis (unit vector)
    """
    optical = np.linalg.inv(R_kappa) @ visual_axis
    return optical / np.linalg.norm(optical)
