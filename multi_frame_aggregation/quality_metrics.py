import numpy as np
from typing import Optional


def compute_geometric_quality(
    yaw: Optional[float],
    pitch: Optional[float],
    brightness: Optional[float],
    sharpness: Optional[float],
    yaw_sensitivity: float = 27.5,
    pitch_sensitivity: float = 22.5,
    brightness_min: float = 50.0,
    brightness_max: float = 200.0,
    sharpness_min: float = 10.0,
    sharpness_max: float = 100.0,
    w_pose: float = 0.5,
    w_brightness: float = 0.25,
    w_sharpness: float = 0.25,
) -> float:
    """
    Compute a combined per-frame geometric quality score in [0, 1].

    The score encodes how trustworthy a frame's emotion prediction is given:
      - Head pose deviation from frontal (yaw/pitch)
      - Illumination adequacy (brightness)
      - Image focus (sharpness via Laplacian variance)

    Formula:
        q_geo = w_pose * p_score + w_brightness * b_score + w_sharpness * s_score

    where
        p_score = exp(-(|yaw|/Y0 + |pitch|/P0))
        b_score = clip((mu - mu_min) / (mu_max - mu_min), 0, 1)
        s_score = clip((LAP - LAP_min) / (LAP_max - LAP_min), 0, 1)

    Missing inputs default to their neutral values (pose=frontal, brightness=mid-range,
    sharpness=sharp) so that absent data does not unfairly penalise a frame.

    Args:
        yaw:               Head yaw in degrees  (positive = turned right).
        pitch:             Head pitch in degrees (positive = looking down).
        brightness:        Mean luminance of the face ROI (0–255).
        sharpness:         Variance of Laplacian for the face ROI.
        yaw_sensitivity:   Half-width for pose decay (degrees).
        pitch_sensitivity: Half-height for pose decay (degrees).
        brightness_min:    Lower brightness clamp.
        brightness_max:    Upper brightness clamp.
        sharpness_min:     Lower sharpness clamp.
        sharpness_max:     Upper sharpness clamp.
        w_pose:            Weight for pose component.
        w_brightness:      Weight for brightness component.
        w_sharpness:       Weight for sharpness component.

    Returns:
        Scalar quality score in [0, 1].
    """
    # Pose score — 1.0 when perfectly frontal, decays exponentially off-axis.
    if yaw is not None and pitch is not None:
        p_score = float(np.exp(-(abs(yaw) / yaw_sensitivity + abs(pitch) / pitch_sensitivity)))
    else:
        p_score = 1.0

    # Brightness score — linear ramp between min and max thresholds.
    if brightness is not None:
        b_score = float(np.clip((brightness - brightness_min) / (brightness_max - brightness_min), 0.0, 1.0))
    else:
        b_score = 1.0

    # Sharpness score — linear ramp between min and max Laplacian variance.
    if sharpness is not None:
        s_score = float(np.clip((sharpness - sharpness_min) / (sharpness_max - sharpness_min), 0.0, 1.0))
    else:
        s_score = 1.0

    return w_pose * p_score + w_brightness * b_score + w_sharpness * s_score
