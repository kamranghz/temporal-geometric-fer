import numpy as np


def compute_brightness_score(
    luminance: float,
    mu_min: float = 50.0,
    mu_max: float = 200.0,
) -> float:
    """
    Linear brightness quality score in [0, 1].

    Score is 0 below `mu_min` (too dark), rises linearly to 1 at `mu_max`
    (well-lit), and is clipped to 1 above that.

    Args:
        luminance: Mean pixel intensity of the face ROI (0–255).
        mu_min:    Minimum acceptable brightness.
        mu_max:    Maximum acceptable brightness.
    """
    return float(np.clip((luminance - mu_min) / (mu_max - mu_min), 0.0, 1.0))


def compute_sharpness_score(
    laplacian_var: float,
    lap_min: float = 10.0,
    lap_max: float = 100.0,
) -> float:
    """
    Linear sharpness quality score in [0, 1].

    Uses the variance of the Laplacian as a focus measure (higher = sharper).

    Args:
        laplacian_var: Variance of the Laplacian over the face ROI.
        lap_min:       Threshold below which the image is considered blurry.
        lap_max:       Threshold above which the image is considered sharp.
    """
    return float(np.clip((laplacian_var - lap_min) / (lap_max - lap_min), 0.0, 1.0))


def compute_pose_score(
    yaw: float,
    pitch: float,
    yaw_sensitivity: float = 27.5,
    pitch_sensitivity: float = 22.5,
) -> float:
    """
    Exponential pose quality score in [0, 1].

    Score is 1.0 for a perfectly frontal face and decays as yaw or pitch
    increase.  Inspired by AffectNet annotation guidelines, which note
    significantly higher annotation error rates beyond ±30° yaw.

    Formula::

        p = exp(-(|yaw| / Y0 + |pitch| / P0))

    Args:
        yaw:               Head yaw in degrees.
        pitch:             Head pitch in degrees.
        yaw_sensitivity:   Characteristic decay angle for yaw.
        pitch_sensitivity: Characteristic decay angle for pitch.
    """
    return float(np.exp(-(abs(yaw) / yaw_sensitivity + abs(pitch) / pitch_sensitivity)))


def compute_combined_quality(
    brightness_score: float,
    sharpness_score: float,
    pose_score: float,
    w_brightness: float = 0.3,
    w_sharpness: float = 0.3,
    w_pose: float = 0.4,
) -> float:
    """
    Weighted combination of brightness, sharpness, and pose scores.

    Pose carries the highest default weight because extreme angles degrade
    expression visibility most severely.

    Args:
        brightness_score: Output of :func:`compute_brightness_score`.
        sharpness_score:  Output of :func:`compute_sharpness_score`.
        pose_score:       Output of :func:`compute_pose_score`.
        w_brightness:     Weight for the brightness component (default 0.3).
        w_sharpness:      Weight for the sharpness component (default 0.3).
        w_pose:           Weight for the pose component (default 0.4).

    Returns:
        Combined quality score in [0, 1].
    """
    return (
        w_brightness * brightness_score
        + w_sharpness * sharpness_score
        + w_pose * pose_score
    )
