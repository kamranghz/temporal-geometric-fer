import numpy as np
from typing import Dict, Optional

from .quality_scoring import (
    compute_brightness_score,
    compute_sharpness_score,
    compute_pose_score,
    compute_combined_quality,
)


class ContextAwareFilter:
    """
    Adjusts per-frame model confidence based on visual context quality.

    Three independent quality signals — lighting (brightness), focus
    (sharpness), and head orientation (pose) — are combined into a scalar
    quality score.  That score is then used in one of two modes:

    * ``'scale'`` — multiply raw confidence directly:
        c_adjusted = c_model × c_quality

    * ``'threshold'`` — raise the acceptance threshold when quality is low:
        τ_t = τ_0 × (1 + k × (1 − c_quality))

    The threshold mode is preferred when downstream decisions gate on a
    fixed confidence cutoff; the scale mode is simpler for regression tasks.

    Args:
        brightness_min:    Lower brightness clamp (default 50).
        brightness_max:    Upper brightness clamp (default 200).
        sharpness_min:     Lower Laplacian variance clamp (default 10).
        sharpness_max:     Upper Laplacian variance clamp (default 100).
        yaw_sensitivity:   Characteristic yaw decay angle in degrees (default 27.5).
        pitch_sensitivity: Characteristic pitch decay angle in degrees (default 22.5).
        w_brightness:      Weight for brightness component (default 0.3).
        w_sharpness:       Weight for sharpness component (default 0.3).
        w_pose:            Weight for pose component (default 0.4).
        threshold_k:       Sensitivity coefficient for adaptive threshold (default 0.5).
        base_threshold:    Baseline confidence threshold τ_0 (default 0.6).
        mode:              ``'scale'`` or ``'threshold'`` (default ``'scale'``).
    """

    def __init__(
        self,
        brightness_min: float = 50.0,
        brightness_max: float = 200.0,
        sharpness_min: float = 10.0,
        sharpness_max: float = 100.0,
        yaw_sensitivity: float = 27.5,
        pitch_sensitivity: float = 22.5,
        w_brightness: float = 0.3,
        w_sharpness: float = 0.3,
        w_pose: float = 0.4,
        threshold_k: float = 0.5,
        base_threshold: float = 0.6,
        mode: str = "scale",
    ) -> None:
        if mode not in ("scale", "threshold"):
            raise ValueError(f"mode must be 'scale' or 'threshold', got '{mode}'")

        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.sharpness_min = sharpness_min
        self.sharpness_max = sharpness_max
        self.yaw_sensitivity = yaw_sensitivity
        self.pitch_sensitivity = pitch_sensitivity
        self.w_brightness = w_brightness
        self.w_sharpness = w_sharpness
        self.w_pose = w_pose
        self.threshold_k = threshold_k
        self.base_threshold = base_threshold
        self.mode = mode

    def update(
        self,
        probs: Optional[np.ndarray] = None,
        confidence: Optional[float] = None,
        brightness: Optional[float] = None,
        sharpness: Optional[float] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute quality-adjusted confidence for the current frame.

        Either ``probs`` or ``confidence`` should be supplied; if both are
        given, ``confidence`` takes precedence.

        Args:
            probs:      Softmax probabilities, shape (K,).  Used to derive
                        ``confidence = max(probs)`` when ``confidence`` is None.
            confidence: Raw model confidence (max softmax probability).
            brightness: Mean luminance of the face ROI.
            sharpness:  Variance of Laplacian for the face ROI.
            yaw:        Head yaw angle in degrees.
            pitch:      Head pitch angle in degrees.

        Returns:
            Dictionary containing:
                ``c_qual``       – Combined quality score in [0, 1].
                ``b_score``      – Brightness sub-score.
                ``s_score``      – Sharpness sub-score.
                ``p_score``      – Pose sub-score.
                ``c_model``      – Raw model confidence (before adjustment).
                ``c_adj``        – Adjusted confidence (scale mode only).
                ``tau_adaptive`` – Adaptive threshold (threshold mode only).
        """
        if confidence is None:
            confidence = float(np.max(probs)) if probs is not None else 0.5

        b_score = (
            compute_brightness_score(brightness, self.brightness_min, self.brightness_max)
            if brightness is not None else 1.0
        )
        s_score = (
            compute_sharpness_score(sharpness, self.sharpness_min, self.sharpness_max)
            if sharpness is not None else 1.0
        )
        p_score = (
            compute_pose_score(yaw, pitch, self.yaw_sensitivity, self.pitch_sensitivity)
            if (yaw is not None and pitch is not None) else 1.0
        )

        c_qual = compute_combined_quality(b_score, s_score, p_score, self.w_brightness, self.w_sharpness, self.w_pose)

        result: Dict[str, float] = {
            "c_qual": c_qual,
            "b_score": b_score,
            "s_score": s_score,
            "p_score": p_score,
            "c_model": confidence,
        }

        if self.mode == "scale":
            result["c_adj"] = confidence * c_qual
        else:
            result["tau_adaptive"] = self.base_threshold * (1.0 + self.threshold_k * (1.0 - c_qual))
            result["c_adj"] = confidence  # threshold mode does not rescale confidence

        return result
