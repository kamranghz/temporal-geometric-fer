import numpy as np
from collections import deque
from typing import Optional, Dict

from .quality_metrics import compute_geometric_quality


class MultiFrameAggregator:
    """
    Quality-weighted temporal aggregation over a sliding window of frames.

    Emotion probabilities are noisy in isolation; averaging over recent frames
    dramatically reduces flicker while preserving genuine expression changes.
    Each frame is weighted by a geometric quality score (pose, brightness,
    sharpness) so low-quality frames contribute less to the aggregate.

    The output is a convex blend of:
        - the quality-weighted window average  (captures temporal context)
        - the current frame's probabilities    (preserves responsiveness)

    controlled by `lambda_weight`:
        stable_t = lambda * p_aggregated + (1 - lambda) * p_current

    Args:
        window_size:   Number of past frames to retain (default 5).
        tau:           Softmax temperature applied to quality weights before
                       aggregation.  Higher tau → sharper focus on the best
                       frames; tau=1 is standard softmax (default 1.5).
        lambda_weight: Blend coefficient between aggregated and current frame
                       (default 0.4; increase toward 1.0 for smoother but
                       more lagging output).
    """

    def __init__(
        self,
        window_size: int = 5,
        tau: float = 1.5,
        lambda_weight: float = 0.4,
    ) -> None:
        self.window_size = window_size
        self.tau = tau
        self.lambda_weight = lambda_weight

        # Each entry: (probs: np.ndarray, quality: float)
        self._window: deque = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        probs: np.ndarray,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        brightness: Optional[float] = None,
        sharpness: Optional[float] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """
        Ingest a new frame and return the stabilised probability distribution.

        Args:
            probs:      Softmax emotion probabilities, shape (K,).
            yaw:        Head yaw in degrees (from MediaPipe / OpenFace).
            pitch:      Head pitch in degrees.
            brightness: Mean luminance of the face ROI.
            sharpness:  Variance of Laplacian for the face ROI.
            landmarks:  (optional) Raw landmark array; reserved for future
                        motion-based quality extensions.

        Returns:
            Dictionary with:
                stable_probs  – Quality-weighted temporally smoothed distribution.
                geo_quality   – Quality score of the current frame in [0, 1].
                window_depth  – Number of frames currently in the window.
        """
        probs = np.asarray(probs, dtype=np.float64)

        geo_quality = compute_geometric_quality(yaw, pitch, brightness, sharpness)
        self._window.append((probs.copy(), geo_quality))

        stable_probs = self._aggregate(probs)

        return {
            "stable_probs": stable_probs,
            "geo_quality": geo_quality,
            "window_depth": len(self._window),
        }

    def reset(self) -> None:
        """Clear the internal frame buffer."""
        self._window.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate(self, current_probs: np.ndarray) -> np.ndarray:
        """Compute quality-weighted mean over the window and blend with current."""
        if len(self._window) == 1:
            return current_probs.copy()

        stacked = np.stack([p for p, _ in self._window])          # (N, K)
        qualities = np.array([q for _, q in self._window])        # (N,)

        # Softmax with temperature over quality scores
        shifted = qualities * self.tau - qualities.max() * self.tau  # numerical stability
        weights = np.exp(shifted)
        weights /= weights.sum()

        aggregated = (stacked * weights[:, None]).sum(axis=0)     # (K,)

        blended = self.lambda_weight * aggregated + (1.0 - self.lambda_weight) * current_probs

        # Renormalise to a valid probability distribution
        total = blended.sum()
        if total > 1e-8:
            blended /= total

        return blended
