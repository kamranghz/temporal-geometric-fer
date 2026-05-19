import time
from collections import deque
from typing import Dict, Optional

import numpy as np


class StabilityIndex:
    """
    Online temporal stability metric for emotion probability streams.

    The Stability Index (SI) quantifies how consistent emotion predictions
    are across consecutive frames.  It combines:

    * A **per-frame distance** measuring the total variation between the
      current and previous probability distributions.
    * A **confidence × quality weight** (ω_t) that discounts noisy frames.
    * An **EMA update** to produce a smoothly evolving scalar metric.

    Formula::

        D_t  = 0.5 × ||q̂_t − q̂_{t-1}||₁
        ω_t  = c_confidence × c_quality
        s_t  = 1 − ω_t × D_t / (ω_t + ε)
        SI_t = (1 − β) × SI_{t-1} + β × s_t

    Additionally, a **windowed SI** is reported using a weighted sum over the
    last W frames (useful for segment-level evaluation).

    Args:
        beta:        EMA decay for the online SI update (default 0.2; higher
                     values make the metric more reactive to recent frames).
        window_size: Number of frames retained for windowed SI (default 30).
        epsilon:     Numerical stability constant (default 1e-6).
    """

    def __init__(
        self,
        beta: float = 0.2,
        window_size: int = 30,
        epsilon: float = 1e-6,
    ) -> None:
        self.beta = beta
        self.window_size = window_size
        self.epsilon = epsilon

        self._q_prev: Optional[np.ndarray] = None
        self._si_prev: Optional[float] = None
        self._label_prev: Optional[int] = None

        self._distance_buf: deque = deque(maxlen=window_size)
        self._weight_buf: deque = deque(maxlen=window_size)
        self._label_buf: deque = deque(maxlen=window_size)

        self._frame_count: int = 0
        self._label_changes: int = 0
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        probs: np.ndarray,
        confidence: float,
        quality: float,
        label: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Ingest a new frame and return updated stability metrics.

        Args:
            probs:      Smoothed emotion probabilities from the aggregation
                        pipeline, shape (K,).
            confidence: Per-frame confidence score (e.g. 1 − H(q)/log K).
            quality:    Per-frame quality score from :class:`ContextAwareFilter`.
            label:      Predicted emotion label (optional; enables flicker-rate
                        and label-stability reporting).

        Returns:
            Dictionary with:
                ``SI``              – Online EMA stability index in [0, 1].
                ``SI_windowed``     – Windowed weighted stability estimate.
                ``distance``        – Total variation D_t for this frame.
                ``stability_score`` – Per-frame score s_t before EMA smoothing.
                ``label_stability`` – Fraction of consecutive identical labels
                                      over the window (requires *label*).
                ``flicker_rate``    – Label switches per second.
                ``frame_count``     – Running frame counter.
        """
        if self._start_time is None:
            self._start_time = time.time()

        probs = np.asarray(probs, dtype=np.float64)

        distance = (
            0.5 * np.abs(probs - self._q_prev).sum()
            if self._q_prev is not None else 0.0
        )
        omega = confidence * quality
        per_frame_score = 1.0 - (omega * distance) / (omega + self.epsilon)

        si = (
            (1.0 - self.beta) * self._si_prev + self.beta * per_frame_score
            if self._si_prev is not None else per_frame_score
        )

        self._q_prev = probs.copy()
        self._si_prev = si

        self._distance_buf.append(distance)
        self._weight_buf.append(omega)

        if len(self._distance_buf) > 1:
            wsum = sum(w * d for w, d in zip(self._weight_buf, self._distance_buf))
            si_windowed = 1.0 - wsum / (sum(self._weight_buf) + self.epsilon)
        else:
            si_windowed = si

        if label is not None:
            self._label_buf.append(label)
            if self._label_prev is not None and label != self._label_prev:
                self._label_changes += 1
            self._label_prev = label

        if len(self._label_buf) > 1:
            consecutive = sum(
                1 for i in range(1, len(self._label_buf))
                if self._label_buf[i] == self._label_buf[i - 1]
            )
            label_stability = consecutive / (len(self._label_buf) - 1)
        else:
            label_stability = 1.0

        elapsed = time.time() - self._start_time
        flicker_rate = self._label_changes / elapsed if elapsed > 0 else 0.0

        self._frame_count += 1

        return {
            "SI": si,
            "SI_windowed": si_windowed,
            "distance": distance,
            "stability_score": per_frame_score,
            "label_stability": label_stability,
            "flicker_rate": flicker_rate,
            "frame_count": self._frame_count,
        }

    def stability_level(self, si: float) -> str:
        """Categorise a scalar SI value as 'Stable', 'Moderate', or 'Unstable'."""
        if si >= 0.8:
            return "Stable"
        if si >= 0.6:
            return "Moderate"
        return "Unstable"

    def reset(self) -> None:
        """Reset all internal state for a new session."""
        self._q_prev = None
        self._si_prev = None
        self._label_prev = None
        self._distance_buf.clear()
        self._weight_buf.clear()
        self._label_buf.clear()
        self._frame_count = 0
        self._label_changes = 0
        self._start_time = None

    @property
    def statistics(self) -> Dict[str, float]:
        """Summary statistics for the current session."""
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        return {
            "frame_count": self._frame_count,
            "label_changes": self._label_changes,
            "elapsed_seconds": elapsed,
            "flicker_rate": self._label_changes / elapsed if elapsed > 0 else 0.0,
            "current_SI": self._si_prev if self._si_prev is not None else 0.0,
        }
