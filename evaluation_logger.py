"""
Dual-pipeline evaluation logger for live FER sessions.

Records per-frame emotion data from both the OpenFace-only and Hybrid-Geometric
pipelines in JSONL format.  JSONL (one JSON object per line) allows real-time
appending without holding the entire session in memory, and simplifies
post-session analysis with pandas or any line-oriented tool.
"""

import json
import time
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np


PipelineType = Literal["OpenFaceOnly", "HybridGeometric"]


class SessionLogger:
    """
    Per-session JSONL writer for one pipeline.

    A separate instance should be created for each pipeline (OpenFaceOnly and
    HybridGeometric) so that log files remain independent and the comparison
    analysis does not require parsing interleaved streams.

    Args:
        pipeline:  Which pipeline this logger belongs to.
        log_dir:   Directory for output files (created if absent).
    """

    _FILENAME: Dict[str, str] = {
        "OpenFaceOnly":    "openface_only.jsonl",
        "HybridGeometric": "hybrid_geometric.jsonl",
    }

    def __init__(self, pipeline: PipelineType, log_dir: str = "logs/component_1") -> None:
        self.pipeline = pipeline
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.log_dir / self._FILENAME[pipeline]
        self._file = open(self.jsonl_path, "w", encoding="utf-8")

        self._session_start = time.time()
        self._frame_count = 0
        self._prev_raw_probs: Optional[np.ndarray] = None
        self._prev_ema_probs: Optional[np.ndarray] = None
        self._prev_ema_label: Optional[str] = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def log_frame(
        self,
        raw_probs: np.ndarray,
        raw_label: str,
        ema_probs: np.ndarray,
        ema_label: str,
        stability_index: float,
        confidence_score: float,
        quality_score: float,
        flicker_rate_30f: float,
        label_switching_rate: float,
    ) -> None:
        """
        Append one frame record to the JSONL file.

        Args:
            raw_probs:           Raw softmax output from the backbone, shape (8,).
            raw_label:           Argmax emotion label from raw_probs.
            ema_probs:           EMA-smoothed probabilities, shape (8,).
            ema_label:           Argmax emotion label from ema_probs.
            stability_index:     Current SI value in [0, 1].
            confidence_score:    Model confidence (max softmax probability).
            quality_score:       Quality score from ContextAwareFilter.
            flicker_rate_30f:    Label switches per 30 frames.
            label_switching_rate: Label switching rate from StabilityIndex.
        """
        now = time.time()

        ftc = (
            float(0.5 * np.abs(ema_probs - self._prev_ema_probs).sum())
            if self._prev_ema_probs is not None else 0.0
        )
        label_changed = (
            int(ema_label != self._prev_ema_label)
            if self._prev_ema_label is not None else 0
        )

        record = {
            "timestamp": now,
            "elapsed_seconds": now - self._session_start,
            "frame_index": self._frame_count,
            "dataset_source": "LiveUser",
            "pipeline_type": self.pipeline,
            "raw_probabilities": [round(float(p), 6) for p in raw_probs],
            "raw_label": raw_label,
            "ema_probabilities": [round(float(p), 6) for p in ema_probs],
            "ema_label": ema_label,
            "frame_to_frame_change": round(ftc, 6),
            "label_changed": label_changed,
            "stability_index": round(stability_index, 6),
            "confidence_score": round(confidence_score, 6),
            "quality_score": round(quality_score, 6),
            "flicker_rate_30f": round(flicker_rate_30f, 4),
            "label_switching_rate": round(label_switching_rate, 6),
        }

        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        self._prev_raw_probs = raw_probs.copy()
        self._prev_ema_probs = ema_probs.copy()
        self._prev_ema_label = ema_label
        self._frame_count += 1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def frame_count(self) -> int:
        return self._frame_count


# ---------------------------------------------------------------------------
# Backwards-compatible alias used in test_evaluation_logger.py
# ---------------------------------------------------------------------------

class Component1Logger(SessionLogger):
    """Alias for SessionLogger kept for backwards compatibility."""

    def __init__(self, pipeline_type: PipelineType, log_dir: str = "logs/component_1") -> None:
        super().__init__(pipeline=pipeline_type, log_dir=log_dir)
