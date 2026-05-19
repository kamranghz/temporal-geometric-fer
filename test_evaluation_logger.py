"""
Unit tests for the dual-pipeline evaluation logger.

Run with:  python test_evaluation_logger.py
"""

import json
import time
from pathlib import Path

import numpy as np

from evaluation_logger import Component1Logger


_EMOTIONS = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
_LOG_DIR  = "logs/test_component1"


def _random_probs() -> np.ndarray:
    p = np.random.rand(8).astype(np.float64)
    return p / p.sum()


def _log_n_frames(logger: Component1Logger, n: int, base_stability: float = 0.7) -> None:
    for _ in range(n):
        raw   = _random_probs()
        ema   = (_random_probs() * 0.2 + raw * 0.8)
        ema  /= ema.sum()

        logger.log_frame(
            raw_probs=raw,
            raw_label=_EMOTIONS[int(np.argmax(raw))],
            ema_probs=ema,
            ema_label=_EMOTIONS[int(np.argmax(ema))],
            stability_index=base_stability + np.random.rand() * 0.1,
            confidence_score=float(np.max(ema)),
            quality_score=0.7 + np.random.rand() * 0.2,
            flicker_rate_30f=5.0 + np.random.rand() * 2.0,
            label_switching_rate=0.2 + np.random.rand() * 0.1,
        )
        time.sleep(0.01)


def test_openface_logger() -> None:
    print("TEST 1: OpenFaceOnly logger")
    with Component1Logger("OpenFaceOnly", _LOG_DIR) as logger:
        _log_n_frames(logger, 5)
        path = logger.jsonl_path
        count = logger.frame_count

    assert count == 5, f"Expected 5 frames, got {count}"

    lines = Path(path).read_text().splitlines()
    assert len(lines) == 5

    for line in lines:
        rec = json.loads(line)
        assert rec["dataset_source"] == "LiveUser"
        assert rec["pipeline_type"] == "OpenFaceOnly"
        assert "raw_probabilities" in rec
        assert "ema_probabilities" in rec

    print("  PASSED\n")


def test_hybrid_logger() -> None:
    print("TEST 2: HybridGeometric logger")
    with Component1Logger("HybridGeometric", _LOG_DIR) as logger:
        _log_n_frames(logger, 5, base_stability=0.85)
        path = logger.jsonl_path
        count = logger.frame_count

    assert count == 5

    lines = Path(path).read_text().splitlines()
    for line in lines:
        rec = json.loads(line)
        assert rec["dataset_source"] == "LiveUser"
        assert rec["pipeline_type"] == "HybridGeometric"

    print("  PASSED\n")


def test_jsonl_schema() -> None:
    print("TEST 3: JSONL schema validation")
    required_keys = {
        "timestamp", "frame_index", "dataset_source", "pipeline_type",
        "raw_probabilities", "ema_probabilities", "frame_to_frame_change",
        "stability_index", "confidence_score", "quality_score",
    }

    for filename in ("openface_only.jsonl", "hybrid_geometric.jsonl"):
        path = Path(_LOG_DIR) / filename
        lines = path.read_text().splitlines()
        for line in lines:
            rec = json.loads(line)
            missing = required_keys - set(rec.keys())
            assert not missing, f"Missing keys in {filename}: {missing}"
        print(f"  {filename}: OK ({len(lines)} records)")

    print("  PASSED\n")


if __name__ == "__main__":
    np.random.seed(42)
    test_openface_logger()
    test_hybrid_logger()
    test_jsonl_schema()
    print("All tests passed.")
