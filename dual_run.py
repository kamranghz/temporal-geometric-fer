"""
Dual-pipeline evaluation launcher.

Starts the live webcam demo in dual-logging mode so that both the
OpenFace-only and Hybrid-Geometric pipelines run simultaneously on the same
video stream.  Outputs are written to separate JSONL files for later
side-by-side comparison.

Usage::

    python dual_run.py [--log_dir logs/session_01]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dual-pipeline FER evaluation (OpenFace-only vs Hybrid)."
    )
    parser.add_argument(
        "--log_dir",
        default="logs/component_1",
        help="Directory for output JSONL logs (default: logs/component_1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CARE-AI  |  Dual-Pipeline Evaluation")
    print("=" * 72)
    print()
    print("Pipelines:")
    print("  A — OpenFace-only  →  openface_only.jsonl")
    print("  B — Hybrid Geometric  →  hybrid_geometric.jsonl")
    print()
    print(f"Log directory: {log_dir}")
    print()
    print("Controls (in the webcam window):")
    print("  1–5  Toggle pipeline components")
    print("  B    Baseline mode (all components off)")
    print("  F    Full system (all components on)")
    print("  R    Reset session metrics")
    print("  Q    Quit")
    print("=" * 72)

    cmd = [
        sys.executable,
        "live_webcam_unified.py",
        "--use_hybrid", "True",
        "--enable_evaluation",
        "--log_dir", str(log_dir),
    ]

    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()

    print()
    print("=" * 72)
    print("Session complete.")
    print(f"Logs saved to: {log_dir}/")
    print("  openface_only.jsonl")
    print("  hybrid_geometric.jsonl")
    print("=" * 72)


if __name__ == "__main__":
    main()
