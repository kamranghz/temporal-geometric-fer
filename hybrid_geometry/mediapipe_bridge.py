import cv2
import numpy as np
from typing import Dict, Optional, Tuple

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False


class MediaPipeBridge:
    """
    Thin wrapper around MediaPipe FaceMesh for 3D landmark extraction.

    Extracts up to 478 normalised 3D landmarks (x, y, z) per frame when
    ``refine_landmarks=True``, or 468 landmarks otherwise.  The refined
    mode includes iris landmarks, which improve gaze and blink estimation.

    The bridge also provides a lightweight head-pose estimate derived from
    the asymmetry between key facial anchor points — sufficient for the
    pose quality score used in :class:`~context_aware_filtering.ContextAwareFilter`
    without the overhead of a full PnP solve.

    Args:
        refine_landmarks:          Enable iris refinement (478 vs 468 pts).
        min_detection_confidence:  Minimum confidence for initial detection.
        min_tracking_confidence:   Minimum confidence to keep tracking.
    """

    def __init__(
        self,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not _HAS_MEDIAPIPE:
            raise ImportError(
                "MediaPipe is not installed. "
                "Install it with:  pip install mediapipe"
            )

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.num_landmarks = 478 if refine_landmarks else 468

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 3D landmarks from a BGR video frame.

        Args:
            frame_bgr: OpenCV BGR image of any resolution.

        Returns:
            Array of shape (N, 3) with normalised (x, y, z) coordinates,
            or ``None`` if no face is detected.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        lms = result.multi_face_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

    def extract_with_pose(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        """
        Extract landmarks and estimate head pose in a single call.

        Returns:
            Tuple of (landmarks, pose_info).  ``pose_info`` contains keys
            ``'yaw'``, ``'pitch'``, and ``'roll'`` in degrees.  All values
            are ``None`` when no face is detected.
        """
        landmarks = self.extract(frame_bgr)
        if landmarks is None:
            return None, {"yaw": None, "pitch": None, "roll": None}

        yaw, pitch = self._estimate_pose(landmarks)
        return landmarks, {"yaw": yaw, "pitch": pitch, "roll": 0.0}

    def close(self) -> None:
        """Release the underlying MediaPipe graph."""
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_pose(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Lightweight yaw/pitch estimate from facial symmetry.

        Uses the horizontal asymmetry between the nose tip and the face
        boundary anchors to approximate yaw, and the vertical position of
        the nose tip to approximate pitch.  This is a fast proxy — for
        precise pose angles use a full PnP solver with the 3D face model.
        """
        # Indices: 234=left cheek anchor, 454=right cheek anchor, 1=nose tip
        left_x   = landmarks[234, 0]
        right_x  = landmarks[454, 0]
        nose_x   = landmarks[1,   0]
        nose_y   = landmarks[1,   1]

        left_dist  = abs(nose_x - left_x)
        right_dist = abs(nose_x - right_x)
        # Scale ~90° over the full face width
        yaw   = (right_dist - left_dist) * 90.0
        pitch = (nose_y - 0.5) * 60.0

        return float(yaw), float(pitch)
