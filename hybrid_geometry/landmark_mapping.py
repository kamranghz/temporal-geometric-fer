import numpy as np
from typing import Optional


# Mapping from the 98-point OpenFace format to MediaPipe's 478-landmark indices.
# Each key is an OpenFace index (0–97); each value is the corresponding
# MediaPipe landmark index to sample from a (478, 3) array.
_OF98_TO_MP478: dict[int, int] = {
    # Face contour (0–32)
    0: 234,  1: 93,   2: 132,  3: 58,   4: 172,  5: 136,  6: 150,  7: 149,
    8: 176,  9: 148, 10: 152, 11: 377, 12: 400, 13: 378, 14: 379, 15: 365,
   16: 397, 17: 288, 18: 361, 19: 323, 20: 454, 21: 356, 22: 389, 23: 251,
   24: 284, 25: 332, 26: 297, 27: 338, 28:  10, 29: 109, 30:  67, 31: 103, 32:  54,
    # Right eyebrow (33–41)
   33:  70, 34:  63, 35: 105, 36:  66, 37: 107, 38:  55, 39:  65, 40:  52, 41:  53,
    # Left eyebrow (42–50)
   42: 282, 43: 295, 44: 285, 45: 300, 46: 293, 47: 334, 48: 296, 49: 336, 50: 285,
    # Nose (51–59)
   51: 168, 52:   6, 53: 197, 54: 195, 55:   5, 56:   4, 57:   1, 58:  19, 59:  94,
    # Right eye (60–67)
   60:  33, 61: 246, 62: 161, 63: 160, 64: 159, 65: 158, 66: 157, 67: 173,
    # Left eye (68–75)
   68: 263, 69: 249, 70: 390, 71: 373, 72: 374, 73: 380, 74: 381, 75: 382,
    # Outer lips (76–87)
   76:  61, 77: 185, 78:  40, 79:  39, 80:  37, 81:   0, 82: 267, 83: 269,
   84: 270, 85: 409, 86: 291, 87: 375,
    # Inner lips (88–97)
   88:  78, 89: 191, 90:  80, 91:  81, 92:  82, 93:  13, 94: 312, 95: 311,
   96: 310, 97: 415,
}

# Pre-built index array for fast vectorised lookup
_SOURCE_INDICES = np.array([_OF98_TO_MP478[i] for i in range(98)], dtype=np.int32)


class LandmarkMapper:
    """
    Maps MediaPipe's 478 landmarks to OpenFace's 98-point layout.

    OpenFace 3.0 was trained on the 98-point face alignment standard used by
    WFLW and COFW datasets.  When using MediaPipe as the geometry front-end,
    the 478 raw points must be projected into this canonical 98-point space
    before being passed to the OpenFace model.

    The mapping is a fixed index-selection matrix::

        L_98 = X_478[idx],   idx ∈ ℤ^{98}

    No interpolation is performed; each of the 98 target points corresponds
    directly to one of the 478 MediaPipe landmarks.

    Args:
        scale_to_pixels: If ``True``, multiply (x, y) by image (width, height)
                         to convert from normalised [0, 1] to pixel coordinates.
                         The z channel is left normalised.
        image_size:      (width, height) tuple used when ``scale_to_pixels=True``.
    """

    def __init__(
        self,
        scale_to_pixels: bool = False,
        image_size: tuple[int, int] = (640, 480),
    ) -> None:
        self.scale_to_pixels = scale_to_pixels
        self.image_size = image_size

    def map(self, landmarks_478: np.ndarray) -> np.ndarray:
        """
        Select the 98 OpenFace-compatible points from a MediaPipe result.

        Args:
            landmarks_478: Array of shape (478, 3) or (478, 2) from
                           :class:`~hybrid_geometry.MediaPipeBridge`.

        Returns:
            Array of shape (98, 3) or (98, 2) in the same coordinate system.

        Raises:
            ValueError: If the input does not have at least 478 rows.
        """
        if landmarks_478.shape[0] < 478:
            raise ValueError(
                f"Expected at least 478 landmarks, got {landmarks_478.shape[0]}"
            )

        mapped = landmarks_478[_SOURCE_INDICES].copy()  # (98, *)

        if self.scale_to_pixels and mapped.shape[1] >= 2:
            w, h = self.image_size
            mapped[:, 0] *= w
            mapped[:, 1] *= h

        return mapped

    def map_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Vectorised mapping for a batch of frames.

        Args:
            batch: Array of shape (T, 478, C).

        Returns:
            Array of shape (T, 98, C).
        """
        return batch[:, _SOURCE_INDICES, :]
