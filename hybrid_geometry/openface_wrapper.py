from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.MLT import MLT


EMOTION_LABELS: List[str] = [
    "Neutral", "Happy", "Sad", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt",
]

# Input image size expected by EfficientNet-B0 backbone
_INPUT_SIZE: int = 224


class OpenFaceWrapper:
    """
    Inference wrapper for the OpenFace 3.0 multitask model.

    The underlying :class:`~model.MLT` network simultaneously predicts:
      * **Emotion** — 8-class softmax over AffectNet categories.
      * **Gaze**    — 2D gaze direction regression.
      * **AUs**     — 8 Action Unit regression scores.

    The wrapper handles preprocessing (face crop → normalise → to tensor) and
    postprocessing (softmax → numpy), exposing a clean single-call interface
    for the live inference pipeline.

    Args:
        model_path: Path to the pre-trained ``MTL_backbone.pth`` checkpoint.
        device:     Torch device string (``'cuda'`` or ``'cpu'``).
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: str = "weights/MTL_backbone.pth",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        self._model = MLT(
            base_model_name="tf_efficientnet_b0_ns",
            expr_classes=8,
            au_numbers=8,
        )

        ckpt_path = Path(model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {ckpt_path}\n"
                "Download weights from the link in weights/README.md"
            )

        state = torch.load(ckpt_path, map_location=self.device)
        self._model.load_state_dict(state)
        self._model.to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, face_bgr: np.ndarray) -> Dict[str, object]:
        """
        Run a single face crop through the model.

        Args:
            face_bgr: Cropped and aligned face image (BGR, any resolution).
                      Will be resized to 224 × 224 internally.

        Returns:
            Dictionary with keys:
                ``emotion_probs``  – Softmax probabilities, shape (8,).
                ``emotion_label``  – Argmax emotion label string.
                ``gaze``           – Gaze vector, shape (2,).
                ``au_scores``      – Action unit scores, shape (8,).
        """
        tensor = self._preprocess(face_bgr)

        with torch.no_grad():
            emotion_logits, gaze_out, au_out = self._model(tensor)

        emotion_probs = F.softmax(emotion_logits, dim=1).squeeze(0).cpu().numpy()
        gaze = gaze_out.squeeze(0).cpu().numpy()
        aus  = au_out.squeeze(0).cpu().numpy()

        return {
            "emotion_probs": emotion_probs,
            "emotion_label": EMOTION_LABELS[int(np.argmax(emotion_probs))],
            "gaze": gaze,
            "au_scores": aus,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, face_bgr: np.ndarray) -> torch.Tensor:
        """Resize, normalise, and convert to a (1, 3, 224, 224) tensor."""
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE))

        img = resized.astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        img = img.transpose(2, 0, 1)          # (H, W, C) → (C, H, W)

        return torch.from_numpy(img).unsqueeze(0).to(self.device)
