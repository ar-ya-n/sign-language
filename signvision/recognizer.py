from pathlib import Path
from typing import Any, Optional

import cv2

from .models import GesturePrediction


class GestureRecognizer:
    """Thin adapter around MediaPipe Tasks Gesture Recognizer in VIDEO mode."""

    def __init__(self, model_path: Path, score_threshold: float = 0.5, max_results: int = 1) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run `python scripts/download_model.py` first."
            )
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("mediapipe is not installed; run pip install -r requirements.txt") from exc
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=score_threshold,
            min_hand_presence_confidence=score_threshold,
            min_tracking_confidence=score_threshold,
        )
        self._mp = mp
        self._recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

    def recognize(self, frame: Any, timestamp_ms: int) -> Optional[GesturePrediction]:
        """Recognize an OpenCV BGR frame, returning the top gesture or None."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._recognizer.recognize_for_video(rgb, timestamp_ms)
        if not result.gestures or not result.gestures[0]:
            return None
        category = result.gestures[0][0]
        return GesturePrediction(
            label=category.category_name or category.display_name or "Unknown",
            score=float(category.score or 0.0),
            category_name=category.display_name,
            timestamp_ms=timestamp_ms,
            landmarks=result.hand_landmarks[0] if result.hand_landmarks else None,
        )

    def close(self) -> None:
        close = getattr(self._recognizer, "close", None)
        if close:
            close()
