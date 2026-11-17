import threading
import tkinter as tk
from pathlib import Path

import cv2
from PIL import Image

from .camera import CameraWorker
from .config import AppConfig
from .history import RecognitionHistory
from .mapping import map_gesture
from .recognizer import GestureRecognizer
from .stabilizer import PredictionStabilizer
from .ui import SignVisionUI


class SignVisionApp:
    def __init__(self, root: tk.Tk, config: AppConfig | None = None):
        self.config = config or AppConfig()
        self.root = root
        self.history = RecognitionHistory(self.config.history_limit)
        self.stabilizer = PredictionStabilizer(self.config.stabilization_window)
        self.recognizer = None
        self.worker = None
        self._lock = threading.Lock()
        self.ui = SignVisionUI(root, self.start, self.stop, self.clear_history, self.close)

    def start(self) -> None:
        if self.worker:
            return
        try:
            self.recognizer = GestureRecognizer(self.config.model_path, self.config.score_threshold, self.config.max_results)
            self.worker = CameraWorker(self.config.camera_index, self.config.camera_width, self.config.camera_height,
                                       self._on_frame, self._on_error)
            self.worker.start()
            self.ui.set_status("Running")
        except Exception as exc:
            self.worker = None
            self.ui.show_error(str(exc))

    def stop(self) -> None:
        worker, self.worker = self.worker, None
        if worker: worker.stop()
        if self.recognizer: self.recognizer.close()
        self.recognizer = None
        self.ui.set_status("Stopped")

    def _on_frame(self, frame, timestamp_ms: int) -> None:
        try:
            prediction = self.recognizer.recognize(frame, timestamp_ms)
            display = cv2.flip(frame, 1)
            if prediction and prediction.landmarks:
                self._draw_landmarks(display, prediction.landmarks)
            stable_raw = self.stabilizer.update(prediction.label if prediction else None)
            gesture, meaning = map_gesture(
                stable_raw, prediction.score if prediction else 0.0, self.config.score_threshold
            )
            if meaning not in {"NO HAND DETECTED", "UNCERTAIN"} and (
                not self.history.entries() or self.history.entries()[-1].label != meaning
            ):
                self.history.add(meaning, prediction.score if prediction else 0.0)
            image = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            self.root.after(0, self._update_ui, image, gesture, meaning, prediction.score if prediction else 0)
        except Exception as exc:
            self._on_error(exc)

    @staticmethod
    def _draw_landmarks(frame, landmarks) -> None:
        height, width = frame.shape[:2]
        # The preview is mirrored for a natural selfie view, so mirror the
        # normalized landmark x-coordinate as well.
        points = [
            (int((1.0 - point.x) * width), int(point.y * height))
            for point in landmarks
        ]
        connections = (
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
            (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
        )
        for start, end in connections:
            cv2.line(frame, points[start], points[end], (255, 190, 0), 2)
        for point in points:
            cv2.circle(frame, point, 5, (0, 255, 180), -1)

    def _update_ui(self, image, gesture, meaning, score) -> None:
        self.ui.set_image(image)
        self.ui.set_result(gesture, meaning, score)
        self.ui.refresh_history(self.history.entries())

    def _on_error(self, exc: Exception) -> None:
        self.root.after(0, lambda: self.ui.show_error(str(exc)))
        self.stop()

    def clear_history(self) -> None:
        self.history.clear(); self.stabilizer.reset(); self.ui.refresh_history([])

    def close(self) -> None:
        self.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    SignVisionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
