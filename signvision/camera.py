import threading
import time
from typing import Callable, Optional

import cv2


class CameraWorker:
    """Capture and recognition loop; all camera work stays off Tk's UI thread."""

    def __init__(self, camera_index: int, width: int, height: int, on_frame: Callable, on_error: Callable):
        self.camera_index, self.width, self.height = camera_index, width, height
        self.on_frame, self.on_error = on_frame, on_error
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="signvision-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        self._thread = None

    def _run(self) -> None:
        capture = cv2.VideoCapture(self.camera_index)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Unable to open camera {self.camera_index}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            start = time.monotonic()
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera returned an empty frame")
                timestamp_ms = int((time.monotonic() - start) * 1000)
                self.on_frame(frame, timestamp_ms)
        except Exception as exc:
            if not self._stop.is_set():
                self.on_error(exc)
        finally:
            capture.release()

