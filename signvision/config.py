from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    model_path: Path = Path(__file__).resolve().parent.parent / "models" / "gesture_recognizer.task"
    alphabet_model_path: Path = Path(__file__).resolve().parent.parent / "models" / "asl_classifier.tflite"
    alphabet_metadata_path: Path = Path(__file__).resolve().parent.parent / "models" / "asl_classifier.json"
    camera_index: int = 0
    score_threshold: float = 0.60
    max_results: int = 1
    stabilization_window: int = 5
    history_limit: int = 10
    camera_width: int = 960
    camera_height: int = 540
    frame_interval_ms: int = 15
