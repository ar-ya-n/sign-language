"""Optional TensorFlow Lite inference; dependencies are intentionally lazy."""
from pathlib import Path
from typing import Any

from .model_metadata import ModelMetadata


class TFLiteClassifier:
    """Run a user-trained .tflite model, never pretending a model is trained."""

    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Classifier not found: {model_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        self.metadata = ModelMetadata.load(metadata_path)
        try:
            import tensorflow as tf  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is optional. Install `tensorflow` (or tensorflow-cpu) "
                "to use trained MobileNetV2 inference."
            ) from exc
        self._interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]

    def predict(self, image: Any) -> tuple[str, float]:
        import numpy as np
        from PIL import Image

        width, height = self.metadata.input_size
        array = np.asarray(Image.fromarray(image).convert("RGB").resize((width, height)), dtype=np.float32)
        array = (array / 127.5) - 1.0
        array = np.expand_dims(array, 0).astype(self._input["dtype"])
        self._interpreter.set_tensor(self._input["index"], array)
        self._interpreter.invoke()
        scores = self._interpreter.get_tensor(self._output["index"])[0]
        index = int(np.argmax(scores))
        return self.metadata.labels[index], float(scores[index])
