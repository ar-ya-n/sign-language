"""Portable metadata for trained SignVision classifiers."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelMetadata:
    architecture: str
    labels: tuple[str, ...]
    input_size: tuple[int, int] = (224, 224)
    color_order: str = "RGB"
    normalization: str = "mobile-net-v2 [-1, 1]"
    dataset: str = "unspecified"
    trained: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["labels"] = list(self.labels)
        data["input_size"] = list(self.input_size)
        return data

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ModelMetadata":
        data = json.loads(path.read_text(encoding="utf-8"))
        data["labels"] = tuple(data["labels"])
        data["input_size"] = tuple(data.get("input_size", (224, 224)))
        return cls(**data)
