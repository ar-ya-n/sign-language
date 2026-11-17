from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class GesturePrediction:
    label: str
    score: float
    category_name: Optional[str] = None
    timestamp_ms: int = 0
    landmarks: Optional[Any] = None
