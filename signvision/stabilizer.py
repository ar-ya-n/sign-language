from collections import Counter, deque
from typing import Deque, Optional


class PredictionStabilizer:
    """Return a label only after it dominates a rolling window."""

    def __init__(self, window_size: int = 5, min_ratio: float = 0.6) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0 < min_ratio <= 1:
            raise ValueError("min_ratio must be in (0, 1]")
        self.window_size = window_size
        self.min_ratio = min_ratio
        self._labels: Deque[Optional[str]] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._labels.clear()

    def update(self, label: Optional[str]) -> Optional[str]:
        self._labels.append(label)
        valid = [item for item in self._labels if item]
        if not valid:
            return None
        winner, count = Counter(valid).most_common(1)[0]
        # Use the configured window as the denominator so one early sample
        # cannot immediately become a "stable" prediction.
        return winner if count / self.window_size >= self.min_ratio else None
