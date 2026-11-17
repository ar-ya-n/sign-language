"""Dependency-free landmark feature preparation for optional classifiers."""
from typing import Iterable, Sequence


def normalize_landmarks(points: Iterable[object]) -> list[float]:
    """Translate 21 (x, y, z) hand landmarks to wrist-relative features."""
    values = [(float(p.x), float(p.y), float(getattr(p, "z", 0.0))) for p in points]
    if not values:
        return []
    wrist = values[0]
    return [component - wrist[index % 3] for point in values for index, component in enumerate(point)]


def flatten_landmark_rows(rows: Sequence[Iterable[object]]) -> list[list[float]]:
    return [normalize_landmarks(row) for row in rows]
