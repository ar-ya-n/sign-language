from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass(frozen=True)
class HistoryEntry:
    label: str
    score: float
    timestamp: datetime


class RecognitionHistory:
    def __init__(self, limit: int = 100) -> None:
        if limit < 1:
            raise ValueError("limit must be positive")
        self.limit = limit
        self._entries: List[HistoryEntry] = []

    def add(self, label: str, score: float, timestamp: datetime | None = None) -> None:
        self._entries.append(HistoryEntry(label, score, timestamp or datetime.now()))
        del self._entries[:-self.limit]

    def clear(self) -> None:
        self._entries.clear()

    def entries(self) -> List[HistoryEntry]:
        return list(self._entries)

