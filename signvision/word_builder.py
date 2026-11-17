"""Turn stable alphabet predictions into words without accidental duplicates."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WordEvent:
    word: str
    committed: Optional[str] = None


class WordBuilder:
    """A small, deterministic state machine for alphabet-to-word entry.

    A letter is accepted once, then must leave the stable stream (or be
    replaced by another stable letter) before the same letter can be accepted
    again.  This makes words such as ``BOOK`` possible without frame-rate
    dependent repeats.
    """

    def __init__(self, max_length: int = 64) -> None:
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.max_length = max_length
        self._letters: list[str] = []
        self._armed = True
        self._last_stable: Optional[str] = None

    @property
    def word(self) -> str:
        return "".join(self._letters)

    def reset(self) -> None:
        self._letters.clear()
        self._armed = True
        self._last_stable = None

    def backspace(self) -> None:
        if self._letters:
            self._letters.pop()
        self._armed = True

    def space(self) -> WordEvent:
        if self._letters and self._letters[-1] != " ":
            self._letters.append(" ")
        self._armed = True
        return WordEvent(self.word, " ")

    def update(self, stable_letter: Optional[str]) -> WordEvent:
        letter = (stable_letter or "").strip().upper()
        if len(letter) != 1 or not letter.isalpha():
            self._armed = True
            self._last_stable = None
            return WordEvent(self.word)
        if letter != self._last_stable:
            self._armed = True
        self._last_stable = letter
        committed = None
        if self._armed and len(self._letters) < self.max_length:
            self._letters.append(letter)
            committed = letter
            self._armed = False
        return WordEvent(self.word, committed)
