"""Mapping from MediaPipe labels to the prototype's custom meanings."""

GESTURE_LABELS = {
    "Open_Palm": ("Open palm", "HELLO"),
    "Closed_Fist": ("Closed fist", "STOP"),
    "Thumb_Up": ("Thumbs up", "YES"),
    "Thumb_Down": ("Thumbs down", "NO"),
    "Victory": ("Victory", "PEACE"),
    "Pointing_Up": ("Pointing up", "ONE"),
    "ILoveYou": ("I love you", "I LOVE YOU"),
}


def map_gesture(label: str | None, score: float, threshold: float) -> tuple[str, str]:
    """Return (gesture name, meaning), applying the confidence policy."""
    if not label:
        return "NO HAND DETECTED", "NO HAND DETECTED"
    if score < threshold:
        return "UNCERTAIN", "UNCERTAIN"
    return GESTURE_LABELS.get(label, (label.replace("_", " ").title(), "UNKNOWN"))


def display_name(label: str) -> str:
    return GESTURE_LABELS.get(label, (label.replace("_", " ").title(), ""))[0]
