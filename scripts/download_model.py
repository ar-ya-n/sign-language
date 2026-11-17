"""Download Google's official MediaPipe Gesture Recognizer model."""
from pathlib import Path
from urllib.request import urlopen

URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
DEST = Path(__file__).resolve().parents[1] / "models" / "gesture_recognizer.task"

def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official model to {DEST}")
    with urlopen(URL, timeout=60) as source, DEST.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    print("Done.")

if __name__ == "__main__":
    main()
