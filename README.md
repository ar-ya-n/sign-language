# SignVision AI

Real-time desktop sign gesture recognition using **MediaPipe Tasks Gesture
Recognizer in VIDEO mode**, OpenCV, and Tkinter. Camera capture and inference
run on a daemon background thread; Tkinter updates are marshalled to the UI
thread. Predictions are stabilized over a rolling window and retained in a
bounded history.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\download_model.py
python main.py
```

The download script retrieves Google's official `gesture_recognizer.task`
binary; no fake or bundled model is used. A camera is required. Stop releases
the camera and model resources. Missing model, missing dependency, camera
failure, and frame/inference errors are surfaced in the UI.

Run core checks with `python -m unittest discover -s tests`.
