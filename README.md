# SignVision AI — V2 (practical ISL prototype)

Real-time desktop sign gesture recognition using **MediaPipe Tasks Gesture
Recognizer in VIDEO mode**, OpenCV, and Tkinter. Camera capture and inference
run on a daemon background thread; Tkinter updates are marshalled to the UI
thread. Predictions are stabilized over a rolling window and retained in a
bounded history.

V2 adds an honest alphabet/word workflow, deterministic repeated-letter
rearming (`BOOK` does not become `BOOOOK`), portable model metadata, optional
MobileNetV2 training/export, and dependency-free landmark feature preparation.
The bundled MediaPipe model is a generic gesture demo, not an ISL alphabet
model; therefore it must not be presented as ISL accuracy.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\download_model.py
python main.py
```

## Dataset and training

The intended image-folder layout is `data/realsign/<label>/*.jpg`. The
`scripts/download_dataset.py` command requires an official URL supplied by the
operator; it never guesses or fabricates a download. RealSign is referenced
as a CC0 source where the official release is used: retain its attribution and
license information with any redistribution. Inspect before training:

```powershell
python scripts/inspect_dataset.py data/realsign
python scripts/train_classifier.py data/realsign --epochs 10
```

TensorFlow is optional and is intentionally not in the default requirements.
Training prints the metrics from that local run and writes a `.tflite` model
plus JSON metadata. No model quality, dataset download, or validation result
is claimed by this repository until the operator runs and records it.

## Scope and limitations

Alphabet mode displays the stabilized classifier label. Word mode accepts
single-letter labels and provides space/backspace/clear controls. Dynamic
signs, continuous sentence translation, two-hand grammar, occlusion,
lighting, camera angle, and signer independence are **not solved** by this
prototype. The existing generic MediaPipe gestures remain available as a V1
fallback. Landmark features are a representation option, not a trained model.

The download script retrieves Google's official `gesture_recognizer.task`
binary; no fake or bundled model is used. A camera is required. Stop releases
the camera and model resources. Missing model, missing dependency, camera
failure, and frame/inference errors are surfaced in the UI.

Run core checks with `python -m unittest discover -s tests`.

Validation status: focused unit tests and Python syntax checks are the only
validation performed in this workspace; no ISL dataset was downloaded and no
classifier was trained here.
