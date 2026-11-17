# SignVision AI - ASL Alphabet Prototype

SignVision AI is a real-time webcam application for **static American Sign
Language (ASL) fingerspelling**. It can display a trained alphabet classifier's
letter and build words one letter at a time. It is not a complete ASL
translation system: grammar, facial expressions, body signs, and dynamic
letters are outside this prototype.

The bundled `gesture_recognizer.task` is only a generic MediaPipe gesture demo.
It is a fallback and is **not** an ASL alphabet model. ASL recognition starts
only after you train and export `models/asl_classifier.tflite`.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\download_model.py
python main.py
```

TensorFlow is optional during normal use. Install it only for training:

```powershell
pip install tensorflow-cpu
```

## Recommended ASL datasets

Choose one dataset and keep its license/attribution with your project:

1. **ASL Alphabet (Kaggle)**  
   https://www.kaggle.com/datasets/grassknoted/asl-alphabet  
   This is a large static image dataset with folders such as `A` through `Z`
   plus non-letter classes. For this project, train on the 26 letter folders
   only. Kaggle download normally requires a Kaggle account/API token:

   ```powershell
   kaggle datasets download -d grassknoted/asl-alphabet -p data
   ```

2. **Sign Language MNIST (Kaggle)**  
   https://www.kaggle.com/datasets/datamunge/sign-language-mnist  
   This is a smaller CSV-based dataset of static ASL letters. It excludes
   dynamic `J` and `Z`, so it is useful for a 24-letter baseline, not complete
   A-Z recognition.

Do not mix datasets until their label definitions, image composition, and
licenses have been checked. Do not horizontally flip signs automatically:
orientation can affect the label.

## Prepare and inspect the dataset

The training script expects one folder per class:

```text
data/asl/
  A/
  B/
  ...
  Z/
```

Extract the Kaggle archive and copy only the 26 letter folders into
`data/asl/`. Then inspect it:

```powershell
python scripts/inspect_dataset.py data/asl
```

The inspection output is a factual file count only. It does not claim model
accuracy or signer-independent performance.

## Train the ASL classifier

```powershell
python scripts/train_classifier.py data/asl --epochs 10
```

The script uses a frozen ImageNet MobileNetV2 backbone, a global-average
pooling layer, dropout, and a softmax classification head. It writes:

- `models/asl_classifier.tflite`
- `models/asl_classifier.json` (labels and preprocessing metadata)

The webcam inference path uses the same 224x224 RGB preprocessing and
`[-1, 1]` normalization. Training output is the only source of accuracy
claims; this repository does not invent metrics. Evaluate on a held-out test
set before relying on the model.

## Using word mode

1. Start the application and select **Word** mode.
2. Click **START WORD ANALYSIS**.
3. Hold one static ASL letter until it stabilizes.
4. Move to a neutral pose before signing the next letter.
5. Use Pause, Resume, Add letter, Undo letter, Space, Clear word, and Finish
   word as needed.

The re-arm state prevents one held pose from being appended every frame and
allows deliberate repeated letters such as `LL` or `OO`.

## Current scope

- Static ASL alphabet classification only, after a locally trained model is
  installed.
- `J` and `Z` require motion in standard ASL and should not be claimed as
  supported by a still-image model unless temporal training and testing are
  added.
- No complete ASL sentence translation, grammar, facial-expression analysis,
  or signer-independent accuracy claim.
- No dataset or ASL classifier was trained in this workspace yet.

Run the available unit tests with:

```powershell
python -m unittest discover -s tests
```
