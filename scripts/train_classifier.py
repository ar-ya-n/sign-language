"""Train/export an optional MobileNetV2 image classifier from class folders.

This script is deliberately explicit: it only reports metrics from a run the
user actually starts, and writes metadata alongside the exported model.
"""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models") / "isl_classifier.tflite")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "Optional training requires TensorFlow. Install `tensorflow` or `tensorflow-cpu`; "
            "the existing MediaPipe demo does not require it."
        ) from exc
    if not args.dataset.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {args.dataset}")
    image_size = (224, 224)
    train = tf.keras.utils.image_dataset_from_directory(
        args.dataset, validation_split=0.2, subset="training", seed=42, image_size=image_size
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        args.dataset, validation_split=0.2, subset="validation", seed=42, image_size=image_size
    )
    labels = tuple(train.class_names)
    base = tf.keras.applications.MobileNetV2(input_shape=(*image_size, 3), include_top=False, weights="imagenet")
    base.trainable = False
    model = tf.keras.Sequential([base, tf.keras.layers.GlobalAveragePooling2D(),
                                  tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(len(labels), activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=validation, epochs=args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    args.output.write_bytes(converter.convert())
    from signvision.model_metadata import ModelMetadata
    ModelMetadata("MobileNetV2", labels, dataset=str(args.dataset), trained=True,
                  notes="Metrics above are from this local training run.").save(args.output.with_suffix(".json"))
    print(f"Exported {args.output} and metadata. No accuracy is claimed beyond the run output.")


if __name__ == "__main__":
    main()
