from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from datasets.rgb_flow_dataset import (
    build_dataset,
    horizontal_flip_rgbflow,
)
from models.four_conv3d import build_model


def configure_runtime(seed):
    keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")

    print("TensorFlow:", tf.__version__)
    print("GPUs:", gpus)
    print("Seed:", seed)

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision:", keras.mixed_precision.global_policy())


def collect_probabilities(model, dataset, use_tta):
    probabilities = []

    for videos, _ in dataset:
        probs = model(videos, training=False).numpy()

        if use_tta:
            flipped = horizontal_flip_rgbflow(videos)
            flipped_probs = model(flipped, training=False).numpy()
            probs = (probs + flipped_probs) / 2.0

        probabilities.extend(probs[:, 1].tolist())

    return np.asarray(probabilities, dtype=np.float64)


def aggregate_video_level(metadata, clip_probabilities, method):
    temp = metadata.copy()
    temp["probability"] = clip_probabilities

    labels = []
    probabilities = []

    for _, group in temp.groupby("source_video_path", sort=False):
        values = group["probability"].to_numpy(dtype=np.float64)

        if method == "mean":
            probability = float(values.mean())
        elif method == "top2_mean":
            top = np.sort(values)[-min(2, len(values)):]
            probability = float(top.mean())
        elif method == "max":
            probability = float(values.max())
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        labels.append(int(group["label"].iloc[0]))
        probabilities.append(probability)

    return (
        np.asarray(labels, dtype=np.int32),
        np.asarray(probabilities, dtype=np.float64),
    )


def metric_result(y_true, y_probability, threshold):
    prediction = (y_probability >= threshold).astype(np.int32)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "auc": float(roc_auc_score(y_true, y_probability)),
        "precision": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, prediction, zero_division=0)
        ),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }


def best_threshold(y_true, y_probability):
    sorted_probability = np.sort(y_probability)

    midpoints = (
        (sorted_probability[:-1] + sorted_probability[1:]) / 2.0
        if len(sorted_probability) > 1
        else np.asarray([])
    )

    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([0.5]),
                sorted_probability,
                midpoints,
                np.linspace(0.05, 0.95, 181),
            ]
        )
    )

    best = None

    for threshold in candidates:
        result = metric_result(
            y_true,
            y_probability,
            float(threshold),
        )

        if (
            best is None
            or result["accuracy"] > best["accuracy"]
            or (
                result["accuracy"] == best["accuracy"]
                and result["f1"] > best["f1"]
            )
        ):
            best = result

    return best


def evaluate_video_level(model, val_dataset, val_metadata, use_tta):
    clip_probabilities = collect_probabilities(
        model,
        val_dataset,
        use_tta,
    )

    results = {}

    for method in ("mean", "top2_mean", "max"):
        y_true, y_probability = aggregate_video_level(
            val_metadata,
            clip_probabilities,
            method,
        )

        results[method] = {
            "at_0_5": metric_result(
                y_true,
                y_probability,
                0.5,
            ),
            "best_validation_threshold": best_threshold(
                y_true,
                y_probability,
            ),
        }

    selected_method = max(
        results,
        key=lambda method: (
            results[method]["best_validation_threshold"]["accuracy"],
            results[method]["best_validation_threshold"]["f1"],
            results[method]["best_validation_threshold"]["auc"],
        ),
    )

    selected = results[selected_method]["best_validation_threshold"]
    return results, selected_method, selected


class VideoMetricsCheckpoint(keras.callbacks.Callback):
    """Evaluate and save the best video-level validation checkpoint."""

    def __init__(self, val_dataset, val_metadata, use_tta, checkpoint_path):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_metadata = val_metadata
        self.use_tta = use_tta
        self.checkpoint_path = Path(checkpoint_path)
        self.best_accuracy = -1.0
        self.best_f1 = -1.0
        self.best_auc = -1.0
        self.best_epoch = 0
        self.best_method = None
        self.best_result = None

    def on_epoch_end(self, epoch, logs=None):
        _, method, result = evaluate_video_level(
            self.model,
            self.val_dataset,
            self.val_metadata,
            self.use_tta,
        )

        accuracy = result["accuracy"]
        f1 = result["f1"]
        auc = result["auc"]

        improved = (
            accuracy > self.best_accuracy + 1e-12
            or (
                abs(accuracy - self.best_accuracy) <= 1e-12
                and f1 > self.best_f1 + 1e-12
            )
            or (
                abs(accuracy - self.best_accuracy) <= 1e-12
                and abs(f1 - self.best_f1) <= 1e-12
                and auc > self.best_auc + 1e-12
            )
        )

        if improved:
            self.best_accuracy = accuracy
            self.best_f1 = f1
            self.best_auc = auc
            self.best_epoch = epoch + 1
            self.best_method = method
            self.best_result = result
            self.model.save_weights(self.checkpoint_path)

        print()
        print(f"Aggregation : {method}")
        print(f"Threshold   : {result['threshold']:.3f}")
        print(f"Accuracy    : {accuracy * 100:.2f}%")
        print(f"AUC         : {auc * 100:.2f}%")
        print(f"Precision   : {result['precision'] * 100:.2f}%")
        print(f"Recall      : {result['recall'] * 100:.2f}%")
        print(f"F1-score    : {f1 * 100:.2f}%")

        if improved:
            print(
                f"BEST SO FAR : {self.best_accuracy * 100:.2f}% "
                f"(epoch {self.best_epoch}, saved checkpoint)"
            )
        else:
            print(
                f"BEST SO FAR : {self.best_accuracy * 100:.2f}% "
                f"(epoch {self.best_epoch})"
            )


def build_optimizer(learning_rate, weight_decay):
    if hasattr(keras.optimizers, "AdamW"):
        return keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=1.0,
        )

    return keras.optimizers.experimental.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clipnorm=1.0,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the generic four-layer 3D-CNN on preprocessed "
            "binary video data."
        )
    )

    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--output-prefix", default="experiment")
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="Optional compatible checkpoint for transfer learning.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-units", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training augmentation.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable horizontal-flip test-time augmentation.",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=0,
        help=(
            "Optional patience for clip-level validation-accuracy early "
            "stopping. Use 0 to run every requested epoch."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    configure_runtime(args.seed)

    root = Path(__file__).resolve().parents[1]
    checkpoint_dir = root / "checkpoints"
    log_dir = root / "logs"
    result_dir = root / "results" / "metrics"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / (
        args.output_prefix + "_best.weights.h5"
    )
    result_path = result_dir / (
        args.output_prefix + "_validation.json"
    )
    csv_log_path = log_dir / (
        args.output_prefix + "_training.csv"
    )

    train_dataset, train_metadata = build_dataset(
        args.train_manifest,
        args.batch_size,
        training=True,
        seed=args.seed,
        augment=not args.no_augment,
    )

    val_dataset, val_metadata = build_dataset(
        args.val_manifest,
        args.batch_size,
        training=False,
        seed=args.seed,
        augment=False,
    )

    model = build_model(
        hidden_units=args.hidden_units,
        dropout=args.dropout,
    )

    if args.init_weights is not None:
        if not args.init_weights.exists():
            raise FileNotFoundError(args.init_weights)
        model.load_weights(args.init_weights)
        initialization = str(args.init_weights)
    else:
        initialization = "random initialization"

    model.compile(
        optimizer=build_optimizer(
            args.learning_rate,
            args.weight_decay,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        ],
    )

    print(
        "Train clips:",
        len(train_metadata),
        "videos:",
        train_metadata["source_video_path"].nunique(),
    )
    print(
        "Validation clips:",
        len(val_metadata),
        "videos:",
        val_metadata["source_video_path"].nunique(),
    )
    print("Initialization:", initialization)
    print("Training augmentation:", not args.no_augment)
    print("Horizontal-flip TTA:", not args.no_tta)

    model.summary()

    video_checkpoint = VideoMetricsCheckpoint(
        val_dataset,
        val_metadata,
        use_tta=not args.no_tta,
        checkpoint_path=checkpoint_path,
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        video_checkpoint,
        keras.callbacks.CSVLogger(str(csv_log_path)),
    ]

    if args.early_stopping > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=args.early_stopping,
                restore_best_weights=False,
                verbose=1,
            )
        )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    if not checkpoint_path.exists():
        raise RuntimeError("No video-level checkpoint was saved.")

    best_model = build_model(
        hidden_units=args.hidden_units,
        dropout=args.dropout,
    )
    best_model.load_weights(checkpoint_path)

    aggregation_results, selected_method, selected = (
        evaluate_video_level(
            best_model,
            val_dataset,
            val_metadata,
            use_tta=not args.no_tta,
        )
    )

    report = {
        "architecture": {
            "input": [16, 112, 112, 6],
            "conv3d_layers": 4,
            "hidden_fc_layers": 1,
            "hidden_units": args.hidden_units,
            "output_units": 2,
            "output_activation": "softmax",
        },
        "initialization": initialization,
        "epochs_requested": args.epochs,
        "checkpoint_selection": (
            "best video-level validation accuracy; ties by F1 then AUC"
        ),
        "best_epoch": video_checkpoint.best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "seed": args.seed,
        "training_augmentation": not args.no_augment,
        "horizontal_flip_tta": not args.no_tta,
        "train_videos": int(
            train_metadata["source_video_path"].nunique()
        ),
        "validation_videos": int(
            val_metadata["source_video_path"].nunique()
        ),
        "aggregation_results": aggregation_results,
        "selected_validation_aggregation": selected_method,
        "selected_validation_result": selected,
    }

    result_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("FINAL BEST VIDEO-LEVEL RESULT")
    print(f"Best epoch   : {video_checkpoint.best_epoch}")
    print(f"Aggregation  : {selected_method}")
    print(f"Accuracy     : {selected['accuracy'] * 100:.2f}%")
    print(f"AUC          : {selected['auc'] * 100:.2f}%")
    print(f"Precision    : {selected['precision'] * 100:.2f}%")
    print(f"Recall       : {selected['recall'] * 100:.2f}%")
    print(f"F1-score     : {selected['f1'] * 100:.2f}%")
    print(f"Threshold    : {selected['threshold']:.3f}")
    print("Saved checkpoint:", checkpoint_path)
    print("Saved result:", result_path)


if __name__ == "__main__":
    main()
