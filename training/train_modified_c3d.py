from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from datasets.rgb_flow_dataset import (
    build_dataset,
    horizontal_flip_rgbflow,
)
from models.modified_c3d import (
    build_modified_c3d,
)

ROOT = Path(__file__).resolve().parents[1]

TRAIN_MANIFEST = Path(
    os.environ.get(
        "TRAIN_MANIFEST",
        str(ROOT / "datasets/splits/train_multiclip_rgbflow.csv"),
    )
)

VAL_MANIFEST = Path(
    os.environ.get(
        "VAL_MANIFEST",
        str(ROOT / "datasets/splits/val_multiclip_rgbflow.csv"),
    )
)

EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-5"))
SEED = int(os.environ.get("SEED", "42"))
HIDDEN_UNITS = int(os.environ.get("HIDDEN_UNITS", "1024"))
DROPOUT = float(os.environ.get("DROPOUT", "0.30"))

CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
RESULT_DIR = ROOT / "results/metrics"

BEST_WEIGHTS = (
    CHECKPOINT_DIR
    / "modified_c3d_best.weights.h5"
)

REPORT_PATH = (
    RESULT_DIR
    / "modified_c3d_validation.json"
)


def configure_runtime():
    keras.utils.set_random_seed(SEED)

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    print("TensorFlow:", tf.__version__)
    print("GPUs:", gpus)

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(
                    gpu,
                    True,
                )
            except RuntimeError:
                pass

        keras.mixed_precision.set_global_policy(
            "mixed_float16"
        )

        print(
            "Mixed precision:",
            keras.mixed_precision.global_policy(),
        )


def cast_labels_to_int(ds):
    return ds.map(
        lambda x, y: (
            x,
            tf.cast(y, tf.int32),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def collect_probabilities(model, dataset):
    y_true = []
    p_pos = []

    for videos, labels in dataset:
        probs = model(
            videos,
            training=False,
        ).numpy()

        flipped = horizontal_flip_rgbflow(
            videos
        )

        probs_flip = model(
            flipped,
            training=False,
        ).numpy()

        probs = (
            probs + probs_flip
        ) / 2.0

        y_true.extend(
            labels.numpy()
            .astype(np.int32)
            .tolist()
        )

        p_pos.extend(
            probs[:, 1].tolist()
        )

    return (
        np.asarray(
            y_true,
            dtype=np.int32,
        ),
        np.asarray(
            p_pos,
            dtype=np.float64,
        ),
    )


def aggregate_video_level(
    metadata,
    clip_probs,
    method,
):
    tmp = metadata.copy()
    tmp["prob"] = clip_probs

    labels = []
    probabilities = []

    for _, group in tmp.groupby(
        "source_video_path",
        sort=False,
    ):
        values = group[
            "prob"
        ].to_numpy(
            dtype=np.float64
        )

        if method == "mean":
            prob = float(
                values.mean()
            )
        elif method == "top2_mean":
            top = np.sort(
                values
            )[-min(2, len(values)):]
            prob = float(
                top.mean()
            )
        elif method == "max":
            prob = float(
                values.max()
            )
        else:
            raise ValueError(method)

        labels.append(
            int(
                group["label"].iloc[0]
            )
        )
        probabilities.append(prob)

    return (
        np.asarray(
            labels,
            dtype=np.int32,
        ),
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
    )


def metrics(y_true, y_prob, threshold):
    y_pred = (
        y_prob >= threshold
    ).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "auc": float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
    }


def threshold_search(
    y_true,
    y_prob,
):
    sorted_prob = np.sort(
        y_prob
    )

    mids = (
        (
            sorted_prob[:-1]
            + sorted_prob[1:]
        )
        / 2.0
        if len(sorted_prob) > 1
        else np.asarray([])
    )

    candidates = np.unique(
        np.concatenate(
            [
                np.asarray(
                    [0.5]
                ),
                sorted_prob,
                mids,
                np.linspace(
                    0.10,
                    0.90,
                    161,
                ),
            ]
        )
    )

    best = None

    for threshold in candidates:
        result = metrics(
            y_true,
            y_prob,
            float(threshold),
        )

        if (
            best is None
            or result["accuracy"]
            > best["accuracy"]
            or (
                result["accuracy"]
                == best["accuracy"]
                and result["f1"]
                > best["f1"]
            )
        ):
            best = result

    return best


def main():
    configure_runtime()

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_ds, train_meta = build_dataset(
        TRAIN_MANIFEST,
        BATCH_SIZE,
        True,
        SEED,
    )

    val_ds, val_meta = build_dataset(
        VAL_MANIFEST,
        BATCH_SIZE,
        False,
        SEED,
    )

    train_ds = cast_labels_to_int(
        train_ds
    )
    val_ds = cast_labels_to_int(
        val_ds
    )

    model = build_modified_c3d(
        hidden_units=HIDDEN_UNITS,
        dropout=DROPOUT,
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            clipnorm=1.0,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            ),
        ],
    )

    model.summary()

    print("Model: Modified C3D")
    print(
        "Train clips:",
        len(train_meta),
        "videos:",
        train_meta[
            "source_video_path"
        ].nunique(),
    )
    print(
        "Val clips:",
        len(val_meta),
        "videos:",
        val_meta[
            "source_video_path"
        ].nunique(),
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(BEST_WEIGHTS),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            restore_best_weights=False,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(
                LOG_DIR
                / "modified_c3d_training.csv"
            )
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    best_model = build_modified_c3d(
        hidden_units=HIDDEN_UNITS,
        dropout=DROPOUT,
    )

    best_model.load_weights(
        BEST_WEIGHTS
    )

    _, clip_probs = collect_probabilities(
        best_model,
        val_ds,
    )

    results = {}

    for method in (
        "mean",
        "top2_mean",
        "max",
    ):
        y_true, video_prob = aggregate_video_level(
            val_meta,
            clip_probs,
            method,
        )

        results[method] = {
            "at_0_5": metrics(
                y_true,
                video_prob,
                0.5,
            ),
            "best_validation_threshold": (
                threshold_search(
                    y_true,
                    video_prob,
                )
            ),
        }

    best_method = max(
        results,
        key=lambda m: (
            results[m]
            ["best_validation_threshold"]
            ["accuracy"],
            results[m]
            ["best_validation_threshold"]
            ["f1"],
        ),
    )

    report = {
        "model": "Modified C3D",
        "output_activation": "softmax",
        "output_units": 2,
        "loss": (
            "SparseCategoricalCrossentropy"
        ),
        "architecture": {
            "conv3d_layers": 4,
            "hidden_fc_layers": 1,
            "hidden_units": HIDDEN_UNITS,
            "input": [
                16,
                112,
                112,
                6,
            ],
        },
        "training_from_scratch": True,
        "train_videos": int(
            train_meta[
                "source_video_path"
            ].nunique()
        ),
        "validation_videos": int(
            val_meta[
                "source_video_path"
            ].nunique()
        ),
        "aggregation_results": (
            results
        ),
        "selected_validation_aggregation": (
            best_method
        ),
        "selected_validation_result": (
            results[best_method]
            ["best_validation_threshold"]
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
