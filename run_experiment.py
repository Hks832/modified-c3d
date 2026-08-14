from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def run_command(args):
    printable = " ".join(str(part) for part in args)
    print()
    print("$", printable)
    print()
    subprocess.run(args, cwd=ROOT, check=True)


def normalize_run_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise argparse.ArgumentTypeError(
            "Run name may contain only letters, numbers, '.', '_' and '-'."
        )
    return value


def subset_manifest(path: Path, videos_per_class: int):
    df = pd.read_csv(path)

    required = {"video_path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    selected = []

    for label in sorted(df["label"].unique()):
        part = df[df["label"] == label].head(videos_per_class)
        selected.append(part)

    out = pd.concat(selected, ignore_index=True)
    out.to_csv(path, index=False)

    print(
        f"Smoke subset: {path} -> "
        f"{len(out)} videos "
        f"({out['label'].value_counts().sort_index().to_dict()})"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete generic binary video-classification pipeline: "
            "split raw videos, create RGB+flow clips, train the four-layer "
            "3D-CNN, and report video-level validation metrics."
        )
    )

    parser.add_argument(
        "--negative-dir",
        required=True,
        type=Path,
        help="Directory containing class-0 videos.",
    )
    parser.add_argument(
        "--positive-dir",
        required=True,
        type=Path,
        help="Directory containing class-1 videos.",
    )
    parser.add_argument(
        "--run-name",
        default="experiment",
        type=normalize_run_name,
        help="Name used for generated outputs.",
    )
    parser.add_argument(
        "--runs-dir",
        default=ROOT / "runs",
        type=Path,
        help="Directory used for generated split files and RGB+flow features.",
    )
    parser.add_argument(
        "--val-size",
        default=0.20,
        type=float,
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-4,
        type=float,
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-5,
        type=float,
    )
    parser.add_argument(
        "--hidden-units",
        default=1024,
        type=int,
    )
    parser.add_argument(
        "--dropout",
        default=0.30,
        type=float,
    )
    parser.add_argument(
        "--init-weights",
        default=None,
        type=Path,
        help="Optional compatible checkpoint for transfer learning.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
    )
    parser.add_argument(
        "--early-stopping",
        default=0,
        type=int,
        help="Use 0 to run every requested epoch.",
    )
    parser.add_argument(
        "--flow-clip",
        default=12.0,
        type=float,
    )
    parser.add_argument(
        "--overwrite-features",
        action="store_true",
    )
    parser.add_argument(
        "--smoke-videos-per-class",
        default=0,
        type=int,
        help=(
            "If greater than zero, reduce each train/validation manifest to "
            "this many videos per class before preprocessing."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    negative_dir = args.negative_dir.expanduser().resolve()
    positive_dir = args.positive_dir.expanduser().resolve()
    runs_dir = args.runs_dir.expanduser().resolve()

    if not negative_dir.is_dir():
        raise NotADirectoryError(negative_dir)
    if not positive_dir.is_dir():
        raise NotADirectoryError(positive_dir)
    if not (0.0 < args.val_size < 1.0):
        raise ValueError("--val-size must be between 0 and 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    run_dir = runs_dir / args.run_name
    split_dir = run_dir / "splits"
    feature_dir = run_dir / "features"

    raw_train_manifest = split_dir / "train.csv"
    raw_val_manifest = split_dir / "val.csv"

    train_feature_manifest = split_dir / "train_rgbflow.csv"
    val_feature_manifest = split_dir / "val_rgbflow.csv"

    train_feature_dir = feature_dir / "train"
    val_feature_dir = feature_dir / "val"

    run_dir.mkdir(parents=True, exist_ok=True)

    print("Run directory:", run_dir)
    print("Negative class:", negative_dir)
    print("Positive class:", positive_dir)

    run_command(
        [
            sys.executable,
            "-m",
            "preprocessing.prepare_dataset",
            "--negative-dir",
            str(negative_dir),
            "--positive-dir",
            str(positive_dir),
            "--out-dir",
            str(split_dir),
            "--val-size",
            str(args.val_size),
            "--seed",
            str(args.seed),
        ]
    )

    if args.smoke_videos_per_class > 0:
        subset_manifest(
            raw_train_manifest,
            args.smoke_videos_per_class,
        )
        subset_manifest(
            raw_val_manifest,
            args.smoke_videos_per_class,
        )

    preprocess_common = [
        "--flow-clip",
        str(args.flow_clip),
    ]

    if args.overwrite_features:
        preprocess_common.append("--overwrite")

    run_command(
        [
            sys.executable,
            "-m",
            "preprocessing.rgb_flow_preprocessing",
            "--manifest",
            str(raw_train_manifest),
            "--output-dir",
            str(train_feature_dir),
            "--output-manifest",
            str(train_feature_manifest),
            *preprocess_common,
        ]
    )

    run_command(
        [
            sys.executable,
            "-m",
            "preprocessing.rgb_flow_preprocessing",
            "--manifest",
            str(raw_val_manifest),
            "--output-dir",
            str(val_feature_dir),
            "--output-manifest",
            str(val_feature_manifest),
            *preprocess_common,
        ]
    )

    training_command = [
        sys.executable,
        "-m",
        "training.train",
        "--train-manifest",
        str(train_feature_manifest),
        "--val-manifest",
        str(val_feature_manifest),
        "--output-prefix",
        args.run_name,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--hidden-units",
        str(args.hidden_units),
        "--dropout",
        str(args.dropout),
        "--seed",
        str(args.seed),
        "--early-stopping",
        str(args.early_stopping),
    ]

    if args.init_weights is not None:
        init_weights = args.init_weights.expanduser().resolve()
        if not init_weights.is_file():
            raise FileNotFoundError(init_weights)
        training_command.extend(
            [
                "--init-weights",
                str(init_weights),
            ]
        )

    if args.no_augment:
        training_command.append("--no-augment")

    if args.no_tta:
        training_command.append("--no-tta")

    run_command(training_command)

    print()
    print("Experiment complete.")
    print(
        "Checkpoint:",
        ROOT / "checkpoints" / f"{args.run_name}_best.weights.h5",
    )
    print(
        "Metrics:",
        ROOT / "results" / "metrics" / f"{args.run_name}_validation.json",
    )
    print(
        "Training log:",
        ROOT / "logs" / f"{args.run_name}_training.csv",
    )


if __name__ == "__main__":
    main()
