from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

PATH_COLUMNS = (
    "video_path",
    "source_video_path",
    "path",
    "filepath",
    "file_path",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_path_column(df: pd.DataFrame) -> str:
    for column in PATH_COLUMNS:
        if column in df.columns:
            return column
    raise KeyError(
        f"Could not find a video-path column. Tried {PATH_COLUMNS}. "
        f"Present columns: {list(df.columns)}"
    )


def resolve_paths(df: pd.DataFrame, manifest_path: Path):
    path_column = find_path_column(df)
    paths = []

    for value in df[path_column].astype(str):
        path = Path(value).expanduser()
        if not path.is_absolute():
            candidates = [
                manifest_path.parent / path,
                Path.cwd() / path,
            ]
            path = next(
                (candidate for candidate in candidates if candidate.exists()),
                candidates[0],
            )
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)

    return path_column, paths


def portable_split(
    source_df: pd.DataFrame,
    source_manifest: Path,
    dataset_root: Path,
    split_name: str,
):
    if "label" not in source_df.columns:
        raise KeyError(f"{source_manifest} has no 'label' column")

    _, paths = resolve_paths(source_df, source_manifest)
    rows = []
    source_group_column = (
        "source_group" if "source_group" in source_df.columns else None
    )

    print(f"Hashing {split_name} videos for dataset fingerprinting...")

    for row_index, (path, label) in enumerate(
        zip(paths, source_df["label"].astype(int)),
        start=1,
    ):
        try:
            relative = path.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(
                f"Video is outside dataset root: {path}\n"
                f"dataset root: {dataset_root}"
            ) from exc

        item = {
            "relative_path": relative.as_posix(),
            "label": int(label),
            "file_sha256": sha256(path),
        }

        if source_group_column is not None:
            item["source_group"] = str(
                source_df.iloc[row_index - 1][source_group_column]
            )

        rows.append(item)

        if row_index % 50 == 0 or row_index == len(paths):
            print(f"  {row_index}/{len(paths)} {split_name} videos hashed")

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export existing train/validation manifests as a portable, "
            "version-controlled fixed experimental protocol."
        )
    )
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Root directory replaced by --dataset-root on another machine. "
            "If omitted, a common root is inferred."
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-units", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flow-clip", type=float, default=12.0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--early-stopping", type=int, default=0)
    parser.add_argument(
        "--requires-init-weights",
        action="store_true",
        help="Mark this protocol as requiring source checkpoint weights.",
    )
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help=(
            "Exact source checkpoint used by the experiment. Its SHA-256 is "
            "stored in experiment.json; the checkpoint itself is not copied."
        ),
    )
    parser.add_argument(
        "--initialization-note",
        default="random initialization",
        help="Human-readable description of the intended initialization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_manifest = args.train_manifest.expanduser().resolve()
    val_manifest = args.val_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest)

    _, train_paths = resolve_paths(train_df, train_manifest)
    _, val_paths = resolve_paths(val_df, val_manifest)

    if args.dataset_root is None:
        common = os.path.commonpath(
            [str(path) for path in train_paths + val_paths]
        )
        dataset_root = Path(common)
        if dataset_root.is_file():
            dataset_root = dataset_root.parent
    else:
        dataset_root = args.dataset_root.expanduser().resolve()

    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)

    init_weights = None
    init_hash = None
    if args.init_weights is not None:
        init_weights = args.init_weights.expanduser().resolve()
        if not init_weights.is_file():
            raise FileNotFoundError(init_weights)
        print("Hashing initialization checkpoint...")
        init_hash = sha256(init_weights)

    if args.requires_init_weights and init_weights is None:
        raise RuntimeError(
            "--requires-init-weights also requires --init-weights so the "
            "exact source checkpoint can be fingerprinted."
        )

    train_protocol = portable_split(
        train_df,
        train_manifest,
        dataset_root,
        "train",
    )
    val_protocol = portable_split(
        val_df,
        val_manifest,
        dataset_root,
        "validation",
    )

    train_paths_set = set(train_protocol["relative_path"])
    val_paths_set = set(val_protocol["relative_path"])
    video_overlap = sorted(train_paths_set & val_paths_set)
    if video_overlap:
        raise RuntimeError(
            f"Train/validation video overlap detected: {video_overlap[:10]}"
        )

    source_group_overlap = []
    if (
        "source_group" in train_protocol.columns
        and "source_group" in val_protocol.columns
    ):
        source_group_overlap = sorted(
            set(train_protocol["source_group"])
            & set(val_protocol["source_group"])
        )
        if source_group_overlap:
            raise RuntimeError(
                "Train/validation source-group overlap detected: "
                f"{source_group_overlap[:10]}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    config_path = output_dir / "experiment.json"

    train_protocol.to_csv(train_path, index=False)
    val_protocol.to_csv(val_path, index=False)

    config = {
        "protocol_version": 2,
        "portable_dataset_root_from_export_machine": str(dataset_root),
        "dataset_video_hashes": "sha256 per file in train.csv and val.csv",
        "train_videos": int(len(train_protocol)),
        "validation_videos": int(len(val_protocol)),
        "train_class_counts": {
            str(k): int(v)
            for k, v in train_protocol["label"].value_counts().sort_index().items()
        },
        "validation_class_counts": {
            str(k): int(v)
            for k, v in val_protocol["label"].value_counts().sort_index().items()
        },
        "video_overlap": 0,
        "source_group_overlap": len(source_group_overlap),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_units": args.hidden_units,
            "dropout": args.dropout,
            "seed": args.seed,
            "early_stopping": args.early_stopping,
            "augment": not args.no_augment,
            "tta": not args.no_tta,
        },
        "preprocessing": {
            "clips_per_video": 3,
            "frames_per_clip": 16,
            "height": 112,
            "width": 112,
            "channels": 6,
            "flow_clip": args.flow_clip,
        },
        "initialization": {
            "requires_init_weights": args.requires_init_weights,
            "note": args.initialization_note,
            "checkpoint_filename_at_export": (
                init_weights.name if init_weights is not None else None
            ),
            "checkpoint_sha256": init_hash,
        },
    }

    config["files"] = {
        "train_csv_sha256": sha256(train_path),
        "val_csv_sha256": sha256(val_path),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print()
    print("Portable protocol exported.")
    print("Dataset root used:", dataset_root)
    print("Train videos:", len(train_protocol))
    print("Validation videos:", len(val_protocol))
    print("Source-group overlap:", len(source_group_overlap))
    print("Train CSV:", train_path)
    print("Validation CSV:", val_path)
    print("Configuration:", config_path)
    print("Train SHA256:", config["files"]["train_csv_sha256"])
    print("Validation SHA256:", config["files"]["val_csv_sha256"])
    if init_hash:
        print("Initialization checkpoint SHA256:", init_hash)


if __name__ == "__main__":
    main()
