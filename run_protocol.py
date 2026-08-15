from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(parts):
    print()
    print("$", " ".join(str(part) for part in parts))
    print()
    subprocess.run(parts, cwd=ROOT, check=True)


def load_protocol_split(path: Path, dataset_root: Path, split_name: str):
    df = pd.read_csv(path)
    required = {"relative_path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    rows = []
    missing_files = []
    hash_mismatches = []
    verify_hashes = "file_sha256" in df.columns

    if verify_hashes:
        print(f"Verifying {split_name} dataset fingerprints...")

    for index, row in enumerate(df.itertuples(index=False), start=1):
        video_path = (dataset_root / str(row.relative_path)).resolve()

        if not video_path.is_file():
            missing_files.append(str(video_path))
        elif verify_hashes:
            expected_hash = str(row.file_sha256).strip().lower()
            actual_hash = sha256(video_path)
            if actual_hash != expected_hash:
                hash_mismatches.append(
                    (str(row.relative_path), expected_hash, actual_hash)
                )

        item = {
            "video_path": str(video_path),
            "label": int(row.label),
        }
        if hasattr(row, "source_group"):
            item["source_group"] = str(row.source_group)
        rows.append(item)

        if verify_hashes and (index % 50 == 0 or index == len(df)):
            print(f"  {index}/{len(df)} {split_name} videos verified")

    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(
            f"{len(missing_files)} protocol video(s) were not found under "
            f"dataset root {dataset_root}. First missing paths:\n{preview}"
        )

    if hash_mismatches:
        preview = "\n".join(
            f"{path}\n expected {expected}\n actual   {actual}"
            for path, expected, actual in hash_mismatches[:5]
        )
        raise RuntimeError(
            f"{len(hash_mismatches)} {split_name} video SHA256 mismatch(es). "
            f"The local dataset is not byte-identical to the frozen protocol.\n"
            f"First mismatch(es):\n{preview}"
        )

    return pd.DataFrame(rows), verify_hashes


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce a frozen experimental protocol using version-controlled "
            "split membership, dataset fingerprints and experiment settings."
        )
    )
    parser.add_argument("--protocol-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--init-weights", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    protocol_dir = args.protocol_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    runs_dir = args.runs_dir.expanduser().resolve()

    train_protocol_path = protocol_dir / "train.csv"
    val_protocol_path = protocol_dir / "val.csv"
    config_path = protocol_dir / "experiment.json"

    for path in (train_protocol_path, val_protocol_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_files = config.get("files", {})

    actual_train_hash = sha256(train_protocol_path)
    actual_val_hash = sha256(val_protocol_path)
    expected_train_hash = expected_files.get("train_csv_sha256")
    expected_val_hash = expected_files.get("val_csv_sha256")

    if expected_train_hash and actual_train_hash != expected_train_hash:
        raise RuntimeError("train.csv SHA256 does not match experiment.json")
    if expected_val_hash and actual_val_hash != expected_val_hash:
        raise RuntimeError("val.csv SHA256 does not match experiment.json")

    train_df, train_data_hashed = load_protocol_split(
        train_protocol_path,
        dataset_root,
        "train",
    )
    val_df, val_data_hashed = load_protocol_split(
        val_protocol_path,
        dataset_root,
        "validation",
    )

    train_paths = set(train_df["video_path"])
    val_paths = set(val_df["video_path"])
    overlap = train_paths & val_paths
    if overlap:
        raise RuntimeError(
            f"Frozen protocol has {len(overlap)} train/validation video overlaps."
        )

    source_overlap = set()
    if (
        "source_group" in train_df.columns
        and "source_group" in val_df.columns
    ):
        source_overlap = (
            set(train_df["source_group"]) & set(val_df["source_group"])
        )
        if source_overlap:
            raise RuntimeError(
                "Frozen protocol has train/validation source-group overlap: "
                f"{sorted(source_overlap)[:10]}"
            )

    expected_train = config.get("train_videos")
    expected_val = config.get("validation_videos")
    if expected_train is not None and len(train_df) != int(expected_train):
        raise RuntimeError(
            f"Expected {expected_train} train videos, found {len(train_df)}"
        )
    if expected_val is not None and len(val_df) != int(expected_val):
        raise RuntimeError(
            f"Expected {expected_val} validation videos, found {len(val_df)}"
        )

    training = config.get("training", {})
    preprocessing = config.get("preprocessing", {})
    initialization = config.get("initialization", {})

    epochs = (
        args.epochs
        if args.epochs is not None
        else int(training.get("epochs", 20))
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(training.get("batch_size", 16))
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(training.get("learning_rate", 1e-4))
    )
    weight_decay = float(training.get("weight_decay", 1e-5))
    hidden_units = int(training.get("hidden_units", 1024))
    dropout = float(training.get("dropout", 0.30))
    seed = int(training.get("seed", 42))
    early_stopping = int(training.get("early_stopping", 0))
    augment = bool(training.get("augment", True))
    tta = bool(training.get("tta", True))
    flow_clip = float(preprocessing.get("flow_clip", 12.0))

    requires_init = bool(initialization.get("requires_init_weights", False))
    expected_init_hash = initialization.get("checkpoint_sha256")
    init_weights = None
    actual_init_hash = None

    if args.init_weights is not None:
        init_weights = args.init_weights.expanduser().resolve()
        if not init_weights.is_file():
            raise FileNotFoundError(init_weights)
        print("Verifying initialization checkpoint...")
        actual_init_hash = sha256(init_weights)
        if expected_init_hash and actual_init_hash != expected_init_hash:
            raise RuntimeError(
                "Initialization checkpoint SHA256 does not match the frozen "
                "protocol.\n"
                f"Expected: {expected_init_hash}\n"
                f"Actual:   {actual_init_hash}"
            )
    elif requires_init:
        raise RuntimeError(
            "This protocol requires the exact compatible source checkpoint. "
            "Pass it with --init-weights."
        )

    if requires_init and not expected_init_hash:
        print(
            "WARNING: this older protocol requires initialization weights but "
            "does not contain a checkpoint SHA256. Re-export it with the "
            "current tools for exact checkpoint verification."
        )

    run_name = args.run_name or protocol_dir.name
    run_dir = runs_dir / run_name
    split_dir = run_dir / "splits"
    feature_dir = run_dir / "features"
    split_dir.mkdir(parents=True, exist_ok=True)

    raw_train_manifest = split_dir / "train.csv"
    raw_val_manifest = split_dir / "val.csv"
    train_feature_manifest = split_dir / "train_rgbflow.csv"
    val_feature_manifest = split_dir / "val_rgbflow.csv"

    train_df.to_csv(raw_train_manifest, index=False)
    val_df.to_csv(raw_val_manifest, index=False)

    metadata = {
        "protocol_dir": str(protocol_dir),
        "dataset_root": str(dataset_root),
        "protocol_version": config.get("protocol_version"),
        "train_protocol_sha256": actual_train_hash,
        "val_protocol_sha256": actual_val_hash,
        "dataset_file_hashes_verified": bool(
            train_data_hashed and val_data_hashed
        ),
        "source_group_overlap": len(source_overlap),
        "train_videos": len(train_df),
        "validation_videos": len(val_df),
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "hidden_units": hidden_units,
            "dropout": dropout,
            "seed": seed,
            "early_stopping": early_stopping,
            "augment": augment,
            "tta": tta,
        },
        "preprocessing": {
            "flow_clip": flow_clip,
        },
        "initialization": {
            "requires_init_weights": requires_init,
            "provided_init_weights": str(init_weights) if init_weights else None,
            "expected_checkpoint_sha256": expected_init_hash,
            "actual_checkpoint_sha256": actual_init_hash,
            "note": initialization.get("note", ""),
        },
    }
    (run_dir / "protocol_run.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("Frozen protocol verified.")
    print("Protocol:", protocol_dir)
    print("Dataset root:", dataset_root)
    print("Train videos:", len(train_df))
    print("Validation videos:", len(val_df))
    print("Video overlap: 0")
    if "source_group" in train_df.columns and "source_group" in val_df.columns:
        print("Source-group overlap: 0")
    print("Train CSV SHA256:", actual_train_hash)
    print("Validation CSV SHA256:", actual_val_hash)
    print(
        "Raw video SHA256 verification:",
        "passed" if train_data_hashed and val_data_hashed else "not present",
    )
    if actual_init_hash:
        print("Initialization checkpoint SHA256:", actual_init_hash)

    preprocess_extra = ["--flow-clip", str(flow_clip)]
    if args.overwrite_features:
        preprocess_extra.append("--overwrite")

    run_command(
        [
            sys.executable,
            "-m",
            "preprocessing.rgb_flow_preprocessing",
            "--manifest",
            str(raw_train_manifest),
            "--output-dir",
            str(feature_dir / "train"),
            "--output-manifest",
            str(train_feature_manifest),
            *preprocess_extra,
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
            str(feature_dir / "val"),
            "--output-manifest",
            str(val_feature_manifest),
            *preprocess_extra,
        ]
    )

    command = [
        sys.executable,
        "-m",
        "training.train",
        "--train-manifest",
        str(train_feature_manifest),
        "--val-manifest",
        str(val_feature_manifest),
        "--output-prefix",
        run_name,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
        "--hidden-units",
        str(hidden_units),
        "--dropout",
        str(dropout),
        "--seed",
        str(seed),
        "--early-stopping",
        str(early_stopping),
    ]

    if init_weights is not None:
        command.extend(["--init-weights", str(init_weights)])
    if not augment:
        command.append("--no-augment")
    if not tta:
        command.append("--no-tta")

    run_command(command)

    print()
    print("Protocol experiment complete.")
    print("Run metadata:", run_dir / "protocol_run.json")
    print("Checkpoint:", ROOT / "checkpoints" / f"{run_name}_best.weights.h5")
    print(
        "Metrics:",
        ROOT / "results" / "metrics" / f"{run_name}_validation.json",
    )


if __name__ == "__main__":
    main()
