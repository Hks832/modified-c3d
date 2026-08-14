from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
    ".mpeg",
    ".mpg",
    ".wmv",
    ".m4v",
}


def collect(folder: Path, label: int):
    files = sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    return [
        {
            "video_path": str(path.resolve()),
            "label": int(label),
        }
        for path in files
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a stratified binary train/validation split. "
            "Class 0 is the negative class and class 1 is the positive class."
        )
    )
    parser.add_argument(
        "--positive-dir",
        required=True,
        type=Path,
        help="Directory containing class-1 videos.",
    )
    parser.add_argument(
        "--negative-dir",
        required=True,
        type=Path,
        help="Directory containing class-0 videos.",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets/splits",
        type=Path,
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    rows = (
        collect(args.negative_dir, 0)
        + collect(args.positive_dir, 1)
    )
    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No supported video files were found.")

    counts = df["label"].value_counts().sort_index()
    if set(counts.index.tolist()) != {0, 1}:
        raise RuntimeError(
            "Both classes must contain at least one supported video."
        )

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=df["label"],
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.out_dir / "train.csv"
    val_path = args.out_dir / "val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print("Total videos:", len(df))
    print("Class counts:")
    print(counts)
    print()
    print("Train videos:", len(train_df))
    print(train_df["label"].value_counts().sort_index())
    print()
    print("Validation videos:", len(val_df))
    print(val_df["label"].value_counts().sort_index())
    print()
    print("Saved:")
    print(train_path)
    print(val_path)


if __name__ == "__main__":
    main()
