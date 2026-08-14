from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"}

def collect(folder: Path, label: int):
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    return [{"video_path": str(p.resolve()), "label": label} for p in files]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", required=True, type=Path)
    ap.add_argument("--nofights", required=True, type=Path)
    ap.add_argument("--out-dir", default="datasets/splits", type=Path)
    ap.add_argument("--val-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = collect(args.nofights, 0) + collect(args.fights, 1)
    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No videos found.")

    counts = df["label"].value_counts().sort_index()
    print("Found videos:")
    print(counts)

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=df["label"],
        shuffle=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.out_dir / "train_internal.csv"
    val_path = args.out_dir / "val_internal.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print()
    print("Train:", len(train_df))
    print(train_df["label"].value_counts().sort_index())
    print("Validation:", len(val_df))
    print(val_df["label"].value_counts().sort_index())
    print()
    print("Saved:")
    print(train_path)
    print(val_path)

if __name__ == "__main__":
    main()
