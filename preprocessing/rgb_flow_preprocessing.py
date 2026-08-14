from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

N_CLIPS = 3
FRAMES_PER_CLIP = 16
SIZE = 112
CLIP_CENTERS = (0.25, 0.50, 0.75)
HALF_WINDOW = 0.23

PATH_COLUMNS = ("video_path","path","filepath","file_path","source_path")
LABEL_COLUMNS = ("label","class","target")

def find_column(df, candidates, kind):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find {kind} column. Tried {candidates}. Present columns: {list(df.columns)}")

def resolve_video_path(value, manifest_path):
    path = Path(str(value))
    if path.is_absolute():
        return path
    for candidate in (ROOT / path, manifest_path.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return path

def read_frame(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {index}")
    return frame

def resize_rgb(frame):
    resized = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0

def build_clip(cap, total_frames, center_fraction, flow_clip=12.0):
    last = max(total_frames - 1, 0)
    start_fraction = max(0.0, center_fraction - HALF_WINDOW)
    end_fraction = min(1.0, center_fraction + HALF_WINDOW)
    start = int(round(start_fraction * last))
    end = int(round(end_fraction * last))

    indices = np.rint(
        np.linspace(start, max(start, end), FRAMES_PER_CLIP)
    ).astype(np.int32)

    rgb_frames = [resize_rgb(read_frame(cap, int(idx))) for idx in indices]
    rgb = np.asarray(rgb_frames, dtype=np.float32)

    flow_x = np.zeros((FRAMES_PER_CLIP, SIZE, SIZE, 1), dtype=np.float32)
    flow_y = np.zeros_like(flow_x)
    flow_mag = np.zeros_like(flow_x)

    prev_gray = cv2.cvtColor((rgb[0] * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    for t in range(1, FRAMES_PER_CLIP):
        curr_gray = cv2.cvtColor((rgb[t] * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        fx = np.clip(flow[..., 0] / float(flow_clip), -1.0, 1.0)
        fy = np.clip(flow[..., 1] / float(flow_clip), -1.0, 1.0)
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mag = np.clip(mag / float(flow_clip), 0.0, 1.0)

        flow_x[t, ..., 0] = fx
        flow_y[t, ..., 0] = fy
        flow_mag[t, ..., 0] = mag
        prev_gray = curr_gray

    clip = np.concatenate([rgb, flow_x, flow_y, flow_mag], axis=-1)

    expected = (FRAMES_PER_CLIP, SIZE, SIZE, 6)
    if clip.shape != expected:
        raise RuntimeError(f"Expected {expected}, got {clip.shape}")

    return clip.astype(np.float16), indices

def process_video(video_path, flow_clip):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame count for {video_path}")

    clips = []
    try:
        for clip_index, center in enumerate(CLIP_CENTERS):
            tensor, indices = build_clip(
                cap, total, center, flow_clip=flow_clip
            )
            clips.append((clip_index, tensor, indices))
    finally:
        cap.release()

    return clips, total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--flow-clip", type=float, default=12.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_df = pd.read_csv(args.manifest)
    path_col = find_column(source_df, PATH_COLUMNS, "video path")
    label_col = find_column(source_df, LABEL_COLUMNS, "label")

    if args.limit is not None:
        source_df = source_df.iloc[:args.limit].copy().reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    successful_videos = 0

    for video_index, row in source_df.iterrows():
        source_path = resolve_video_path(row[path_col], args.manifest)
        label = int(row[label_col])

        try:
            clips, total_frames = process_video(source_path, args.flow_clip)
            video_rows = []

            for clip_index, tensor, indices in clips:
                feature_path = (
                    args.output_dir /
                    f"{video_index:05d}_{source_path.stem}_clip{clip_index}.npz"
                )

                if args.overwrite or not feature_path.exists():
                    np.savez_compressed(feature_path, video=tensor)

                video_rows.append({
                    "feature_path": str(feature_path),
                    "label": label,
                    "processing_success": True,
                    "source_video_path": str(source_path),
                    "video_index": int(video_index),
                    "clip_index": int(clip_index),
                    "total_source_frames": int(total_frames),
                    "first_source_frame": int(indices[0]),
                    "last_source_frame": int(indices[-1]),
                    "error": "",
                })

            rows.extend(video_rows)
            successful_videos += 1

        except Exception as exc:
            rows.append({
                "feature_path": "",
                "label": label,
                "processing_success": False,
                "source_video_path": str(source_path),
                "video_index": int(video_index),
                "clip_index": -1,
                "total_source_frames": -1,
                "first_source_frame": -1,
                "last_source_frame": -1,
                "error": str(exc),
            })

        if ((video_index + 1) % 25 == 0) or (video_index + 1 == len(source_df)):
            print(f"{video_index + 1}/{len(source_df)} videos | successful={successful_videos}")

    out = pd.DataFrame(rows)
    out.to_csv(args.output_manifest, index=False)

    successful_clips = int(
        out["processing_success"].astype(str).str.lower().isin(["true","1","yes"]).sum()
    )

    print()
    print("Multi-clip RGB+Flow preprocessing")
    print(f"Videos successful: {successful_videos}/{len(source_df)}")
    print(f"Clip rows successful: {successful_clips}")
    print("Expected clips per successful video: 3")
    print("Per-clip tensor: (16,112,112,6)")

if __name__ == "__main__":
    main()
