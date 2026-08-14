from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

FRAMES_PER_CLIP = 16
SIZE = 112
CHANNELS = 6
CLIP_CENTERS = (0.25, 0.50, 0.75)
HALF_WINDOW = 0.23
FLOW_CLIP = 12.0


def read_all_frames(video_path: Path):
    """Decode sequentially to avoid unreliable random frame seeking."""
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is not None and frame.size:
            frames.append(frame)

    capture.release()

    if len(frames) < 2:
        raise RuntimeError(
            f"Only {len(frames)} readable frame(s): {video_path}"
        )

    return frames


def sample_indices(frame_count, center_fraction):
    last = frame_count - 1
    start_fraction = max(0.0, center_fraction - HALF_WINDOW)
    end_fraction = min(1.0, center_fraction + HALF_WINDOW)

    indices = np.linspace(
        start_fraction * last,
        end_fraction * last,
        FRAMES_PER_CLIP,
    )

    return np.clip(
        np.rint(indices).astype(np.int32),
        0,
        last,
    )


def resize_rgb(frame):
    resized = cv2.resize(
        frame,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def build_clip(frames, indices, flow_clip):
    rgb = np.asarray(
        [resize_rgb(frames[int(index)]) for index in indices],
        dtype=np.float32,
    )

    output = np.zeros(
        (FRAMES_PER_CLIP, SIZE, SIZE, CHANNELS),
        dtype=np.float32,
    )
    output[..., :3] = rgb

    previous_gray = None

    for frame_index in range(FRAMES_PER_CLIP):
        gray = cv2.cvtColor(
            (rgb[frame_index] * 255.0).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )

        if previous_gray is None:
            flow = np.zeros((SIZE, SIZE, 2), dtype=np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            ).astype(np.float32)

        output[frame_index, ..., 3] = np.clip(
            flow[..., 0] / flow_clip,
            -1.0,
            1.0,
        )
        output[frame_index, ..., 4] = np.clip(
            flow[..., 1] / flow_clip,
            -1.0,
            1.0,
        )

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        output[frame_index, ..., 5] = np.clip(
            magnitude / flow_clip,
            0.0,
            1.0,
        )

        previous_gray = gray

    expected = (FRAMES_PER_CLIP, SIZE, SIZE, CHANNELS)
    if output.shape != expected:
        raise RuntimeError(f"Expected {expected}, got {output.shape}")

    return output.astype(np.float16)


def process_video(video_path, flow_clip):
    frames = read_all_frames(video_path)
    clips = []

    for clip_index, center in enumerate(CLIP_CENTERS):
        indices = sample_indices(len(frames), center)
        tensor = build_clip(frames, indices, flow_clip)
        clips.append((clip_index, tensor, indices))

    return clips, len(frames)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create three RGB+optical-flow clips per video. "
            "Each saved tensor has shape (16,112,112,6)."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--flow-clip", type=float, default=FLOW_CLIP)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_df = pd.read_csv(args.manifest)

    required = {"video_path", "label"}
    missing = required - set(source_df.columns)
    if missing:
        raise KeyError(
            f"Input manifest is missing columns: {sorted(missing)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []

    for video_index, row in enumerate(
        tqdm(
            source_df.itertuples(index=False),
            total=len(source_df),
            desc="RGB+Flow preprocessing",
        )
    ):
        source_path = Path(str(row.video_path))
        label = int(row.label)

        try:
            clips, total_frames = process_video(source_path, args.flow_clip)

            for clip_index, tensor, indices in clips:
                feature_path = (
                    args.output_dir
                    / f"{video_index:05d}_{source_path.stem}_clip{clip_index}.npz"
                )

                if args.overwrite or not feature_path.exists():
                    np.savez_compressed(feature_path, video=tensor)

                rows.append(
                    {
                        "feature_path": str(feature_path.resolve()),
                        "label": label,
                        "processing_success": True,
                        "source_video_path": str(source_path.resolve()),
                        "video_index": int(video_index),
                        "clip_index": int(clip_index),
                        "total_source_frames": int(total_frames),
                        "first_source_frame": int(indices[0]),
                        "last_source_frame": int(indices[-1]),
                        "error": "",
                    }
                )

        except Exception as exc:
            failures.append(
                {
                    "feature_path": "",
                    "label": label,
                    "processing_success": False,
                    "source_video_path": str(source_path.resolve()),
                    "video_index": int(video_index),
                    "clip_index": -1,
                    "total_source_frames": 0,
                    "first_source_frame": -1,
                    "last_source_frame": -1,
                    "error": str(exc),
                }
            )

    output_df = pd.DataFrame(rows + failures)
    output_df.to_csv(args.output_manifest, index=False)

    successful = pd.DataFrame(rows)
    successful_videos = (
        successful["source_video_path"].nunique()
        if not successful.empty
        else 0
    )

    print()
    print("Preprocessing complete")
    print(f"Videos successful: {successful_videos}/{len(source_df)}")
    print(f"Clips created: {len(successful)}")
    print(f"Expected clips: {successful_videos * 3}")
    print(f"Failures: {len(failures)}")
    print("Per-clip tensor: (16,112,112,6)")
    print("Saved manifest:", args.output_manifest)

    if failures:
        failure_path = (
            args.output_manifest.parent
            / (args.output_manifest.stem + "_failures.csv")
        )
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        print("Failure log:", failure_path)


if __name__ == "__main__":
    main()
