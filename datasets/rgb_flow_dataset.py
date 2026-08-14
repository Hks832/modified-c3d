from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

INPUT_SHAPE = (16, 112, 112, 6)


def load_manifest(path):
    df = pd.read_csv(Path(path))

    required = {"feature_path", "label", "source_video_path"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    if "processing_success" in df.columns:
        good = (
            df["processing_success"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        df = df[good].copy().reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"No usable rows found in manifest: {path}")

    return df


def _load_npz(path, label):
    if isinstance(path, np.ndarray):
        path = path.item()
    if isinstance(path, bytes):
        path = path.decode("utf-8")

    with np.load(str(path), allow_pickle=False) as saved:
        video = saved["video"].astype(np.float32)

    if video.shape != INPUT_SHAPE:
        raise ValueError(
            f"{path}: expected {INPUT_SHAPE}, got {video.shape}"
        )

    return video, np.int32(label)


def _tf_load(path, label):
    video, label = tf.numpy_function(
        _load_npz,
        [path, label],
        [tf.float32, tf.int32],
    )
    video.set_shape(INPUT_SHAPE)
    label.set_shape(())
    return video, label


def horizontal_flip_rgbflow(video):
    """Horizontal flip while correcting horizontal optical-flow direction."""
    flipped = tf.reverse(video, axis=[2])
    rgb = flipped[..., :3]
    flow_x = -flipped[..., 3:4]
    flow_y_and_magnitude = flipped[..., 4:]
    return tf.concat(
        [rgb, flow_x, flow_y_and_magnitude],
        axis=-1,
    )


def _augment(video, label):
    do_flip = tf.random.uniform(()) < 0.5
    video = tf.cond(
        do_flip,
        lambda: horizontal_flip_rgbflow(video),
        lambda: video,
    )

    pad = 5
    padded = tf.pad(
        video,
        paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]],
        mode="REFLECT",
    )

    offset_y = tf.random.uniform((), 0, 2 * pad + 1, dtype=tf.int32)
    offset_x = tf.random.uniform((), 0, 2 * pad + 1, dtype=tf.int32)

    video = padded[
        :,
        offset_y:offset_y + 112,
        offset_x:offset_x + 112,
        :,
    ]

    rgb = video[..., :3]
    flow = video[..., 3:]

    contrast = tf.random.uniform((), 0.85, 1.15)
    brightness = tf.random.uniform((), -0.06, 0.06)
    rgb = tf.clip_by_value(
        (rgb - 0.5) * contrast + 0.5 + brightness,
        0.0,
        1.0,
    )

    flow_gain = tf.random.uniform((), 0.90, 1.10)
    flow = flow * flow_gain
    flow_xy = tf.clip_by_value(flow[..., :2], -1.0, 1.0)
    magnitude = tf.clip_by_value(flow[..., 2:3], 0.0, 1.0)

    return tf.concat([rgb, flow_xy, magnitude], axis=-1), label


def build_dataset(
    manifest_path,
    batch_size,
    training,
    seed=42,
    augment=True,
):
    df = load_manifest(manifest_path)

    paths = df["feature_path"].astype(str).to_numpy()
    labels = df["label"].astype(np.int32).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            len(df),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(_tf_load, num_parallel_calls=tf.data.AUTOTUNE)

    if training and augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, df
