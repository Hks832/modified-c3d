# Generic 4-Conv3D RGB+Flow Video Classifier

This repository contains a dataset-agnostic implementation of a binary video-classification pipeline. No dataset name, local dataset path, or dataset-specific class name is hard-coded in the training code.

## Architecture

Each model input is a `16 x 112 x 112 x 6` tensor:

- 3 RGB channels
- horizontal optical flow
- vertical optical flow
- optical-flow magnitude

The classifier uses:

1. Conv3D(64) + BatchNorm + Swish + MaxPool3D
2. Conv3D(128) + BatchNorm + Swish + MaxPool3D
3. Conv3D(256) + BatchNorm + Swish + MaxPool3D
4. Conv3D(512) + BatchNorm + Swish + MaxPool3D
5. GlobalAveragePooling3D + GlobalMaxPooling3D
6. one hidden Dense layer with 1024 units
7. Dropout
8. two-unit SoftMax output

The architecture is fixed at four Conv3D layers and one hidden fully connected layer.

## Environment

Python 3.10 is recommended.

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify TensorFlow before a long run:

```bash
python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY
```

The repository pins TensorFlow 2.15.0. TensorRT is not required.

## 1. Create a train/validation split

The input dataset only needs two class folders. Class 0 is the negative class and class 1 is the positive class.

```bash
python -m preprocessing.prepare_dataset \
  --negative-dir "/path/to/class_0" \
  --positive-dir "/path/to/class_1" \
  --out-dir datasets/splits \
  --val-size 0.20 \
  --seed 42
```

This creates:

```text
datasets/splits/train.csv
datasets/splits/val.csv
```

## 2. Preprocess the training split

The preprocessor sequentially decodes each video, which is more reliable than seeking directly to arbitrary frame numbers for videos with imperfect codec metadata.

```bash
python -m preprocessing.rgb_flow_preprocessing \
  --manifest datasets/splits/train.csv \
  --output-dir features/train \
  --output-manifest datasets/splits/train_rgbflow.csv
```

## 3. Preprocess the validation split

```bash
python -m preprocessing.rgb_flow_preprocessing \
  --manifest datasets/splits/val.csv \
  --output-dir features/val \
  --output-manifest datasets/splits/val_rgbflow.csv
```

For each successfully decoded video, the preprocessor creates three temporal clips. Each clip contains 16 frames and has shape `(16, 112, 112, 6)`.

## 4. Train from scratch

```bash
python -m training.train \
  --train-manifest datasets/splits/train_rgbflow.csv \
  --val-manifest datasets/splits/val_rgbflow.csv \
  --output-prefix run_from_scratch \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --hidden-units 1024 \
  --dropout 0.30 \
  --seed 42
```

The terminal prints clip-level Keras metrics and video-level validation metrics after every epoch.

Outputs are saved as:

```text
checkpoints/<output-prefix>_best.weights.h5
logs/<output-prefix>_training.csv
results/metrics/<output-prefix>_validation.json
```

## 5. Transfer learning

The same script supports transfer learning from any checkpoint produced by the same architecture.

```bash
python -m training.train \
  --train-manifest datasets/splits/train_rgbflow.csv \
  --val-manifest datasets/splits/val_rgbflow.csv \
  --init-weights /path/to/source_best.weights.h5 \
  --output-prefix transfer_run \
  --epochs 12 \
  --learning-rate 2e-5 \
  --seed 42
```

All layers are fine-tuned. The architecture does not change.

## Evaluation

Three video-level aggregation methods are evaluated:

- mean
- mean of the two highest clip probabilities
- maximum clip probability

The JSON result contains both the result at threshold `0.5` and the best threshold selected on the validation set.

A threshold selected on the validation set is a validation/model-selection result. It should not be described as untouched test-set performance.

## Reproducibility notes

- Keep the same split CSVs when comparing training configurations.
- Keep the seed fixed when reproducing a run.
- GPU kernels can still introduce small run-to-run variation.
- Do not rename or move generated feature files after preprocessing because the feature manifest stores their paths.
- Checkpoints are intentionally ignored by Git because they are large. A source checkpoint used for transfer learning should be supplied separately or reproduced first.

## Quick checks

Syntax check:

```bash
python -m py_compile \
  models/four_conv3d.py \
  datasets/rgb_flow_dataset.py \
  preprocessing/prepare_dataset.py \
  preprocessing/rgb_flow_preprocessing.py \
  training/train.py \
  smoke_test.py
```

Model smoke test:

```bash
python smoke_test.py
```
