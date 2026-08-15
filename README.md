

This repository contains a dataset-agnostic binary video-classification pipeline. No dataset name, local dataset path, or dataset-specific class name is hard-coded in the model or training code.

## Architecture

Each input clip has shape `16 x 112 x 112 x 6`:

- 3 RGB channels
- horizontal optical flow
- vertical optical flow
- optical-flow magnitude

The classifier uses four Conv3D blocks (64, 128, 256, 512 filters), BatchNorm, Swish, MaxPool3D, global average + max pooling, one hidden Dense layer with 1024 units, Dropout, and a two-unit SoftMax output.

## Quickest reproducibility workflow

After cloning the repository and installing the Python dependencies, a complete experiment can be run from the two raw class folders with one command. The runner automatically creates the train/validation split, preprocesses every video into three RGB+flow clips, trains the model, evaluates video-level metrics, and saves the checkpoint and result JSON.

```bash
python run_experiment.py \
  --negative-dir "/path/to/class_0" \
  --positive-dir "/path/to/class_1" \
  --run-name verification \
  --epochs 20 \
  --seed 42
```

The command above performs the complete pipeline:

```text
raw class folders
    -> stratified train/validation split
    -> sequential video decoding
    -> three RGB+optical-flow clips per video
    -> 4-Conv3D training
    -> video-level aggregation and threshold evaluation
    -> checkpoint + CSV training log + JSON metrics
```

Generated preprocessing data is kept under `runs/<run-name>/`. Training outputs are saved as:

```text
checkpoints/<run-name>_best.weights.h5
logs/<run-name>_training.csv
results/metrics/<run-name>_validation.json
```

For a quick end-to-end functionality test before a long run, use a small number of videos per class:

```bash
python run_experiment.py \
  --negative-dir "/path/to/class_0" \
  --positive-dir "/path/to/class_1" \
  --run-name smoke_test \
  --epochs 1 \
  --batch-size 2 \
  --smoke-videos-per-class 2 \
  --no-augment \
  --no-tta
```

A smoke-test score is only a software functionality check; it is not a reported experimental result.

## Environment

Python 3.10 is recommended.

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Linux systems using an NVIDIA GPU, an optional GPU dependency file is also provided:

```bash
pip install -r requirements-gpu.txt
```

TensorRT is not required.

Verify the model implementation:

```bash
python smoke_test.py
```

Expected architecture checks include input shape `(None, 16, 112, 112, 6)`, output shape `(None, 2)`, four Conv3D layers, and two Dense layers (one hidden layer plus the SoftMax output layer).


The same one-command runner supports a compatible source checkpoint:

```bash
python run_experiment.py \
  --negative-dir "/path/to/class_0" \
  --positive-dir "/path/to/class_1" \
  --run-name transfer_verification \
  --init-weights "/path/to/source_best.weights.h5" \
  --epochs 12 \
  --learning-rate 2e-5 \
  --seed 42
```

All layers are fine-tuned; the architecture does not change.

## Evaluation

After every epoch, the training script evaluates three video-level aggregation methods:

- mean clip probability
- mean of the two highest clip probabilities
- maximum clip probability

The best video-level validation checkpoint is selected by validation accuracy, with F1 and AUC used as tie-breakers. The result JSON stores the aggregation results, the selected validation aggregation method, threshold, accuracy, AUC, precision, recall, and F1 score.



## Manual workflow (advanced)

The one-command runner simply orchestrates the reusable generic modules. They can also be run separately when needed.

Create a split:

```bash
python -m preprocessing.prepare_dataset \
  --negative-dir "/path/to/class_0" \
  --positive-dir "/path/to/class_1" \
  --out-dir datasets/splits \
  --val-size 0.20 \
  --seed 42
```

Preprocess training and validation videos:

```bash
python -m preprocessing.rgb_flow_preprocessing \
  --manifest datasets/splits/train.csv \
  --output-dir features/train \
  --output-manifest datasets/splits/train_rgbflow.csv

python -m preprocessing.rgb_flow_preprocessing \
  --manifest datasets/splits/val.csv \
  --output-dir features/val \
  --output-manifest datasets/splits/val_rgbflow.csv
```

Train directly from the generated manifests:

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

## Reproducibility notes

- Use the same raw dataset version and the same seed when reproducing a generated split.
- Preserve any official or source-disjoint split required by a specific experiment rather than regenerating it with a simple random split.
- GPU kernels can introduce small run-to-run variation.
- Generated features can require substantial disk space and preprocessing time.
- Checkpoints are intentionally ignored by Git because of their size. A transfer-learning checkpoint must be provided separately or reproduced first.
- The raw datasets are not distributed in this repository; the user must obtain them from their original sources.

## Quick syntax check

```bash
python -m py_compile \
  run_experiment.py \
  models/four_conv3d.py \
  datasets/rgb_flow_dataset.py \
  preprocessing/prepare_dataset.py \
  preprocessing/rgb_flow_preprocessing.py \
  training/train.py \
  smoke_test.py
```
