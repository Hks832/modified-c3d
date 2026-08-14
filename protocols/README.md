# Frozen experimental protocols

This directory stores version-controlled train/validation membership and experiment settings for exact reproducibility.

Each protocol directory must contain:

```text
train.csv
val.csv
experiment.json
```

`train.csv` and `val.csv` store `relative_path,label` rather than absolute machine-specific paths. A protocol may also include `source_group` when source-disjoint verification is required.

`experiment.json` records the preprocessing and training settings plus SHA-256 hashes of the split CSV files. `run_protocol.py` verifies those hashes, checks that every expected raw video exists under the supplied dataset root, verifies train/validation video overlap is zero, verifies source-group overlap when that column is available, preprocesses the exact membership, and trains with the stored settings.

A frozen protocol is run with:

```bash
python run_protocol.py \
  --protocol-dir protocols/<protocol-name> \
  --dataset-root "/path/to/extracted/dataset"
```

For transfer-learning protocols that require a source checkpoint:

```bash
python run_protocol.py \
  --protocol-dir protocols/<protocol-name> \
  --dataset-root "/path/to/extracted/dataset" \
  --init-weights "/path/to/source_checkpoint.weights.h5"
```

The raw datasets and large checkpoints are not stored in Git. Only the exact membership and configuration needed to reproduce the experiment are version controlled.

Existing local split manifests can be converted to a portable protocol with `tools/export_protocol.py`. Example:

```bash
python tools/export_protocol.py \
  --train-manifest /path/to/original_train_manifest.csv \
  --val-manifest /path/to/original_val_manifest.csv \
  --dataset-root "/path/to/dataset/root" \
  --output-dir protocols/<protocol-name> \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --hidden-units 1024 \
  --dropout 0.30 \
  --seed 42
```

For a transfer-learning protocol, add `--requires-init-weights` and an `--initialization-note` describing the required source checkpoint.
