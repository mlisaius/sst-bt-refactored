# SST-BT — Satellite Time-Series Barlow Twins

This repository is a clean, refactored implementation of a self-supervised learning pipeline for satellite time-series crop classification. The core idea is to pre-train an encoder on a large pool of *unlabelled* satellite pixels using the [Barlow Twins](https://arxiv.org/abs/2103.03230) objective, then transfer those learned representations to a supervised crop-type classification task with only a small amount of labelled data.

---

## What problem does this solve?

Labelling satellite pixels with crop types is expensive and time-consuming. However, there is an abundance of *unlabelled* satellite imagery. Barlow Twins lets the model learn rich temporal and spectral representations from that unlabelled data first — without any labels — and the resulting encoder is then fine-tuned on a small labelled dataset. This typically outperforms training a classifier from scratch on the labelled data alone.

---

## What is a d-pixel?

A *d-pixel* (date-pixel) is the core data unit. It represents a single geographic pixel observed across multiple satellite acquisition dates. Each d-pixel is stored as one row in a CSV:

```
[label, id, band_0_date_0, band_1_date_0, ..., cloud_flag_date_0, band_0_date_1, ...]
```

- **Column 0** — crop-type class label (integer; 0 in unlabelled data)
- **Column 1** — pixel ID (overwritten with sequential integers at load time)
- **Remaining columns** — spectral bands packed by date, with an optional cloud-quality flag as the last band of each date

The data modules compute NDVI and GCVI vegetation indices on the fly and append them as extra features before passing data to the encoder.

---

## Repository layout

```
refactored/
├── train.py                        # SSL pre-training entry point
├── evaluate.py                     # Random Forest evaluation of frozen embeddings
├── finetune.py                     # Two-phase supervised fine-tuning entry point
│
├── config/
│   └── FORPAPER_123_100.yaml       # Main configuration file (edit this to run experiments)
│
└── src/
    ├── datamodules/
    │   ├── btdmunlabelledVIs.py    # Dataset + DataModule for unlabelled SSL pre-training
    │   ├── btdmVIs.py              # Dataset + DataModule for labelled fine-tuning & evaluation
    │   └── __init__.py
    │
    └── models/
        ├── barlowtwins_unlabelled_VIs_speed.py   # Barlow Twins SSL model (used by train.py)
        ├── barlowtwins_VIs.py                    # Barlow Twins variant (used by evaluate.py)
        ├── classification.py                     # Two-phase fine-tuning model
        └── __init__.py
```

---

## The three-stage pipeline

### Stage 1 — SSL pre-training (`train.py`)

The encoder is trained on a large pool of unlabelled d-pixels. Two independent sparse temporal samples are drawn from each d-pixel (the "two views" required by Barlow Twins), and the loss pushes the encoder to produce similar representations for both views of the same pixel while decorrelating different dimensions of the output.

```bash
python train.py config/FORPAPER_123_100.yaml
```

Checkpoints are saved to `outputs/` after every epoch. Pick the one with the lowest `val_loss` as your pre-trained encoder — this path goes into `model.cont_epoch` in the config.

---

### Stage 2 — Fine-tuning (`finetune.py`)

Fine-tuning happens in two phases to make the most of a small labelled dataset.

**Phase 1 — Linear probe (encoder frozen)**
The pre-trained encoder weights are locked and only a small classification head is trained. This is fast, prevents the encoder from forgetting its SSL representations, and confirms that those representations are actually useful for crop classification before any expensive end-to-end training.

**Phase 2 — Full fine-tune (encoder unfrozen)**
The best probe checkpoint is loaded, the encoder is unfrozen, and the entire network is trained end-to-end. The encoder uses a 10× lower learning rate than the classification head so the pre-trained features adapt gradually rather than being wiped out.

```bash
python finetune.py config/FORPAPER_123_100.yaml
```

Phase 1 checkpoints are saved to `outputs/probe/` and Phase 2 checkpoints to `outputs/finetune/`. Both are monitored by `val_acc` — the best checkpoint is kept automatically.

---

### Stage 3 — RF evaluation (`evaluate.py`)

As a baseline (and sanity check), the frozen encoder from a pre-trained checkpoint is used to extract fixed embeddings for the labelled train and test sets. A Random Forest classifier is then fitted on the training embeddings and evaluated on the test embeddings. Results are written to `outputs/model_performance_output_VIs.txt`.

```bash
python evaluate.py config/FORPAPER_123_100.yaml
```

This is useful for quickly checking that the SSL pre-training produced meaningful representations before committing to the more expensive fine-tuning run.

---

## Configuration

All three scripts read from the same YAML config file. The key sections are:

```yaml
program:
    seed: 123           # RNG seed for reproducibility

datamodule:
    # Unlabelled data — used by train.py
    train_data_dir: "data/unlabelled_train.csv"
    val_data_dir:   "data/unlabelled_val.csv"
    test_data_dir:  "data/unlabelled_test.csv"

    # Labelled data — used by finetune.py and evaluate.py
    train_data: "data/labelled_train.csv"
    val_data:   "data/labelled_val.csv"
    test_data:  "data/labelled_test.csv"

    batch_size:      128
    n_ssamples:      15    # timesteps per sparse sample (the main augmentation)
    n_augmentations: 15    # how many stochastic copies of the dataset per epoch
    nbands:          10    # spectral bands per timestep (NDVI/GCVI added on top)
    cmask:           1     # cloud masking: 0=none, 1=drop cloudy dates, 2=artificial

model:
    encoder_no:      2     # 1=3-layer FC, 2=4-layer FC, 4=ResNet-18
    encoder_out_dim: 256   # encoder output / representation size
    z_dim:           256   # Barlow Twins projection head output size
    max_epochs:      100
    cont_epoch: "outputs/my-ssl-checkpoint.ckpt"   # pre-trained SSL checkpoint

finetune:
    num_classes:     13    # number of crop-type classes
    probe_epochs:    50    # epochs for Phase 1 (head only)
    finetune_epochs: 50    # epochs for Phase 2 (end-to-end)
    head_lr:         4.0e-4
    encoder_lr:      4.0e-5   # 10x lower than head_lr
```

The only things you normally need to change between experiments are the data paths, `cont_epoch` (the SSL checkpoint), and the `seed`.

---

## How augmentation works

The primary augmentation in this pipeline is *sparse temporal sampling*. Instead of presenting all available dates to the encoder, each training step randomly draws `n_ssamples` dates from the full time series. Two independent draws from the same d-pixel form the positive pair for the Barlow Twins loss. Setting `n_augmentations` to a value like 15 effectively replicas the dataset 15× per epoch, each replica drawing different random date subsets.

Cloud masking (`cmask`) can further increase augmentation diversity:
- `cmask=0` — use all dates as-is
- `cmask=1` — drop cloud-flagged dates before sampling (recommended)
- `cmask=2` — artificially remove `n_masked` random dates to simulate cloud gaps

---

## Model architecture

The encoder is a fully-connected network selected by `encoder_no`:

| `encoder_no` | Architecture |
|---|---|
| 1 | 3-layer FC with ReLU, hidden size 512 |
| 2 | 4-layer FC with LeakyReLU, hidden size 512 (default) |
| 4 | Adapted ResNet-18 (3×3 stem, custom FC head) |

The encoder output is fed through a two-layer projection head (with BatchNorm) during SSL pre-training. At fine-tuning time the projection head is discarded and replaced by:

```
BatchNorm1d → Linear(256, 512) → ReLU → Linear(512, 256) → Dropout(0.1) → Linear(256, num_classes)
```

---

## Dependencies

- Python 3.8
- `torch`, `torchvision`
- `pytorch-lightning`
- `torchmetrics`
- `scikit-learn`
- `numpy`, `pandas`
- `wandb` (for experiment tracking — runs are logged automatically)
- `scipy`, `pyyaml`

See `../requirements_stbt.txt` for the full pinned list.

---

## Metrics and logging

All runs log to [Weights & Biases](https://wandb.ai) under the project name set in `logger.log_dir`. Fine-tuning Phase 1 logs to `<log_dir>-probe` and Phase 2 to `<log_dir>-finetune` so the two phases are easy to compare side-by-side.

The RF evaluation baseline writes a plain-text summary to `outputs/model_performance_output_VIs.txt` containing weighted F1, balanced accuracy, overall accuracy, and Cohen's kappa.
