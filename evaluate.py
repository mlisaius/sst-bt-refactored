"""Evaluation entry point for the Barlow Twins satellite time-series model.

This script loads a YAML configuration file, restores a pre-trained encoder
from a Lightning checkpoint, extracts frozen embeddings for the train and test
splits, fits a Random Forest classifier on the train embeddings, and reports
classification metrics on the test split.

Usage:
    python evaluate.py <path_to_config.yaml>

Example:
    python evaluate.py config/FORPAPER_123_100.yaml
"""

import sys
import yaml
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)

from src.datamodules import create_test_dataloaderVIb as create_test_dataloader
from src.models import BarlowTwinsVIs as BarlowTwins


def save_representations(dataloader, n_ssamples, batch_size, this_encoder, save_reps):
    """Extract encoder embeddings from a dataloader and optionally save them to CSV.

    Iterates over a single batch (the dataloader is expected to yield one batch
    containing the entire dataset), runs the frozen encoder on both augmented
    views, and concatenates the results along with their labels and pixel IDs.

    Args:
        dataloader: A PyTorch DataLoader yielding ``((img1, img2), label, id)``
            tuples.
        n_ssamples (int): Number of sparse temporal samples per d-pixel.  Used
            only for constructing the output filename when ``save_reps=True``.
        batch_size (int): Batch size.  Used only for constructing the output
            filename when ``save_reps=True``.
        this_encoder (nn.Module): Frozen encoder extracted from the loaded
            checkpoint.  Called with ``torch.no_grad`` semantics via
            ``.detach()``.
        save_reps (bool): If ``True``, write the concatenated embedding
            DataFrame to a CSV file in the current working directory and return
            it.  If ``False``, return the DataFrame without writing to disk.

    Returns:
        pd.DataFrame: A DataFrame whose columns are ``[label, id,
            emb_0, ..., emb_N]``, with one row per (pixel, view) pair.
    """
    for batch in dataloader:
        (img1, img2), label, id = batch

        embeddings1_np = this_encoder(img1).detach().numpy()
        embeddings2_np = this_encoder(img2).detach().numpy()
        label_np = label.detach().numpy()
        id_np = id.detach().numpy()

        embeddings1_labels_np = np.c_[label_np, id_np, embeddings1_np]
        embeddings2_labels_np = np.c_[label_np, id_np, embeddings2_np]
        embeddings_full = np.concatenate(
            (embeddings1_labels_np, embeddings2_labels_np), axis=0
        )
        df = pd.DataFrame(embeddings_full)

        if save_reps:
            name = (
                "representations_ssamples"
                + str(n_ssamples)
                + "_batchsize"
                + str(batch_size)
                + "_traindata.csv"
            )
            df.to_csv(name, index=False)

        return df


def main(conf):
    """Load a checkpoint, extract embeddings, and evaluate with a Random Forest.

    Reads train and test CSV files from the config, builds dataloaders, extracts
    frozen encoder embeddings for both splits, trains a Random Forest on the
    train embeddings, and writes classification metrics to
    ``outputs/model_performance_output_VIs.txt``.

    Args:
        conf (dict): Parsed YAML configuration.  Expected top-level keys are
            ``program``, ``datamodule``, ``model``, and ``trainer``.  The
            ``datamodule`` section must contain ``train_data``, ``test_data``,
            ``n_augmentations``, ``num_workers``, ``n_ssamples``, ``n_masked``,
            ``nbands``, and ``cmask``.  The ``model`` section must contain
            ``encoder_no``, ``encoder_out_dim``, ``batch_size``, ``z_dim``,
            ``max_epochs``, and ``cont_epoch`` (checkpoint path).
    """
    # Build the model architecture so Lightning can restore weights from
    # the checkpoint.  num_training_samples is not used during inference.
    model = BarlowTwins(
        encoder_no=conf["model"]["encoder_no"],
        encoder_out_dim=conf["model"]["encoder_out_dim"],
        num_training_samples=conf["model"]["num_training_samples"],
        batch_size=conf["datamodule"]["batch_size"],
        z_dim=conf["model"]["z_dim"],
        max_epochs=conf["model"]["max_epochs"],
        n_ssamples=conf["datamodule"]["n_ssamples"],
        nbands=conf["datamodule"]["nbands"],
    )

    model_loaded = model.load_from_checkpoint(
        checkpoint_path=conf["model"]["cont_epoch"]
    )
    this_encoder = model_loaded.encoder

    train_data = np.asarray(pd.read_csv(conf["datamodule"]["train_data"]))
    test_data = np.asarray(pd.read_csv(conf["datamodule"]["test_data"]))
    print("Train data shape:", train_data.shape[0])
    print("Test data shape:", test_data.shape[0])
    print("Loaded data.")

    # Create dataloaders that return the full dataset in a single batch so that
    # save_representations can process all embeddings at once.
    train_dataloader = create_test_dataloader(
        train_data,
        n_augmentations=conf["datamodule"]["n_augmentations"],
        batch_size=train_data.shape[0],
        num_workers=conf["datamodule"]["num_workers"],
        n_ssamples=conf["datamodule"]["n_ssamples"],
        n_masked=0,
        maskset=[],
        nbands=conf["datamodule"]["nbands"],
        cmask=conf["datamodule"]["cmask"],
        shuffle_val=True,
    )

    test_dataloader = create_test_dataloader(
        test_data,
        n_augmentations=conf["datamodule"]["n_augmentations"],
        batch_size=test_data.shape[0],
        num_workers=conf["datamodule"]["num_workers"],
        n_ssamples=conf["datamodule"]["n_ssamples"],
        n_masked=0,
        maskset=[],
        nbands=conf["datamodule"]["nbands"],
        cmask=conf["datamodule"]["cmask"],
        shuffle_val=False,
    )

    train_representations = save_representations(
        train_dataloader,
        n_ssamples=conf["datamodule"]["n_ssamples"],
        batch_size=train_data.shape[0],
        this_encoder=this_encoder,
        save_reps=False,
    )

    test_representations = save_representations(
        test_dataloader,
        n_ssamples=conf["datamodule"]["n_ssamples"],
        batch_size=test_data.shape[0],
        this_encoder=this_encoder,
        save_reps=False,
    )

    print("Train representations shape:", train_representations.shape)

    # Columns: [label, id, emb_0, ..., emb_N].  Embeddings start at column 2.
    X_tr = np.asarray(train_representations)[:, 2 : train_representations.shape[1]]
    y_tr = np.asarray(train_representations)[:, 0]
    X = np.asarray(test_representations)[:, 2 : train_representations.shape[1]]
    y = np.asarray(test_representations)[:, 0]

    clf = RandomForestClassifier(random_state=0, n_estimators=500)
    clf.fit(X_tr, y_tr)
    print("Model trained. Now classifying.")
    y_predictrf = clf.predict(X)

    score_rf = f1_score(y, y_predictrf, average="weighted")
    average_accurate = accuracy_score(y, y_predictrf, normalize=True)
    balanced_accuracy = balanced_accuracy_score(y, y_predictrf)
    kappa = cohen_kappa_score(y, y_predictrf)

    print(
        "Weighted F1:", score_rf,
        "Balanced Accuracy:", balanced_accuracy,
        "Average Accuracy:", average_accurate,
        "Kappa:", kappa,
    )

    with open("outputs/model_performance_output_VIs.txt", "w") as text_file:
        text_file.write(
            "Weighted F1: {0}; Balanced Accuracy: {1}; "
            "Average Accuracy: {2}; Kappa: {3}".format(
                score_rf, balanced_accuracy, average_accurate, kappa
            )
        )


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        cfg = yaml.safe_load(f)

    pl.seed_everything(cfg["program"]["seed"])
    main(cfg)
