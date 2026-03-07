"""SSL pre-training entry point for the Barlow Twins satellite time-series model.

This script loads a YAML configuration file, constructs the data module and
model, and runs the PyTorch Lightning training loop. Checkpoints are saved to
the directory specified in the config under ``trainer.default_root_dir``, and
all metrics are logged to Weights & Biases.

Usage:
    python train.py <path_to_config.yaml>

Example:
    python train.py config/FORPAPER_123_100.yaml
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.datamodules import BarlowTwinsDataModuleUVI as BarlowTwinsDataModule
from src.models import BarlowTwinsUVIsSp as BarlowTwins


def main(conf):
    """Build and train the Barlow Twins model from a configuration dictionary.

    Reads the training CSV to determine the dataset size, constructs the data
    module and model, attaches a WandB logger and a checkpoint callback, then
    calls ``trainer.fit``.

    Args:
        conf (dict): Parsed YAML configuration. Expected top-level keys are
            ``program``, ``datamodule``, ``model``, ``trainer``, and ``logger``.
    """
    # Row count is used to compute the number of training iterations per epoch,
    # which in turn drives the learning-rate warm-up scheduler.
    data_len = np.asarray(pd.read_csv(conf["datamodule"]["train_data_dir"])).shape[0]

    # Create the checkpoint output directory if it does not already exist.
    if not os.path.isdir(conf["trainer"]["default_root_dir"]):
        os.makedirs(conf["trainer"]["default_root_dir"])

    datamodule = BarlowTwinsDataModule(
        train_data_dir=conf["datamodule"]["train_data_dir"],
        val_data_dir=conf["datamodule"]["val_data_dir"],
        test_data_dir=conf["datamodule"]["test_data_dir"],
        batch_size=conf["datamodule"]["batch_size"],
        num_workers=16,
        cmask=conf["datamodule"]["cmask"],
        n_masked=conf["datamodule"]["n_masked"],
        n_augmentations=conf["datamodule"]["n_augmentations"],
        n_ssamples=conf["datamodule"]["n_ssamples"],
        nbands=conf["datamodule"]["nbands"],
    )

    model = BarlowTwins(
        encoder_no=conf["model"]["encoder_no"],
        encoder_out_dim=conf["model"]["encoder_out_dim"],
        num_training_samples=data_len,
        batch_size=conf["datamodule"]["batch_size"],
        z_dim=conf["model"]["z_dim"],
        max_epochs=conf["model"]["max_epochs"],
        n_ssamples=conf["datamodule"]["n_ssamples"],
        nbands=int(conf["datamodule"]["nbands"]),
    )

    wandb_logger = WandbLogger(project=conf["logger"]["log_dir"], log_model="all")

    # Save a checkpoint after every epoch, retaining all of them (save_top_k=-1)
    # so that any epoch can be used as a starting point for fine-tuning.
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=-1,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            dirpath="outputs/",
            filename=conf["model"]["ckpt_name"] + "-{epoch:02d}-{val_loss:.2f}",
        )
    ]

    trainer = Trainer(
        callbacks=callbacks,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=conf["model"]["max_epochs"],
        default_root_dir=conf["trainer"]["default_root_dir"],
        log_every_n_steps=conf["trainer"]["log_every_n_steps"],
        logger=wandb_logger,
        enable_progress_bar=False,
    )

    # Register the model with WandB before training so that gradients and
    # model topology are tracked throughout the run.
    wandb_logger.watch(model)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    # Accept the path to a YAML config file as the sole command-line argument.
    with open(sys.argv[1], "r") as f:
        cfg = yaml.safe_load(f)

    # Seed all random number generators for reproducibility.
    pl.seed_everything(cfg["program"]["seed"])

    main(cfg)
