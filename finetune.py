"""Two-phase fine-tuning entry point for the Barlow Twins satellite time-series model.

Phase 1 — Linear probe: The pre-trained encoder is frozen and only the
classification head is trained.  This confirms that the SSL representations
carry useful crop-type signal and produces a well-initialised head checkpoint.

Phase 2 — Full fine-tune: The probe checkpoint is loaded, the encoder is
unfrozen, and the whole network is trained end-to-end with a 10× lower
learning rate on the encoder than on the head to preserve pre-trained features.

Usage:
    python finetune.py <path_to_config.yaml>

Example:
    python finetune.py config/FORPAPER_123_100.yaml
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

# BarlowTwinsDataModuleVIs handles the labelled train/val/test CSVs.
from src.datamodules import BarlowTwinsDataModuleVIs

# STBTClassification is the two-phase Lightning module defined in classification.py.
from src.models import STBTClassification


def _build_datamodule(conf):
    """Construct the labelled datamodule from config.

    Args:
        conf (dict): Parsed YAML configuration.

    Returns:
        BarlowTwinsDataModuleVIs: Configured datamodule for fine-tuning.
    """
    # Alias the nested dict to keep the call site readable.
    dm = conf["datamodule"]

    # train_data / val_data / test_data are the labelled CSV paths; they differ
    # from train_data_dir / val_data_dir which point to the unlabelled SSL data.
    return BarlowTwinsDataModuleVIs(
        train_data_dir=dm["train_data"],    # labelled training split
        val_data_dir=dm["val_data"],        # labelled validation split
        test_data_dir=dm["test_data"],      # labelled test split (unused during finetune)
        batch_size=dm["batch_size"],
        num_workers=dm["num_workers"],
        cmask=dm["cmask"],                  # cloud-masking mode (0/1/2)
        n_masked=dm["n_masked"],            # dates to mask when cmask==2
        n_augmentations=dm["n_augmentations"],  # number of stochastic copies per epoch
        n_ssamples=dm["n_ssamples"],        # timesteps drawn per sparse sample
        nbands=dm["nbands"],                # spectral bands per timestep (excl. VIs)
    )


def _build_model(conf, data_len, phase, ckpt_override=None):
    """Instantiate STBTClassification for the requested phase.

    Args:
        conf (dict): Parsed YAML configuration.
        data_len (int): Number of training samples (for warm-up scheduler).
        phase (str): ``"probe"`` or ``"finetune"``.
        ckpt_override (str or None): If provided, load this Lightning checkpoint
            and override ``phase`` with the value passed here.  Used for Phase 2
            to resume from the best probe checkpoint.

    Returns:
        STBTClassification: Initialised model.
    """
    # Short aliases for frequently used config sub-dicts.
    dm = conf["datamodule"]
    m  = conf["model"]
    ft = conf["finetune"]

    # Collect all constructor kwargs in one place so they can be forwarded
    # either to __init__ directly or to load_from_checkpoint as overrides.
    init_kwargs = dict(
        encoder_no=m["encoder_no"],             # encoder architecture (1–4)
        encoder_out_dim=m["encoder_out_dim"],   # representation dimensionality
        num_training_samples=data_len,          # used to compute warm-up steps
        batch_size=dm["batch_size"],
        z_dim=m["z_dim"],                       # SSL projection dim (ckpt compat.)
        max_epochs=m["max_epochs"],
        n_ssamples=dm["n_ssamples"],
        nbands=int(dm["nbands"]),               # cast to int; YAML may give float
        head_lr=ft["head_lr"],                  # learning rate for head params
        encoder_lr=ft["encoder_lr"],            # learning rate for encoder in Phase 2
        num_classes=ft["num_classes"],          # number of crop-type classes
        ckpt=m["cont_epoch"],                   # path to the pre-trained SSL checkpoint
        phase=phase,
    )

    if ckpt_override is not None:
        # Phase 2: start from the best probe checkpoint rather than scratch.
        # load_from_checkpoint reconstructs __init__ using the saved hparams,
        # but we override `phase` so the encoder is not re-frozen, and pass
        # all other kwargs so Lightning uses the current config values rather
        # than whatever was stored in the probe checkpoint.
        # strict=False is required because the probe optimizer state contains
        # only head params; we discard it and let configure_optimizers rebuild.
        model = STBTClassification.load_from_checkpoint(
            ckpt_override,
            phase=phase,      # override: "finetune" instead of saved "probe"
            strict=False,     # allow optimizer/scheduler state mismatch
            **{k: v for k, v in init_kwargs.items() if k != "phase"},  # other overrides
        )

        # Belt-and-suspenders: explicitly clear requires_grad=False flags that
        # the probe __init__ set.  load_from_checkpoint calls __init__ with
        # phase="finetune" so the flags should already be clear, but this
        # guards against edge cases where the checkpoint restores them.
        for p in model.encoder.parameters():
            p.requires_grad = True
    else:
        # Phase 1: fresh instantiation — encoder is loaded from the SSL checkpoint
        # and frozen inside STBTClassification.__init__.
        model = STBTClassification(**init_kwargs)

    return model


def run_probe(conf, datamodule, data_len):
    """Phase 1: train the classification head with the encoder frozen.

    Args:
        conf (dict): Parsed YAML configuration.
        datamodule (BarlowTwinsDataModuleVIs): Ready-to-use datamodule.
        data_len (int): Training set size.

    Returns:
        str: Path to the best probe checkpoint (highest ``val_acc``).
    """
    # Build the model with encoder frozen (phase="probe").
    model = _build_model(conf, data_len, phase="probe")

    # Ensure the output directory exists before the trainer tries to write to it.
    os.makedirs("outputs/probe", exist_ok=True)

    # Save only the single best checkpoint to avoid filling disk with probe ckpts.
    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",                         # pick checkpoint with highest val accuracy
        mode="max",                                # higher is better
        save_top_k=1,                              # keep only the best
        dirpath="outputs/probe",
        filename="probe-{epoch:02d}-{val_acc:.3f}",
    )

    # Use a separate WandB project suffix so probe and finetune runs are distinguishable.
    wandb_logger = WandbLogger(
        project=conf["logger"]["log_dir"] + "-probe",
        log_model=False,  # don't upload model artefacts to save bandwidth
    )

    trainer = Trainer(
        max_epochs=conf["finetune"]["probe_epochs"],   # shorter run; head trains quickly
        callbacks=[checkpoint_cb],
        accelerator="auto",                            # GPU if available, else CPU
        devices=1 if torch.cuda.is_available() else None,
        default_root_dir=conf["trainer"]["default_root_dir"],
        log_every_n_steps=conf["trainer"]["log_every_n_steps"],
        logger=wandb_logger,
        enable_progress_bar=False,  # suppress tqdm bars in HPC log files
    )

    trainer.fit(model, datamodule)

    # Print the path so it is visible in HPC job logs even without WandB access.
    print("Phase 1 best checkpoint:", checkpoint_cb.best_model_path)

    # Return the path so run_finetune can load it directly.
    return checkpoint_cb.best_model_path


def run_finetune(conf, datamodule, data_len, probe_ckpt):
    """Phase 2: unfreeze the encoder and fine-tune end-to-end.

    Loads the best probe checkpoint so the head starts from a good
    initialisation, then trains the full network with differential learning
    rates.

    Args:
        conf (dict): Parsed YAML configuration.
        datamodule (BarlowTwinsDataModuleVIs): Ready-to-use datamodule.
        data_len (int): Training set size.
        probe_ckpt (str): Path to the best Phase 1 checkpoint.
    """
    # Load probe weights and switch to finetune mode (encoder unfrozen).
    model = _build_model(conf, data_len, phase="finetune", ckpt_override=probe_ckpt)

    os.makedirs("outputs/finetune", exist_ok=True)

    # Keep the top-2 checkpoints in case the best has a transient spike and the
    # second-best is actually more representative.
    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        dirpath="outputs/finetune",
        filename="finetune-{epoch:02d}-{val_acc:.3f}",
    )

    wandb_logger = WandbLogger(
        project=conf["logger"]["log_dir"] + "-finetune",
        log_model=False,
    )

    trainer = Trainer(
        max_epochs=conf["finetune"]["finetune_epochs"],
        callbacks=[checkpoint_cb],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        default_root_dir=conf["trainer"]["default_root_dir"],
        log_every_n_steps=conf["trainer"]["log_every_n_steps"],
        logger=wandb_logger,
        enable_progress_bar=False,
    )

    trainer.fit(model, datamodule)
    print("Phase 2 best checkpoint:", checkpoint_cb.best_model_path)


def main(conf):
    """Run the full two-phase fine-tuning pipeline.

    Args:
        conf (dict): Parsed YAML configuration. Expected top-level keys are
            ``program``, ``datamodule``, ``model``, ``trainer``, ``logger``,
            and ``finetune``.
    """
    # Read the number of training rows from the CSV to configure the warm-up
    # scheduler.  We use pandas rather than loading the full array into numpy
    # so that only the shape is needed; the actual data is loaded by the datamodule.
    data_len = pd.read_csv(conf["datamodule"]["train_data"]).shape[0]
    print("Training set size:", data_len)

    # A single datamodule instance is reused across both phases so that the
    # same random seed governs both train/val splits.
    datamodule = _build_datamodule(conf)

    # Phase 1: train head only, returns path to best probe checkpoint.
    probe_ckpt = run_probe(conf, datamodule, data_len)

    # Phase 2: load probe checkpoint, unfreeze encoder, train end-to-end.
    run_finetune(conf, datamodule, data_len, probe_ckpt)


if __name__ == "__main__":
    # Accept the path to a YAML config file as the sole command-line argument,
    # mirroring the calling convention of train.py and evaluate.py.
    with open(sys.argv[1], "r") as f:
        cfg = yaml.safe_load(f)

    # Seed all RNGs (Python, NumPy, PyTorch) for reproducibility.
    pl.seed_everything(cfg["program"]["seed"])

    main(cfg)
