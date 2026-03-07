"""Barlow Twins Lightning module for labelled satellite time-series d-pixels.

This module is used during the evaluation phase.  It is architecturally
identical to the unlabelled pre-training variant
(:mod:`src.models.barlowtwins_unlabelled_VIs_speed`) with three differences:

1. ``shared_step`` unpacks ``(x1, x2), label, id = batch`` — the label and
   pixel ID emitted by the labelled datamodule are received but not used in
   the loss computation.
2. The learning rate is stored as a constructor parameter (``learning_rate``)
   and referenced as ``self.learning_rate`` in the optimiser, making it
   straightforward to vary from config.
3. ``Encoder2`` uses ``nn.ReLU`` instead of ``nn.LeakyReLU``.

Additionally, the ``BarlowTwins`` constructor accepts an ``embeddings_name``
argument that is used by the ``test_step`` method to write embedding CSV files.
"""

from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchvision.models.resnet import resnet18


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class BarlowTwinsLoss(nn.Module):
    """Cross-correlation matrix loss for Barlow Twins.

    Drives the on-diagonal elements of the normalised cross-correlation matrix
    towards 1 (invariance) and the off-diagonal elements towards 0
    (redundancy reduction).

    Args:
        batch_size (int): Training batch size.  Used to normalise the
            cross-correlation matrix.
        lambda_coeff (float): Weight applied to the off-diagonal penalty term.
            Defaults to ``5e-3``.
        z_dim (int): Dimensionality of the projection head output.

    Note:
        ``self.bn`` is registered as a submodule even though it is not called
        in ``forward``.  Its parameters appear in saved checkpoint state dicts
        (under ``loss_fn.bn.*``), so removing it would break checkpoint
        compatibility.
    """

    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super().__init__()
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff
        self.bn = nn.BatchNorm1d(z_dim, affine=False)

    def off_diagonal_ele(self, x):
        """Return a flattened view of the off-diagonal elements of a square matrix.

        Implementation taken from the official Barlow Twins repository:
        https://github.com/facebookresearch/barlowtwins/blob/main/main.py

        Args:
            x (torch.Tensor): Square 2-D tensor of shape ``(n, n)``.

        Returns:
            torch.Tensor: 1-D tensor containing the ``n * (n - 1)``
                off-diagonal elements in row-major order.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        """Compute the Barlow Twins loss.

        Args:
            z1 (torch.Tensor): Projected embeddings for the first view,
                shape ``(N, z_dim)``.
            z2 (torch.Tensor): Projected embeddings for the second view,
                shape ``(N, z_dim)``.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class Encoder1(nn.Module):
    """Three-layer fully-connected encoder.

    Args:
        encoder_out_dim (int): Output representation dimensionality.
        nbands (int): Number of feature channels per timestep (spectral bands
            plus any appended vegetation indices).
        n_ssamples (int): Number of sparse temporal samples per d-pixel.
    """

    def __init__(self, encoder_out_dim, nbands, n_ssamples):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(nbands * n_ssamples, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, encoder_out_dim),
        )

    def forward(self, x):
        return self.net(self.flatten(x))


class Encoder2(nn.Module):
    """Four-layer fully-connected encoder.

    Args:
        encoder_out_dim (int): Output representation dimensionality.
        nbands (int): Number of feature channels per timestep.
        n_ssamples (int): Number of sparse temporal samples per d-pixel.
    """

    def __init__(self, encoder_out_dim=20, nbands=10, n_ssamples=15):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(nbands * n_ssamples, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, encoder_out_dim),
        )

    def forward(self, x):
        return self.net(self.flatten(x))


class Encoder3(nn.Module):
    """Experimental 1-D CNN encoder (not yet validated).

    Args:
        encoder_out_dim (int): Output representation dimensionality.
    """

    def __init__(self, encoder_out_dim=20):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.Linear(64, encoder_out_dim),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.net(self.flatten(x))


def define_encoder(encoderno, encoder_out_dim, nbands, n_ssamples):
    """Instantiate the requested encoder variant.

    Args:
        encoderno (int): Encoder architecture selector.

            * ``1`` — three-layer MLP (:class:`Encoder1`).
            * ``2`` — four-layer MLP (:class:`Encoder2`).
            * ``3`` — experimental CNN (:class:`Encoder3`).
            * ``4`` — adapted ResNet-18 backbone.

        encoder_out_dim (int): Output representation dimensionality.
        nbands (int): Number of feature channels per timestep (spectral bands
            plus any vegetation indices).
        n_ssamples (int): Number of sparse temporal samples per d-pixel.

    Returns:
        nn.Module: Instantiated encoder.
    """
    if encoderno == 1:
        return Encoder1(encoder_out_dim, nbands, n_ssamples)
    elif encoderno == 2:
        return Encoder2(encoder_out_dim, nbands, n_ssamples)
    elif encoderno == 3:
        return Encoder3(encoder_out_dim)
    elif encoderno == 4:
        # Adapted ResNet-18: replace the first 7x7 conv with a 3x3 conv,
        # remove the max-pool, and replace the classification head with a
        # linear layer of the desired output dimension.
        this_encoder = resnet18()
        this_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        this_encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        this_encoder.fc = nn.Linear(512, encoder_out_dim, bias=False)
        return this_encoder


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """Two-layer MLP projection head.

    Args:
        input_dim (int): Input dimensionality (encoder output size).
        hidden_dim (int): Hidden layer dimensionality.
        output_dim (int): Output dimensionality (embedding size for loss).
    """

    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def _warmup_fn(warmup_steps, step):
    """Return the learning-rate multiplier for a linear warm-up schedule.

    Args:
        warmup_steps (int): Number of optimiser steps over which to ramp the
            learning rate from 0 to its target value.
        step (int): Current optimiser step.

    Returns:
        float: Multiplier in ``(0, 1]``.
    """
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 1.0


def linear_warmup_decay(warmup_steps):
    """Create a :func:`partial` of :func:`_warmup_fn` with ``warmup_steps`` bound.

    Args:
        warmup_steps (int): Total warm-up steps.

    Returns:
        Callable[[int], float]: A single-argument function suitable for use
            with ``torch.optim.lr_scheduler.LambdaLR``.
    """
    return partial(_warmup_fn, warmup_steps)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class BarlowTwins(LightningModule):
    """Barlow Twins Lightning module for labelled satellite time-series data.

    Used during evaluation: the model is loaded from a pre-trained checkpoint
    and its ``encoder`` attribute is extracted to produce frozen embeddings for
    downstream classification.

    The ``shared_step`` unpacks ``(x1, x2), label, id = batch``, matching the
    three-element output of :class:`~src.datamodules.btdmVIs.CustomDPixDataset`.

    Args:
        encoder_no (int): Encoder architecture selector (see
            :func:`define_encoder`).
        encoder_out_dim (int): Encoder output dimensionality.
        num_training_samples (int): Total training samples.  Used to compute
            the number of steps per epoch for the warm-up scheduler.
        batch_size (int): Training batch size.
        lambda_coeff (float): Off-diagonal loss weight.
        z_dim (int): Projection head output dimensionality.
        learning_rate (float): Base learning rate passed to Adam.
        warmup_epochs (int): Number of warm-up epochs.
        max_epochs (int): Total training epochs.
        n_ssamples (int): Sparse temporal sample count per d-pixel.
        nbands (int): Spectral band count per date.
        embeddings_name (str): Output CSV filename used by ``test_step``.
    """

    def __init__(
        self,
        encoder_no=1,
        encoder_out_dim=20,
        num_training_samples=100,
        batch_size=128,
        lambda_coeff=5e-3,
        z_dim=20,
        learning_rate=1e-4,
        warmup_epochs=10,
        max_epochs=200,
        n_ssamples=15,
        nbands=9,
        embeddings_name="embeddings.csv",
    ):
        super().__init__()
        # nbands + 2 accounts for the NDVI and GCVI columns appended in the dataset.
        self.encoder = define_encoder(encoder_no, encoder_out_dim, nbands + 2, n_ssamples)
        self.projection_head = ProjectionHead(
            input_dim=encoder_out_dim,
            hidden_dim=encoder_out_dim,
            output_dim=z_dim,
        )
        self.loss_fn = BarlowTwinsLoss(
            batch_size=batch_size,
            lambda_coeff=lambda_coeff,
            z_dim=z_dim,
        )
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.train_iters_per_epoch = num_training_samples // batch_size
        self.embeddings_name = embeddings_name
        self.save_hyperparameters(
            "encoder_no", "encoder_out_dim", "num_training_samples",
            "batch_size", "lambda_coeff", "z_dim", "learning_rate",
            "warmup_epochs", "max_epochs", "n_ssamples", "nbands",
        )

    def forward(self, x):
        """Run the encoder on input ``x``.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoder output.
        """
        return self.encoder(x)

    def shared_step(self, batch):
        """Compute the Barlow Twins loss for one batch.

        The label and pixel ID emitted by the labelled datamodule are received
        but not used in the loss.

        Args:
            batch: Tuple of ``((x1, x2), label, id)`` as returned by
                :class:`~src.datamodules.btdmVIs.CustomDPixDataset`.

        Returns:
            torch.Tensor: Scalar Barlow Twins loss.
        """
        (x1, x2), m, z = batch
        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))
        return self.loss_fn(z1, z2)

    def training_step(self, batch, batch_idx):
        """Execute one training step and log the per-step loss.

        Args:
            batch: Tuple of ``((x1, x2), label, id)`` from the training DataLoader.
            batch_idx (int): Index of the current batch within the epoch.

        Returns:
            torch.Tensor: Scalar training loss used for backpropagation.
        """
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Execute one validation step and log the per-epoch loss.

        Args:
            batch: Tuple of ``((x1, x2), label, id)`` from the validation DataLoader.
            batch_idx (int): Index of the current batch within the epoch.
        """
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Extract embeddings and write them to a CSV file.

        Args:
            batch: Tuple of ``((x1, x2), label, id)``.
            batch_idx (int): Batch index (unused).

        Returns:
            np.ndarray: Concatenated embedding array of shape
                ``(2 * N, 2 + encoder_out_dim)``.
        """
        (x1, x2), label, id = batch
        embeddings1_np = self.encoder(x1).detach().numpy()
        embeddings2_np = self.encoder(x2).detach().numpy()
        id_np = id.detach().numpy() - 1  # Correct for 1-based ID offset.
        label_np = label.detach().numpy()
        embeddings1_labels_np = np.c_[id_np, label_np, embeddings1_np]
        embeddings2_labels_np = np.c_[id_np, label_np, embeddings2_np]
        embeddings_full = np.concatenate(
            (embeddings1_labels_np, embeddings2_labels_np), axis=0
        )
        print("Embeddings shape:", embeddings_full.shape)
        pd.DataFrame(embeddings_full).to_csv(self.embeddings_name)
        return embeddings_full

    def pred_step(self, batch, batch_idx):
        """Extract embeddings without saving (predict-stage helper).

        Args:
            batch: Tuple of ``((x1, x2), label, id)``.
            batch_idx (int): Batch index (unused).

        Returns:
            np.ndarray: Concatenated embedding array.
        """
        (x1, x2), label, id = batch
        embeddings1_np = self.encoder(x1).detach().numpy()
        embeddings2_np = self.encoder(x2).detach().numpy()
        id_np = id.detach().numpy()
        embeddings1_labels_np = np.c_[id_np, embeddings1_np]
        embeddings2_labels_np = np.c_[id_np, embeddings2_np]
        return np.concatenate((embeddings1_labels_np, embeddings2_labels_np), axis=0)

    def configure_optimizers(self):
        """Configure the Adam optimiser with a linear warm-up LR schedule.

        Returns:
            Tuple[list, list]: A tuple of ``([optimizer], [scheduler_dict])``.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
