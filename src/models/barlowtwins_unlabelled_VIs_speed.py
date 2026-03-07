"""Barlow Twins model components for satellite time-series d-pixel representation learning.

This module provides all building blocks for the self-supervised Barlow Twins
framework adapted for use on satellite time-series data:

- **BarlowTwinsLoss**: The cross-correlation matrix loss from Zbontar et al. (2021).
- **Encoder1 / Encoder2 / Encoder3**: Three fully-connected (and one CNN) encoder
  options of increasing depth.
- **define_encoder**: Factory function that selects and constructs an encoder by number.
- **ProjectionHead**: Two-layer MLP with BatchNorm that maps encoder outputs to the
  embedding space in which the loss is computed.
- **linear_warmup_decay**: Learning-rate schedule with a linear warm-up phase.
- **BarlowTwins**: The complete PyTorch Lightning module that wires the encoder,
  projection head, and loss together for training, validation, and inference.

Input tensors are expected to have shape ``(batch, 1, n_ssamples, nbands + 2)``,
where the ``+ 2`` accounts for the NDVI and GCVI vegetation indices appended
by the data module, and the leading channel dimension of 1 is required by
certain encoder architectures (e.g. Encoder 4 / ResNet).

Reference:
    Zbontar, J. et al. (2021). Barlow Twins: Self-Supervised Learning via
    Redundancy Reduction. ICML 2021. https://arxiv.org/abs/2103.03230
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchvision.models.resnet import resnet18


class BarlowTwinsLoss(nn.Module):
    """Cross-correlation matrix loss for Barlow Twins self-supervised learning.

    Given two sets of embeddings ``z1`` and ``z2`` produced from two augmented
    views of the same input, this loss minimises the difference between their
    cross-correlation matrix and the identity matrix. The on-diagonal terms
    encourage invariance to augmentation (pushing the two views together), while
    the off-diagonal terms reduce redundancy between embedding dimensions.

    The loss is defined as:

        L = Σ_i (1 - C_ii)² + λ · Σ_{i≠j} C_ij²

    where C is the cross-correlation matrix normalised by batch size and λ
    (``lambda_coeff``) down-weights the off-diagonal penalty.

    Args:
        batch_size (int): Number of samples per batch; used to normalise the
            cross-correlation matrix.
        lambda_coeff (float): Weighting factor for the off-diagonal penalty.
            Defaults to ``5e-3`` as per the original paper.
        z_dim (int): Dimensionality of the embedding vectors. Defaults to ``128``.
    """

    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super().__init__()
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff
        # Retained for checkpoint compatibility with prior saved models.
        # Not used in the forward pass (affine=False means no learnable params).
        self.bn = nn.BatchNorm1d(z_dim, affine=False)

    def off_diagonal_ele(self, x):
        """Extract the off-diagonal elements of a square matrix as a 1-D tensor.

        Implementation adapted from the official Barlow Twins repository:
        https://github.com/facebookresearch/barlowtwins/blob/main/main.py

        Args:
            x (torch.Tensor): Square matrix of shape ``(n, n)``.

        Returns:
            torch.Tensor: 1-D tensor containing all ``n * (n - 1)`` off-diagonal
                elements in row-major order.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        """Compute the Barlow Twins loss for a batch of embedding pairs.

        NaN values in the embeddings (which can arise from degenerate inputs)
        are replaced with zero before normalisation to prevent the loss from
        becoming NaN and corrupting training.

        Args:
            z1 (torch.Tensor): Embeddings for the first view, shape ``(N, D)``.
            z2 (torch.Tensor): Embeddings for the second view, shape ``(N, D)``.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        z1 = torch.nan_to_num(z1)
        z2 = torch.nan_to_num(z2)

        # Normalise each embedding dimension across the batch dimension.
        z1_norm = (z1 - z1.mean(dim=0)) / z1.std(dim=0)
        z2_norm = (z2 - z2.mean(dim=0)) / z2.std(dim=0)

        # Compute the empirical cross-correlation matrix (D × D).
        cross_corr = torch.nan_to_num(
            torch.matmul(z1_norm.T, z2_norm) / self.batch_size
        )

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()
        return on_diag + self.lambda_coeff * off_diag


class Encoder1(nn.Module):
    """Three-layer fully-connected encoder with ReLU activations.

    The simplest encoder option. Suitable for rapid prototyping or when a
    lightweight model is preferred. Hidden layer width is fixed at 512.

    Args:
        encoder_out_dim (int): Dimensionality of the output representation.
        nbands (int): Number of input features per timestep
            (spectral bands + VIs).
        n_ssamples (int): Number of timesteps in each sparse temporal sample.
            The encoder input size is ``nbands * n_ssamples`` after flattening.
    """

    def __init__(self, encoder_out_dim, nbands, n_ssamples):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nbands * n_ssamples, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, encoder_out_dim),
        )

    def forward(self, x):
        """Pass input through the encoder network.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(batch, 1, n_ssamples, nbands)``.

        Returns:
            torch.Tensor: Representation tensor of shape
                ``(batch, encoder_out_dim)``.
        """
        return self.net(x)


class Encoder2(nn.Module):
    """Four-layer fully-connected encoder with LeakyReLU activations.

    A deeper alternative to ``Encoder1``. The additional hidden layer and
    LeakyReLU activations (negative slope 0.01) help with gradient flow in
    deeper networks. Hidden layer width is fixed at 512.

    Args:
        encoder_out_dim (int): Dimensionality of the output representation.
            Defaults to ``20``.
        nbands (int): Number of input features per timestep. Defaults to ``10``.
        n_ssamples (int): Number of timesteps per sparse sample. Defaults to ``15``.
    """

    def __init__(self, encoder_out_dim=20, nbands=10, n_ssamples=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nbands * n_ssamples, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, encoder_out_dim),
        )

    def forward(self, x):
        """Pass input through the encoder network.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(batch, 1, n_ssamples, nbands)``.

        Returns:
            torch.Tensor: Representation tensor of shape
                ``(batch, encoder_out_dim)``.
        """
        return self.net(x)


class Encoder3(nn.Module):
    """Experimental three-layer 1D-CNN encoder (not production-ready).

    This encoder was intended to apply 1D convolutions along the spectral
    dimension, but the current implementation flattens the input before the
    convolutional layers, which is not the intended behaviour. Retained for
    experimental reference only. Use Encoder1 or Encoder2 for production runs.

    Args:
        encoder_out_dim (int): Dimensionality of the output representation.
            Defaults to ``20``.
    """

    def __init__(self, encoder_out_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Conv1d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.Linear(64, encoder_out_dim),
            nn.Softmax(),
        )

    def forward(self, x):
        """Pass input through the experimental CNN encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch, encoder_out_dim)``.
        """
        return self.net(x)


def define_encoder(encoderno, encoder_out_dim, nbands, n_ssamples):
    """Construct and return the requested encoder variant.

    Args:
        encoderno (int): Encoder variant selector.
            ``1`` – three-layer FC with ReLU (``Encoder1``).
            ``2`` – four-layer FC with LeakyReLU (``Encoder2``).
            ``3`` – experimental 1D-CNN (``Encoder3``).
            ``4`` – adapted ResNet-18 with a modified first convolution and
                    a custom fully-connected output layer.
        encoder_out_dim (int): Dimensionality of the output representation.
        nbands (int): Number of input features per timestep.
        n_ssamples (int): Number of timesteps per sparse sample.

    Returns:
        nn.Module: Instantiated encoder module.
    """
    if encoderno == 1:
        return Encoder1(encoder_out_dim, nbands, n_ssamples)
    elif encoderno == 2:
        return Encoder2(encoder_out_dim, nbands, n_ssamples)
    elif encoderno == 3:
        return Encoder3(encoder_out_dim)
    elif encoderno == 4:
        # Adapt a standard ResNet-18 backbone for 1-channel d-pixel inputs:
        # replace the 7×7 stem conv with a 3×3 conv to suit smaller inputs,
        # remove the initial max-pool to preserve spatial resolution, and
        # replace the classification head with a linear projection to the
        # desired output dimensionality.
        encoder = resnet18()
        encoder.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        encoder.fc = nn.Linear(512, encoder_out_dim, bias=False)
        return encoder


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head with BatchNorm and ReLU.

    Maps encoder output representations into the embedding space in which the
    Barlow Twins loss is computed. Following the original paper, a BatchNorm
    layer is inserted between the two linear layers.

    Args:
        input_dim (int): Dimensionality of the encoder output. Defaults to ``2048``.
        hidden_dim (int): Width of the hidden layer. Defaults to ``2048``.
        output_dim (int): Dimensionality of the final embedding. Defaults to ``128``.
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
        """Project encoder representations into the embedding space.

        Args:
            x (torch.Tensor): Encoder output of shape ``(batch, input_dim)``.

        Returns:
            torch.Tensor: Projected embeddings of shape ``(batch, output_dim)``.
        """
        return self.projection_head(x)


def _warmup_fn(warmup_steps, step):
    """Compute the learning-rate scaling factor for a linear warm-up schedule.

    Returns a linearly increasing multiplier during the warm-up phase and a
    constant multiplier of 1.0 thereafter.

    Args:
        warmup_steps (int): Total number of warm-up optimisation steps.
        step (int): Current optimisation step.

    Returns:
        float: Scaling factor in the range ``(0, 1]``.
    """
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 1.0


def linear_warmup_decay(warmup_steps):
    """Return a ``LambdaLR``-compatible scheduler function with linear warm-up.

    Uses ``functools.partial`` to bind ``warmup_steps`` so that the returned
    callable matches the single-argument signature expected by
    ``torch.optim.lr_scheduler.LambdaLR``.

    Args:
        warmup_steps (int): Number of steps over which to linearly ramp the
            learning rate from 0 to its full value.

    Returns:
        Callable[[int], float]: A function ``f(step) -> scale_factor``.
    """
    return partial(_warmup_fn, warmup_steps)


class BarlowTwins(LightningModule):
    """Full Barlow Twins SSL model for satellite time-series d-pixel inputs.

    Combines an encoder, a projection head, and the Barlow Twins loss into a
    single PyTorch Lightning module. The model is trained end-to-end on pairs
    of augmented sparse temporal samples of each d-pixel.

    At test / prediction time the encoder (without the projection head) is used
    to extract fixed-length representations that can be fed to downstream
    classifiers (e.g. random forest or fine-tuned linear probes).

    Args:
        encoder_no (int): Encoder variant (1–4). See ``define_encoder``.
        encoder_out_dim (int): Output dimensionality of the encoder.
        num_training_samples (int): Total number of training samples; used to
            compute ``train_iters_per_epoch`` for the warm-up scheduler.
        batch_size (int): Training batch size.
        lambda_coeff (float): Off-diagonal penalty weight in the loss.
        z_dim (int): Dimensionality of the projection head output.
        learning_rate (float): Base learning rate for the Adam optimiser.
        warmup_epochs (int): Number of epochs over which to ramp the LR.
        max_epochs (int): Total training epochs.
        n_ssamples (int): Number of timesteps per sparse temporal sample.
        nbands (int): Number of spectral bands per timestep (excluding VIs).
            Two extra features (NDVI, GCVI) are added internally, so the
            encoder receives ``nbands + 2`` features per timestep.
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
    ):
        super().__init__()
        # The data module appends NDVI and GCVI (+2 bands) before the tensors
        # reach the encoder, so the true input width is nbands + 2.
        self.encoder = define_encoder(encoder_no, encoder_out_dim, nbands + 2, n_ssamples)
        self.projection_head = ProjectionHead(encoder_out_dim, encoder_out_dim, z_dim)
        self.loss_fn = BarlowTwinsLoss(batch_size, lambda_coeff, z_dim)
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.train_iters_per_epoch = num_training_samples // batch_size
        self.save_hyperparameters(
            "encoder_no", "encoder_out_dim", "num_training_samples", "batch_size",
            "lambda_coeff", "z_dim", "learning_rate", "warmup_epochs", "max_epochs",
            "n_ssamples", "nbands",
        )

    def forward(self, x):
        """Run the encoder on a batch of d-pixel views.

        Args:
            x (torch.Tensor): Input of shape ``(batch, 1, n_ssamples, nbands+2)``.

        Returns:
            torch.Tensor: Encoder representations of shape
                ``(batch, encoder_out_dim)``.
        """
        return self.encoder(x)

    def shared_step(self, batch):
        """Compute the Barlow Twins loss for a single batch.

        Used by both ``training_step`` and ``validation_step`` to avoid
        code duplication.

        Args:
            batch (tuple): A batch of the form ``((x1, x2), _)`` where ``x1``
                and ``x2`` are tensors of shape ``(N, 1, n_ssamples, nbands+2)``
                representing the two augmented views, and ``_`` is the pixel id
                (unused in the loss computation).

        Returns:
            torch.Tensor: Scalar Barlow Twins loss.
        """
        (x1, x2), _ = batch
        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))
        return self.loss_fn(z1, z2)

    def training_step(self, batch, batch_idx):
        """Execute one training step and log the loss.

        Args:
            batch (tuple): Batch from the training DataLoader.
            batch_idx (int): Index of the current batch within the epoch.

        Returns:
            torch.Tensor: Scalar training loss for backpropagation.
        """
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Execute one validation step and log the loss.

        Args:
            batch (tuple): Batch from the validation DataLoader.
            batch_idx (int): Index of the current batch within the epoch.
        """
        loss = self.shared_step(batch)
        # sync_dist=True averages the loss across all devices in distributed
        # training, ensuring the logged value reflects the full validation set.
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """Extract and return encoder embeddings for a test batch.

        Embeddings from both views are concatenated with their pixel IDs so
        that downstream code can match representations back to ground-truth
        labels. The projection head is intentionally bypassed here; the raw
        encoder output is more suitable as a fixed feature for classification.

        Args:
            batch (tuple): A batch of the form ``((x1, x2), ids)`` where
                ``ids`` is a 1-D tensor of integer pixel identifiers.
            batch_idx (int): Index of the current batch.

        Returns:
            np.ndarray: Array of shape ``(2N, 1 + encoder_out_dim)`` where the
                first column contains pixel IDs and the remaining columns contain
                the encoder embeddings. Rows 0..N-1 correspond to view 1 and
                rows N..2N-1 correspond to view 2.
        """
        (x1, x2), ids = batch
        e1 = self.encoder(x1).detach().numpy()
        e2 = self.encoder(x2).detach().numpy()
        ids = ids.detach().numpy()
        return np.concatenate([np.c_[ids, e1], np.c_[ids, e2]], axis=0)

    def pred_step(self, batch, batch_idx):
        """Extract embeddings during a predict run; delegates to ``test_step``.

        Args:
            batch (tuple): A batch of the form ``((x1, x2), ids)``.
            batch_idx (int): Index of the current batch.

        Returns:
            np.ndarray: Same format as ``test_step``.
        """
        return self.test_step(batch, batch_idx)

    def configure_optimizers(self):
        """Set up the Adam optimiser with a linear warm-up learning-rate schedule.

        The learning rate is linearly ramped from 0 to the base rate over
        ``warmup_epochs`` epochs (measured in optimisation steps), then held
        constant. The Adam betas are set to ``(0.5, 0.999)`` as found to work
        well empirically for this dataset.

        Returns:
            tuple: A 2-tuple ``([optimizer], [scheduler_dict])`` in the format
                expected by PyTorch Lightning.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.5, 0.999))
        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, linear_warmup_decay(warmup_steps)
            ),
            "interval": "step",   # Update the LR after every optimisation step.
            "frequency": 1,
        }
        return [optimizer], [scheduler]
