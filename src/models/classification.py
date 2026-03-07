"""Two-phase supervised fine-tuning of a pre-trained Barlow Twins encoder.

Phase 1 (probe): The encoder is frozen and only the classification head is
trained.  This verifies that the SSL representations are meaningful before
committing to full end-to-end training.

Phase 2 (finetune): The encoder is unfrozen with a 10× lower learning rate
than the head, allowing the pre-trained features to adapt without being
destroyed.

The module is designed so that the Phase 1 checkpoint can be loaded directly
by ``STBTClassification.load_from_checkpoint`` with ``phase="finetune"``
overriding the saved hparam, enabling seamless transition between phases.
"""

# partial is used to bind warmup_steps into the LambdaLR callable.
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

# BarlowTwinsUVIsSp is the pre-trained SSL model whose encoder we extract.
# linear_warmup_decay builds the LR schedule function used in both phases.
from .barlowtwins_unlabelled_VIs_speed import BarlowTwins as BarlowTwinsUVIsSp
from .barlowtwins_unlabelled_VIs_speed import linear_warmup_decay


class MulticlassClassification(nn.Module):
    """Three-layer MLP classification head.

    Architecture: Linear(in_dim, 512) → ReLU → Linear(512, 256) →
    Dropout(0.1) → Linear(256, num_classes).

    Args:
        in_dim (int): Input feature dimensionality (encoder output size).
        num_classes (int): Number of target crop-type classes.
    """

    def __init__(self, in_dim, num_classes):
        super().__init__()
        # Three linear layers with reducing width; Dropout after the second
        # layer regularises training when labelled data is scarce.
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),   # project encoder dim → 512
            nn.ReLU(),                 # non-linearity after first projection
            nn.Linear(512, 256),       # compress to 256
            nn.Dropout(0.1),           # light dropout to reduce overfitting
            nn.Linear(256, num_classes),  # final logit layer; one output per class
        )

    def forward(self, x):
        # Pass the (batch, in_dim) tensor through all layers in sequence.
        return self.net(x)


class STBTClassLoss(nn.Module):
    """Cross-entropy loss averaged across two augmented views.

    Returns both the scalar loss and the mean per-batch accuracy computed
    from the averaged logits of the two views.

    Args:
        num_classes (int): Number of target classes (stored for reference).
    """

    def __init__(self, num_classes):
        super().__init__()
        # Store num_classes for potential future use (e.g. per-class metrics).
        self.num_classes = num_classes

    def forward(self, z1, z2, label):
        """Compute loss and accuracy for a pair of view logits.

        Args:
            z1 (torch.Tensor): Logits for view 1, shape ``(N, num_classes)``.
            z2 (torch.Tensor): Logits for view 2, shape ``(N, num_classes)``.
            label (torch.Tensor): Ground-truth class indices, shape ``(N,)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(loss, accuracy)`` where both
                are scalar tensors.  ``accuracy`` is in the range ``[0, 1]``.
        """
        # Labels arrive as floats from the dataloader; cross_entropy needs longs.
        label = label.long()

        # Compute cross-entropy for each view independently, then average.
        # Training on both views acts as an implicit augmentation of supervision.
        loss = (F.cross_entropy(z1, label) + F.cross_entropy(z2, label)) / 2.0

        # Average the two sets of logits to get a single prediction per sample.
        # This is more stable than picking one view arbitrarily at eval time.
        avg_logits = (z1 + z2) / 2.0

        # argmax picks the predicted class; compare to ground truth and take mean.
        acc = (avg_logits.argmax(dim=1) == label).float().mean()

        return loss, acc


class STBTClassification(LightningModule):
    """PyTorch Lightning module for two-phase fine-tuning of a Barlow Twins encoder.

    Loads a pre-trained ``BarlowTwinsUVIsSp`` checkpoint, extracts its encoder,
    and attaches a fresh classification head.  The ``phase`` argument controls
    whether the encoder is frozen (``"probe"``) or trainable (``"finetune"``).

    In the ``"finetune"`` phase, two Adam parameter groups are used so that the
    encoder receives a 10× lower learning rate than the head, preserving the
    pre-trained features while still allowing end-to-end adaptation.

    Args:
        encoder_no (int): Encoder variant used during SSL pre-training (1–4).
        encoder_out_dim (int): Encoder output dimensionality.
        num_training_samples (int): Size of the training set; used to compute
            the number of optimisation steps per epoch for the warm-up
            scheduler.
        batch_size (int): Training batch size.
        z_dim (int): Projection head output dim of the SSL model (kept for
            checkpoint compatibility).
        max_epochs (int): Total training epochs for this phase.
        n_ssamples (int): Number of sparse temporal samples per d-pixel.
        nbands (int): Number of spectral bands per timestep (excluding VIs).
        head_lr (float): Learning rate for the classification head.
        encoder_lr (float): Learning rate for the encoder in Phase 2 (ignored
            in Phase 1 when the encoder is frozen).
        num_classes (int): Number of crop-type target classes.
        ckpt (str): Path to the pre-trained SSL checkpoint from which the
            encoder is extracted.
        phase (str): ``"probe"`` to freeze the encoder, ``"finetune"`` to
            train it end-to-end.  Defaults to ``"probe"``.
    """

    def __init__(
        self,
        encoder_no,
        encoder_out_dim,
        num_training_samples,
        batch_size,
        z_dim,
        max_epochs,
        n_ssamples,
        nbands,
        head_lr,
        encoder_lr,
        num_classes,
        ckpt,
        phase="probe",
    ):
        super().__init__()

        # Persist all constructor arguments to the checkpoint so that
        # load_from_checkpoint can reconstruct the model without extra kwargs.
        self.save_hyperparameters()

        # Restore the full pre-trained Barlow Twins model from the SSL checkpoint,
        # then keep only the encoder sub-module and discard everything else
        # (projection head, loss, optimiser state).  This avoids carrying
        # unused SSL parameters into fine-tuning.
        pretrained = BarlowTwinsUVIsSp.load_from_checkpoint(ckpt)
        self.encoder = pretrained.encoder  # nn.Module: maps d-pixel views → embeddings

        # In the probe phase the encoder weights must not change so that we can
        # measure the quality of the fixed SSL representations before committing
        # to expensive end-to-end training.
        if phase == "probe":
            for p in self.encoder.parameters():
                p.requires_grad = False  # prevent gradients flowing into encoder

        # BatchNorm1d normalises the encoder output distribution before the
        # linear layers, which stabilises early training and reduces sensitivity
        # to the choice of head_lr.  MulticlassClassification then maps the
        # normalised embeddings to per-class logits.
        self.head = nn.Sequential(
            nn.BatchNorm1d(encoder_out_dim),                      # normalise embeddings
            MulticlassClassification(encoder_out_dim, num_classes),  # → class logits
        )

        # Loss function shared by training and validation steps.
        self.loss_fn = STBTClassLoss(num_classes)

        # Store phase and learning rates as instance attributes so they are
        # accessible in configure_optimizers without going through hparams.
        self.phase = phase
        self.head_lr = head_lr
        self.encoder_lr = encoder_lr

        # Number of gradient steps per epoch; drives the warm-up scheduler.
        # Integer division is intentional: drop_last=True in the dataloader
        # means the last partial batch is never seen.
        self.train_iters_per_epoch = num_training_samples // batch_size

    def shared_step(self, batch):
        """Forward pass and loss computation shared by train and val steps.

        Args:
            batch (tuple): ``((x1, x2), label, id)`` from the labelled
                datamodule.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(loss, accuracy)``.
        """
        # Unpack the two augmented views and their labels; pixel id is unused here.
        (x1, x2), label, _ = batch

        # Run each view through the frozen/trainable encoder, then through the head.
        # Both views share the same encoder and head weights (no siamese split).
        z1 = self.head(self.encoder(x1))  # logits from view 1, shape (N, num_classes)
        z2 = self.head(self.encoder(x2))  # logits from view 2, shape (N, num_classes)

        # Compute averaged cross-entropy loss and accuracy across the two views.
        return self.loss_fn(z1, z2, label)

    def training_step(self, batch, batch_idx):
        # Compute loss and accuracy for this training batch.
        loss, acc = self.shared_step(batch)

        # Log per-step so we can see learning dynamics within each epoch.
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log("train_acc",  acc,  on_step=True, on_epoch=False)

        # Lightning expects the loss scalar to be returned for backpropagation.
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)

        # Log per-epoch only; sync_dist=True averages across GPUs in DDP.
        # val_acc is what ModelCheckpoint monitors to select the best checkpoint.
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_acc",  acc,  on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        """Set up Adam with a one-epoch linear warm-up schedule.

        In the probe phase only head parameters are passed to the optimiser;
        the encoder is frozen so its gradients are never computed.  In the
        finetune phase two parameter groups are created with different learning
        rates (encoder at ``encoder_lr``, head at ``head_lr``).

        Returns:
            tuple: ``([optimizer], [scheduler_dict])`` as expected by Lightning.
        """
        if self.phase == "probe":
            # Only optimise the head; the encoder requires_grad is already False,
            # but restricting the parameter group makes the intention explicit
            # and avoids accidentally creating zero-grad entries in the optimiser.
            param_groups = self.head.parameters()
        else:
            # Differential learning rates: the encoder has already learned useful
            # representations, so we update it much more slowly than the head to
            # avoid catastrophic forgetting of the SSL pre-training signal.
            param_groups = [
                {"params": self.encoder.parameters(), "lr": self.encoder_lr},  # 10× lower
                {"params": self.head.parameters(),    "lr": self.head_lr},
            ]

        # head_lr serves as the default lr for groups that don't set their own.
        optimizer = Adam(param_groups, lr=self.head_lr)

        # Linearly ramp the learning rate from 0 to its full value over the
        # first epoch's worth of steps, then hold constant.  This prevents
        # destructively large updates at the very start of training.
        warmup_steps = self.train_iters_per_epoch  # one full epoch of warm-up
        scheduler = LambdaLR(optimizer, linear_warmup_decay(warmup_steps))

        # "interval": "step" tells Lightning to call scheduler.step() after
        # every optimisation step rather than after every epoch.
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
