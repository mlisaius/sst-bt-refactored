"""Datamodule and dataset utilities for labelled satellite time-series d-pixels.

This module provides the dataset class and dataloader factory functions used
during evaluation.  Unlike the unlabelled counterpart, each d-pixel here
carries a class label and a unique pixel ID, both of which are propagated
through the dataloader so that embeddings can be matched to ground-truth
annotations after encoding.

Key differences from the unlabelled datamodule
-----------------------------------------------
* ``CustomDPixDataset.__getitem__`` returns ``(views, label, id)`` instead of
  just the views.
* Input CSV columns are assumed to be ``[label, id, band_0, ...]`` — the label
  occupies column 0 and the ID column 1.
* No per-pixel normalisation step is applied.
* Vegetation indices (NDVI, GCVI) and sinusoidal day-of-year features are
  appended inside ``__getitem__`` before sparse sampling; the two sinusoidal
  columns are then dropped from the sampled tensor.
"""

import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


def create_maskset(data, n_masked):
    """Create a per-pixel cloud-masking lookup table.

    For each of the ``npixels`` rows in ``data``, draws ``n_masked`` date
    indices uniformly at random without replacement.  The resulting array is
    passed to ``CustomDPixDataset`` so that the same random dates are dropped
    consistently across augmentation copies.

    Args:
        data (np.ndarray): Full d-pixel array with shape
            ``(npixels, 2 + ndates * nbands)``.  The number of dates is
            inferred as ``(ncols - 2) / 10``.
        n_masked (int): Number of dates to mask per pixel.

    Returns:
        np.ndarray: Integer array of shape ``(npixels, n_masked)`` where each
            row contains the date indices to be masked for that pixel.
    """
    npixels = int(data.shape[0])
    ndates = int((data.shape[1] - 2) / 10)
    random_datemask = np.empty((npixels, n_masked), dtype=np.int8)
    for n in range(npixels):
        random_datemask[n] = np.array(random.sample(range(ndates), n_masked))
    return random_datemask


class CustomDPixDataset:
    """Dataset for labelled satellite time-series d-pixels.

    Each row in the input array represents one d-pixel: the first column is the
    class label, the second is a pixel ID, and the remaining columns are
    spectral reflectance values arranged as
    ``[date_0_band_0, ..., date_0_band_N, date_1_band_0, ...]``.

    Vegetation indices (NDVI and GCVI) and sinusoidal day-of-year encodings are
    computed and appended before sparse temporal sampling.  The sinusoidal
    features are dropped from the final tensor so that the encoder receives
    only spectral + VI features.

    The ``cmask`` argument controls cloud filtering:

    * ``0`` — no masking.
    * ``1`` — keep only rows where the last band (flag column) equals 0, then
      drop the flag column.
    * ``2`` — artificially remove ``n_masked`` dates using a pre-computed
      ``maskset``.

    Args:
        data (np.ndarray): Input data array with shape
            ``(npixels, 2 + ndates * nbands)``.
        n_ssamples (int): Number of timesteps to draw for each sparse sample.
        cmask (int): Cloud-masking mode (0, 1, or 2).
        n_masked (int): Number of dates to artificially mask when
            ``cmask == 2``.
        maskset (np.ndarray or list): Pre-computed date-mask indices produced by
            :func:`create_maskset`.  Only consulted when ``cmask == 2``.
        nbands (int): Number of spectral bands per date (excluding any flag
            column).
    """

    def __init__(self, data, n_ssamples, cmask, n_masked, maskset, nbands):
        self.data = data
        self.nbands = nbands
        self.samplesize = n_ssamples
        self.cmask = cmask
        self.n_masked = n_masked
        self.maskset = maskset

    def __getitem__(self, index):
        data = self.data
        label = int(np.asarray(data)[index, 0])
        id = int(np.asarray(data)[index, 1])

        if self.cmask == 1:
            ndates = int((data.shape[1] - 2) / (self.nbands + 1))
            dpixel = np.asarray(data)[index, 2:].reshape((ndates, self.nbands + 1))
        else:
            ndates = int((data.shape[1] - 2) / self.nbands)
            dpixel = np.asarray(data)[index, 2:].reshape((ndates, self.nbands))

        if self.cmask == 1:
            # Retain only cloud-free observations (flag == 0) and drop the flag.
            dpixel = dpixel[(dpixel[:, -1] == 0), :]
            ndates = int(dpixel.shape[0])
            dpixel = dpixel[:, :-1]
        elif self.cmask == 2:
            dpixel = (np.delete(dpixel, self.maskset[index], axis=0)).astype(int)

        # Compute vegetation indices and append as extra feature columns.
        NDVI = np.divide(
            np.subtract(dpixel[:, 6], dpixel[:, 2]),
            np.add(dpixel[:, 6], dpixel[:, 2]),
        )
        GCVI = np.divide(dpixel[:, 6], dpixel[:, 1]) - 1
        dpixel = np.c_[dpixel, NDVI, GCVI]

        # Append sinusoidal day-of-year encodings (used for positional context
        # during sampling; dropped from the encoder input below).
        dpixel = np.c_[
            dpixel,
            np.sin(2 * np.pi * np.arange(ndates) / ndates),
            np.cos(2 * np.pi * np.arange(ndates) / ndates),
        ]

        # Draw two independent sparse temporal samples to form the positive pair.
        sparsesample1 = dpixel[
            np.sort(random.sample(range(ndates - self.n_masked), self.samplesize)), :
        ]
        sparsesample2 = dpixel[
            np.sort(random.sample(range(ndates - self.n_masked), self.samplesize)), :
        ]

        # Drop the two trailing sinusoidal columns before converting to tensors.
        sparsesample1_t = (
            torch.from_numpy(sparsesample1[:, :-2].astype(float)).float()
        )[None, :]
        sparsesample2_t = (
            torch.from_numpy(sparsesample2[:, :-2].astype(float)).float()
        )[None, :]

        return (sparsesample1_t, sparsesample2_t), label, id

    def __len__(self):
        """Return the number of d-pixels in the dataset.

        Returns:
            int: Total number of rows in the underlying data array.
        """
        return np.asarray(self.data).shape[0]


def create_dataloaders(
    dataset, n_augmentations, batch_size, num_workers,
    n_ssamples, n_masked, maskset, nbands, cmask, shuffle_val,
):
    """Build a training DataLoader with optional dataset augmentation copies.

    Each augmentation copy is an independently sampled ``CustomDPixDataset``
    wrapping the same underlying array.  Copies are concatenated so that one
    epoch sees ``n_augmentations * 2 * len(dataset)`` samples.

    Args:
        dataset (np.ndarray): Labelled d-pixel array.
        n_augmentations (int): Number of augmentation copies per epoch
            (multiplied by 2 internally so that each logical augmentation
            produces two independently sampled views).
        batch_size (int): Dataloader batch size.
        num_workers (int): Number of DataLoader worker processes.
        n_ssamples (int): Number of sparse temporal samples per d-pixel.
        n_masked (int): Number of dates to artificially mask (``cmask == 2``).
        maskset (np.ndarray or list): Pre-computed mask indices.
        nbands (int): Number of spectral bands per date.
        cmask (int): Cloud-masking mode.
        shuffle_val (bool): Whether to shuffle the dataloader.

    Returns:
        DataLoader: A PyTorch DataLoader ready for training.
    """
    n_augmentations = n_augmentations * 2

    if n_augmentations == 0:
        train_dataset = CustomDPixDataset(
            dataset, n_ssamples, cmask, n_masked, maskset, nbands
        )
    else:
        copies = [
            CustomDPixDataset(dataset, n_ssamples, cmask, n_masked, maskset, nbands)
            for _ in range(n_augmentations)
        ]
        train_dataset = torch.utils.data.ConcatDataset(copies)

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_val,
        num_workers=num_workers,
        drop_last=True,
    )


def create_test_dataloader(
    test_data, n_augmentations, batch_size, num_workers,
    n_ssamples, n_masked, maskset, nbands, cmask, shuffle_val,
):
    """Build an evaluation DataLoader with optional dataset augmentation copies.

    Identical to :func:`create_dataloaders` except that the DataLoader always
    uses ``shuffle=False`` regardless of ``shuffle_val``, ensuring deterministic
    ordering for embedding extraction.

    Args:
        test_data (np.ndarray): Labelled d-pixel array.
        n_augmentations (int): Number of augmentation copies (not doubled here,
            unlike :func:`create_dataloaders`).
        batch_size (int): Dataloader batch size.
        num_workers (int): Number of DataLoader worker processes.
        n_ssamples (int): Number of sparse temporal samples per d-pixel.
        n_masked (int): Number of dates to artificially mask.
        maskset (np.ndarray or list): Pre-computed mask indices.
        nbands (int): Number of spectral bands per date.
        cmask (int): Cloud-masking mode.
        shuffle_val (bool): Ignored — test loaders never shuffle.

    Returns:
        DataLoader: A PyTorch DataLoader ready for evaluation.
    """
    if n_augmentations == 0:
        test_dataset = CustomDPixDataset(
            test_data, n_ssamples, cmask, n_masked, maskset, nbands
        )
    else:
        copies = [
            CustomDPixDataset(test_data, n_ssamples, cmask, n_masked, maskset, nbands)
            for _ in range(n_augmentations)
        ]
        test_dataset = torch.utils.data.ConcatDataset(copies)

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )


class BarlowTwinsDataModule(pl.LightningDataModule):
    """Lightning DataModule for labelled satellite time-series d-pixels.

    Handles CSV loading, unique-ID assignment, optional cloud masking, and
    dataloader construction for train / validation / test / predict stages.

    Args:
        train_data_dir (str): Path to the training CSV.
        val_data_dir (str): Path to the validation CSV.
        test_data_dir (str): Path to the test/predict CSV.
        batch_size (int): Batch size for train and val loaders.
        num_workers (int): DataLoader worker count.
        cmask (int): Cloud-masking mode (0, 1, or 2).
        n_masked (int): Dates to mask when ``cmask == 2``.
        n_augmentations (int): Augmentation copy count per epoch.
        n_ssamples (int): Sparse temporal samples per d-pixel.
        nbands (int): Spectral band count per date.
    """

    def __init__(
        self,
        train_data_dir: str = "./",
        val_data_dir: str = "./",
        test_data_dir: str = "./",
        batch_size: int = 128,
        num_workers: int = 0,
        cmask: int = 1,
        n_masked: int = 0,
        n_augmentations: int = 0,
        n_ssamples: int = 0,
        nbands: int = 9,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cmask = cmask
        self.n_masked = n_masked
        self.n_augmentations = n_augmentations
        self.n_ssamples = n_ssamples
        self.nbands = nbands

    def _load_csv(self, path):
        """Load a labelled CSV and replace the ID column with sequential IDs.

        The CSV is expected to have a spare leading column (dropped via
        ``[:, 1:]``), with the resulting column 0 being the class label and
        column 1 being a pixel identifier that is overwritten with a unique
        sequential integer to ensure consistent lookup during evaluation.

        Args:
            path (str): Path to the CSV file.

        Returns:
            np.ndarray: Array with shape ``(npixels, ncols - 1)`` where column
                1 contains unique sequential IDs starting from 1.
        """
        data = np.asarray(pd.read_csv(path))[:, 1:]
        data[:, 1] = np.arange(1, data.shape[0] + 1)
        return data

    def setup(self, stage: str):
        """Load datasets for the requested training stage.

        Args:
            stage (str): One of ``"fit"``, ``"test"``, or ``"predict"``.
        """
        if stage == "fit":
            self.data_train = self._load_csv(self.train_data_dir)
            print("Train data sanity check:", self.data_train[1, 0:15])
            self.data_val = self._load_csv(self.val_data_dir)
        elif stage in ("test", "predict"):
            self.data_test = self._load_csv(self.test_data_dir)

    def train_dataloader(self):
        """Return the training DataLoader."""
        maskset = (
            create_maskset(self.data_train, self.n_masked)
            if self.cmask == 2
            else []
        )
        return create_dataloaders(
            self.data_train,
            self.n_augmentations,
            self.batch_size,
            self.num_workers,
            self.n_ssamples,
            self.n_masked,
            maskset,
            self.nbands,
            self.cmask,
            True,
        )

    def val_dataloader(self):
        """Return the validation DataLoader."""
        maskset = (
            create_maskset(self.data_val, self.n_masked)
            if self.cmask == 2
            else []
        )
        return create_dataloaders(
            self.data_val,
            self.n_augmentations,
            self.batch_size,
            self.num_workers,
            self.n_ssamples,
            self.n_masked,
            maskset,
            self.nbands,
            self.cmask,
            False,
        )

    def test_dataloader(self):
        """Return the test DataLoader (full dataset in one batch)."""
        self.batch_size = int(self.data_test.shape[0]) * self.n_augmentations
        return create_test_dataloader(
            self.data_test,
            self.n_augmentations,
            self.batch_size,
            self.num_workers,
            self.n_ssamples,
            self.n_masked,
            [],
            self.nbands,
            self.cmask,
            False,
        )

    def predict_dataloader(self):
        """Return the predict DataLoader (full dataset in one batch)."""
        self.batch_size = int(self.data_test.shape[0]) * self.n_augmentations
        return create_test_dataloader(
            self.data_test,
            self.n_augmentations,
            self.batch_size,
            self.num_workers,
            self.n_ssamples,
            self.n_masked,
            [],
            self.nbands,
            self.cmask,
            False,
        )
