"""Data module for unlabelled satellite time-series d-pixels with vegetation indices.

A *d-pixel* (date-pixel) is a single geographic pixel observed across multiple
dates. Each row of the input CSV represents one d-pixel with the layout:

    [code, id, band_0_t0, ..., band_N_t0, flag_t0, band_0_t1, ..., flag_tT]

where ``flag`` is an optional cloud-quality indicator whose interpretation
is controlled by the ``cmask`` parameter.

On-the-fly feature engineering appends two vegetation indices per timestep:

- **NDVI** = (NIR − Red) / (NIR + Red)  [bands 6 and 2 respectively]
- **GCVI** = (NIR / Green) − 1          [bands 6 and 1 respectively]

Sinusoidal day-of-year (DOY) encodings are also appended to provide the model
with temporal position information. The full feature vector for each timestep
is then per-pixel standardised before sampling.

The Barlow Twins augmentation strategy is implemented via *sparse temporal
sampling*: two independent subsets of ``n_ssamples`` timesteps are drawn
uniformly at random from the available dates for each d-pixel, forming the
pair of views fed to the contrastive loss.
"""

import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


def create_maskset(data, n_masked):
    """Build a random cloud-mask index array for artificial cloud simulation.

    For each pixel, ``n_masked`` date indices are sampled without replacement
    from the full set of available dates. The resulting array is passed to
    ``CustomDPixDataset`` when ``cmask=2`` to artificially remove those dates
    before sparse temporal sampling.

    Args:
        data (np.ndarray): Full dataset array of shape ``(n_pixels, n_cols)``.
            Column count is used to infer the number of dates; the formula
            assumes 10 bands per timestep and 2 header columns.
        n_masked (int): Number of dates to mask per pixel.

    Returns:
        np.ndarray: Integer array of shape ``(n_pixels, n_masked)`` containing
            the date indices to remove for each pixel.
    """
    npixels = int(data.shape[0])
    # Infer the number of timesteps from the total column count, assuming
    # 10 spectral bands per date plus 2 header columns (code, id).
    ndates = int((data.shape[1] - 2) / 10)
    random_datemask = np.empty((npixels, n_masked), dtype=np.int8)
    for n in range(npixels):
        random_datemask[n] = np.array(random.sample(range(ndates), n_masked))
    return random_datemask


class CustomDPixDataset:
    """PyTorch-compatible dataset that yields augmented view pairs from d-pixels.

    Each call to ``__getitem__`` applies the full preprocessing pipeline to a
    single d-pixel and returns two independently sparse-sampled views of that
    pixel for use as the positive pair in Barlow Twins training.

    The preprocessing pipeline per d-pixel is:
        1. Reshape the flat band array into a ``(ndates, nbands)`` matrix.
        2. Apply cloud masking according to ``cmask``.
        3. Compute NDVI and GCVI vegetation indices and append them as columns.
        4. Append sinusoidal DOY encodings (sin and cos) as positional features.
        5. Standardise each feature column across dates (zero mean, unit std).
        6. Draw two independent sparse temporal samples of ``n_ssamples`` rows.
        7. Strip the DOY encoding columns (last two) before converting to tensors.

    The final tensors have shape ``(1, n_ssamples, nbands + 2)``, where the
    extra channel dimension is required by the encoder.

    Args:
        data (np.ndarray): Full dataset array of shape ``(n_pixels, n_cols)``.
        n_ssamples (int): Number of timesteps to include in each sparse sample.
        cmask (int): Cloud-masking mode.
            ``0`` – retain dates where the quality flag equals 1 (clear sky).
            ``1`` – retain dates where the quality flag equals 0 (no cloud).
            ``2`` – artificially remove the ``n_masked`` dates listed in
                    ``maskset`` for each pixel.
        n_masked (int): Number of masked dates used when sampling the sparse
            views; the sampling range is ``[0, ndates - n_masked)``.
        maskset (np.ndarray or list): Per-pixel date indices to remove when
            ``cmask=2``; ignored otherwise. Pass an empty list for other modes.
        nbands (int): Number of spectral bands per timestep, excluding the
            cloud-quality flag column.
        startindex (int): Column index at which the band data begins (i.e. the
            number of header columns at the start of each row).
    """

    def __init__(self, data, n_ssamples, cmask, n_masked, maskset, nbands, startindex):
        self.data = data
        self.nbands = nbands
        self.samplesize = n_ssamples
        self.cmask = cmask
        self.n_masked = n_masked
        self.maskset = maskset
        self.startindex = startindex

    def __getitem__(self, index):
        """Return a pair of augmented view tensors and the pixel identifier.

        Args:
            index (int): Row index into the dataset array.

        Returns:
            tuple: A 2-tuple ``((s1, s2), id)`` where ``s1`` and ``s2`` are
                float32 tensors of shape ``(1, n_ssamples, nbands + 2)`` and
                ``id`` is an integer pixel identifier taken from column 0.
        """
        data = self.data
        # Column 0 holds the crop-class code, which is used as the pixel
        # identifier for downstream evaluation.
        id = int(np.asarray(data)[index, 0])

        # When a cloud-quality flag column is present (cmask 0 or 1), each
        # timestep occupies nbands + 1 columns; otherwise it occupies nbands.
        if self.cmask in (0, 1):
            ndates = int((data.shape[1] - 2) / (self.nbands + 1))
            dpixel = np.asarray(data)[index, self.startindex:].reshape(
                (ndates, self.nbands + 1)
            )
        else:
            ndates = int((data.shape[1] - 2) / self.nbands)
            dpixel = np.asarray(data)[index, self.startindex:].reshape(
                (ndates, self.nbands)
            )

        # Apply cloud masking: retain only the valid dates and remove the flag
        # column, or artificially delete the pre-computed masked date rows.
        if self.cmask == 0:
            # Flag value 1 indicates a clear-sky observation.
            dpixel = dpixel[dpixel[:, -1] == 1, :]
            ndates = int(dpixel.shape[0])
            dpixel = dpixel[:, :-1]  # Drop the flag column.
        elif self.cmask == 1:
            # Flag value 0 indicates a cloud-free observation.
            dpixel = dpixel[dpixel[:, -1] == 0, :]
            ndates = int(dpixel.shape[0])
            dpixel = dpixel[:, :-1]  # Drop the flag column.
        elif self.cmask == 2:
            # Remove the pre-selected dates to simulate cloud occlusion.
            dpixel = np.delete(dpixel, self.maskset[index], axis=0).astype(int)

        # --- Vegetation index computation -----------------------------------
        # NDVI: Normalised Difference Vegetation Index using NIR (band 6) and
        #       Red (band 2).
        NDVI = np.divide(
            np.subtract(dpixel[:, 6], dpixel[:, 2]),
            np.add(dpixel[:, 6], dpixel[:, 2]),
        )
        # GCVI: Green Chlorophyll Vegetation Index using NIR (band 6) and
        #       Green (band 1).
        GCVI = np.divide(dpixel[:, 6], dpixel[:, 1]) - 1
        dpixel = np.c_[dpixel, NDVI, GCVI]

        # --- Temporal position encoding -------------------------------------
        # Append sinusoidal DOY features so the encoder can distinguish
        # observations at different positions within the growing season.
        # These two columns are stripped again before the final tensor is
        # returned; they exist solely to preserve temporal order during sampling.
        dpixel = np.c_[
            dpixel,
            np.sin(2 * np.pi * np.arange(ndates) / ndates),
            np.cos(2 * np.pi * np.arange(ndates) / ndates),
        ]

        # Per-pixel, per-feature standardisation. A small epsilon (1e-6)
        # prevents division by zero for constant-valued feature columns.
        dpixel = (dpixel - dpixel.mean(axis=0)) / (dpixel.std(axis=0) + 1e-6)

        # --- Sparse temporal sampling (augmentation) ------------------------
        # Two independent samples of n_ssamples timestep indices are drawn
        # without replacement. Sorting the indices preserves temporal order
        # within each view.
        sparsesample1 = dpixel[
            np.sort(random.sample(range(ndates - self.n_masked), self.samplesize)), :
        ]
        sparsesample2 = dpixel[
            np.sort(random.sample(range(ndates - self.n_masked), self.samplesize)), :
        ]

        # Strip the DOY encoding columns (last two) before creating tensors.
        # NaN values (which can arise from band ratios at zero denominators)
        # are replaced with zero to prevent gradient corruption.
        s1 = torch.nan_to_num(
            torch.from_numpy(sparsesample1[:, :-2].astype(float)).float()[None, :]
        )
        s2 = torch.nan_to_num(
            torch.from_numpy(sparsesample2[:, :-2].astype(float)).float()[None, :]
        )

        return (s1, s2), id

    def __len__(self):
        """Return the number of d-pixels in the dataset.

        Returns:
            int: Total number of rows (pixels) in the underlying array.
        """
        return np.asarray(self.data).shape[0]


def _make_dataset(data, n_augmentations, n_ssamples, cmask, n_masked, maskset, nbands, startindex):
    """Construct a dataset by optionally concatenating multiple augmented copies.

    When ``n_augmentations > 0``, the dataset is replicated ``n_augmentations``
    times via ``ConcatDataset``. Because ``CustomDPixDataset`` samples
    stochastically on each call to ``__getitem__``, each replicated copy yields
    a different set of sparse temporal views, effectively multiplying the number
    of training pairs seen per epoch.

    Args:
        data (np.ndarray): Dataset array of shape ``(n_pixels, n_cols)``.
        n_augmentations (int): Number of dataset copies to concatenate.
            Pass ``0`` to return a single copy.
        n_ssamples (int): Timesteps per sparse sample.
        cmask (int): Cloud-masking mode (0, 1, or 2).
        n_masked (int): Number of dates to exclude from the sampling range.
        maskset (np.ndarray or list): Per-pixel date indices to remove (cmask=2).
        nbands (int): Number of spectral bands per timestep.
        startindex (int): Column index at which band data begins.

    Returns:
        CustomDPixDataset or ConcatDataset: A single dataset or a concatenation
            of ``n_augmentations`` independent copies.
    """
    if n_augmentations == 0:
        return CustomDPixDataset(data, n_ssamples, cmask, n_masked, maskset, nbands, startindex)
    return torch.utils.data.ConcatDataset([
        CustomDPixDataset(data, n_ssamples, cmask, n_masked, maskset, nbands, startindex)
        for _ in range(n_augmentations)
    ])


def create_dataloaders(
    dataset, n_augmentations, batch_size, num_workers,
    n_ssamples, n_masked, maskset, nbands, cmask, shuffle_val, startindex,
):
    """Create a DataLoader for training or validation with augmentation.

    The number of augmented copies is doubled internally (``n_augmentations * 2``)
    to ensure that each d-pixel contributes multiple distinct view pairs per
    epoch, consistent with the original training protocol.

    Args:
        dataset (np.ndarray): Dataset array of shape ``(n_pixels, n_cols)``.
        n_augmentations (int): Number of augmentation replications before
            doubling. Pass ``0`` to use a single un-replicated copy.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of parallel data-loading workers.
        n_ssamples (int): Timesteps per sparse sample.
        n_masked (int): Number of dates excluded from the sampling range.
        maskset (np.ndarray or list): Per-pixel mask indices (cmask=2 only).
        nbands (int): Number of spectral bands per timestep.
        cmask (int): Cloud-masking mode (0, 1, or 2).
        shuffle_val (bool): Whether to shuffle the DataLoader.
        startindex (int): Column index at which band data begins.

    Returns:
        DataLoader: A configured PyTorch DataLoader with ``drop_last=True``
            to ensure all batches are the same size.
    """
    # Multiply by 2 so that each epoch presents twice as many augmented pairs
    # as the nominal n_augmentations value, matching the original training setup.
    ds = _make_dataset(
        dataset, n_augmentations * 2, n_ssamples, cmask, n_masked, maskset, nbands, startindex
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle_val,
        num_workers=num_workers, drop_last=True,
    )


def create_test_dataloader(
    test_data, n_augmentations, batch_size, num_workers,
    n_ssamples, n_masked, maskset, nbands, cmask, shuffle_val, startindex,
):
    """Create a DataLoader for test or prediction, without the 2× augmentation factor.

    Unlike ``create_dataloaders``, this function does **not** double
    ``n_augmentations``, as the test protocol uses the raw replication count
    to control how many embedding samples are extracted per pixel.

    Args:
        test_data (np.ndarray): Dataset array of shape ``(n_pixels, n_cols)``.
        n_augmentations (int): Number of dataset copies to concatenate.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of parallel data-loading workers.
        n_ssamples (int): Timesteps per sparse sample.
        n_masked (int): Number of dates excluded from the sampling range.
        maskset (np.ndarray or list): Per-pixel mask indices (cmask=2 only).
        nbands (int): Number of spectral bands per timestep.
        cmask (int): Cloud-masking mode (0, 1, or 2).
        shuffle_val (bool): Unused; shuffle is always ``False`` for evaluation.
        startindex (int): Column index at which band data begins.

    Returns:
        DataLoader: A configured PyTorch DataLoader with ``shuffle=False`` and
            ``drop_last=True``.
    """
    ds = _make_dataset(
        test_data, n_augmentations, n_ssamples, cmask, n_masked, maskset, nbands, startindex
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True,
    )


class BarlowTwinsDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for unlabelled d-pixel datasets with VIs.

    Handles CSV loading, unique-ID assignment, cloud-mask construction, and
    DataLoader creation for the train, validation, test, and predict stages.
    All configuration is provided at construction time; ``setup()`` is called
    automatically by the Lightning ``Trainer`` before each stage begins.

    Args:
        train_data_dir (str): Path to the training CSV file.
        val_data_dir (str): Path to the validation CSV file.
        test_data_dir (str): Path to the test / prediction CSV file.
        batch_size (int): Number of d-pixels per batch.
        num_workers (int): Number of subprocesses for data loading.
        cmask (int): Cloud-masking mode applied to all splits.
            ``0`` – retain clear-sky dates (flag == 1).
            ``1`` – retain cloud-free dates (flag == 0).
            ``2`` – artificially mask ``n_masked`` random dates per pixel.
        n_masked (int): Number of dates to mask per pixel when ``cmask=2``.
        n_augmentations (int): Number of stochastic augmentation replications.
            The training loader doubles this value internally.
        n_ssamples (int): Number of timesteps per sparse temporal sample.
        nbands (int): Number of spectral bands per timestep (excluding flag).
        startindex (int): Column index at which the band data begins.
    """

    def __init__(
        self,
        train_data_dir="./",
        val_data_dir="./",
        test_data_dir="./",
        batch_size=128,
        num_workers=0,
        cmask=1,
        n_masked=0,
        n_augmentations=0,
        n_ssamples=0,
        nbands=9,
        startindex=3,
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
        self.startindex = startindex

    def _load_csv(self, path):
        """Load a CSV into a NumPy array and assign contiguous unique row IDs.

        The value in column 1 (the original ID field) is overwritten with a
        contiguous integer sequence starting at 1. This ensures each d-pixel
        has a unique identifier regardless of any duplication or shuffling in
        the source file.

        Args:
            path (str): Path to the CSV file to load.

        Returns:
            np.ndarray: Array of shape ``(n_rows, n_cols)`` with column 1
                replaced by ``[1, 2, ..., n_rows]``.
        """
        data = np.asarray(pd.read_csv(path))
        data[:, 1] = np.arange(1, data.shape[0] + 1)
        return data

    def setup(self, stage=None):
        """Load the appropriate CSV splits for the requested training stage.

        Called automatically by the Lightning ``Trainer`` before the
        corresponding dataloaders are first accessed.

        Args:
            stage (str or None): One of ``"fit"``, ``"test"``, ``"predict"``,
                or ``None``. When ``None``, all splits are loaded.
        """
        if stage in ("fit", None):
            self.data_train = self._load_csv(self.train_data_dir)
            self.data_val = self._load_csv(self.val_data_dir)
        if stage in ("test", "predict", None):
            self.data_test = self._load_csv(self.test_data_dir)

    def train_dataloader(self):
        """Return the training DataLoader with shuffling and augmentation.

        Returns:
            DataLoader: Shuffled loader with ``n_augmentations * 2`` dataset
                copies and ``drop_last=True``.
        """
        maskset = create_maskset(self.data_train, self.n_masked) if self.cmask == 2 else []
        return create_dataloaders(
            self.data_train, self.n_augmentations, self.batch_size,
            self.num_workers, self.n_ssamples, self.n_masked,
            maskset, self.nbands, self.cmask, True, self.startindex,
        )

    def val_dataloader(self):
        """Return the validation DataLoader without shuffling.

        Returns:
            DataLoader: Non-shuffled loader with ``n_augmentations * 2`` dataset
                copies and ``drop_last=True``.
        """
        maskset = create_maskset(self.data_val, self.n_masked) if self.cmask == 2 else []
        return create_dataloaders(
            self.data_val, self.n_augmentations, self.batch_size,
            self.num_workers, self.n_ssamples, self.n_masked,
            maskset, self.nbands, self.cmask, False, self.startindex,
        )

    def test_dataloader(self):
        """Return the test DataLoader sized to yield all embeddings in one batch.

        The batch size is set to ``n_pixels * n_augmentations`` so that a
        single forward pass collects all embedding samples at once, which is
        required by the downstream representation-extraction workflow.

        Returns:
            DataLoader: Non-shuffled loader with a single large batch.
        """
        self.batch_size = int(self.data_test.shape[0]) * self.n_augmentations
        return create_test_dataloader(
            self.data_test, self.n_augmentations, self.batch_size,
            self.num_workers, self.n_ssamples, self.n_masked,
            [], self.nbands, self.cmask, False, self.startindex,
        )

    def predict_dataloader(self):
        """Return the prediction DataLoader, identical in construction to the test loader.

        Returns:
            DataLoader: Non-shuffled loader with a single large batch.
        """
        self.batch_size = int(self.data_test.shape[0]) * self.n_augmentations
        return create_test_dataloader(
            self.data_test, self.n_augmentations, self.batch_size,
            self.num_workers, self.n_ssamples, self.n_masked,
            [], self.nbands, self.cmask, False, self.startindex,
        )
