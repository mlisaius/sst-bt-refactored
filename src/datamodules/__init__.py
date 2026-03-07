"""Public API for the datamodules package.

Exports:
    BarlowTwinsDataModuleUVI: Lightning DataModule for unlabelled SSL pre-training.
    create_test_dataloaderVIb: Dataloader factory for labelled evaluation data.
    BarlowTwinsDataModuleVIs: Lightning DataModule for labelled fine-tuning data.
"""

# BarlowTwinsDataModuleUVI wraps the *unlabelled* CSV data used during SSL
# pre-training.  Each sample is a d-pixel with no crop-type label; the
# datamodule applies stochastic sparse temporal sampling to produce two views.
from .btdmunlabelledVIs import BarlowTwinsDataModule as BarlowTwinsDataModuleUVI

# create_test_dataloaderVIb builds a single-batch dataloader from a *labelled*
# CSV for embedding extraction in evaluate.py.  The full dataset is loaded in
# one batch so all embeddings can be collected at once for the RF classifier.
from .btdmVIs import create_test_dataloader as create_test_dataloaderVIb

# BarlowTwinsDataModuleVIs wraps the *labelled* train/val/test CSV splits used
# during fine-tuning.  Each sample carries a crop-type label and pixel ID so
# that classification loss and accuracy can be computed in STBTClassification.
from .btdmVIs import BarlowTwinsDataModule as BarlowTwinsDataModuleVIs

__all__ = ("BarlowTwinsDataModuleUVI", "create_test_dataloaderVIb", "BarlowTwinsDataModuleVIs")
