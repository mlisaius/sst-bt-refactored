"""Public API for the datamodules package.

Exports the primary data module used for SSL pre-training on unlabelled
satellite time-series d-pixels with on-the-fly vegetation index computation.
"""

from .btdmunlabelledVIs import BarlowTwinsDataModule as BarlowTwinsDataModuleUVI

__all__ = ("BarlowTwinsDataModuleUVI",)
