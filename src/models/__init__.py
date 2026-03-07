"""Public API for the models package.

Exports:
    BarlowTwinsUVIsSp: Lightning module for unlabelled SSL pre-training.
    BarlowTwinsVIs: Lightning module for labelled evaluation (checkpoint loading
        and embedding extraction).
"""

from .barlowtwins_unlabelled_VIs_speed import BarlowTwins as BarlowTwinsUVIsSp
from .barlowtwins_VIs import BarlowTwins as BarlowTwinsVIs

__all__ = ("BarlowTwinsUVIsSp", "BarlowTwinsVIs")
