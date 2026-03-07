"""Public API for the models package.

Exports the primary Barlow Twins Lightning module used for SSL pre-training
on satellite time-series d-pixels.
"""

from .barlowtwins_unlabelled_VIs_speed import BarlowTwins as BarlowTwinsUVIsSp

__all__ = ("BarlowTwinsUVIsSp",)
