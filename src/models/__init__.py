"""Public API for the models package.

Exports:
    BarlowTwinsUVIsSp: Lightning module for unlabelled SSL pre-training.
    BarlowTwinsVIs: Lightning module for labelled evaluation (checkpoint loading
        and embedding extraction).
    STBTClassification: Lightning module for two-phase supervised fine-tuning
        of a pre-trained BarlowTwinsUVIsSp encoder.
"""

# BarlowTwinsUVIsSp: full SSL model (encoder + projection head + BT loss).
# Used by train.py for pre-training and by STBTClassification to extract
# the encoder from a saved checkpoint.
from .barlowtwins_unlabelled_VIs_speed import BarlowTwins as BarlowTwinsUVIsSp

# BarlowTwinsVIs: alternate SSL model variant used by evaluate.py to load
# a checkpoint and extract frozen encoder embeddings for RF evaluation.
from .barlowtwins_VIs import BarlowTwins as BarlowTwinsVIs

# STBTClassification: two-phase fine-tuning module.
# Phase 1 (probe) freezes the encoder; Phase 2 (finetune) unfreezes it
# with a lower learning rate to preserve pre-trained representations.
from .classification import STBTClassification

__all__ = ("BarlowTwinsUVIsSp", "BarlowTwinsVIs", "STBTClassification")
