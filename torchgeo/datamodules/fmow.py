# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Functional Map of the World datamodule."""

from typing import Any

import torch

from ..datasets import FMoW
from .geo import NonGeoDataModule
from .utils import collate_fn_detection


class FMoWDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the fMoW dataset.

    .. versionadded:: 0.11
    """

    # fMoW-RGB has no published per-channel statistics (checked the official
    # https://github.com/fMoW/baseline preprocessing code and SatMAE's fMoW paper
    # appendix, neither documents one). These are the standard ImageNet RGB
    # statistics, used as a documented, well-understood default rather than an
    # fMoW-specific value, until real per-channel statistics are computed.
    mean = torch.tensor([123.675, 116.28, 103.53])
    std = torch.tensor([58.395, 57.12, 57.375])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new FMoWDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.FMoW`.
        """
        super().__init__(FMoW, batch_size, num_workers, **kwargs)
        self.collate_fn = collate_fn_detection
