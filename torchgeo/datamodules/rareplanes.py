# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""RarePlanes datamodule."""

from typing import Any

import torch
from torch.utils.data import random_split

from ..datasets import RarePlanes
from .geo import NonGeoDataModule
from .utils import collate_fn_detection


class RarePlanesDataModule(NonGeoDataModule):
    """LightningDataModule for the RarePlanes object detection dataset.

    Validation uses a fixed 20% split of the training samples. Testing and
    prediction use the official test split. For batches larger than one, all
    images in the batch must be the same size.

    .. versionadded:: 0.11
    """

    std = torch.tensor(255)

    def __init__(
        self, batch_size: int = 1, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new RarePlanesDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.RarePlanes`.
        """
        super().__init__(RarePlanes, batch_size, num_workers, **kwargs)
        self.collate_fn = collate_fn_detection

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            dataset = RarePlanes(split='train', **self.kwargs)
            self.train_dataset, self.val_dataset = random_split(
                dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
            )
        if stage == 'test':
            self.test_dataset = RarePlanes(split='test', **self.kwargs)
        if stage == 'predict':
            self.predict_dataset = RarePlanes(split='test', **self.kwargs)
