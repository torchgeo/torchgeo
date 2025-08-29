# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""PASTIS datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch import Tensor
from torch.utils.data import random_split

from ..datasets import PASTIS
from .geo import NonGeoDataModule


class PASTISDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the PASTIS dataset.

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        target_t: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize a new PASTISDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            target_t: Target number of temporal frames. Sequences will be padded or truncated to this length.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.PASTIS`.
        """
        super().__init__(
            PASTIS, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.target_t = target_t
        self.aug = K.AugmentationSequential(
            K.VideoSequential(K.Normalize(mean=self.mean, std=self.std)),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = PASTIS(**self.kwargs)
        generator = torch.Generator().manual_seed(0)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [
                1 - self.val_split_pct - self.test_split_pct,
                self.val_split_pct,
                self.test_split_pct,
            ],
            generator,
        )

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply augmentations to batch after transferring to device.

        Args:
            batch: A batch of data that needs to be augmented.
            dataloader_idx: The index of the dataloader.

        Returns:
            A batch of augmented data.
        """
        if 'image' in batch and batch['image'].ndim == 5:
            B, T, C, H, W = batch['image'].shape
            target_t = self.target_t

            if T < target_t:
                pad = batch['image'].new_zeros((B, target_t - T, C, H, W))
                batch['image'] = torch.cat([batch['image'], pad], dim=1)
            elif T > target_t:
                batch['image'] = batch['image'][:, :target_t, :, :, :]

        return super().on_after_batch_transfer(batch, dataloader_idx)
