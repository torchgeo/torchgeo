# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""BioMassters datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch.utils.data import random_split

from ..datasets import BioMassters
from .geo import NonGeoDataModule


class BioMasstersDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the BioMassters dataset.

    Samples have the fused spatiotemporal regression format
    ``{'image': (T, C, H, W), 'mask': (H, W)}``.

    .. versionadded:: 0.11
    """

    # The task uses these to denormalize predictions; masks remain unnormalized.
    target_mean = 0.0
    target_std = 1.0
    # Training statistics published by https://github.com/quqixun/BioMassters.
    s1_mean = torch.tensor([-11.440976, -18.056156, -12.975023, -24.132893])
    s1_std = torch.tensor([3.167056, 4.362354, 5.392603, 17.264702])

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split_pct: float = 0.3,
        **kwargs: Any,
    ) -> None:
        """Initialize a new BioMasstersDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the labeled train split used for validation.
            **kwargs: Additional keyword arguments passed to the dataset.
        """
        super().__init__(
            BioMassters, batch_size, num_workers, as_time_series=True, **kwargs
        )
        self.lengths = (1 - val_split_pct, val_split_pct)

        sensors = kwargs.get('sensors', BioMassters.valid_sensors)
        means = {'S1': self.s1_mean, 'S2': torch.zeros(11)}
        stds = {'S1': self.s1_std, 'S2': torch.ones(11)}
        mean = torch.cat([means[sensor] for sensor in sensors])
        std = torch.cat([stds[sensor] for sensor in sensors])
        self.aug = K.AugmentationSequential(
            K.VideoSequential(K.Normalize(mean=mean, std=std)),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            self.dataset = BioMassters(split='train', **self.kwargs)
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, self.lengths, torch.Generator().manual_seed(0)
            )
        if stage == 'test':
            self.test_dataset = BioMassters(split='test', **self.kwargs)
        if stage == 'predict':
            self.predict_dataset = BioMassters(split='test', **self.kwargs)
