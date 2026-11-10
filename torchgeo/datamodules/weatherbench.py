# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""WeatherBench datamodule."""

from collections.abc import Sequence
from typing import Any

import pandas as pd
import torch

from ..datasets import WeatherBench2
from ..datasets.utils import Sample
from ..samplers import SequentialTimedeltaSampler
from .geo import GeoDataModule


def _collate_fn(samples: Sequence[Sample]) -> Sample:
    """Collate weather samples sharing a grid and static fields.

    Args:
        samples: Weather samples with input and target fields.

    Returns:
        A batch with shared 2D fields and timestamps kept in nanoseconds.
    """
    return {
        key: value
        if value.ndim == 2
        else torch.stack([sample[key] for sample in samples])
        for key, value in samples[0].items()
    }


class WeatherBench2DataModule(GeoDataModule):
    """LightningDataModule implementation for the WeatherBench 2 dataset.

    Uses chronological train/validation/test splits. Each window stays within
    its split, and predict uses the test split. Values stay in physical units.
    The dataset splits each window into input and target tensors.
    Static fields remain 2D, and timestamps are kept in nanoseconds.

    .. versionadded:: 0.11
    """

    def __init__(
        self,
        batch_size: int = 1,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        input_steps: int = 2,
        target_steps: int = 1,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new WeatherBench2DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            val_split_pct: Fraction of timesteps used for validation.
            test_split_pct: Fraction of timesteps used for testing.
            input_steps: Number of input time steps to use.
            target_steps: Number of target time steps to use.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.WeatherBench2`.
        """
        super().__init__(
            WeatherBench2,
            batch_size,
            num_workers=num_workers,
            target_steps=target_steps,
            **kwargs,
        )

        self.collate_fn = _collate_fn
        self.input_steps = input_steps

        level = kwargs.get('level')
        if isinstance(level, int | float):
            self.kwargs['level'] = [level]
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = WeatherBench2(**self.kwargs)
        self.dataset = dataset
        times = dataset.data.time
        timestep = pd.Timestamp(times[1].values) - pd.Timestamp(times[0].values)
        window_size = self.input_steps + dataset.target_steps
        num_times = dataset.data.sizes['time']
        val_size = round(self.val_split_pct * num_times)
        test_size = round(self.test_split_pct * num_times)
        train_size = num_times - val_size - test_size

        def sampler(start: int, stop: int) -> SequentialTimedeltaSampler:
            return SequentialTimedeltaSampler(
                dataset,
                delta=(window_size - 1) * timestep,
                stride=timestep,
                toi=pd.Interval(
                    pd.Timestamp(times[start].values),
                    pd.Timestamp(times[stop - 1].values),
                    closed='both',
                ),
            )

        if stage in ['fit']:
            self.train_sampler = sampler(0, train_size)
        if stage in ['fit', 'validate']:
            self.val_sampler = sampler(train_size, train_size + val_size)
        if stage in ['test']:
            self.test_sampler = sampler(train_size + val_size, num_times)
        if stage in ['predict']:
            self.predict_sampler = sampler(train_size + val_size, num_times)

    def on_after_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        """Skip normalization and keep values in physical units.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            The unchanged batch.
        """
        return batch
