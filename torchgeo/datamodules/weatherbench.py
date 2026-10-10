# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""WeatherBench datamodule."""

from typing import Any

import pandas as pd

from ..datasets import WeatherBench2
from ..datasets.utils import Sample
from ..samplers import SequentialTimedeltaSampler
from .geo import GeoDataModule


class WeatherBench2DataModule(GeoDataModule):
    """LightningDataModule implementation for the WeatherBench 2 dataset.

    Uses chronological train/validation/test splits. Each window stays within
    its split, and predict uses the test split. Values stay in physical units.

    .. versionadded:: 0.11
    """

    def __init__(
        self,
        batch_size: int = 1,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        timestep: str = '6h',
        history: int = 2,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new WeatherBench2DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            val_split_pct: Fraction of timesteps used for validation.
            test_split_pct: Fraction of timesteps used for testing.
            timestep: Time between consecutive states.
            history: Number of input states before the target state.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.WeatherBench2`.
        """
        super().__init__(WeatherBench2, batch_size, num_workers=num_workers, **kwargs)

        self.timestep = pd.Timedelta(timestep)
        self.history = history

        level = kwargs.get('level')
        if isinstance(level, int | float):
            self.kwargs['level'] = [level]
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.

        Raises:
            ValueError: If a split cannot contain a window.
        """
        self.dataset = WeatherBench2(**self.kwargs)
        times = pd.DatetimeIndex(self.dataset.data.time.values)
        val_size = round(self.val_split_pct * len(times))
        test_size = round(self.test_split_pct * len(times))
        train_size = len(times) - val_size - test_size
        test = times[train_size + val_size :]

        def sampler(
            timestamps: pd.DatetimeIndex, split: str
        ) -> SequentialTimedeltaSampler:
            return SequentialTimedeltaSampler(
                self.dataset,
                delta=self.history * self.timestep,
                stride=self.timestep,
                toi=pd.Interval(timestamps[0], timestamps[-1], closed='both'),
            )

        if stage in ['fit']:
            self.train_sampler = sampler(times[:train_size], 'train')
        if stage in ['fit', 'validate']:
            self.val_sampler = sampler(times[train_size : train_size + val_size], 'val')
        if stage in ['test']:
            self.test_sampler = sampler(test, 'test')
        if stage in ['predict']:
            self.predict_sampler = sampler(test, 'predict')

    def on_after_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        """Skip normalization and keep values in physical units.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            The unchanged batch.
        """
        return batch
