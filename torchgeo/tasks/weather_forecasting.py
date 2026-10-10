# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tasks for weather forecasting."""

from collections.abc import Mapping
from typing import Any, Literal

import torch
from torch import Tensor
from torchmetrics import MeanSquaredError, MetricCollection
from torchvision.models._api import WeightsEnum

from ..datasets.utils import Sample
from ..models import aurora_swin_unet
from .base import BaseTask
from .mixins import RegressionMixin


class WeatherForecasting(RegressionMixin, BaseTask):
    """One-step weather forecasting.

    Uses ``history`` previous states to predict the next state. Surface tensors
    have shape ``(B, T, H, W)``, atmospheric tensors have shape ``(B, T, L, H, W)``,
    and static tensors have shape ``(B, H, W)`` or ``(B, T, H, W)``.

    The model returns predicted variables under the same names with shape
    ``(B, 1, H, W)`` or ``(B, 1, L, H, W)``.

    .. versionadded:: 0.11
    """

    def __init__(
        self,
        model: str = 'aurora',
        weights: WeightsEnum | None = None,
        variable_mapping: Mapping[str, str] | None = None,
        history: int = 2,
        loss: Literal['mae', 'mse'] = 'mae',
        lr: float = 1e-4,
        patience: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize a new WeatherForecasting instance.

        Args:
            model: Weather forecasting model name. Supported value is ``'aurora'``.
            weights: Initial model weights.
            variable_mapping: Mapping from dataset variable names to model names.
                Keys are the variables to predict.
            history: Number of previous states provided to predict the next state.
            loss: Loss function, one of ``'mse'`` or ``'mae'``.
            lr: Learning rate for the optimizer.
            patience: Patience for the learning rate scheduler.
            **kwargs: Additional keyword arguments passed to the model constructor.
        """
        self.weights = weights
        self.kwargs = kwargs
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* is invalid.
        """
        model: str = self.hparams['model']
        if model == 'aurora':
            self.model = aurora_swin_unet(
                weights=self.weights,
                variable_mapping=self.hparams['variable_mapping'],
                **self.kwargs,
            )
        else:
            msg = f"Invalid model type '{model}'. Supported model: 'aurora'"
            raise ValueError(msg)

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        metrics = MetricCollection(
            {
                f'{name}_RMSE': MeanSquaredError(squared=False)
                for name in self.hparams['variable_mapping']
            },
            compute_groups=False,
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, batch: Sample) -> Sample:
        """Predict the next weather state from the requested history.

        Args:
            batch: Weather states and coordinates.

        Returns:
            Predicted tensors under their dataset variable names, with a time
            dimension of length one.
        """
        history = self.hparams['history']
        x = {
            name: value[:, :history] if name == 'time' or value.ndim >= 4 else value
            for name, value in batch.items()
        }
        y_hat: Sample = self.model(x)
        return y_hat

    def _shared_step(self, batch: Sample, stage: str) -> Tensor:
        """Compute the loss and metrics for a given stage."""
        history = self.hparams['history']
        batch_size = batch['time'].shape[0]
        predictions = self(batch)
        metrics = getattr(self, f'{stage}_metrics')
        losses: list[Tensor] = []
        for name in self.hparams['variable_mapping']:
            y_hat = predictions[name]
            y = batch[name][:, history].unsqueeze(dim=1)
            y = y[..., : y_hat.shape[-2], : y_hat.shape[-1]].to(y_hat)
            if y_hat.shape != y.shape:
                raise ValueError('Predictions and targets must have the same shape.')
            losses.append(self.criterion(y_hat, y))
            metrics[f'{stage}_{name}_RMSE'].update(y_hat.flatten(), y.flatten())
        loss = torch.stack(losses).mean()
        self.log(f'{stage}_loss', loss, batch_size=batch_size)
        return loss

    def training_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics."""
        return self._shared_step(batch, 'train')

    def validation_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics."""
        self._shared_step(batch, 'val')

    def test_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics."""
        self._shared_step(batch, 'test')

    def predict_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> Sample:
        """Predict the next state from the requested history."""
        return self(batch)
