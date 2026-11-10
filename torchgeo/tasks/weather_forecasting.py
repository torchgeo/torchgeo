# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tasks for weather forecasting."""

from collections.abc import Sequence
from typing import Any, Literal, cast

import einops
import torch
from torch import Tensor, nn
from torchmetrics import Metric, MetricCollection
from torchmetrics.wrappers import ClasswiseWrapper
from torchvision.models._api import WeightsEnum

from ..datasets.utils import Sample
from ..metrics import MeanSquaredError
from ..models import aurora_swin_unet
from .base import BaseTask


class WeatherForecasting(BaseTask):
    """Weather forecasting from prepared input and target states.

    Batches contain ``input_<variable>`` and ``target_<variable>`` fields. Surface
    tensors have shape ``(B, T, H, W)``, atmospheric tensors have shape
    ``(B, T, L, H, W)``, and static tensors have shape ``(H, W)``.

    The model receives input fields without the ``input_`` prefix and returns
    predicted variables under their dataset names with matching target shapes.

    Latitude-weighted MSE and RMSE are reported separately for each variable,
    forecast lead in hours, and atmospheric pressure level in hPa. Surface and
    atmospheric variables use separate metric collections. Only the batch and
    spatial dimensions are averaged. Batches must share a monotonic latitude
    grid, pressure levels, and forecast leads. Timestamps are Unix nanoseconds.

    For example, ``val_RMSE_temperature_6h_500hPa`` reports the six-hour forecast
    error at 500 hPa. Latitude is reordered only for metric computation; model
    inputs and the training loss retain their original ordering.

    .. versionadded:: 0.11
    """

    def __init__(
        self,
        data_vars: Sequence[str],
        model: Literal['aurora'] = 'aurora',
        weights: WeightsEnum | None = None,
        loss: Literal['mae', 'mse'] = 'mae',
        lr: float = 1e-4,
        patience: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize a new WeatherForecasting instance.

        Args:
            data_vars: Data variables to use for loss and metrics.
            model: Weather forecasting model name. Supported value is ``'aurora'``.
            weights: Initial model weights.
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
        match model:
            case 'aurora':
                self.model = aurora_swin_unet(weights=self.weights, **self.kwargs)
            case _:
                msg = f"Invalid model type '{model}'. Supported model: 'aurora'"
                raise ValueError(msg)

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        match self.hparams['loss']:
            case 'mse':
                self.criterion: nn.Module = nn.MSELoss()
            case 'mae':
                self.criterion = nn.L1Loss()

    def configure_metrics(self, batch: Sample | None = None) -> None:
        """Initialize latitude-weighted MSE and RMSE from the first scoring batch.

        Args:
            batch: Batch supplying the grid, forecast leads, and pressure levels.
                Initialization is deferred until these coordinates are available.
        """
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()
        self.test_metrics = nn.ModuleDict()
        self.metric_variables: dict[str, list[str]] = {}
        if batch is None:
            return

        latitude = batch['target_latitude'][0]
        self.flip_latitude = bool(latitude[0] > latitude[-1])
        if self.flip_latitude:
            latitude = latitude.flip(0)
        leads = (
            (batch['target_time'][0] - batch['input_time'][0, -1]) / (3600 * 1e9)
        ).tolist()
        for group, ndim in [('surface', 4), ('atmosphere', 5)]:
            names = [
                name
                for name in self.hparams['data_vars']
                if batch[f'target_{name}'].ndim == ndim
            ]
            if not names:
                continue
            self.metric_variables[group] = names
            labels = [f'{name}_{lead:g}h' for name in names for lead in leads]
            if group == 'atmosphere':
                labels = [
                    f'{label}_{level:g}hPa'
                    for label in labels
                    for level in batch['target_level'][0].tolist()
                ]
            metrics_dict: dict[str, Metric | MetricCollection] = {}
            for name, squared in [('MSE', True), ('RMSE', False)]:
                metric = MeanSquaredError(latitude, (len(labels),), squared=squared)
                if len(labels) > 1:
                    metrics_dict[name] = ClasswiseWrapper(
                        metric, labels=labels, prefix=f'{name}_'
                    )
                else:
                    metrics_dict[f'{name}_{labels[0]}'] = metric
            metrics = MetricCollection(metrics_dict, compute_groups=False).to(
                self.device
            )
            self.train_metrics[group] = metrics.clone(prefix='train_')
            self.val_metrics[group] = metrics.clone(prefix='val_')
            self.test_metrics[group] = metrics.clone(prefix='test_')

    def forward(self, batch: Sample) -> Sample:
        """Predict weather states from the prepared inputs.

        Args:
            batch: Input and target states and their coordinates.

        Returns:
            Predicted tensors under their dataset variable names.
        """
        x = {
            name.removeprefix('input_'): value
            for name, value in batch.items()
            if name.startswith('input_')
        }
        y_hat: Sample = self.model(x)
        return y_hat

    def _shared_step(self, batch: Sample, stage: str) -> Tensor:
        """Compute the loss and metrics for a given stage.

        Raises:
            ValueError: If predictions and targets have different shapes.
        """
        batch_size = batch['input_time'].shape[0]
        predictions = self(batch)
        targets: Sample = {}
        losses: list[Tensor] = []
        for name in self.hparams['data_vars']:
            y_hat = predictions[name]
            y = batch[f'target_{name}'].to(y_hat)
            if y_hat.shape != y.shape:
                raise ValueError('Predictions and targets must have the same shape.')
            targets[name] = y
            losses.append(self.criterion(y_hat, y))
        if not self.train_metrics:
            # Validation may run first under Lightning's inference mode.
            with torch.inference_mode(False):
                self.configure_metrics(batch)
        metrics = getattr(self, f'{stage}_metrics')
        for group, names in self.metric_variables.items():
            y_hat = torch.stack([predictions[name] for name in names], dim=1)
            y = torch.stack([targets[name] for name in names], dim=1)
            # Flatten output labels without averaging variables, leads, or levels.
            y_hat = einops.rearrange(y_hat, 'b ... h w -> b (...) h w')
            y = einops.rearrange(y, 'b ... h w -> b (...) h w')
            if self.flip_latitude:
                y_hat = y_hat.flip(-2)
                y = y.flip(-2)
            metrics[group].update(y_hat, y)
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
        """Predict weather states from the prepared inputs."""
        return self(batch)

    def on_train_epoch_end(self) -> None:
        """Log train metrics."""
        for collection in self.train_metrics.values():
            metrics = cast(MetricCollection, collection)
            self.log_dict(metrics.compute())
            metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        for collection in self.val_metrics.values():
            metrics = cast(MetricCollection, collection)
            self.log_dict(metrics.compute())
            metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Log test metrics."""
        for collection in self.test_metrics.values():
            metrics = cast(MetricCollection, collection)
            self.log_dict(metrics.compute())
            metrics.reset()
