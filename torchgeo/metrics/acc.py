# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Anomaly correlation coefficient (ACC)."""

from collections.abc import Sequence
from typing import Any

import einops
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import Metric

from .utils import latitude_weights


class AnomalyCorrCoef(Metric):
    r"""Compute Anomaly correlation coefficient (ACC).

    ACC is computed as the Pearson correlation coefficient of the anomalies
    with respect to the climatology :math:`c`:

    .. math::

       f_{t,l,i,j}^\prime = f_{t,l,i,j} - c_{t,l,i,j};
       \quad o_{t,i,j}^\prime = o_{t,i,j} - c_{t,i,j}

    where :math:`c_{t,l,i,j}` and :math:`c_{t,i,j}` indicate the climatology
    corresponding to the appropriate *tod-doy* combination. The ACC then is defined as:

    .. math::

       \text{ACC}_l = \frac{1}{T} \sum_t^T \frac{
           \sum_i^I \sum_j^J w(i) f_{t,l,i,j}^\prime o_{t,i,j}^\prime
       }{
           \sqrt{
               \sum_i^I \sum_j^J w(i) {f_{t,l,i,j}^\prime}^2
               \sum_i^I \sum_j^J w(i) {o_{t,i,j}^\prime}^2
           }
       }

    Where:

    =========  ====================  =================
    Symbol     Range                 Description
    =========  ====================  =================
    :math:`f`                        Forecast
    :math:`o`                        Observation
    :math:`c`                        Climatology
    :math:`w`                        Latitude weight
    :math:`t`  :math:`1, \ldots, T`  Verification time
    :math:`l`  :math:`1, \ldots, L`  Lead time
    :math:`i`  :math:`1, \ldots, I`  Latitude index
    :math:`j`  :math:`1, \ldots, J`  Longitude index
    =========  ====================  =================

    The range of ACC values goes from 1, indicating perfect correlation,
    to -1, indicating perfect anti-correlation. A climatological forecast
    has an ACC value of zero.

    For more details on this formulation, see equation 5 in:

    * https://arxiv.org/abs/2308.15560

    .. versionadded:: 0.11
    """

    is_differentiable = True
    higher_is_better = None
    full_state_update = True
    plot_lower_bound = -1.0
    plot_upper_bound = 1.0

    ratio: Tensor
    total: Tensor

    def __init__(
        self, latitude: Tensor, num_outputs: tuple[int, ...], **kwargs: Any
    ) -> None:
        r"""Initialize a new AnomalyCorrCoef instance.

        Args:
            latitude: Latitude of :math:`f` and :math:`o` in degrees
                :math:`[-90, \ldots, 90]`.
            num_outputs: Dimensions of output metric after mean across
                :math:`t`, :math:`i`, and :math:`j` dimensions.
        """
        super().__init__(**kwargs)

        self.weights = einops.rearrange(latitude_weights(latitude), 'y -> y 1')

        self.add_state('ratio', default=torch.zeros(num_outputs), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Forecast - climatology of shape
                [:math:`B`, ..., :math:`H`, :math:`W`].
            target: Observations - climatology of shape
                [:math:`B`, ..., :math:`H`, :math:`W`].
        """
        covariance = torch.sum(self.weights * preds * target, dim=[-2, -1])
        variance1 = torch.sum(self.weights * preds**2, dim=[-2, -1])
        variance2 = torch.sum(self.weights * target**2, dim=[-2, -1])
        self.ratio += torch.sum(covariance / torch.sqrt(variance1 * variance2), dim=0)
        self.total += preds.shape[0]

    def compute(self) -> Tensor:
        """Compute anomaly correlation coefficient over state.

        Returns:
            Anomaly correlation coefficient.
        """
        return self.ratio / self.total

    def plot(
        self, val: Tensor | Sequence[Tensor] | None = None, ax: Axes | None = None
    ) -> tuple[Figure, Axes]:
        """Plot a single or multiple values from the metric.

        Args:
            val: The output of :meth:`AnomalyCorrCoef.compute` or a list of these
                results. If *val* is not provided, will automatically call
                :meth:`AnomalyCorrCoef.compute` and plot that result.
            ax: A matplotlib axis object. If provided will add plot to that axis.

        Returns:
            The resulting figure and axis.
        """
        return self._plot(val, ax)  # ty: ignore[invalid-return-type]
