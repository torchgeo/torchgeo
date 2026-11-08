# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch
from matplotlib import pyplot as plt

from torchgeo.metrics import AnomalyCorrCoef

B = 2
T = 3
Z = 4
Y = 5
X = 6
latitude = torch.tensor([-60.0, -30.0, 0.0, 30.0, 60.0])


@pytest.mark.parametrize('dims', [(), (T,), (T, Z)])
def test_acc(dims: tuple[int, ...]) -> None:
    preds = torch.ones(B, *dims, Y, X)
    target = torch.ones(B, *dims, Y, X)
    expected = torch.ones(dims)
    metric = AnomalyCorrCoef(latitude, dims)
    metric.update(preds, target)  # ty: ignore[invalid-argument-type]
    acc = metric.compute()  # ty: ignore[missing-argument]
    assert torch.allclose(acc, expected)


@pytest.mark.parametrize('dims', [(), (T,)])
def test_plot(dims: tuple[int, ...]) -> None:
    preds = torch.rand(B, *dims, Y, X)
    target = torch.rand(B, *dims, Y, X)
    metric = AnomalyCorrCoef(latitude, dims)
    metric.update(preds, target)  # ty: ignore[invalid-argument-type]
    metric.plot()
    plt.close()
