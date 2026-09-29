# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import torch

from torchgeo.datamodules import PASTIS100DataModule, PASTISDataModule


def test_normalization() -> None:
    for datamodule in [PASTISDataModule, PASTIS100DataModule]:
        assert torch.equal(datamodule.mean, torch.tensor(0))
        assert torch.equal(datamodule.std, torch.tensor(10000))
