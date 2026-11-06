# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from _pytest.fixtures import SubRequest
from torch import nn

from torchgeo.datasets import DatasetNotFoundError, RarePlanes


class TestRarePlanes:
    @pytest.fixture(params=product(['real', 'synthetic'], ['train', 'test']))
    def dataset(self, request: SubRequest) -> RarePlanes:
        root = os.path.join('tests', 'data', 'rareplanes')
        dataset_type, split = request.param
        return RarePlanes(root, dataset_type, split, transforms=nn.Identity())

    def test_getitem(self, dataset: RarePlanes) -> None:
        sample = dataset[0]
        assert sample['image'].shape == (3, 8, 8)
        assert torch.allclose(
            sample['bbox_xyxy'], torch.tensor([[1, 2, 4, 6]], dtype=torch.float32)
        )
        assert torch.equal(sample['label'], torch.tensor([1]))

    def test_len(self, dataset: RarePlanes) -> None:
        assert len(dataset) == 1

    def test_invalid_dataset_type(self) -> None:
        with pytest.raises(AssertionError):
            RarePlanes(
                dataset_type='invalid'  # ty: ignore[invalid-argument-type]
            )

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            RarePlanes(split='invalid')  # ty: ignore[invalid-argument-type]

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            RarePlanes(tmp_path)

    @pytest.mark.parametrize('predictions', [False, True])
    def test_plot(self, dataset: RarePlanes, predictions: bool) -> None:
        sample = dataset[0]
        if predictions:
            sample['prediction_bbox_xyxy'] = torch.tensor([[2, 1, 6, 4]])
        fig = dataset.plot(sample, show_titles=not predictions, suptitle='RarePlanes')
        assert len(fig.axes) == 1 + predictions
        assert fig.axes[0].patches[0].get_xy() == pytest.approx((1, 2))
        if predictions:
            assert fig.axes[1].patches[0].get_xy() == pytest.approx((2, 1))
        plt.close(fig)
