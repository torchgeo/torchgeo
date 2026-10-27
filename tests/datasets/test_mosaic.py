# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import nn

from torchgeo.datasets import DatasetNotFoundError, MosaicV0, RGBBandsMissingError

root = os.path.join('tests', 'data', 'mosaic')


class TestMosaicV0:
    @pytest.fixture(
        params=[
            ('sen1floods11', 'train', MosaicV0.s1_bands, False),
            ('pilot', 'test', MosaicV0.s2_bands, True),
            ('event', 'test', MosaicV0.all_bands, False),
        ]
    )
    def dataset(self, request: SubRequest) -> MosaicV0:
        split_by, split, bands, include_sources = request.param
        transforms = nn.Identity()
        return MosaicV0(
            root,
            split=split,
            split_by=split_by,
            bands=bands,
            include_sources=include_sources,
            transforms=transforms,
        )

    def test_getitem(self, dataset: MosaicV0) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert x['image'].shape == (len(dataset.bands), 32, 32)
        assert x['image'].dtype == torch.float32
        assert not x['image'].isnan().any()
        assert x['mask'].shape == (32, 32)
        assert x['mask'].dtype == torch.int64
        assert x['posterior'].shape == (32, 32)
        assert x['flags'].dtype == torch.uint8
        assert x['landcover'].max() < len(dataset.landcover_groups)
        if dataset.include_sources:
            assert x['sources'].shape == (4, 32, 32)
            assert set(x['sources'].unique().tolist()) <= {-1, 0, 1}
        else:
            assert 'sources' not in x

    @pytest.mark.parametrize(
        'split_by,split,length',
        [
            ('sen1floods11', 'train', 2),
            ('sen1floods11', 'valid', 1),
            ('sen1floods11', 'test', 1),
            ('sen1floods11', 'bolivia', 1),
            ('pilot', 'anchor', 1),
            ('pilot', 'test', 4),
            ('event', 'train', 3),
            ('event', 'test', 2),
        ],
    )
    def test_len(
        self,
        split_by: Literal['sen1floods11', 'pilot', 'event'],
        split: str,
        length: int,
    ) -> None:
        ds = MosaicV0(root, split=split, split_by=split_by, fold=2)
        assert len(ds) == length

    def test_posterior_setting(self) -> None:
        f = MosaicV0(root, posterior='f')[0]['posterior']
        g = MosaicV0(root, posterior='g')[0]['posterior']
        assert not torch.equal(f.nan_to_num(), g.nan_to_num())

    def test_download(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        url = os.path.join(root, MosaicV0.filename)
        monkeypatch.setattr(MosaicV0, 'url', url)
        monkeypatch.setattr(
            MosaicV0, 'originals_url', os.path.join(root, 'sen1floods11', '')
        )
        MosaicV0(tmp_path, include_sources=True, download=True)

    def test_extract(self, tmp_path: Path) -> None:
        shutil.copy(os.path.join(root, MosaicV0.filename), tmp_path)
        shutil.copytree(os.path.join(root, 'sen1floods11'), tmp_path / 'sen1floods11')
        MosaicV0(tmp_path, checksum=False)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(tmp_path / MosaicV0.filename, 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted.'):
            MosaicV0(tmp_path, checksum=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            MosaicV0(tmp_path)

    def test_originals_not_downloaded(self, tmp_path: Path) -> None:
        shutil.copytree(
            os.path.join(root, MosaicV0.directory), tmp_path / MosaicV0.directory
        )
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            MosaicV0(tmp_path)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            MosaicV0(root, split='anchor')

    def test_invalid_fold(self) -> None:
        with pytest.raises(AssertionError):
            MosaicV0(root, split='test', split_by='event', fold=99)

    def test_invalid_bands(self) -> None:
        with pytest.raises(AssertionError):
            MosaicV0(root, bands=('foo',))

    def test_plot(self, dataset: MosaicV0) -> None:
        x = dataset[0]
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask'].clone()
        dataset.plot(x)
        plt.close()

    def test_plot_rgb(self) -> None:
        dataset = MosaicV0(root, bands=('B1',))
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            dataset.plot(dataset[0])
