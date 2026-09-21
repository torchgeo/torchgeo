# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import io
import pickle
import threading
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from pyproj import CRS
from pytest import MonkeyPatch

from torchgeo.datasets import WMSDataset
from torchgeo.datasets.utils import Sample

pytest.importorskip('owslib', minversion='0.29.3')

URL = 'https://example.com/wms'
BOUNDS = (-1.0, 50.0, 1.0, 52.0)


class MockLayer:
    def __init__(self, crs_options: list[str] | None) -> None:
        self.boundingBoxWGS84: tuple[float, float, float, float] | None = BOUNDS
        self.crsOptions = crs_options


class MockWebMapService:
    def __init__(self, url: str, version: str = '1.3.0') -> None:
        self.url = url
        self.version = version
        # owslib keeps parsed XML, which cannot be pickled either.
        self.lock = threading.Lock()
        self.contents = {
            'imagery': MockLayer(['EPSG:3857', 'EPSG:4326']),
            'boundaries': MockLayer(['EPSG:3857']),
        }

    def getmap(self, **kwargs: Any) -> io.BytesIO:
        self.request = kwargs
        width, height = kwargs['size']
        return encode(np.zeros((height, width, 3), dtype=np.uint8))


def encode(array: np.ndarray) -> io.BytesIO:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    buffer.seek(0)
    return buffer


class TestWMSDataset:
    @pytest.fixture(autouse=True)
    def mock_service(self, monkeypatch: MonkeyPatch) -> None:
        import owslib.wms

        monkeypatch.setattr(owslib.wms, 'WebMapService', MockWebMapService)

    @pytest.fixture
    def dataset(self) -> WMSDataset:
        return WMSDataset(URL, 'imagery', res=0.01)

    def test_getitem(self, dataset: WMSDataset) -> None:
        x = dataset[-1:0:0.01, 50:51:0.01]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].shape == (3, 100, 100)
        assert x['image'].dtype == torch.float32

    def test_getitem_requests_the_queried_window(self, dataset: WMSDataset) -> None:
        dataset[-1:0:0.5, 50:51:0.25]
        request = dataset.service.request
        assert request['layers'] == ['imagery']
        assert request['srs'] == 'EPSG:4326'
        assert request['bbox'] == (-1.0, 50.0, 0.0, 51.0)
        assert request['size'] == (2, 4)

    def test_crs(self, dataset: WMSDataset) -> None:
        assert dataset.crs == CRS.from_epsg(4326)

    def test_res(self, dataset: WMSDataset) -> None:
        assert dataset.res == (0.01, 0.01)

    def test_explicit_crs(self) -> None:
        crs = CRS.from_epsg(3857)
        dataset = WMSDataset(URL, 'imagery', res=10, crs=crs, format='image/jpeg')
        assert dataset.crs == crs
        assert dataset.srs == 'EPSG:3857'
        assert dataset.index.total_bounds == pytest.approx(
            (-111319.49, 6446275.84, 111319.49, 6800125.45), rel=1e-6
        )

    def test_set_crs(self, dataset: WMSDataset) -> None:
        dataset.crs = CRS.from_epsg(3857)
        dataset[-111319:0:1000, 6446276:6500000:1000]
        assert dataset.service.request['srs'] == 'EPSG:3857'

    def test_set_crs_not_offered(self, dataset: WMSDataset) -> None:
        with pytest.raises(ValueError, match='imagery is not offered in EPSG:32718'):
            dataset.crs = CRS.from_epsg(32718)

    def test_transforms(self) -> None:
        def transforms(sample: Sample) -> Sample:
            sample['image'] += 1
            return sample

        dataset = WMSDataset(URL, 'imagery', res=0.01, transforms=transforms)
        assert dataset[-1:0:0.01, 50:51:0.01]['image'].min() == 1

    def test_non_rgb_response(
        self, dataset: WMSDataset, monkeypatch: MonkeyPatch
    ) -> None:
        def getmap(**kwargs: Any) -> io.BytesIO:
            return encode(np.zeros((8, 8), dtype=np.uint8))

        monkeypatch.setattr(dataset.service, 'getmap', getmap)
        assert dataset[-1:0:0.25, 50:51:0.125]['image'].shape == (3, 8, 8)

    def test_picklable(self, dataset: WMSDataset) -> None:
        restored = pickle.loads(pickle.dumps(dataset))
        assert restored.srs == dataset.srs
        assert restored[-1:0:0.01, 50:51:0.01]['image'].shape == (3, 100, 100)

    def test_invalid_layer(self) -> None:
        with pytest.raises(ValueError, match='Available layers: boundaries, imagery'):
            WMSDataset(URL, 'elevation', res=0.01)

    def test_no_extent(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(
            MockLayer,
            '__init__',
            lambda self, _: setattr(self, 'boundingBoxWGS84', None),
        )
        with pytest.raises(ValueError, match='imagery does not advertise'):
            WMSDataset(URL, 'imagery', res=0.01)

    def test_default_crs_not_offered(self) -> None:
        with pytest.raises(ValueError, match='boundaries is not offered in EPSG:4326'):
            WMSDataset(URL, 'boundaries', res=0.01)

    def test_crs_without_authority(self) -> None:
        crs = CRS.from_proj4('+proj=laea +lat_0=45 +lon_0=-100')
        with pytest.raises(ValueError, match='has no authority code'):
            WMSDataset(URL, 'imagery', res=10, crs=crs)

    def test_invalid_query(self, dataset: WMSDataset) -> None:
        with pytest.raises(IndexError, match='not found in dataset with bounds'):
            dataset[10:11:0.01, 10:11:0.01]
