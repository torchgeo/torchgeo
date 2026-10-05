# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import nn

from torchgeo.models import (
    OlmoEarthBase_Weights,
    OlmoEarthLarge_Weights,
    OlmoEarthNano_Weights,
    OlmoEarthSmall_Weights,
    OlmoEarthTiny_Weights,
    OlmoEarthV1_Weights,
    olmoearth_base,
    olmoearth_large,
    olmoearth_nano,
    olmoearth_small,
    olmoearth_tiny,
    olmoearth_v1,
    olmoearth_v1_unet_decoder,
)

olmoearth_pretrain_minimal = pytest.importorskip('olmoearth_pretrain_minimal')


class TestOlmoEarthPinnedDownload:
    @pytest.fixture
    def downloads(self, monkeypatch: MonkeyPatch) -> list[str]:
        urls: list[str] = []

        def download_url_to_file(
            url: str, dst: str, hash_prefix: str | None = None
        ) -> None:
            urls.append(url)
            Path(dst).touch()

        def load_model_from_path(path: str) -> nn.Module:
            assert os.path.exists(os.path.join(path, 'config.json'))
            assert os.path.exists(os.path.join(path, 'weights.pth'))
            return nn.Identity()

        monkeypatch.setattr(torch.hub, 'download_url_to_file', download_url_to_file)
        monkeypatch.setattr(
            olmoearth_pretrain_minimal, 'load_model_from_path', load_model_from_path
        )
        return urls

    def test_download(self, downloads: list[str]) -> None:
        weights = OlmoEarthNano_Weights.V1
        model = olmoearth_nano(weights=weights)
        assert isinstance(model, nn.Identity)
        prefix = (
            f'https://huggingface.co/{weights.meta["hf_repo"]}'
            f'/resolve/{weights.meta["revision"]}'
        )
        assert downloads == [f'{prefix}/config.json', f'{prefix}/weights.pth']

    def test_cached(self, downloads: list[str]) -> None:
        olmoearth_nano(weights=OlmoEarthNano_Weights.V1)
        olmoearth_nano(weights=OlmoEarthNano_Weights.V1)
        assert len(downloads) == 2


class TestOlmoEarthNano:
    @pytest.fixture(params=[*OlmoEarthNano_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthNano_Weights:
        return request.param

    def test_olmoearth(self) -> None:
        olmoearth_nano()

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthNano_Weights) -> None:
        olmoearth_nano(weights=weights)


class TestOlmoEarthTiny:
    @pytest.fixture(params=[*OlmoEarthTiny_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthTiny_Weights:
        return request.param

    def test_olmoearth(self) -> None:
        olmoearth_tiny()

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthTiny_Weights) -> None:
        olmoearth_tiny(weights=weights)


class TestOlmoEarthSmall:
    @pytest.fixture(params=[*OlmoEarthSmall_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthSmall_Weights:
        return request.param

    def test_olmoearth(self) -> None:
        olmoearth_small()

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthSmall_Weights) -> None:
        olmoearth_small(weights=weights)


class TestOlmoEarthBase:
    @pytest.fixture(params=[*OlmoEarthBase_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthBase_Weights:
        return request.param

    def test_olmoearth(self) -> None:
        olmoearth_base()

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthBase_Weights) -> None:
        olmoearth_base(weights=weights)


class TestOlmoEarthLarge:
    @pytest.fixture(params=[*OlmoEarthLarge_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthLarge_Weights:
        return request.param

    def test_olmoearth(self) -> None:
        olmoearth_large()

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthLarge_Weights) -> None:
        olmoearth_large(weights=weights)


@pytest.mark.filterwarnings(
    'ignore:Use torchgeo.models.olmoearth_nano.* instead:DeprecationWarning'
)
class TestOlmoEarthV1:
    @pytest.fixture(params=[*OlmoEarthV1_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthV1_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> OlmoEarthV1_Weights:
        weights = OlmoEarthV1_Weights.NANO
        path = tmp_path / 'weights.pth'
        model = olmoearth_v1(model_size='nano')
        torch.save(model.model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_olmoearth_v1(self) -> None:
        olmoearth_v1()

    def test_olmoearth_v1_deprecated(self) -> None:
        with pytest.warns(DeprecationWarning, match='olmoearth_nano'):
            olmoearth_v1()

    def test_olmoearth_v1_weights(self, mocked_weights: OlmoEarthV1_Weights) -> None:
        olmoearth_v1(weights=mocked_weights)

    def test_olmoearth_v1_weights_are_applied(
        self, mocked_weights: OlmoEarthV1_Weights
    ) -> None:
        one = olmoearth_v1(weights=mocked_weights).state_dict()
        two = olmoearth_v1(weights=mocked_weights).state_dict()
        assert one.keys() == two.keys()
        for key, value in one.items():
            assert torch.equal(value, two[key]), key

    @pytest.mark.slow
    def test_olmoearth_v1_download(self, weights: OlmoEarthV1_Weights) -> None:
        olmoearth_v1(weights=weights)


class TestOlmoEarthV1UNetDecoder:
    def test_olmoearth_v1_unet_decoder(self) -> None:
        olmoearth_v1_unet_decoder()

    def test_forward(self) -> None:
        in_dim, num_classes, patch_size = 32, 5, 8
        decoder = olmoearth_v1_unet_decoder(
            in_dim=in_dim, num_classes=num_classes, patch_size=patch_size
        )
        # Patch tokens: (B, H_p, W_p, in_dim) -> logits (B, num_classes, H, W).
        x = torch.randn(2, 4, 4, in_dim)
        out = decoder(x)
        assert out.shape == (2, num_classes, 4 * patch_size, 4 * patch_size)

    def test_invalid_patch_size(self) -> None:
        with pytest.raises(ValueError, match='patch_size must be a power of two'):
            olmoearth_v1_unet_decoder(patch_size=6)
