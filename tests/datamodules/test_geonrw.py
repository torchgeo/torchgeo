# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from torchgeo.datamodules import GeoNRWDataModule, SustainBenchCropYieldDataModule


class TestGeoNRWDataModule:
    def test_no_rescaling(self) -> None:
        """Dataset images are already in [0, 1]; normalization must be identity."""
        dm = GeoNRWDataModule()
        assert dm.mean == 0
        assert dm.std == 1


class TestSustainBenchCropYieldDataModule:
    def test_no_rescaling(self) -> None:
        """Dataset images are already in [0, 1]; normalization must be identity."""
        dm = SustainBenchCropYieldDataModule()
        assert dm.mean == 0
        assert dm.std == 1
