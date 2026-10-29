# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Web Map Service dataset."""

import io
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import shapely
import torch
from geopandas import GeoDataFrame
from PIL import Image
from pyproj import CRS as PROJ_CRS

from .geo import GeoDataset
from .utils import GeoSlice, Sample, lazy_import


class WMSDataset(GeoDataset):
    """Dataset backed by a Web Map Service.

    A `Web Map Service (WMS) <https://www.ogc.org/publications/standard/wms/>`__
    renders georeferenced imagery on demand: the client asks for a bounding box, a
    :term:`coordinate reference system (CRS)`, and an image size, and the server
    reprojects and resamples its data to match. Many mapping agencies publish aerial
    imagery this way, which makes a WMS a practical source of imagery for regions
    where downloading and storing the entire dataset is not an option.

    Every query issues one ``GetMap`` request and returns the rendered picture as
    RGB; nothing is written to disk. The server decides how the imagery is
    resampled, so querying a different CRS or resolution costs nothing on the
    client side. Servers usually limit the image size they will return, which
    bounds how large a single query can be.

    This dataset requires the following additional library to be installed:

    * `owslib <https://pypi.org/project/OWSLib/>`_ to query the service

    .. versionadded:: 0.11
    """

    def __init__(
        self,
        url: str,
        layer: str,
        res: float | tuple[float, float],
        crs: PROJ_CRS | None = None,
        format: str = 'image/png',
        version: str = '1.3.0',
        transforms: Callable[[Sample], Sample] | None = None,
    ) -> None:
        """Initialize a new WMSDataset instance.

        Args:
            url: URL of the service, without any query parameters.
            layer: Name of the layer to request.
            res: Resolution of a query in units of CRS. A WMS has no native
                resolution, so this only provides the default used by samplers.
            crs: CRS to request the imagery in. Defaults to EPSG:4326, which the
                layer must offer.
            format: MIME type of the imagery to request.
            version: Version of the WMS specification to speak.
            transforms: A function/transform that takes an input sample and returns a
                transformed version.

        Raises:
            DependencyNotFoundError: If owslib is not installed.
            ValueError: If *layer* is not offered by the service, does not advertise
                its extent, or is not offered in *crs*.
        """
        self.url = url
        self.version = version
        self._service: Any = None

        if layer not in self.service.contents:
            offered = ', '.join(sorted(self.service.contents))
            msg = f'{layer} is not offered by {url}. Available layers: {offered}'
            raise ValueError(msg)

        metadata = self.service.contents[layer]
        if metadata.boundingBoxWGS84 is None:
            msg = f'{layer} does not advertise its geographic extent.'
            raise ValueError(msg)

        self.layer = layer
        self.format = format
        self.transforms = transforms
        self.crs_options: list[str] = list(metadata.crsOptions or [])

        if crs is None:
            crs = PROJ_CRS.from_epsg(4326)
        self.srs = self._srs(crs)

        if isinstance(res, int | float):
            res = (res, res)
        self._res = res

        # The service advertises the extent of every layer in EPSG:4326.
        geometry = [shapely.box(*metadata.boundingBoxWGS84)]
        datetimes = [(pd.Timestamp.min, pd.Timestamp.max)]
        index = pd.IntervalIndex.from_tuples(datetimes, closed='both', name='datetime')
        self.index = GeoDataFrame(
            index=index, geometry=geometry, crs=PROJ_CRS.from_epsg(4326)
        ).to_crs(crs)

    @property
    def service(self) -> Any:
        """Client for the service, created on first use.

        The client holds parsed XML that cannot be pickled, so it is left out of
        the pickled state and recreated in each DataLoader worker.
        """
        if self._service is None:
            wms = lazy_import('owslib.wms')
            self._service = wms.WebMapService(self.url, version=self.version)
        return self._service

    def __getstate__(self) -> dict[str, Any]:
        """Pickle everything but the service client."""
        return {**self.__dict__, '_service': None}

    def _srs(self, crs: PROJ_CRS) -> str:
        """Identifier the service uses for *crs*.

        Args:
            crs: CRS to request the imagery in.

        Returns:
            The ``AUTHORITY:CODE`` identifier of *crs*.

        Raises:
            ValueError: If *crs* has no authority code or the layer is not offered
                in it.
        """
        authority = crs.to_authority()
        if authority is None:
            msg = f'{crs.name} has no authority code, so a WMS cannot serve it.'
            raise ValueError(msg)

        srs = ':'.join(authority)
        if srs not in self.crs_options:
            offered = ', '.join(sorted(self.crs_options))
            msg = f'{self.layer} is not offered in {srs}. Available CRSs: {offered}'
            raise ValueError(msg)
        return srs

    @GeoDataset.crs.setter
    def crs(self, new_crs: PROJ_CRS) -> None:
        """Change the CRS of the dataset and of every request it issues.

        Args:
            new_crs: New CRS, which the layer must offer.
        """
        self.srs = self._srs(new_crs)
        GeoDataset.crs.fset(self, new_crs)

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            index: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *index* is not found in the dataset.
        """
        x, y, _ = self._disambiguate_slice(index)

        if self.index.cx[x.start : x.stop, y.start : y.stop].empty:
            raise IndexError(
                f'index: {index} not found in dataset with bounds: {self.bounds}'
            )

        width = round((x.stop - x.start) / x.step)
        height = round((y.stop - y.start) / y.step)
        response = self.service.getmap(
            layers=[self.layer],
            srs=self.srs,
            bbox=(x.start, y.start, x.stop, y.stop),
            size=(width, height),
            format=self.format,
        )
        image = Image.open(io.BytesIO(response.read())).convert('RGB')
        data = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(torch.float32)

        sample: Sample = {
            'image': data,
            'bounds': self._slice_to_tensor(index),
            'transform': torch.tensor(
                rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
            ),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
