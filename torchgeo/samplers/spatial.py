# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Spatial sampling routines."""

import math
from collections.abc import Iterator
from typing import Literal

import geopandas as gpd
import numpy as np
import shapely
from geopandas import GeoSeries
from numpy.random import BitGenerator, Generator, RandomState, SeedSequence
from shapely import Polygon

from ..datasets import GeoDataset
from .base import SpatialSampler
from .constants import Units
from .utils import _to_tuple, convolution_arithmetic


class RandomPatchSampler(SpatialSampler):
    """Sample locations from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible. Note that
    randomly sampled chips may overlap.

    .. versionadded:: 0.10
    """

    strategy = 'random'

    def __init__(
        self,
        dataset: GeoDataset,
        *,
        size: tuple[float, float] | float,
        length: int | None = None,
        roi: Polygon | None = None,
        units: Units = Units.PIXELS,
        generator: int
        | BitGenerator
        | Generator
        | RandomState
        | SeedSequence
        | None = None,
        snap2pixel: bool = True,
        snap2block: bool = False,
        origin: Literal['upper', 'lower'] = 'upper',
    ) -> None:
        """Initialize a new RandomPatchSampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: Dataset to sample from.
            size: Dimensions of each :term:`patch`.
            length: Number of random samples to draw per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset).
            roi: Region of interest to sample from
                (defaults to the bounds of ``dataset.index``).
            units: Defines if ``size`` is in pixel or CRS units.
            generator: Pseudo-random number generator (PRNG).
            snap2pixel: Snap sampled patches to the nearest pixel. Results in a slight
                decrease in data diversity but a large increase in I/O speed.
            snap2block: Snap sampled patches to the nearest block. Results in a large
                decrease in data diversity but a larger increase in I/O speed. Set
                *size* to a factor or multiple of your COG block size for best effect.
            origin: Whether the [0, 0] origin is in the upper left or lower left corner
                of each raster file. Only matters when *snap2block* is True.

        .. versionadded:: 0.11
           The *snap2pixel*, *snap2block*, and *origin* parameters.
        """
        super().__init__(dataset, roi=roi)

        self.size = _to_tuple(size)
        self.generator = np.random.default_rng(generator)
        self.snap2pixel = snap2pixel
        self.snap2block = snap2block
        self.origin = origin

        # Convert from pixel units to CRS units
        if units == Units.PIXELS:
            self.size = (self.size[0] * dataset.res[1], self.size[1] * dataset.res[0])

        # Default to approximate number of non-overlapping patches
        total_area = shapely.area(self.geometry)
        patch_area = self.size[0] * self.size[1]
        self._length = length or convolution_arithmetic(total_area, patch_area)

        # Erosion to avoid out-of-bounds sampling
        # Purposefully conservative radius calculation
        distance = math.sqrt((self.size[0] / 2) ** 2 + (self.size[1] / 2) ** 2)
        self.series = GeoSeries([shapely.buffer(self.geometry, -distance)])

    def __iter__(self) -> Iterator[tuple[slice, slice]]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [xmin:xmax, ymin:ymax] coordinates to index a dataset.
        """
        # Ensure a new set of random points for each epoch
        points = self.series.sample_points(size=len(self), rng=self.generator)

        # Points are random, but order is not
        # Remove once https://github.com/geopandas/geopandas/pull/3773 is in min version
        points = points.explode().sample(frac=1, random_state=self.generator)

        # Shift by half a patch to make it easier to snap to a corner
        points = points.translate(-self.size[1] / 2, -self.size[0] / 2)

        if self.snap2pixel or self.snap2block:
            xorigin, ymin, _, ymax = self.bounds
            yorigin = ymax if self.origin == 'upper' else ymin
            scale = self.size[::-1] if self.snap2block else self.res
            # Convert from geospatial coords to pixel/block coords
            points = points.translate(-xorigin, -yorigin)
            points = points.scale(1 / scale[0], 1 / scale[1], origin=(0, 0))
            # Round to the nearest pixel/block
            x = points.x.round()
            y = points.y.round()
            # Convert from pixel/block coords to geospatial coords
            points = gpd.points_from_xy(x, y)
            points = points.scale(scale[0], scale[1], origin=(0, 0))
            points = points.translate(xorigin, yorigin)

        for point in points:
            xmin = point.x
            xmax = point.x + self.size[1]
            ymin = point.y
            ymax = point.y + self.size[0]
            yield slice(xmin, xmax), slice(ymin, ymax)


class GriddedPatchSampler(SpatialSampler):
    """Sample locations from a region of interest in a grid-like fashion.

    This is particularly useful during evaluation when you want to make predictions for
    an entire region of interest. You want to minimize the amount of redundant
    computation by minimizing overlap between :term:`chips <chip>`.

    Usually the stride should be slightly smaller than the chip size such that each chip
    has some small overlap with surrounding chips. This is used to prevent `stitching
    artifacts <https://arxiv.org/abs/1805.12219>`_ when combining each prediction patch.
    The overlap between each chip (``size - stride``) should be approximately equal to
    the `receptive field <https://distill.pub/2019/computing-receptive-fields/>`_ of the
    CNN.

    .. versionadded:: 0.10
    """

    strategy = 'sequential'

    def __init__(
        self,
        dataset: GeoDataset,
        *,
        size: tuple[float, float] | float,
        stride: tuple[float, float] | float | None = None,
        roi: Polygon | None = None,
        units: Units = Units.PIXELS,
        snap2pixel: bool = True,
        snap2block: bool = False,
        origin: Literal['upper', 'lower'] = 'upper',
    ) -> None:
        """Initialize a new GriddedPatchSampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: Dataset to sample from.
            size: Dimensions of each :term:`patch`.
            stride: Distance to skip between each patch (defaults to *size*).
            roi: Region of interest to sample from
                (defaults to the bounds of ``dataset.index``).
            units: Defines if ``size`` and ``stride`` are in pixel or CRS units.
            snap2pixel: Snap sampled patches to the nearest pixel. Results in a slight
                decrease in data diversity but a large increase in I/O speed.
            snap2block: Snap sampled patches to the nearest block. Results in a large
                decrease in data diversity but a larger increase in I/O speed. Set
                *size* and *stride* to a factor or multiple of your COG block size for
                best effect.
            origin: Whether the [0, 0] origin is in the upper left or lower left corner
                of each raster file. Only matters when *snap2block* is True.

        .. versionadded:: 0.11
           The *snap2pixel*, *snap2block*, and *origin* parameters.
        """
        super().__init__(dataset, roi=roi)

        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride or size)
        self.snap2pixel = snap2pixel
        self.snap2block = snap2block
        self.origin = origin

        # Convert from pixel units to CRS units
        if units == Units.PIXELS:
            self.size = (self.size[0] * dataset.res[1], self.size[1] * dataset.res[0])
            self.stride = (
                self.stride[0] * dataset.res[1],
                self.stride[1] * dataset.res[0],
            )

    def __iter__(self) -> Iterator[tuple[slice, slice]]:
        """Iterate over generated sample locations for each epoch.

        Yields:
            [xmin:xmax, ymin:ymax] coordinates to index a dataset.
        """
        xmin, ymin, xmax, ymax = self.geometry.bounds
        xorigin = self.bounds[0]
        yorigin = self.bounds[3] if self.origin == 'upper' else self.bounds[1]

        if self.snap2pixel or self.snap2block:
            scale = self.size[::-1] if self.snap2block else self.res
            # Convert from geospatial coords to pixel/block coords
            xmin = (xmin - xorigin) / scale[0]
            xmax = (xmax - xorigin) / scale[0]
            ymin = (ymin - yorigin) / scale[1]
            ymax = (ymax - yorigin) / scale[1]
            # Round to the nearest pixel/block
            xmin = math.floor(xmin)
            xmax = math.ceil(xmax)
            ymin = math.floor(ymin)
            ymax = math.ceil(ymax)
            # Convert from pixel/block coords to geospatial coords
            xmin = xmin * scale[0] + xorigin
            xmax = xmax * scale[0] + xorigin
            ymin = ymin * scale[1] + yorigin
            ymax = ymax * scale[1] + yorigin

        rows = convolution_arithmetic(ymax - ymin, self.size[0], self.stride[0])
        cols = convolution_arithmetic(xmax - xmin, self.size[1], self.stride[1])

        # For each row...
        for i in range(rows):
            if self.origin == 'upper':
                y = ymax - i * self.stride[0]
            else:
                y = ymin + i * self.stride[0]

            # For each column...
            for j in range(cols):
                x = xmin + j * self.stride[1]

                # Check for intersection
                if self.origin == 'upper':
                    bbox = shapely.box(x, y - self.size[0], x + self.size[1], y)
                else:
                    bbox = shapely.box(x, y, x + self.size[1], y + self.size[0])
                if self.geometry.intersects(bbox) and not self.geometry.touches(bbox):
                    if self.origin == 'upper':
                        yield slice(x, x + self.size[1]), slice(y - self.size[0], y)
                    else:
                        yield slice(x, x + self.size[1]), slice(y, y + self.size[0])
