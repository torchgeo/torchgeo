# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Mixins for dataset classes."""

from collections.abc import Sequence
from typing import ClassVar

from matplotlib.colors import Colormap


class PlottingMixin:
    """Mixin for dataset plotting.

    .. versionadded:: 0.10
    """

    #: Names of all available bands in the dataset.
    all_bands: ClassVar[tuple[str, ...]] = ()

    #: Names of RGB bands in the dataset.
    rgb_bands: ClassVar[tuple[str, ...]] = ()

    #: Subset of *all_bands* selected by user.
    bands: Sequence[str]

    #: Color map for the dataset
    cmap: str | Colormap | None = None
