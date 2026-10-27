# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Mosaic v0 dataset."""

import os
from collections.abc import Callable, Sequence
from typing import ClassVar, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import (
    Path,
    Sample,
    check_integrity,
    download_and_extract_archive,
    download_url,
    extract_archive,
    quantile_normalization,
)


class MosaicV0(NonGeoDataset):
    """Mosaic v0 dataset.

    `Mosaic v0 <https://doi.org/10.5281/zenodo.23582250>`__ adds per-pixel
    reliability layers and event-blocked splits to the 446 hand-labeled chips of
    `Sen1Floods11 v1.1 <https://github.com/cloudtostreet/Sen1Floods11>`__.
    For each pixel, a Dawid-Skene model fit on four disagreeing label sources
    (S1-Otsu, S2-MNDWI, JRC permanent water, and an MMFlood-trained U-Net) gives a
    posterior probability of water and a bitmask recording which sources disagree.

    Mosaic v0 does not redistribute the Sen1Floods11 imagery or labels. This class
    downloads the Mosaic v0 package from Zenodo and the original Sen1Floods11 files
    from the official public bucket (``gs://sen1floods11``, over HTTPS), verifying
    each original against the SHA-256 recorded in the package.

    Dataset features:

    * 446 chips from 11 flood events, 512x512 px, about 10 m, EPSG:4326
    * Sentinel-1 GRD (VV, VH in dB) and Sentinel-2 L1C (13 bands) imagery
    * hand labels (1 water, 0 not water, -1 unlabeled)
    * posterior P(water) under the package's settings (f) and (g)
      (NaN where fewer than two sources are valid)
    * per-pixel disagreement flags (uint8 bitmask, see :attr:`flag_bits`)
    * ESA WorldCover 2021 land-cover group (see :attr:`landcover_groups`)
    * optionally, the four label sources (1 water, 0 not water, -1 invalid)

    Dataset splits (*split_by*):

    * ``'sen1floods11'``: the official Sen1Floods11 v1.1 hand-labeled split
      (train 252, valid 89, test 90, bolivia 15)
    * ``'pilot'``: the pilot's anchor/test assignment (anchor 46, test 400). Anchor
      hand labels were used to fit the posterior, so evaluate posteriors on ``'test'``
    * ``'event'``: leave-one-event-out; ``'test'`` holds the event with
      ``event_fold == fold`` and ``'train'`` holds the other 10 events

    Dataset format:

    * Mosaic layers are GeoTIFFs in the Zenodo package; originals are GeoTIFFs
      stored under ``root/sen1floods11/`` mirroring the bucket layout

    If you use this dataset in your research, please cite both Mosaic v0 and
    Sen1Floods11:

    * https://doi.org/10.5281/zenodo.23582250
    * https://doi.org/10.1109/CVPRW50498.2020.00113

    The Mosaic v0 layers and tables are released under CC BY 4.0. The Sen1Floods11
    originals remain under their own terms.

    .. note::

       NaN values in Sentinel-1 imagery are replaced with 0 in *image*.

    .. versionadded:: 0.11
    """

    url = 'https://zenodo.org/records/23582250/files/mosaic-v0-0.1.0.zip?download=1'
    filename = 'mosaic-v0-0.1.0.zip'
    sha256 = 'ab4c77b717a93873e4d3a80ce954e81558a2d54b9bea6fd39820c397185db985'
    directory = 'mosaic-v0'

    #: Base URL of the official Sen1Floods11 public bucket.
    originals_url = 'https://storage.googleapis.com/sen1floods11/'
    originals_dir = 'sen1floods11'

    s1_bands = ('VV', 'VH')
    s2_bands = (
        'B1',
        'B2',
        'B3',
        'B4',
        'B5',
        'B6',
        'B7',
        'B8',
        'B8A',
        'B9',
        'B10',
        'B11',
        'B12',
    )
    all_bands = s1_bands + s2_bands
    rgb_bands = ('B4', 'B3', 'B2')

    sources = ('S1-Otsu', 'S2-MNDWI', 'JRC', 'MMFlood-UNet')

    #: Meaning of each bit of the *flags* bitmask.
    flag_bits: ClassVar[dict[int, str]] = {
        0: 'S1-Otsu disagrees with the MAP of posterior (g)',
        1: 'S2-MNDWI disagrees with the MAP of posterior (g)',
        2: 'JRC disagrees with the MAP of posterior (g)',
        3: 'MMFlood-UNet disagrees with the MAP of posterior (g)',
        4: 'the valid sources disagree among themselves',
        5: 'the S1 family (Otsu = U-Net) disagrees with S2-MNDWI',
        6: '0.2 < P(g) < 0.8',
        7: 'fewer than 2 valid sources',
    }

    #: Land-cover groups (from ESA WorldCover 2021 v200).
    landcover_groups = (
        'Tree/shrub/mangrove',
        'Grass/wetland',
        'Cropland',
        'Built/bare',
        'Water',
    )

    ignore_index = -1

    _splits: ClassVar[dict[str, tuple[str, ...]]] = {
        'sen1floods11': ('train', 'valid', 'test', 'bolivia'),
        'pilot': ('anchor', 'test'),
        'event': ('train', 'test'),
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        split_by: Literal['sen1floods11', 'pilot', 'event'] = 'sen1floods11',
        fold: int = 0,
        bands: Sequence[str] = s1_bands,
        posterior: Literal['f', 'g'] = 'g',
        include_sources: bool = False,
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = True,
    ) -> None:
        """Initialize a new MosaicV0 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: split to load; valid values depend on *split_by*
            split_by: one of 'sen1floods11', 'pilot', or 'event'
            fold: event fold held out as 'test' when *split_by* is 'event'
            bands: Sentinel-1 and/or Sentinel-2 bands to return as *image*
            posterior: posterior setting to return, 'f' or 'g'
            include_sources: if True, also return the four label sources as
                *sources* (downloads the S1-Otsu and JRC originals)
            transforms: a function/transform that takes input sample and its target
                as entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, verify the SHA-256 of the downloaded files (may be slow)

        Raises:
            AssertionError: If *split*, *split_by*, *fold*, *bands*, or
                *posterior* are invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split_by in self._splits
        assert split in self._splits[split_by]
        assert len(bands) > 0
        assert set(bands) <= set(self.all_bands)
        assert posterior in ('f', 'g')

        self.root = root
        self.split = split
        self.split_by = split_by
        self.fold = fold
        self.bands = bands
        self.posterior = posterior
        self.include_sources = include_sources
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self.s1_indices = [self.s1_bands.index(b) for b in bands if b in self.s1_bands]
        self.s2_indices = [self.s2_bands.index(b) for b in bands if b in self.s2_bands]
        self.layers = ['LabelHand']
        if self.s1_indices:
            self.layers.append('S1Hand')
        if self.s2_indices:
            self.layers.append('S2Hand')
        if include_sources:
            self.layers += ['S1OtsuLabelHand', 'JRCWaterHand']

        self._verify_package()
        package = os.path.join(self.root, self.directory)
        chips = pd.read_csv(os.path.join(package, 'tables', 'chips.csv'))
        match split_by:
            case 'sen1floods11':
                chips = chips[chips['sen1floods11_split'] == split]
            case 'pilot':
                chips = chips[chips['pilot_role'] == split]
            case 'event':
                assert fold in set(chips['event_fold'])
                held_out = chips['event_fold'] == fold
                chips = chips[held_out if split == 'test' else ~held_out]
        self.chips = chips.reset_index(drop=True)

        originals = pd.read_csv(os.path.join(package, 'tables', 'originals.csv'))
        originals = originals[originals['chip'].isin(self.chips['chip'])]
        originals = originals[originals['layer'].isin(self.layers)]
        # (chip, layer) -> (path relative to the bucket root, SHA-256)
        self.originals = {
            (str(chip), str(layer)): (
                str(gs_uri).removeprefix('gs://sen1floods11/'),
                str(sha256),
            )
            for chip, layer, gs_uri, sha256 in zip(
                originals['chip'],
                originals['layer'],
                originals['gs_uri'],
                originals['sha256'],
            )
        }
        self._verify_originals()

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.chips)

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index
        """
        row = self.chips.iloc[index]
        chip = row['chip']
        package = os.path.join(self.root, self.directory)

        images = []
        if self.s1_indices:
            s1 = self._load(chip, 'S1Hand').astype(np.float32)[self.s1_indices]
            images.append(np.nan_to_num(s1, nan=0.0))
        if self.s2_indices:
            s2 = self._load(chip, 'S2Hand').astype(np.float32)[self.s2_indices]
            images.append(s2)
        image = torch.from_numpy(np.concatenate(images, axis=0))
        mask = torch.from_numpy(self._load(chip, 'LabelHand')[0].astype(np.int64))

        with rasterio.open(os.path.join(package, row['posterior_file'])) as f:
            posterior = f.read(1 if self.posterior == 'f' else 2)
        with rasterio.open(os.path.join(package, row['aux_file'])) as f:
            aux = f.read()

        sample = {
            'image': image,
            'mask': mask,
            'posterior': torch.from_numpy(posterior.astype(np.float32)),
            'flags': torch.from_numpy(aux[3]),
            'landcover': torch.from_numpy(aux[2].astype(np.int64)),
        }

        if self.include_sources:
            otsu = self._load(chip, 'S1OtsuLabelHand')[0].astype(np.int64)
            jrc = self._load(chip, 'JRCWaterHand')[0].astype(np.int64)
            mndwi = aux[0].astype(np.int64)
            unet = aux[1].astype(np.int64)
            mndwi[mndwi == 255] = -1
            unet[unet == 255] = -1
            sample['sources'] = torch.from_numpy(np.stack([otsu, mndwi, jrc, unet]))

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _original_path(self, chip: str, layer: str) -> str:
        """Return the local path of an original Sen1Floods11 file.

        Args:
            chip: chip id
            layer: Sen1Floods11 layer name

        Returns:
            path under *root* that mirrors the bucket layout
        """
        relpath = self.originals[chip, layer][0]
        return os.path.join(self.root, self.originals_dir, relpath)

    def _load(self, chip: str, layer: str) -> np.typing.NDArray[np.generic]:
        """Load an original Sen1Floods11 file.

        Args:
            chip: chip id
            layer: Sen1Floods11 layer name

        Returns:
            array of shape (C, H, W)
        """
        with rasterio.open(self._original_path(chip, layer)) as f:
            return f.read()

    def _verify_package(self) -> None:
        """Verify the integrity of the Mosaic v0 package."""
        # Check if the extracted package already exists
        tables = os.path.join(self.root, self.directory, 'tables', 'chips.csv')
        if os.path.isfile(tables):
            return

        # Check if the zip file has already been downloaded
        path = os.path.join(self.root, self.filename)
        if os.path.isfile(path):
            if self.checksum and not check_integrity(path, sha256=self.sha256):
                raise RuntimeError('Dataset found, but corrupted.')
            extract_archive(path)
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the package
        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.filename,
            sha256=self.sha256 if self.checksum else None,
        )

    def _verify_originals(self) -> None:
        """Verify that the needed Sen1Floods11 originals exist, downloading them."""
        for (chip, layer), (relpath, sha256) in self.originals.items():
            path = self._original_path(chip, layer)
            if os.path.isfile(path):
                continue

            if not self.download:
                raise DatasetNotFoundError(self)

            download_url(
                self.originals_url + relpath,
                os.path.dirname(path),
                sha256=sha256 if self.checksum else None,
            )

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            RGBBandsMissingError: If *bands* includes neither all RGB bands
                nor both Sentinel-1 bands.
        """
        bands = list(self.bands)
        if all(b in bands for b in self.rgb_bands):
            channels = [sample['image'][bands.index(b)] for b in self.rgb_bands]
        elif all(b in bands for b in self.s1_bands):
            vv = sample['image'][bands.index('VV')]
            vh = sample['image'][bands.index('VH')]
            # False color composite: VV, VH, VV - VH (dB)
            channels = [vv, vh, vv - vh]
        else:
            raise RGBBandsMissingError()
        nodata = torch.stack(channels).eq(0).all(dim=0)
        image = torch.stack([quantile_normalization(c) for c in channels], dim=-1)
        image[nodata] = 0

        mask = sample['mask'].numpy()
        # gray: unlabeled (-1), black: not water (0), white: water (1)
        label_cmap = ListedColormap(['gray', 'black', 'white'])
        posterior = sample['posterior'].numpy()
        disagree = ((sample['flags'].numpy() >> 4) & 1).astype(np.uint8)

        panels = [
            ('Image', image.numpy(), {}),
            ('Hand label', mask, {'cmap': label_cmap, 'vmin': -1, 'vmax': 1}),
            (
                f'Posterior P(water) ({self.posterior})',
                posterior,
                {'cmap': 'viridis', 'vmin': 0, 'vmax': 1},
            ),
            ('Sources disagree', disagree, {'cmap': 'gray', 'vmin': 0, 'vmax': 1}),
        ]
        if 'prediction' in sample:
            prediction = sample['prediction'].numpy()
            panels.append(
                ('Prediction', prediction, {'cmap': 'gray', 'vmin': 0, 'vmax': 1})
            )

        ncols = len(panels)
        fig, axs = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        for ax, (title, data, kwargs) in zip(axs, panels):
            ax.imshow(data, **kwargs)
            ax.axis('off')
            if show_titles:
                ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
