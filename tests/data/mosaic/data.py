#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

SIZE = 32
PACKAGE = 'mosaic-v0'
ORIGINALS = os.path.join('sen1floods11', 'v1.1', 'data', 'flood_events', 'HandLabeled')

np.random.seed(0)

# chip, event, event_fold, pilot_role, sen1floods11_split
CHIPS = [
    ('Bolivia_1', 'Bolivia', 0, 'test', 'bolivia'),
    ('Ghana_1', 'Ghana', 1, 'anchor', 'train'),
    ('Ghana_2', 'Ghana', 1, 'test', 'valid'),
    ('India_1', 'India', 2, 'test', 'test'),
    ('India_2', 'India', 2, 'test', 'train'),
]

# layer: (count, dtype)
LAYERS = {
    'S1Hand': (2, 'float32'),
    'S2Hand': (13, 'int16'),
    'LabelHand': (1, 'int16'),
    'S1OtsuLabelHand': (1, 'int16'),
    'JRCWaterHand': (1, 'uint8'),
}


def write(path: str, data: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = {
        'driver': 'GTiff',
        'height': SIZE,
        'width': SIZE,
        'count': data.shape[0],
        'dtype': data.dtype,
        'crs': 'EPSG:4326',
        'transform': from_origin(-65.0, -14.0, 0.0001, 0.0001),
    }
    with rasterio.open(path, 'w', **profile) as f:
        f.write(data)


def sha256(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def original(layer: str) -> np.ndarray:
    count, dtype = LAYERS[layer]
    shape = (count, SIZE, SIZE)
    if layer == 'S1Hand':
        data = np.random.uniform(-30, 0, shape).astype(dtype)
        data[:, 0, 0] = np.nan
    elif layer == 'S2Hand':
        data = np.random.randint(0, 5000, shape).astype(dtype)
    elif layer == 'JRCWaterHand':
        data = np.random.randint(0, 2, shape).astype(dtype)
    else:
        data = np.random.randint(-1, 2, shape).astype(dtype)
    return data


if __name__ == '__main__':
    for path in [PACKAGE, 'sen1floods11', f'{PACKAGE}-0.1.0.zip']:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)

    chip_rows = []
    original_rows = []
    for chip, event, fold, role, split in CHIPS:
        posterior_file = f'layers/sen1floods11/posterior/{chip}_posterior.tif'
        aux_file = f'layers/sen1floods11/aux/{chip}_aux.tif'

        posterior = np.random.rand(2, SIZE, SIZE).astype(np.float32)
        posterior[:, 0, 0] = np.nan
        write(os.path.join(PACKAGE, posterior_file), posterior)

        aux = np.stack(
            [
                np.random.choice([0, 1, 255], (SIZE, SIZE)),
                np.random.choice([0, 1, 255], (SIZE, SIZE)),
                np.random.randint(0, 5, (SIZE, SIZE)),
                np.random.randint(0, 256, (SIZE, SIZE)),
            ]
        ).astype(np.uint8)
        write(os.path.join(PACKAGE, aux_file), aux)

        chip_rows.append(
            {
                'chip': chip,
                'event': event,
                'event_fold': fold,
                'pilot_role': role,
                'sen1floods11_split': split,
                'posterior_file': posterior_file,
                'aux_file': aux_file,
            }
        )

        for layer in LAYERS:
            name = f'{chip}_{layer}.tif'
            path = os.path.join(ORIGINALS, layer, name)
            write(path, original(layer))
            gs_uri = (
                f'gs://sen1floods11/v1.1/data/flood_events/HandLabeled/{layer}/{name}'
            )
            original_rows.append(
                {
                    'chip': chip,
                    'layer': layer,
                    'url': gs_uri.replace('gs://', 'https://storage.googleapis.com/'),
                    'gs_uri': gs_uri,
                    'bytes': os.path.getsize(path),
                    'sha256': sha256(path),
                }
            )

    tables = os.path.join(PACKAGE, 'tables')
    os.makedirs(tables, exist_ok=True)
    pd.DataFrame(chip_rows).to_csv(os.path.join(tables, 'chips.csv'), index=False)
    pd.DataFrame(original_rows).to_csv(
        os.path.join(tables, 'originals.csv'), index=False
    )

    shutil.make_archive(f'{PACKAGE}-0.1.0', 'zip', '.', PACKAGE)
    print(f'{PACKAGE}-0.1.0.zip', sha256(f'{PACKAGE}-0.1.0.zip'))
