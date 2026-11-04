#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import json
import os
import shutil
from datetime import date, timedelta

import numpy as np

SIZE = 32
NUM_SAMPLES = 5
MAX_NUM_TIME_STEPS = 10
np.random.seed(0)

FILENAME_HIERARCHY = dict[str, 'FILENAME_HIERARCHY'] | list[str]

filenames: FILENAME_HIERARCHY = {
    'DATA_S2': ['S2'],
    'DATA_S1A': ['S1A'],
    'DATA_S1D': ['S1D'],
    'ANNOTATIONS': ['TARGET'],
    'INSTANCE_ANNOTATIONS': ['INSTANCES'],
}


def create_file(path: str) -> None:
    for i in range(NUM_SAMPLES):
        new_path = f'{path}_{i}.npy'
        fn = os.path.basename(new_path)
        t = np.random.randint(1, MAX_NUM_TIME_STEPS)
        if fn.startswith('S2'):
            data = np.random.randint(0, 256, size=(t, 10, SIZE, SIZE)).astype(np.int16)
        elif fn.startswith(('S1A', 'S1D')):
            data = np.random.randint(0, 256, size=(t, 3, SIZE, SIZE)).astype(np.float16)
        elif fn.startswith('TARGET'):
            data = np.random.randint(0, 20, size=(3, SIZE, SIZE)).astype(np.uint8)
        elif fn.startswith('INSTANCES'):
            data = np.random.randint(0, 100, size=(SIZE, SIZE)).astype(np.int64)
        np.save(new_path, data)


def create_directory(directory: str, hierarchy: FILENAME_HIERARCHY) -> None:
    if isinstance(hierarchy, dict):
        # Recursive case
        for key, value in hierarchy.items():
            path = os.path.join(directory, key)
            os.makedirs(path, exist_ok=True)
            create_directory(path, value)
    else:
        # Base case
        for value in hierarchy:
            path = os.path.join(directory, value)
            create_file(path)


def make_dates(t: int) -> dict[str, int]:
    start = date(2018, 10, 4)
    return {
        str(i): int((start + timedelta(days=6 * i)).strftime('%Y%m%d'))
        for i in range(t)
    }


def num_steps(modality: str, i: int) -> int:
    path = os.path.join('PASTIS-R', f'DATA_{modality}', f'{modality}_{i}.npy')
    return np.load(path).shape[0]


if __name__ == '__main__':
    create_directory('PASTIS-R', filenames)

    features = []
    for i in range(NUM_SAMPLES):
        features.append(
            {
                'id': str(i),
                'type': 'Feature',
                'properties': {
                    'Fold': (i % 5) + 1,
                    'ID_PATCH': i,
                    'dates-S1A': make_dates(num_steps('S1A', i)),
                    'dates-S1D': make_dates(num_steps('S1D', i)),
                    'dates-S2': make_dates(num_steps('S2', i)),
                    'id': str(i),
                },
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                },
            }
        )

    with open(os.path.join('PASTIS-R', 'metadata.geojson'), 'w') as f:
        json.dump({'type': 'FeatureCollection', 'features': features}, f, indent=2)

    filename = 'PASTIS-R.zip'
    shutil.make_archive(filename.replace('.zip', ''), 'zip', '.', 'PASTIS-R')
