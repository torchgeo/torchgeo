#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import shapely.geometry
from rasterio.crs import CRS
from rasterio.transform import Affine
from torchvision.datasets.utils import calculate_md5

suffix_to_key_map = {
    'a_naip': 'naip',
    'b_nlcd': 'nlcd',
    'c_roads': 'roads',
    'd_water': 'water',
    'd1_waterways': 'waterways',
    'd2_waterbodies': 'waterbodies',
    'e_buildings': 'buildings',
    'h_highres_labels': 'lc',
    'prior_from_cooccurrences_101_31': 'prior',
    'prior_from_cooccurrences_101_31_no_osm_no_buildings': 'prior_no_osm_no_buildings',
}

layer_data_profiles: dict[str, dict[Any, Any]] = {
    'a_naip': {
        'profile': {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'count': 4,
            'crs': CRS.from_epsg(26914),
            'blockxsize': 512,
            'blockysize': 512,
            'tiled': True,
            'compress': 'deflate',
            'interleave': 'pixel',
        },
        'data_type': 'continuous',
        'vals': (4, 255),
    },
    'b_nlcd': {
        'profile': {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'count': 1,
            'crs': CRS.from_epsg(26914),
            'blockxsize': 512,
            'blockysize': 512,
            'tiled': True,
            'compress': 'deflate',
            'interleave': 'band',
        },
        'data_type': 'categorical',
        'vals': [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15],
    },
    'c_roads': {
        'profile': {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'count': 1,
            'crs': CRS.from_epsg(26914),
            'blockxsize': 512,
            'blockysize': 512,
            'tiled': True,
            'compress': 'deflate',
            'interleave': 'band',
        },
        'data_type': 'categorical',
        'vals': [0, 1],
    },
    'd1_waterways': {
        'profile': {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'count': 1,
            'crs': CRS.from_epsg(26914),
            'blockxsize': 512,
            'blockysize': 512,
            'tiled': True,
            'compress': 'deflate',
            'interleave': 'band',
        },
        'data_type': 'categorical',
        'vals': [0, 1],
    },
    'd2_waterbodies': {
        'profile': {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'count': 1,
            'crs': CRS.from_epsg(26914),
            'blockxsize': 512,
            'blockysize': 512,
            'tiled': True,
            'compress': 'deflate',
            'interleave': 'band',
        },
        'data_type': 'categorical',
        'vals': [0, 1],
    },
    'd_water': {
        'profile': {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'count': 1,
            'crs': CRS.from_epsg(26914),
            'blockxsize': 512,
            'blockysize': 512,
            'tiled': True,
            'compress': 'deflate',
            'interleave': 'band',
        },
        'data_type': 'categorical',
        'vals': [0, 1],
    },
    'e_buildings': {
        'profile': {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'count': 1,
            'crs': CRS.from_epsg(26914),
            'blockxsize': 512,
            'blockysize': 512,
            'tiled': True,
            'compress': 'deflate',
            'interleave': 'band',
        },
        'data_type': 'categorical',
        'vals': [0, 1],
    },
    'h_highres_labels': {
        'profile': {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'count': 1,
            'crs': CRS.from_epsg(26914),
            'blockxsize': 512,
            'blockysize': 512,
            'tiled': True,
            'compress': 'deflate',
            'interleave': 'band',
        },
        'data_type': 'categorical',
        'vals': [10, 20, 30, 40, 70],
    },
    'prior_from_cooccurrences_101_31': {
        'profile': {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'count': 5,
            'crs': CRS.from_epsg(26914),
            'blockxsize': 512,
            'blockysize': 512,
            'tiled': True,
            'compress': 'deflate',
            'interleave': 'band',
        },
        'data_type': 'continuous',
        'vals': (0, 225),
    },
    'prior_from_cooccurrences_101_31_no_osm_no_buildings': {
        'profile': {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'count': 5,
            'crs': CRS.from_epsg(26914),
            'blockxsize': 512,
            'blockysize': 512,
            'tiled': True,
            'compress': 'deflate',
            'interleave': 'band',
        },
        'data_type': 'continuous',
        'vals': (0, 220),
    },
}

tile_list = [
    'pittsburgh_pa-2010_1m-train_tiles-debuffered/4007925_se',
    'austin_tx-2012_1m-test_tiles-debuffered/3009726_sw',
]


def write_data(path: str, profile: dict[Any, Any], data_type: Any, vals: Any) -> None:
    assert all(key in profile for key in ('count', 'height', 'width', 'dtype'))
    with rasterio.open(path, 'w', **profile) as dst:
        size = (profile['count'], profile['height'], profile['width'])
        dtype = np.dtype(profile['dtype'])
        if data_type == 'continuous':
            data = np.random.randint(vals[0], vals[1] + 1, size=size, dtype=dtype)
        elif data_type == 'categorical':
            data = np.random.choice(vals, size=size).astype(dtype)
        else:
            raise ValueError(f'{data_type} is not recognized')
        dst.write(data)


def generate_test_data(root: str) -> str:
    """Creates test data archive for the EnviroAtlas dataset and returns its md5 hash.

    Args:
        root (str): Path to store test data

    Returns:
        str: md5 hash of created archive
    """
    size = (64, 64)
    folder_path = os.path.join(root, 'enviroatlas_lotp')

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for prefix in tile_list:
        for suffix, data_profile in layer_data_profiles.items():
            img_path = os.path.join(folder_path, f'{prefix}_{suffix}.tif')
            img_dir = os.path.dirname(img_path)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            data_profile['profile']['height'] = size[0]
            data_profile['profile']['width'] = size[1]
            data_profile['profile']['transform'] = Affine(
                1.0, 0.0, 608170.0, 0.0, -1.0, 3381430.0
            )

            write_data(
                img_path,
                data_profile['profile'],
                data_profile['data_type'],
                data_profile['vals'],
            )

    # build the spatial index
    # Create a list to store all the features
    features = []

    for prefix in tile_list:
        img_path = os.path.join(folder_path, f'{prefix}_a_naip.tif')
        with rasterio.open(img_path) as f:
            # Create a box geometry from the raster bounds
            box_geom = shapely.geometry.box(*f.bounds)
            # Transform the geometry to EPSG:3857 using GeoPandas
            box_gdf = gpd.GeoDataFrame(geometry=[box_geom], crs=f.crs)
            transformed_gdf = box_gdf.to_crs('EPSG:3857')
            geom = transformed_gdf.geometry.iloc[0]

        # Create properties dictionary with all required columns
        properties = {
            'split': prefix.split('/')[0].replace('_tiles-debuffered', ''),
            'naip': '',
            'nlcd': '',
            'roads': '',
            'water': '',
            'waterways': '',
            'waterbodies': '',
            'buildings': '',
            'lc': '',
            'prior_no_osm_no_buildings': '',
            'prior': '',
        }

        # Fill in the actual values
        for suffix, data_profile in layer_data_profiles.items():
            key = suffix_to_key_map[suffix]
            properties[key] = f'{prefix}_{suffix}.tif'

        # Create feature dictionary
        feature = {'geometry': geom, 'properties': properties}
        features.append(feature)

    # Create GeoDataFrame from features with explicit column types
    gdf = gpd.GeoDataFrame.from_features(features, crs='EPSG:3857')

    # Ensure all string columns have the correct dtype
    string_columns = [
        'split',
        'naip',
        'nlcd',
        'roads',
        'water',
        'waterways',
        'waterbodies',
        'buildings',
        'lc',
        'prior_no_osm_no_buildings',
        'prior',
    ]
    for col in string_columns:
        gdf[col] = gdf[col].astype('string')

    # Save to GeoJSON file
    gdf.to_file(os.path.join(folder_path, 'spatial_index.geojson'), driver='GeoJSON')

    # Create archive
    archive_path = os.path.join(root, 'enviroatlas_lotp')
    shutil.make_archive(archive_path, 'zip', root_dir=root, base_dir='enviroatlas_lotp')
    shutil.rmtree(folder_path)
    md5: str = calculate_md5(archive_path + '.zip')
    return md5


if __name__ == '__main__':
    md5_hash = generate_test_data(os.getcwd())
    print(md5_hash)
