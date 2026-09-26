# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Blending utilities for tiled inference."""

import math
import pathlib
from collections import defaultdict
from typing import Any, Literal, NotRequired, TypedDict

import numpy as np
import rasterio
from pyproj import CRS as PROJ_CRS
from rasterio.crs import CRS as RIO_CRS
from rasterio.transform import Affine
from tqdm import tqdm

from torchgeo.datasets.utils import Path


class PatchMetadata(TypedDict):
    """Metadata for a single prediction patch.

    .. versionadded:: 0.11
    """

    patch_id: int
    file: pathlib.Path
    geo_bbox: tuple[float, float, float, float]
    transform: list[float]
    bbox: NotRequired[tuple[int, int, int, int]]
    edge_deltas: NotRequired[tuple[int, int, int, int]]
    boundary_edges: NotRequired[tuple[bool, bool, bool, bool]]


def _patch_bounds_in_output_crs(
    meta: PatchMetadata, output_crs: PROJ_CRS | None
) -> tuple[float, float, float, float]:
    """Return a patch's geographic bounds expressed in the output CRS.

    Today every patch is in the output CRS, and the bounds are returned directly.
    When native-CRS samples are implemented, reproject the bounds to the output CRS.
    ``rasterio.warp.transform_bounds(patch_crs, output_crs, *meta['geo_bbox'])``

    Args:
        meta: Metadata for a single patch.
        output_crs: CRS the mosaic is written in.

    Returns:
        Patch bounds as ``(xmin, ymin, xmax, ymax)`` in the output CRS.
    """
    return meta['geo_bbox']


def _snap(value: float) -> int:
    """Round half up, so a constant sub-pixel offset shifts every patch the same way.

    Python's :func:`round` rounds half to even, which turns a constant 0.5 px
    offset into a 1 px jitter between neighbouring patches.

    Args:
        value: Pixel coordinate to snap.

    Returns:
        Nearest integer, ties rounded up.
    """
    return math.floor(value + 0.5)


def _patch_crs(src: rasterio.io.DatasetReader, patch_id: int) -> PROJ_CRS:
    """Read the CRS of a patch file.

    Args:
        src: Open patch dataset.
        patch_id: Identifier used in the error message.

    Returns:
        The patch CRS.

    Raises:
        ValueError: If the patch has no CRS.
    """
    if src.crs is None:
        raise ValueError(f'Patch {patch_id} has no CRS')
    return PROJ_CRS.from_user_input(src.crs)


def _get_boundary_edges(
    geo_bbox: tuple[float, float, float, float],
    scene_bounds: tuple[float, float, float, float],
    pixel_size: tuple[float, float],
    *,
    north_up: bool = True,
) -> tuple[bool, bool, bool, bool]:
    """Determine which edges of a patch touch the scene boundary.

    A tolerance of half a pixel per axis is used for boundary detection to handle
    floating-point imprecision, so a patch edge is boundary-touching only if it is
    closer to the scene edge than to the next pixel.

    Edges are given in **array order**: top means first rows of the array, bottom
    means last rows. For north-up rasters the first row is geo ymax; for south-up
    it is geo ymin.

    Args:
        geo_bbox: Patch bounds as (xmin, ymin, xmax, ymax) in geo coordinates.
        scene_bounds: Scene bounds as (minx, miny, maxx, maxy) in geo coordinates.
        pixel_size: Size of one pixel as (xres, yres) in geo units.
        north_up: Whether the raster has a negative y-resolution (north at the top).
            South-up rasters (positive y-resolution) swap top/bottom assignments.

    Returns:
        Tuple of (top, bottom, left, right) flags, True where the edge touches the
        scene boundary.
    """
    minx, miny, maxx, maxy = scene_bounds
    tolerance_x = abs(pixel_size[0]) * 0.5
    tolerance_y = abs(pixel_size[1]) * 0.5

    left = abs(geo_bbox[0] - minx) < tolerance_x
    right = abs(geo_bbox[2] - maxx) < tolerance_x
    at_geo_top = abs(geo_bbox[3] - maxy) < tolerance_y
    at_geo_bottom = abs(geo_bbox[1] - miny) < tolerance_y

    if north_up:
        # Array row 0 = geo ymax (north)
        top, bottom = at_geo_top, at_geo_bottom
    else:
        # Array row 0 = geo ymin (south)
        top, bottom = at_geo_bottom, at_geo_top

    return top, bottom, left, right


def _get_edge_deltas(
    geo_bbox: tuple[float, float, float, float],
    scene_bounds: tuple[float, float, float, float],
    pixel_size: tuple[float, float],
    delta: int,
    *,
    north_up: bool = True,
) -> tuple[int, int, int, int]:
    """Compute per-edge crop amounts based on boundary proximity.

    Patches touching scene boundaries preserve their edge pixels (delta=0 on that
    edge) to avoid black borders. Interior edges are cropped normally to remove
    neural network edge artifacts. See :func:`_get_boundary_edges` for how
    boundary edges are detected and for the array-order convention.

    Args:
        geo_bbox: Patch bounds as (xmin, ymin, xmax, ymax) in geo coordinates.
        scene_bounds: Scene bounds as (minx, miny, maxx, maxy) in geo coordinates.
        pixel_size: Size of one pixel as (xres, yres) in geo units.
        delta: Default pixels to crop from edges.
        north_up: Whether the raster has a negative y-resolution (north at the top).

    Returns:
        Tuple of (top, bottom, left, right) crop amounts in pixels, where top/bottom
        refer to array rows (not geographic direction).
    """
    boundary = _get_boundary_edges(
        geo_bbox, scene_bounds, pixel_size, north_up=north_up
    )
    top, bottom, left, right = (0 if at_boundary else delta for at_boundary in boundary)
    return top, bottom, left, right


def _reconstruct_scene_from_patches(
    patch_metadata: list[PatchMetadata],
    patch_size: tuple[int, int],
    delta: int = 0,
    output_crs: PROJ_CRS | None = None,
) -> tuple[tuple[int, int], Affine]:
    """Reconstruct scene-level transform and shape from per-patch transforms.

    This leverages per-patch transforms to reconstruct the full scene metadata
    without needing upfront dataset information. Converts geo coordinates to
    pixel coordinates using a global reference frame.

    Supports both north-up (negative y-resolution) and south-up (positive
    y-resolution) rasters, determined automatically from the patch transforms.

    Args:
        patch_metadata: List of dicts with 'geo_bbox' and 'transform'.
            geo_bbox is (geo_xmin, geo_ymin, geo_xmax, geo_ymax) in geo coordinates.
            transform is the affine as a list [a, b, c, d, e, f]:
                | a  b  c |   where c, f are the origin
                | d  e  f |   and a, e are x_res, y_res
                | 0  0  1 |
        patch_size: Size of each patch as (height, width) in pixels.
        delta: Pixels to crop from patch edges before blending.
        output_crs: CRS the scene is reconstructed in (used to place each patch).

    Returns:
        output_shape: (height, width) of full scene.
        scene_transform: Affine transform for the full scene.

    Raises:
        ValueError: If patches have inconsistent resolutions or metadata is empty.
    """
    if not patch_metadata:
        raise ValueError('patch_metadata is empty')

    first_transform = patch_metadata[0]['transform']
    x_res = first_transform[0]
    y_res = first_transform[4]  # signed: negative for north-up, positive for south-up

    for meta in patch_metadata[1:]:
        t = meta['transform']
        if abs(t[0] - x_res) > 1e-6 or abs(t[4] - y_res) > 1e-6:
            raise ValueError(
                f'Inconsistent resolutions: first patch has ({x_res}, {y_res}), '
                f'but patch {meta["patch_id"]} has ({t[0]}, {t[4]})'
            )

    boxes = [_patch_bounds_in_output_crs(meta, output_crs) for meta in patch_metadata]
    all_geo_xmin = [b[0] for b in boxes]
    all_geo_xmax = [b[2] for b in boxes]
    all_geo_ymin = [b[1] for b in boxes]
    all_geo_ymax = [b[3] for b in boxes]

    scene_bounds = (
        min(all_geo_xmin),
        min(all_geo_ymin),
        max(all_geo_xmax),
        max(all_geo_ymax),
    )
    global_geo_xmin = scene_bounds[0]
    north_up = y_res < 0
    # For north-up the origin row sits at ymax; for south-up at ymin
    origin_y = scene_bounds[3] if north_up else scene_bounds[1]

    patch_h, patch_w = patch_size

    for meta in patch_metadata:
        geo_bbox = _patch_bounds_in_output_crs(meta, output_crs)
        top, bottom, left, right = _get_edge_deltas(
            geo_bbox, scene_bounds, (x_res, abs(y_res)), delta, north_up=north_up
        )
        meta['edge_deltas'] = (top, bottom, left, right)
        meta['boundary_edges'] = _get_boundary_edges(
            geo_bbox, scene_bounds, (x_res, abs(y_res)), north_up=north_up
        )

        patch_geo_xmin = geo_bbox[0] + left * x_res
        effective_patch_w = patch_w - left - right
        effective_patch_h = patch_h - top - bottom

        # Compute the y coordinate of this patch's array origin (row 0 after crop)
        if north_up:
            patch_origin_y = geo_bbox[3] - top * abs(y_res)
        else:
            patch_origin_y = geo_bbox[1] + top * y_res

        patch_col_start = _snap((patch_geo_xmin - global_geo_xmin) / x_res)
        patch_row_start = _snap((patch_origin_y - origin_y) / y_res)

        meta['bbox'] = (
            patch_col_start,
            patch_row_start,
            patch_col_start + effective_patch_w,
            patch_row_start + effective_patch_h,
        )

    all_x_starts = [meta['bbox'][0] for meta in patch_metadata]
    all_y_starts = [meta['bbox'][1] for meta in patch_metadata]
    all_x_stops = [meta['bbox'][2] for meta in patch_metadata]
    all_y_stops = [meta['bbox'][3] for meta in patch_metadata]

    min_x = min(all_x_starts)
    min_y = min(all_y_starts)
    max_x = max(all_x_stops)
    max_y = max(all_y_stops)

    for meta in patch_metadata:
        bbox = meta['bbox']
        meta['bbox'] = (
            bbox[0] - min_x,
            bbox[1] - min_y,
            bbox[2] - min_x,
            bbox[3] - min_y,
        )

    scene_geo_xmin = global_geo_xmin + min_x * x_res
    scene_origin_y = origin_y + min_y * y_res

    scene_transform = Affine(x_res, 0, scene_geo_xmin, 0, y_res, scene_origin_y)

    output_width = max_x - min_x
    output_height = max_y - min_y

    return (output_height, output_width), scene_transform


def get_blend_mask(
    patch_size: int | tuple[int, int],
    overlap: int,
    delta: int,
    method: Literal['cosine', 'linear'] = 'cosine',
    edge_deltas: tuple[int, int, int, int] | None = None,
    boundary_edges: tuple[bool, bool, bool, bool] | None = None,
) -> np.typing.NDArray[np.floating[Any]]:
    """Generate blend mask for weighted patch merging.

    Args:
        patch_size: Size of patch (H, W) or single int.
        overlap: Overlap in pixels on each side.
        delta: Default pixels to crop from edges.
        method: Blending method ('cosine' or 'linear').
        edge_deltas: Optional per-edge crop amounts (top, bottom, left, right).
            Defaults to *delta* on every edge.
        boundary_edges: Optional per-edge flags (top, bottom, left, right). Edges
            marked True touch the scene boundary and get no blend ramp, so the
            scene edge keeps full weight. Defaults to ramps on every edge.

    Returns:
        Blend mask with values in [0, 1].

    Raises:
        ValueError: If method is not 'cosine' or 'linear', the crop removes the
            whole patch, or the overlap exceeds half of the cropped patch.
    """
    if isinstance(patch_size, int):
        h = w = patch_size
    else:
        h, w = patch_size

    if edge_deltas is not None:
        top, bottom, left, right = edge_deltas
    else:
        top = bottom = left = right = delta

    if boundary_edges is None:
        boundary_edges = (False, False, False, False)
    apply_top_ramp, apply_bottom_ramp, apply_left_ramp, apply_right_ramp = (
        not at_boundary for at_boundary in boundary_edges
    )

    h_crop = h - top - bottom
    w_crop = w - left - right

    if h_crop <= 0 or w_crop <= 0:
        raise ValueError('delta crops away the entire patch')
    if 2 * overlap > min(h_crop, w_crop):
        raise ValueError(
            'overlap exceeds half of the cropped patch dimensions, so the ramps on '
            'opposite edges would overwrite each other'
        )

    if method == 'cosine':
        y = np.ones(h_crop, dtype=np.float32)
        if overlap > 0:
            ramp = np.cos(np.pi * (np.arange(overlap) + 1) / (overlap + 1)) / 2 + 0.5
            if apply_top_ramp:
                y[:overlap] = ramp[::-1]
            if apply_bottom_ramp:
                y[-overlap:] = ramp

        x = np.ones(w_crop, dtype=np.float32)
        if overlap > 0:
            ramp = np.cos(np.pi * (np.arange(overlap) + 1) / (overlap + 1)) / 2 + 0.5
            if apply_left_ramp:
                x[:overlap] = ramp[::-1]
            if apply_right_ramp:
                x[-overlap:] = ramp

        mask = y[:, None] * x[None, :]
    elif method == 'linear':
        y = np.ones(h_crop, dtype=np.float32)
        if overlap > 0:
            # Skip the exact 0 and 1 endpoints so opposite ramps sum to 1 and no
            # pixel gets zero weight, mirroring the cosine ramp.
            ramp = np.linspace(0, 1, overlap + 2, dtype=np.float32)[1:-1]
            if apply_top_ramp:
                y[:overlap] = ramp
            if apply_bottom_ramp:
                y[-overlap:] = ramp[::-1]

        x = np.ones(w_crop, dtype=np.float32)
        if overlap > 0:
            # Skip the exact 0 and 1 endpoints so opposite ramps sum to 1 and no
            # pixel gets zero weight, mirroring the cosine ramp.
            ramp = np.linspace(0, 1, overlap + 2, dtype=np.float32)[1:-1]
            if apply_left_ramp:
                x[:overlap] = ramp
            if apply_right_ramp:
                x[-overlap:] = ramp[::-1]

        mask = y[:, None] * x[None, :]
    else:
        raise ValueError(f'Unknown blend method: {method}')

    return mask


def _build_grid_index(
    patch_metadata: list[PatchMetadata], grid_size: int
) -> dict[tuple[int, int], list[int]]:
    """Build simple grid-based spatial index for fast patch lookup.

    Args:
        patch_metadata: List of dicts with 'bbox' (x_start, y_start, x_stop, y_stop).
        grid_size: Size of grid cells in pixels.

    Returns:
        Dict mapping (grid_row, grid_col) to list of patch indices.
    """
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)

    for idx, meta in enumerate(patch_metadata):
        bbox = meta['bbox']
        x_start, y_start, x_stop, y_stop = bbox

        grid_col_start = x_start // grid_size
        grid_col_end = (x_stop - 1) // grid_size
        grid_row_start = y_start // grid_size
        grid_row_end = (y_stop - 1) // grid_size

        for gr in range(grid_row_start, grid_row_end + 1):
            for gc in range(grid_col_start, grid_col_end + 1):
                grid[(gr, gc)].append(idx)

    return dict(grid)


def _query_grid_index(
    grid: dict[tuple[int, int], list[int]],
    patch_metadata: list[PatchMetadata],
    chunk_y: int,
    chunk_x: int,
    chunk_h: int,
    chunk_w: int,
    grid_size: int,
) -> list[PatchMetadata]:
    """Query grid index for patches overlapping chunk.

    Args:
        grid: Grid index from _build_grid_index.
        patch_metadata: Original patch metadata list.
        chunk_y: Top-left y coordinate of chunk.
        chunk_x: Top-left x coordinate of chunk.
        chunk_h: Height of chunk.
        chunk_w: Width of chunk.
        grid_size: Size of grid cells in pixels.

    Returns:
        List of patch metadata dicts that may overlap the chunk.
    """
    grid_col_start = chunk_x // grid_size
    grid_col_end = (chunk_x + chunk_w - 1) // grid_size
    grid_row_start = chunk_y // grid_size
    grid_row_end = (chunk_y + chunk_h - 1) // grid_size

    candidate_indices: set[int] = set()
    for gr in range(grid_row_start, grid_row_end + 1):
        for gc in range(grid_col_start, grid_col_end + 1):
            candidate_indices.update(grid.get((gr, gc), []))

    return [patch_metadata[idx] for idx in candidate_indices]


def _resolve_output_grid(
    patch_metadata: list[PatchMetadata],
    output_crs: PROJ_CRS | None,
    patch_size: tuple[int, int],
    delta: int,
    dataset_bounds: tuple[float, float, float, float] | None,
    dataset_res: tuple[float, float] | None,
) -> tuple[tuple[int, int], Affine]:
    """Resolve the output grid (shape + transform) and assign each patch's pixel window.

    Single decision point for the output grid. Extent and resolution come from the
    dataset when known, otherwise from the patches. Sets ``meta['bbox']`` and
    ``meta['edge_deltas']`` and ``meta['boundary_edges']`` on each patch as a side
    effect, which the merge relies on.

    Args:
        patch_metadata: Metadata for every written patch.
        output_crs: CRS the mosaic is written in.
        patch_size: Patch size as ``(height, width)`` in pixels.
        delta: Pixels to crop from patch edges before blending.
        dataset_bounds: Scene bounds ``(minx, miny, maxx, maxy)``, if known upfront.
        dataset_res: Scene resolution ``(xres, yres)``, if known upfront.

    Returns:
        output_shape: ``(height, width)`` of the full scene.
        scene_transform: Affine transform for the full scene.

    Raises:
        ValueError: If *dataset_res* does not match the resolution of the patches.
    """
    patch_h, patch_w = patch_size

    if dataset_bounds is not None and dataset_res is not None:
        minx, miny, maxx, maxy = dataset_bounds
        x_res_ds, y_res_ds = dataset_res
        output_width = round((maxx - minx) / x_res_ds)
        output_height = round((maxy - miny) / y_res_ds)
        output_shape = (output_height, output_width)

        first_transform = patch_metadata[0]['transform']
        x_res = first_transform[0]
        y_res_signed = first_transform[4]
        north_up = y_res_signed < 0
        y_res = abs(y_res_signed)

        if abs(x_res - x_res_ds) > 1e-6 or abs(y_res - y_res_ds) > 1e-6:
            raise ValueError(
                f'dataset_res {dataset_res} does not match the patch resolution '
                f'({x_res}, {y_res}); patches are placed in patch pixel units'
            )

        if north_up:
            scene_transform = Affine(x_res_ds, 0, minx, 0, -y_res_ds, maxy)
        else:
            scene_transform = Affine(x_res_ds, 0, minx, 0, y_res_ds, miny)

        scene_bounds = dataset_bounds
        origin_y = maxy if north_up else miny

        for meta in patch_metadata:
            geo_bbox = _patch_bounds_in_output_crs(meta, output_crs)
            top, bottom, left, right = _get_edge_deltas(
                geo_bbox, scene_bounds, (x_res, y_res), delta, north_up=north_up
            )
            meta['edge_deltas'] = (top, bottom, left, right)
            meta['boundary_edges'] = _get_boundary_edges(
                geo_bbox, scene_bounds, (x_res, y_res), north_up=north_up
            )

            patch_geo_xmin = geo_bbox[0] + left * x_res
            effective_patch_w = patch_w - left - right
            effective_patch_h = patch_h - top - bottom

            if north_up:
                patch_origin_y = geo_bbox[3] - top * y_res
            else:
                patch_origin_y = geo_bbox[1] + top * y_res

            patch_col_start = _snap((patch_geo_xmin - minx) / x_res)
            patch_row_start = _snap((patch_origin_y - origin_y) / y_res_signed)

            meta['bbox'] = (
                patch_col_start,
                patch_row_start,
                patch_col_start + effective_patch_w,
                patch_row_start + effective_patch_h,
            )

        return output_shape, scene_transform
    else:
        return _reconstruct_scene_from_patches(
            patch_metadata, (patch_h, patch_w), delta, output_crs
        )


def weighted_merge(
    patch_metadata: list[PatchMetadata],
    num_classes: int,
    overlap: int,
    delta: int,
    output_path: Path,
    blend_method: Literal['cosine', 'linear'] = 'cosine',
    crs: PROJ_CRS | None = None,
    chunk_size: int = 4096,
    dataset_bounds: tuple[float, float, float, float] | None = None,
    dataset_res: tuple[float, float] | None = None,
    nodata: int | None = 255,
    **kwargs: Any,
) -> None:
    """Merge patches from disk with weighted blending.

    Uses chunked processing with spatial indexing for memory-efficient
    merging of arbitrarily large scenes. Pixels not covered by any patch, for
    example outside a sampled region of interest, are written as *nodata*.

    Args:
        patch_metadata: List of dicts with 'file', 'geo_bbox', 'transform'.
        num_classes: Number of classes.
        overlap: Overlap in pixels.
        delta: Pixels to crop from edges.
        output_path: Where to save GeoTIFF.
        blend_method: 'cosine' or 'linear'. Cosine blending uses a Hann window
            weight mask to reduce edge artifacts, as recommended by
            https://doi.org/10.1371/journal.pone.0229839.
        crs: :term:`coordinate reference system (CRS)` the mosaic is written in.
            Defaults to the CRS of the patches. Every patch must already be in
            this CRS; reprojecting patches is not supported.
        chunk_size: Size of chunks for processing. Peak memory is about
            ``num_classes * chunk_size**2 * 4`` bytes.
        dataset_bounds: Original dataset bounds (minx, miny, maxx, maxy).
        dataset_res: Original dataset resolution as (xres, yres).
        nodata: Value written where no patch contributes, and recorded as the
            nodata value of the output. Must not collide with a class index.
            If None, uncovered pixels are written as class 0 and the output has
            no nodata value.
        **kwargs: Additional keyword arguments passed to
            :class:`~torchgeo.callbacks.writer.GeoTIFFWriter` (e.g. ``compress``,
            ``overview_resampling``).

    Raises:
        ValueError: If *patch_metadata* is empty, *num_classes* does not fit the
            uint8 output, *nodata* collides with a class index, *delta* exceeds
            half of *overlap* (neighbouring patches would no longer meet after
            cropping), or a patch has no CRS, a different CRS, or a different
            size than the first patch.

    .. versionadded:: 0.11
    """
    from torchgeo.callbacks.writer import GeoTIFFWriter

    if not patch_metadata:
        raise ValueError('patch_metadata is empty')
    if num_classes > 256:
        raise ValueError(f'num_classes ({num_classes}) does not fit the uint8 output')
    if nodata is not None and not num_classes <= nodata <= 255:
        raise ValueError(
            f'nodata ({nodata}) must be a uint8 value outside the class range '
            f'0..{num_classes - 1}'
        )
    if 2 * delta > overlap:
        raise ValueError(
            f'delta ({delta}) must not exceed half of overlap ({overlap}), otherwise '
            'neighbouring patches no longer meet after cropping and leave gaps'
        )

    with rasterio.open(patch_metadata[0]['file']) as src:
        patch_h, patch_w = src.height, src.width
        if crs is None:
            crs = _patch_crs(src, patch_metadata[0]['patch_id'])

    output_shape, scene_transform = _resolve_output_grid(
        patch_metadata, crs, (patch_h, patch_w), delta, dataset_bounds, dataset_res
    )

    grid_size = chunk_size * 2
    grid = _build_grid_index(patch_metadata, grid_size)

    effective_overlap = overlap - 2 * delta
    mask_cache: dict[
        tuple[tuple[int, int, int, int], tuple[bool, bool, bool, bool]],
        np.typing.NDArray[np.floating[Any]],
    ] = {}

    writer = GeoTIFFWriter(
        output_path=output_path,
        width=output_shape[1],
        height=output_shape[0],
        num_bands=1,
        crs=RIO_CRS.from_user_input(crs),
        transform=scene_transform,
        nodata=nodata,
        **kwargs,
    )

    height, width = output_shape

    with writer:
        chunk_iter = [
            (cy, cx)
            for cy in range(0, height, chunk_size)
            for cx in range(0, width, chunk_size)
        ]
        for chunk_y, chunk_x in tqdm(chunk_iter, desc='Merging chunks'):
            chunk_h = min(chunk_size, height - chunk_y)
            chunk_w = min(chunk_size, width - chunk_x)

            chunk_output = np.zeros((num_classes, chunk_h, chunk_w), dtype=np.float32)
            chunk_weights = np.zeros((chunk_h, chunk_w), dtype=np.float32)

            overlapping = _query_grid_index(
                grid, patch_metadata, chunk_y, chunk_x, chunk_h, chunk_w, grid_size
            )
            for meta in overlapping:
                with rasterio.open(meta['file']) as src:
                    if _patch_crs(src, meta['patch_id']) != crs:
                        raise ValueError(
                            f'Patch {meta["patch_id"]} is in CRS {src.crs}, but the '
                            f'output CRS is {crs}. Reprojecting patches is not supported.'
                        )
                    if (src.height, src.width) != (patch_h, patch_w):
                        raise ValueError(
                            f'Patch {meta["patch_id"]} has size '
                            f'{(src.height, src.width)}, but the first patch has size '
                            f'{(patch_h, patch_w)}; all patches must share one size'
                        )
                    patch_data = src.read().astype(np.float32)

                edge_deltas = meta.get('edge_deltas', (delta, delta, delta, delta))
                boundary_edges = meta.get(
                    'boundary_edges', (False, False, False, False)
                )
                top, bottom, left, right = edge_deltas

                bottom_slice = -bottom if bottom > 0 else None
                right_slice = -right if right > 0 else None
                patch_data = patch_data[:, top:bottom_slice, left:right_slice]

                mask_key = (edge_deltas, boundary_edges)
                if mask_key not in mask_cache:
                    mask_cache[mask_key] = get_blend_mask(
                        (patch_h, patch_w),
                        effective_overlap,
                        delta,
                        blend_method,
                        edge_deltas=edge_deltas,
                        boundary_edges=boundary_edges,
                    )
                blend_mask = mask_cache[mask_key]

                bbox = meta['bbox']
                patch_col_start = bbox[0]
                patch_row_start = bbox[1]

                current_patch_h, current_patch_w = (
                    patch_data.shape[1],
                    patch_data.shape[2],
                )

                overlap_col_start = max(0, patch_col_start - chunk_x)
                overlap_row_start = max(0, patch_row_start - chunk_y)
                overlap_col_end = min(
                    chunk_w, patch_col_start + current_patch_w - chunk_x
                )
                overlap_row_end = min(
                    chunk_h, patch_row_start + current_patch_h - chunk_y
                )

                if (
                    overlap_col_end <= overlap_col_start
                    or overlap_row_end <= overlap_row_start
                ):
                    continue

                patch_col_start_local = max(0, chunk_x - patch_col_start)
                patch_row_start_local = max(0, chunk_y - patch_row_start)
                patch_col_end_local = patch_col_start_local + (
                    overlap_col_end - overlap_col_start
                )
                patch_row_end_local = patch_row_start_local + (
                    overlap_row_end - overlap_row_start
                )

                patch_region = patch_data[
                    :,
                    patch_row_start_local:patch_row_end_local,
                    patch_col_start_local:patch_col_end_local,
                ]
                mask_region = blend_mask[
                    patch_row_start_local:patch_row_end_local,
                    patch_col_start_local:patch_col_end_local,
                ]

                chunk_output[
                    :,
                    overlap_row_start:overlap_row_end,
                    overlap_col_start:overlap_col_end,
                ] += patch_region * mask_region[None, :, :]
                chunk_weights[
                    overlap_row_start:overlap_row_end, overlap_col_start:overlap_col_end
                ] += mask_region

            uncovered = chunk_weights == 0
            min_weight = 1e-6
            chunk_weights = np.maximum(chunk_weights, min_weight)
            chunk_output = chunk_output / chunk_weights[None, :, :]
            chunk_labels = np.argmax(chunk_output, axis=0).astype(np.uint8)
            if nodata is not None:
                chunk_labels[uncovered] = nodata

            writer.write_chunk(chunk_labels, chunk_y, chunk_x)
