# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for blending utilities."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import rasterio
import torch
from pyproj import CRS
from rasterio.transform import Affine

from torchgeo.callbacks.blending import (
    PatchMetadata,
    _build_grid_index,
    _get_edge_deltas,
    _query_grid_index,
    _reconstruct_scene_from_patches,
    get_blend_mask,
    weighted_merge,
)


def _save_test_patch(
    path: Path, logits: torch.Tensor, transform: list[float], crs: str = 'EPSG:32631'
) -> None:
    """Save test patch as GeoTIFF with one-hot encoded predictions.

    Args:
        path: Output file path.
        logits: Logits tensor of shape (num_classes, H, W).
        transform: Affine transform as list [a, b, c, d, e, f].
        crs: Coordinate reference system.
    """
    num_classes = logits.shape[0]
    class_predictions = logits.argmax(dim=0)
    one_hot = (
        torch.nn.functional.one_hot(class_predictions.long(), num_classes=num_classes)
        .permute(2, 0, 1)
        .to(torch.uint8)
        .numpy()
    )
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=one_hot.shape[1],
        width=one_hot.shape[2],
        count=one_hot.shape[0],
        dtype='uint8',
        transform=Affine(*transform),
        crs=crs,
    ) as dst:
        dst.write(one_hot)


def _make_meta(
    patch_id: int,
    geo_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    transform: list[float] | None = None,
    bbox: tuple[int, int, int, int] | None = None,
) -> PatchMetadata:
    """Build patch metadata for tests that never read the patch file."""
    meta: PatchMetadata = {
        'patch_id': patch_id,
        'file': Path('unused'),
        'geo_bbox': geo_bbox,
        'transform': transform if transform is not None else [1.0, 0, 0, 0, -1.0, 0],
    }
    if bbox is not None:
        meta['bbox'] = bbox
    return meta


class TestReconstructSceneFromPatches:
    """Tests for _reconstruct_scene_from_patches."""

    def test_single_patch(self) -> None:
        """Test reconstruction with single patch."""
        meta = [
            _make_meta(0, (0.0, 360.0, 640.0, 1000.0), [10.0, 0, 0, 0, -10.0, 1000])
        ]

        shape, transform = _reconstruct_scene_from_patches(meta, (64, 64), delta=0)

        assert shape == (64, 64)
        assert transform == Affine(10.0, 0, 0, 0, -10.0, 1000)
        assert meta[0]['bbox'] == (0, 0, 64, 64)

    def test_two_patches_horizontal(self) -> None:
        """Test reconstruction with two horizontal patches."""
        meta = [
            _make_meta(
                0, (100.0, 136.0, 164.0, 200.0), [1.0, 0, 100.0, 0, -1.0, 200.0]
            ),
            _make_meta(
                1, (132.0, 136.0, 196.0, 200.0), [1.0, 0, 132.0, 0, -1.0, 200.0]
            ),
        ]

        shape, transform = _reconstruct_scene_from_patches(meta, (64, 64), delta=0)

        assert shape == (64, 96)
        assert transform == Affine(1.0, 0, 100.0, 0, -1.0, 200.0)
        assert meta[0]['bbox'] == (0, 0, 64, 64)
        assert meta[1]['bbox'] == (32, 0, 96, 64)

    def test_inconsistent_resolutions_raises(self) -> None:
        """Test error on inconsistent resolutions."""
        meta = [
            _make_meta(0, (0.0, 36.0, 64.0, 100.0), [1.0, 0, 0, 0, -1.0, 100]),
            _make_meta(1, (64.0, 36.0, 192.0, 100.0), [2.0, 0, 64, 0, -1.0, 100]),
        ]

        with pytest.raises(ValueError, match='Inconsistent resolutions'):
            _reconstruct_scene_from_patches(meta, (64, 64), delta=0)

    def test_single_patch_south_up(self) -> None:
        """Test reconstruction with a south-up raster (positive y-resolution)."""
        meta = [_make_meta(0, (0.0, 0.0, 640.0, 640.0), [10.0, 0, 0, 0, 10.0, 0])]

        shape, transform = _reconstruct_scene_from_patches(meta, (64, 64), delta=0)

        assert shape == (64, 64)
        assert transform == Affine(10.0, 0, 0, 0, 10.0, 0)
        assert meta[0]['bbox'] == (0, 0, 64, 64)

    def test_two_patches_vertical_south_up(self) -> None:
        """Test reconstruction with two vertical south-up patches."""
        meta = [
            _make_meta(0, (0.0, 0.0, 64.0, 64.0), [1.0, 0, 0.0, 0, 1.0, 0.0]),
            _make_meta(1, (0.0, 32.0, 64.0, 96.0), [1.0, 0, 0.0, 0, 1.0, 32.0]),
        ]

        shape, transform = _reconstruct_scene_from_patches(meta, (64, 64), delta=0)

        assert shape == (96, 64)
        assert transform == Affine(1.0, 0, 0.0, 0, 1.0, 0.0)
        assert meta[0]['bbox'] == (0, 0, 64, 64)
        assert meta[1]['bbox'] == (0, 32, 64, 96)

    def test_two_patches_vertical_south_up_with_delta(self) -> None:
        """South-up reconstruction with delta > 0 crops correct array edges."""
        meta = [
            _make_meta(0, (0.0, 0.0, 64.0, 64.0), [1.0, 0, 0.0, 0, 1.0, 0.0]),
            _make_meta(1, (0.0, 32.0, 64.0, 96.0), [1.0, 0, 0.0, 0, 1.0, 32.0]),
        ]

        shape, _transform = _reconstruct_scene_from_patches(meta, (64, 64), delta=8)

        # Patch 0 at geo ymin: array-top (row 0) is scene boundary -> top=0
        assert meta[0]['edge_deltas'] == (0, 8, 0, 0)
        # Patch 1 at geo ymax: array-bottom (last row) is scene boundary -> bottom=0
        assert meta[1]['edge_deltas'] == (8, 0, 0, 0)
        assert meta[0]['boundary_edges'] == (True, False, True, True)
        assert meta[1]['boundary_edges'] == (False, True, True, True)
        # Scene covers 96 geo units; with delta=8 cropped from interior edges only,
        # effective height = 96 - 0 = 96 (boundary edges preserved)
        assert shape == (96, 64)

    def test_empty_metadata_raises(self) -> None:
        """Test error on empty patch_metadata."""
        with pytest.raises(ValueError, match='patch_metadata is empty'):
            _reconstruct_scene_from_patches([], (64, 64), delta=0)


class TestGetEdgeDeltas:
    """Tests for _get_edge_deltas."""

    def test_corner_patch_two_edges_zero(self) -> None:
        """Corner patches get delta=0 on two boundary-touching edges."""
        scene_bounds = (0.0, 0.0, 100.0, 100.0)
        pixel_size = (1.0, 1.0)
        delta = 8

        top_left = (0.0, 36.0, 64.0, 100.0)
        result = _get_edge_deltas(top_left, scene_bounds, pixel_size, delta)
        assert result == (0, delta, 0, delta)

        bottom_right = (36.0, 0.0, 100.0, 64.0)
        result = _get_edge_deltas(bottom_right, scene_bounds, pixel_size, delta)
        assert result == (delta, 0, delta, 0)

    def test_interior_patch_all_edges_delta(self) -> None:
        """Interior patches get delta on all edges."""
        scene_bounds = (0.0, 0.0, 200.0, 200.0)
        interior_patch = (68.0, 68.0, 132.0, 132.0)
        pixel_size = (1.0, 1.0)
        delta = 8

        result = _get_edge_deltas(interior_patch, scene_bounds, pixel_size, delta)
        assert result == (delta, delta, delta, delta)

    def test_boundary_edge_single_edge_zero(self) -> None:
        """Patches on one boundary get delta=0 on that edge only."""
        scene_bounds = (0.0, 0.0, 200.0, 200.0)
        pixel_size = (1.0, 1.0)
        delta = 8

        top_edge = (68.0, 136.0, 132.0, 200.0)
        result = _get_edge_deltas(top_edge, scene_bounds, pixel_size, delta)
        assert result == (0, delta, delta, delta)

        left_edge = (0.0, 68.0, 64.0, 132.0)
        result = _get_edge_deltas(left_edge, scene_bounds, pixel_size, delta)
        assert result == (delta, delta, 0, delta)

    def test_tolerance_within_threshold(self) -> None:
        """Patches within half a pixel of the boundary count as boundary-touching."""
        scene_bounds = (0.0, 0.0, 100.0, 100.0)
        pixel_size = (1.0, 1.0)
        delta = 8

        almost_top = (32.0, 36.0, 68.0, 99.8)
        result = _get_edge_deltas(almost_top, scene_bounds, pixel_size, delta)
        assert result[0] == 0

    def test_tolerance_outside_threshold(self) -> None:
        """Patches a full pixel from the boundary don't count as boundary."""
        scene_bounds = (0.0, 0.0, 100.0, 100.0)
        pixel_size = (1.0, 1.0)
        delta = 8

        not_at_top = (32.0, 36.0, 68.0, 99.0)
        result = _get_edge_deltas(not_at_top, scene_bounds, pixel_size, delta)
        assert result[0] == delta

    def test_single_patch_all_edges_zero(self) -> None:
        """Single patch covering entire scene has delta=0 on all edges."""
        scene_bounds = (0.0, 0.0, 64.0, 64.0)
        patch = (0.0, 0.0, 64.0, 64.0)
        pixel_size = (1.0, 1.0)
        delta = 8

        result = _get_edge_deltas(patch, scene_bounds, pixel_size, delta)
        assert result == (0, 0, 0, 0)

    def test_anisotropic_pixels_use_per_axis_tolerance(self) -> None:
        """The y tolerance follows the y resolution, not the x resolution."""
        scene_bounds = (0.0, 0.0, 600.0, 100.0)
        delta = 8

        # Five y-pixels above the scene bottom: interior with yres=10, even though
        # the coarse xres=60 would have hidden it inside a 1.5 * xres tolerance.
        patch = (0.0, 50.0, 600.0, 100.0)
        result = _get_edge_deltas(patch, scene_bounds, (60.0, 10.0), delta)
        assert result == (0, delta, 0, 0)

        # A third of a y-pixel short of the bottom with yres=60: still boundary.
        patch = (0.0, 20.0, 600.0, 100.0)
        result = _get_edge_deltas(patch, scene_bounds, (10.0, 60.0), delta)
        assert result == (0, 0, 0, 0)

    def test_south_up_corner_patch(self) -> None:
        """South-up rasters swap top/bottom in array order."""
        scene_bounds = (0.0, 0.0, 100.0, 100.0)
        pixel_size = (1.0, 1.0)
        delta = 8

        # Geo bottom-left corner = array top-left for south-up
        bottom_left = (0.0, 0.0, 64.0, 64.0)
        result = _get_edge_deltas(
            bottom_left, scene_bounds, pixel_size, delta, north_up=False
        )
        # Array top (row 0) touches geo ymin -> top=0
        # Array bottom (last row) not at geo ymax -> bottom=delta
        assert result == (0, delta, 0, delta)

        # Geo top-right corner = array bottom-right for south-up
        top_right = (36.0, 36.0, 100.0, 100.0)
        result = _get_edge_deltas(
            top_right, scene_bounds, pixel_size, delta, north_up=False
        )
        # Array top not at geo ymin -> top=delta
        # Array bottom touches geo ymax -> bottom=0
        assert result == (delta, 0, delta, 0)


class TestGetBlendMask:
    """Tests for get_blend_mask."""

    def test_no_overlap(self) -> None:
        """Test blend mask with no overlap."""
        mask = get_blend_mask(64, overlap=0, delta=0, method='cosine')

        assert mask.shape == (64, 64)
        np.testing.assert_allclose(mask, np.ones((64, 64)), rtol=1e-5)

    def test_with_overlap_cosine(self) -> None:
        """Test cosine blend mask."""
        mask = get_blend_mask(64, overlap=8, delta=0, method='cosine')

        assert mask.shape == (64, 64)
        assert mask[0, 32] < mask[32, 32]
        assert mask[32, 32] == pytest.approx(1.0)

    def test_with_overlap_linear(self) -> None:
        """Test linear blend mask."""
        mask = get_blend_mask(64, overlap=8, delta=0, method='linear')

        assert mask.shape == (64, 64)
        assert mask[0, 32] < mask[32, 32]
        assert mask[32, 32] == pytest.approx(1.0)

    @pytest.mark.parametrize('method', ['cosine', 'linear'])
    @pytest.mark.parametrize('overlap', [1, 2, 8])
    def test_opposite_ramps_sum_to_one(self, method: str, overlap: int) -> None:
        """Ramps of neighbouring patches form a partition of unity, never zero."""
        mask = get_blend_mask(32, overlap=overlap, delta=0, method=method)  # ty: ignore[invalid-argument-type]

        left = mask[16, :overlap]
        right = mask[16, -overlap:]
        # A right edge of one patch lands pixel-for-pixel on the left edge of the
        # next, so the two ramps are summed without reversing.
        np.testing.assert_allclose(left + right, 1.0, rtol=1e-5)
        assert np.all(mask > 0)

    def test_invalid_method_raises(self) -> None:
        """Test invalid blend method raises error."""
        with pytest.raises(ValueError, match='Unknown blend method'):
            get_blend_mask(64, overlap=8, delta=0, method='invalid')  # ty: ignore[invalid-argument-type]

    def test_delta_crops_entire_patch_raises(self) -> None:
        """Test delta that crops away the entire patch raises error."""
        with pytest.raises(ValueError, match='delta crops away the entire patch'):
            get_blend_mask(64, overlap=0, delta=32, method='cosine')

    @pytest.mark.parametrize('overlap', [25, 50])
    def test_overlap_exceeds_half_cropped_patch_raises(self, overlap: int) -> None:
        """Overlap above half the cropped patch (48 px here) raises.

        Between half and full size the two ramps would overwrite each other;
        above full size they would not even fit.
        """
        with pytest.raises(ValueError, match='overlap exceeds half'):
            get_blend_mask(64, overlap=overlap, delta=8, method='cosine')

    def test_overlap_at_half_cropped_patch_is_valid(self) -> None:
        """Exactly half the cropped patch (50% overlap) is the largest valid ramp."""
        mask = get_blend_mask(64, overlap=24, delta=8, method='cosine')
        assert mask.shape == (48, 48)
        # Ramps meet in the middle, so the peak is symmetric and just below 1
        assert mask[23, 23] == pytest.approx(mask[24, 24])
        assert 0.99 < mask[24, 24] < 1.0

    def test_edge_deltas_suppresses_ramp(self) -> None:
        """Edges with delta=0 in edge_deltas get no blend ramp (weight=1)."""
        edge_deltas = (0, 8, 0, 8)
        mask = get_blend_mask(
            64,
            overlap=8,
            delta=8,
            method='cosine',
            edge_deltas=edge_deltas,
            boundary_edges=(True, False, True, False),
        )

        assert mask.shape == (56, 56)
        assert mask[0, 28] == pytest.approx(1.0, rel=1e-4)
        assert mask[-1, 28] < 1.0

    def test_asymmetric_edge_deltas(self) -> None:
        """Different deltas on each edge produce asymmetric mask."""
        edge_deltas = (4, 8, 2, 6)
        mask = get_blend_mask(
            64, overlap=8, delta=8, method='cosine', edge_deltas=edge_deltas
        )

        expected_h = 64 - 4 - 8
        expected_w = 64 - 2 - 6
        assert mask.shape == (expected_h, expected_w)

    def test_delta_zero_keeps_interior_ramps(self) -> None:
        """delta=0 still ramps interior edges; only boundary edges are flat."""
        mask = get_blend_mask(
            64,
            overlap=8,
            delta=0,
            method='cosine',
            edge_deltas=(0, 0, 0, 0),
            boundary_edges=(True, False, False, False),
        )

        assert mask.shape == (64, 64)
        assert mask[0, 32] == pytest.approx(1.0)
        assert mask[-1, 32] < 1.0
        assert mask[32, 0] < 1.0
        assert mask[32, -1] < 1.0

    def test_edge_deltas_none_uses_uniform_delta(self) -> None:
        """When edge_deltas is None, uniform delta is used."""
        mask_uniform = get_blend_mask(64, overlap=8, delta=8, method='cosine')
        mask_explicit = get_blend_mask(
            64, overlap=8, delta=8, method='cosine', edge_deltas=(8, 8, 8, 8)
        )

        np.testing.assert_allclose(mask_uniform, mask_explicit)


class TestGridIndexing:
    """Tests for grid-based spatial indexing utilities."""

    def test_build_and_query(self) -> None:
        """Test building and querying grid index."""
        meta = [
            _make_meta(0, bbox=(0, 0, 64, 64)),
            _make_meta(1, bbox=(200, 200, 264, 264)),
        ]
        grid_size = 128
        grid = _build_grid_index(meta, grid_size)

        results = _query_grid_index(
            grid,
            meta,
            chunk_y=0,
            chunk_x=0,
            chunk_h=64,
            chunk_w=64,
            grid_size=grid_size,
        )

        assert len(results) == 1
        assert results[0]['patch_id'] == 0

    def test_query_multiple_patches(self) -> None:
        """Test querying returns multiple overlapping patches."""
        meta = [
            _make_meta(0, bbox=(0, 0, 64, 64)),
            _make_meta(1, bbox=(32, 0, 96, 64)),
            _make_meta(2, bbox=(200, 200, 264, 264)),
        ]
        grid_size = 128
        grid = _build_grid_index(meta, grid_size)

        results = _query_grid_index(
            grid,
            meta,
            chunk_y=0,
            chunk_x=0,
            chunk_h=100,
            chunk_w=100,
            grid_size=grid_size,
        )

        patch_ids = {r['patch_id'] for r in results}
        assert 0 in patch_ids
        assert 1 in patch_ids
        assert 2 not in patch_ids

    def test_query_returns_non_overlapping_in_same_cell(self) -> None:
        """Test grid query returns patches in same cell even if no pixel overlap.

        This tests the scenario where patches in the same grid cell are returned
        but don't actually overlap with the chunk at pixel level.
        """
        meta = [
            _make_meta(0, bbox=(0, 0, 32, 32)),
            _make_meta(1, bbox=(96, 96, 128, 128)),
        ]
        grid_size = 128
        grid = _build_grid_index(meta, grid_size)

        results = _query_grid_index(
            grid,
            meta,
            chunk_y=0,
            chunk_x=0,
            chunk_h=64,
            chunk_w=64,
            grid_size=grid_size,
        )

        patch_ids = {r['patch_id'] for r in results}
        assert 0 in patch_ids
        assert 1 in patch_ids


class TestExtentMismatch:
    """Tests for output extent accuracy."""

    def test_extent_matches_input_bounds_3x3_grid(self) -> None:
        """Verify output extent matches original input extent exactly.

        Creates a 3x3 grid of overlapping patches covering exactly 160x160 pixels
        at 1m resolution. The output shape and transform should match exactly.
        """
        patch_size = 64
        stride = 48

        origin_x, origin_y = 1000.0, 2000.0
        res = 1.0

        meta: list[PatchMetadata] = []
        for row in range(3):
            for col in range(3):
                geo_xmin = origin_x + col * stride * res
                geo_ymax = origin_y - row * stride * res
                geo_xmax = geo_xmin + patch_size * res
                geo_ymin = geo_ymax - patch_size * res

                meta.append(
                    {
                        'patch_id': row * 3 + col,
                        'file': Path('unused'),
                        'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                        'transform': [res, 0, geo_xmin, 0, -res, geo_ymax],
                    }
                )

        shape, transform = _reconstruct_scene_from_patches(
            meta, (patch_size, patch_size)
        )

        expected_width = 2 * stride + patch_size
        expected_height = 2 * stride + patch_size
        assert shape == (expected_height, expected_width)
        assert transform.c == pytest.approx(origin_x)
        assert transform.f == pytest.approx(origin_y)


class TestBlackBorder:
    """Tests for edge blending artifacts."""

    def test_no_black_border_at_edges(self, tmp_path: Path) -> None:
        """Verify edge pixels have valid values after blending.

        Creates a 2x2 grid of patches where all patches predict class 1.
        After blending, all pixels including edges/corners should be class 1.
        This verifies that edge patches are correctly processed without artifacts.
        """
        patch_size = 64
        overlap = 16
        delta = 8
        num_classes = 3
        expected_class = 1
        stride = patch_size - 2 * overlap

        origin_x, origin_y = 0.0, 128.0
        res = 1.0

        patch_metadata: list[PatchMetadata] = []
        for row in range(2):
            for col in range(2):
                patch_id = row * 2 + col
                geo_xmin = origin_x + col * stride * res
                geo_ymax = origin_y - row * stride * res
                geo_xmax = geo_xmin + patch_size * res
                geo_ymin = geo_ymax - patch_size * res

                logits = torch.zeros(num_classes, patch_size, patch_size)
                logits[expected_class] = 1.0

                patch_file = tmp_path / f'patch_{patch_id:06d}.tif'
                transform = [res, 0, geo_xmin, 0, -res, geo_ymax]
                _save_test_patch(patch_file, logits, transform)

                patch_metadata.append(
                    {
                        'patch_id': patch_id,
                        'file': patch_file,
                        'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                        'transform': transform,
                    }
                )

        output_path = tmp_path / 'output.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=overlap,
            delta=delta,
            blend_method='cosine',
            output_path=output_path,
            chunk_size=256,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)
            # crs was not given, so it is taken from the patches
            assert CRS.from_user_input(src.crs) == CRS.from_epsg(32631)

        assert data[0, 0] == expected_class, (
            f'Top-left corner: {data[0, 0]} != {expected_class}'
        )
        assert data[0, -1] == expected_class, (
            f'Top-right corner: {data[0, -1]} != {expected_class}'
        )
        assert data[-1, 0] == expected_class, (
            f'Bottom-left corner: {data[-1, 0]} != {expected_class}'
        )
        assert data[-1, -1] == expected_class, (
            f'Bottom-right corner: {data[-1, -1]} != {expected_class}'
        )

        assert np.all(data[0, :] == expected_class), 'Top edge has wrong values'
        assert np.all(data[-1, :] == expected_class), 'Bottom edge has wrong values'
        assert np.all(data[:, 0] == expected_class), 'Left edge has wrong values'
        assert np.all(data[:, -1] == expected_class), 'Right edge has wrong values'

        assert np.all(data == expected_class), (
            f'Not all pixels are class {expected_class}'
        )


class TestNonOverlappingPatches:
    """Tests for handling patches that don't overlap with chunks."""

    def test_weighted_merge_skips_non_overlapping_patches(self, tmp_path: Path) -> None:
        """Test weighted_merge skips patches that don't overlap with current chunk.

        Creates two patches far apart but in the same grid cell (grid_size=1024).
        When processing chunks, the grid query returns both patches, but the
        overlap calculation should skip patches that don't actually overlap.
        """
        patch_size = 64
        num_classes = 2
        res = 1.0
        origin_y = 564.0

        patch_metadata: list[PatchMetadata] = []
        positions = [(0, 0), (500, 500)]
        for patch_id, (col_offset, row_offset) in enumerate(positions):
            geo_xmin = float(col_offset)
            geo_ymax = origin_y - row_offset
            geo_xmax = geo_xmin + patch_size * res
            geo_ymin = geo_ymax - patch_size * res

            logits = torch.zeros(num_classes, patch_size, patch_size)
            logits[1] = 1.0

            patch_file = tmp_path / f'patch_{patch_id:06d}.tif'
            transform = [res, 0, geo_xmin, 0, -res, geo_ymax]
            _save_test_patch(patch_file, logits, transform)

            patch_metadata.append(
                {
                    'patch_id': patch_id,
                    'file': patch_file,
                    'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                    'transform': transform,
                }
            )

        output_path = tmp_path / 'output_sparse.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=0,
            delta=0,
            blend_method='cosine',
            crs=CRS.from_epsg(32631),
            output_path=output_path,
            chunk_size=128,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)
            assert src.nodata == 255
            assert data[0, 0] == 1
            assert data[500, 500] == 1

        # Everything between the two patches is uncovered and gets nodata
        assert np.all(data[:64, 64:500] == 255)
        assert np.all(data[64:500, :] == 255)
        # A nodata of None keeps the old behaviour: class 0, no nodata tag
        output_path = tmp_path / 'output_sparse_no_nodata.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=0,
            delta=0,
            output_path=output_path,
            chunk_size=128,
            nodata=None,
        )
        with rasterio.open(output_path) as src:
            assert src.nodata is None
            assert np.all(src.read(1)[64:500, :] == 0)


class TestDatasetBoundsMode:
    """Tests for weighted_merge with dataset_bounds parameter."""

    def test_weighted_merge_with_dataset_bounds(self, tmp_path: Path) -> None:
        """Test weighted_merge uses dataset_bounds for output extent."""
        patch_size = 64
        overlap = 16
        delta = 8
        num_classes = 3
        expected_class = 1

        dataset_bounds = (0.0, 0.0, 128.0, 128.0)
        dataset_res = (1.0, 1.0)

        patch_metadata: list[PatchMetadata] = []
        for row in range(2):
            for col in range(2):
                patch_id = row * 2 + col
                geo_xmin = col * 32.0
                geo_ymax = 128.0 - row * 32.0
                geo_xmax = geo_xmin + patch_size
                geo_ymin = geo_ymax - patch_size

                logits = torch.zeros(num_classes, patch_size, patch_size)
                logits[expected_class] = 1.0

                patch_file = tmp_path / f'patch_{patch_id:06d}.tif'
                transform = [1.0, 0, geo_xmin, 0, -1.0, geo_ymax]
                _save_test_patch(patch_file, logits, transform)

                patch_metadata.append(
                    {
                        'patch_id': patch_id,
                        'file': patch_file,
                        'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                        'transform': transform,
                    }
                )

        output_path = tmp_path / 'output_with_bounds.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=overlap,
            delta=delta,
            blend_method='cosine',
            crs=CRS.from_epsg(32631),
            output_path=output_path,
            chunk_size=256,
            dataset_bounds=dataset_bounds,
            dataset_res=dataset_res,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)
            assert src.width == 128
            assert src.height == 128
            assert data.shape == (128, 128)

    def test_dataset_res_mismatch_raises(self, tmp_path: Path) -> None:
        """dataset_res that differs from the patch resolution raises."""
        patch_size = 64
        transform = [1.0, 0, 0.0, 0, -1.0, 64.0]
        logits = torch.zeros(2, patch_size, patch_size)
        patch_file = tmp_path / 'patch_res.tif'
        _save_test_patch(patch_file, logits, transform)

        patch_metadata: list[PatchMetadata] = [
            {
                'patch_id': 0,
                'file': patch_file,
                'geo_bbox': (0.0, 0.0, 64.0, 64.0),
                'transform': transform,
            }
        ]

        with pytest.raises(ValueError, match='does not match the patch resolution'):
            weighted_merge(
                patch_metadata=patch_metadata,
                num_classes=2,
                overlap=0,
                delta=0,
                output_path=tmp_path / 'output.tif',
                dataset_bounds=(0.0, 0.0, 64.0, 64.0),
                dataset_res=(2.0, 2.0),
            )

    def test_dataset_bounds_edge_coverage(self, tmp_path: Path) -> None:
        """Using dataset_bounds still produces full edge coverage with delta > 0."""
        patch_size = 64
        overlap = 16
        delta = 8
        num_classes = 2
        expected_class = 1

        dataset_bounds = (0.0, 0.0, 96.0, 96.0)
        dataset_res = (1.0, 1.0)

        patch_metadata: list[PatchMetadata] = []
        for row in range(2):
            for col in range(2):
                patch_id = row * 2 + col
                geo_xmin = col * 32.0
                geo_ymax = 96.0 - row * 32.0
                geo_xmax = geo_xmin + patch_size
                geo_ymin = geo_ymax - patch_size

                logits = torch.zeros(num_classes, patch_size, patch_size)
                logits[expected_class] = 1.0

                patch_file = tmp_path / f'patch_bounds_{patch_id:06d}.tif'
                transform = [1.0, 0, geo_xmin, 0, -1.0, geo_ymax]
                _save_test_patch(patch_file, logits, transform)

                patch_metadata.append(
                    {
                        'patch_id': patch_id,
                        'file': patch_file,
                        'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                        'transform': transform,
                    }
                )

        output_path = tmp_path / 'output_bounds_edge.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=overlap,
            delta=delta,
            blend_method='cosine',
            crs=CRS.from_epsg(32631),
            output_path=output_path,
            chunk_size=256,
            dataset_bounds=dataset_bounds,
            dataset_res=dataset_res,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)

        assert np.all(data[0, :] == expected_class), 'Top edge has wrong values'
        assert np.all(data[-1, :] == expected_class), 'Bottom edge has wrong values'
        assert np.all(data[:, 0] == expected_class), 'Left edge has wrong values'
        assert np.all(data[:, -1] == expected_class), 'Right edge has wrong values'


class TestWeightedMergeValidation:
    """Tests for weighted_merge argument validation."""

    def test_empty_metadata_raises(self, tmp_path: Path) -> None:
        """Test weighted_merge with no patches raises before touching the disk."""
        with pytest.raises(ValueError, match='patch_metadata is empty'):
            weighted_merge(
                patch_metadata=[],
                num_classes=2,
                overlap=0,
                delta=0,
                output_path=tmp_path / 'output.tif',
            )

    def test_nodata_collides_with_class_raises(self, tmp_path: Path) -> None:
        """A nodata value inside the class range raises."""
        with pytest.raises(ValueError, match='outside the class range'):
            weighted_merge(
                patch_metadata=[_make_meta(0)],
                num_classes=3,
                overlap=0,
                delta=0,
                output_path=tmp_path / 'output.tif',
                nodata=2,
            )

    def test_too_many_classes_raises(self, tmp_path: Path) -> None:
        """More classes than uint8 can hold raise."""
        with pytest.raises(ValueError, match='does not fit the uint8 output'):
            weighted_merge(
                patch_metadata=[_make_meta(0)],
                num_classes=257,
                overlap=0,
                delta=0,
                output_path=tmp_path / 'output.tif',
            )

    def test_delta_exceeds_half_overlap_raises(self, tmp_path: Path) -> None:
        """Cropping more than half the overlap would leave gaps between patches."""
        with pytest.raises(ValueError, match='must not exceed half of overlap'):
            weighted_merge(
                patch_metadata=[_make_meta(0)],
                num_classes=2,
                overlap=8,
                delta=5,
                output_path=tmp_path / 'output.tif',
            )

    def test_patch_without_crs_raises(self, tmp_path: Path) -> None:
        """A patch file with no CRS raises a clear error."""
        patch_size = 64
        transform = [1.0, 0, 0.0, 0, -1.0, 64.0]
        one_hot = np.zeros((2, patch_size, patch_size), dtype=np.uint8)
        patch_file = tmp_path / 'no_crs.tif'
        with rasterio.open(
            patch_file,
            'w',
            driver='GTiff',
            height=patch_size,
            width=patch_size,
            count=2,
            dtype='uint8',
            transform=Affine(*transform),
        ) as dst:
            dst.write(one_hot)

        with pytest.raises(ValueError, match='has no CRS'):
            weighted_merge(
                patch_metadata=[
                    _make_meta(0, (0.0, 0.0, 64.0, 64.0), transform)
                    | {'file': patch_file}
                ],
                num_classes=2,
                overlap=0,
                delta=0,
                output_path=tmp_path / 'output.tif',
            )

    def test_mixed_patch_sizes_raise(self, tmp_path: Path) -> None:
        """Patches of different sizes raise instead of mis-sizing the scene."""
        patch_metadata: list[PatchMetadata] = []
        for patch_id, size in enumerate([16, 8]):
            geo_xmin = float(patch_id * 16)
            logits = torch.zeros(2, size, size)
            patch_file = tmp_path / f'mixed_{patch_id}.tif'
            transform = [1.0, 0, geo_xmin, 0, -1.0, 16.0]
            _save_test_patch(patch_file, logits, transform)
            patch_metadata.append(
                {
                    'patch_id': patch_id,
                    'file': patch_file,
                    'geo_bbox': (geo_xmin, 16.0 - size, geo_xmin + size, 16.0),
                    'transform': transform,
                }
            )

        with pytest.raises(ValueError, match='all patches must share one size'):
            weighted_merge(
                patch_metadata=patch_metadata,
                num_classes=2,
                overlap=0,
                delta=0,
                output_path=tmp_path / 'output.tif',
            )

    def test_patch_crs_mismatch_raises(self, tmp_path: Path) -> None:
        """Test a patch outside the output CRS raises instead of merging silently."""
        patch_size = 64
        transform = [1.0, 0, 0.0, 0, -1.0, 64.0]
        logits = torch.zeros(2, patch_size, patch_size)
        patch_file = tmp_path / 'patch_000000.tif'
        _save_test_patch(patch_file, logits, transform, crs='EPSG:32632')

        patch_metadata: list[PatchMetadata] = [
            {
                'patch_id': 0,
                'file': patch_file,
                'geo_bbox': (0.0, 0.0, 64.0, 64.0),
                'transform': transform,
            }
        ]

        with pytest.raises(ValueError, match='is in CRS'):
            weighted_merge(
                patch_metadata=patch_metadata,
                num_classes=2,
                overlap=0,
                delta=0,
                crs=CRS.from_epsg(32631),
                output_path=tmp_path / 'output.tif',
            )


class TestSinglePatchScene:
    """Tests for single-patch scenes."""

    def test_single_patch_no_black_border(self, tmp_path: Path) -> None:
        """Single patch covering entire scene has no black borders with delta > 0."""
        patch_size = 64
        delta = 8
        num_classes = 2
        expected_class = 1
        res = 1.0

        geo_xmin = 0.0
        geo_ymax = 64.0
        geo_xmax = 64.0
        geo_ymin = 0.0

        logits = torch.zeros(num_classes, patch_size, patch_size)
        logits[expected_class] = 1.0

        patch_file = tmp_path / 'single_patch.tif'
        transform = [res, 0, geo_xmin, 0, -res, geo_ymax]
        _save_test_patch(patch_file, logits, transform)

        patch_metadata: list[PatchMetadata] = [
            {
                'patch_id': 0,
                'file': patch_file,
                'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                'transform': transform,
            }
        ]

        output_path = tmp_path / 'single_patch_output.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=16,
            delta=delta,
            blend_method='cosine',
            crs=CRS.from_epsg(32631),
            output_path=output_path,
            chunk_size=256,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)

        assert data.shape == (64, 64)
        assert np.all(data == expected_class), 'Single patch has wrong values'
        assert data[0, 0] == expected_class, 'Top-left corner wrong'
        assert data[0, -1] == expected_class, 'Top-right corner wrong'
        assert data[-1, 0] == expected_class, 'Bottom-left corner wrong'
        assert data[-1, -1] == expected_class, 'Bottom-right corner wrong'


class TestSouthUpRasters:
    """Tests for south-up raster support (positive y-resolution)."""

    def test_weighted_merge_south_up(self, tmp_path: Path) -> None:
        """South-up rasters merge correctly with weighted blending."""
        patch_size = 64
        overlap = 16
        delta = 8
        num_classes = 2
        expected_class = 1
        stride = patch_size - 2 * overlap
        res = 1.0

        patch_metadata: list[PatchMetadata] = []
        for row in range(2):
            for col in range(2):
                patch_id = row * 2 + col
                geo_xmin = col * stride * res
                # South-up: origin at ymin, y increases upward
                geo_ymin = row * stride * res
                geo_xmax = geo_xmin + patch_size * res
                geo_ymax = geo_ymin + patch_size * res

                logits = torch.zeros(num_classes, patch_size, patch_size)
                logits[expected_class] = 1.0

                patch_file = tmp_path / f'south_up_{patch_id:06d}.tif'
                transform = [res, 0, geo_xmin, 0, res, geo_ymin]
                _save_test_patch(patch_file, logits, transform)

                patch_metadata.append(
                    {
                        'patch_id': patch_id,
                        'file': patch_file,
                        'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                        'transform': transform,
                    }
                )

        output_path = tmp_path / 'south_up_output.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=overlap,
            delta=delta,
            blend_method='cosine',
            crs=CRS.from_epsg(32631),
            output_path=output_path,
            chunk_size=256,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)
            # Positive y-resolution in the output transform
            assert src.transform.e > 0

        assert np.all(data == expected_class), 'South-up merge has wrong values'
        assert data[0, 0] == expected_class, 'Top-left corner wrong'
        assert data[-1, -1] == expected_class, 'Bottom-right corner wrong'

    def test_weighted_merge_south_up_varying_classes(self, tmp_path: Path) -> None:
        """South-up merge places patches at correct pixel positions.

        Each patch predicts a different class so positional errors (e.g. row
        mirroring) would put the wrong class in the wrong quadrant.
        """
        patch_size = 64
        num_classes = 5
        res = 1.0

        # 2x2 grid, no overlap, no delta: simple tiling
        patch_metadata: list[PatchMetadata] = []
        expected_classes = [1, 2, 3, 4]
        for row in range(2):
            for col in range(2):
                patch_id = row * 2 + col
                geo_xmin = col * patch_size * res
                geo_ymin = row * patch_size * res
                geo_xmax = geo_xmin + patch_size * res
                geo_ymax = geo_ymin + patch_size * res

                logits = torch.zeros(num_classes, patch_size, patch_size)
                logits[expected_classes[patch_id]] = 1.0

                patch_file = tmp_path / f'south_up_vary_{patch_id:06d}.tif'
                transform = [res, 0, geo_xmin, 0, res, geo_ymin]
                _save_test_patch(patch_file, logits, transform)

                patch_metadata.append(
                    {
                        'patch_id': patch_id,
                        'file': patch_file,
                        'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                        'transform': transform,
                    }
                )

        output_path = tmp_path / 'south_up_vary_output.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=0,
            delta=0,
            blend_method='cosine',
            crs=CRS.from_epsg(32631),
            output_path=output_path,
            chunk_size=256,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)
            assert src.transform.e > 0

        # South-up: array row 0 = geo ymin (bottom row of patches)
        # Patch 0 (class 1): geo (0,0)-(64,64)   -> array rows 0:64,  cols 0:64
        # Patch 1 (class 2): geo (64,0)-(128,64)  -> array rows 0:64,  cols 64:128
        # Patch 2 (class 3): geo (0,64)-(64,128)  -> array rows 64:128, cols 0:64
        # Patch 3 (class 4): geo (64,64)-(128,128) -> array rows 64:128, cols 64:128
        assert data[0, 0] == 1, 'Bottom-left quadrant wrong'
        assert data[0, -1] == 2, 'Bottom-right quadrant wrong'
        assert data[-1, 0] == 3, 'Top-left quadrant wrong'
        assert data[-1, -1] == 4, 'Top-right quadrant wrong'

    def test_dataset_bounds_south_up(self, tmp_path: Path) -> None:
        """South-up rasters work correctly with dataset_bounds mode."""
        patch_size = 64
        num_classes = 2
        expected_class = 1
        res = 1.0

        dataset_bounds = (0.0, 0.0, 64.0, 64.0)
        dataset_res = (1.0, 1.0)

        geo_xmin, geo_ymin = 0.0, 0.0
        geo_xmax = geo_xmin + patch_size * res
        geo_ymax = geo_ymin + patch_size * res

        logits = torch.zeros(num_classes, patch_size, patch_size)
        logits[expected_class] = 1.0

        patch_file = tmp_path / 'south_up_bounds.tif'
        transform = [res, 0, geo_xmin, 0, res, geo_ymin]
        _save_test_patch(patch_file, logits, transform)

        patch_metadata: list[PatchMetadata] = [
            {
                'patch_id': 0,
                'file': patch_file,
                'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                'transform': transform,
            }
        ]

        output_path = tmp_path / 'south_up_bounds_output.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=0,
            delta=0,
            blend_method='cosine',
            crs=CRS.from_epsg(32631),
            output_path=output_path,
            chunk_size=256,
            dataset_bounds=dataset_bounds,
            dataset_res=dataset_res,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)
            assert src.width == 64
            assert src.height == 64
            assert src.transform.e > 0

        assert np.all(data == expected_class)


class TestBlendTransition:
    """Tests that exercise the blend ramps with conflicting predictions.

    Every other weighted_merge test uses an effective overlap of zero
    (overlap - 2 * delta) or identical classes, so the ramps never matter.
    """

    @pytest.mark.parametrize('delta', [0, 4])
    @pytest.mark.parametrize('blend_method', ['cosine', 'linear'])
    def test_two_patches_transition_at_overlap_midpoint(
        self, tmp_path: Path, delta: int, blend_method: Literal['cosine', 'linear']
    ) -> None:
        """Class boundary between two conflicting patches sits mid-overlap.

        With delta=0 nothing is cropped, so the ramps span the full overlap.
        """
        patch_size = 64
        overlap = 24
        num_classes = 3
        stride = patch_size - overlap
        classes = [1, 2]

        patch_metadata: list[PatchMetadata] = []
        for patch_id, expected_class in enumerate(classes):
            geo_xmin = float(patch_id * stride)
            geo_xmax = geo_xmin + patch_size
            logits = torch.zeros(num_classes, patch_size, patch_size)
            logits[expected_class] = 1.0

            patch_file = tmp_path / f'blend_{patch_id:06d}.tif'
            transform = [1.0, 0, geo_xmin, 0, -1.0, float(patch_size)]
            _save_test_patch(patch_file, logits, transform)
            patch_metadata.append(
                {
                    'patch_id': patch_id,
                    'file': patch_file,
                    'geo_bbox': (geo_xmin, 0.0, geo_xmax, float(patch_size)),
                    'transform': transform,
                }
            )

        output_path = tmp_path / 'blend_output.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=overlap,
            delta=delta,
            blend_method=blend_method,
            crs=CRS.from_epsg(32631),
            output_path=output_path,
            chunk_size=256,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)

        # After cropping delta from the interior edges, the patches overlap on
        # [stride + delta, patch_size - delta). Symmetric ramps hand each half
        # of that window to the nearer patch.
        midpoint = (stride + delta + patch_size - delta) // 2
        assert data.shape == (patch_size, stride + patch_size)
        assert np.all(data[:, :midpoint] == classes[0])
        assert np.all(data[:, midpoint:] == classes[1])

    @pytest.mark.parametrize('blend_method', ['cosine', 'linear'])
    def test_checkerboard(
        self, tmp_path: Path, blend_method: Literal['cosine', 'linear']
    ) -> None:
        """A 3x3 checkerboard of overlapping patches stitches into a checkerboard."""
        patch_size = 64
        overlap = 24
        delta = 4
        num_classes = 2
        grid = 3
        stride = patch_size - overlap
        scene_size = (grid - 1) * stride + patch_size

        patch_metadata: list[PatchMetadata] = []
        for row in range(grid):
            for col in range(grid):
                patch_id = row * grid + col
                geo_xmin = float(col * stride)
                geo_ymax = float(scene_size - row * stride)
                logits = torch.zeros(num_classes, patch_size, patch_size)
                logits[(row + col) % 2] = 1.0

                patch_file = tmp_path / f'checker_{patch_id:06d}.tif'
                transform = [1.0, 0, geo_xmin, 0, -1.0, geo_ymax]
                _save_test_patch(patch_file, logits, transform)
                patch_metadata.append(
                    {
                        'patch_id': patch_id,
                        'file': patch_file,
                        'geo_bbox': (
                            geo_xmin,
                            geo_ymax - patch_size,
                            geo_xmin + patch_size,
                            geo_ymax,
                        ),
                        'transform': transform,
                    }
                )

        output_path = tmp_path / 'checker_output.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=overlap,
            delta=delta,
            blend_method=blend_method,
            crs=CRS.from_epsg(32631),
            output_path=output_path,
            chunk_size=256,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)

        # Boundaries between neighbouring patches fall mid-overlap, so pixel
        # (r, c) belongs to the patch whose cropped window centre is nearest.
        boundaries = [i * stride + (stride + patch_size) // 2 for i in range(grid - 1)]
        owner = np.searchsorted(boundaries, np.arange(scene_size), side='right')
        expected = (owner[:, None] + owner[None, :]) % 2

        assert data.shape == (scene_size, scene_size)
        np.testing.assert_array_equal(data, expected)

    def test_linear_single_pixel_overlap_has_no_hole(self, tmp_path: Path) -> None:
        """An effective overlap of one pixel gives that pixel positive weight.

        Regression test: a linear ramp that started at exactly zero left the
        shared column with zero weight, which argmax turned into class 0.
        """
        patch_size = 8
        overlap = 1
        num_classes = 2
        stride = patch_size - overlap

        patch_metadata: list[PatchMetadata] = []
        for patch_id in range(3):
            geo_xmin = float(patch_id * stride)
            logits = torch.zeros(num_classes, patch_size, patch_size)
            logits[1] = 1.0
            patch_file = tmp_path / f'thin_{patch_id:06d}.tif'
            transform = [1.0, 0, geo_xmin, 0, -1.0, float(patch_size)]
            _save_test_patch(patch_file, logits, transform)
            patch_metadata.append(
                {
                    'patch_id': patch_id,
                    'file': patch_file,
                    'geo_bbox': (
                        geo_xmin,
                        0.0,
                        geo_xmin + patch_size,
                        float(patch_size),
                    ),
                    'transform': transform,
                }
            )

        output_path = tmp_path / 'thin_output.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=overlap,
            delta=0,
            blend_method='linear',
            output_path=output_path,
            chunk_size=256,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)

        assert np.all(data == 1)
