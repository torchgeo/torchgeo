# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import re

import numpy as np
import pytest
import torch
from torch import Tensor

from torchgeo.datamodules.utils import collate_fn_detection, group_shuffle_split


def test_group_shuffle_split() -> None:
    train_indices = [0, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14]
    test_indices = [1, 3, 4, 12]
    np.random.seed(0)
    alphabet = np.array(list('abc'))
    groups = np.random.randint(0, 3, size=(15))
    groups = alphabet[groups]

    with pytest.raises(ValueError, match='You must specify `train_size`'):
        group_shuffle_split(groups, train_size=None, test_size=None)
    with pytest.raises(ValueError, match='`train_size` and `test_size` must sum to 1'):
        group_shuffle_split(groups, train_size=0.2, test_size=1.0)
    with pytest.raises(
        ValueError,
        match=re.escape('`train_size` and `test_size` must be in the range (0,1)'),
    ):
        group_shuffle_split(groups, train_size=-0.2, test_size=1.2)
    with pytest.raises(ValueError, match='3 groups were found, however the current'):
        group_shuffle_split(groups, train_size=None, test_size=0.999)

    test_cases = [(None, 0.2, 42), (0.8, None, 42)]

    for train_size, test_size, random_state in test_cases:
        train_indices1, test_indices1 = group_shuffle_split(
            groups,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )
        # Check that the results are the same as expected
        assert np.array_equal(train_indices, train_indices1)
        assert np.array_equal(test_indices, test_indices1)

        assert len(set(train_indices1) & set(test_indices1)) == 0
        assert len(set(groups[train_indices1])) == 2


def _positive_chip() -> dict[str, Tensor]:
    return {
        'image': torch.rand(3, 8, 8),
        'bbox_xyxy': torch.tensor([[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 2.0, 2.0]]),
        'label': torch.tensor([1, 2], dtype=torch.int32),
        'mask': torch.ones(2, 8, 8, dtype=torch.uint8),
    }


def _negative_chip() -> dict[str, Tensor]:
    """A chip with no objects is missing the keys entirely.

    This is what a UnionDataset returns when it skips a VectorDataset that does
    not intersect the chip.
    """
    return {'image': torch.rand(3, 8, 8)}


@pytest.mark.parametrize('negative_first', [False, True])
def test_collate_fn_detection_negative_chips(negative_first: bool) -> None:
    """A batch may mix chips that have objects with chips that have none."""
    batch = [_positive_chip(), _negative_chip()]
    if negative_first:
        batch.reverse()
    neg = 0 if negative_first else 1
    pos = 1 - neg

    collated = collate_fn_detection(batch)

    assert collated['image'].shape == (2, 3, 8, 8)

    # Empty entries keep the dtype and trailing shape of the real ones, so an
    # empty target still concatenates with the rest.
    assert collated['bbox_xyxy'][pos].shape == (2, 4)
    assert collated['bbox_xyxy'][neg].shape == (0, 4)

    assert collated['label'][pos].shape == (2,)
    assert collated['label'][neg].shape == (0,)
    assert collated['label'][neg].dtype == torch.int32

    assert collated['mask'][pos].shape == (2, 8, 8)
    assert collated['mask'][neg].shape == (0, 8, 8)
    assert collated['mask'][neg].dtype == torch.uint8


def test_collate_fn_detection_labels_default() -> None:
    """Without labels, every box gets label 1, including on negative chips."""
    batch = [
        {
            'image': torch.rand(3, 8, 8),
            'bbox_xyxy': torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
        },
        {'image': torch.rand(3, 8, 8)},
    ]

    collated = collate_fn_detection(batch)

    assert torch.equal(collated['label'][0], torch.tensor([1]))
    assert collated['label'][1].shape == (0,)
