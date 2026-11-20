# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from shutil import copyfile

import pytest
from torch.utils.data import Subset

from torchgeo.datamodules import RarePlanesDataModule


@pytest.fixture
def datamodule(tmp_path: Path) -> RarePlanesDataModule:
    fixture = Path('tests/data/rareplanes/synthetic/train')
    for split, size in [('train', 5), ('test', 2)]:
        root = tmp_path / 'synthetic' / split
        (root / 'images').mkdir(parents=True)
        (root / 'xmls').mkdir()
        for index in range(size):
            copyfile(fixture / 'images/scene.png', root / f'images/{index}.png')
            annotation = root / f'xmls/{index}.xml'
            if index == 0:
                annotation.write_text('<image/>')
            else:
                copyfile(fixture / 'xmls/scene.xml', annotation)
    return RarePlanesDataModule(root=tmp_path, dataset_type='synthetic', batch_size=2)


def test_train_and_validation(datamodule: RarePlanesDataModule) -> None:
    datamodule.setup('fit')
    train = datamodule.train_dataset
    val = datamodule.val_dataset
    assert isinstance(train, Subset)
    assert isinstance(val, Subset)
    assert len(train) == 4
    assert len(val) == 1
    assert set(train.indices).isdisjoint(val.indices)
    next(iter(datamodule.train_dataloader()))

    datamodule.setup('validate')
    assert isinstance(datamodule.val_dataset, Subset)
    assert datamodule.val_dataset.indices == val.indices
    next(iter(datamodule.val_dataloader()))


@pytest.mark.parametrize('stage', ['test', 'predict'])
def test_held_out_batches(datamodule: RarePlanesDataModule, stage: str) -> None:
    datamodule.setup(stage)
    loader = getattr(datamodule, f'{stage}_dataloader')()
    batch = next(iter(loader))
    assert batch['image'].shape == (2, 3, 8, 8)
    assert [boxes.shape for boxes in batch['bbox_xyxy']] == [(0, 4), (1, 4)]
    assert [labels.tolist() for labels in batch['label']] == [[], [1]]
