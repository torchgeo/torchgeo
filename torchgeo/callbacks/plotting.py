# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Plot datasets via a logger."""

import kornia.augmentation as K
from lightning.pytorch import Callback, LightningModule, Trainer

from ..datasets import unbind_samples
from ..datasets.utils import Sample


class TensorBoardCallback(Callback):
    """Callback to log dataset plots to TensorBoard.

    .. versionadded:: 0.11
    """

    def on_validation_batch_end(  # ty: ignore[invalid-method-override]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sample,
        batch: Sample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch ends.

        Args:
            trainer: The trainer.
            pl_module: The task.
            outputs: The output of :meth:`LightningModule.validation_step`.
            batch: The input of :meth:`LightningModule.validation_step`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
        """
        if batch_idx > 10:
            return

        dm = trainer.datamodule
        aug = K.AugmentationSequential(K.Denormalize(dm.mean, dm.std), data_keys=None)
        outputs = aug(outputs)
        sample = unbind_samples(outputs)[0]
        sample = {k: v.to(device='cpu') for k, v in sample.items()}
        fig = dm.plot(sample)
        trainer.logger.experiment.add_figure(f'image/{batch_idx}', fig)
