Self-Supervised Learning Tasks
==============================

Self-supervised learning (SSL) trains encoders without labels. To assess the learned features, we use them to classify labeled images. Training loss alone is insufficient: each SSL method optimizes a different objective, and a decreasing loss can accompany representation collapse, where the encoder produces nearly identical features for different images.

This page describes TorchGeo's SSL tasks and a EuroSAT benchmark for comparing them. It includes results and configurations for the existing tasks, along with evaluation guidance for contributors adding a new task.

.. contents::
   :local:
   :depth: 1

Available tasks
---------------

The tasks below subclass :class:`~torchgeo.tasks.BaseTask`. The ``model`` argument selects a `timm <https://huggingface.co/docs/timm/reference/models>`__ encoder, subject to the restrictions noted below, and ``in_channels`` sets the number of input bands for multispectral imagery.

.. list-table::
   :header-rows: 1
   :widths: 14 46 40

   * - Task
     - Approach
     - Notes
   * - :class:`~torchgeo.tasks.SimCLR`
     - Uses NT-Xent to bring representations of two augmented views of the same image closer together and separate those of different images.
     - ``version`` selects SimCLR v1 or v2. Large batches provide more negative examples.
   * - :class:`~torchgeo.tasks.MoCo`
     - Uses a momentum-updated target encoder. Versions 1 and 2 store negative examples in a queue.
     - ``version`` selects MoCo v1, v2, or v3. v3 drops the queue and uses a predictor head.
   * - :class:`~torchgeo.tasks.BYOL`
     - Predicts the target network's projection of one view from the online network's projection of another, without negative examples.
     - Uses a fixed 224x224 input resolution internally and does not expose ``size`` or augmentation arguments.
   * - :class:`~torchgeo.tasks.MAE`
     - Reconstructs masked image patches.
     - Supports vision transformers only, since it operates on patch tokens.


Benchmarking
------------

We pretrain on EuroSAT's 13-band multispectral images without labels, then freeze the encoder and evaluate its features with a k-nearest-neighbor (kNN) classifier. This evaluation follows `Corley et al. 2024, "Revisiting pre-trained remote sensing model benchmarks: resizing and normalization matters" <https://arxiv.org/abs/2305.13456>`_. Using kNN avoids the optimizer, learning-rate, and regularization choices needed to train a linear probe.

Use the following settings when comparing runs with the results below.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Setting
     - Value
   * - Dataset
     - EuroSAT, all 13 Sentinel-2 bands, TorchGeo splits (16,200 train / 5,400 val / 5,400 test)
   * - Normalization
     - Per-band standardization using ``MEAN`` and ``STD`` from ``torchgeo.datamodules.eurosat``, which is what :class:`~torchgeo.datamodules.EuroSATDataModule` applies by default
   * - Input size
     - 224x224, produced by each task's ``RandomResizedCrop`` from the native 64x64 imagery
   * - Pretraining
     - 60 epochs, batch size 128, mixed precision, one GPU, seed 0
   * - Features
     - ``forward_head(forward_features(x), pre_logits=True)`` on the frozen encoder, evaluated on unaugmented images
   * - Classifier
     - ``sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)``, Euclidean
   * - Scaling
     - Evaluate kNN with and without ``StandardScaler`` and report the better accuracy, following the reference paper
   * - Selection
     - Select the learning rate using validation accuracy, then evaluate the selected model once on the test set

Resizing and normalization affect the comparison. The reference paper reports changes in model rankings when images are evaluated at their native 64x64 resolution instead of 224x224. In our runs, changing preprocessing produced a 0.43 range in kNN accuracy for the same ImageNet-pretrained ResNet-50.

Results
-------

The table reports EuroSAT test-set top-1 accuracy using kNN with five neighbors. For each SSL task and encoder, we selected the learning rate by validation accuracy and evaluated the selected model once on the test set. The baselines use image statistics or encoder features without SSL pretraining on EuroSAT.

.. list-table::
   :header-rows: 1
   :widths: 26 18 14 12 30

   * - Task
     - Encoder
     - Test acc
     - lr
     - Config
   * - Image statistics [#floor]_
     - none
     - 0.8937
     - --
     - --
   * - Random initialization
     - ResNet-50
     - 0.8622
     - --
     - --
   * - Random initialization
     - ViT-S/16
     - 0.8507
     - --
     - --
   * - Supervised ImageNet [#imagenet]_
     - ResNet-50
     - 0.8948
     - --
     - --
   * - Supervised ImageNet [#imagenet]_
     - ViT-S/16
     - 0.9178
     - --
     - --
   * - :class:`~torchgeo.tasks.MoCo` v3
     - ResNet-50
     - **0.9494**
     - 1e-2
     - `moco_resnet50.yaml <https://github.com/torchgeo/torchgeo/blob/main/configs/ssl_benchmarking/moco_resnet50.yaml>`__
   * - :class:`~torchgeo.tasks.MoCo` v3
     - ViT-S/16
     - 0.9413
     - 1e-4
     - `moco_vit_small.yaml <https://github.com/torchgeo/torchgeo/blob/main/configs/ssl_benchmarking/moco_vit_small.yaml>`__
   * - :class:`~torchgeo.tasks.SimCLR`
     - ResNet-50
     - 0.9356
     - 1.5
     - `simclr_resnet50.yaml <https://github.com/torchgeo/torchgeo/blob/main/configs/ssl_benchmarking/simclr_resnet50.yaml>`__
   * - :class:`~torchgeo.tasks.SimCLR`
     - ViT-S/16
     - 0.9296
     - 1.5
     - `simclr_vit_small.yaml <https://github.com/torchgeo/torchgeo/blob/main/configs/ssl_benchmarking/simclr_vit_small.yaml>`__
   * - :class:`~torchgeo.tasks.BYOL`
     - ResNet-50
     - 0.9294
     - 1e-3
     - `byol_resnet50.yaml <https://github.com/torchgeo/torchgeo/blob/main/configs/ssl_benchmarking/byol_resnet50.yaml>`__
   * - :class:`~torchgeo.tasks.BYOL`
     - ViT-S/16
     - 0.9241
     - 1e-4
     - `byol_vit_small.yaml <https://github.com/torchgeo/torchgeo/blob/main/configs/ssl_benchmarking/byol_vit_small.yaml>`__

The image-statistics baseline scores 0.8937, higher than either randomly initialized encoder. The ImageNet-pretrained ViT scores 0.9178. Compare against these baselines as well as random initialization when assessing whether SSL improves classification accuracy.

The selected learning rate depends on both the task and the encoder. The rates in this table span four orders of magnitude: MoCo v3 uses 1e-2 with ResNet-50 and 1e-4 with ViT-S/16. Sweep learning rates for each combination rather than assuming one setting will work for all of them.

Three of the sixteen MoCo and SimCLR runs completed all 60 epochs but produced checkpoints with entirely NaN features. Another run collapsed to a near-constant embedding while its loss stayed flat. ``torchgeo fit`` exited successfully in each case. The feature checks described below help identify these failures.

All results use a single seed. Differences of a few thousandths, such as 0.005, need repeated runs before they can support a reliable ranking.

.. rubric:: Footnotes

.. [#floor] The image-statistics baseline concatenates each image's per-band mean, standard deviation, minimum, and maximum into a 52-dimensional vector and uses the same kNN classifier. It does not use a neural network.

.. [#imagenet] These ImageNet-pretrained encoders use ``in_chans=13``. To adapt the pretrained input convolution, ``timm.models.adapt_input_conv`` tiles the RGB filters ``ceil(13 / 3) = 5`` times, truncates to 13 channels, and rescales by ``3 / 13`` to preserve the activation magnitude. The remaining layers retain their ImageNet weights.

Running the benchmark
---------------------

Pretrain using a TorchGeo task and :class:`~torchgeo.datamodules.EuroSATDataModule`, which applies the per-band standardization used in this benchmark. Save the following configuration as ``moco_resnet50.yaml`` to train MoCo v3 with a ResNet-50 encoder:

.. code-block:: yaml

   seed_everything: 0
   trainer:
     accelerator: gpu
     devices: 1
     max_epochs: 60
     precision: 16-mixed
     benchmark: true
   model:
     class_path: MoCo
     init_args:
       model: resnet50
       in_channels: 13
       version: 3
       lr: 0.01
       size: 224
   data:
     class_path: EuroSATDataModule
     init_args:
       batch_size: 128
       num_workers: 8
     dict_kwargs:
       root: data/eurosat

.. code-block:: console

   $ python -m torchgeo fit --config moco_resnet50.yaml

The complete configurations for the six SSL runs are in `configs/ssl_benchmarking <https://github.com/torchgeo/torchgeo/tree/main/configs/ssl_benchmarking>`__. They include each task's options, along with logging and checkpoint settings. Each learning rate was selected from a four-rate sweep using validation accuracy.

After training, load the checkpoint and evaluate the frozen encoder. ``torchgeo fit`` does not run the kNN evaluation. The :doc:`/tutorials/ssl_knn_eval` tutorial shows how to load the encoder, preprocess the images, and fit the classifier. The functions below extract features and evaluate them with kNN:

.. code-block:: python

   import torch
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.preprocessing import StandardScaler


   @torch.no_grad()
   def features(backbone, loader, device):
       backbone.eval().to(device)
       out, targets = [], []
       for batch in loader:
           x = batch['image'].to(device)
           z = backbone.forward_head(backbone.forward_features(x), pre_logits=True)
           out.append(z.flatten(1).cpu())
           targets.append(batch['label'])
       return torch.cat(out).numpy(), torch.cat(targets).numpy()


   def knn_score(train, train_y, test, test_y):
       scores = []
       for scaler in (None, StandardScaler()):
           a, b = (train, test) if scaler is None else (
               scaler.fit_transform(train), scaler.transform(test)
           )
           probe = KNeighborsClassifier(n_neighbors=5).fit(a, train_y)
           scores.append(probe.score(b, test_y))
       return max(scores)

Extract features without random augmentations, using the normalization and input size specified above. The encoder is available as ``backbone`` for :class:`~torchgeo.tasks.SimCLR` and :class:`~torchgeo.tasks.MoCo`, and as ``model.backbone.model`` for :class:`~torchgeo.tasks.BYOL`.

For each checkpoint, check that the features are finite and examine the L2-normalized embeddings. Near-zero standard deviation across images and a mean pairwise cosine similarity near one indicate representation collapse. Report these checks alongside accuracy.

Adding a new task
-----------------

When adding an SSL task, include:

#. ``torchgeo/tasks/foo.py``, subclassing :class:`~torchgeo.tasks.BaseTask`.
#. An entry in ``torchgeo/tasks/__init__.py``.
#. Tests in ``tests/tasks/test_foo.py``.
#. An API documentation entry in ``docs/api/tasks.rst``.

Evaluate the task using the benchmark settings above, trying at least three or four learning rates on the EuroSAT validation split. The current MoCo v3 and SimCLR defaults are scaled for a batch size of 4096: ``lr=9.6`` is ``0.6 x 4096 / 256`` for MoCo v3, and ``lr=4.8`` is ``0.3 x 4096 / 256`` for SimCLR. These defaults were too large for the batch size of 128 used here.

Select the learning rate using validation accuracy, then evaluate the selected model once on the test split. Include the configuration, kNN accuracy, and checks for collapse in the pull request so reviewers can compare the result with the baselines. If accuracy is below the image-statistics baseline, investigate the training and evaluation setup before drawing conclusions about the task.