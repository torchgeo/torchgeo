Self-Supervised Learning Tasks
==============================

Self-supervised learning (SSL) trains encoders without labels. Training loss alone is insufficient to determine whether an SSL training method is working as a decreasing loss can accompany representation collapse, where the encoder produces nearly identical features for different images. To properly evaluate SSL methods, we use the features from the encoder as input to a downstream task with labeled data.

This page describes TorchGeo's SSL tasks and a EuroSAT benchmark for comparing them as well as any newly proposed SSL task. It includes results and configurations for the existing tasks, along with evaluation guidance for contributors adding a new task.

.. contents::
   :local:
   :depth: 1

Available tasks
---------------

See the :ref:`SSL task overview <ssl-tasks>` for supported tasks, their approaches, and encoder restrictions.


Benchmarking
------------

We benchmark by pretraining on EuroSAT's 13-band multispectral images without labels, then freeze the encoder and evaluate its features with a k-nearest-neighbor (kNN) classifier. This evaluation follows `Corley et al. 2024, "Revisiting pre-trained remote sensing model benchmarks: resizing and normalization matters" <https://arxiv.org/abs/2305.13456>`_. Using kNN avoids the optimizer, learning-rate, and regularization choices needed to train a linear probe.

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
     - ``sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)``, Euclidean distance
   * - Scaling
     - Evaluate kNN with and without ``StandardScaler`` and report the better accuracy, following the reference paper
   * - Selection
     - Select the learning rate using validation accuracy, then evaluate the selected model once on the test set

Resizing and normalization affect the comparison. The reference paper reports changes in model rankings when images are evaluated at their native 64x64 resolution instead of 224x224. In our runs, for example, changing preprocessing produced a 0.43 range in kNN accuracy for the same ImageNet-pretrained ResNet-50.

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
   * - Supervised ImageNet
     - ResNet-50
     - 0.8948
     - --
     - --
   * - Supervised ImageNet
     - ViT-S/16
     - 0.9178
     - --
     - --
   * - :class:`~torchgeo.tasks.MoCo` v3
     - ResNet-50
     - **0.9494**
     - 1e-2
     - `moco_resnet50.yaml <https://github.com/torchgeo/torchgeo/blob/main/tests/conf/ssl/moco_resnet50.yaml>`__
   * - :class:`~torchgeo.tasks.MoCo` v3
     - ViT-S/16
     - 0.9413
     - 1e-4
     - `moco_vit_small.yaml <https://github.com/torchgeo/torchgeo/blob/main/tests/conf/ssl/moco_vit_small.yaml>`__
   * - :class:`~torchgeo.tasks.SimCLR`
     - ResNet-50
     - 0.9356
     - 1.5
     - `simclr_resnet50.yaml <https://github.com/torchgeo/torchgeo/blob/main/tests/conf/ssl/simclr_resnet50.yaml>`__
   * - :class:`~torchgeo.tasks.SimCLR`
     - ViT-S/16
     - 0.9296
     - 1.5
     - `simclr_vit_small.yaml <https://github.com/torchgeo/torchgeo/blob/main/tests/conf/ssl/simclr_vit_small.yaml>`__
   * - :class:`~torchgeo.tasks.BYOL`
     - ResNet-50
     - 0.9294
     - 1e-3
     - `byol_resnet50.yaml <https://github.com/torchgeo/torchgeo/blob/main/tests/conf/ssl/byol_resnet50.yaml>`__
   * - :class:`~torchgeo.tasks.BYOL`
     - ViT-S/16
     - 0.9241
     - 1e-4
     - `byol_vit_small.yaml <https://github.com/torchgeo/torchgeo/blob/main/tests/conf/ssl/byol_vit_small.yaml>`__

The image-statistics baseline scores 0.8937, higher than either randomly initialized encoder. The ImageNet-pretrained ViT scores 0.9178. Compare against these baselines as well as random initialization when assessing whether SSL improves classification accuracy.

The selected learning rate depends on both the task and the encoder. The rates in this table span four orders of magnitude: MoCo v3 uses 1e-2 with ResNet-50 and 1e-4 with ViT-S/16. Sweep learning rates for each combination rather than assuming one setting will work for all of them. For example, three of the sixteen MoCo and SimCLR runs that we did produced checkpoints with entirely NaN features. Another run collapsed to a near-constant embedding while its loss stayed flat.

.. rubric:: Footnotes

.. [#floor] The image-statistics baseline concatenates each image's per-band mean, standard deviation, minimum, and maximum into a 52-dimensional vector and uses the same kNN classifier. It does not use a neural network.

Running the benchmark
---------------------

Use the configurations in `tests/conf/ssl <https://github.com/torchgeo/torchgeo/tree/main/tests/conf/ssl>`__ to run the six SSL integration tests. For example:

.. code-block:: console

   $ torchgeo fit --config tests/conf/ssl/byol_resnet50.yaml

See :doc:`/tutorials/ssl_knn_eval` for evaluation code and checks for collapse. The tutorial uses EuroSAT100 for a short demonstration; use the full EuroSAT dataset and the settings above for this benchmark.

Adding a new task
-----------------

Follow the :ref:`SSL contribution checklist <contributing-ssl>` when adding a new SSL task.