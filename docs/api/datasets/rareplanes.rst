.. _RarePlanes:

RarePlanes
==========

.. currentmodule:: torchgeo.datasets
.. autoclass:: RarePlanes


Plotting a sample
-----------------

After downloading the dataset, display the pixel bounding boxes with:

.. code-block:: python

    from torchgeo.datasets import RarePlanes

    dataset = RarePlanes(root="data/rareplanes", dataset_type="real", split="train")
    figure = dataset.plot(dataset[0])

Aircraft use label 1; label 0 is reserved for background. Add
``prediction_bbox_xyxy`` to the sample to display predicted boxes in a second panel.
