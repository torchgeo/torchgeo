CMAViT
======

.. currentmodule:: torchgeo.models
.. autoclass:: CMAViT

Inputs and Outputs
^^^^^^^^^^^^^^^^^^

CMAViT takes three modalities and returns one or more predicted maps:

* ``img``: images of shape :math:`\scriptstyle B \times T \times C \times H \times W`
* ``met``: meteorology of shape :math:`\scriptstyle B \times T \times C`, with any number of variables. All the variables of a timestep are embedded as a single token, as meteorology is coarse and has no spatial extent
* ``context``: a sequence of :math:`\scriptstyle B` raw text strings

The number of timesteps :math:`\scriptstyle T` may be smaller than ``num_observations``, but must be the same for images and meteorology.

With ``timeseries=True``, the model returns :math:`\scriptstyle T` maps of shape :math:`\scriptstyle B \times 1 \times H_{out} \times W_{out}`, where the :math:`\scriptstyle i`-th map only uses the first :math:`\scriptstyle i` observations. This allows yield to be forecast throughout the growing season. Otherwise, it returns a single map that uses all observations.

The attention weights are large and only useful for interpretation, so they are only returned with ``return_attention=True``. They then include the cross-attention weights of each window, which show which text tokens each image patch attends to.
