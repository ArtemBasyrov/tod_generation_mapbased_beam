tod\_core
=========

Batch-level TOD kernel.  :func:`beam_tod_batch` rotates the beam onto each
boresight sample, gathers the sky map at every beam node with the configured
interpolation method (``'bilinear'`` or ``'nearest'``), applies the spin-2
Q/U transport, and accumulates the beam-weighted sum.

.. automodule:: tod_core
   :members: beam_tod_batch
   :undoc-members: False
   :show-inheritance: True
