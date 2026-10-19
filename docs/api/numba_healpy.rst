numba\_healpy
=============

Numba JIT re-implementation of the HEALPix RING helpers the kernels need.

These are inlined into the Numba nopython kernels to eliminate Python-level
dispatch overhead in the hot loop. The bilinear weights are bit-identical to
``healpy.get_interp_weights``. All functions operate in RING ordering only.

.. automodule:: numba_healpy
   :members: get_interp_weights_numba
   :undoc-members: False
   :show-inheritance: True
