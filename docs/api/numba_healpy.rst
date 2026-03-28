numba\_healpy
=============

Numba JIT re-implementations of HEALPix helper functions.

These are drop-in replacements for the equivalent ``healpy`` functions,
inlined into the Numba nopython kernels to eliminate Python-level dispatch
overhead in the hot tile loop. All functions operate in RING ordering only.

.. automodule:: numba_healpy
   :members: get_interp_weights_numba, pix2ang_numba, query_disc_numba
   :undoc-members: False
   :show-inheritance: True
