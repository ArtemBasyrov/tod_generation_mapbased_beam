tod\_calibrate
==============

Runtime and beam-clustering calibration.

:func:`calibrate_runtime` measures sustained throughput over process count,
Numba threads per worker and batch size, using interleaved timing repeats to
avoid cache warm-up bias.  :func:`calibrate_beam_clustering` sweeps
``(tail_fraction × n_clusters × whiten)`` against the B_ℓ width bound and the
exact :math:`m = \pm 2` shape bound computed by :func:`beam_m2_multipoles`.
Both write their results back to the active config file.

.. automodule:: tod_calibrate
   :members: calibrate_runtime, calibrate_beam_clustering, beam_m2_multipoles
   :undoc-members: False
   :show-inheritance: True
