sample\_based\_tod\_generation\_gridint
=======================================

Main pipeline entry point. Loads the sky map and beam data, runs the
clustering and runtime calibrations when enabled, spawns a multiprocessing
pool, and writes one output per observation day: a TOAST HDF5 observation
(``furax_export: true``) or a raw ``(3, n_samples)`` ``.npy``.

.. automodule:: sample_based_tod_generation_gridint
   :members: main, tod_exact_gen_batched
   :undoc-members: False
   :show-inheritance: True
