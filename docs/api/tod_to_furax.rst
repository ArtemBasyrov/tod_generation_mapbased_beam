tod\_to\_furax
==============

TOAST HDF5 export for the furax mapmaker.  :func:`write_day_observation` is
the writer the generator calls when ``furax_export: true``.  Run as a script
to re-export existing ``tod_day_N.npy`` files::

    python tod_to_furax.py [--output DIR] [--start-day N] [--end-day N]
                           [--t0 ISO8601] [--split-per-day N]
                           [--precision float32|float64]

.. automodule:: tod_to_furax
   :members: write_day_observation, convert_day, observation_paths
   :undoc-members: False
   :show-inheritance: True
