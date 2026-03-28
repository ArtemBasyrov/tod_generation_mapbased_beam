tod\_calibrate
==============

Empirical batch-size and process-count calibration.

The calibration runs once before the day loop, measures sustained throughput
over a range of candidate configurations using interleaved timing repeats (to
avoid L3 cache warm-up bias), and writes the optimal values to the config file
for subsequent runs. All calibration functions are internal helpers (prefixed
with ``_``) and are not part of the public API.

.. automodule:: tod_calibrate
   :undoc-members: False
   :show-inheritance: True
