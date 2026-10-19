tod\_pipeline\_helpers
======================

Shared set-up for the two entry scripts: beam loading and pixel selection
(with the ``cos(dec)`` solid-angle Jacobian), beam clustering, merging of the
per-component beam entries, the spin-2 skip-band threshold, HWP modulation,
and writing calibration results back to the config.

.. automodule:: tod_pipeline_helpers
   :members: prepare_beam_data, apply_beam_clustering, merge_beam_entries,
             resolve_spin2_skip_threshold, apply_hwp_modulation,
             save_runtime_calibration, save_clustering_calibration
   :undoc-members: False
   :show-inheritance: True
