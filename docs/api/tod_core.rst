tod\_core
=========

Core numerical routines for sample-based TOD generation. All public functions
are stateless and accept only arrays as arguments.

.. automodule:: tod_core
   :members: precompute_rotation_vector_batch, beam_tod_batch
   :undoc-members: False
   :show-inheritance: True

.. rubric:: Execution paths in :func:`beam_tod_batch`

Three paths are selected automatically based on the contents of ``beam_data``:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Path
     - Required cache keys
     - Description
   * - Flat-sky
     - ``vec_rolled``, ``psi_grid``, ``dtheta``, ``dphi``, ``mp_stacked``
     - Both Rodrigues rotations and ``vec2ang`` are skipped. Valid for narrow
       beams (≲ 5°). Fastest.
   * - Single-Rodrigues
     - ``vec_rolled``, ``psi_grid``, ``mp_stacked``
     - psi-roll is precomputed; only the recentering rotation is applied at
       runtime. ~2× faster than the full path.
   * - Full double-Rodrigues
     - *(no cache)*
     - Both rotations applied per ``(B, S)`` element at runtime. Most general.
