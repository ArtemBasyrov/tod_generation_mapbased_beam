Configuration Reference
=======================

All settings are read from ``config.yaml`` in the project directory. If a
``config_local.yaml`` file is present it takes precedence over ``config.yaml``,
allowing per-machine overrides without modifying the tracked file.

Paths
-----

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Key
     - Type
     - Description
   * - ``FOLDER_SCAN``
     - ``str``
     - Directory containing scan files (``theta_N.npy``, ``phi_N.npy``,
       ``psi_N.npy``). Must end with a path separator.
   * - ``FOLDER_TOD_OUTPUT``
     - ``str``
     - Output directory for TOD files (``obs_day_N.h5`` by default, or
       ``tod_day_N.npy`` when ``furax_export: false``). Created automatically
       if absent.
   * - ``path_to_map``
     - ``str``
     - Path to the HEALPix sky-map FITS file containing Stokes I, Q, U
       in fields 0, 1, 2 respectively.
   * - ``FOLDER_BEAM``
     - ``str``
     - Directory containing beam FITS files.
   * - ``beam_file_I``
     - ``str``
     - Filename of the intensity (I) beam map inside ``FOLDER_BEAM``.
       Must be a `pixell / enmap <https://pixell.readthedocs.io/en/latest/usage.html#usagepage>`_ FITS file.
   * - ``beam_file_Q``
     - ``str``
     - Filename of the Q-polarisation beam map inside ``FOLDER_BEAM``.
   * - ``beam_file_U``
     - ``str``
     - Filename of the U-polarisation beam map inside ``FOLDER_BEAM``.

Only the ``beam_file_*`` / ``power_fraction_threshold_*`` entries whose Stokes
index appears in ``map_fields`` are required; entries for inactive components
may be omitted or set to ``null``.

Stokes Component Selection
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 50

   * - Key
     - Type
     - Default
     - Description
   * - ``map_fields``
     - ``list[int]``
     - ``[0, 1, 2]``
     - Which Stokes components to read from the input FITS map — a non-empty
       subset of ``[0, 1, 2]`` = ``[T, Q, U]``. Use ``[0]`` for a
       temperature-only map.

The spin-2 Q/U frame rotation runs only when **both** Q (``1``) and U (``2``)
are active. Any other subset takes the scalar gather path. The output TOD shape
is always ``(3, n_samples)``; rows for inactive components are written as zeros.

Beam Centre
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 50

   * - Key
     - Type
     - Default
     - Description
   * - ``beam_center_x``
     - ``int | null``
     - ``null``
     - Row index of the beam-centre pixel in the beam array. ``null`` →
       ``H // 2``.
   * - ``beam_center_y``
     - ``int | null``
     - ``null``
     - Column index of the beam-centre pixel. ``null`` → ``W // 2``.

Beam Pixel Selection
--------------------

.. list-table::
   :header-rows: 1
   :widths: 35 10 10 45

   * - Key
     - Type
     - Default
     - Description
   * - ``power_fraction_threshold_I``
     - ``float``
     - ``0.99``
     - Fraction of total beam power to retain for the I beam. Pixels whose
       dB value falls below the implied cut are discarded. Increase toward
       ``1.0`` for higher fidelity; decrease toward ``0.9`` to aggressively
       prune faint sidelobes.
   * - ``power_fraction_threshold_Q``
     - ``float``
     - ``0.99``
     - Same for the Q beam.
   * - ``power_fraction_threshold_U``
     - ``float``
     - ``0.99``
     - Same for the U beam.

Batch Range
-----------

These keys control which scan files are processed. The pipeline uses the term
*day* for the index suffix of the scan files (``theta_N.npy``, etc.), but the
index can represent any batching unit you choose — an observation session, a
CES, an hour of data, etc.

.. list-table::
   :header-rows: 1
   :widths: 20 10 20 50

   * - Key
     - Type
     - Default
     - Description
   * - ``start_day``
     - ``int``
     - ``0``
     - First batch index to process (inclusive).
   * - ``end_day``
     - ``int``
     - total batches
     - Last batch index to process (exclusive). Set to ``null`` to process all
       batches found in ``FOLDER_SCAN``.

Multiprocessing
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Key
     - Type
     - Description
   * - ``n_processes``
     - ``int``
     - Maximum worker processes on a local machine. On a cluster the scheduler
       allocation (``SLURM_CPUS_PER_TASK``, etc.) takes precedence and this
       value is used only as a cap.
   * - ``max_memory_per_process``
     - ``float``
     - Per-process memory budget in GB. Used as a fallback when ``psutil``
       is unavailable.
   * - ``mp_start_method``
     - ``str``
     - Multiprocessing start method. ``'spawn'`` (default) is safe everywhere;
       ``'fork'`` is faster on Linux (avoids re-running Numba JIT per worker)
       but may deadlock on macOS.

Calibration Cache
-----------------

The first run measures sustained throughput at several batch sizes and
process counts, writes the optimal values back to the active config, and
sets ``calibration_enabled: false`` for future runs so calibration is
skipped automatically.

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 50

   * - Key
     - Type
     - Default
     - Description
   * - ``calibration_enabled``
     - ``bool``
     - ``true``
     - Run calibration on this invocation. Automatically reset to ``false``
       after calibration completes so subsequent runs reuse cached values.
   * - ``calibration_n_processes``
     - ``int | null``
     - ``null``
     - Cached optimal process count. Written automatically after calibration.
   * - ``calibration_numba_threads``
     - ``int | null``
     - ``null``
     - Cached optimal Numba ``prange`` thread count per worker. Written
       automatically after calibration.
   * - ``calibration_batch_size``
     - ``int | null``
     - ``null``
     - Cached optimal batch size. Written automatically after calibration.

Beam Interpolation
------------------

.. list-table::
   :header-rows: 1
   :widths: 28 10 12 50

   * - Key
     - Type
     - Default
     - Description
   * - ``beam_interp_method``
     - ``str``
     - ``'bilinear'``
     - Interpolation strategy. See the table below.
   * - ``beam_interp_sigma_deg``
     - ``float | null``
     - ``null``
     - Gaussian kernel width [degrees]. Defaults to one HEALPix pixel
       resolution. Ignored when ``beam_interp_method != 'gaussian'``.
   * - ``beam_interp_radius_deg``
     - ``float | null``
     - ``null``
     - Neighbour search radius [degrees]. Defaults to ``3 × sigma``.
       Ignored when ``beam_interp_method != 'gaussian'``.

**Available interpolation methods:**

.. list-table::
   :header-rows: 1
   :widths: 15 70 15

   * - Value
     - Description
     - Speed
   * - ``'nearest'``
     - Single nearest-pixel lookup. No blending between adjacent pixels.
       Fastest option; suitable when the beam pixel resolution is much finer
       than the sky-map resolution.
     - Fastest
   * - ``'bilinear'`` *(recommended)*
     - 4-pixel bilinear HEALPix interpolation via a fused Numba kernel.
       Best balance of speed and accuracy for most beams. **This is the
       recommended method.**
     - Fast
   * - ``'gaussian'``
     - Isotropic Gaussian kernel over all HEALPix pixels within
       ``radius_deg``. Avoids grid-aligned interpolation artefacts; best
       for wide or asymmetric beams. ``sigma_deg`` and ``radius_deg``
       are active only for this method. Available on the ``gaussian`` branch.
     - Slow
   * - ``'totalconvolve'``
     - ducc0 NUFFT synthesis (``synthesis_general``, spin-0 for T, spin-2 for
       Q/U). Accuracy set by ``totalconvolve_epsilon``; eliminates the flat
       high-ℓ noise floor. Requires ``pip install ducc0``. Available on the
       ``totalconvolve`` branch.
     - 5–15× slower

.. _configuration:Beam Pixel Clustering:

Beam Pixel Clustering
---------------------

Spatial k-means clustering on the unit sphere can reduce the number of
effective beam pixels during TOD generation.  Only the low-power *tail*
of the beam is clustered; the high-power main-lobe pixels are kept
pixel-exact.  This trades a small, controllable accuracy loss for a
proportional reduction in computation.

**Workflow**

1. Set ``clustering_calibration_enabled: true`` and run the pipeline once.
   The calibration sweeps a ``(tail_fraction × n_clusters)`` grid.  For
   each pair it clusters the beam pixels and computes the beam transfer
   function B_ℓ from the clustered geometry, comparing it against the
   reference B_ℓ of the unclustered beam.  The pair that maximises the
   pixel-count speedup while keeping B_ℓ divergence below
   ``clustering_error_threshold`` is written back to the config.
2. On all subsequent runs the saved values are used directly and clustering
   calibration is skipped (``clustering_calibration_enabled`` is reset to
   ``false`` automatically).

No scan data or TOD generation is needed during calibration — the metric
is computed purely from beam geometry.  See :doc:`beam_cluster_calibration`
for a detailed description of the B_ℓ divergence metric and guidance on
choosing the threshold.

**Manual override:** set ``clustering_calibration_enabled: false`` and
fill in ``n_beam_clusters`` and ``beam_cluster_tail_fraction`` by hand.

**Disable entirely:** set ``n_beam_clusters: null``.

.. list-table::
   :header-rows: 1
   :widths: 38 10 10 42

   * - Key
     - Type
     - Default
     - Description
   * - ``n_beam_clusters``
     - ``int | null``
     - ``null``
     - Maximum number of clusters for the tail.  ``null`` disables
       clustering entirely.  Set automatically by calibration.
   * - ``beam_cluster_tail_fraction``
     - ``float | null``
     - ``null``
     - Fraction of total beam power treated as the "tail" to be clustered.
       The remaining ``(1 − fraction)`` of power pixels are kept pixel-exact.
       Set automatically by calibration.
   * - ``clustering_calibration_enabled``
     - ``bool``
     - ``false``
     - Run the clustering calibration sweep on this invocation.
       Automatically reset to ``false`` after calibration completes.
   * - ``clustering_error_threshold``
     - ``float``
     - ``1.0e-5``
     - Maximum tolerated relative RMS B_ℓ divergence between the clustered
       and reference beam transfer function.  The calibration selects the
       pair that maximises speedup subject to this constraint.  See
       :doc:`beam_cluster_calibration` for metric definition and
       tier-based recommendations.

Spin-2 Skip Optimisation
------------------------

The spin-2 Q/U frame correction is negligible near the equator and grows toward
the poles. Skipping it inside an equatorial band saves work at a bounded cost.

.. list-table::
   :header-rows: 1
   :widths: 28 15 12 45

   * - Key
     - Type
     - Default
     - Description
   * - ``spin2_skip_tolerance``
     - ``float | null``
     - ``null``
     - Maximum per-sample fractional Q/U error tolerated by skipping the
       correction in the equatorial band. ``null`` or ``0`` always applies the
       correction. Recommended when used: ``0.01`` (1 %).

Working Precision
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 56

   * - Key
     - Type
     - Default
     - Description
   * - ``precision``
     - ``str``
     - ``'float32'``
     - Working precision for the float32-side of the pipeline (sky map,
       pointings, beam values, Rodrigues rotation, saved TOD). ``'float64'`` is
       a validation knob — slower and more memory, isolating whether high-ℓ
       residuals are precision-limited. Float64 surfaces (bilinear weights,
       spin-2 cache, accumulators, B_ℓ) are unchanged either way.

Half-Wave Plate (HWP)
---------------------

When enabled, the generator rotates the per-sample ``(Q, U)`` output by
``4·φ_HWP(t)``, with ``φ_HWP(t) = 2π·f_HWP·t + φ₀`` and ``t`` continuous across
days. The rotation is applied **after** beam convolution and does not alter the
beam shape or the Rodrigues rotation.

.. list-table::
   :header-rows: 1
   :widths: 32 12 12 44

   * - Key
     - Type
     - Default
     - Description
   * - ``hwp_enabled``
     - ``bool``
     - ``false``
     - Enable HWP modulation of the output Q/U.
   * - ``hwp_rotation_frequency_hz``
     - ``float``
     - ``0.0``
     - Physical HWP rotation rate ``f_HWP`` [Hz].
   * - ``hwp_initial_phase_rad``
     - ``float``
     - ``0.0``
     - Phase ``φ₀`` at ``t = 0`` (start of day 0) [rad].

Focal-Plane Detectors
---------------------

By default the boresight is a single implicit detector. A ``detectors`` list
simulates a focal plane: each detector is offset from the boresight in the
TOAST xi-eta-gamma convention (degrees) and its per-detector pointing is
composed on the fly via quaternions that mirror ``furax.math.quaternion``
exactly, so the generator and furax compose identical pointing.

.. list-table::
   :header-rows: 1
   :widths: 22 15 10 53

   * - Key
     - Type
     - Default
     - Description
   * - ``detectors``
     - ``list | null``
     - ``null``
     - Focal-plane detector list. Each entry is
       ``{name, xi_deg, eta_deg, gamma_deg}``. ``gamma_deg`` is the
       polarisation-angle offset (an A/B pair shares xi/eta and differs by
       ``gamma_deg = 90``). ``null`` → single boresight detector.
   * - ``detector_subset``
     - ``list | null``
     - ``null``
     - Run only part of the focal plane (detector names or 0-based indices).
       Used to shard a focal plane across HPC nodes. ``null`` → all detectors.

A detector entry may additionally carry ``beam_file_{I,Q,U}``,
``power_fraction_threshold_{I,Q,U}``, ``n_beam_clusters`` and/or
``beam_cluster_tail_fraction``. Any omitted key falls back to the global value.
Detectors whose resolved beam files **and** clustering coincide share one beam
set in memory, so an A/B pair on the same beams is loaded and clustered once —
this is how genuinely asymmetric per-detector beams are run.

.. code-block:: yaml

   detectors:
     - {name: det_000A, xi_deg: 0.0, eta_deg: 0.0, gamma_deg: 0.0}
     - {name: det_000B, xi_deg: 0.0, eta_deg: 0.0, gamma_deg: 90.0}
     - {name: det_001A, xi_deg: 0.3, eta_deg: 0.1, gamma_deg: 0.0,
        beam_file_I: beam_001_I.fits, beam_file_Q: beam_001_Q.fits,
        beam_file_U: beam_001_U.fits,
        n_beam_clusters: 100, beam_cluster_tail_fraction: 0.03}

furax HDF5 Export
-----------------

By default the generator writes one furax-compatible TOAST HDF5 observation per
day (``obs_day_N.h5``, a ``(n_det, n_samples)`` detdata block) instead of
``.npy`` files. The standalone ``tod_to_furax.py`` re-exports existing ``.npy``
files with its own ``--output`` / ``--precision`` / ``--split-per-day`` flags.

.. list-table::
   :header-rows: 1
   :widths: 24 12 24 40

   * - Key
     - Type
     - Default
     - Description
   * - ``furax_export``
     - ``bool``
     - ``true``
     - ``true`` → write ``obs_day_N.h5`` per day (no ``.npy``). ``false`` →
       write raw per-detector ``tod_day_N.npy``. Requires the ``toast`` stack
       (imported lazily).
   * - ``furax_export_t0``
     - ``str``
     - ``'2030-01-01T00:00:00+00:00'``
     - UTC time of sample 0 of day 0 (timestream time origin).

Full Example
------------

.. code-block:: yaml

   ---
     FOLDER_SCAN:       "/data/scan/"
     FOLDER_TOD_OUTPUT: "/data/tod/"
     path_to_map:       "/data/maps/cmb_IQU.fits"
     map_fields: [0, 1, 2]

     FOLDER_BEAM:  "/data/beams/"
     beam_file_I:  "beam_I.fits"
     beam_file_Q:  "beam_Q.fits"
     beam_file_U:  "beam_U.fits"
     power_fraction_threshold_I: 0.99
     power_fraction_threshold_Q: 0.99
     power_fraction_threshold_U: 0.99
     beam_center_x: null
     beam_center_y: null

     start_day: 0
     end_day: 366

     n_processes: 8
     max_memory_per_process: 2.0
     mp_start_method: 'spawn'

     calibration_enabled: true
     calibration_n_processes: null
     calibration_numba_threads: null
     calibration_batch_size: null

     beam_interp_method: 'bilinear'
     spin2_skip_tolerance: null
     precision: 'float32'

     n_beam_clusters: null
     beam_cluster_tail_fraction: null
     clustering_calibration_enabled: false
     clustering_error_threshold: 1.0e-5

     hwp_enabled: false
     hwp_rotation_frequency_hz: 0.0
     hwp_initial_phase_rad: 0.0

     detectors: null
     detector_subset: null

     furax_export: true
     furax_export_t0: '2030-01-01T00:00:00+00:00'
