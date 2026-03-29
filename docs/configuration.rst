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
     - Output directory for TOD files (``tod_day_N.npy``). Created
       automatically if absent.
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
   * - ``calibration_batch_size``
     - ``int | null``
     - ``null``
     - Cached optimal batch size. Written automatically after calibration.

Beam Rotation Cache
-------------------

Pre-computing the psi-roll rotation (via :mod:`precompute_beam_cache`)
eliminates one of the two Rodrigues rotations per sample, yielding roughly a
25 % speed-up. An additional flat-sky approximation can eliminate the second
rotation for narrow beams. Because the psi-roll is evaluated on a discrete
grid rather than continuously, using the cache introduces a small
interpolation error. **Not recommended for experiments requiring high
precision.**

.. list-table::
   :header-rows: 1
   :widths: 25 10 10 55

   * - Key
     - Type
     - Default
     - Description
   * - ``beam_cache_dir``
     - ``str | null``
     - ``null``
     - Path to the directory containing ``.npz`` cache files. ``null``
       disables caching (full double-Rodrigues path used).
   * - ``beam_cache_n_psi``
     - ``int``
     - ``720``
     - Number of psi bins in the cache. **Must match** the ``--n_psi``
       value used when generating the cache.

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
       are active only for this method.
     - Slow

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
   The calibration sweeps a fixed ``(tail_fraction × n_clusters)`` grid,
   measures relative RMS TOD error on a uniformly-strided probe day, and
   writes the best pair back to the config.
2. On all subsequent runs the saved values are used directly and clustering
   calibration is skipped (``clustering_calibration_enabled`` is reset to
   ``false`` automatically).

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
     - ``1.0e-3``
     - Maximum tolerated relative RMS TOD error during calibration.
       The calibration selects the pair that maximises speedup subject
       to this constraint.  See :doc:`beam_interpolation_accuracy` for
       measured interpolation floors and tier-based recommendations.

Full Example
------------

.. code-block:: yaml

   ---
     FOLDER_SCAN:       "/data/scan/"
     FOLDER_TOD_OUTPUT: "/data/tod/"
     path_to_map:       "/data/maps/cmb_IQU.fits"

     FOLDER_BEAM:  "/data/beams/"
     beam_file_I:  "beam_I.fits"
     beam_file_Q:  "beam_Q.fits"
     beam_file_U:  "beam_U.fits"
     power_fraction_threshold_I: 0.99
     power_fraction_threshold_Q: 0.99
     power_fraction_threshold_U: 0.99

     start_day: 0
     end_day: 366

     n_processes: 8
     max_memory_per_process: 2.0

     calibration_enabled: true
     calibration_n_processes: null
     calibration_batch_size: null

     beam_cache_dir: "/data/beam_cache/"
     beam_cache_n_psi: 720

     beam_interp_method: 'bilinear'
     beam_interp_sigma_deg: null
     beam_interp_radius_deg: null

     n_beam_clusters: null
     beam_cluster_tail_fraction: null
     clustering_calibration_enabled: false
     clustering_error_threshold: 1.0e-3
