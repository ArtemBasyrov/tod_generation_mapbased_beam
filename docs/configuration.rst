Configuration Reference
=======================

All settings are read from ``config.yaml`` in the project directory. If a
``config_local.yaml`` file is present it is read instead, allowing
per-machine settings without modifying the tracked file.  Both calibration
steps write their results back into whichever file is active.

Paths and Input Maps
--------------------

.. list-table::
   :header-rows: 1
   :widths: 28 12 12 48

   * - Key
     - Type
     - Default
     - Description
   * - ``FOLDER_SCAN``
     - ``str``
     - required
     - Directory containing scan files (``theta_N.npy``, ``phi_N.npy``,
       ``psi_N.npy``). Must end with a path separator.
   * - ``FOLDER_TOD_OUTPUT``
     - ``str``
     - required
     - Output directory. Created automatically if absent.
   * - ``path_to_map``
     - ``str``
     - required
     - HEALPix sky-map FITS file (RING ordering) with Stokes I, Q, U in
       fields 0, 1, 2.
   * - ``map_fields``
     - ``list[int]``
     - ``[0, 1, 2]``
     - Stokes components to read: ``0`` = T, ``1`` = Q, ``2`` = U.  Any
       non-empty subset is valid.  See :ref:`partial_stokes`.
   * - ``FOLDER_BEAM``
     - ``str``
     - required
     - Directory containing beam FITS files. Must end with a path separator.
   * - ``beam_file_I``
     - ``str``
     - —
     - I beam file inside ``FOLDER_BEAM``, a
       `pixell / enmap <https://pixell.readthedocs.io/en/latest/usage.html#usagepage>`_
       FITS file.  Required when ``0`` is in ``map_fields``.
   * - ``beam_file_Q``
     - ``str``
     - —
     - Q beam file.  Required when ``1`` is in ``map_fields``.
   * - ``beam_file_U``
     - ``str``
     - —
     - U beam file.  Required when ``2`` is in ``map_fields``.
   * - ``beam_center_x``
     - ``int | null``
     - ``null``
     - Row (Dec-axis) index of the beam-centre pixel.  ``null`` uses
       ``H // 2``.
   * - ``beam_center_y``
     - ``int | null``
     - ``null``
     - Column (RA-axis) index of the beam-centre pixel.  ``null`` uses
       ``W // 2``.

Components may share one beam file, and Q and U may use different beams.
A Q/U weight difference is applied in the detector polarisation basis, so the
TOD does not depend on ψ or on the sky coordinate system.

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
     - —
     - Fraction of the integrated I beam power to retain.  Required when
       ``0`` is in ``map_fields``.
   * - ``power_fraction_threshold_Q``
     - ``float``
     - —
     - Same for the Q beam.
   * - ``power_fraction_threshold_U``
     - ``float``
     - —
     - Same for the U beam.

Pixels are ranked by the solid-angle-weighted value ``B cos(dec)``, so the
retained fraction is a fraction of *integrated* power.  ``1.00`` keeps every
non-zero pixel; lower values drop the faintest ones.  With beam clustering
enabled a 100 % beam is affordable.

Batch Range
-----------

The pipeline uses the term *day* for the index suffix of the scan files, but
the index can represent any batching unit — an observation session, a CES, an
hour of data, etc.

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Key
     - Type
     - Default
     - Description
   * - ``start_day``
     - ``int``
     - required
     - First day to process (inclusive).
   * - ``end_day``
     - ``int | null``
     - required
     - Last day to process (exclusive). ``null`` processes every day found in
       ``FOLDER_SCAN``.

Multiprocessing and Runtime Calibration
---------------------------------------

The first run measures sustained throughput over process count, Numba threads
per worker and batch size, writes the best values back to the active config,
and sets ``calibration_enabled: false`` so later runs skip it.

.. list-table::
   :header-rows: 1
   :widths: 30 12 12 46

   * - Key
     - Type
     - Default
     - Description
   * - ``n_processes``
     - ``int``
     - required
     - Upper bound on worker processes. On a cluster the scheduler allocation
       (``SLURM_CPUS_PER_TASK``, etc.) takes precedence and this value is only
       a cap.
   * - ``mp_start_method``
     - ``str``
     - ``'spawn'``
     - ``'spawn'`` is safe on all platforms.  ``'fork'`` starts workers faster
       on Linux but may deadlock on macOS.
   * - ``calibration_enabled``
     - ``bool``
     - ``true``
     - Run the runtime calibration on this invocation.  Reset to ``false``
       once it completes.
   * - ``calibration_n_processes``
     - ``int | null``
     - ``null``
     - Calibrated process count.  Written automatically.
   * - ``calibration_numba_threads``
     - ``int | null``
     - ``null``
     - Calibrated Numba threads per worker.  Written automatically.
   * - ``calibration_batch_size``
     - ``int | null``
     - ``null``
     - Calibrated samples per batch.  Written automatically.

The per-process memory budget is derived with ``psutil``: 75 % of available
RAM on a local machine, all of it on a detected cluster (SLURM, PBS, LSF,
SGE).

Numerics
--------

.. list-table::
   :header-rows: 1
   :widths: 28 12 12 48

   * - Key
     - Type
     - Default
     - Description
   * - ``beam_interp_method``
     - ``str``
     - ``'bilinear'``
     - ``'bilinear'`` or ``'nearest'``.  See the table below.
   * - ``precision``
     - ``str``
     - ``'float32'``
     - ``'float32'`` or ``'float64'``.  Sets the dtype of the sky map, the
       beam weights and node vectors, and the stored TOD.
   * - ``spin2_skip_tolerance``
     - ``float | null``
     - ``null``
     - Skip the spin-2 Q/U transport for boresights in an equatorial band
       where the rotation it would apply, :math:`|2\delta|`, stays below this
       tolerance [rad].  ``null`` or ``0`` always applies it.  ``0.01`` is a
       reasonable speed setting; leave it ``null`` for validation runs.

``precision`` does not affect the pointing, which is always float64, nor the
bilinear weights, the spin-2 cache, the accumulators or B_ℓ.  At float32 the
remaining cost is the beam node vectors, at 3.8e-6 of the TOD rms.

**Interpolation methods:**

.. list-table::
   :header-rows: 1
   :widths: 15 70 15

   * - Value
     - Description
     - Speed
   * - ``'bilinear'`` *(recommended)*
     - 4-pixel bilinear HEALPix interpolation via a fused Numba kernel.
       Smooth under rotation and accurate below 0.1 % at nside 2048.
     - 4,272 samples/s/core
   * - ``'nearest'``
     - Uses the HEALPix pixel that *contains* each beam node, evaluated in
       closed form.  Jumps discretely when nodes cross pixel boundaries under
       rotation.  **Do not use for polarisation analysis.**
     - 7,684 samples/s/core

Throughputs were measured on an Apple M4 Pro core at nside 1024 with a 30′
beam at 100 % power, ``precision: float64`` and the spin-2 transport on.
See :doc:`beam_interpolation_accuracy` for accuracy measurements.

Half-Wave Plate
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 50

   * - Key
     - Type
     - Default
     - Description
   * - ``hwp_enabled``
     - ``bool``
     - ``false``
     - Modulate Q/U by an ideal rotating HWP.  Requires both Q and U in
       ``map_fields``.
   * - ``hwp_rotation_frequency_hz``
     - ``float``
     - ``0.0``
     - Physical HWP rotation rate :math:`f_{\rm HWP}` [Hz].  Must be positive
       when ``hwp_enabled`` is true.
   * - ``hwp_initial_phase_rad``
     - ``float``
     - ``0.0``
     - HWP angle :math:`\varphi_0` at the start of day 0 [rad].

The HWP angle is :math:`\varphi(t) = 2\pi f_{\rm HWP}\,t + \varphi_0`,
wrapped to :math:`[0, 2\pi)`, with *t* running continuously from the start of
day 0.  The beam-convolved Q/U are rotated by :math:`4\varphi`, so a mapmaker
still forms :math:`d = I + Q'\cos 2\psi + U'\sin 2\psi`.  The HDF5 export
stores the same wrapped angle, bit for bit.

Output Format
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 10 25 40

   * - Key
     - Type
     - Default
     - Description
   * - ``furax_export``
     - ``bool``
     - ``true``
     - ``true`` writes one TOAST HDF5 observation ``obs_day_N.h5`` per day
       and no ``.npy``.  ``false`` writes the raw ``tod_day_N.npy`` instead.
   * - ``furax_export_t0``
     - ``str``
     - ``'2030-01-01T00:00:00+00:00'``
     - ISO-8601 UTC time of sample 0 of day 0.

See :doc:`data_formats` for both output formats.

.. _configuration:Beam Pixel Clustering:

Beam Pixel Clustering
---------------------

Spatial k-means clustering on the unit sphere reduces the number of
effective beam pixels during TOD generation.  Only the low-power *tail*
of the beam is clustered; the high-power main-lobe pixels are kept
pixel-exact.  This trades a small, controllable accuracy loss for a
proportional reduction in computation.

**Workflow**

1. Set ``clustering_calibration_enabled: true`` and run the pipeline once.
   The calibration sweeps a ``(tail_fraction × n_clusters × whiten)`` grid.
   For each point it clusters the beam pixels and measures two errors: the
   divergence of the beam transfer function B_ℓ against the unclustered
   reference, and the ellipticity the clustering adds to the beam.  The
   point that maximises the pixel-count speedup subject to both bounds is
   written back to the config as ``n_beam_clusters``,
   ``beam_cluster_tail_fraction`` and ``beam_cluster_whiten``.
2. On all subsequent runs the saved values are used directly and clustering
   calibration is skipped (``clustering_calibration_enabled`` is reset to
   ``false`` automatically).

No scan data or TOD generation is needed during calibration — both metrics
are computed purely from beam geometry.  See :doc:`beam_cluster_calibration`
for the metric definitions and guidance on choosing the bounds.

.. warning::

   The two bounds constrain different harmonics and neither implies the
   other.  B_ℓ depends on each beam node only through its distance from the
   beam centre, so it is the :math:`m = 0` harmonic — it bounds the beam
   *width* and is blind to any reshaping.  ``clustering_ellipticity_tolerance``
   bounds the :math:`m = \pm 2` term, which is the one that sources T→P
   leakage.

**Manual override:** set ``clustering_calibration_enabled: false`` and
fill in ``n_beam_clusters``, ``beam_cluster_tail_fraction`` and
``beam_cluster_whiten`` by hand.

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
     - Number of clusters for the tail.  ``null`` disables clustering
       entirely.  Set automatically by calibration.
   * - ``beam_cluster_tail_fraction``
     - ``float | null``
     - ``null``
     - Fraction of integrated beam power treated as the "tail" to be
       clustered.  The remaining ``(1 − fraction)`` of power is kept
       pixel-exact.  Set automatically by calibration.
   * - ``beam_cluster_whiten``
     - ``bool``
     - ``true``
     - Assign tail pixels to clusters in the frame where the beam's second
       moment is isotropic.  Whether this helps is a property of the beam and
       must be measured, so the calibration sweeps it and writes the winner.
       Ignored when ``beam_symmetric: true``.
   * - ``beam_symmetric``
     - ``bool``
     - ``false``
     - Declare the beam a function of angular distance alone.  Clustering then
       groups one quadrant and copies the partition onto the other three by
       90° rotation, so each cell's manufactured ``m = ±2`` is cancelled by its
       three siblings.  Only set this for a beam that really is symmetric — the
       measured 90° asymmetry is printed at load but is not enforced.  Requires an odd-sized beam grid with
       equal RA and Dec spacing.  See :doc:`beam_cluster_calibration`.
   * - ``clustering_calibration_enabled``
     - ``bool``
     - ``false``
     - Run the clustering calibration sweep on this invocation.
       Automatically reset to ``false`` after calibration completes.
   * - ``clustering_error_threshold``
     - ``float``
     - ``1.0e-3``
     - **m = 0 bound.**  Maximum tolerated relative RMS B_ℓ divergence
       between the clustered and reference beam transfer function.  Bounds
       the beam width.  The ``config.yaml`` template sets ``1.0e-5``.  See
       :doc:`beam_cluster_calibration` for tier-based recommendations.
   * - ``clustering_ellipticity_tolerance``
     - ``float | null``
     - ``1.0``
     - **m = ±2 bound.**  Maximum tolerated error in the beam's own
       ``b_{l,2}`` multipoles, computed exactly on the sphere,
       as a factor over the least the sweep achieves for this beam.  Relative
       rather than absolute because what is achievable depends on the beam,
       the beam-grid spacing and the cluster budget together.  ``1.0`` keeps
       only the least-ellipticity configuration; larger values trade shape
       for speed; ``null`` reports without gating.  Values below ``1.0``
       raise at config load — they would reject every candidate.

.. _partial_stokes:

Partial-Stokes Maps
-------------------

``map_fields`` selects which Stokes components are read from the sky map.
Inactive components are never allocated and their beam files are never
opened, and their ``beam_file_*`` / ``power_fraction_threshold_*`` entries may
be omitted or set to ``null``.  ``tod_config`` raises a ``ValueError`` at
import listing any entry missing for an active component.

The spin-2 Q/U transport runs only when both Q and U are active
(``[1, 2]`` or ``[0, 1, 2]``).  Every other subset uses the scalar path.  The
output always carries all three ``[I, Q, U]`` rows; inactive rows are zero.

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
     power_fraction_threshold_I: 1.00
     power_fraction_threshold_Q: 1.00
     power_fraction_threshold_U: 1.00
     beam_center_x: null
     beam_center_y: null

     start_day: 0
     end_day: 366

     n_processes: 8
     mp_start_method: 'spawn'
     calibration_enabled: true
     calibration_n_processes: null
     calibration_numba_threads: null
     calibration_batch_size: null

     beam_interp_method: 'bilinear'
     precision: 'float32'
     spin2_skip_tolerance: null

     n_beam_clusters: null
     beam_cluster_tail_fraction: null
     beam_cluster_whiten: true
     beam_symmetric: false
     clustering_calibration_enabled: true
     clustering_error_threshold: 1.0e-5
     clustering_ellipticity_tolerance: 1.0

     hwp_enabled: false
     hwp_rotation_frequency_hz: 0.0
     hwp_initial_phase_rad: 0.0

     furax_export: true
     furax_export_t0: '2030-01-01T00:00:00+00:00'
