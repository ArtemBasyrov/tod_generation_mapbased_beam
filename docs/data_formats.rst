Data Formats
============

This page documents all input and output file formats used by the pipeline.

Input Files
-----------

Sky Map (``path_to_map``)
~~~~~~~~~~~~~~~~~~~~~~~~~

A HEALPix FITS file readable by ``healpy.read_map``. Must contain exactly
three fields:

* **Field 0** — Stokes I (intensity)
* **Field 1** — Stokes Q (linear polarisation)
* **Field 2** — Stokes U (linear polarisation)

All three fields must share the same ``nside``. Values are loaded and stored
as ``float32``. Any ``healpy``-compatible HEALPix FITS file (RING ordering)
is accepted.

Beam Files (``FOLDER_BEAM / beam_file_{I,Q,U}``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`pixell / enmap <https://pixell.readthedocs.io/en/latest/usage.html#usagepage>`_ FITS format (2-D map). Requirements:

* The beam must be centred such that the beam axis falls on the grid centre
  pixel ``(H // 2, W // 2)`` for a map of shape ``(H, W)``.  RA and Dec
  coordinates are read from the WCS header and expressed as offsets relative
  to that centre pixel.
* Values represent beam amplitude in linear units (not dB).
* The I, Q and U beams may share the same file (set all three ``beam_file_*``
  keys to the same filename).
* Normalisation of the beam amplitude is **not required**. The pipeline
  re-normalises beam weights internally based on the power threshold, so the
  absolute scale of the beam file does not affect the TOD signal amplitude.

For a worked example of how to generate a synthetic beam file see
:doc:`beam_creation`.

Scan Files (``FOLDER_SCAN``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One triplet of ``.npy`` files per processing unit (referred to as a *day* in
the filenames, but this can represent any convenient batch — an observation
session, a CES, an hour of data, etc.):

.. code-block:: text

   theta_{day_index}.npy   # boresight colatitude  [rad]
   phi_{day_index}.npy     # boresight longitude   [rad]
   psi_{day_index}.npy     # polarisation roll     [rad]

Each file is a 1-D array with one element per detector sample. The dtype may
be ``float32`` or ``float64``; all three are converted to ``float32`` when
loaded. Files are opened as ``numpy`` memory-maps so only the currently
processed batch is resident in RAM.

The total number of processing units is inferred from the highest index found
among ``psi_*.npy`` files. The sampling rate is estimated as
``len(psi_0.npy) / 86400`` samples per second.

Beam Rotation Cache (``beam_cache_dir``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optional ``.npz`` files produced by ``precompute_beam_cache.py``. The cache
filename pattern is::

    {beam_stem}_cache_npsi{N_psi}.npz

Contents:

.. list-table::
   :header-rows: 1
   :widths: 15 20 10 55

   * - Key
     - Shape
     - dtype
     - Description
   * - ``psi_grid``
     - ``(N_psi,)``
     - float32
     - psi bin centres [rad] covering ``[0, 2π)``.
   * - ``vec_rolled``
     - ``(N_psi, S, 3)``
     - float32
     - Beam-pixel unit vectors after psi-roll for each bin. ``S`` is the
       number of pixels selected by the power threshold.
   * - ``beam_vals``
     - ``(S,)``
     - float32
     - Normalised beam weights for the ``S`` selected pixels.
   * - ``beam_ctr``
     - ``(3,)``
     - float32
     - Beam-centre unit vector. Always ``[1, 0, 0]`` for correctly centred
       beam maps.
   * - ``dtheta``
     - ``(N_psi, S)``
     - float32
     - Flat-sky colatitude offsets [rad] from the beam centre.
       Present only when the cache was generated without ``--no_offsets``.
       Valid for narrow beams (≲ 5°).
   * - ``dphi``
     - ``(N_psi, S)``
     - float32
     - Flat-sky phi offsets [rad] (raw; divide by ``sin(theta_b)`` at
       runtime). Present together with ``dtheta``.

When both ``dtheta`` and ``dphi`` are present the pipeline uses the
*flat-sky* execution path, which skips both Rodrigues rotations and the
``vec2ang`` call entirely.

Output Files
------------

TOD Files (``FOLDER_TOD_OUTPUT``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One ``.npy`` file per observation day::

    tod_day_{day_index}.npy

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Property
     - Value
     - Notes
   * - Shape
     - ``(3, n_samples)``
     - Axis 0: Stokes component ``[I, Q, U]``. Axis 1: detector sample.
   * - dtype
     - ``float32``
     -
   * - Format
     - NumPy ``.npy``
     - Load with ``numpy.load('tod_day_N.npy')``.

Example
~~~~~~~

.. code-block:: python

   import numpy as np

   tod = np.load("tod_day_0.npy")
   tod_I = tod[0]   # shape (n_samples,)
   tod_Q = tod[1]
   tod_U = tod[2]
