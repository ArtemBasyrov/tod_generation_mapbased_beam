Data Formats
============

This page documents all input and output file formats used by the pipeline.

Input Files
-----------

Sky Map (``path_to_map``)
~~~~~~~~~~~~~~~~~~~~~~~~~

A HEALPix FITS file in RING ordering, readable by ``healpy.read_map``:

* **Field 0** — Stokes I (intensity)
* **Field 1** — Stokes Q (linear polarisation)
* **Field 2** — Stokes U (linear polarisation)

Only the fields listed in ``map_fields`` are read, so a temperature-only file
works with ``map_fields: [0]``.  The loaded fields must share one ``nside``
and are cast to ``precision``.

Q and U must follow the HEALPix/COSMO polarisation convention, as written by
healpy and PySM.  An IAU-convention map has the opposite sign of U, and the
spin-2 transport would then rotate it the wrong way without raising any error.

Beam Files (``FOLDER_BEAM / beam_file_{I,Q,U}``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`pixell / enmap <https://pixell.readthedocs.io/en/latest/usage.html#usagepage>`_ FITS format (2-D CAR map). Requirements:

* Only the **first component** of the map is read.  Stokes beams that differ
  must therefore be stored in separate files; components that share a beam
  may point at the same file.
* RA and Dec are read from the WCS header and expressed as offsets from the
  beam-centre pixel.  That pixel is ``(H // 2, W // 2)`` for a map of shape
  ``(H, W)``, unless ``beam_center_x`` / ``beam_center_y`` are set.
* Use an odd-sized grid (e.g. 201 × 201) so the centre pixel is unambiguous.
  ``beam_symmetric: true`` requires an odd-sized grid with equal RA and Dec
  spacing.
* Values represent beam amplitude in linear units (not dB).
* Normalisation of the beam amplitude is **not required**.  The pipeline
  multiplies each pixel by its ``cos(dec)`` solid-angle Jacobian, keeps the
  pixels carrying ``power_fraction_threshold`` of the integrated power, drops
  zero-weight pixels, and normalises the weights to sum to one.

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
   psi_{day_index}.npy     # polarisation angle    [rad]

Each file is a 1-D array with one element per detector sample, in
``float32`` or ``float64``.  Files are opened as ``numpy`` memory-maps, and
each batch is converted to ``float64`` when it is read, whatever
``precision`` says.

The total number of processing units is inferred from the highest index found
among ``psi_*.npy`` files. The sampling rate is estimated as
``len(psi_0.npy) / 86400`` samples per second.

Output Files
------------

TOAST HDF5 (``furax_export: true``, default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One TOAST observation per day in ``FOLDER_TOD_OUTPUT``::

    obs_day_{day_index}.h5
    index.sqlite              # observation index written by SaveHDF5

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Property
     - Value
     - Notes
   * - Detectors
     - 1 (boresight)
     - Signal block of shape ``(1, n_samples)``.
   * - Signal
     - :math:`I + Q\cos 2\psi + U\sin 2\psi`
     - Q/U after any HWP modulation.  furax applies the ideal polariser's
       factor 1/2 itself when it reads the file.
   * - dtype
     - ``precision``
     - Must match furax's ``double_precision``.
   * - Time origin
     - ``furax_export_t0``
     - UTC time of sample 0 of day 0.
   * - HWP angle
     - stored when ``hwp_enabled``
     - The wrapped angle the TOD was modulated with, bit for bit.

No ``.npy`` file is written in this mode.  The export needs ``toast``, which
is imported at start-up so a missing install fails before any generation.

Raw TOD (``furax_export: false``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One ``.npy`` file per day::

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
       Rows for components not in ``map_fields`` are zero.
   * - dtype
     - ``precision``
     -
   * - Format
     - NumPy ``.npy``
     - Load with ``numpy.load('tod_day_N.npy')``.

.. code-block:: python

   import numpy as np

   tod = np.load("tod_day_0.npy")
   tod_I = tod[0]   # shape (n_samples,)
   tod_Q = tod[1]
   tod_U = tod[2]

``python tod_to_furax.py`` re-exports these files to TOAST HDF5 afterwards.
It writes to ``<FOLDER_TOD_OUTPUT>/furax_h5/`` unless ``--output`` is given,
and also takes ``--start-day``, ``--end-day``, ``--t0``, ``--split-per-day``
and ``--precision`` (default ``float64``).
