Data Formats
============

This page documents all input and output file formats used by the pipeline.

Input Files
-----------

Sky Map (``path_to_map``)
~~~~~~~~~~~~~~~~~~~~~~~~~

A HEALPix FITS file readable by ``healpy.read_map``. Fields are indexed:

* **Field 0** — Stokes I (intensity)
* **Field 1** — Stokes Q (linear polarisation)
* **Field 2** — Stokes U (linear polarisation)

Only the fields listed in ``map_fields`` (default ``[0, 1, 2]``) are read and
allocated; the others are never loaded. All read fields must share the same
``nside`` and are loaded at the configured ``precision`` (``float32`` by
default). Any ``healpy``-compatible HEALPix FITS file (RING ordering) is
accepted.

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

Output Files
------------

The output format depends on the ``furax_export`` setting.

furax HDF5 Observations (``furax_export: true``, default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One TOAST HDF5 observation per day::

    obs_day_{day_index}.h5

Each observation bundles every configured detector (the focal plane from
``detectors``) into a ``(n_det, n_samples)`` detdata block, with timestamps
anchored at ``furax_export_t0``. The stored signal dtype follows ``precision``.
These files load directly into furax's ``MultiObservationMapMaker``.

NumPy TOD Files (``furax_export: false``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One ``.npy`` file per processing day and detector::

    tod_day_{day_index}.npy            # implicit boresight detector
    tod_day_{day_index}_{name}.npy     # per configured focal-plane detector

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Property
     - Value
     - Notes
   * - Shape
     - ``(3, n_samples)``
     - Axis 0: Stokes component ``[I, Q, U]`` (rows for components absent from
       ``map_fields`` are zero). Axis 1: detector sample.
   * - dtype
     - follows ``precision``
     - ``float32`` by default.
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
