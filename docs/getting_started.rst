Getting Started
===============

Installation
------------

Install the required Python packages::

    pip install -r requirements.txt

This installs ``numpy healpy pixell numba pyyaml psutil`` for TOD generation
and ``toast astropy h5py`` for the furax HDF5 export, which is on by default.
``psutil`` is required: the pipeline uses it to size the per-process memory
budget.

Quick Start
-----------

1. **Copy and edit the config**::

       cp config.yaml config_local.yaml
       $EDITOR config_local.yaml

   At minimum set:

   * ``FOLDER_SCAN`` — directory with ``theta_N.npy`` / ``phi_N.npy`` /
     ``psi_N.npy`` scan files.
   * ``FOLDER_TOD_OUTPUT`` — where the output files are written.
   * ``path_to_map`` — HEALPix FITS sky map.
   * ``FOLDER_BEAM`` and ``beam_file_I/Q/U`` — beam FITS files.  Only the
     entries for components listed in ``map_fields`` are required.

2. **(Optional) Calibrate the beam clustering** by setting
   ``clustering_calibration_enabled: true``.  The next run sweeps the
   clustering parameters, writes the chosen values back to the config and
   resets the flag.  See :doc:`beam_cluster_calibration`.

3. **Run the pipeline**::

       python sample_based_tod_generation_gridint.py

   On the first run the pipeline measures throughput over process count,
   Numba threads per worker and batch size, writes the best values to the
   config, and processes all days.  Later runs skip this calibration.
   ``python run_calibration.py`` runs the calibration alone, without
   generating TODs.

Running on HPC / SLURM
-----------------------

The pipeline detects SLURM, PBS, LSF and SGE allocations.  On a cluster it
uses the allocated CPUs and the full node memory, and the calibration finds
the best ``n_processes``, Numba threads and ``batch_size`` for them::

    #!/bin/bash
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=32
    #SBATCH --mem=128G

    python sample_based_tod_generation_gridint.py

On memory-constrained nodes the best process count is often *fewer* than the
allocated CPUs, and the calibration captures this.

Output Files
------------

With ``furax_export: true`` (the default) each day becomes one TOAST HDF5
observation for the furax mapmaker::

    FOLDER_TOD_OUTPUT/obs_day_0.h5
    FOLDER_TOD_OUTPUT/obs_day_1.h5
    ...
    FOLDER_TOD_OUTPUT/index.sqlite

With ``furax_export: false`` each day is the raw TOD instead::

    FOLDER_TOD_OUTPUT/tod_day_0.npy
    FOLDER_TOD_OUTPUT/tod_day_1.npy
    ...

Each ``.npy`` has shape ``(3, n_samples)`` with axis 0 the Stokes component
``[I, Q, U]``, and dtype set by ``precision``.  See :doc:`data_formats`.
