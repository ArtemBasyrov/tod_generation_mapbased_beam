Getting Started
===============

Installation
------------

Install the required Python packages::

    pip install numpy healpy pixell numba pyyaml psutil   # core TOD generation
    pip install toast astropy h5py                         # furax HDF5 export (default-on)

``psutil`` is optional but recommended — it enables automatic CPU and memory
detection on both local machines and HPC clusters. The ``toast`` / ``astropy``
/ ``h5py`` stack is needed only for the default furax HDF5 export; set
``furax_export: false`` to write raw ``.npy`` files without it.

Quick Start
-----------

1. **Copy and edit the config**::

       cp config.yaml config_local.yaml
       $EDITOR config_local.yaml

   At minimum set:

   * ``FOLDER_SCAN`` — directory with ``theta_N.npy`` / ``phi_N.npy`` /
     ``psi_N.npy`` scan files.
   * ``FOLDER_TOD_OUTPUT`` — where output files (``obs_day_N.h5`` or
     ``tod_day_N.npy``) are written.
   * ``path_to_map`` — HEALPix FITS file containing I, Q, U.
   * ``FOLDER_BEAM`` and ``beam_file_I/Q/U`` — beam FITS files.

2. **Run the pipeline**::

       python sample_based_tod_generation_gridint.py

   On first run the pipeline measures throughput at several batch sizes and
   process counts, writes the optimal values to the config, and processes all
   days. Subsequent runs skip calibration automatically. By default it writes
   one furax-compatible TOAST HDF5 observation per day (``obs_day_N.h5``); set
   ``furax_export: false`` to write raw ``tod_day_N.npy`` files instead.

Running on HPC / SLURM
-----------------------

The pipeline is SLURM-aware. Set ``--cpus-per-task`` and ``--mem`` in your
job script; calibration will find the best ``n_processes`` and ``batch_size``
for the allocated resources::

    #!/bin/bash
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=32
    #SBATCH --mem=128G

    python sample_based_tod_generation_gridint.py

On memory-constrained nodes the optimal process count is often *fewer* than the
total allocated CPUs — the calibration captures this correctly.

Output Files
------------

By default one TOAST HDF5 observation is written per processing day (the
filename uses a *day* index by convention, but it can represent any batching
unit)::

    FOLDER_TOD_OUTPUT/obs_day_0.h5
    FOLDER_TOD_OUTPUT/obs_day_1.h5
    ...

Each observation holds a ``(n_det, n_samples)`` detdata block for the configured
focal plane. With ``furax_export: false`` the pipeline instead writes raw NumPy
files (``tod_day_N.npy`` for the boresight, ``tod_day_N_{name}.npy`` per
detector), each of shape ``(3, n_samples)`` with axis 0 the Stokes component
``[I, Q, U]``. See :doc:`data_formats` for full details.
