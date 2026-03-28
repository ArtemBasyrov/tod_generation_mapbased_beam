Getting Started
===============

Installation
------------

Install the required Python packages::

    pip install numpy healpy pixell numba pyyaml psutil

``psutil`` is optional but recommended — it enables automatic CPU and memory
detection on both local machines and HPC clusters.

Quick Start
-----------

1. **Copy and edit the config**::

       cp config.yaml config_local.yaml
       $EDITOR config_local.yaml

   At minimum set:

   * ``FOLDER_SCAN`` — directory with ``theta_N.npy`` / ``phi_N.npy`` /
     ``psi_N.npy`` scan files.
   * ``FOLDER_TOD_OUTPUT`` — where output ``tod_day_N.npy`` files are written.
   * ``path_to_map`` — HEALPix FITS file containing I, Q, U.
   * ``FOLDER_BEAM`` and ``beam_file_I/Q/U`` — beam FITS files.

2. **(Optional but recommended) Pre-compute the beam rotation cache**::

       python precompute_beam_cache.py --n_psi 720

   This eliminates one or both Rodrigues rotations per sample at runtime.
   Then set ``beam_cache_dir`` in your config to the output directory.

3. **Run the pipeline**::

       python sample_based_tod_generation_gridint.py

   On first run the pipeline measures throughput at several batch sizes and
   process counts, writes the optimal values to the config, and processes all
   days. Subsequent runs skip calibration automatically.

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

One ``.npy`` file is written per observation day::

    FOLDER_TOD_OUTPUT/tod_day_0.npy
    FOLDER_TOD_OUTPUT/tod_day_1.npy
    ...

Each file has shape ``(3, n_samples)`` and dtype ``float32``.
Axis 0 is the Stokes component: ``[I, Q, U]``.
