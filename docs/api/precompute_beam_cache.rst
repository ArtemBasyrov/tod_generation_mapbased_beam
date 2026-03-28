precompute\_beam\_cache
=======================

One-time pre-computation script for beam rotation caches.

Run this script once before the main pipeline to precompute psi-roll
rotations for each beam file. The resulting ``.npz`` files allow the
runtime pipeline to skip one (or both) of the Rodrigues rotations per
sample.

Usage::

    python precompute_beam_cache.py [--n_psi 720] [--output_dir /path/to/cache/]
                                    [--no_offsets] [--config config.yaml]

Command-line arguments:

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Flag
     - Default
     - Description
   * - ``--n_psi``
     - ``720``
     - Number of psi bins. 720 gives 0.5° resolution. Increase for
       beams wider than ~5°.
   * - ``--output_dir``
     - ``FOLDER_BEAM``
     - Where to write ``.npz`` cache files. Defaults to the beam folder
       from the config.
   * - ``--no_offsets``
     - *(flag)*
     - Skip flat-sky angular-offset pre-computation. Use for beams wider
       than ~5° where the flat-sky approximation is invalid.
   * - ``--config``
     - ``config.yaml``
     - Path to the config file to read beam settings from.

.. automodule:: precompute_beam_cache
   :members: precompute_beam
   :undoc-members: False
   :show-inheritance: True
