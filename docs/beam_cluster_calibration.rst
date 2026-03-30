Beam Pixel Clustering & Calibration
=====================================

This page explains how the beam pixel clustering calibration works, defines
the B_ℓ divergence quality metric, and gives guidance on choosing
``clustering_error_threshold``.

Overview
--------

Spatial k-means clustering on the unit sphere reduces the number of effective
beam pixels before TOD generation.  Only the low-power *tail* of the beam is
clustered (controlled by ``beam_cluster_tail_fraction``); the bright main-lobe
pixels are kept pixel-exact.  The gain is a proportional speed-up in the
innermost Numba gather loops with a small, controllable accuracy loss.

.. note::

   Clustering is applied **only** to the TOD-generation path.  The beam
   transfer function B_ℓ must always be computed from the full, unclustered
   beam pixel set.  Legendre polynomial oscillations on scales ~π/ℓ are
   destroyed by pixel merging, so any B_ℓ computation must bypass this step.

How the Calibration Works
--------------------------

When ``clustering_calibration_enabled: true`` is set, the pipeline sweeps a
``(tail_fraction × n_clusters)`` grid.  For each candidate pair it:

1. Clusters a copy of the beam pixels using the candidate parameters.
2. Computes the beam transfer function B_ℓ from the clustered centroids
   (power_cut = 1.0).
3. Computes the reference B_ℓ from the full unclustered beam (computed once,
   reused for all grid points).
4. Measures the relative RMS B_ℓ divergence (see below).
5. Records the pixel-count speedup as ``S / K_out``.

The pair that achieves the highest speedup while keeping B_ℓ divergence
below ``clustering_error_threshold`` is written to the config.  If no pair
qualifies, the pair with the lowest divergence is used with a warning.

No scan data or TOD generation is performed during calibration — the metric
depends only on beam geometry, making the sweep fast.

B_ℓ Divergence Metric
-----------------------

The quality of a ``(tail_fraction, n_clusters)`` pair is quantified by:

.. math::

   \varepsilon_{B_\ell} =
       \frac{\mathrm{RMS}_\ell\!\left(B_\ell^{\mathrm{clust}} -
                                      B_\ell^{\mathrm{ref}}\right)}
            {\mathrm{RMS}_\ell\!\left(B_\ell^{\mathrm{ref}}\right)}

where:

* :math:`B_\ell^{\mathrm{ref}}` is the beam transfer function computed from
  the full unclustered pixel set with ``power_cut = 1.0``.
* :math:`B_\ell^{\mathrm{clust}}` is the beam transfer function computed from
  the centroid pixels produced by the candidate pair.
* The RMS is taken over multipoles :math:`\ell = 0 \ldots \ell_{\max}`, where
  :math:`\ell_{\max}` defaults to ``2 × nside`` of the sky map (or 500 if no
  map is available).

**Why B_ℓ divergence rather than TOD error?**

* **No scan data needed.** The metric is computed from beam geometry alone,
  so the calibration sweep is fast and can be run independently of the
  observation schedule.
* **Direct beam fidelity.** B_ℓ controls how the beam couples to each angular
  scale of the sky.  A clustering that faithfully reproduces B_ℓ will also
  reproduce the TOD accurately, because TOD errors ultimately arise from
  beam-shape distortions that are captured in B_ℓ.

Calibration Output Table
-------------------------

When the calibration runs it prints an ASCII table of the form::

   [clust_calib] error_threshold=1.0e-05
    tail%      K   K_out   speedup   B_ell div  status
   --------------------------------------------------------
     0.5%     10      10      1.00   1.23e-07  ✓
     0.5%     20      20      1.00   1.23e-07  ✓
     ...
     5.0%    500     487      2.63   8.41e-06  ✓
     5.0%   1000     912      3.11   3.92e-06  ✓
   --------------------------------------------------------

   [clust_calib] Recommendation: tail_fraction=0.05, n_clusters=1000
     (speedup=3.11x, B_ell div=3.92e-06)

Columns:

* **tail%** — fraction of total beam power in the clustered tail.
* **K** — requested number of tail clusters.
* **K_out** — actual number of output pixels (``n_main + K_tail``).
* **speedup** — ratio ``S / K_out`` where ``S`` is the original pixel count.
* **B_ell div** — :math:`\varepsilon_{B_\ell}` for this pair.
* **status** — ✓ if ``B_ell div ≤ clustering_error_threshold``, ✗ otherwise.

.. _clustering_error_threshold_guidance:

Choosing ``clustering_error_threshold``
-----------------------------------------

The threshold controls the strictness of the B_ℓ fidelity requirement.
Lower values preserve more of the beam shape but allow less aggressive
clustering (smaller speedup).

.. list-table::
   :header-rows: 1
   :widths: 38 22 40

   * - Precision tier
     - ``clustering_error_threshold``
     - Notes
   * - Conservative (default)
     - ``1.0e-5``
     - Safe for science-grade pipelines.  Typical speedup 2–4× for a
       5 % tail with 500–1000 clusters.
   * - Moderate
     - ``1.0e-4``
     - Suitable for survey-speed optimisation where a small B_ℓ bias
       is acceptable.  Allows more aggressive tail truncation.
   * - Loose / exploratory
     - ``1.0e-3``
     - Useful for rapid prototyping.  The B_ℓ shape may be visibly
       distorted at high ℓ.

Practical notes:

* **Start with the default** (``1.0e-5``) and inspect the calibration table.
  If all grid points pass, you can relax the threshold to gain more speedup, 
  it's an interplay between the noise level and the accuracy of B_ℓ characterization;
  if none pass, tighten the ``tail_fraction`` range or increase ``n_clusters``.
* Interpolation errors (see :doc:`beam_interpolation_accuracy`) set a separate
  noise floor on sky-map lookups and are independent of this threshold.  There
  is no strict relationship between the two metrics; they should be chosen
  independently.
* B_ℓ divergence is **independent of scan strategy** — it depends
  only on beam geometry and the clustering parameters.
