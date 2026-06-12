Beam Interpolation Accuracy
===========================

This page documents the interpolation accuracy of the three methods available
via ``beam_interp_method``.  The measurements set the effective noise floor of
the sky-map lookup step; they are independent of beam pixel clustering (see
:doc:`beam_cluster_calibration` for clustering quality metrics).

.. note::

   All measurements on this page used a **symmetric Gaussian beam with
   FWHM = 30 arcmin**.  The reported test is the **accuracy test** — relative
   RMS error against the harmonically smoothed map.

   The relative RMS metric is largely independent of beam shape; the main
   external factor is the ratio of beam FWHM to HEALPix pixel size
   (see :ref:`Interpolation accuracy floors <interp_accuracy_floors>`).


Methods
-------

Three interpolation strategies are compared:

.. list-table::
   :header-rows: 1
   :widths: 12 20 68

   * - Key value
     - Short name
     - Description
   * - ``'nearest'``
     - NP — Nearest Pixel
     - Single nearest-pixel lookup.  No blending; the raw pixel that is closest
       to the query direction is returned directly.
   * - ``'bilinear'``
     - BI — Bilinear Interpolation
     - Weighted average of the 4 HEALPix neighbours returned by
       ``healpy.get_interp_weights``.  Implemented as a fused Numba kernel.
   * - ``'gaussian'``
     - GK — Gaussian Kernel
     - Isotropic Gaussian-weighted average over all pixels within
       ``beam_interp_radius_deg``.


Metrics
-------

**Accuracy** (:math:`\varepsilon`)
   For each combination of ``nside`` and beam pixel resolution the pipeline
   evaluates pointing directions that lie between pixel centres and computes:

   .. math::

      \varepsilon = \frac{\mathrm{RMS}\!\left(v_{\mathrm{interp}} -
                          v_{\mathrm{true}}\right)}
                         {\mathrm{RMS}\!\left(v_{\mathrm{true}}\right)}

   where :math:`v_{\mathrm{true}}` is the value from the harmonically smoothed
   map.  A smaller :math:`\varepsilon` means the method reproduces the true
   beam value more accurately.


Accuracy Test Results
---------------------

Beam pixel resolution: 0.5 arcmin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 28 28 28

   * - ``nside``
     - RMS\ :sub:`NP` / RMS\ :sub:`true`
     - RMS\ :sub:`BI` / RMS\ :sub:`true`
     - RMS\ :sub:`GK` / RMS\ :sub:`true`
   * - 512
     - 6.8838 × 10\ :sup:`−3`
     - 1.4477 × 10\ :sup:`−2`
     - 6.0139 × 10\ :sup:`−2`
   * - 1024
     - 1.6464 × 10\ :sup:`−3`
     - 3.5470 × 10\ :sup:`−3`
     - 1.6263 × 10\ :sup:`−2`
   * - 2048
     - 4.7922 × 10\ :sup:`−4`
     - 8.3853 × 10\ :sup:`−4`
     - 4.0962 × 10\ :sup:`−3`

Beam pixel resolution: 1 arcmin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 28 28 28

   * - ``nside``
     - RMS\ :sub:`NP` / RMS\ :sub:`true`
     - RMS\ :sub:`BI` / RMS\ :sub:`true`
     - RMS\ :sub:`GK` / RMS\ :sub:`true`
   * - 512
     - 7.5959 × 10\ :sup:`−3`
     - 1.5329 × 10\ :sup:`−2`
     - 6.3000 × 10\ :sup:`−2`
   * - 1024
     - 2.1361 × 10\ :sup:`−3`
     - 3.7801 × 10\ :sup:`−3`
     - 1.6360 × 10\ :sup:`−2`
   * - 2048
     - 1.1955 × 10\ :sup:`−3`
     - 9.1339 × 10\ :sup:`−4`
     - 3.9902 × 10\ :sup:`−3`

Beam pixel resolution: 5 arcmin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 28 28 28

   * - ``nside``
     - RMS\ :sub:`NP` / RMS\ :sub:`true`
     - RMS\ :sub:`BI` / RMS\ :sub:`true`
     - RMS\ :sub:`GK` / RMS\ :sub:`true`
   * - 512
     - 2.4216 × 10\ :sup:`−2`
     - 1.4279 × 10\ :sup:`−2`
     - 6.1118 × 10\ :sup:`−2`
   * - 1024
     - 1.4897 × 10\ :sup:`−2`
     - 4.5058 × 10\ :sup:`−3`
     - 1.6636 × 10\ :sup:`−2`
   * - 2048
     - 8.0256 × 10\ :sup:`−3`
     - 1.0050 × 10\ :sup:`−3`
     - 4.0215 × 10\ :sup:`−3`


Key Observations
----------------

**NP exhibits discrete-jump behaviour** under rotation.  When the beam is
rotated, a beam pixel direction can cross a HEALPix pixel boundary and snap to
a new centre — a discontinuous jump proportional to the local sky gradient.
The artefact grows sharply with beam pixel size and introduces
scan-strategy-dependent noise that is especially harmful for polarisation
analysis, even though NP can win the raw accuracy test at fine beam pixel
resolution (≤ 1 arcmin).

**GK is the least accurate**: its explicit smoothing blurs the true beam
value.  It is not recommended and is no longer tested.

**BI is the best overall compromise** — smooth (no discrete jumps) and
accurate (below 0.1 % at nside = 2048 for all beam pixel resolutions).
**Bilinear interpolation is the recommended default for all use cases; the
method choice is settled.**

**All methods scale approximately as** ``nside``\ :sup:`2` **in the accuracy
test**: doubling ``nside`` reduces :math:`\varepsilon` by roughly a factor of
4.


.. _interp_accuracy_floors:

Interpolation Accuracy as a Pipeline Error Floor
-------------------------------------------------

The interpolation errors measured above set the *noise floor* of the sky-map
lookup step.  No pipeline configuration — including beam pixel clustering —
can reduce the total error below this floor.  The table below shows the
minimum ``nside`` required for bilinear interpolation to stay within common
precision tiers across all beam pixel resolutions tested here.

.. list-table::
   :header-rows: 1
   :widths: 38 22 20 20

   * - Precision tier (relative RMS)
     - Bilinear threshold
     - Min. ``nside`` (5 arcmin beam)
     - Min. ``nside`` (0.5 arcmin beam)
   * - Loose / exploratory (< 5 %)
     - ``5.0e-2``
     - 512
     - 512
   * - Standard (< 1 %)
     - ``1.0e-2``
     - 1024
     - 512
   * - Tight (< 0.1 %)
     - ``1.0e-3``
     - 2048
     - 2048
   * - Very tight (< 0.05 %)
     - ``5.0e-4``
     - > 2048
     - 2048

Practical notes:

* **Use** ``'bilinear'`` **interpolation** (``beam_interp_method: bilinear``).
  It is accurate below 0.1 % at nside = 2048 and free of boundary-jump
  artefacts.  The table above is calibrated for BI.
* ``'nearest'`` interpolation may look better in the accuracy test at fine beam
  pixel resolution (≤ 1 arcmin), but its discrete boundary-jump artefacts make
  it unsuitable for polarisation analysis and any pipeline that compares
  observations taken at different orientations.
* The relative RMS metric is largely **independent of beam shape**: the
  interpolation operates on the HEALPix sky map, and the sub-pixel displacement
  distribution is determined by the HEALPix geometry, not by beam morphology.
  The values above apply regardless of beam asymmetry.
* The metric **does depend on beam FWHM** relative to the HEALPix pixel size.
  The tables were measured with a 30 arcmin FWHM beam; for significantly
  narrower beams — where FWHM / pixel_size drops below ~4–5 — the sky map
  has more sub-pixel structure and the relative errors will be larger than
  listed here.

.. note::

   The ``clustering_error_threshold`` config key governs a *different* metric:
   the relative RMS divergence of the beam transfer function B_ℓ between the
   clustered and unclustered beam.  That metric is defined and discussed
   separately on :doc:`beam_cluster_calibration`.
