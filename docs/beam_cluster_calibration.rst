Beam Pixel Clustering & Calibration
=====================================

This page explains how the beam pixel clustering calibration works, defines
the two quality metrics it enforces, and gives guidance on choosing
``clustering_error_threshold`` and ``clustering_ellipticity_tolerance``.

Overview
--------

Spatial k-means clustering on the unit sphere reduces the number of effective
beam pixels before TOD generation.  Only the low-power *tail* of the beam is
clustered (controlled by ``beam_cluster_tail_fraction``); the bright main-lobe
pixels are kept pixel-exact.  The gain is a proportional speed-up in the
innermost Numba gather loops with a small, controllable accuracy loss.

Pixels are grouped in the frame where the beam's own second moment is
isotropic, so that the clustering narrows the beam without reshaping it.  See
:ref:`clustering_error_structure` for why that matters and
:ref:`whitened_assignment` for what it does.

.. note::

   Clustering is applied **only** to the TOD-generation path.  The beam
   transfer function B_ℓ must always be computed from the full, unclustered
   beam pixel set.  Legendre polynomial oscillations on scales ~π/ℓ are
   destroyed by pixel merging, so any B_ℓ computation must bypass this step.

.. _clustering_error_structure:

What the Clustering Error Is
-----------------------------

Clustering replaces each group of beam nodes by a single centroid.  Mass and
dipole are preserved exactly — the centroid *is* the weighted mean of its
members — so the whole leading error is a deficit in the beam's second moment,

.. math::

   \Delta M_{ij} \,=\, -\bar\Sigma_{ij} \,,\qquad
   \bar\Sigma_{ij} \,=\, \sum_k W_k \,\Sigma^{(k)}_{ij}

with :math:`\Sigma^{(k)}` the within-cluster covariance of cluster *k* and
:math:`W_k` its mass.  That deficit splits into two pieces that behave very
differently:

* the part **proportional to the beam's own second moment** :math:`M` is a
  uniform narrowing.  It moves :math:`b_\ell` but leaves the beam's
  ellipticity untouched, so it adds no systematic that was not already there.
* the remainder is **ellipticity added to the beam**, an :math:`m = \pm 2`
  change.  A symmetric beam manufactures it out of nothing, and it is what
  sources T→P leakage.

Only the second is dangerous, and ``B_ℓ`` cannot see it: ``B_ℓ`` depends on
each node only through its angular distance from the beam centre, so it is the
:math:`m = 0` harmonic alone and **no rearrangement of nodes at fixed radius
changes it**.  A clustering that reshapes the beam scores identically to one
that does not.  This is why the calibration enforces two bounds.

How the Calibration Works
--------------------------

When ``clustering_calibration_enabled: true`` is set, the pipeline sweeps a
``(tail_fraction × n_clusters)`` grid.  For each candidate pair it:

1. Clusters a copy of the beam pixels using the candidate parameters.
2. Computes the beam transfer function B_ℓ from the clustered centroids
   (power_cut = 1.0).
3. Computes the reference B_ℓ from the full unclustered beam (computed once,
   reused for all grid points).
4. Measures the relative RMS B_ℓ divergence — the :math:`m = 0` bound.
5. Measures the added ellipticity :math:`q_2` from the cluster labels — the
   :math:`m = \pm 2` bound.
6. Records the pixel-count speedup as ``S / K_out``.

The pair that achieves the highest speedup while staying inside **both** bounds
is written to the config.  If no pair qualifies, the pair with the **least
added ellipticity** is used with a warning — an :math:`m = \pm 2` error
reshapes the beam, where the :math:`m = 0` term only rescales its width.

No scan data or TOD generation is performed during calibration — both metrics
depend only on beam geometry, making the sweep fast.

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

**Why B_ℓ divergence?**

* **No scan data needed.** The metric is computed from beam geometry alone,
  so the calibration sweep is fast and can be run independently of the
  observation schedule.
* **Direct width fidelity.** B_ℓ controls how the azimuthally averaged beam
  couples to each angular scale of the sky.

.. warning::

   B_ℓ is the :math:`m = 0` harmonic and bounds the beam *width* only.  It
   cannot bound the ellipticity clustering adds, because no rearrangement of
   nodes at fixed radius changes it.  Measured on the SAT 90 GHz beam, a
   shape-preserving clustering scores 10 % *worse* on B_ℓ divergence than one
   that reshapes the beam, while committing 10× less added ellipticity.  The
   second metric below exists for exactly that reason.

Added-Ellipticity Metric
--------------------------

The shape error is measured **exactly on the sphere**, as the change the
clustering makes to the beam's own :math:`m = \pm2` multipoles:

.. math::

   b_{\ell,2} \,=\, \sum_i w_i\,\bar P_\ell^{\,2}(\cos\theta_i)\,e^{-2i\phi_i}\,,

with :math:`(\theta, \phi)` in the beam-centred frame and :math:`\bar P` the
normalised associated Legendre function.  This is the quantity the spin-2
response couples to, so a clustering that changes it has changed the effective
beam's ellipticity.  The reported error is the RMS over the sky band of
:math:`\lvert b_{\ell,2}[\text{clustered}] - b_{\ell,2}[\text{exact}]\rvert`,
both evaluated in the same frame (``tod_calibrate.beam_m2_multipoles``).

.. warning::

   The second-moment deficit :math:`\bar\Sigma` is **not** a sufficient
   criterion, and an earlier version of this gate used it.  :math:`\bar\Sigma`
   is only the :math:`k^2` Taylor coefficient of the transfer.  Above
   :math:`\ell \simeq 100` the centroid phases decohere, so the error is a
   phase-incoherent sum of per-cell covariances and stops following its own
   low-order expansion.

   Measured: a reweighting of the node weights drives
   :math:`\lvert\Delta M^{\rm shape}\rvert` from 4.7e-04 to 6.0e-14 — ten
   orders — while the :math:`m = \pm2` error does not move at any multipole.
   A gate built on :math:`\bar\Sigma` can therefore certify a configuration
   that leaks exactly as much as before.  See
   ``error_analytical/03_clustering.md`` §13.1.

Because the multipoles are computed in spherical harmonics on the beam-centred
frame, no tangent-plane chart enters and the :math:`\sigma^2/2` artifact that
the ``(RA, Dec)`` offset chart manufactures cannot contaminate the metric.

The table also reports the error as a fraction of the beam's *own*
:math:`m = \pm2`.  That is the number to weigh against a leakage budget.

.. note::

   For a symmetric beam that denominator is a noise floor rather than a
   signal, so the ratio column is not meaningful — correctly so, since an
   ellipticity created from nothing has no natural scale to be measured
   against.  Use ``clustering_ellipticity_tolerance: 1.0`` for such beams, and
   see *Symmetric Beams* below for removing the error outright.

Calibration Output Table
-------------------------

When the calibration runs it prints an ASCII table of the form::

   [clust_calib] B_ell (m=0) <= 1.0e-05
   [clust_calib] added ellipticity q2(l=2048) <= 1 x 3.37e-04 = 3.37e-04
                 (beam's own quadrupole q2_beam = 2.22e+00)
    tail%      K   K_out   speedup   B_ell div    q2 added   /q2_beam  status
   --------------------------------------------------------------------------
    15.0%    500    2421     14.42    1.97e-04    2.87e-03   1.29e-03  ✗ m=±2
    15.0%   1000    2921     11.95    2.06e-04    9.54e-04   4.31e-04  ✗ m=±2
    15.0%   2000    3921      8.91    2.41e-04    3.37e-04   1.52e-04  ✓
    15.0%   4000    5921      5.90    2.55e-04    8.87e-04   4.00e-04  ✗ m=±2
   --------------------------------------------------------------------------

   [clust_calib] Recommendation: tail_fraction=0.15, n_clusters=2000
     (speedup=8.91x, B_ell div=2.41e-04, q2 added=3.37e-04)

Columns:

* **tail%** — fraction of total beam power in the clustered tail.
* **K** — requested number of tail clusters.
* **K_out** — actual number of output pixels (``n_main + K_tail``).
* **speedup** — ratio ``S / K_out`` where ``S`` is the original pixel count.
* **B_ell div** — :math:`\varepsilon_{B_\ell}`, the :math:`m = 0` bound.
* **q2 added** — the ellipticity this pair adds to the beam.
* **/q2_beam** — that ellipticity as a fraction of the beam's own quadrupole.
* **status** — ✓ if both bounds hold; ``✗ m=0`` or ``✗ m=±2`` names which one
  failed.

.. note::

   Added ellipticity is **not** monotone in ``n_clusters``.  In the table above
   it bottoms at K = 2000 and rises again by 2.6× at K = 4000, because a large
   cluster budget forces near-degenerate cells: a cell holding two grid nodes
   has a rank-1 covariance locked to a grid direction, and such cells carry
   87 % of the manufactured quadrupole at K = 4000 while holding only 36 % of
   the width error.  More clusters buys width accuracy and can cost shape
   accuracy.

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

.. _ellipticity_tolerance_guidance:

Choosing ``clustering_ellipticity_tolerance``
-----------------------------------------------

This bound is **relative, not absolute**: it is a factor over the least added
ellipticity any point in the sweep achieves *for this beam*.

An absolute bound cannot be chosen before running.  What is achievable depends
on the beam, the beam-grid spacing and the cluster budget together, so there is
no value a user can know in advance — and setting one too low simply causes
every candidate to fail, dropping the calibration into its fallback.  The
calibrator measures the floor instead and expresses the tolerance against it.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Value
     - Behaviour
   * - ``1.0`` (default)
     - Keep only the least-ellipticity configuration.  Use this for symmetric
       or near-symmetric beams, where any added quadrupole is manufactured
       from nothing.
   * - ``2.0``
     - Accept up to twice the floor in exchange for speed.  Reasonable for a
       strongly asymmetric beam, where the added term perturbs an existing
       asymmetry rather than creating one.
   * - ``null``
     - Report the term without gating on it.  The columns still appear in the
       table.

Values below ``1.0`` raise a ``ValueError`` at config load: no configuration
can beat the sweep's own floor, so such a bound would reject every candidate
including the best one.

.. _whitened_assignment:

Shape-Preserving Assignment
-----------------------------

Pixels are grouped in the frame where the beam's second moment is isotropic,

.. math::

   W \,=\, \sqrt{\bar\lambda}\; M^{-1/2} \,,

so that k-means — which minimises an *isotropic* distortion — is optimising the
right functional.  Mapping back gives
:math:`\bar\Sigma = \sigma^2 M`, proportional to the beam's own second moment
by construction: the clustering narrows the beam without reshaping it.

Only the assignment is whitened.  Centroids remain the weighted mean of the
original vectors, so dipole preservation and the exact
:math:`\Delta M = -\bar\Sigma` identity are untouched.

Measured on the SAT 90 GHz beam at ``n_clusters=2000``, ``tail_fraction=0.15``:
added ellipticity falls from 3.8e-02 to 3.8e-03 — a factor of 10 — for +0.02 %
on the width term, with the deficit axis moving from −4.4° (the beam-grid axis)
to +65.5°, against the beam's own axis of +71.8°.

.. note::

   On a beam that is a function of angular distance alone the whitener is
   exactly the identity and the transform is skipped, so the partition is
   reproduced bit for bit.  Whitening re-aims the second-moment deficit onto
   the beam's own axes; a beam with no axes has nothing to re-aim onto.  Such
   a beam's residual ellipticity is grid-induced and is not addressed by this
   mechanism.

   Set ``beam_cluster_whiten: false`` (or ``whiten=False`` when calling
   :func:`beam_cluster.cluster_beam_pixels` directly) to recover a plain
   spherical k-means.  The calibration sweeps this axis, because whitening is
   shape-preserving only below the crossover cluster count
   :math:`K_\times` and whether it pays above that depends on the beam.

   ``beam_cluster_whiten`` is **ignored entirely** when ``beam_symmetric:
   true``: the quadrant path described below clusters in the original frame
   whatever the flag says, and the calibration collapses the ``whiten`` axis
   to a single leg rather than sweeping one that measures nothing.

Symmetric Beams — ``beam_symmetric``
--------------------------------------

Merging beam nodes makes each cell a small **anisotropic** smear, so the
effective beam comes out slightly elliptical even when the real one is round.
That manufactured :math:`m = \pm2` sources T→P leakage and, unlike the
interpolation kernel, is **not** in the mapmaker's forward model, so it
survives into the maps rather than being deconvolved.  On a symmetric beam it
*dominates* the beam's own :math:`m = \pm2` — measured at 3.5–181× of it.

With ``beam_symmetric: true`` the clustering groups **one quadrant** of the
beam and copies the partition onto the other three by 90° rotation (the group
:math:`C_4`).  Every cell then has three siblings at 90°, 180° and 270°.
Ellipticity is a two-fold pattern — rotating an ellipse by 90° exchanges its
axes — so the four contributions cancel:

.. math::

   \sum_{n=0}^{3} e^{2i(\phi + n\pi/2)} \,=\, 0 \,.

The cancellation is exact and independent of the cluster count, of
:math:`K/K_\times`, and of any threshold.  Node count is unchanged, so it is
free.

The two mechanisms are alternatives on the same axis, not a stack: with
``beam_symmetric: true`` the quadrant k-means runs in the original frame and
:ref:`whitened assignment <whitened_assignment>` never engages, whatever
``beam_cluster_whiten`` is set to.  Whitening re-aims the second-moment
deficit onto the beam's own axes, which a round beam does not have; the
:math:`C_4` construction cancels the deficit's :math:`m = \pm2` part outright
instead.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - beam
     - plain k-means
     - ``beam_symmetric: true``
   * - SAT 90 GHz symmetric (real, noisy)
     - 3.5–181× the beam's own
     - **0.004–0.71×**
   * - analytic round Gaussian (:math:`\varepsilon_{\rm eff}`)
     - 1.67e-04
     - **1.04e-07** (1610×)

.. warning::

   Only set this for a beam that really is symmetric.  The cancellation needs
   the beam's own weights to be 90°-invariant; on a genuinely asymmetric beam
   the construction both fails to help (0.1–10×, no consistent gain) and costs
   node reduction (5.0× instead of 8×).  The measured C4 asymmetry is printed
   when the beam loads — roughly 1e-4 for a symmetric beam and 1e-2 for an
   asymmetric one — but it is reported, not enforced.

   Asymmetric beams need no fix: their clustering :math:`m = \pm2` is
   1.6e-04 of the beam's own, four orders below the signal.

Two implementation details are load-bearing.  The rotation centre is the
beam-centre **pixel** (``beam_center_x`` / ``beam_center_y``, default
``H//2, W//2``), never the peak pixel — detector noise puts the peak a pixel
off, which inflates the measured asymmetry sixfold and destroys the gain.  And
beam maps must be **odd-sized** and centred on that pixel, or orbits do not
close: a 200×200 map loses its edge pixels to singletons and 8 % of the
reduction, where 201×201 closes every orbit.

Symmetrising the beam map itself is deliberately *not* offered here.  It
changes the beam values, which is a claim about the instrument rather than a
numerical choice, and applied to an asymmetric beam it silently deletes the
asymmetry the pipeline exists to measure.  Do it upstream of the pipeline if
you want it.
