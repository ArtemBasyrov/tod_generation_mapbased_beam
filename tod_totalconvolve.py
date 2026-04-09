"""Sky-map interpolation via ducc0.sht.experimental.synthesis_general.

Replaces the HEALPix bilinear gather + accumulate step in
:func:`tod_core.beam_tod_batch` with NUFFT-based spherical harmonic
synthesis at arbitrary sky positions, accurate to a user-specified
tolerance and free of the flat noise floor produced by pixel-domain
methods.

Design
------
At startup (once per worker process):
  - T is converted to spin-0 alm via healpy.map2alm.
  - Q and U are decomposed jointly into spin-2 alm via healpy.map2alm_spin,
    preserving the local-frame reference needed for correct polar behaviour.
  - No Interpolator object is built, avoiding the O(σ·lmax³) grid-construction
    cost of ducc0.totalconvolve.

At runtime (per batch):
  - The full double-Rodrigues rotation is applied to obtain sky unit vectors.
  - All B×S positions are passed to synthesis_general() in one call for all
    Stokes components simultaneously (the NUFFT point-grid is shared).
  - The result is reshaped and accumulated via a matrix multiply.

Accuracy:
  ``epsilon=1e-6``  → ~6 significant digits (~0.0001 % RMS).
  ``epsilon=1e-10`` → ~10 significant digits.
  Errors are uniform across all ℓ — no flat noise floor at high ℓ.

Setup cost vs totalconvolve:
  Only 3 × map2alm — O(lmax² × nside × iter).
  No Interpolator construction (saves O(σ·lmax³) per component).
"""

import numpy as np
import healpy as hp

try:
    from ducc0.sht.experimental import synthesis_general as _synthesis_general
except ImportError as _e:
    raise ImportError(
        "ducc0 is required for beam_interp_method='totalconvolve'. "
        "Install with: pip install ducc0"
    ) from _e

_TWO_PI = 2.0 * np.pi


class TotalconvolveInterpolator:
    """Stateful sky-map interpolator backed by ducc0 NUFFT-based SHT synthesis.

    Build once per process via the worker initialiser or the single-process
    path in ``main()``.  Call :meth:`sample` repeatedly.

    T is treated as a spin-0 scalar field.  Q and U are treated as a spin-2
    pair: they are decomposed jointly via ``healpy.map2alm_spin`` into
    gradient/curl alm coefficients, and synthesised via
    ``synthesis_general(spin=2)``, which applies the parallactic-angle
    rotation factor e^{±2iα} at each evaluation point.  This eliminates the
    polar frame-rotation error that arises when Q/U are interpolated as
    independent scalars.

    Args:
        mp (list of array-like): ``[T, Q, U]`` HEALPix maps in RING ordering.
            float32 or float64; converted to float64 internally.
        lmax (int, optional): Harmonic band limit. Defaults to ``2 * nside``.
        epsilon (float): NUFFT accuracy target. ``1e-6`` gives ~6 significant
            digits. Lower values are more accurate but increase per-call cost.
        nthreads (int): ducc0 thread count.  Set to ``1`` in multiprocessing
            workers to avoid over-subscription; ``0`` uses all available cores.
        map2alm_iter (int): Number of Jacobi iterations for
            ``healpy.map2alm`` (T only). ``3`` is the healpy default.
            ``healpy.map2alm_spin`` does not support iterative refinement.
    """

    def __init__(self, mp, lmax=None, epsilon=1e-6, nthreads=1, map2alm_iter=3):
        nside = hp.get_nside(np.asarray(mp[0]))
        if lmax is None:
            lmax = 2 * nside

        self.lmax = lmax
        self.epsilon = epsilon
        self.nthreads = nthreads
        self.n_comp = len(mp)

        print(
            f"  [totalconvolve] Computing alm for {len(mp)} components: "
            f"nside={nside}, lmax={lmax}, epsilon={epsilon}"
        )

        # T: scalar field (spin-0).
        self._alm_T = hp.map2alm(
            np.asarray(mp[0], dtype=np.float64),
            lmax=lmax,
            iter=map2alm_iter,
        )

        # Q, U: spin-2 pair — decomposed jointly so that synthesis_general
        # can apply the local-frame rotation at each evaluation point.
        alm_g, alm_c = hp.map2alm_spin(
            [np.asarray(mp[1], dtype=np.float64), np.asarray(mp[2], dtype=np.float64)],
            spin=2,
            lmax=lmax,
        )
        self._alm_QU = np.array([alm_g, alm_c])  # (2, nalm) complex128

        print("  [totalconvolve] Ready.")

    def sample(self, theta, phi, comp_indices=None):
        """Evaluate Stokes components at arbitrary sky positions.

        T (comp 0) uses a spin-0 synthesis call.  Q (comp 1) and U (comp 2)
        are synthesised together in a single spin-2 call — the NUFFT point-
        grid is shared, and the result is cached so requesting both Q and U
        costs no more than requesting one.

        Args:
            theta (numpy.ndarray): Colatitude [rad], shape ``(N,)``.
            phi (numpy.ndarray): Longitude [rad] in ``[0, 2π)``, shape ``(N,)``.
            comp_indices (list of int, optional): Which components to evaluate
                (0=T, 1=Q, 2=U).  ``None`` evaluates all components.

        Returns:
            numpy.ndarray: Interpolated values, shape ``(C, N)`` float64.
        """
        if comp_indices is None:
            comp_indices = list(range(self.n_comp))

        theta = np.asarray(theta, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)

        # loc: (N, 2) — [colatitude, longitude] as required by synthesis_general
        loc = np.empty((len(theta), 2), dtype=np.float64)
        loc[:, 0] = theta
        loc[:, 1] = phi

        # T: spin-0 synthesis, evaluated only if requested.
        t_synth = None
        if 0 in comp_indices:
            t_synth = _synthesis_general(
                alm=self._alm_T[np.newaxis],  # (1, nalm)
                loc=loc,
                spin=0,
                lmax=self.lmax,
                epsilon=self.epsilon,
                nthreads=self.nthreads,
            )[0]  # (N,)

        # Q and U: always synthesised together in one spin-2 call.
        qu_synth = None
        if 1 in comp_indices or 2 in comp_indices:
            qu_synth = _synthesis_general(
                alm=self._alm_QU,  # (2, nalm)
                loc=loc,
                spin=2,
                lmax=self.lmax,
                epsilon=self.epsilon,
                nthreads=self.nthreads,
            )  # (2, N): [Q_local, U_local]

        lookup = {
            0: t_synth,
            1: qu_synth[0] if qu_synth is not None else None,
            2: qu_synth[1] if qu_synth is not None else None,
        }
        return np.array([lookup[i] for i in comp_indices])


def _gather_accum_totalconvolve(vec_rot, beam_vals, comp_indices, interp, tod):
    """Replace HEALPix gather + accumulate using ducc0 NUFFT synthesis.

    Converts rotated beam-pixel unit vectors to ``(θ, φ)``, calls
    :meth:`TotalconvolveInterpolator.sample` once with all B×S positions
    for all Stokes components, then multiplies by beam weights and
    accumulates into ``tod``.

    Args:
        vec_rot (numpy.ndarray): Rotated beam-pixel unit vectors in the sky
            frame, shape ``(B, S, 3)``, dtype float32.
        beam_vals (numpy.ndarray): Normalised beam pixel weights, shape
            ``(S,)``, dtype float32.
        comp_indices (list of int): Stokes component indices to accumulate.
        interp (TotalconvolveInterpolator): Pre-built interpolator.
        tod (dict): Mapping ``{comp_idx: (B,) float32}`` updated in place.
    """
    B, S = vec_rot.shape[:2]

    vf = vec_rot.reshape(-1, 3).astype(np.float64)  # (B*S, 3)
    r_xy = np.sqrt(vf[:, 0] ** 2 + vf[:, 1] ** 2)
    theta_sky = np.arctan2(r_xy, vf[:, 2])  # [0, π]
    phi_sky = np.arctan2(vf[:, 1], vf[:, 0])  # [-π, π]
    phi_sky += (phi_sky < 0.0) * _TWO_PI  # wrap to [0, 2π)

    # Single call for all components — (C, B*S) float64.
    vals = interp.sample(theta_sky, phi_sky, comp_indices)

    bv = beam_vals.astype(np.float64)
    for i, comp in enumerate(comp_indices):
        tod[comp] += (vals[i].reshape(B, S) @ bv).astype(np.float32)
