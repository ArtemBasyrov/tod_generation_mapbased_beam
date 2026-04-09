"""Tests for the tod_totalconvolve module.

Covers:
  TotalconvolveInterpolator    — construction, scalar accuracy, Q/U polar accuracy
  _gather_accum_totalconvolve  — shape, beam weighting, single-pixel identity

Can be run independently::

    pytest tests/test_tod_totalconvolve.py -v
    python tests/test_tod_totalconvolve.py
"""

import os
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Path / stub setup (mirrors the pattern used in other test files)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

for _mod_name in ["pixell", "pixell.enmap"]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

if "tod_io" not in sys.modules:
    sys.modules["tod_io"] = MagicMock()

import numpy as np
import numpy.testing as npt
import healpy as hp
import pytest
import ducc0.sht.experimental as _dsht

from tod_totalconvolve import (
    TotalconvolveInterpolator,
    _gather_accum_totalconvolve,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(0)
_NSIDE = 32
_LMAX = 2 * _NSIDE
_NPIX = hp.nside2npix(_NSIDE)


def _random_map(rng=_RNG):
    """Return a float32 HEALPix map with unit-variance Gaussian values."""
    return rng.standard_normal(_NPIX).astype(np.float32)


def _build_interp(mp=None, lmax=_LMAX, epsilon=1e-7):
    """Build a TotalconvolveInterpolator on the shared test maps."""
    if mp is None:
        mp = [_random_map() for _ in range(3)]
    return TotalconvolveInterpolator(mp, lmax=lmax, epsilon=epsilon, nthreads=1)


def _random_unit_vectors(n, rng=_RNG):
    """Return (n, 3) array of random unit vectors on the sphere."""
    v = rng.standard_normal((n, 3)).astype(np.float32)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return v


# ===========================================================================
# TestTotalconvolveInterpolator
# ===========================================================================


class TestTotalconvolveInterpolator:
    """Tests for TotalconvolveInterpolator."""

    # ── Construction ──────────────────────────────────────────────────────────

    def test_builds_without_error(self):
        """Constructor completes for a standard 3-component map."""
        interp = _build_interp()
        assert interp.n_comp == 3
        assert interp.lmax == _LMAX

    def test_lmax_default_is_2nside(self):
        """Default lmax is 2 * nside when not specified."""
        mp = [_random_map() for _ in range(3)]
        interp = TotalconvolveInterpolator(mp, nthreads=1)
        assert interp.lmax == 2 * _NSIDE

    def test_stores_epsilon(self):
        """Epsilon is stored as an attribute."""
        interp = _build_interp(epsilon=1e-5)
        assert interp.epsilon == 1e-5

    # ── sample() output shape ─────────────────────────────────────────────────

    def test_sample_shape_all_components(self):
        """sample() returns (n_comp, N) when comp_indices=None."""
        interp = _build_interp()
        N = 50
        theta = _RNG.uniform(0.1, np.pi - 0.1, N)
        phi = _RNG.uniform(0.0, 2 * np.pi, N)
        vals = interp.sample(theta, phi)
        assert vals.shape == (3, N)

    def test_sample_shape_subset(self):
        """sample() returns (len(comp_indices), N) for a subset."""
        interp = _build_interp()
        N = 20
        theta = _RNG.uniform(0.1, np.pi - 0.1, N)
        phi = _RNG.uniform(0.0, 2 * np.pi, N)
        vals = interp.sample(theta, phi, comp_indices=[1, 2])
        assert vals.shape == (2, N)

    def test_sample_dtype_float64(self):
        """sample() output is float64."""
        interp = _build_interp()
        theta = np.array([1.0, 1.5])
        phi = np.array([0.5, 1.0])
        vals = interp.sample(theta, phi)
        assert vals.dtype == np.float64

    # ── Accuracy ──────────────────────────────────────────────────────────────

    def test_scalar_accuracy_equator(self):
        """Interpolated T values match exact harmonic synthesis within epsilon.

        Strategy: build a bandlimited T map from known alm, construct the
        interpolator, and verify that sampling at equatorial positions matches
        ``ducc0.sht.experimental.synthesis_general`` (exact to machine precision)
        to better than 100 × epsilon.  The generous margin accounts for
        map2alm discretisation noise when reconstructing alm from a pixelised
        map at nside=64.
        """
        nside = 64
        lmax = 128
        rng = np.random.default_rng(1)

        alm_true = (
            rng.standard_normal(hp.Alm.getsize(lmax))
            + 1j * rng.standard_normal(hp.Alm.getsize(lmax))
        ) * 0.5
        m = hp.alm2map(alm_true.astype(np.complex128), nside=nside).astype(np.float32)

        epsilon = 1e-7
        interp = TotalconvolveInterpolator(
            [m, m, m], lmax=lmax, epsilon=epsilon, nthreads=1
        )

        n_pts = 100
        theta = np.full(n_pts, np.pi / 2.0) + rng.uniform(-0.3, 0.3, n_pts)
        phi = np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False)

        vals = interp.sample(theta, phi, comp_indices=[0])[0]

        # Reference: exact harmonic synthesis at the same positions.
        # We use the alm recovered from the pixelised map (same as the
        # interpolator uses internally) so that map2alm discretisation noise
        # cancels and the comparison isolates the NUFFT accuracy.
        alm_recovered = hp.map2alm(m.astype(np.float64), lmax=lmax, iter=3)
        loc = np.column_stack([theta, phi])
        ref = _dsht.synthesis_general(
            alm=alm_recovered.reshape(1, -1),
            spin=0,
            lmax=lmax,
            loc=loc,
            epsilon=1e-12,
            nthreads=1,
        )[0]

        rms = np.sqrt(np.mean((vals - ref) ** 2))
        map_std = float(np.std(m))
        assert rms / map_std < 100 * epsilon, (
            f"Equatorial RMS {rms / map_std:.2e} exceeds 100×epsilon={100 * epsilon:.2e}"
        )

    def test_polar_accuracy_comparable_to_equator(self):
        """Q/U spin-2 interpolation near the poles is not substantially worse than at equator.

        Uses a bandlimited spin-2 field (Q/U synthesised from known alm_E,
        alm_B).  The reference at each evaluation point is exact spin-2
        synthesis from the *recovered* alms (same alms the interpolator holds
        internally), so the comparison isolates NUFFT accuracy from
        map2alm_spin discretisation noise and the 100×epsilon bound applies
        cleanly.  The key scientific assertion is that the polar error does
        not exceed 5× the equatorial error.
        """
        nside = 64
        lmax = 64
        rng = np.random.default_rng(2)
        nalm = hp.Alm.getsize(lmax)

        # Bandlimited spin-2 field: l=0,1 must be zero for spin-2.
        alm_E = (rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)).astype(
            np.complex128
        ) * 0.5
        alm_B = (rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)).astype(
            np.complex128
        ) * 0.5
        for ell in range(2):
            for m in range(ell + 1):
                idx = hp.Alm.getidx(lmax, ell, m)
                alm_E[idx] = 0.0
                alm_B[idx] = 0.0

        alm_T_zero = np.zeros(nalm, dtype=np.complex128)
        _, q_map, u_map = hp.alm2map([alm_T_zero, alm_E, alm_B], nside=nside)

        epsilon = 1e-8
        t_map = np.zeros(hp.nside2npix(nside), dtype=np.float32)
        interp = TotalconvolveInterpolator(
            [t_map, q_map.astype(np.float32), u_map.astype(np.float32)],
            lmax=lmax,
            epsilon=epsilon,
            nthreads=1,
        )

        n_pts = 50

        def _rms_vs_recovered(theta, phi, comp, qu_idx):
            """NUFFT error vs exact synthesis from the same recovered alms."""
            vals = interp.sample(theta, phi, comp_indices=[comp])[0]
            loc = np.column_stack([theta, phi])
            ref = _dsht.synthesis_general(
                alm=interp._alm_QU,  # same alms as the interpolator
                spin=2,
                lmax=lmax,
                loc=loc,
                epsilon=1e-12,
                nthreads=1,
            )[qu_idx]  # 0 = Q, 1 = U
            return float(np.sqrt(np.mean((vals - ref) ** 2)))

        # Equatorial band: θ ∈ [80°, 100°]
        theta_eq = np.linspace(np.radians(80), np.radians(100), n_pts)
        phi_eq = np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False)
        rms_eq = _rms_vs_recovered(theta_eq, phi_eq, comp=1, qu_idx=0)

        # Polar cap: θ ∈ [2°, 10°]
        theta_pol = np.linspace(np.radians(2), np.radians(10), n_pts)
        phi_pol = np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False)
        rms_pol = _rms_vs_recovered(theta_pol, phi_pol, comp=1, qu_idx=0)

        map_std = float(np.std(q_map))
        rms_eq_rel = rms_eq / map_std
        rms_pol_rel = rms_pol / map_std

        assert rms_eq_rel < 100 * epsilon, f"Equatorial RMS {rms_eq_rel:.2e} too large"
        assert rms_pol_rel < 100 * epsilon, f"Polar RMS {rms_pol_rel:.2e} too large"
        assert rms_pol_rel < 5 * rms_eq_rel + 1e-12, (
            f"Polar RMS ({rms_pol_rel:.2e}) is >5× equatorial ({rms_eq_rel:.2e})"
        )

    def test_spin2_roundtrip_at_pixel_centers(self):
        """Convention check: map2alm_spin + synthesis_general(spin=2) round-trip.

        Builds a Q/U map from known spin-2 alms, constructs the interpolator,
        and samples at every HEALPix pixel center (including polar rings).
        The recovered values must match the original map to within 1%.

        A sign or phase convention mismatch between healpy.map2alm_spin and
        ducc0.synthesis_general(spin=2) would produce O(1) relative errors,
        so even this loose tolerance cleanly catches incorrect conventions.
        Polar pixels are included because that is where the local reference
        frame rotates most rapidly.
        """
        nside = 32
        lmax = 16  # lmax = nside/2: well below Nyquist, map2alm_spin very accurate
        rng = np.random.default_rng(7)
        nalm = hp.Alm.getsize(lmax)

        alm_E = (rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)).astype(
            np.complex128
        )
        alm_B = (rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)).astype(
            np.complex128
        )
        for ell in range(2):
            for m in range(ell + 1):
                idx = hp.Alm.getidx(lmax, ell, m)
                alm_E[idx] = 0.0
                alm_B[idx] = 0.0

        alm_T_zero = np.zeros(nalm, dtype=np.complex128)
        _, q_map, u_map = hp.alm2map([alm_T_zero, alm_E, alm_B], nside=nside)

        t_map = np.zeros(hp.nside2npix(nside), dtype=np.float32)
        interp = TotalconvolveInterpolator(
            [t_map, q_map.astype(np.float32), u_map.astype(np.float32)],
            lmax=lmax,
            epsilon=1e-9,
            nthreads=1,
        )

        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        vals = interp.sample(theta, phi, comp_indices=[1, 2])

        q_rms = float(np.sqrt(np.mean((vals[0] - q_map) ** 2)))
        u_rms = float(np.sqrt(np.mean((vals[1] - u_map) ** 2)))
        q_std = float(np.std(q_map))
        u_std = float(np.std(u_map))

        assert q_rms / q_std < 0.01, f"Q round-trip RMS {q_rms / q_std:.3e} > 1%"
        assert u_rms / u_std < 0.01, f"U round-trip RMS {u_rms / u_std:.3e} > 1%"

    def test_qu_synthesized_jointly(self):
        """Requesting Q alone, U alone, or both together gives the same values.

        Q and U are synthesised in a single spin-2 call; requesting a subset
        must not change the returned values.  Near-polar positions are
        included to exercise the spin-2 code path fully.
        """
        interp = _build_interp()
        N = 30
        rng = np.random.default_rng(11)
        # Mix equatorial and near-polar positions
        theta = np.concatenate(
            [
                rng.uniform(0.05, 0.3, N // 2),  # near north pole
                rng.uniform(np.pi - 0.3, np.pi - 0.05, N // 2),  # near south pole
            ]
        )
        phi = rng.uniform(0.0, 2 * np.pi, N)

        vals_Q_only = interp.sample(theta, phi, comp_indices=[1])[0]
        vals_U_only = interp.sample(theta, phi, comp_indices=[2])[0]
        vals_QU = interp.sample(theta, phi, comp_indices=[1, 2])

        npt.assert_array_equal(vals_QU[0], vals_Q_only)
        npt.assert_array_equal(vals_QU[1], vals_U_only)

    def test_single_position(self):
        """sample() works for a single position (N=1)."""
        interp = _build_interp()
        vals = interp.sample(np.array([1.0]), np.array([0.5]))
        assert vals.shape == (3, 1)

    def test_phi_wrap(self):
        """Positions near φ=0 and φ=2π give the same result (periodicity)."""
        interp = _build_interp()
        theta = np.array([1.2, 1.2])
        phi_near_zero = np.array([1e-8, 2 * np.pi - 1e-8])
        vals = interp.sample(theta, phi_near_zero)
        # Neighbouring φ values should give nearly identical T values
        npt.assert_allclose(vals[:, 0], vals[:, 1], atol=1e-4)


# ===========================================================================
# TestGatherAccumTotalconvolve
# ===========================================================================


class TestGatherAccumTotalconvolve:
    """Tests for _gather_accum_totalconvolve."""

    def _make_tod(self, comp_indices, B):
        return {c: np.zeros(B, dtype=np.float32) for c in comp_indices}

    # ── Output shape and type ─────────────────────────────────────────────────

    def test_output_shape(self):
        """tod arrays have shape (B,) after accumulation."""
        interp = _build_interp()
        B, S = 6, 15
        vec_rot = _random_unit_vectors(B * S).reshape(B, S, 3)
        bv = np.ones(S, dtype=np.float32) / S
        tod = self._make_tod([0, 1, 2], B)
        _gather_accum_totalconvolve(vec_rot, bv, [0, 1, 2], interp, tod)
        for c in [0, 1, 2]:
            assert tod[c].shape == (B,)

    def test_output_dtype(self):
        """Accumulated values are float32."""
        interp = _build_interp()
        B, S = 4, 8
        vec_rot = _random_unit_vectors(B * S).reshape(B, S, 3)
        bv = np.ones(S, dtype=np.float32) / S
        tod = self._make_tod([0], B)
        _gather_accum_totalconvolve(vec_rot, bv, [0], interp, tod)
        assert tod[0].dtype == np.float32

    def test_accumulates_in_place(self):
        """tod values from a second call are added to those from the first."""
        interp = _build_interp()
        B, S = 4, 8
        vec_rot = _random_unit_vectors(B * S).reshape(B, S, 3)
        bv = np.ones(S, dtype=np.float32) / S

        tod_once = self._make_tod([0], B)
        _gather_accum_totalconvolve(vec_rot, bv, [0], interp, tod_once)

        tod_twice = self._make_tod([0], B)
        _gather_accum_totalconvolve(vec_rot, bv, [0], interp, tod_twice)
        _gather_accum_totalconvolve(vec_rot, bv, [0], interp, tod_twice)

        npt.assert_allclose(tod_twice[0], 2.0 * tod_once[0], rtol=1e-5)

    # ── Beam weighting ────────────────────────────────────────────────────────

    def test_single_beam_pixel_identity(self):
        """With S=1 and beam_val=1, result equals direct sky sample at that position.

        The single beam pixel is placed exactly at the boresight (vec_rot[:, 0, :] =
        pointing direction), so the TOD value should match interp.sample() at
        those positions.
        """
        mp = [_random_map() for _ in range(3)]
        interp = TotalconvolveInterpolator(mp, lmax=_LMAX, epsilon=1e-8, nthreads=1)

        B = 10
        # S=1: one beam pixel per sample, centred on the pointing direction
        pointing = _random_unit_vectors(B)  # (B, 3) float32
        vec_rot = pointing[:, np.newaxis, :]  # (B, 1, 3)

        beam_vals = np.ones(1, dtype=np.float32)
        tod = self._make_tod([0], B)
        _gather_accum_totalconvolve(vec_rot, beam_vals, [0], interp, tod)

        # Reference: directly sample at the same positions
        vf = pointing.astype(np.float64)
        r_xy = np.sqrt(vf[:, 0] ** 2 + vf[:, 1] ** 2)
        theta = np.arctan2(r_xy, vf[:, 2])
        phi = np.arctan2(vf[:, 1], vf[:, 0])
        phi = np.where(phi < 0, phi + 2 * np.pi, phi)
        ref = interp.sample(theta, phi, comp_indices=[0])[0]

        npt.assert_allclose(tod[0], ref.astype(np.float32), rtol=1e-5)

    def test_uniform_beam_is_average(self):
        """Uniform beam weights sum to 1 — result is the mean of sky samples."""
        mp = [_random_map() for _ in range(3)]
        interp = TotalconvolveInterpolator(mp, lmax=_LMAX, epsilon=1e-8, nthreads=1)

        B, S = 3, 12
        vec_rot = _random_unit_vectors(B * S).reshape(B, S, 3)
        beam_vals = np.ones(S, dtype=np.float32) / S

        tod = self._make_tod([0], B)
        _gather_accum_totalconvolve(vec_rot, beam_vals, [0], interp, tod)

        # Compute reference: sample each position individually and average
        vf = vec_rot.reshape(-1, 3).astype(np.float64)
        r_xy = np.sqrt(vf[:, 0] ** 2 + vf[:, 1] ** 2)
        theta = np.arctan2(r_xy, vf[:, 2])
        phi = np.arctan2(vf[:, 1], vf[:, 0])
        phi = np.where(phi < 0, phi + 2 * np.pi, phi)
        sky_vals = interp.sample(theta, phi, comp_indices=[0])[0].reshape(B, S)
        ref = sky_vals.mean(axis=1).astype(np.float32)

        npt.assert_allclose(tod[0], ref, rtol=1e-5)

    def test_zero_beam_gives_zero_tod(self):
        """Zero beam weights produce zero TOD regardless of sky values."""
        interp = _build_interp()
        B, S = 5, 10
        vec_rot = _random_unit_vectors(B * S).reshape(B, S, 3)
        beam_vals = np.zeros(S, dtype=np.float32)
        tod = self._make_tod([0, 1], B)
        _gather_accum_totalconvolve(vec_rot, beam_vals, [0, 1], interp, tod)
        npt.assert_array_equal(tod[0], 0.0)
        npt.assert_array_equal(tod[1], 0.0)

    def test_partial_comp_indices(self):
        """Only specified components are accumulated; others remain zero."""
        interp = _build_interp()
        B, S = 4, 6
        vec_rot = _random_unit_vectors(B * S).reshape(B, S, 3)
        bv = np.ones(S, dtype=np.float32) / S

        # Accumulate only component 1
        tod = self._make_tod([0, 1, 2], B)
        _gather_accum_totalconvolve(vec_rot, bv, [1], interp, tod)

        npt.assert_array_equal(tod[0], 0.0)
        assert not np.all(tod[1] == 0.0)
        npt.assert_array_equal(tod[2], 0.0)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
