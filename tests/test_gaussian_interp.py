"""
Tests for Gaussian kernel interpolation in tod_core.

Covers:
  _angular_distance           — great-circle distance helper
  _gaussian_interp_accum      — Gaussian kernel interpolation + beam accumulation
  beam_tod_batch (interp_mode='gaussian')
      - original path (double Rodrigues, no cache)
      - single-Rodrigues path (vec_rolled / psi_grid cache)
      - flat-sky path (vec_rolled / psi_grid / dtheta / dphi cache)

Can be run independently:
    pytest tests/test_gaussian_interp.py -v
    python tests/test_gaussian_interp.py
"""

import os
import sys
import math
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Standalone imports (conftest.py handles this automatically under pytest)
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

from tod_core import (
    _angular_distance,
    _gaussian_interp_accum,
    _gaussian_interp_accum_jit,
    beam_tod_batch,
    precompute_rotation_vector_batch,
)

# ---------------------------------------------------------------------------
# Shared RNG
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(42)

# ===========================================================================
# Helpers shared across test classes
# ===========================================================================

def _build_data(S=30, nside=32):
    """Synthetic beam data dict with S pixels near the north pole."""
    rng = np.random.default_rng(99)
    theta_beam = rng.uniform(0.0, 0.05, S)
    phi_beam   = rng.uniform(0, 2 * np.pi, S)
    vec_orig   = np.stack([
        np.sin(theta_beam) * np.cos(phi_beam),
        np.sin(theta_beam) * np.sin(phi_beam),
        np.cos(theta_beam),
    ], axis=-1).astype(np.float32)
    beam_vals  = rng.uniform(0.5, 1.5, S).astype(np.float32)
    beam_vals /= beam_vals.sum()
    return {
        "vec_orig":     vec_orig,
        "beam_vals":    beam_vals,
        "comp_indices": ["I", "Q", "U"],
        "mp_stacked":   None,
    }


def _build_scan(B=8, N=201):
    """B scan directions near the equator."""
    rng = np.random.default_rng(77)
    ra  = np.zeros((N, N))
    dec = np.zeros((N, N))
    phi_batch   = rng.uniform(0, 0.04, B)
    theta_batch = rng.uniform(np.pi / 2 - 0.04, np.pi / 2, B)
    rot_vecs, betas = precompute_rotation_vector_batch(
        ra, dec, phi_batch, theta_batch, center_idx=(100, 100)
    )
    return phi_batch, theta_batch, -betas, rot_vecs


def _constant_maps(nside=32):
    npix = hp.nside2npix(nside)
    return {c: np.ones(npix, dtype=np.float32) for c in ["I", "Q", "U"]}


def _mp_stacked(mp, comp_indices):
    return np.stack([mp[c] for c in comp_indices]).astype(np.float32)


# ===========================================================================
# TestAngularDistance
# ===========================================================================

class TestAngularDistance:
    """Tests for tod_core._angular_distance."""

    def test_zero_distance_collocated(self):
        """Identical directions give distance 0."""
        th = 0.5
        ph = 1.2
        dist = _angular_distance(th, ph, np.array([th]), np.array([ph]))
        npt.assert_allclose(dist, [0.0], atol=1e-12)

    def test_antipodal_distance_is_pi(self):
        """Antipodal points give distance π."""
        dist = _angular_distance(0.0, 0.0, np.array([np.pi]), np.array([0.0]))
        npt.assert_allclose(dist, [np.pi], atol=1e-12)

    def test_known_90deg_separation(self):
        """North pole and equator are π/2 apart."""
        dist = _angular_distance(0.0, 0.0, np.array([np.pi / 2]), np.array([0.0]))
        npt.assert_allclose(dist, [np.pi / 2], atol=1e-10)

    def test_vectorized_output_shape(self):
        """Output has the same length as the neighbour arrays."""
        th = np.pi / 4
        ph = 0.3
        n  = 50
        rng = np.random.default_rng(1)
        th_arr = rng.uniform(0, np.pi, n)
        ph_arr = rng.uniform(0, 2 * np.pi, n)
        dist = _angular_distance(th, ph, th_arr, ph_arr)
        assert dist.shape == (n,)

    def test_distance_in_zero_to_pi(self):
        """All output distances are in [0, π]."""
        rng = np.random.default_rng(2)
        n   = 100
        th1 = rng.uniform(0, np.pi)
        ph1 = rng.uniform(0, 2 * np.pi)
        th2 = rng.uniform(0, np.pi, n)
        ph2 = rng.uniform(0, 2 * np.pi, n)
        dist = _angular_distance(th1, ph1, th2, ph2)
        assert np.all(dist >= 0.0), "Distances must be non-negative"
        assert np.all(dist <= np.pi + 1e-12), "Distances must not exceed π"

    def test_symmetry(self):
        """d(a, b) == d(b, a)."""
        rng = np.random.default_rng(3)
        n   = 20
        th1, ph1 = rng.uniform(0, np.pi, n), rng.uniform(0, 2 * np.pi, n)
        th2, ph2 = rng.uniform(0, np.pi, n), rng.uniform(0, 2 * np.pi, n)
        d_ab = _angular_distance(th1[0], ph1[0], th2, ph2)
        d_ba = _angular_distance(th2[0], ph2[0], th1, ph1)
        # Vectorised: check one pair each direction
        npt.assert_allclose(
            _angular_distance(th1[0], ph1[0], th2[:1], ph2[:1]),
            _angular_distance(th2[0], ph2[0], th1[:1], ph1[:1]),
            atol=1e-12,
        )


# ===========================================================================
# TestGaussianInterpAccum
# ===========================================================================

class TestGaussianInterpAccum:
    """Tests for tod_core._gaussian_interp_accum."""

    @staticmethod
    def _build_inputs(B, Sc, nside, mp_val=1.0, rng=None):
        """
        Build minimal inputs for _gaussian_interp_accum pointing near the equator.
        Returns theta_flat, phi_flat (B*Sc,), mp_stacked (1, npix), beam_vals (Sc,),
        tod_arr (1, B).
        """
        if rng is None:
            rng = np.random.default_rng(0)
        npix = hp.nside2npix(nside)
        theta_flat = rng.uniform(np.pi / 2 - 0.1, np.pi / 2 + 0.1, B * Sc)
        phi_flat   = rng.uniform(0, 2 * np.pi, B * Sc)
        mp_stacked = np.full((1, npix), mp_val, dtype=np.float32)
        beam_vals  = np.ones(Sc, dtype=np.float32) / Sc
        tod_arr    = np.zeros((1, B), dtype=np.float64)
        return theta_flat, phi_flat, mp_stacked, beam_vals, tod_arr

    def test_constant_map_gives_constant(self):
        """Gaussian interp on a constant map returns that constant for every (c, b)."""
        nside = 32
        B, Sc = 4, 8
        val   = 3.7
        theta_flat, phi_flat, mp_stacked, beam_vals, tod_arr = \
            self._build_inputs(B, Sc, nside, mp_val=val)
        _gaussian_interp_accum(
            theta_flat, phi_flat, B, Sc,
            nside, mp_stacked, beam_vals, tod_arr,
            sigma_deg=2.0, radius_deg=6.0,
        )
        # beam is normalised (sum=1), so tod should equal val
        npt.assert_allclose(tod_arr[0], np.full(B, val), atol=1e-3)

    def test_accumulates_inplace(self):
        """Calling twice with same inputs doubles the result."""
        nside = 32
        B, Sc = 3, 5
        theta_flat, phi_flat, mp_stacked, beam_vals, tod_arr = \
            self._build_inputs(B, Sc, nside, mp_val=1.0)
        _gaussian_interp_accum(
            theta_flat, phi_flat, B, Sc,
            nside, mp_stacked, beam_vals, tod_arr,
            sigma_deg=2.0, radius_deg=6.0,
        )
        first = tod_arr.copy()
        _gaussian_interp_accum(
            theta_flat, phi_flat, B, Sc,
            nside, mp_stacked, beam_vals, tod_arr,
            sigma_deg=2.0, radius_deg=6.0,
        )
        npt.assert_allclose(tod_arr, 2 * first, atol=1e-10)

    def test_zero_beam_vals_gives_zero(self):
        """Zero beam weights produce no contribution."""
        nside = 32
        B, Sc = 3, 5
        theta_flat, phi_flat, mp_stacked, beam_vals, tod_arr = \
            self._build_inputs(B, Sc, nside, mp_val=5.0)
        beam_vals[:] = 0.0
        _gaussian_interp_accum(
            theta_flat, phi_flat, B, Sc,
            nside, mp_stacked, beam_vals, tod_arr,
            sigma_deg=2.0, radius_deg=6.0,
        )
        npt.assert_allclose(tod_arr, np.zeros((1, B)), atol=1e-10)

    def test_output_shape_unchanged(self):
        """tod_arr shape (C, B) is not modified by the call."""
        nside = 32
        C, B, Sc = 2, 5, 7
        rng = np.random.default_rng(4)
        npix       = hp.nside2npix(nside)
        theta_flat = rng.uniform(0.1, np.pi - 0.1, B * Sc)
        phi_flat   = rng.uniform(0, 2 * np.pi, B * Sc)
        mp_stacked = np.ones((C, npix), dtype=np.float32)
        beam_vals  = np.ones(Sc, dtype=np.float32) / Sc
        tod_arr    = np.zeros((C, B), dtype=np.float64)
        _gaussian_interp_accum(
            theta_flat, phi_flat, B, Sc,
            nside, mp_stacked, beam_vals, tod_arr,
            sigma_deg=1.5, radius_deg=4.5,
        )
        assert tod_arr.shape == (C, B)

    def test_wider_sigma_smooths_point_source(self):
        """
        A point-source map: querying exactly at the source with a narrow sigma
        gives a higher value than with a wide sigma that dilutes the source.
        """
        nside = 64
        npix  = hp.nside2npix(nside)
        # Place a point source at the north-pole pixel
        source_pix = hp.ang2pix(nside, 0.01, 0.0)
        mp_stacked = np.zeros((1, npix), dtype=np.float32)
        mp_stacked[0, source_pix] = 1.0

        # Query direction exactly at the source
        th_src, ph_src = hp.pix2ang(nside, source_pix)
        theta_flat = np.array([th_src])
        phi_flat   = np.array([ph_src])
        beam_vals  = np.array([1.0], dtype=np.float32)

        tod_narrow = np.zeros((1, 1), dtype=np.float64)
        tod_wide   = np.zeros((1, 1), dtype=np.float64)

        _gaussian_interp_accum(
            theta_flat, phi_flat, 1, 1,
            nside, mp_stacked, beam_vals, tod_narrow,
            sigma_deg=0.05, radius_deg=0.15,
        )
        _gaussian_interp_accum(
            theta_flat, phi_flat, 1, 1,
            nside, mp_stacked, beam_vals, tod_wide,
            sigma_deg=2.0, radius_deg=6.0,
        )
        assert tod_narrow[0, 0] > tod_wide[0, 0], (
            "Narrow sigma should give higher value at the source than wide sigma"
        )

    def test_multi_component_map(self):
        """Multiple components are interpolated independently and correctly."""
        nside = 32
        C, B, Sc = 3, 4, 6
        npix      = hp.nside2npix(nside)
        rng       = np.random.default_rng(5)
        theta_flat = rng.uniform(np.pi / 2 - 0.1, np.pi / 2 + 0.1, B * Sc)
        phi_flat   = rng.uniform(0, 2 * np.pi, B * Sc)
        # Each component has a different constant value
        values     = np.array([1.0, 2.0, 3.0])
        mp_stacked = np.stack([
            np.full(npix, v, dtype=np.float32) for v in values
        ])
        beam_vals  = np.ones(Sc, dtype=np.float32) / Sc
        tod_arr    = np.zeros((C, B), dtype=np.float64)
        _gaussian_interp_accum(
            theta_flat, phi_flat, B, Sc,
            nside, mp_stacked, beam_vals, tod_arr,
            sigma_deg=2.0, radius_deg=6.0,
        )
        for c, v in enumerate(values):
            npt.assert_allclose(tod_arr[c], np.full(B, v), atol=1e-3,
                                err_msg=f"Component {c} (value={v}) failed")


# ===========================================================================
# TestBeamTodBatchGaussianOriginalPath
# ===========================================================================

class TestBeamTodBatchGaussianOriginalPath:
    """
    Tests for beam_tod_batch with interp_mode='gaussian' on the original
    (double Rodrigues, no cache) path.
    """

    def test_output_keys_shape_dtype(self):
        """Gaussian mode returns the same keys, shape (B,), dtype float32 as bilinear."""
        nside = 32
        B     = 5
        data  = _build_data(S=30, nside=nside)
        data["mp_stacked"] = _mp_stacked(_constant_maps(nside), data["comp_indices"])
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)

        tod = beam_tod_batch(
            nside, _constant_maps(nside), data,
            rot_vecs, phi_b, theta_b, psis_b,
            interp_mode="gaussian",
        )
        assert set(tod.keys()) == set(data["comp_indices"])
        for comp in data["comp_indices"]:
            assert tod[comp].shape == (B,)
            assert tod[comp].dtype == np.float32

    def test_constant_map_gives_ones(self):
        """Constant (all-ones) sky map with normalised beam gives tod ≈ 1.0."""
        nside = 32
        B     = 8
        data  = _build_data(S=30, nside=nside)
        data["mp_stacked"] = _mp_stacked(_constant_maps(nside), data["comp_indices"])
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)

        tod = beam_tod_batch(
            nside, _constant_maps(nside), data,
            rot_vecs, phi_b, theta_b, psis_b,
            interp_mode="gaussian",
        )
        for comp in data["comp_indices"]:
            npt.assert_allclose(tod[comp], np.ones(B), atol=1e-3,
                                err_msg=f"Constant-map TOD not ≈ 1 for comp={comp}")

    def test_gaussian_vs_bilinear_constant_map(self):
        """On a constant map Gaussian and bilinear must agree exactly (both give 1.0)."""
        nside = 32
        B     = 6
        data  = _build_data(S=30, nside=nside)
        mp    = _constant_maps(nside)
        data["mp_stacked"] = _mp_stacked(mp, data["comp_indices"])
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)

        tod_bi = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                                interp_mode="bilinear")
        tod_ga = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                                interp_mode="gaussian")
        for comp in data["comp_indices"]:
            npt.assert_allclose(tod_ga[comp], tod_bi[comp], atol=1e-3)

    def test_gaussian_vs_bilinear_smooth_map(self):
        """
        On a spatially smooth map Gaussian and bilinear agree to within 1e-2.
        Both are valid interpolators; on smooth data the difference is at the
        sub-percent level.
        """
        nside = 64
        B     = 8
        rng   = np.random.default_rng(10)
        npix  = hp.nside2npix(nside)
        # Smooth map: just a low-amplitude sinusoidal modulation
        pix_theta, pix_phi = hp.pix2ang(nside, np.arange(npix))
        smooth_map = (1.0 + 0.05 * np.sin(pix_phi)).astype(np.float32)
        mp = {c: smooth_map.copy() for c in ["I", "Q", "U"]}

        data = _build_data(S=20, nside=nside)
        data["mp_stacked"] = _mp_stacked(mp, data["comp_indices"])
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)

        tod_bi = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                                interp_mode="bilinear")
        tod_ga = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                                interp_mode="gaussian",
                                sigma_deg=0.05, radius_deg=0.15)
        for comp in data["comp_indices"]:
            npt.assert_allclose(tod_ga[comp], tod_bi[comp], atol=1e-2,
                                err_msg=f"Gaussian/bilinear disagree on smooth map for {comp}")

    def test_sigma_affects_result(self):
        """Different sigma_deg values produce different TOD on a non-constant map."""
        nside = 64
        B     = 4
        npix  = hp.nside2npix(nside)
        rng   = np.random.default_rng(11)
        pix_theta, pix_phi = hp.pix2ang(nside, np.arange(npix))
        # Map with a sharp feature so sigma matters
        mp_arr = (1.0 + 0.5 * np.sin(4 * pix_phi)).astype(np.float32)
        mp = {c: mp_arr.copy() for c in ["I", "Q", "U"]}

        data = _build_data(S=20, nside=nside)
        data["mp_stacked"] = _mp_stacked(mp, data["comp_indices"])
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)

        tod_narrow = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                                    interp_mode="gaussian",
                                    sigma_deg=0.05, radius_deg=0.15)
        tod_wide   = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                                    interp_mode="gaussian",
                                    sigma_deg=1.0, radius_deg=3.0)

        # At least one component must differ; the sharp map ensures they do
        any_diff = False
        for comp in data["comp_indices"]:
            if not np.allclose(tod_narrow[comp], tod_wide[comp], atol=1e-4):
                any_diff = True
                break
        assert any_diff, "Different sigma_deg values should produce different TOD on a non-flat map"

    def test_default_sigma_radius(self):
        """Calling with sigma_deg=None and radius_deg=None runs without error."""
        nside = 32
        B     = 4
        data  = _build_data(S=20, nside=nside)
        data["mp_stacked"] = _mp_stacked(_constant_maps(nside), data["comp_indices"])
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)
        mp = _constant_maps(nside)

        tod = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                             interp_mode="gaussian",
                             sigma_deg=None, radius_deg=None)
        for comp in data["comp_indices"]:
            assert np.all(np.isfinite(tod[comp])), f"NaN/Inf in tod[{comp}]"

    def test_bilinear_mode_unchanged(self):
        """Passing interp_mode='bilinear' explicitly gives identical results to the default."""
        nside = 32
        B     = 6
        data  = _build_data(S=30, nside=nside)
        data["mp_stacked"] = _mp_stacked(_constant_maps(nside), data["comp_indices"])
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)
        mp = _constant_maps(nside)

        tod_default  = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b)
        tod_explicit = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                                      interp_mode="bilinear")
        for comp in data["comp_indices"]:
            npt.assert_array_equal(tod_default[comp], tod_explicit[comp])


# ===========================================================================
# TestBeamTodBatchGaussianCachedPaths
# ===========================================================================

class TestBeamTodBatchGaussianCachedPaths:
    """
    Tests for beam_tod_batch with interp_mode='gaussian' on the single-Rodrigues
    (vec_rolled / psi_grid) and flat-sky (+ dtheta / dphi) cached paths.
    """

    @staticmethod
    def _build_data_with_cache(S=30, N_psi=8, nside=32, include_flatsky=False):
        """
        Build a data dict that includes precomputed cache arrays so that
        beam_tod_batch takes the single-Rodrigues or flat-sky path.

        vec_rolled is built by applying the polarisation roll at each psi bin
        to the same vec_orig used in the no-cache case (simplified: just
        copy vec_orig for every bin so the roll is identity-like).
        """
        base = _build_data(S=S, nside=nside)
        rng  = np.random.default_rng(20)

        # psi_grid: N_psi evenly spaced angles in [0, 2π)
        psi_grid = np.linspace(0, 2 * np.pi, N_psi, endpoint=False).astype(np.float32)

        # vec_rolled: (N_psi, S, 3) — use the same vec_orig for every psi bin
        # (the pol-roll at a given bin is baked in; here we approximate it as
        # identity for the purpose of testing the dispatch path).
        vec_rolled = np.tile(
            base["vec_orig"][np.newaxis], (N_psi, 1, 1)
        ).astype(np.float32)

        mp = _constant_maps(nside)
        mp_stacked = np.stack(
            [mp[c] for c in base["comp_indices"]]
        ).astype(np.float32)

        data = dict(base)
        data["vec_rolled"]  = vec_rolled
        data["psi_grid"]    = psi_grid
        data["mp_stacked"]  = mp_stacked

        if include_flatsky:
            # dtheta / dphi: (N_psi, S) small offsets (all zeros → pointing = beam centre)
            data["dtheta"] = np.zeros((N_psi, S), dtype=np.float32)
            data["dphi"]   = np.zeros((N_psi, S), dtype=np.float32)

        return data

    # ── single-Rodrigues path ────────────────────────────────────────────────

    def test_single_rodrigues_output_shape_dtype(self):
        """Single-Rodrigues + Gaussian: output shape (B,) float32 per component."""
        nside = 32
        B     = 5
        data  = self._build_data_with_cache(S=20, N_psi=8, nside=nside)
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)
        mp = _constant_maps(nside)

        tod = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                             interp_mode="gaussian")
        for comp in data["comp_indices"]:
            assert tod[comp].shape == (B,)
            assert tod[comp].dtype == np.float32

    def test_single_rodrigues_constant_map_gives_ones(self):
        """Single-Rodrigues + Gaussian on constant map gives tod ≈ 1.0."""
        nside = 32
        B     = 6
        data  = self._build_data_with_cache(S=20, N_psi=8, nside=nside)
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)
        mp = _constant_maps(nside)

        tod = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                             interp_mode="gaussian")
        for comp in data["comp_indices"]:
            npt.assert_allclose(tod[comp], np.ones(B), atol=1e-3)

    def test_single_rodrigues_gaussian_vs_bilinear_constant(self):
        """Single-Rodrigues path: Gaussian and bilinear agree on a constant map."""
        nside = 32
        B     = 6
        data  = self._build_data_with_cache(S=20, N_psi=8, nside=nside)
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)
        mp = _constant_maps(nside)

        tod_bi = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                                interp_mode="bilinear")
        tod_ga = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                                interp_mode="gaussian")
        for comp in data["comp_indices"]:
            npt.assert_allclose(tod_ga[comp], tod_bi[comp], atol=1e-3)

    # ── flat-sky path ────────────────────────────────────────────────────────

    def test_flatsky_output_shape_dtype(self):
        """Flat-sky + Gaussian: output shape (B,) float32 per component."""
        nside = 32
        B     = 5
        data  = self._build_data_with_cache(S=20, N_psi=8, nside=nside,
                                            include_flatsky=True)
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)
        mp = _constant_maps(nside)

        tod = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                             interp_mode="gaussian")
        for comp in data["comp_indices"]:
            assert tod[comp].shape == (B,)
            assert tod[comp].dtype == np.float32

    def test_flatsky_constant_map_gives_ones(self):
        """Flat-sky + Gaussian on constant map gives tod ≈ 1.0."""
        nside = 32
        B     = 6
        data  = self._build_data_with_cache(S=20, N_psi=8, nside=nside,
                                            include_flatsky=True)
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)
        mp = _constant_maps(nside)

        tod = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                             interp_mode="gaussian")
        for comp in data["comp_indices"]:
            npt.assert_allclose(tod[comp], np.ones(B), atol=1e-3)

    def test_flatsky_gaussian_vs_bilinear_constant(self):
        """Flat-sky path: Gaussian and bilinear agree on a constant map."""
        nside = 32
        B     = 6
        data  = self._build_data_with_cache(S=20, N_psi=8, nside=nside,
                                            include_flatsky=True)
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)
        mp = _constant_maps(nside)

        tod_bi = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                                interp_mode="bilinear")
        tod_ga = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                                interp_mode="gaussian")
        for comp in data["comp_indices"]:
            npt.assert_allclose(tod_ga[comp], tod_bi[comp], atol=1e-3)

    def test_flatsky_no_nan_inf(self):
        """Flat-sky + Gaussian produces only finite values."""
        nside = 32
        B     = 8
        data  = self._build_data_with_cache(S=20, N_psi=8, nside=nside,
                                            include_flatsky=True)
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)
        mp = _constant_maps(nside)

        tod = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                             interp_mode="gaussian")
        for comp in data["comp_indices"]:
            assert np.all(np.isfinite(tod[comp])), \
                f"NaN or Inf in flat-sky Gaussian tod[{comp}]"


# ===========================================================================
# TestGaussianInterpAccumJit
# ===========================================================================

class TestGaussianInterpAccumJit:
    """
    Tests for tod_core._gaussian_interp_accum_jit — the Numba JIT kernel.

    Verifies:
      • constant map → constant output (correctness of weighted sum)
      • results agree with the Python reference (hp.query_disc path) to ~1e-4
        relative error (small difference is expected: numba disc includes a few
        extra boundary pixels with tiny weights)
      • inplace accumulation (calling twice doubles the result)
      • zero beam weights produce zero contribution
      • fallback when the disc is nearly empty (small radius, inclusive=True
        still finds the nearest pixel)
      • parallel (multi-beam) result matches serial loop
    """

    @staticmethod
    def _ref_accum(theta_flat, phi_flat, B, Sc, nside, mp_stacked, beam_vals,
                   sigma_rad, radius_rad):
        """Python reference using hp.query_disc."""
        two_sigma2 = 2.0 * sigma_rad ** 2
        C   = mp_stacked.shape[0]
        out = np.zeros((C, B), dtype=np.float64)
        for b in range(B):
            for s in range(Sc):
                n  = b * Sc + s
                th = theta_flat[n];  ph = phi_flat[n]
                bv = float(beam_vals[s])
                vec_q   = hp.ang2vec(th, ph)
                pix_idx = hp.query_disc(nside, vec_q, radius_rad,
                                        inclusive=True, nest=False)
                if pix_idx.size == 0:
                    nearest = hp.ang2pix(nside, th, ph)
                    for c in range(C):
                        out[c, b] += mp_stacked[c, nearest] * bv
                    continue
                th_n, ph_n = hp.pix2ang(nside, pix_idx)
                cos_d = (np.sin(th) * np.sin(th_n) * np.cos(ph - ph_n)
                         + np.cos(th) * np.cos(th_n))
                dist  = np.arccos(np.clip(cos_d, -1.0, 1.0))
                w     = np.exp(-dist ** 2 / two_sigma2)
                w_sum = w.sum()
                if w_sum < 1e-300:
                    nearest = hp.ang2pix(nside, th, ph)
                    for c in range(C):
                        out[c, b] += mp_stacked[c, nearest] * bv
                else:
                    w_norm = w / w_sum
                    for c in range(C):
                        out[c, b] += (w_norm @ mp_stacked[c][pix_idx]) * bv
        return out

    def test_constant_map_gives_constant(self):
        """JIT kernel on a constant map returns that constant for every (c, b)."""
        nside = 32
        B, Sc = 4, 8
        val   = 2.5
        rng   = np.random.default_rng(0)
        npix  = hp.nside2npix(nside)
        theta_flat = rng.uniform(math.pi/2 - 0.1, math.pi/2 + 0.1, B * Sc)
        phi_flat   = rng.uniform(0, 2 * math.pi, B * Sc)
        mp_stacked = np.full((1, npix), val, dtype=np.float32)
        beam_vals  = np.ones(Sc, dtype=np.float32) / Sc
        tod_arr    = np.zeros((1, B), dtype=np.float64)

        _gaussian_interp_accum_jit(theta_flat, phi_flat, B, Sc,
                                   nside, mp_stacked, beam_vals, tod_arr,
                                   math.radians(1.0), math.radians(3.0))
        np.testing.assert_allclose(tod_arr[0], np.full(B, val), atol=1e-3)

    def test_agrees_with_healpy_reference(self):
        """
        JIT kernel agrees with the hp.query_disc reference to < 1e-3 relative
        error on a random map.  Small differences arise from the numba disc
        including slightly more boundary pixels.
        """
        nside  = 64
        B, Sc  = 6, 5
        np.random.seed(7)
        npix   = hp.nside2npix(nside)
        theta_flat = np.random.uniform(0.2, math.pi - 0.2, B * Sc)
        phi_flat   = np.random.uniform(0.0, 2 * math.pi, B * Sc)
        mp_stacked = np.random.rand(2, npix).astype(np.float32)
        beam_vals  = np.random.rand(Sc).astype(np.float32)
        sigma_rad  = math.radians(0.5)
        radius_rad = math.radians(1.5)

        tod_jit = np.zeros((2, B), dtype=np.float64)
        _gaussian_interp_accum_jit(theta_flat, phi_flat, B, Sc,
                                   nside, mp_stacked, beam_vals, tod_jit,
                                   sigma_rad, radius_rad)
        tod_ref = self._ref_accum(theta_flat, phi_flat, B, Sc,
                                  nside, mp_stacked, beam_vals,
                                  sigma_rad, radius_rad)
        np.testing.assert_allclose(tod_jit, tod_ref,
                                   rtol=1e-3, atol=1e-5,
                                   err_msg="JIT kernel diverges from healpy reference")

    def test_accumulates_inplace(self):
        """Calling the JIT kernel twice with the same inputs doubles the output."""
        nside = 32
        B, Sc = 3, 4
        rng   = np.random.default_rng(1)
        npix  = hp.nside2npix(nside)
        theta_flat = rng.uniform(0.1, math.pi - 0.1, B * Sc)
        phi_flat   = rng.uniform(0.0, 2 * math.pi, B * Sc)
        mp_stacked = rng.random((2, npix)).astype(np.float32)
        beam_vals  = rng.random(Sc).astype(np.float32)
        tod_arr    = np.zeros((2, B), dtype=np.float64)
        kwargs = dict(sigma_rad=math.radians(1.0), radius_rad=math.radians(3.0))

        _gaussian_interp_accum_jit(theta_flat, phi_flat, B, Sc,
                                   nside, mp_stacked, beam_vals, tod_arr, **kwargs)
        first = tod_arr.copy()
        _gaussian_interp_accum_jit(theta_flat, phi_flat, B, Sc,
                                   nside, mp_stacked, beam_vals, tod_arr, **kwargs)
        np.testing.assert_allclose(tod_arr, 2 * first, atol=1e-12)

    def test_zero_beam_vals_gives_zero(self):
        """Zero beam weights produce no contribution."""
        nside = 32
        B, Sc = 3, 5
        npix  = hp.nside2npix(nside)
        theta_flat = np.random.default_rng(2).uniform(0.1, math.pi-0.1, B*Sc)
        phi_flat   = np.random.default_rng(2).uniform(0.0, 2*math.pi, B*Sc)
        mp_stacked = np.ones((1, npix), dtype=np.float32)
        beam_vals  = np.zeros(Sc, dtype=np.float32)
        tod_arr    = np.zeros((1, B), dtype=np.float64)
        _gaussian_interp_accum_jit(theta_flat, phi_flat, B, Sc,
                                   nside, mp_stacked, beam_vals, tod_arr,
                                   math.radians(1.0), math.radians(3.0))
        np.testing.assert_allclose(tod_arr, 0.0, atol=1e-12)

    def test_tiny_radius_uses_nearest_pixel(self):
        """
        With a tiny search radius (< one pixel) the inclusive disc always
        returns at least one pixel and the result is finite and non-zero for
        a non-zero map.
        """
        nside  = 32
        B, Sc  = 2, 2
        npix   = hp.nside2npix(nside)
        mp_stacked = np.ones((1, npix), dtype=np.float32)
        beam_vals  = np.ones(Sc, dtype=np.float32)
        theta_flat = np.full(B * Sc, math.pi / 2, dtype=np.float64)
        phi_flat   = np.full(B * Sc, 0.1, dtype=np.float64)
        tod_arr    = np.zeros((1, B), dtype=np.float64)
        tiny_rad   = 1e-6   # much smaller than a pixel

        _gaussian_interp_accum_jit(theta_flat, phi_flat, B, Sc,
                                   nside, mp_stacked, beam_vals, tod_arr,
                                   tiny_rad, tiny_rad)
        assert np.all(np.isfinite(tod_arr)), "NaN/Inf with tiny radius"
        assert np.all(tod_arr > 0), "Expected non-zero output for non-zero map"

    def test_multi_component_independence(self):
        """
        Each map component is accumulated independently: scaling one component's
        map by 2 doubles only that component's output.
        """
        nside  = 32
        B, Sc  = 4, 3
        npix   = hp.nside2npix(nside)
        rng    = np.random.default_rng(3)
        theta_flat = rng.uniform(0.1, math.pi-0.1, B*Sc)
        phi_flat   = rng.uniform(0.0, 2*math.pi, B*Sc)
        beam_vals  = rng.random(Sc).astype(np.float32)
        mp_base    = rng.random(npix).astype(np.float32)
        sigma_rad  = math.radians(1.0)
        radius_rad = math.radians(3.0)

        # Two-component map: comp 0 = base, comp 1 = 2 * base
        mp_stacked = np.stack([mp_base, 2.0 * mp_base])
        tod_arr    = np.zeros((2, B), dtype=np.float64)
        _gaussian_interp_accum_jit(theta_flat, phi_flat, B, Sc,
                                   nside, mp_stacked, beam_vals, tod_arr,
                                   sigma_rad, radius_rad)
        np.testing.assert_allclose(tod_arr[1], 2.0 * tod_arr[0], rtol=1e-6,
                                   err_msg="Component 1 should be 2x component 0")

    def test_parallel_matches_serial(self):
        """
        Running the parallel JIT kernel on B beams gives the same result as
        running it B times with B=1 (serial decomposition).
        """
        nside  = 32
        B, Sc  = 6, 4
        npix   = hp.nside2npix(nside)
        rng    = np.random.default_rng(4)
        theta_flat = rng.uniform(0.1, math.pi-0.1, B*Sc)
        phi_flat   = rng.uniform(0.0, 2*math.pi, B*Sc)
        mp_stacked = rng.random((2, npix)).astype(np.float32)
        beam_vals  = rng.random(Sc).astype(np.float32)
        sigma_rad  = math.radians(0.8)
        radius_rad = math.radians(2.4)

        # Parallel: all B at once
        tod_par = np.zeros((2, B), dtype=np.float64)
        _gaussian_interp_accum_jit(theta_flat, phi_flat, B, Sc,
                                   nside, mp_stacked, beam_vals, tod_par,
                                   sigma_rad, radius_rad)

        # Serial: one beam at a time
        tod_ser = np.zeros((2, B), dtype=np.float64)
        for b in range(B):
            tf_b = theta_flat[b*Sc:(b+1)*Sc]
            pf_b = phi_flat  [b*Sc:(b+1)*Sc]
            buf  = np.zeros((2, 1), dtype=np.float64)
            _gaussian_interp_accum_jit(tf_b, pf_b, 1, Sc,
                                       nside, mp_stacked, beam_vals, buf,
                                       sigma_rad, radius_rad)
            tod_ser[:, b] = buf[:, 0]

        np.testing.assert_allclose(tod_par, tod_ser, atol=1e-12,
                                   err_msg="Parallel and serial results differ")


# ===========================================================================
# Entry point for standalone execution
# ===========================================================================

if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
