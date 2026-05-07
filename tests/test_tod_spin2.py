"""
Tests for the spin-2 frame-rotation primitives in tod_spin2.

- _spin2_cos2d_sin2d_jit         : spherical-trig spin-2 rotation angles
- _spin2_lookup_cached           : direct-mapped cache wrapper
- compute_spin2_skip_z_threshold : skip-band derivation
- _max_two_delta_at_boresight    : worst-case |2δ| sweep helper

Can be run independently:
    pytest tests/test_tod_spin2.py -v
    python tests/test_tod_spin2.py
"""

import os
import sys
import math
from unittest.mock import MagicMock

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
import pytest

from tod_spin2 import (
    _spin2_cos2d_sin2d_jit,
    _spin2_lookup_cached,
    compute_spin2_skip_z_threshold,
    _max_two_delta_at_boresight,
)


# ===========================================================================
# TestSpin2Cos2dSin2d
# ===========================================================================


class TestSpin2Cos2dSin2d:
    """Unit tests for _spin2_cos2d_sin2d_jit (haversine / tan(δ) formulation)."""

    @staticmethod
    def _angles(theta, phi):
        return math.cos(theta), math.sin(theta), phi

    def test_identity_same_point(self):
        """Exact same pixel and target → no rotation (cos=1, sin=0)."""
        z, s, p = self._angles(0.5, 1.2)
        c2, s2 = _spin2_cos2d_sin2d_jit(z, s, p, z, s, p)
        assert abs(c2 - 1.0) < 1e-12
        assert abs(s2) < 1e-12

    def test_unit_norm(self):
        """cos²(2δ) + sin²(2δ) == 1 for random pairs (algebraically exact in new formula)."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            t1, p1 = rng.uniform(0.05, math.pi - 0.05), rng.uniform(0.0, 2 * math.pi)
            t2, p2 = rng.uniform(0.05, math.pi - 0.05), rng.uniform(0.0, 2 * math.pi)
            z1, s1 = math.cos(t1), math.sin(t1)
            z2, s2 = math.cos(t2), math.sin(t2)
            c2, si2 = _spin2_cos2d_sin2d_jit(z1, s1, p1, z2, s2, p2)
            npt.assert_allclose(c2**2 + si2**2, 1.0, atol=1e-14)

    def test_antisymmetry(self):
        """Swapping pix ↔ target negates δ: cos(2δ) is even, sin(2δ) is odd."""
        t1, p1 = 0.7, 0.3
        t2, p2 = 1.4, 2.1
        z1, s1 = math.cos(t1), math.sin(t1)
        z2, s2 = math.cos(t2), math.sin(t2)
        c_fwd, s_fwd = _spin2_cos2d_sin2d_jit(z1, s1, p1, z2, s2, p2)
        c_bwd, s_bwd = _spin2_cos2d_sin2d_jit(z2, s2, p2, z1, s1, p1)
        npt.assert_allclose(c_fwd, c_bwd, atol=1e-12)
        npt.assert_allclose(s_fwd, -s_bwd, atol=1e-12)

    def test_agrees_with_atan2_reference(self):
        """
        Agrees with the explicit atan2 computation from the toy model
        (example_QU_convolution.py) to machine precision.
        """
        rng = np.random.default_rng(42)
        for _ in range(200):
            t1, p1 = rng.uniform(0.05, math.pi - 0.05), rng.uniform(0.0, 2 * math.pi)
            t2, p2 = rng.uniform(0.05, math.pi - 0.05), rng.uniform(0.0, 2 * math.pi)
            z1, s1 = math.cos(t1), math.sin(t1)
            z2, s2 = math.cos(t2), math.sin(t2)

            # Reference: toy-model formula with atan2
            dphi = p2 - p1
            cd, sd = math.cos(dphi), math.sin(dphi)
            cos_beta = s1 * s2 * cd + z1 * z2
            cos_beta = max(-1.0, min(1.0, cos_beta))
            sin_beta = math.sqrt(max(0.0, 1.0 - cos_beta**2))
            if sin_beta < 1e-12:
                continue
            isb = 1.0 / sin_beta
            sa_ref = s2 * sd * isb
            ca_ref = (s2 * z1 * cd - z2 * s1) * isb
            sg_ref = s1 * sd * isb
            cg_ref = -(s1 * z2 * cd - z1 * s2) * isb
            alpha = math.atan2(sa_ref, ca_ref)
            gamma = math.atan2(sg_ref, cg_ref)
            delta = alpha - gamma
            c2_ref = math.cos(2 * delta)
            s2_ref = math.sin(2 * delta)

            c2, s2 = _spin2_cos2d_sin2d_jit(z1, s1, p1, z2, s2, p2)
            npt.assert_allclose(c2, c2_ref, atol=1e-12)
            npt.assert_allclose(s2, s2_ref, atol=1e-12)

    def test_same_meridian_no_rotation(self):
        """Points on the same meridian (dphi=0) → cos=1, sin=0 identically (N=0 branch)."""
        cases = [(0.3, 1.0), (1.0, 2.5), (0.1, math.pi - 0.1)]
        for t1, t2 in cases:
            z1, s1 = math.cos(t1), math.sin(t1)
            z2, s2 = math.cos(t2), math.sin(t2)
            c2, si2 = _spin2_cos2d_sin2d_jit(z1, s1, 0.0, z2, s2, 0.0)
            npt.assert_allclose(c2, 1.0, atol=1e-12, err_msg=f"t1={t1}, t2={t2}")
            npt.assert_allclose(si2, 0.0, atol=1e-12, err_msg=f"t1={t1}, t2={t2}")

    def test_exact_coincident_pole_returns_identity(self):
        """Exact same point at the north pole → equality guard returns (1, 0)."""
        c2, s2 = _spin2_cos2d_sin2d_jit(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        assert c2 == 1.0 and s2 == 0.0

    def test_near_coincident_is_finite(self):
        """Near-coincident (not exactly equal) points → finite result close to (1, 0)."""
        eps = 1e-7
        t0, p0 = 1.0, 1.5
        z0, s0 = math.cos(t0), math.sin(t0)
        z1, s1 = math.cos(t0 + eps), math.sin(t0 + eps)
        c2, si2 = _spin2_cos2d_sin2d_jit(z0, s0, p0, z1, s1, p0 + eps)
        assert math.isfinite(c2) and math.isfinite(si2)
        npt.assert_allclose(c2**2 + si2**2, 1.0, atol=1e-12)

    def test_near_pole_is_finite(self):
        """Near-pole (non-degenerate) pair → finite result on the unit circle."""
        z_pix, sth_pix, phi_pix = math.cos(0.01), math.sin(0.01), 0.0
        z_pts, sth_pts, phi_pts = math.cos(1.0), math.sin(1.0), 1.0
        c2, si2 = _spin2_cos2d_sin2d_jit(
            z_pix, sth_pix, phi_pix, z_pts, sth_pts, phi_pts
        )
        assert math.isfinite(c2) and math.isfinite(si2)
        npt.assert_allclose(c2**2 + si2**2, 1.0, atol=1e-12)


# ===========================================================================
# TestSpin2LookupCached
# ===========================================================================


class TestSpin2LookupCached:
    """Tests for the direct-mapped spin-2 cache helper.

    The helper wraps :func:`_spin2_cos2d_sin2d_jit` with a per-pixel cache.
    Correctness requirements:

    - On a cold slot (pixel index -1 or a different pixel) the helper must
      return the same ``(cos 2δ, sin 2δ)`` as a direct call to
      ``_spin2_cos2d_sin2d_jit``, AND install those values in the slot.
    - On a warm slot (same pixel as the one stored) the cached values are
      returned without re-computing — verified by flipping the cache entry to a
      sentinel value and ensuring that value is returned.
    - When two pixels hash to the same slot, the newer one evicts the older
      one, and subsequent lookups of the evicted pixel are a miss (recompute).
    """

    N_CACHE = 1024
    CMASK = N_CACHE - 1

    def _empty_cache(self):
        return (
            np.full(self.N_CACHE, -1, dtype=np.int64),
            np.empty(self.N_CACHE, dtype=np.float64),
            np.empty(self.N_CACHE, dtype=np.float64),
        )

    def _ref(self, z_n, phi_n, z_pts, sth_pts, phi_pts):
        sth_n = math.sqrt(max(0.0, 1.0 - z_n * z_n))
        return _spin2_cos2d_sin2d_jit(z_n, sth_n, phi_n, z_pts, sth_pts, phi_pts)

    def test_miss_matches_direct_call(self):
        """On a cold cache the helper returns exactly _spin2_cos2d_sin2d_jit."""
        cache_pix, cache_c2d, cache_s2d = self._empty_cache()
        z_pts, sth_pts, phi_pts = 0.6, 0.8, 1.2

        rng = np.random.default_rng(0)
        for _ in range(50):
            p = int(rng.integers(0, 10_000_000))
            theta_n = rng.uniform(0.05, math.pi - 0.05)
            phi_n = rng.uniform(0.0, 2 * math.pi)
            z_n = math.cos(theta_n)

            c2d_ref, s2d_ref = self._ref(z_n, phi_n, z_pts, sth_pts, phi_pts)
            c2d, s2d = _spin2_lookup_cached(
                p,
                z_n,
                phi_n,
                z_pts,
                sth_pts,
                phi_pts,
                cache_pix,
                cache_c2d,
                cache_s2d,
                self.CMASK,
            )
            npt.assert_allclose(c2d, c2d_ref, atol=1e-14)
            npt.assert_allclose(s2d, s2d_ref, atol=1e-14)

    def test_miss_populates_slot(self):
        """After a miss the slot contains the queried pixel and its spin-2 values."""
        cache_pix, cache_c2d, cache_s2d = self._empty_cache()
        z_pts, sth_pts, phi_pts = 0.6, 0.8, 1.2
        p, z_n, phi_n = 12345, 0.3, 0.7

        c2d, s2d = _spin2_lookup_cached(
            p,
            z_n,
            phi_n,
            z_pts,
            sth_pts,
            phi_pts,
            cache_pix,
            cache_c2d,
            cache_s2d,
            self.CMASK,
        )
        slot = (p * 2654435769) & self.CMASK
        assert cache_pix[slot] == p
        npt.assert_allclose(cache_c2d[slot], c2d, atol=0.0)
        npt.assert_allclose(cache_s2d[slot], s2d, atol=0.0)

    def test_hit_returns_stored_value(self):
        """A warm slot returns the cached value without recomputing.

        Verified by corrupting the cached (c2d, s2d) with a sentinel; the
        helper must return the sentinel rather than the true spin-2 values.
        """
        cache_pix, cache_c2d, cache_s2d = self._empty_cache()
        z_pts, sth_pts, phi_pts = 0.6, 0.8, 1.2
        p, z_n, phi_n = 42, 0.2, 1.1

        # First call: miss, populates slot.
        _spin2_lookup_cached(
            p,
            z_n,
            phi_n,
            z_pts,
            sth_pts,
            phi_pts,
            cache_pix,
            cache_c2d,
            cache_s2d,
            self.CMASK,
        )
        slot = (p * 2654435769) & self.CMASK
        # Overwrite stored values with a sentinel.
        cache_c2d[slot] = -123.456
        cache_s2d[slot] = 789.012
        # Second call: hit — must return the sentinel.
        c2d, s2d = _spin2_lookup_cached(
            p,
            z_n,
            phi_n,
            z_pts,
            sth_pts,
            phi_pts,
            cache_pix,
            cache_c2d,
            cache_s2d,
            self.CMASK,
        )
        assert c2d == -123.456
        assert s2d == 789.012

    def test_eviction_on_hash_collision(self):
        """Two pixels hashing to the same slot: second evicts first."""
        cache_pix, cache_c2d, cache_s2d = self._empty_cache()
        z_pts, sth_pts, phi_pts = 0.6, 0.8, 1.2

        # Find two distinct pixels that collide.
        p_a = 13
        slot_a = (p_a * 2654435769) & self.CMASK
        p_b = None
        for candidate in range(p_a + 1, p_a + 200_000):
            if ((candidate * 2654435769) & self.CMASK) == slot_a:
                p_b = candidate
                break
        assert p_b is not None, "Could not find a collision within search range"

        z_a, phi_a = 0.2, 0.5
        z_b, phi_b = -0.4, 2.1

        _spin2_lookup_cached(
            p_a,
            z_a,
            phi_a,
            z_pts,
            sth_pts,
            phi_pts,
            cache_pix,
            cache_c2d,
            cache_s2d,
            self.CMASK,
        )
        assert cache_pix[slot_a] == p_a
        # Lookup p_b: miss (different pixel), evicts p_a.
        c2d_b, s2d_b = _spin2_lookup_cached(
            p_b,
            z_b,
            phi_b,
            z_pts,
            sth_pts,
            phi_pts,
            cache_pix,
            cache_c2d,
            cache_s2d,
            self.CMASK,
        )
        assert cache_pix[slot_a] == p_b
        c2d_ref, s2d_ref = self._ref(z_b, phi_b, z_pts, sth_pts, phi_pts)
        npt.assert_allclose(c2d_b, c2d_ref, atol=1e-14)
        npt.assert_allclose(s2d_b, s2d_ref, atol=1e-14)

        # Re-querying p_a is now a miss again (slot holds p_b).
        c2d_a, s2d_a = _spin2_lookup_cached(
            p_a,
            z_a,
            phi_a,
            z_pts,
            sth_pts,
            phi_pts,
            cache_pix,
            cache_c2d,
            cache_s2d,
            self.CMASK,
        )
        c2d_a_ref, s2d_a_ref = self._ref(z_a, phi_a, z_pts, sth_pts, phi_pts)
        npt.assert_allclose(c2d_a, c2d_a_ref, atol=1e-14)
        npt.assert_allclose(s2d_a, s2d_a_ref, atol=1e-14)
        assert cache_pix[slot_a] == p_a  # evicted p_b

    def test_reset_slot_treated_as_miss(self):
        """A slot explicitly set back to -1 is treated as a cold miss."""
        cache_pix, cache_c2d, cache_s2d = self._empty_cache()
        z_pts, sth_pts, phi_pts = 0.6, 0.8, 1.2
        p, z_n, phi_n = 77, 0.1, 2.0

        _spin2_lookup_cached(
            p,
            z_n,
            phi_n,
            z_pts,
            sth_pts,
            phi_pts,
            cache_pix,
            cache_c2d,
            cache_s2d,
            self.CMASK,
        )
        slot = (p * 2654435769) & self.CMASK
        # Poison with bad stored values AND reset the sentinel:
        cache_pix[slot] = -1
        cache_c2d[slot] = -999.0
        cache_s2d[slot] = -999.0
        # Now a lookup must recompute — returns the true value, not the garbage.
        c2d, s2d = _spin2_lookup_cached(
            p,
            z_n,
            phi_n,
            z_pts,
            sth_pts,
            phi_pts,
            cache_pix,
            cache_c2d,
            cache_s2d,
            self.CMASK,
        )
        c2d_ref, s2d_ref = self._ref(z_n, phi_n, z_pts, sth_pts, phi_pts)
        npt.assert_allclose(c2d, c2d_ref, atol=1e-14)
        npt.assert_allclose(s2d, s2d_ref, atol=1e-14)


# ===========================================================================
# TestComputeSpin2SkipZThreshold
# ===========================================================================


class TestComputeSpin2SkipZThreshold:
    """Tests for compute_spin2_skip_z_threshold."""

    def test_disabled_when_tol_zero(self):
        assert compute_spin2_skip_z_threshold(0.01, 0.0) == -1.0
        assert compute_spin2_skip_z_threshold(0.01, None) == -1.0
        assert compute_spin2_skip_z_threshold(0.01, -0.5) == -1.0

    def test_zero_beam_radius(self):
        """Zero-radius beam → no correction needed anywhere → z_threshold = 1."""
        assert compute_spin2_skip_z_threshold(0.0, 0.01) == 1.0

    def test_threshold_in_range(self):
        """Returned z_threshold must lie in [-1, 1] and grow with tolerance."""
        beam_radius = math.radians(1.0)  # ~1° beam
        z_loose = compute_spin2_skip_z_threshold(beam_radius, 0.1)
        z_tight = compute_spin2_skip_z_threshold(beam_radius, 0.001)
        assert -1.0 <= z_tight <= z_loose <= 1.0

    def test_within_tolerance_at_threshold(self):
        """For boresights at exactly the returned threshold, sampled |2δ| ≤ tol."""
        beam_radius = math.radians(1.5)
        tol = 0.02
        z_thresh = compute_spin2_skip_z_threshold(beam_radius, tol)
        if z_thresh < 0.0:
            pytest.skip("threshold disabled for this configuration")
        # Sample beam-edge offsets at theta_pts = arccos(z_thresh) and check
        # max |2δ| is below tol.
        theta_pts = math.acos(z_thresh)
        max_2d = _max_two_delta_at_boresight(theta_pts, beam_radius, n_az=256)
        assert max_2d <= tol * 1.05, (
            f"max |2δ| = {max_2d:.6f} exceeds tol = {tol} (z_thresh={z_thresh:.4f})"
        )


# ===========================================================================
# TestMaxTwoDeltaAtBoresight
# ===========================================================================


class TestMaxTwoDeltaAtBoresight:
    """Direct unit tests for the worst-case |2δ| sweep helper.

    This helper is used internally by compute_spin2_skip_z_threshold, but
    has its own correctness contract: for a boresight at colatitude
    ``theta_pts`` it must return the maximum |2δ| over a ring of beam-edge
    azimuths at angular distance ``R``.
    """

    def test_zero_radius_returns_zero(self):
        """R=0 collapses every sample onto the boresight → |2δ| = 0."""
        max_2d = _max_two_delta_at_boresight(theta_pts=0.5, R=0.0, n_az=64)
        assert max_2d == 0.0

    def test_equator_close_boresight_negligible(self):
        """Near the equator, |2δ| → 0 for a small beam (sub-radian)."""
        # theta_pts close to π/2 (boresight near equator).
        max_2d = _max_two_delta_at_boresight(
            theta_pts=math.pi / 2 - 1e-3,
            R=math.radians(0.5),
            n_az=128,
        )
        assert max_2d < 1e-3

    def test_polar_boresight_grows(self):
        """|2δ| grows as the boresight approaches the pole at fixed R."""
        R = math.radians(1.0)
        m_eq = _max_two_delta_at_boresight(theta_pts=math.pi / 2 - 0.1, R=R, n_az=128)
        m_mid = _max_two_delta_at_boresight(theta_pts=0.5, R=R, n_az=128)
        m_pole = _max_two_delta_at_boresight(theta_pts=0.05, R=R, n_az=128)
        assert m_eq < m_mid < m_pole

    def test_grows_with_radius(self):
        """At fixed boresight, |2δ| grows monotonically with R."""
        theta_pts = 0.3
        m_small = _max_two_delta_at_boresight(theta_pts, R=math.radians(0.1), n_az=128)
        m_med = _max_two_delta_at_boresight(theta_pts, R=math.radians(1.0), n_az=128)
        m_large = _max_two_delta_at_boresight(theta_pts, R=math.radians(5.0), n_az=128)
        assert 0.0 <= m_small < m_med < m_large

    def test_finite_and_nonneg(self):
        """For a generic configuration the result is finite and ≥ 0."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            theta_pts = float(rng.uniform(0.01, math.pi / 2 - 0.01))
            R = float(rng.uniform(1e-3, math.radians(5.0)))
            n_az = int(rng.integers(8, 256))
            v = _max_two_delta_at_boresight(theta_pts, R, n_az)
            assert math.isfinite(v) and v >= 0.0


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
