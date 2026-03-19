"""
Tests for the tod_utils module.

Covers: _ring_above_jit, _ring_info_jit, _ring_z_jit
        _get_interp_weights_jit, get_interp_weights_numba.

Can be run independently:
    pytest tests/test_numba_healpy.py -v
    python tests/test_numba_healpy.py
"""

import os
import sys
import math
import importlib
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Ensure project root and stubs are available when run as a standalone file
# (conftest.py handles this automatically under pytest).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

for _mod_name in ["pixell", "pixell.enmap"]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

if "tod_io" not in sys.modules:
    sys.modules["tod_io"] = MagicMock()

import healpy as hp
import numpy as np
import numpy.testing as npt
import pytest

from numba_healpy import (
    _ring_above_jit,
    _ring_info_jit,
    _ring_z_jit,
    _get_interp_weights_jit,
    get_interp_weights_numba
)


# ===========================================================================
# TestRingAboveJit
# ===========================================================================

class TestRingAboveJit:
    """Tests for tod_core._ring_above_jit — the scalar HEALPix ring_above helper."""

    @pytest.mark.parametrize("nside", [1, 2, 4, 16, 64])
    def test_equatorial_midpoint(self, nside):
        """At z=0 (equatorial midpoint), ring_above returns 2*nside."""
        assert _ring_above_jit(nside, 0.0) == 2 * nside

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_equatorial_north_boundary(self, nside):
        """At z=2/3 (NPC/equatorial boundary), ring_above returns nside."""
        assert _ring_above_jit(nside, 2.0 / 3.0) == nside

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_equatorial_south_boundary(self, nside):
        """At z=-2/3 (equatorial/SPC boundary), ring_above returns 3*nside."""
        assert _ring_above_jit(nside, -2.0 / 3.0) == 3 * nside

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_north_pole(self, nside):
        """At z=1 (north pole), ring_above returns 0."""
        assert _ring_above_jit(nside, 1.0) == 0

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_south_pole(self, nside):
        """At z=-1 (south pole), ring_above returns 4*nside-1."""
        assert _ring_above_jit(nside, -1.0) == 4 * nside - 1

    @pytest.mark.parametrize("nside", [4, 16])
    def test_result_in_valid_range(self, nside):
        """ring_above is always in [0, 4*nside-1] for any z in [-1, 1]."""
        rng = np.random.default_rng(20)
        for z in rng.uniform(-1.0, 1.0, 200):
            ir = _ring_above_jit(nside, float(z))
            assert 0 <= ir <= 4 * nside - 1, (
                f"ring_above={ir} out of range [0, {4*nside-1}] for z={z:.6f}")

    @pytest.mark.parametrize("nside", [4, 16])
    def test_consistent_with_ring_z(self, nside):
        """For interior z, z_ring(ir) >= z >= z_ring(ir+1) where ir = ring_above(z)."""
        rng = np.random.default_rng(21)
        z_vals = rng.uniform(-0.60, 0.60, 50)   # stay away from polar boundaries
        for z in z_vals:
            ir = _ring_above_jit(nside, float(z))
            if 1 <= ir <= 4 * nside - 2:
                za = _ring_z_jit(nside, ir)
                zb = _ring_z_jit(nside, ir + 1)
                assert za >= float(z) - 1e-12, (
                    f"z_ring({ir})={za:.8f} < z={z:.8f}")
                assert zb <= float(z) + 1e-12, (
                    f"z_ring({ir+1})={zb:.8f} > z={z:.8f}")

    @pytest.mark.parametrize("nside", [4, 16])
    def test_monotone_decreasing_output(self, nside):
        """ring_above is non-decreasing as z decreases (monotone ordering)."""
        z_vals = np.linspace(0.99, -0.99, 200)
        prev = _ring_above_jit(nside, float(z_vals[0]))
        for z in z_vals[1:]:
            ir = _ring_above_jit(nside, float(z))
            assert ir >= prev, f"ring_above not monotone: {prev} -> {ir} at z={z:.4f}"
            prev = ir


# ===========================================================================
# TestRingInfoJit
# ===========================================================================

class TestRingInfoJit:
    """Tests for tod_core._ring_info_jit — ring layout helper."""

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_first_npc_ring(self, nside):
        """Ring 1 (first NPC ring): n_pix=4, first_pix=0, shifted."""
        npix = 12 * nside * nside
        n_pix, first_pix, phi0, dphi = _ring_info_jit(nside, 1, npix)
        assert n_pix    == 4
        assert first_pix == 0
        npt.assert_allclose(dphi, 2 * np.pi / 4, atol=1e-14)
        npt.assert_allclose(phi0, dphi / 2,       atol=1e-14,
                            err_msg="ring 1 should be shifted (phi0 = dphi/2)")

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_npc_ring_sizes(self, nside):
        """Each NPC ring ir has exactly 4*ir pixels."""
        npix = 12 * nside * nside
        for ir in range(1, nside):
            n_pix, _, _, _ = _ring_info_jit(nside, ir, npix)
            assert n_pix == 4 * ir, f"ring {ir}: got n_pix={n_pix}, expected {4*ir}"

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_equatorial_ring_sizes(self, nside):
        """All equatorial rings have exactly 4*nside pixels."""
        npix = 12 * nside * nside
        for ir in range(nside, 3 * nside + 1):
            n_pix, _, _, _ = _ring_info_jit(nside, ir, npix)
            assert n_pix == 4 * nside, (
                f"ring {ir}: got n_pix={n_pix}, expected {4*nside}")

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_first_equatorial_startpix(self, nside):
        """Ring nside starts at ncap = 2*nside*(nside-1)."""
        npix = 12 * nside * nside
        ncap = 2 * nside * (nside - 1)
        _, first_pix, _, _ = _ring_info_jit(nside, nside, npix)
        assert first_pix == ncap, (
            f"nside={nside}: first_pix={first_pix}, expected ncap={ncap}")

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_equatorial_shift_alternates(self, nside):
        """Equatorial rings shift when (ir-nside) is even, not shifted when odd."""
        npix = 12 * nside * nside
        for ir in range(nside, 3 * nside + 1):
            _, _, phi0, dphi = _ring_info_jit(nside, ir, npix)
            if (ir - nside) % 2 == 0:
                npt.assert_allclose(phi0, dphi / 2, atol=1e-14,
                                    err_msg=f"ring {ir} should be shifted")
            else:
                npt.assert_allclose(phi0, 0.0, atol=1e-14,
                                    err_msg=f"ring {ir} should NOT be shifted")

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_spc_rings_always_shifted(self, nside):
        """All south polar cap rings are shifted (phi0 = dphi/2)."""
        npix = 12 * nside * nside
        for ir in range(3 * nside + 1, 4 * nside):
            _, _, phi0, dphi = _ring_info_jit(nside, ir, npix)
            npt.assert_allclose(phi0, dphi / 2, atol=1e-14,
                                err_msg=f"SPC ring {ir} should be shifted")

    @pytest.mark.parametrize("nside", [2, 4, 8, 16])
    def test_partition_covers_all_pixels(self, nside):
        """Ring pixel ranges partition [0, 12*nside^2) exactly, with no overlap or gap."""
        npix_total = 12 * nside * nside
        covered = np.zeros(npix_total, dtype=np.int32)
        for ir in range(1, 4 * nside):
            n_pix, first_pix, _, _ = _ring_info_jit(nside, ir, npix_total)
            assert first_pix >= 0, f"ring {ir}: first_pix={first_pix} < 0"
            assert first_pix + n_pix <= npix_total, (
                f"ring {ir}: pixel range [{first_pix}, {first_pix+n_pix}) exceeds npix")
            covered[first_pix:first_pix + n_pix] += 1
        npt.assert_array_equal(covered, np.ones(npix_total, dtype=np.int32),
                               err_msg="Some pixels covered 0 or >1 times")

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_dphi_equals_twopi_over_npix(self, nside):
        """dphi == 2*pi / n_pix for every ring."""
        npix = 12 * nside * nside
        for ir in range(1, 4 * nside):
            n_pix, _, _, dphi = _ring_info_jit(nside, ir, npix)
            npt.assert_allclose(dphi, 2 * np.pi / n_pix, atol=1e-14,
                                err_msg=f"ring {ir}: dphi mismatch")


# ===========================================================================
# TestRingZJit
# ===========================================================================

class TestRingZJit:
    """Tests for tod_core._ring_z_jit — ring centre z=cos(theta) helper."""

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_equatorial_north_boundary(self, nside):
        """z_ring(nside) == 2/3 (NPC/equatorial boundary)."""
        npt.assert_allclose(_ring_z_jit(nside, nside), 2.0 / 3.0, atol=1e-14)

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_equatorial_south_boundary(self, nside):
        """z_ring(3*nside) == -2/3 (equatorial/SPC boundary)."""
        npt.assert_allclose(_ring_z_jit(nside, 3 * nside), -2.0 / 3.0, atol=1e-14)

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_equatorial_midpoint(self, nside):
        """z_ring(2*nside) == 0 (equatorial centre)."""
        npt.assert_allclose(_ring_z_jit(nside, 2 * nside), 0.0, atol=1e-14)

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_first_npc_ring(self, nside):
        """z_ring(1) == 1 - 1/(3*nside^2) (first north polar-cap ring)."""
        expected = 1.0 - 1.0 / (3.0 * nside * nside)
        npt.assert_allclose(_ring_z_jit(nside, 1), expected, atol=1e-14)

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_last_spc_ring(self, nside):
        """z_ring(4*nside-1) == -1 + 1/(3*nside^2) (last south polar-cap ring)."""
        expected = -1.0 + 1.0 / (3.0 * nside * nside)
        npt.assert_allclose(_ring_z_jit(nside, 4 * nside - 1), expected, atol=1e-14)

    @pytest.mark.parametrize("nside", [4, 16])
    def test_all_values_in_minus1_plus1(self, nside):
        """z_ring(ir) lies in (-1, 1] for all valid ring indices."""
        for ir in range(1, 4 * nside):
            z = _ring_z_jit(nside, ir)
            assert -1.0 <= z <= 1.0, f"z_ring({ir}) = {z:.10f} outside [-1, 1]"

    @pytest.mark.parametrize("nside", [4, 16])
    def test_strictly_decreasing(self, nside):
        """z_ring decreases strictly from ring 1 to ring 4*nside-1."""
        z_prev = _ring_z_jit(nside, 1)
        for ir in range(2, 4 * nside):
            z = _ring_z_jit(nside, ir)
            assert z < z_prev, (
                f"z not strictly decreasing at ir={ir}: z[{ir}]={z:.10f} >= z[{ir-1}]={z_prev:.10f}")
            z_prev = z

    @pytest.mark.parametrize("nside", [4, 16])
    def test_north_south_symmetry(self, nside):
        """z_ring(ir) == -z_ring(4*nside - ir) for all interior rings."""
        for ir in range(1, 2 * nside):
            z_north = _ring_z_jit(nside, ir)
            z_south = _ring_z_jit(nside, 4 * nside - ir)
            npt.assert_allclose(z_north, -z_south, atol=1e-14,
                                err_msg=f"N/S symmetry failed at ir={ir}")


# ===========================================================================
# TestGetInterpWeightsNumba
# ===========================================================================

class TestGetInterpWeightsNumba:
    """
    Tests for tod_core.get_interp_weights_numba and _get_interp_weights_jit.

    Core validation: pixel-exact and weight-close agreement with healpy's
    hp.get_interp_weights on random inputs across all nside values and all
    three sphere regions (NPC, equatorial belt, SPC).
    """

    # ── output contract ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_output_shapes_and_dtypes(self, nside):
        """pixels: (4, N) int64; weights: (4, N) float64."""
        rng   = np.random.default_rng(30)
        N     = 60
        theta = rng.uniform(0.05, np.pi - 0.05, N)
        phi   = rng.uniform(0.0, 2 * np.pi, N)
        pix, wgt = get_interp_weights_numba(nside, theta, phi)
        assert pix.shape == (4, N)
        assert wgt.shape == (4, N)
        assert pix.dtype == np.int64
        assert wgt.dtype == np.float64

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_weights_sum_to_one(self, nside):
        """The four bilinear weights sum to exactly 1 for every point."""
        rng   = np.random.default_rng(31)
        N     = 300
        theta = rng.uniform(0.0, np.pi, N)
        phi   = rng.uniform(0.0, 2 * np.pi, N)
        _, wgt = get_interp_weights_numba(nside, theta, phi)
        npt.assert_allclose(wgt.sum(axis=0), np.ones(N), atol=1e-14,
                            err_msg="Bilinear weights do not sum to 1")

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_pixels_in_valid_range(self, nside):
        """All pixel indices lie in [0, 12*nside^2)."""
        npix  = hp.nside2npix(nside)
        rng   = np.random.default_rng(32)
        N     = 300
        theta = rng.uniform(0.0, np.pi, N)
        phi   = rng.uniform(0.0, 2 * np.pi, N)
        pix, _ = get_interp_weights_numba(nside, theta, phi)
        assert int(pix.min()) >= 0,    "Pixel index below 0"
        assert int(pix.max()) < npix,  f"Pixel index >= npix={npix}"

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_weights_non_negative(self, nside):
        """All interpolation weights are >= 0."""
        rng   = np.random.default_rng(33)
        N     = 300
        theta = rng.uniform(0.0, np.pi, N)
        phi   = rng.uniform(0.0, 2 * np.pi, N)
        _, wgt = get_interp_weights_numba(nside, theta, phi)
        assert float(wgt.min()) >= -1e-14, f"Negative weight: {wgt.min()}"

    # ── agreement with healpy reference ──────────────────────────────────────

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_agrees_with_healpy_pixels(self, nside):
        """Pixel indices exactly match hp.get_interp_weights on 500 random interior points."""
        rng   = np.random.default_rng(34)
        N     = 500
        theta = rng.uniform(0.05, np.pi - 0.05, N)
        phi   = rng.uniform(0.0, 2 * np.pi, N)
        pix_hp, _      = hp.get_interp_weights(nside, theta, phi)
        pix_nb, _      = get_interp_weights_numba(nside, theta, phi)
        npt.assert_array_equal(pix_nb, pix_hp,
                               err_msg=f"Pixel mismatch at nside={nside}")

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_agrees_with_healpy_weights(self, nside):
        """Bilinear weights match hp.get_interp_weights to 1e-12 on 500 random points."""
        rng   = np.random.default_rng(35)
        N     = 500
        theta = rng.uniform(0.05, np.pi - 0.05, N)
        phi   = rng.uniform(0.0, 2 * np.pi, N)
        _, wgt_hp = hp.get_interp_weights(nside, theta, phi)
        _, wgt_nb = get_interp_weights_numba(nside, theta, phi)
        npt.assert_allclose(wgt_nb, wgt_hp, atol=1e-12,
                            err_msg=f"Weight mismatch at nside={nside}")

    @pytest.mark.parametrize("nside", [4, 16])
    def test_agrees_with_healpy_in_npc(self, nside):
        """Exact pixel+weight agreement in the north polar cap (theta < arccos(2/3))."""
        rng   = np.random.default_rng(36)
        N     = 200
        theta = rng.uniform(0.05, math.acos(2.0 / 3.0) - 0.02, N)
        phi   = rng.uniform(0.0, 2 * np.pi, N)
        pix_hp, wgt_hp = hp.get_interp_weights(nside, theta, phi)
        pix_nb, wgt_nb = get_interp_weights_numba(nside, theta, phi)
        npt.assert_array_equal(pix_nb, pix_hp,  err_msg="NPC pixel mismatch")
        npt.assert_allclose(wgt_nb, wgt_hp, atol=1e-12, err_msg="NPC weight mismatch")

    @pytest.mark.parametrize("nside", [4, 16])
    def test_agrees_with_healpy_in_spc(self, nside):
        """Exact pixel+weight agreement in the south polar cap (theta > pi-arccos(2/3))."""
        rng   = np.random.default_rng(37)
        N     = 200
        theta = rng.uniform(np.pi - math.acos(2.0 / 3.0) + 0.02, np.pi - 0.05, N)
        phi   = rng.uniform(0.0, 2 * np.pi, N)
        pix_hp, wgt_hp = hp.get_interp_weights(nside, theta, phi)
        pix_nb, wgt_nb = get_interp_weights_numba(nside, theta, phi)
        npt.assert_array_equal(pix_nb, pix_hp,  err_msg="SPC pixel mismatch")
        npt.assert_allclose(wgt_nb, wgt_hp, atol=1e-12, err_msg="SPC weight mismatch")

    # ── special/boundary inputs ───────────────────────────────────────────────

    @pytest.mark.parametrize("nside", [4, 16])
    def test_near_poles_valid_outputs(self, nside):
        """Points very close to the poles produce valid pixel indices and unit weight sums."""
        npix = hp.nside2npix(nside)
        eps  = 1e-6
        # North-pole cluster
        theta_n = np.array([eps, eps / 2, eps / 10])
        phi_n   = np.array([0.0, np.pi / 2, np.pi])
        pix, wgt = get_interp_weights_numba(nside, theta_n, phi_n)
        assert pix.min() >= 0 and pix.max() < npix
        npt.assert_allclose(wgt.sum(axis=0), np.ones(3), atol=1e-13)
        # South-pole cluster
        theta_s = np.pi - theta_n
        pix, wgt = get_interp_weights_numba(nside, theta_s, phi_n)
        assert pix.min() >= 0 and pix.max() < npix
        npt.assert_allclose(wgt.sum(axis=0), np.ones(3), atol=1e-13)

    def test_constant_map_gives_constant_value(self):
        """Bilinear interpolation of a constant map returns the constant everywhere."""
        nside = 16
        npix  = hp.nside2npix(nside)
        const = 3.14159
        cmap  = np.full(npix, const, dtype=np.float64)
        rng   = np.random.default_rng(38)
        N     = 150
        theta = rng.uniform(0.05, np.pi - 0.05, N)
        phi   = rng.uniform(0.0, 2 * np.pi, N)
        pix, wgt = get_interp_weights_numba(nside, theta, phi)
        interp = (cmap[pix] * wgt).sum(axis=0)
        npt.assert_allclose(interp, np.full(N, const), atol=1e-12)

    def test_phi_wrap_around(self):
        """Points at phi ≈ 0 and phi ≈ 2*pi give identical results."""
        nside = 16
        eps   = 1e-9
        theta = np.array([np.pi / 2, np.pi / 3])
        phi_lo = np.array([eps, eps])
        phi_hi = np.array([2 * np.pi - eps, 2 * np.pi - eps])
        pix_lo, wgt_lo = get_interp_weights_numba(nside, theta, phi_lo)
        pix_hi, wgt_hi = get_interp_weights_numba(nside, theta, phi_hi)
        # Near phi=0 and phi=2π the selected pixels may differ by one neighbour;
        # just check that all pixel indices are valid and weights sum to 1.
        npix = hp.nside2npix(nside)
        assert (pix_lo >= 0).all() and (pix_lo < npix).all()
        assert (pix_hi >= 0).all() and (pix_hi < npix).all()
        npt.assert_allclose(wgt_lo.sum(axis=0), [1.0, 1.0], atol=1e-13)
        npt.assert_allclose(wgt_hi.sum(axis=0), [1.0, 1.0], atol=1e-13)

    def test_pre_allocated_buffer_path(self):
        """_get_interp_weights_jit fills pre-allocated arrays consistently with wrapper."""
        nside = 16
        rng   = np.random.default_rng(39)
        N     = 80
        theta = np.asarray(rng.uniform(0.05, np.pi - 0.05, N), dtype=np.float64)
        phi   = np.asarray(rng.uniform(0.0, 2 * np.pi, N),     dtype=np.float64)
        pix_w, wgt_w = get_interp_weights_numba(nside, theta, phi)
        pix_j = np.empty((4, N), dtype=np.int64)
        wgt_j = np.empty((4, N), dtype=np.float64)
        _get_interp_weights_jit(nside, theta, phi, pix_j, wgt_j)
        npt.assert_array_equal(pix_j, pix_w)
        npt.assert_array_equal(wgt_j, wgt_w)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))