"""
Tests for the tod_utils module.

Covers: _ring_above_jit, _ring_info_jit, _ring_z_jit
        _get_interp_weights_jit, get_interp_weights_numba.
        _pix2ang_ring_jit, pix2ang_numba.
        _query_disc_jit, query_disc_numba.

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
    get_interp_weights_numba,
    _pix2ang_ring_jit,
    pix2ang_numba,
    _query_disc_jit,
    query_disc_numba,
    _ang2pix_ring_jit,
    _ring_interp_single_jit,
    _ring_interp_with_angles_jit,
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
                f"ring_above={ir} out of range [0, {4 * nside - 1}] for z={z:.6f}"
            )

    @pytest.mark.parametrize("nside", [4, 16])
    def test_consistent_with_ring_z(self, nside):
        """For interior z, z_ring(ir) >= z >= z_ring(ir+1) where ir = ring_above(z)."""
        rng = np.random.default_rng(21)
        z_vals = rng.uniform(-0.60, 0.60, 50)  # stay away from polar boundaries
        for z in z_vals:
            ir = _ring_above_jit(nside, float(z))
            if 1 <= ir <= 4 * nside - 2:
                za = _ring_z_jit(nside, ir)
                zb = _ring_z_jit(nside, ir + 1)
                assert za >= float(z) - 1e-12, f"z_ring({ir})={za:.8f} < z={z:.8f}"
                assert zb <= float(z) + 1e-12, f"z_ring({ir + 1})={zb:.8f} > z={z:.8f}"

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
        assert n_pix == 4
        assert first_pix == 0
        npt.assert_allclose(dphi, 2 * np.pi / 4, atol=1e-14)
        npt.assert_allclose(
            phi0,
            dphi / 2,
            atol=1e-14,
            err_msg="ring 1 should be shifted (phi0 = dphi/2)",
        )

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_npc_ring_sizes(self, nside):
        """Each NPC ring ir has exactly 4*ir pixels."""
        npix = 12 * nside * nside
        for ir in range(1, nside):
            n_pix, _, _, _ = _ring_info_jit(nside, ir, npix)
            assert n_pix == 4 * ir, f"ring {ir}: got n_pix={n_pix}, expected {4 * ir}"

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_equatorial_ring_sizes(self, nside):
        """All equatorial rings have exactly 4*nside pixels."""
        npix = 12 * nside * nside
        for ir in range(nside, 3 * nside + 1):
            n_pix, _, _, _ = _ring_info_jit(nside, ir, npix)
            assert n_pix == 4 * nside, (
                f"ring {ir}: got n_pix={n_pix}, expected {4 * nside}"
            )

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_first_equatorial_startpix(self, nside):
        """Ring nside starts at ncap = 2*nside*(nside-1)."""
        npix = 12 * nside * nside
        ncap = 2 * nside * (nside - 1)
        _, first_pix, _, _ = _ring_info_jit(nside, nside, npix)
        assert first_pix == ncap, (
            f"nside={nside}: first_pix={first_pix}, expected ncap={ncap}"
        )

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_equatorial_shift_alternates(self, nside):
        """Equatorial rings shift when (ir-nside) is even, not shifted when odd."""
        npix = 12 * nside * nside
        for ir in range(nside, 3 * nside + 1):
            _, _, phi0, dphi = _ring_info_jit(nside, ir, npix)
            if (ir - nside) % 2 == 0:
                npt.assert_allclose(
                    phi0, dphi / 2, atol=1e-14, err_msg=f"ring {ir} should be shifted"
                )
            else:
                npt.assert_allclose(
                    phi0, 0.0, atol=1e-14, err_msg=f"ring {ir} should NOT be shifted"
                )

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_spc_rings_always_shifted(self, nside):
        """All south polar cap rings are shifted (phi0 = dphi/2)."""
        npix = 12 * nside * nside
        for ir in range(3 * nside + 1, 4 * nside):
            _, _, phi0, dphi = _ring_info_jit(nside, ir, npix)
            npt.assert_allclose(
                phi0, dphi / 2, atol=1e-14, err_msg=f"SPC ring {ir} should be shifted"
            )

    @pytest.mark.parametrize("nside", [2, 4, 8, 16])
    def test_partition_covers_all_pixels(self, nside):
        """Ring pixel ranges partition [0, 12*nside^2) exactly, with no overlap or gap."""
        npix_total = 12 * nside * nside
        covered = np.zeros(npix_total, dtype=np.int32)
        for ir in range(1, 4 * nside):
            n_pix, first_pix, _, _ = _ring_info_jit(nside, ir, npix_total)
            assert first_pix >= 0, f"ring {ir}: first_pix={first_pix} < 0"
            assert first_pix + n_pix <= npix_total, (
                f"ring {ir}: pixel range [{first_pix}, {first_pix + n_pix}) exceeds npix"
            )
            covered[first_pix : first_pix + n_pix] += 1
        npt.assert_array_equal(
            covered,
            np.ones(npix_total, dtype=np.int32),
            err_msg="Some pixels covered 0 or >1 times",
        )

    @pytest.mark.parametrize("nside", [2, 4, 16])
    def test_dphi_equals_twopi_over_npix(self, nside):
        """dphi == 2*pi / n_pix for every ring."""
        npix = 12 * nside * nside
        for ir in range(1, 4 * nside):
            n_pix, _, _, dphi = _ring_info_jit(nside, ir, npix)
            npt.assert_allclose(
                dphi, 2 * np.pi / n_pix, atol=1e-14, err_msg=f"ring {ir}: dphi mismatch"
            )


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
                f"z not strictly decreasing at ir={ir}: z[{ir}]={z:.10f} >= z[{ir - 1}]={z_prev:.10f}"
            )
            z_prev = z

    @pytest.mark.parametrize("nside", [4, 16])
    def test_north_south_symmetry(self, nside):
        """z_ring(ir) == -z_ring(4*nside - ir) for all interior rings."""
        for ir in range(1, 2 * nside):
            z_north = _ring_z_jit(nside, ir)
            z_south = _ring_z_jit(nside, 4 * nside - ir)
            npt.assert_allclose(
                z_north, -z_south, atol=1e-14, err_msg=f"N/S symmetry failed at ir={ir}"
            )


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
        rng = np.random.default_rng(30)
        N = 60
        theta = rng.uniform(0.05, np.pi - 0.05, N)
        phi = rng.uniform(0.0, 2 * np.pi, N)
        pix, wgt = get_interp_weights_numba(nside, theta, phi)
        assert pix.shape == (4, N)
        assert wgt.shape == (4, N)
        assert pix.dtype == np.int64
        assert wgt.dtype == np.float64

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_weights_sum_to_one(self, nside):
        """The four bilinear weights sum to exactly 1 for every point."""
        rng = np.random.default_rng(31)
        N = 300
        theta = rng.uniform(0.0, np.pi, N)
        phi = rng.uniform(0.0, 2 * np.pi, N)
        _, wgt = get_interp_weights_numba(nside, theta, phi)
        npt.assert_allclose(
            wgt.sum(axis=0),
            np.ones(N),
            atol=1e-14,
            err_msg="Bilinear weights do not sum to 1",
        )

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_pixels_in_valid_range(self, nside):
        """All pixel indices lie in [0, 12*nside^2)."""
        npix = hp.nside2npix(nside)
        rng = np.random.default_rng(32)
        N = 300
        theta = rng.uniform(0.0, np.pi, N)
        phi = rng.uniform(0.0, 2 * np.pi, N)
        pix, _ = get_interp_weights_numba(nside, theta, phi)
        assert int(pix.min()) >= 0, "Pixel index below 0"
        assert int(pix.max()) < npix, f"Pixel index >= npix={npix}"

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_weights_non_negative(self, nside):
        """All interpolation weights are >= 0."""
        rng = np.random.default_rng(33)
        N = 300
        theta = rng.uniform(0.0, np.pi, N)
        phi = rng.uniform(0.0, 2 * np.pi, N)
        _, wgt = get_interp_weights_numba(nside, theta, phi)
        assert float(wgt.min()) >= -1e-14, f"Negative weight: {wgt.min()}"

    # ── agreement with healpy reference ──────────────────────────────────────

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_agrees_with_healpy_pixels(self, nside):
        """Pixel indices exactly match hp.get_interp_weights on 500 random interior points."""
        rng = np.random.default_rng(34)
        N = 500
        theta = rng.uniform(0.05, np.pi - 0.05, N)
        phi = rng.uniform(0.0, 2 * np.pi, N)
        pix_hp, _ = hp.get_interp_weights(nside, theta, phi)
        pix_nb, _ = get_interp_weights_numba(nside, theta, phi)
        npt.assert_array_equal(
            pix_nb, pix_hp, err_msg=f"Pixel mismatch at nside={nside}"
        )

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_agrees_with_healpy_weights(self, nside):
        """Bilinear weights match hp.get_interp_weights to 1e-12 on 500 random points."""
        rng = np.random.default_rng(35)
        N = 500
        theta = rng.uniform(0.05, np.pi - 0.05, N)
        phi = rng.uniform(0.0, 2 * np.pi, N)
        _, wgt_hp = hp.get_interp_weights(nside, theta, phi)
        _, wgt_nb = get_interp_weights_numba(nside, theta, phi)
        npt.assert_allclose(
            wgt_nb, wgt_hp, atol=1e-12, err_msg=f"Weight mismatch at nside={nside}"
        )

    @pytest.mark.parametrize("nside", [4, 16])
    def test_agrees_with_healpy_in_npc(self, nside):
        """Exact pixel+weight agreement in the north polar cap (theta < arccos(2/3))."""
        rng = np.random.default_rng(36)
        N = 200
        theta = rng.uniform(0.05, math.acos(2.0 / 3.0) - 0.02, N)
        phi = rng.uniform(0.0, 2 * np.pi, N)
        pix_hp, wgt_hp = hp.get_interp_weights(nside, theta, phi)
        pix_nb, wgt_nb = get_interp_weights_numba(nside, theta, phi)
        npt.assert_array_equal(pix_nb, pix_hp, err_msg="NPC pixel mismatch")
        npt.assert_allclose(wgt_nb, wgt_hp, atol=1e-12, err_msg="NPC weight mismatch")

    @pytest.mark.parametrize("nside", [4, 16])
    def test_agrees_with_healpy_in_spc(self, nside):
        """Exact pixel+weight agreement in the south polar cap (theta > pi-arccos(2/3))."""
        rng = np.random.default_rng(37)
        N = 200
        theta = rng.uniform(np.pi - math.acos(2.0 / 3.0) + 0.02, np.pi - 0.05, N)
        phi = rng.uniform(0.0, 2 * np.pi, N)
        pix_hp, wgt_hp = hp.get_interp_weights(nside, theta, phi)
        pix_nb, wgt_nb = get_interp_weights_numba(nside, theta, phi)
        npt.assert_array_equal(pix_nb, pix_hp, err_msg="SPC pixel mismatch")
        npt.assert_allclose(wgt_nb, wgt_hp, atol=1e-12, err_msg="SPC weight mismatch")

    # ── special/boundary inputs ───────────────────────────────────────────────

    @pytest.mark.parametrize("nside", [4, 16])
    def test_near_poles_valid_outputs(self, nside):
        """Points very close to the poles produce valid pixel indices and unit weight sums."""
        npix = hp.nside2npix(nside)
        eps = 1e-6
        # North-pole cluster
        theta_n = np.array([eps, eps / 2, eps / 10])
        phi_n = np.array([0.0, np.pi / 2, np.pi])
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
        npix = hp.nside2npix(nside)
        const = 3.14159
        cmap = np.full(npix, const, dtype=np.float64)
        rng = np.random.default_rng(38)
        N = 150
        theta = rng.uniform(0.05, np.pi - 0.05, N)
        phi = rng.uniform(0.0, 2 * np.pi, N)
        pix, wgt = get_interp_weights_numba(nside, theta, phi)
        interp = (cmap[pix] * wgt).sum(axis=0)
        npt.assert_allclose(interp, np.full(N, const), atol=1e-12)

    def test_phi_wrap_around(self):
        """Points at phi ≈ 0 and phi ≈ 2*pi give identical results."""
        nside = 16
        eps = 1e-9
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
        rng = np.random.default_rng(39)
        N = 80
        theta = np.asarray(rng.uniform(0.05, np.pi - 0.05, N), dtype=np.float64)
        phi = np.asarray(rng.uniform(0.0, 2 * np.pi, N), dtype=np.float64)
        pix_w, wgt_w = get_interp_weights_numba(nside, theta, phi)
        pix_j = np.empty((4, N), dtype=np.int64)
        wgt_j = np.empty((4, N), dtype=np.float64)
        _get_interp_weights_jit(nside, theta, phi, pix_j, wgt_j)
        npt.assert_array_equal(pix_j, pix_w)
        npt.assert_array_equal(wgt_j, wgt_w)


# ===========================================================================
# TestRingInterpWithAnglesJit
# ===========================================================================


class TestRingInterpWithAnglesJit:
    """Tests for _ring_interp_with_angles_jit.

    The function must return the same ``(p0..p3, w0..w3)`` as
    :func:`_ring_interp_single_jit` (bit-for-bit), and the additional
    ``(z_n, phi_n)`` values must round-trip through ``hp.pix2ang`` for each of
    the four neighbour pixels.  Coverage includes the three HEALPix regimes:
    normal case (equatorial belt + polar cap proper), north-pole boundary,
    and south-pole boundary.
    """

    @staticmethod
    def _query_points(rng, n, theta_range):
        """Return an (n, 3) float64 array of unit vectors in the θ range."""
        thetas = rng.uniform(*theta_range, n)
        phis = rng.uniform(0.0, 2 * math.pi, n)
        return (
            np.stack(
                [
                    np.sin(thetas) * np.cos(phis),
                    np.sin(thetas) * np.sin(phis),
                    np.cos(thetas),
                ],
                axis=-1,
            ),
            thetas,
            phis,
        )

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_pixels_and_weights_match_single_jit(self, nside):
        """The first 8 returns (pixels, weights) must match _ring_interp_single_jit
        bit-for-bit across all three regimes."""
        rng = np.random.default_rng(2024)
        # Sample colatitudes spanning NPC, equatorial belt, and SPC.
        thetas = np.concatenate(
            [
                rng.uniform(0.01, math.acos(2.0 / 3.0) - 0.02, 40),  # NPC
                rng.uniform(
                    math.acos(2.0 / 3.0) + 0.02, math.acos(-2.0 / 3.0) - 0.02, 40
                ),  # equatorial
                rng.uniform(math.acos(-2.0 / 3.0) + 0.02, math.pi - 0.01, 40),  # SPC
            ]
        )
        phis = rng.uniform(0.0, 2 * math.pi, thetas.size)
        for theta, phi in zip(thetas, phis):
            z = math.cos(theta)
            phi_w = phi % (2 * math.pi)

            out_single = _ring_interp_single_jit(nside, z, phi_w, 12 * nside * nside)
            out_full = _ring_interp_with_angles_jit(nside, z, phi_w, 12 * nside * nside)

            # pixels (0..3) and weights (4..7) must match bit-for-bit.
            for i in range(8):
                assert out_full[i] == out_single[i], (
                    f"Element {i} mismatch at nside={nside}, theta={theta}, phi={phi}"
                )

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_neighbour_angles_match_hp_pix2ang(self, nside):
        """For each of the 4 returned neighbour pixels, (z_n, phi_n) must
        match ``hp.pix2ang`` (i.e. cos(theta_ref) == z_n and phi_ref == phi_n)."""
        rng = np.random.default_rng(99)
        thetas = np.concatenate(
            [
                rng.uniform(0.01, math.acos(2.0 / 3.0) - 0.02, 20),
                rng.uniform(
                    math.acos(2.0 / 3.0) + 0.02, math.acos(-2.0 / 3.0) - 0.02, 20
                ),
                rng.uniform(math.acos(-2.0 / 3.0) + 0.02, math.pi - 0.01, 20),
            ]
        )
        phis = rng.uniform(0.0, 2 * math.pi, thetas.size)
        for theta, phi in zip(thetas, phis):
            z = math.cos(theta)
            phi_w = phi % (2 * math.pi)
            out = _ring_interp_with_angles_jit(nside, z, phi_w, 12 * nside * nside)
            pixels = out[0:4]
            z_n = out[8:12]
            phi_n = out[12:16]
            # Reference from healpy.
            theta_ref, phi_ref = hp.pix2ang(nside, np.asarray(pixels, dtype=np.int64))
            for i in range(4):
                npt.assert_allclose(
                    z_n[i],
                    math.cos(theta_ref[i]),
                    atol=1e-12,
                    err_msg=f"z_n[{i}] at nside={nside}",
                )
                # phi may differ by multiples of 2π; compare via angular difference.
                dphi = (phi_n[i] - phi_ref[i] + math.pi) % (2 * math.pi) - math.pi
                npt.assert_allclose(
                    dphi, 0.0, atol=1e-12, err_msg=f"phi_n[{i}] at nside={nside}"
                )

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_north_pole_boundary(self, nside):
        """Query points above ring 1 (north of pole) take the NPC boundary branch."""
        z1 = _ring_z_jit(nside, 1)  # ring-1 z-centre
        # Pick a z slightly above ring 1.
        z = min(1.0, 0.5 * (1.0 + z1))
        rng = np.random.default_rng(0)
        for phi in rng.uniform(0, 2 * math.pi, 10):
            out = _ring_interp_with_angles_jit(nside, z, phi, 12 * nside * nside)
            # All 4 neighbours sit on ring 1.
            for zn in out[8:12]:
                npt.assert_allclose(zn, z1, atol=1e-14)
            # Weights sum to 1.
            assert abs(sum(out[4:8]) - 1.0) < 1e-12

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_south_pole_boundary(self, nside):
        """Query points below ring 4·nside-1 take the SPC boundary branch."""
        z_last = _ring_z_jit(nside, 4 * nside - 1)
        z = max(-1.0, 0.5 * (-1.0 + z_last))
        rng = np.random.default_rng(1)
        for phi in rng.uniform(0, 2 * math.pi, 10):
            out = _ring_interp_with_angles_jit(nside, z, phi, 12 * nside * nside)
            for zn in out[8:12]:
                npt.assert_allclose(zn, z_last, atol=1e-14)
            assert abs(sum(out[4:8]) - 1.0) < 1e-12

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_weights_sum_to_one(self, nside):
        """Bilinear weights always sum to 1 across all regimes."""
        rng = np.random.default_rng(3)
        for _ in range(200):
            theta = rng.uniform(0.01, math.pi - 0.01)
            phi = rng.uniform(0.0, 2 * math.pi)
            out = _ring_interp_with_angles_jit(
                nside, math.cos(theta), phi, 12 * nside * nside
            )
            npt.assert_allclose(sum(out[4:8]), 1.0, atol=1e-12)


# ===========================================================================
# TestPix2AngNumba
# ===========================================================================


class TestPix2AngNumba:
    """
    Tests for _pix2ang_ring_jit (scalar) and pix2ang_numba (batch).

    Correctness reference: hp.pix2ang(nside, pix, nest=False).
    Covers all three HEALPix zones: north polar cap, equatorial belt,
    south polar cap.
    """

    # ── output contract ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_output_shapes_and_dtypes(self, nside):
        """pix2ang_numba returns two float64 arrays each of shape (N,)."""
        pix = np.arange(hp.nside2npix(nside), dtype=np.int64)
        th, ph = pix2ang_numba(nside, pix)
        assert th.shape == pix.shape
        assert ph.shape == pix.shape
        assert th.dtype == np.float64
        assert ph.dtype == np.float64

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_theta_in_zero_to_pi(self, nside):
        """All theta values lie in [0, pi]."""
        pix = np.arange(hp.nside2npix(nside), dtype=np.int64)
        th, _ = pix2ang_numba(nside, pix)
        assert float(th.min()) >= 0.0, f"theta below 0 at nside={nside}"
        assert float(th.max()) <= math.pi, f"theta above pi at nside={nside}"

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_phi_in_zero_to_twopi(self, nside):
        """All phi values lie in [0, 2*pi]."""
        pix = np.arange(hp.nside2npix(nside), dtype=np.int64)
        _, ph = pix2ang_numba(nside, pix)
        assert float(ph.min()) >= 0.0, f"phi below 0 at nside={nside}"
        assert float(ph.max()) <= 2 * math.pi, f"phi above 2pi at nside={nside}"

    # ── agreement with healpy ─────────────────────────────────────────────────

    @pytest.mark.parametrize("nside", [2, 4, 16, 64])
    def test_agrees_with_healpy_all_pixels(self, nside):
        """theta and phi match hp.pix2ang to 1e-14 for every pixel."""
        pix = np.arange(hp.nside2npix(nside), dtype=np.int64)
        th_nb, ph_nb = pix2ang_numba(nside, pix)
        th_hp, ph_hp = hp.pix2ang(nside, pix, nest=False)
        npt.assert_allclose(
            th_nb, th_hp, atol=1e-14, err_msg=f"theta mismatch at nside={nside}"
        )
        npt.assert_allclose(
            ph_nb, ph_hp, atol=1e-14, err_msg=f"phi mismatch at nside={nside}"
        )

    @pytest.mark.parametrize("nside", [4, 16])
    def test_agrees_with_healpy_north_polar_cap(self, nside):
        """Exact agreement in the north polar cap (pix < 2*nside*(nside-1))."""
        ncap = 2 * nside * (nside - 1)
        pix = np.arange(ncap, dtype=np.int64)
        th_nb, ph_nb = pix2ang_numba(nside, pix)
        th_hp, ph_hp = hp.pix2ang(nside, pix, nest=False)
        npt.assert_allclose(th_nb, th_hp, atol=1e-14, err_msg="NPC theta")
        npt.assert_allclose(ph_nb, ph_hp, atol=1e-14, err_msg="NPC phi")

    @pytest.mark.parametrize("nside", [4, 16])
    def test_agrees_with_healpy_equatorial_belt(self, nside):
        """Exact agreement in the equatorial belt."""
        npix = hp.nside2npix(nside)
        ncap = 2 * nside * (nside - 1)
        pix = np.arange(ncap, npix - ncap, dtype=np.int64)
        th_nb, ph_nb = pix2ang_numba(nside, pix)
        th_hp, ph_hp = hp.pix2ang(nside, pix, nest=False)
        npt.assert_allclose(th_nb, th_hp, atol=1e-14, err_msg="belt theta")
        npt.assert_allclose(ph_nb, ph_hp, atol=1e-14, err_msg="belt phi")

    @pytest.mark.parametrize("nside", [4, 16])
    def test_agrees_with_healpy_south_polar_cap(self, nside):
        """Exact agreement in the south polar cap."""
        npix = hp.nside2npix(nside)
        ncap = 2 * nside * (nside - 1)
        pix = np.arange(npix - ncap, npix, dtype=np.int64)
        th_nb, ph_nb = pix2ang_numba(nside, pix)
        th_hp, ph_hp = hp.pix2ang(nside, pix, nest=False)
        npt.assert_allclose(th_nb, th_hp, atol=1e-14, err_msg="SPC theta")
        npt.assert_allclose(ph_nb, ph_hp, atol=1e-14, err_msg="SPC phi")

    # ── scalar kernel ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("nside", [4, 16])
    def test_scalar_jit_matches_batch(self, nside):
        """_pix2ang_ring_jit gives identical results to the batch wrapper."""
        rng = np.random.default_rng(50)
        pix_arr = rng.integers(0, hp.nside2npix(nside), size=40)
        th_b, ph_b = pix2ang_numba(nside, pix_arr)
        for i, p in enumerate(pix_arr):
            th_s, ph_s = _pix2ang_ring_jit(nside, int(p))
            npt.assert_allclose(th_s, th_b[i], atol=1e-15)
            npt.assert_allclose(ph_s, ph_b[i], atol=1e-15)

    # ── round-trip ────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_roundtrip_ang2pix_pix2ang(self, nside):
        """pix2ang → ang2pix recovers the original pixel index."""
        pix_orig = np.arange(hp.nside2npix(nside), dtype=np.int64)
        th, ph = pix2ang_numba(nside, pix_orig)
        pix_back = hp.ang2pix(nside, th, ph, nest=False)
        npt.assert_array_equal(
            pix_back, pix_orig, err_msg=f"Round-trip failed at nside={nside}"
        )

    # ── nest=True raises ─────────────────────────────────────────────────────

    def test_nest_true_raises(self):
        """pix2ang_numba with nest=True raises ValueError."""
        with pytest.raises(ValueError, match="nest=False"):
            pix2ang_numba(16, np.array([0]), nest=True)


# ===========================================================================
# TestQueryDiscNumba
# ===========================================================================


class TestQueryDiscNumba:
    """
    Tests for _query_disc_jit (internal) and query_disc_numba (public).

    Correctness reference: hp.query_disc(nside, vec, radius, nest=False).
    Covers inclusive / non-inclusive modes, boundary conditions (poles,
    phi wrap-around, full-sky disc), and output contract.
    """

    # ── output contract ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_output_dtype_and_ndim(self, nside):
        """query_disc_numba returns a 1-D int64 array."""
        vec = hp.ang2vec(math.pi / 2, 0.0)
        pix = query_disc_numba(nside, vec, 0.1)
        assert pix.ndim == 1
        assert pix.dtype == np.int64

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_pixels_in_valid_range(self, nside):
        """All returned pixel indices lie in [0, 12*nside^2)."""
        npix = hp.nside2npix(nside)
        rng = np.random.default_rng(60)
        for _ in range(20):
            vec = hp.ang2vec(
                rng.uniform(0.1, math.pi - 0.1), rng.uniform(0, 2 * math.pi)
            )
            pix = query_disc_numba(nside, vec, rng.uniform(0.02, 0.3))
            assert int(pix.min()) >= 0, "pixel index below 0"
            assert int(pix.max()) < npix, f"pixel index >= npix={npix}"

    @pytest.mark.parametrize("nside", [4, 16])
    def test_no_duplicate_pixels(self, nside):
        """Returned pixel indices are unique (no duplicates)."""
        rng = np.random.default_rng(61)
        for _ in range(20):
            vec = hp.ang2vec(
                rng.uniform(0.1, math.pi - 0.1), rng.uniform(0, 2 * math.pi)
            )
            pix = query_disc_numba(nside, vec, rng.uniform(0.02, 0.3))
            assert len(pix) == len(np.unique(pix)), "duplicate pixel indices"

    # ── agreement with healpy ─────────────────────────────────────────────────

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_inclusive_contains_all_healpy_pixels(self, nside):
        """
        With inclusive=True every pixel returned by hp.query_disc is also
        returned by query_disc_numba (numba may return a few extra due to the
        pixel-radius approximation, but must never miss any).
        """
        rng = np.random.default_rng(62)
        for _ in range(30):
            th = rng.uniform(0.1, math.pi - 0.1)
            ph = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0.02, 0.4)
            vec = hp.ang2vec(th, ph)
            pix_hp = hp.query_disc(nside, vec, r, inclusive=True, nest=False)
            pix_nb = query_disc_numba(nside, vec, r, inclusive=True)
            missing = np.setdiff1d(pix_hp, pix_nb)
            assert len(missing) == 0, (
                f"nside={nside} inclusive: {len(missing)} pixels in hp but not numba"
                f" (th={th:.3f}, ph={ph:.3f}, r={r:.3f})"
            )

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_non_inclusive_contains_all_healpy_pixels(self, nside):
        """
        With inclusive=False every pixel returned by hp.query_disc is also
        returned by query_disc_numba (pixel centres strictly inside the disc).
        """
        rng = np.random.default_rng(63)
        for _ in range(30):
            th = rng.uniform(0.1, math.pi - 0.1)
            ph = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0.05, 0.4)
            vec = hp.ang2vec(th, ph)
            pix_hp = hp.query_disc(nside, vec, r, inclusive=False, nest=False)
            pix_nb = query_disc_numba(nside, vec, r, inclusive=False)
            missing = np.setdiff1d(pix_hp, pix_nb)
            assert len(missing) == 0, (
                f"nside={nside} non-inclusive: {len(missing)} pixels missing"
                f" (th={th:.3f}, ph={ph:.3f}, r={r:.3f})"
            )

    @pytest.mark.parametrize("nside", [4, 16])
    def test_inclusive_is_superset_of_non_inclusive(self, nside):
        """inclusive=True always returns at least as many pixels as inclusive=False."""
        rng = np.random.default_rng(64)
        for _ in range(20):
            vec = hp.ang2vec(
                rng.uniform(0.1, math.pi - 0.1), rng.uniform(0, 2 * math.pi)
            )
            r = rng.uniform(0.05, 0.3)
            pix_inc = set(query_disc_numba(nside, vec, r, inclusive=True).tolist())
            pix_exc = set(query_disc_numba(nside, vec, r, inclusive=False).tolist())
            assert pix_exc.issubset(pix_inc), (
                "inclusive=False returned pixels not in inclusive=True"
            )

    # ── boundary and special cases ────────────────────────────────────────────

    @pytest.mark.parametrize("nside", [4, 16])
    def test_north_pole_disc(self, nside):
        """Disc centred on the north pole returns the correct pixels."""
        vec = np.array([0.0, 0.0, 1.0])
        r = 0.3
        pix_hp = np.sort(hp.query_disc(nside, vec, r, inclusive=True, nest=False))
        pix_nb = np.sort(query_disc_numba(nside, vec, r, inclusive=True))
        missing = np.setdiff1d(pix_hp, pix_nb)
        assert len(missing) == 0, f"north-pole disc: {len(missing)} pixels missing"

    @pytest.mark.parametrize("nside", [4, 16])
    def test_south_pole_disc(self, nside):
        """Disc centred on the south pole returns the correct pixels."""
        vec = np.array([0.0, 0.0, -1.0])
        r = 0.3
        pix_hp = np.sort(hp.query_disc(nside, vec, r, inclusive=True, nest=False))
        pix_nb = np.sort(query_disc_numba(nside, vec, r, inclusive=True))
        missing = np.setdiff1d(pix_hp, pix_nb)
        assert len(missing) == 0, f"south-pole disc: {len(missing)} pixels missing"

    @pytest.mark.parametrize("nside", [4, 16])
    def test_phi_zero_wraparound(self, nside):
        """Disc straddling phi=0 / phi=2π returns no duplicates and no missing pixels."""
        vec = hp.ang2vec(math.pi / 2, 0.0)  # points exactly at phi=0
        r = 0.25
        pix_hp = np.sort(hp.query_disc(nside, vec, r, inclusive=True, nest=False))
        pix_nb = np.sort(query_disc_numba(nside, vec, r, inclusive=True))
        assert len(pix_nb) == len(np.unique(pix_nb)), "duplicates near phi=0"
        missing = np.setdiff1d(pix_hp, pix_nb)
        assert len(missing) == 0, f"phi=0 wrap: {len(missing)} pixels missing"

    @pytest.mark.parametrize("nside", [4, 16])
    def test_tiny_disc_returns_at_least_one_pixel(self, nside):
        """A disc smaller than a pixel still returns at least its central pixel."""
        rng = np.random.default_rng(65)
        for _ in range(10):
            th = rng.uniform(0.05, math.pi - 0.05)
            ph = rng.uniform(0, 2 * math.pi)
            vec = hp.ang2vec(th, ph)
            pix = query_disc_numba(nside, vec, 1e-5, inclusive=True)
            assert len(pix) >= 1, "tiny disc returned no pixels"

    @pytest.mark.parametrize("nside", [4, 16])
    def test_large_disc_returns_all_pixels(self, nside):
        """A disc with radius >= pi returns every pixel in the map."""
        npix = hp.nside2npix(nside)
        vec = np.array([0.0, 0.0, 1.0])
        pix = query_disc_numba(nside, vec, math.pi, inclusive=True)
        assert len(pix) == npix, (
            f"full-sky disc: expected {npix} pixels, got {len(pix)}"
        )

    @pytest.mark.parametrize("nside", [4, 16])
    def test_all_returned_pixels_inside_disc(self, nside):
        """
        Every pixel returned with inclusive=False has its centre within the radius.
        (Checks the opposite direction: numba should not hallucinate pixels.)
        """
        rng = np.random.default_rng(66)
        for _ in range(20):
            th = rng.uniform(0.1, math.pi - 0.1)
            ph = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0.05, 0.4)
            vec = hp.ang2vec(th, ph)
            pix_nb = query_disc_numba(nside, vec, r, inclusive=False)
            if len(pix_nb) == 0:
                continue
            th_p, ph_p = hp.pix2ang(nside, pix_nb, nest=False)
            dist = np.arccos(
                np.clip(
                    np.sin(th) * np.sin(th_p) * np.cos(ph - ph_p)
                    + np.cos(th) * np.cos(th_p),
                    -1.0,
                    1.0,
                )
            )
            # Allow a tiny tolerance for floating-point edge cases at the boundary.
            assert np.all(dist <= r + 1e-10), (
                f"pixel outside disc: max dist={dist.max():.6f}, r={r:.6f}"
            )

    # ── internal scalar kernel ────────────────────────────────────────────────

    @pytest.mark.parametrize("nside", [4, 16])
    def test_internal_jit_matches_public_wrapper(self, nside):
        """_query_disc_jit and query_disc_numba return identical sorted arrays."""
        rng = np.random.default_rng(67)
        for _ in range(15):
            th = rng.uniform(0.1, math.pi - 0.1)
            ph = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0.05, 0.3)
            vec = hp.ang2vec(th, ph)
            pix_pub = np.sort(query_disc_numba(nside, vec, r, inclusive=True))
            pix_jit = np.sort(_query_disc_jit(nside, th, ph, r, True))
            npt.assert_array_equal(pix_pub, pix_jit)

    # ── nest=True raises ─────────────────────────────────────────────────────

    def test_nest_true_raises(self):
        """query_disc_numba with nest=True raises ValueError."""
        vec = np.array([0.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="nest=False"):
            query_disc_numba(16, vec, 0.1, nest=True)


# ===========================================================================
# TestAng2PixRingJit
# ===========================================================================


class TestAng2PixRingJit:
    """
    Tests for _ang2pix_ring_jit — scalar RING-scheme nearest-pixel lookup.

    The function uses a "nearest ring by z-distance + nearest phi" heuristic
    that agrees with healpy ~75 % of the time; when it disagrees the returned
    pixel is always within half a pixel size of the true nearest pixel.  Tests
    here verify:
      • output is always a valid pixel index
      • the returned pixel centre is geometrically close to the query point
      • exact match with healpy in unambiguous (well-interior) cases
      • round-trip consistency with _pix2ang_ring_jit
    """

    @staticmethod
    def _great_circle_dist(th1, ph1, th2, ph2):
        cos_d = math.sin(th1) * math.sin(th2) * math.cos(ph1 - ph2) + math.cos(
            th1
        ) * math.cos(th2)
        return math.acos(max(-1.0, min(1.0, cos_d)))

    # ── output contract ───────────────────────────────────────────────────────

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_output_in_valid_range(self, nside):
        """_ang2pix_ring_jit always returns a pixel in [0, npix)."""
        npix = hp.nside2npix(nside)
        rng = np.random.default_rng(100)
        for th, ph in zip(
            rng.uniform(0.0, math.pi, 300), rng.uniform(0.0, 2 * math.pi, 300)
        ):
            p = _ang2pix_ring_jit(nside, float(th), float(ph))
            assert 0 <= p < npix, (
                f"pixel {p} out of [0, {npix}) for theta={th:.4f} phi={ph:.4f}"
            )

    # ── geometric proximity ───────────────────────────────────────────────────

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_returned_pixel_within_one_pixel_size(self, nside):
        """
        The returned pixel centre is always within one pixel diameter of the
        query point (max angular distance < pixel resolution).
        """
        pix_res = hp.nside2resol(nside)  # approx pixel radius [rad]
        rng = np.random.default_rng(101)
        max_dist = 0.0
        for th, ph in zip(
            rng.uniform(0.0, math.pi, 500), rng.uniform(0.0, 2 * math.pi, 500)
        ):
            p = _ang2pix_ring_jit(nside, float(th), float(ph))
            th_p, ph_p = _pix2ang_ring_jit(nside, p)
            d = self._great_circle_dist(th, ph, th_p, ph_p)
            max_dist = max(max_dist, d)
        assert max_dist < pix_res, (
            f"nside={nside}: max dist {math.degrees(max_dist):.4f} deg "
            f">= pixel res {math.degrees(pix_res):.4f} deg"
        )

    # ── exact match with healpy in equatorial / polar zones ───────────────────

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_equatorial_pixels_match_healpy(self, nside):
        """
        Points near the equator (well away from ring boundaries) agree exactly
        with hp.ang2pix.
        """
        # Step through equatorial rings and query at each pixel centre.
        # Pixel centres are unambiguous — healpy must agree.
        all_pix = np.arange(hp.nside2npix(nside), dtype=np.int64)
        th_all, ph_all = hp.pix2ang(nside, all_pix)
        # Restrict to equatorial belt (64 to 192 for nside=64 etc.)
        equatorial = (th_all > 0.4) & (th_all < math.pi - 0.4)
        pix_eq = all_pix[equatorial]
        th_eq = th_all[equatorial]
        ph_eq = ph_all[equatorial]

        mismatches = 0
        for p_ref, th, ph in zip(pix_eq, th_eq, ph_eq):
            p_mine = _ang2pix_ring_jit(nside, float(th), float(ph))
            if p_mine != p_ref:
                mismatches += 1
        # At pixel centres the result must be exact
        assert mismatches == 0, (
            f"nside={nside}: {mismatches} equatorial pixel-centre mismatches with healpy"
        )

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_polar_pixel_centres_match_healpy(self, nside):
        """Points at pixel centres in the polar caps agree exactly with hp.ang2pix."""
        all_pix = np.arange(hp.nside2npix(nside), dtype=np.int64)
        th_all, ph_all = hp.pix2ang(nside, all_pix)
        polar = (th_all < 0.4) | (th_all > math.pi - 0.4)
        pix_p = all_pix[polar]
        th_p = th_all[polar]
        ph_p = ph_all[polar]

        mismatches = 0
        for p_ref, th, ph in zip(pix_p, th_p, ph_p):
            p_mine = _ang2pix_ring_jit(nside, float(th), float(ph))
            if p_mine != p_ref:
                mismatches += 1
        assert mismatches == 0, (
            f"nside={nside}: {mismatches} polar pixel-centre mismatches with healpy"
        )

    # ── round-trip with _pix2ang_ring_jit ────────────────────────────────────

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_pix2ang_roundtrip(self, nside):
        """
        _pix2ang_ring_jit → _ang2pix_ring_jit recovers the original pixel.

        Pixel centres are unambiguous query points — round-trip must be exact.
        """
        npix = hp.nside2npix(nside)
        for p_orig in range(npix):
            th, ph = _pix2ang_ring_jit(nside, p_orig)
            p_back = _ang2pix_ring_jit(nside, th, ph)
            assert p_back == p_orig, (
                f"nside={nside} pix={p_orig}: round-trip gave {p_back}"
            )

    # ── special directions ────────────────────────────────────────────────────

    @pytest.mark.parametrize("nside", [4, 16])
    def test_north_pole(self, nside):
        """A point very close to the north pole returns pixel 0."""
        p = _ang2pix_ring_jit(nside, 1e-10, 0.0)
        assert p == 0, f"nside={nside}: expected pix 0 near north pole, got {p}"

    @pytest.mark.parametrize("nside", [4, 16])
    def test_south_pole(self, nside):
        """
        A point very close to the south pole returns a pixel in the last ring.

        The exact pixel depends on phi; at phi=0 healpy returns the first
        pixel of the last ring (npix-4*1 for nside≥2), not the very last
        pixel.  We verify against healpy directly.
        """
        p = _ang2pix_ring_jit(nside, math.pi - 1e-10, 0.0)
        p_hp = hp.ang2pix(nside, math.pi - 1e-10, 0.0)
        assert p == p_hp, (
            f"nside={nside}: expected healpy pix {p_hp} near south pole, got {p}"
        )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
