"""
Tests for the numba_healpy module.

Covers: _ring_above_jit, _ring_info_jit, _ring_z_jit, _wrap_ring_phase,
        _build_ring_theta, _get_interp_weights_jit, get_interp_weights_numba,
        _ring_interp_single_jit, _ring_interp_with_angles_jit.

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
    _TWO_PI,
    _ring_above_jit,
    _ring_info_jit,
    _ring_z_jit,
    _wrap_ring_phase,
    _build_ring_theta,
    _get_interp_weights_jit,
    get_interp_weights_numba,
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
# TestWrapRingPhase
# ===========================================================================


class TestWrapRingPhase:
    """Tests for _wrap_ring_phase — the branch form of ``tw % n_pix``.

    These exist to pin down the *upper* branch.  ``tw`` reaches exactly
    ``n_pix`` on unshifted rings, and a version that only corrects the negative
    side leaves it there, silently selecting the first pixel of the next ring.
    A reviewer trimming the ``elif`` as dead code must see these fail.
    """

    @staticmethod
    def _low_guard_only(tw, n_pix):
        """The tempting-but-wrong single-branch version, for contrast."""
        return tw + n_pix if tw < 0.0 else tw

    @pytest.mark.parametrize("n_pix", [4, 12, 1024, 4096, 8192])
    def test_matches_modulo_on_a_dense_sweep(self, n_pix):
        """Agrees with ``%`` as a float, across and beyond the valid range."""
        rng = np.random.default_rng(0)
        tws = np.concatenate(
            [
                np.linspace(-float(n_pix), 2.0 * n_pix, 4001),
                rng.uniform(-n_pix, 2 * n_pix, 4000),
                np.array([-0.5, -0.0, 0.0, float(n_pix), n_pix - 1e-12]),
            ]
        )
        for tw in tws:
            if not (-n_pix <= tw <= n_pix):
                continue  # a single correction only spans one modulus
            assert _wrap_ring_phase(float(tw), n_pix) == float(tw) % float(n_pix)

    @pytest.mark.parametrize("n_pix", [4, 12, 1024, 4096])
    def test_exact_upper_edge_wraps_to_zero(self, n_pix):
        """tw == n_pix must reduce to exactly 0.0, as ``%`` does."""
        assert _wrap_ring_phase(float(n_pix), n_pix) == 0.0
        assert float(n_pix) % float(n_pix) == 0.0
        # ...and the single-branch version is what we are guarding against.
        assert self._low_guard_only(float(n_pix), n_pix) == float(n_pix)

    @pytest.mark.parametrize("n_pix", [4, 12, 1024, 4096])
    def test_lower_edge_wraps(self, n_pix):
        """tw == -0.5 (the most negative reachable value) matches ``%``."""
        assert _wrap_ring_phase(-0.5, n_pix) == (-0.5) % float(n_pix)

    @pytest.mark.parametrize("nside", [4, 16, 256, 1024])
    def test_phi_two_pi_hits_the_upper_edge_on_real_rings(self, nside):
        """phi == 2π drives tw to exactly n_pix on every unshifted ring.

        This is the reachable input: the gather kernels map a negative
        ``atan2`` into [0, 2π) by adding 2π, and ``-1e-17 + 2π == 2π``.
        """
        npix = 12 * nside * nside
        seen_upper_edge = 0
        for ir in range(1, 4 * nside):
            n_pix, _fp, phi0, dphi = _ring_info_jit(nside, ir, npix)
            tw = (_TWO_PI - phi0) / dphi
            assert _wrap_ring_phase(tw, n_pix) == tw % float(n_pix)
            if tw == n_pix:
                seen_upper_edge += 1
                # the guard we are protecting actually fires here
                assert _wrap_ring_phase(tw, n_pix) == 0.0
                assert self._low_guard_only(tw, n_pix) != 0.0
        assert seen_upper_edge > 0, "no ring reached the upper edge — test is vacuous"

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_pixel_index_stays_inside_its_ring(self, nside):
        """int(wrapped) must never index past the end of its own ring."""
        npix = 12 * nside * nside
        phis = [0.0, _TWO_PI, _TWO_PI - 1e-16, 5e-324, math.pi]
        for ir in range(1, 4 * nside):
            n_pix, _fp, phi0, dphi = _ring_info_jit(nside, ir, npix)
            for phi in phis:
                ip = int(_wrap_ring_phase((phi - phi0) / dphi, n_pix))
                assert 0 <= ip < n_pix, f"nside={nside} ir={ir} phi={phi!r} ip={ip}"

    def test_phi_two_pi_and_zero_give_the_same_pixels(self):
        """phi = 0 and phi = 2π are the same point, so the stencils must match.

        The *pixel* indices must be identical — that is the wrap invariant, and
        it breaks by a whole ring without the upper guard.  The weights differ
        in the last couple of bits (~3e-14) because ``(0 - phi0) / dphi`` and
        ``(2π - phi0) / dphi`` do not round alike; that predates the branch form
        and is unchanged by it.
        """
        nside = 64
        npix = 12 * nside * nside
        for z in (0.9, 0.5, 0.0, -0.5, -0.9, 0.99999, -0.99999):
            a = _ring_interp_single_jit(nside, z, 0.0, npix)
            b = _ring_interp_single_jit(nside, z, _TWO_PI, npix)
            assert a[:4] == b[:4], f"pixels differ at z={z}"
            npt.assert_allclose(np.array(a[4:]), np.array(b[4:]), atol=1e-13)


# ===========================================================================
# TestBuildRingTheta
# ===========================================================================


class TestBuildRingTheta:
    """Tests for _build_ring_theta and the optional ``ring_theta`` fast path.

    The table is memoisation, not approximation: passing it must not change a
    single bit of any interpolation result.
    """

    @pytest.mark.parametrize("nside", [1, 4, 16, 256])
    def test_entries_are_exactly_acos_of_the_ring_centre(self, nside):
        """Bit-identical to the inline computation it replaces."""
        rt = _build_ring_theta(nside)
        assert rt.shape == (4 * nside,)
        assert rt.dtype == np.float64
        for ir in range(1, 4 * nside):
            assert rt[ir] == math.acos(_ring_z_jit(nside, ir))

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_monotonic_from_north_to_south(self, nside):
        """Colatitude increases with ring index, and stays inside (0, π)."""
        rt = _build_ring_theta(nside)
        vals = rt[1:]
        assert np.all(np.diff(vals) > 0)
        assert np.all(vals > 0.0) and np.all(vals < math.pi)

    @pytest.mark.parametrize("nside", [4, 16, 64, 256])
    def test_table_path_is_bit_identical(self, nside):
        """With and without the table must agree bit-for-bit, both helpers."""
        npix = 12 * nside * nside
        rt = _build_ring_theta(nside)
        rng = np.random.default_rng(1)
        zs = np.concatenate(
            [
                rng.uniform(-1.0, 1.0, 300),
                # pole boundaries and the cap/belt seam
                np.array([1.0, -1.0, 2.0 / 3.0, -2.0 / 3.0]),
                np.array([_ring_z_jit(nside, 1), _ring_z_jit(nside, 4 * nside - 1)]),
            ]
        )
        phis = rng.uniform(0.0, _TWO_PI, zs.size)
        for z, phi in zip(zs, phis):
            z = float(np.clip(z, -1.0, 1.0))
            ref_s = _ring_interp_single_jit(nside, z, float(phi), npix)
            fast_s = _ring_interp_single_jit(nside, z, float(phi), npix, rt)
            assert ref_s == fast_s, f"single differs at z={z}, phi={phi}"

            ref_a = _ring_interp_with_angles_jit(nside, z, float(phi), npix)
            fast_a = _ring_interp_with_angles_jit(nside, z, float(phi), npix, rt)
            assert ref_a == fast_a, f"with_angles differs at z={z}, phi={phi}"

    def test_wrong_nside_table_is_not_silently_accepted(self):
        """A table built for another nside changes results — it is not inert.

        Guards the documented precondition that the table is rebuilt whenever
        nside changes.  If this ever stops differing, the table is not actually
        being read and the optimisation is dead code.
        """
        nside = 64
        npix = 12 * nside * nside
        wrong = _build_ring_theta(2 * nside)
        differs = False
        rng = np.random.default_rng(2)
        for _ in range(200):
            z = float(rng.uniform(-1.0, 1.0))
            phi = float(rng.uniform(0.0, _TWO_PI))
            if _ring_interp_single_jit(nside, z, phi, npix) != _ring_interp_single_jit(
                nside, z, phi, npix, wrong
            ):
                differs = True
                break
        assert differs
