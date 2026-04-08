"""
Tests for bicubic (Keys/Catmull-Rom) interpolation in tod_core and numba_healpy.

Covers:
  _keys_1d_jit                — Keys cubic kernel (tod_core)
  _gather_ring_stencil_jit    — ring-walk stencil gather (numba_healpy)
  _bicubic_interp_accum       — bicubic accumulation wrapper (tod_core)
  beam_tod_batch(interp_mode='bicubic')
      - constant-map reconstruction
      - agreement with bilinear on smooth maps
      - rotational stability (primary science validation criterion)

Can be run independently:
    pytest tests/test_bicubic_interp.py -v
    python tests/test_bicubic_interp.py
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
import numba
import pytest

from tod_bicubic import _bicubic_interp_accum, _keys_1d_jit
from tod_core import beam_tod_batch, precompute_rotation_vector_batch
from numba_healpy import _gather_ring_stencil_jit

# ---------------------------------------------------------------------------
# Shared helpers (mirror test_gaussian_interp.py conventions)
# ---------------------------------------------------------------------------


def _build_data(S=30, nside=32):
    """Synthetic beam data dict with S pixels near the north pole."""
    rng = np.random.default_rng(99)
    theta_beam = rng.uniform(0.0, 0.05, S)
    phi_beam = rng.uniform(0, 2 * np.pi, S)
    vec_orig = np.stack(
        [
            np.sin(theta_beam) * np.cos(phi_beam),
            np.sin(theta_beam) * np.sin(phi_beam),
            np.cos(theta_beam),
        ],
        axis=-1,
    ).astype(np.float32)
    beam_vals = rng.uniform(0.5, 1.5, S).astype(np.float32)
    beam_vals /= beam_vals.sum()
    return {
        "vec_orig": vec_orig,
        "beam_vals": beam_vals,
        "comp_indices": ["I", "Q", "U"],
        "mp_stacked": None,
    }


def _build_scan(B=8, N=201):
    """B scan directions near the equator."""
    rng = np.random.default_rng(77)
    ra = np.zeros((N, N))
    dec = np.zeros((N, N))
    phi_batch = rng.uniform(0, 0.04, B)
    theta_batch = rng.uniform(np.pi / 2 - 0.04, np.pi / 2, B)
    rot_vecs, betas = precompute_rotation_vector_batch(
        ra, dec, phi_batch, theta_batch, center_idx=(N // 2, N // 2)
    )
    return phi_batch, theta_batch, -betas, rot_vecs


def _constant_maps(nside=32):
    npix = hp.nside2npix(nside)
    return {c: np.ones(npix, dtype=np.float32) for c in ["I", "Q", "U"]}


def _mp_stacked(mp, comp_indices):
    return np.stack([mp[c] for c in comp_indices]).astype(np.float32)


# ===========================================================================
# TestKeys1DKernel
# ===========================================================================


class TestKeys1DKernel:
    """Tests for tod_core._keys_1d_jit (Keys/Catmull-Rom 1-D kernel)."""

    def test_value_at_zero(self):
        """K(0) = 1 (kernel is an interpolant)."""
        assert _keys_1d_jit(0.0) == pytest.approx(1.0)

    def test_value_at_one(self):
        """K(1) = 0 (zero at integer distances ≥ 1)."""
        assert _keys_1d_jit(1.0) == pytest.approx(0.0, abs=1e-15)

    def test_value_at_two(self):
        """K(2) = 0 (kernel vanishes at the support boundary)."""
        assert _keys_1d_jit(2.0) == pytest.approx(0.0, abs=1e-15)

    def test_zero_outside_support(self):
        """K(t) = 0 for |t| ≥ 2."""
        for t in [2.0, 2.5, 3.0, 10.0]:
            assert _keys_1d_jit(t) == 0.0, f"Expected 0 for t={t}"
            assert _keys_1d_jit(-t) == 0.0, f"Expected 0 for t={-t}"

    def test_symmetry(self):
        """K(-t) = K(t) for all t."""
        for t in [0.0, 0.3, 0.7, 1.0, 1.5, 1.9, 2.0, 2.5]:
            assert _keys_1d_jit(-t) == pytest.approx(_keys_1d_jit(t), rel=1e-12)

    def test_continuity_at_t1(self):
        """K is C0 at t=1: left and right limits match."""
        eps = 1e-8
        left = _keys_1d_jit(1.0 - eps)
        right = _keys_1d_jit(1.0 + eps)
        assert abs(left - right) < 1e-6

    def test_inner_branch_formula(self):
        """K(0.5) matches the analytic formula for |t| < 1: (3/2)|t|^3 - (5/2)|t|^2 + 1."""
        t = 0.5
        expected = 1.5 * t**3 - 2.5 * t**2 + 1.0
        assert _keys_1d_jit(t) == pytest.approx(expected, rel=1e-12)

    def test_outer_branch_formula(self):
        """K(1.5) matches the analytic formula for 1 ≤ |t| < 2: -0.5|t|^3 + 2.5|t|^2 - 4|t| + 2."""
        t = 1.5
        expected = -0.5 * t**3 + 2.5 * t**2 - 4.0 * t + 2.0
        assert _keys_1d_jit(t) == pytest.approx(expected, rel=1e-12)

    def test_values_in_minus1_1_range(self):
        """K(t) is in [−0.5, 1.0] for t in (−2, 2) (bounded kernel)."""
        ts = np.linspace(-1.99, 1.99, 400)
        for t in ts:
            v = _keys_1d_jit(float(t))
            assert -0.5 - 1e-10 <= v <= 1.0 + 1e-10, (
                f"K({t}) = {v} out of expected range"
            )


# ===========================================================================
# TestGatherRingStencil
# ===========================================================================


class TestGatherRingStencil:
    """Tests for numba_healpy._gather_ring_stencil_jit."""

    @staticmethod
    def _call(nside, vz, ph):
        """Convenience wrapper that allocates buffers and returns (M, pix, z, phi)."""
        out_buf = np.empty(64, dtype=np.int64)
        z_buf = np.empty(64, dtype=np.float64)
        phi_buf = np.empty(64, dtype=np.float64)
        M = _gather_ring_stencil_jit(nside, vz, ph, out_buf, z_buf, phi_buf)
        return M, out_buf[:M], z_buf[:M], phi_buf[:M]

    def test_returns_positive_count_equator(self):
        """Stencil returns at least one pixel for an equatorial query."""
        nside = 64
        M, *_ = self._call(nside, 0.0, 1.0)
        assert M > 0

    def test_returns_positive_count_polar_cap(self):
        """Stencil returns at least one pixel near the north pole."""
        nside = 64
        vz = math.cos(0.05)  # ~3 deg from north pole
        M, *_ = self._call(nside, vz, 0.5)
        assert M > 0

    def test_pixel_indices_in_valid_range(self):
        """All gathered pixel indices must be in [0, 12*nside^2)."""
        nside = 128
        npix = 12 * nside * nside
        rng = np.random.default_rng(7)
        for _ in range(20):
            vz = rng.uniform(-0.9, 0.9)
            ph = rng.uniform(0, 2 * math.pi)
            M, pix, _, _ = self._call(nside, vz, ph)
            assert M > 0
            assert np.all(pix >= 0), "Negative pixel index found"
            assert np.all(pix < npix), "Pixel index out of range"

    def test_no_duplicate_pixels(self):
        """Stencil should not return the same pixel twice."""
        nside = 64
        rng = np.random.default_rng(8)
        for _ in range(20):
            vz = rng.uniform(-0.8, 0.8)
            ph = rng.uniform(0, 2 * math.pi)
            M, pix, _, _ = self._call(nside, vz, ph)
            assert len(np.unique(pix)) == M, "Duplicate pixel indices in stencil"

    def test_z_buf_consistent_with_pixels(self):
        """z_buf[k] should equal cos(theta) for the ring containing pix[k]."""
        nside = 64
        M, pix, z_vals, _ = self._call(nside, 0.0, 0.5)
        for k in range(M):
            th_hp, _ = hp.pix2ang(nside, int(pix[k]))
            z_hp = math.cos(th_hp)
            assert abs(z_vals[k] - z_hp) < 1e-12, (
                f"z_buf[{k}]={z_vals[k]:.6f} != cos(theta_pix)={z_hp:.6f}"
            )

    def test_count_does_not_exceed_buffer(self):
        """M must be ≤ 64 (buffer size used throughout the pipeline)."""
        nside = 2048
        rng = np.random.default_rng(9)
        for _ in range(30):
            vz = rng.uniform(-0.99, 0.99)
            ph = rng.uniform(0, 2 * math.pi)
            M, *_ = self._call(nside, vz, ph)
            assert M <= 64, f"Stencil count {M} exceeds buffer size 64"

    def test_stencil_pixels_near_query(self):
        """All gathered pixels should be within the stencil footprint of the query.

        The stencil covers ±3 rings (≈ ±1.97 h_pix north/south) and ±2 phi pixels
        (≈ ±2.3–3.5 h_pix east/west depending on latitude).  The farthest stencil
        pixel sits at the corner of this box, giving a maximum angular distance of
        ≈ 4.5 h_pix.  Candidates outside the 2 h_pix Keys support get zero weight
        and are skipped by the kernel, so gathering them is harmless.
        """
        nside = 128
        vz = 0.3
        ph = 1.0
        h_pix = hp.nside2resol(nside)
        M, pix, _, _ = self._call(nside, vz, ph)
        # Conservative bound: ±3 rings × ±2 phi → diagonal ≤ ~4.5 h_pix
        max_dist = 5.0 * h_pix
        sin_q = math.sqrt(max(0.0, 1.0 - vz**2))
        for k in range(M):
            th_k, ph_k = hp.pix2ang(nside, int(pix[k]))
            cos_c = math.cos(th_k) * vz + math.sin(th_k) * sin_q * math.cos(ph_k - ph)
            cos_c = min(1.0, max(-1.0, cos_c))
            dist = math.acos(cos_c)
            assert dist <= max_dist + 1e-10, (
                f"Stencil pixel {k} is {dist:.4f} rad from query "
                f"({dist / h_pix:.1f} h_pix), expected ≤ {max_dist:.4f} rad"
            )


# ===========================================================================
# TestBicubicInterpAccum
# ===========================================================================


class TestBicubicInterpAccum:
    """Tests for tod_core._bicubic_interp_accum."""

    @staticmethod
    def _build_inputs(B, Sc, nside, mp_val=1.0, rng=None):
        """
        Build (vec_rot, mp_stacked, beam_vals, tod_arr) with vec_rot pointing
        near the equator.
        """
        if rng is None:
            rng = np.random.default_rng(0)
        npix = hp.nside2npix(nside)
        # Random unit vectors near the equator
        theta = rng.uniform(np.pi / 2 - 0.1, np.pi / 2 + 0.1, (B, Sc))
        phi = rng.uniform(0, 2 * np.pi, (B, Sc))
        vec_rot = np.stack(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ],
            axis=-1,
        ).astype(np.float32)
        mp_stacked = np.full((1, npix), mp_val, dtype=np.float32)
        beam_vals = np.ones(Sc, dtype=np.float32) / Sc
        tod_arr = np.zeros((1, B), dtype=np.float64)
        return vec_rot, mp_stacked, beam_vals, tod_arr

    def test_constant_map_gives_constant(self):
        """Bicubic interp on a constant map returns that constant for every (c, b)."""
        nside = 32
        B, Sc = 4, 8
        val = 3.7
        vec_rot, mp_stacked, beam_vals, tod_arr = self._build_inputs(
            B, Sc, nside, mp_val=val
        )
        _bicubic_interp_accum(vec_rot, B, Sc, nside, mp_stacked, beam_vals, tod_arr)
        npt.assert_allclose(
            tod_arr[0],
            np.full(B, val),
            atol=1e-3,
            err_msg="Constant map not reproduced by bicubic interpolation",
        )

    def test_accumulates_inplace(self):
        """Calling twice with the same inputs doubles the result."""
        nside = 32
        B, Sc = 3, 5
        vec_rot, mp_stacked, beam_vals, tod_arr = self._build_inputs(
            B, Sc, nside, mp_val=1.0
        )
        _bicubic_interp_accum(vec_rot, B, Sc, nside, mp_stacked, beam_vals, tod_arr)
        first = tod_arr.copy()
        _bicubic_interp_accum(vec_rot, B, Sc, nside, mp_stacked, beam_vals, tod_arr)
        npt.assert_allclose(tod_arr, 2 * first, atol=1e-10)

    def test_zero_beam_vals_gives_zero(self):
        """Zero beam weights produce no contribution."""
        nside = 32
        B, Sc = 3, 5
        vec_rot, mp_stacked, beam_vals, tod_arr = self._build_inputs(
            B, Sc, nside, mp_val=5.0
        )
        beam_vals[:] = 0.0
        _bicubic_interp_accum(vec_rot, B, Sc, nside, mp_stacked, beam_vals, tod_arr)
        npt.assert_allclose(tod_arr, np.zeros((1, B)), atol=1e-10)

    def test_output_shape_unchanged(self):
        """tod_arr shape (C, B) is not modified by the call."""
        nside = 32
        C, B, Sc = 3, 5, 7
        rng = np.random.default_rng(4)
        npix = hp.nside2npix(nside)
        theta = rng.uniform(0.1, np.pi - 0.1, (B, Sc))
        phi = rng.uniform(0, 2 * np.pi, (B, Sc))
        vec_rot = np.stack(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ],
            axis=-1,
        ).astype(np.float32)
        mp_stacked = np.ones((C, npix), dtype=np.float32)
        beam_vals = np.ones(Sc, dtype=np.float32) / Sc
        tod_arr = np.zeros((C, B), dtype=np.float64)
        _bicubic_interp_accum(vec_rot, B, Sc, nside, mp_stacked, beam_vals, tod_arr)
        assert tod_arr.shape == (C, B)

    def test_multi_component_constant_maps(self):
        """Each component is interpolated independently with correct values."""
        nside = 32
        C, B, Sc = 3, 4, 6
        rng = np.random.default_rng(5)
        npix = hp.nside2npix(nside)
        theta = rng.uniform(np.pi / 2 - 0.1, np.pi / 2 + 0.1, (B, Sc))
        phi = rng.uniform(0, 2 * np.pi, (B, Sc))
        vec_rot = np.stack(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ],
            axis=-1,
        ).astype(np.float32)
        values = np.array([1.0, 2.0, 3.0])
        mp_stacked = np.stack([np.full(npix, v, dtype=np.float32) for v in values])
        beam_vals = np.ones(Sc, dtype=np.float32) / Sc
        tod_arr = np.zeros((C, B), dtype=np.float64)
        _bicubic_interp_accum(vec_rot, B, Sc, nside, mp_stacked, beam_vals, tod_arr)
        for c, v in enumerate(values):
            npt.assert_allclose(
                tod_arr[c],
                np.full(B, v),
                atol=1e-3,
                err_msg=f"Component {c} (value={v}) failed",
            )

    def test_polar_query_does_not_crash(self):
        """Queries near the north pole fall back gracefully (no exception)."""
        nside = 32
        B, Sc = 2, 4
        npix = hp.nside2npix(nside)
        # Place all vec_rot near the north pole
        eps = 0.001
        theta = np.full((B, Sc), eps)
        phi = np.zeros((B, Sc))
        vec_rot = np.stack(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ],
            axis=-1,
        ).astype(np.float32)
        mp_stacked = np.ones((1, npix), dtype=np.float32)
        beam_vals = np.ones(Sc, dtype=np.float32) / Sc
        tod_arr = np.zeros((1, B), dtype=np.float64)
        _bicubic_interp_accum(vec_rot, B, Sc, nside, mp_stacked, beam_vals, tod_arr)
        # Should produce finite values, not NaN/inf
        assert np.all(np.isfinite(tod_arr))


# ===========================================================================
# TestBeamTodBatchBicubicOriginalPath
# ===========================================================================


class TestBeamTodBatchBicubicOriginalPath:
    """
    Tests for beam_tod_batch with interp_mode='bicubic' on the original
    (double Rodrigues, no cache) path.
    """

    def test_output_keys_shape_dtype(self):
        """Bicubic mode returns the same keys, shape (B,), dtype float32 as bilinear."""
        nside = 32
        B = 5
        data = _build_data(S=30, nside=nside)
        data["mp_stacked"] = _mp_stacked(_constant_maps(nside), data["comp_indices"])
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)

        tod = beam_tod_batch(
            nside,
            _constant_maps(nside),
            data,
            rot_vecs,
            phi_b,
            theta_b,
            psis_b,
            interp_mode="bicubic",
        )
        assert set(tod.keys()) == set(data["comp_indices"])
        for comp in data["comp_indices"]:
            assert tod[comp].shape == (B,)
            assert tod[comp].dtype == np.float32

    def test_constant_map_gives_ones(self):
        """Constant (all-ones) sky map with normalised beam gives tod ≈ 1.0."""
        nside = 32
        B = 8
        data = _build_data(S=30, nside=nside)
        data["mp_stacked"] = _mp_stacked(_constant_maps(nside), data["comp_indices"])
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)

        tod = beam_tod_batch(
            nside,
            _constant_maps(nside),
            data,
            rot_vecs,
            phi_b,
            theta_b,
            psis_b,
            interp_mode="bicubic",
        )
        for comp in data["comp_indices"]:
            npt.assert_allclose(
                tod[comp],
                np.ones(B),
                atol=1e-3,
                err_msg=f"Constant-map TOD not ≈ 1 for comp={comp}",
            )

    def test_bicubic_vs_bilinear_constant_map(self):
        """On a constant map bicubic and bilinear must agree exactly (both give 1.0)."""
        nside = 32
        B = 6
        data = _build_data(S=30, nside=nside)
        mp = _constant_maps(nside)
        data["mp_stacked"] = _mp_stacked(mp, data["comp_indices"])
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)

        tod_bi = beam_tod_batch(
            nside, mp, data, rot_vecs, phi_b, theta_b, psis_b, interp_mode="bilinear"
        )
        tod_bc = beam_tod_batch(
            nside, mp, data, rot_vecs, phi_b, theta_b, psis_b, interp_mode="bicubic"
        )
        for comp in data["comp_indices"]:
            npt.assert_allclose(tod_bc[comp], tod_bi[comp], atol=1e-3)

    def test_bicubic_vs_bilinear_smooth_map(self):
        """
        On a spatially smooth map bicubic and bilinear agree to within 1e-2.
        Both are valid interpolators; on smooth data the difference is sub-percent.
        """
        nside = 64
        B = 8
        npix = hp.nside2npix(nside)
        pix_theta, pix_phi = hp.pix2ang(nside, np.arange(npix))
        smooth_map = (1.0 + 0.05 * np.sin(pix_phi)).astype(np.float32)
        mp = {c: smooth_map.copy() for c in ["I", "Q", "U"]}

        data = _build_data(S=20, nside=nside)
        data["mp_stacked"] = _mp_stacked(mp, data["comp_indices"])
        phi_b, theta_b, psis_b, rot_vecs = _build_scan(B)

        tod_bi = beam_tod_batch(
            nside, mp, data, rot_vecs, phi_b, theta_b, psis_b, interp_mode="bilinear"
        )
        tod_bc = beam_tod_batch(
            nside, mp, data, rot_vecs, phi_b, theta_b, psis_b, interp_mode="bicubic"
        )
        for comp in data["comp_indices"]:
            npt.assert_allclose(
                tod_bc[comp],
                tod_bi[comp],
                atol=1e-2,
                err_msg=f"Bicubic/bilinear disagree on smooth map, comp={comp}",
            )

    def test_rotational_stability_symmetric_beam(self):
        """
        Primary science validation: a single central beam pixel (perfectly symmetric)
        pointing at the same sky location under different psi rotations must produce
        identical TOD values.

        A single pixel at (theta=0, phi=0) is invariant under any psi roll: the
        rotation R(theta, phi, psi) maps the beam centre to the boresight regardless
        of psi, so the sky sample is always the same HEALPix pixel.  Any variation
        is purely a pipeline artifact — the canonical correctness criterion from
        CLAUDE.md.

        This is distinct from a multi-pixel beam test where interpolation noise
        mixes in.  Here the expectation is machine-precision equality (atol ~ 1e-6
        relative, limited only by float32 sky map storage).
        """
        nside = 64
        npix = hp.nside2npix(nside)

        # Single central beam pixel: perfectly rotationally symmetric.
        # With ra=dec=0 at center_idx, the beam center maps to phi=0, theta=pi/2
        # → unit vector (1, 0, 0).  precompute_rotation_vector_batch computes the
        # rotation that brings (1,0,0) → boresight.  Rotating psi around the
        # boresight leaves the central pixel exactly at the boresight, so the
        # sampled sky direction is identical for all psi values.
        vec_orig = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        bv = np.array([1.0], dtype=np.float32)

        # Smooth sky map
        pix_th, pix_ph = hp.pix2ang(nside, np.arange(npix))
        sky = (
            1.0 + 0.3 * np.cos(pix_th) + 0.2 * np.sin(pix_th) * np.cos(pix_ph)
        ).astype(np.float32)
        mp = {
            "I": sky,
            "Q": np.zeros(npix, dtype=np.float32),
            "U": np.zeros(npix, dtype=np.float32),
        }
        mp_stack = np.stack([mp[c] for c in ["I", "Q", "U"]]).astype(np.float32)

        data = {
            "vec_orig": vec_orig,
            "beam_vals": bv,
            "comp_indices": ["I", "Q", "U"],
            "mp_stacked": mp_stack,
        }

        N = 201
        ra = np.zeros((N, N))
        dec = np.zeros((N, N))

        # Same boresight, 8 different psi angles
        B = 8
        theta_center = np.pi / 2 - 0.15
        phi_center = 0.8
        psi_vals = np.linspace(0, 2 * np.pi, B, endpoint=False)
        phi_b = np.full(B, phi_center)
        theta_b = np.full(B, theta_center)

        rot_vecs, betas = precompute_rotation_vector_batch(
            ra, dec, phi_b, theta_b, center_idx=(N // 2, N // 2)
        )
        psis_b = psi_vals - betas

        tod = beam_tod_batch(
            nside,
            mp,
            data,
            rot_vecs,
            phi_b,
            theta_b,
            psis_b,
            interp_mode="bicubic",
        )

        # Central pixel always maps to the same sky direction → identical values
        tod_I = tod["I"]
        rms = float(np.std(tod_I))
        mean = float(np.mean(tod_I))
        rel_rms = rms / (abs(mean) + 1e-30)
        assert rel_rms < 1e-5, (
            f"Bicubic rotational instability too large: relative RMS = {rel_rms:.2e} "
            f"(mean={mean:.6f}, std={rms:.2e})"
        )


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__, "-v"])
