"""
Tests for the tod_core module.

- tod_core    : _rotation_params, _rodrigues_jit,
                    _recenter_and_rotate, precompute_rotation_vector_batch, beam_tod_batch,

Can be run independently:
    pytest tests/test_tod_core.py -v
    python tests/test_tod_core.py
"""

import os
import sys
import importlib
import math
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

import numpy as np
import numpy.testing as npt
import healpy as hp
import pytest

from tod_core import (
    _rotation_params,
    _rodrigues_jit,
    _rodrigues1_from_rolled_jit,
    _recenter_and_rotate,
    precompute_rotation_vector_batch,
    beam_tod_batch,
)

# ---------------------------------------------------------------------------
# Shared RNG (deterministic across all tests)
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(0)


# ===========================================================================
# Helpers
# ===========================================================================


def _random_unit_vectors(n, rng=_RNG):
    """Return (n, 3) array of random unit vectors."""
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return v


def _numpy_ref_rotate(vec_orig, rot_vecs, phi_pix, theta_pix, psis):
    """
    Pure-numpy double-Rodrigues reference implementation used to validate
    the Numba kernel in _recenter_and_rotate.
    """
    angles = np.linalg.norm(rot_vecs, axis=-1, keepdims=True)
    safe = angles > 1e-10
    axes = np.where(safe, rot_vecs / np.where(safe, angles, 1.0), 0.0)
    cos_a = np.cos(angles)[:, :, np.newaxis]
    sin_a = np.sin(angles)[:, :, np.newaxis]
    v = vec_orig[np.newaxis].astype(np.float64)
    k = axes[:, np.newaxis]
    dot = np.sum(k * v, axis=-1, keepdims=True)
    out = v * cos_a + np.cross(k, v) * sin_a + k * dot * (1.0 - cos_a)
    ax = np.stack(
        [
            np.sin(theta_pix) * np.cos(phi_pix),
            np.sin(theta_pix) * np.sin(phi_pix),
            np.cos(theta_pix),
        ],
        axis=-1,
    )
    cos_p = np.cos(psis)[:, np.newaxis, np.newaxis]
    sin_p = np.sin(psis)[:, np.newaxis, np.newaxis]
    dot2 = np.sum(ax[:, np.newaxis] * out, axis=-1, keepdims=True)
    cross = np.cross(ax[:, np.newaxis], out)
    out *= cos_p
    out += cross * sin_p
    out += ax[:, np.newaxis] * (dot2 * (1.0 - cos_p))
    return out


# ===========================================================================
# TestRotationParams
# ===========================================================================


class TestRotationParams:
    """Tests for tod_core._rotation_params."""

    def _make_inputs(self, B, rng=_RNG):
        rot_vecs = rng.standard_normal((B, 3)).astype(np.float64) * 0.5
        phi_b = rng.uniform(0, 2 * np.pi, B)
        theta_b = rng.uniform(0, np.pi, B)
        psis_b = rng.uniform(0, 2 * np.pi, B)
        return rot_vecs, phi_b, theta_b, psis_b

    def test_output_shapes(self):
        """All six returned arrays have correct shapes for B pointing directions."""
        B = 7
        rot_vecs, phi_b, theta_b, psis_b = self._make_inputs(B)
        axes, cos_a, sin_a, ax_pts, cos_p, sin_p = _rotation_params(
            rot_vecs, phi_b, theta_b, psis_b
        )
        assert axes.shape == (B, 3)
        assert cos_a.shape == (B,)
        assert sin_a.shape == (B,)
        assert ax_pts.shape == (B, 3)
        assert cos_p.shape == (B,)
        assert sin_p.shape == (B,)

    def test_all_float32(self):
        """All returned arrays have dtype float32."""
        B = 5
        rot_vecs, phi_b, theta_b, psis_b = self._make_inputs(B)
        outputs = _rotation_params(rot_vecs, phi_b, theta_b, psis_b)
        for arr in outputs:
            assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"

    def test_zero_rot_vecs(self):
        """Zero rot_vecs produce axes=0, cos_a=1, sin_a=0."""
        B = 4
        rot_vecs = np.zeros((B, 3), dtype=np.float64)
        phi_b = np.zeros(B)
        theta_b = np.full(B, np.pi / 2)
        psis_b = np.zeros(B)
        axes, cos_a, sin_a, ax_pts, cos_p, sin_p = _rotation_params(
            rot_vecs, phi_b, theta_b, psis_b
        )
        npt.assert_allclose(axes, np.zeros((B, 3)), atol=1e-6)
        npt.assert_allclose(cos_a, np.ones(B), atol=1e-6)
        npt.assert_allclose(sin_a, np.zeros(B), atol=1e-6)

    def test_ax_pts_are_unit_vectors(self):
        """ax_pts rows are unit vectors for arbitrary phi_b, theta_b."""
        B = 10
        _, phi_b, theta_b, psis_b = self._make_inputs(B)
        rot_vecs = np.zeros((B, 3))
        _, _, _, ax_pts, _, _ = _rotation_params(rot_vecs, phi_b, theta_b, psis_b)
        norms = np.linalg.norm(ax_pts.astype(np.float64), axis=-1)
        npt.assert_allclose(norms, np.ones(B), atol=1e-6)


# ===========================================================================
# TestRodriguesJit
# ===========================================================================


class TestRodriguesJit:
    """Tests for tod_core._rodrigues_jit Numba kernel."""

    def _zero_pol_roll_params(self, B, phi, theta):
        """Build ax_pts, cos_p, sin_p for zero polarisation roll around given axis."""
        phi_f = np.full(B, phi, dtype=np.float32)
        theta_f = np.full(B, theta, dtype=np.float32)
        ax_pts = np.stack(
            [
                np.sin(theta_f) * np.cos(phi_f),
                np.sin(theta_f) * np.sin(phi_f),
                np.cos(theta_f),
            ],
            axis=-1,
        )
        cos_p = np.ones(B, dtype=np.float32)
        sin_p = np.zeros(B, dtype=np.float32)
        return ax_pts, cos_p, sin_p

    def test_identity_rotation(self):
        """Identity rotation (angle=0) with zero pol-roll leaves vectors unchanged."""
        rng = np.random.default_rng(42)
        B = 3
        S = 5
        vec_orig = _random_unit_vectors(S, rng).astype(np.float32)
        axes = np.zeros((B, 3), dtype=np.float32)
        cos_a = np.ones(B, dtype=np.float32)
        sin_a = np.zeros(B, dtype=np.float32)
        ax_pts, cos_p, sin_p = self._zero_pol_roll_params(B, 0.0, 0.0)
        out = np.empty((B, S, 3), dtype=np.float32)
        _rodrigues_jit(vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, out)
        for b in range(B):
            npt.assert_allclose(out[b], vec_orig, atol=1e-5)

    def test_90deg_rotation_around_z(self):
        """90 degree rotation around z maps [1,0,0] to [0,1,0]."""
        B = 1
        S = 1
        vec_orig = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        # Axis = z = [0, 0, 1], angle = pi/2
        axes = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        angle = np.float32(np.pi / 2)
        cos_a = np.array([np.cos(angle)], dtype=np.float32)
        sin_a = np.array([np.sin(angle)], dtype=np.float32)
        ax_pts, cos_p, sin_p = self._zero_pol_roll_params(B, 0.0, 0.0)
        out = np.empty((B, S, 3), dtype=np.float32)
        _rodrigues_jit(vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, out)
        npt.assert_allclose(out[0, 0], [0.0, 1.0, 0.0], atol=1e-5)

    def test_combined_rotation_and_pol_roll(self):
        """[1,0,0] rotated 90 deg around z then 90 deg pol-roll around z gives [-1,0,0]."""
        B = 1
        S = 1
        vec_orig = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        axes = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        angle = np.float32(np.pi / 2)
        cos_a = np.array([np.cos(angle)], dtype=np.float32)
        sin_a = np.array([np.sin(angle)], dtype=np.float32)
        # pol-roll axis = z, psi = pi/2
        psi = np.float32(np.pi / 2)
        # ax_pts for z axis: phi=0, theta=0 -> [0, 0, 1]
        ax_pts = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        cos_p = np.array([np.cos(psi)], dtype=np.float32)
        sin_p = np.array([np.sin(psi)], dtype=np.float32)
        out = np.empty((B, S, 3), dtype=np.float32)
        _rodrigues_jit(vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, out)
        # After first rotation: [1,0,0] -> [0,1,0]
        # After second rotation (90 deg around z): [0,1,0] -> [-1,0,0]
        npt.assert_allclose(out[0, 0], [-1.0, 0.0, 0.0], atol=1e-5)

    def test_output_dtype_float32(self):
        """Output array has dtype float32."""
        B = 2
        S = 3
        vec_orig = _random_unit_vectors(S).astype(np.float32)
        axes = np.zeros((B, 3), dtype=np.float32)
        cos_a = np.ones(B, dtype=np.float32)
        sin_a = np.zeros(B, dtype=np.float32)
        ax_pts, cos_p, sin_p = self._zero_pol_roll_params(B, 0.0, np.pi / 2)
        out = np.empty((B, S, 3), dtype=np.float32)
        _rodrigues_jit(vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, out)
        assert out.dtype == np.float32

    def test_unit_vectors_preserved(self):
        """Unit vector inputs produce unit vector outputs (norm preserved to float32 tolerance)."""
        B = 4
        S = 6
        rng = np.random.default_rng(7)
        vec_orig = _random_unit_vectors(S, rng).astype(np.float32)
        rot_vecs = rng.standard_normal((B, 3)).astype(np.float32) * 0.3
        phi_b = rng.uniform(0, 2 * np.pi, B)
        theta_b = rng.uniform(0.1, np.pi - 0.1, B)
        psis_b = rng.uniform(0, 2 * np.pi, B)
        axes, cos_a, sin_a, ax_pts, cos_p, sin_p = _rotation_params(
            rot_vecs, phi_b, theta_b, psis_b
        )
        out = np.empty((B, S, 3), dtype=np.float32)
        _rodrigues_jit(vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, out)
        norms = np.linalg.norm(out.astype(np.float64), axis=-1)
        npt.assert_allclose(norms, np.ones((B, S)), atol=1e-4)


# ===========================================================================
# TestRecenterAndRotate
# ===========================================================================


class TestRecenterAndRotate:
    """Tests for tod_core._recenter_and_rotate."""

    def _make_inputs(self, B, S, rng=_RNG):
        vec_orig = _random_unit_vectors(S, rng)
        rot_vecs = rng.standard_normal((B, 3)) * 0.3
        phi_b = rng.uniform(0, 2 * np.pi, B)
        theta_b = rng.uniform(0.1, np.pi - 0.1, B)
        psis = rng.uniform(0, 2 * np.pi, B)
        return vec_orig, rot_vecs, phi_b, theta_b, psis

    def test_output_shape(self):
        """Output shape is (B, S, 3) for B pointing directions and S beam pixels."""
        B, S = 5, 12
        vec_orig, rot_vecs, phi_b, theta_b, psis = self._make_inputs(B, S)
        out = _recenter_and_rotate(vec_orig, rot_vecs, phi_b, theta_b, psis)
        assert out.shape == (B, S, 3)

    def test_output_dtype_float32(self):
        """Output dtype is float32."""
        B, S = 3, 8
        vec_orig, rot_vecs, phi_b, theta_b, psis = self._make_inputs(B, S)
        out = _recenter_and_rotate(vec_orig, rot_vecs, phi_b, theta_b, psis)
        assert out.dtype == np.float32

    def test_identity_zero_rot_vecs_zero_psis(self):
        """Zero rot_vecs and zero psis leave vec_orig unchanged (broadcast over B)."""
        B, S = 4, 6
        rng = np.random.default_rng(1)
        vec_orig = _random_unit_vectors(S, rng)
        rot_vecs = np.zeros((B, 3))
        phi_b = rng.uniform(0, 2 * np.pi, B)
        theta_b = rng.uniform(0.1, np.pi - 0.1, B)
        psis = np.zeros(B)
        out = _recenter_and_rotate(vec_orig, rot_vecs, phi_b, theta_b, psis)
        for b in range(B):
            npt.assert_allclose(out[b], vec_orig.astype(np.float32), atol=1e-5)

    def test_norm_preservation(self):
        """All output vectors have L2 norm ≈ 1 when inputs are unit vectors."""
        B, S = 6, 10
        vec_orig, rot_vecs, phi_b, theta_b, psis = self._make_inputs(B, S)
        out = _recenter_and_rotate(vec_orig, rot_vecs, phi_b, theta_b, psis)
        norms = np.linalg.norm(out.astype(np.float64), axis=-1)
        npt.assert_allclose(norms, np.ones((B, S)), atol=1e-4)

    def test_matches_numpy_reference(self):
        """Matches the pure-numpy double-Rodrigues reference to tolerance 1e-4."""
        rng = np.random.default_rng(2)
        B, S = 5, 10
        vec_orig = _random_unit_vectors(S, rng)
        rot_vecs = rng.standard_normal((B, 3)) * 0.4
        phi_b = rng.uniform(0, 2 * np.pi, B)
        theta_b = rng.uniform(0.1, np.pi - 0.1, B)
        psis = rng.uniform(0, 2 * np.pi, B)

        out_numba = _recenter_and_rotate(vec_orig, rot_vecs, phi_b, theta_b, psis)
        out_numpy = _numpy_ref_rotate(vec_orig, rot_vecs, phi_b, theta_b, psis)
        npt.assert_allclose(out_numba.astype(np.float64), out_numpy, atol=1e-4)


# ===========================================================================
# TestPrecomputeRotationVectorBatch
# ===========================================================================


class TestPrecomputeRotationVectorBatch:
    """Tests for tod_core.precompute_rotation_vector_batch."""

    def _make_grid(self, N=201):
        """Return (ra, dec) zero-grids of shape (N, N) suitable for centre_idx=(N//2, N//2)."""
        ra = np.zeros((N, N))
        dec = np.zeros((N, N))
        return ra, dec

    def test_output_shapes(self):
        """rot_vector is (B, 3) and beta is (B,) for B batch pointing directions."""
        N = 201
        ra, dec = self._make_grid(N)
        B = 5
        phi_batch = np.linspace(0, np.pi / 4, B)
        theta_batch = np.linspace(np.pi / 4, np.pi / 2, B)
        rot_vector, beta = precompute_rotation_vector_batch(
            ra, dec, phi_batch, theta_batch, center_idx=(N // 2, N // 2)
        )
        assert rot_vector.shape == (B, 3)
        assert beta.shape == (B,)

    def test_beta_in_zero_to_2pi(self):
        """All beta values are in [0, 2pi)."""
        N = 201
        ra, dec = self._make_grid(N)
        rng = np.random.default_rng(3)
        B = 20
        phi_batch = rng.uniform(0, np.pi / 6, B)
        theta_batch = rng.uniform(np.pi / 3, 2 * np.pi / 3, B)
        _, beta = precompute_rotation_vector_batch(
            ra, dec, phi_batch, theta_batch, center_idx=(N // 2, N // 2)
        )
        assert np.all(beta >= 0.0), "Some beta values are negative"
        assert np.all(beta < 2 * np.pi), "Some beta values exceed 2pi"

    def test_zero_rotation_at_beam_centre(self):
        """Pointing at the beam centre produces |rot_vector| < 1e-10."""
        N = 201
        ra, dec = self._make_grid(N)
        # centre_idx=(N//2, N//2)=(100,100), ra=0, dec=0 -> phi=0, theta=pi/2
        phi_batch = np.array([0.0])
        theta_batch = np.array([np.pi / 2])
        rot_vector, _ = precompute_rotation_vector_batch(
            ra, dec, phi_batch, theta_batch, center_idx=(N // 2, N // 2)
        )
        npt.assert_array_less(np.linalg.norm(rot_vector, axis=-1), 1e-10)

    def test_90deg_separation_gives_pi_over_2(self):
        """Pointing 90 degrees away from the centre gives |rot_vector| ≈ pi/2."""
        N = 201
        ra, dec = self._make_grid(N)
        # Centre is at phi=0, theta=pi/2 (equator)
        # Point 90 degrees away in phi: phi=pi/2, theta=pi/2
        phi_batch = np.array([np.pi / 2])
        theta_batch = np.array([np.pi / 2])
        rot_vector, _ = precompute_rotation_vector_batch(
            ra, dec, phi_batch, theta_batch, center_idx=(N // 2, N // 2)
        )
        npt.assert_allclose(np.linalg.norm(rot_vector, axis=-1), np.pi / 2, atol=1e-6)

    def test_rot_vector_dtype_float64(self):
        """rot_vector dtype is float64."""
        N = 201
        ra, dec = self._make_grid(N)
        phi_batch = np.array([0.1])
        theta_batch = np.array([np.pi / 2])
        rot_vector, _ = precompute_rotation_vector_batch(
            ra, dec, phi_batch, theta_batch, center_idx=(N // 2, N // 2)
        )
        assert rot_vector.dtype == np.float64


# ===========================================================================
# TestBeamTodBatch
# ===========================================================================


class TestBeamTodBatch:
    """Tests for tod_core.beam_tod_batch."""

    @staticmethod
    def _build_data(S=30, nside=32, use_stacked=True):
        """
        Build a synthetic data dict with S beam pixels near the north pole.
        beam_vals are normalised to sum to 1.
        """
        rng = np.random.default_rng(99)

        # Beam pixels near north pole: theta small
        theta_beam = rng.uniform(0.0, 0.05, S)
        phi_beam = rng.uniform(0, 2 * np.pi, S)
        vec_orig = np.stack(
            [
                np.sin(theta_beam) * np.cos(phi_beam),
                np.sin(theta_beam) * np.sin(phi_beam),
                np.cos(theta_beam),
            ],
            axis=-1,
        )

        beam_vals = rng.uniform(0.5, 1.5, S)
        beam_vals /= beam_vals.sum()

        comp_indices = ["I", "Q", "U"]

        data = {
            "vec_orig": vec_orig,
            "beam_vals": beam_vals.astype(np.float32),
            "comp_indices": comp_indices,
            "mp_stacked": None,
        }
        return data

    @staticmethod
    def _build_scan(B=10, N=201):
        """Build B scan-pointing directions using a zero ra/dec grid."""
        rng = np.random.default_rng(77)
        ra = np.zeros((N, N))
        dec = np.zeros((N, N))
        phi_batch = rng.uniform(0, 0.04, B)
        theta_batch = rng.uniform(np.pi / 2 - 0.04, np.pi / 2, B)
        rot_vecs, betas = precompute_rotation_vector_batch(
            ra, dec, phi_batch, theta_batch, center_idx=(N // 2, N // 2)
        )
        psis_b = -betas
        return phi_batch, theta_batch, psis_b, rot_vecs

    @staticmethod
    def _constant_maps(nside=32):
        """Return IQU maps (dict keyed by component name) that are all-ones."""
        npix = hp.nside2npix(nside)
        return {
            "I": np.ones(npix, dtype=np.float32),
            "Q": np.ones(npix, dtype=np.float32),
            "U": np.ones(npix, dtype=np.float32),
        }

    def test_output_keys_shape_dtype(self):
        """Output is a dict with exactly comp_indices keys; each value is shape (B,) float32."""
        nside = 32
        B = 5
        data = self._build_data(S=30)
        phi_b, theta_b, psis_b, rot_vecs = self._build_scan(B)
        mp = self._constant_maps(nside)

        tod = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b)

        assert set(tod.keys()) == set(data["comp_indices"])
        for comp in data["comp_indices"]:
            assert tod[comp].shape == (B,), f"Wrong shape for comp={comp}"
            assert tod[comp].dtype == np.float32, f"Wrong dtype for comp={comp}"

    def test_constant_sky_map_gives_ones(self):
        """Constant (all-ones) sky map with normalised beam gives tod ≈ 1.0 for all comps."""
        nside = 32
        B = 8
        data = self._build_data(S=30)
        phi_b, theta_b, psis_b, rot_vecs = self._build_scan(B)
        mp = self._constant_maps(nside)

        tod = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b)

        for comp in data["comp_indices"]:
            npt.assert_allclose(
                tod[comp],
                np.ones(B),
                atol=1e-3,
                err_msg=f"TOD values not ≈ 1 for comp={comp}",
            )

    def test_mp_stacked_matches_numpy_fallback(self):
        """Numba (mp_stacked) path and numpy fallback path agree to within 1e-4."""
        nside = 32
        B = 6
        mp = self._constant_maps(nside)
        data_base = self._build_data(S=30)
        phi_b, theta_b, psis_b, rot_vecs = self._build_scan(B)
        comp_indices = data_base["comp_indices"]

        # Numpy fallback path: mp_stacked = None
        data_numpy = dict(data_base)
        data_numpy["mp_stacked"] = None
        tod_numpy = beam_tod_batch(
            nside, mp, data_numpy, rot_vecs, phi_b, theta_b, psis_b
        )

        # Numba path: build mp_stacked (stacked in comp_indices order)
        mp_stacked = np.stack([mp[c] for c in comp_indices]).astype(np.float64)
        data_numba = dict(data_base)
        data_numba["mp_stacked"] = mp_stacked
        tod_numba = beam_tod_batch(
            nside, mp, data_numba, rot_vecs, phi_b, theta_b, psis_b
        )

        for comp in comp_indices:
            npt.assert_allclose(
                tod_numba[comp],
                tod_numpy[comp],
                atol=1e-4,
                err_msg=f"Numba and numpy paths disagree for comp={comp}",
            )

    def test_single_sample_batch(self):
        """Single-sample batch (B=1) runs without error and produces shape (1,) per comp."""
        nside = 32
        B = 1
        data = self._build_data(S=30)
        phi_b, theta_b, psis_b, rot_vecs = self._build_scan(B)
        mp = self._constant_maps(nside)

        tod = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b)

        for comp in data["comp_indices"]:
            assert tod[comp].shape == (1,)

    def test_additivity(self):
        """Two calls with half-weight beams summed equals one call with full-weight beam."""
        nside = 32
        B = 5
        mp = self._constant_maps(nside)
        data_base = self._build_data(S=30)
        phi_b, theta_b, psis_b, rot_vecs = self._build_scan(B)

        # Half-weight copies
        data_half1 = dict(data_base)
        data_half1["beam_vals"] = data_base["beam_vals"].copy() * 0.5
        data_half1["mp_stacked"] = None

        data_half2 = dict(data_base)
        data_half2["beam_vals"] = data_base["beam_vals"].copy() * 0.5
        data_half2["mp_stacked"] = None

        # Full-weight (original normalised)
        data_full = dict(data_base)
        data_full["mp_stacked"] = None

        tod1 = beam_tod_batch(nside, mp, data_half1, rot_vecs, phi_b, theta_b, psis_b)
        tod2 = beam_tod_batch(nside, mp, data_half2, rot_vecs, phi_b, theta_b, psis_b)
        tod_ref = beam_tod_batch(nside, mp, data_full, rot_vecs, phi_b, theta_b, psis_b)

        for comp in data_base["comp_indices"]:
            npt.assert_allclose(
                tod1[comp] + tod2[comp],
                tod_ref[comp],
                atol=1e-5,
                err_msg=f"Additivity violated for comp={comp}",
            )

    def test_fused_path_float32_mp_stacked(self):
        """Fused path with float32 mp_stacked agrees with the numpy fallback to 1e-4."""
        nside = 32
        B = 8
        mp = self._constant_maps(nside)
        data_base = self._build_data(S=30)
        phi_b, theta_b, psis_b, rot_vecs = self._build_scan(B)
        comp_indices = data_base["comp_indices"]

        # numpy fallback (mp_stacked = None)
        data_np = dict(data_base)
        data_np["mp_stacked"] = None
        tod_np = beam_tod_batch(nside, mp, data_np, rot_vecs, phi_b, theta_b, psis_b)

        # Fused path with float32 mp_stacked (the intended production dtype)
        mp_stacked_f32 = np.stack([mp[c] for c in comp_indices]).astype(np.float32)
        data_f32 = dict(data_base)
        data_f32["mp_stacked"] = mp_stacked_f32
        tod_f32 = beam_tod_batch(nside, mp, data_f32, rot_vecs, phi_b, theta_b, psis_b)

        for comp in comp_indices:
            npt.assert_allclose(
                tod_f32[comp],
                tod_np[comp],
                atol=1e-4,
                err_msg=f"float32-path disagrees with numpy fallback for {comp}",
            )

    def test_fused_path_nontrivial_map(self):
        """Fused path with a random (non-constant) sky map agrees with the numpy fallback."""
        nside = 32
        B = 8
        rng = np.random.default_rng(55)
        npix = hp.nside2npix(nside)

        # Random IQU maps (non-constant so any systematic bias is detectable)
        mp = {
            c: rng.uniform(0.5, 1.5, npix).astype(np.float32) for c in ["I", "Q", "U"]
        }

        data_base = self._build_data(S=40)
        phi_b, theta_b, psis_b, rot_vecs = self._build_scan(B)
        comp_indices = data_base["comp_indices"]

        data_np = dict(data_base)
        data_np["mp_stacked"] = None
        tod_np = beam_tod_batch(nside, mp, data_np, rot_vecs, phi_b, theta_b, psis_b)

        mp_stacked = np.stack([mp[c] for c in comp_indices]).astype(np.float32)
        data_fused = dict(data_base)
        data_fused["mp_stacked"] = mp_stacked
        tod_fused = beam_tod_batch(
            nside, mp, data_fused, rot_vecs, phi_b, theta_b, psis_b
        )

        for comp in comp_indices:
            npt.assert_allclose(
                tod_fused[comp],
                tod_np[comp],
                atol=1e-4,
                err_msg=f"Fused path disagrees with numpy fallback on random map, comp={comp}",
            )


# ===========================================================================
# TestRodrigues1FromRolledJit
# ===========================================================================


class TestRodrigues1FromRolledJit:
    """Tests for tod_core._rodrigues1_from_rolled_jit Numba kernel."""

    def test_identity_rotation(self):
        """Identity rotation (cos_a=1, sin_a=0) leaves vectors unchanged."""
        rng = np.random.default_rng(60)
        B, Sc = 3, 5
        vec_rolled_b = rng.standard_normal((B, Sc, 3)).astype(np.float32)
        vec_rolled_b /= np.linalg.norm(vec_rolled_b, axis=-1, keepdims=True)
        axes = np.zeros((B, 3), dtype=np.float32)
        cos_a = np.ones(B, dtype=np.float32)
        sin_a = np.zeros(B, dtype=np.float32)
        out = np.empty((B, Sc, 3), dtype=np.float32)
        _rodrigues1_from_rolled_jit(vec_rolled_b, axes, cos_a, sin_a, out)
        npt.assert_allclose(out, vec_rolled_b, atol=1e-5)

    def test_output_shape(self):
        """Output buffer has shape (B, Sc, 3) after the kernel call."""
        B, Sc = 4, 7
        vec_rolled_b = np.zeros((B, Sc, 3), dtype=np.float32)
        axes = np.zeros((B, 3), dtype=np.float32)
        cos_a = np.ones(B, dtype=np.float32)
        sin_a = np.zeros(B, dtype=np.float32)
        out = np.empty((B, Sc, 3), dtype=np.float32)
        _rodrigues1_from_rolled_jit(vec_rolled_b, axes, cos_a, sin_a, out)
        assert out.shape == (B, Sc, 3)

    def test_unit_vector_norms_preserved(self):
        """Rodrigues rotation preserves unit vector norms."""
        rng = np.random.default_rng(61)
        B, Sc = 4, 6
        vec_rolled_b = rng.standard_normal((B, Sc, 3)).astype(np.float32)
        vec_rolled_b /= np.linalg.norm(vec_rolled_b, axis=-1, keepdims=True)
        rot_vecs = rng.standard_normal((B, 3)) * 0.5
        angles = np.linalg.norm(rot_vecs, axis=-1)
        axes = (rot_vecs / np.where(angles > 1e-10, angles, 1.0)[:, None]).astype(
            np.float32
        )
        cos_a = np.cos(angles).astype(np.float32)
        sin_a = np.sin(angles).astype(np.float32)
        out = np.empty((B, Sc, 3), dtype=np.float32)
        _rodrigues1_from_rolled_jit(vec_rolled_b, axes, cos_a, sin_a, out)
        norms = np.linalg.norm(out.astype(np.float64), axis=-1)
        npt.assert_allclose(norms, np.ones((B, Sc)), atol=1e-4)

    def test_matches_numpy_reference(self):
        """Matches a pure-numpy single Rodrigues rotation to tolerance 1e-5."""
        rng = np.random.default_rng(62)
        B, Sc = 5, 8
        vec_rolled_b = rng.standard_normal((B, Sc, 3)).astype(np.float32)
        vec_rolled_b /= np.linalg.norm(vec_rolled_b, axis=-1, keepdims=True)
        rot_vecs = rng.standard_normal((B, 3)) * 0.4
        angles = np.linalg.norm(rot_vecs, axis=-1)
        axes_f64 = rot_vecs / np.where(angles > 1e-10, angles, 1.0)[:, None]
        cos_a = np.cos(angles).astype(np.float32)
        sin_a = np.sin(angles).astype(np.float32)
        axes = axes_f64.astype(np.float32)

        out = np.empty((B, Sc, 3), dtype=np.float32)
        _rodrigues1_from_rolled_jit(vec_rolled_b, axes, cos_a, sin_a, out)

        # Pure-numpy reference: single Rodrigues rotation per (b, s)
        ref = np.empty_like(out, dtype=np.float64)
        for b in range(B):
            k = axes_f64[b]
            ca, sa, oma = float(cos_a[b]), float(sin_a[b]), 1.0 - float(cos_a[b])
            for s in range(Sc):
                v = vec_rolled_b[b, s].astype(np.float64)
                dkv = np.dot(k, v)
                ref[b, s] = v * ca + np.cross(k, v) * sa + k * dkv * oma

        npt.assert_allclose(out.astype(np.float64), ref, atol=1e-5)

    def test_90deg_rotation_around_z(self):
        """90-degree rotation around z maps [1,0,0] to [0,1,0] for all B samples."""
        B, Sc = 3, 1
        vec_rolled_b = np.tile([[1.0, 0.0, 0.0]], (B, Sc, 1)).astype(np.float32)
        axes = np.tile([0.0, 0.0, 1.0], (B, 1)).astype(np.float32)
        angle = np.float32(np.pi / 2)
        cos_a = np.full(B, np.cos(angle), dtype=np.float32)
        sin_a = np.full(B, np.sin(angle), dtype=np.float32)
        out = np.empty((B, Sc, 3), dtype=np.float32)
        _rodrigues1_from_rolled_jit(vec_rolled_b, axes, cos_a, sin_a, out)
        for b in range(B):
            npt.assert_allclose(out[b, 0], [0.0, 1.0, 0.0], atol=1e-5)

    def test_output_dtype_float32(self):
        """Output buffer dtype is float32 (written in-place by the kernel)."""
        B, Sc = 2, 4
        vec_rolled_b = _random_unit_vectors(B * Sc).reshape(B, Sc, 3).astype(np.float32)
        axes = np.zeros((B, 3), dtype=np.float32)
        cos_a = np.ones(B, dtype=np.float32)
        sin_a = np.zeros(B, dtype=np.float32)
        out = np.empty((B, Sc, 3), dtype=np.float32)
        _rodrigues1_from_rolled_jit(vec_rolled_b, axes, cos_a, sin_a, out)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
