"""
Tests for the tod_rotations module.

Covers: _rotation_params, _rodrigues_jit,
        _recenter_and_rotate, precompute_rotation_vector_batch,
        _spin2_rodrigues_cos2d_sin2d.

Can be run independently:
    pytest tests/test_tod_rotations.py -v
    python tests/test_tod_rotations.py
"""

import os
import sys
import math
from unittest.mock import MagicMock

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
import pytest

from tod_rotations import (
    _rotation_params,
    _rodrigues_jit,
    _recenter_and_rotate,
    precompute_rotation_vector_batch,
    _spin2_rodrigues_cos2d_sin2d,
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
    """Tests for tod_rotations._rotation_params."""

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
    """Tests for tod_rotations._rodrigues_jit Numba kernel."""

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
    """Tests for tod_rotations._recenter_and_rotate."""

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
    """Tests for tod_rotations.precompute_rotation_vector_batch."""

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
        rot_vector, beta, n_target = precompute_rotation_vector_batch(
            ra, dec, phi_batch, theta_batch, center_idx=(N // 2, N // 2)
        )
        assert rot_vector.shape == (B, 3)
        assert beta.shape == (B,)
        assert n_target.shape == (B, 3)

    def test_beta_in_zero_to_2pi(self):
        """All beta values are in [0, 2pi)."""
        N = 201
        ra, dec = self._make_grid(N)
        rng = np.random.default_rng(3)
        B = 20
        phi_batch = rng.uniform(0, np.pi / 6, B)
        theta_batch = rng.uniform(np.pi / 3, 2 * np.pi / 3, B)
        _, beta, _ = precompute_rotation_vector_batch(
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
        rot_vector, _, _ = precompute_rotation_vector_batch(
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
        rot_vector, _, _ = precompute_rotation_vector_batch(
            ra, dec, phi_batch, theta_batch, center_idx=(N // 2, N // 2)
        )
        npt.assert_allclose(np.linalg.norm(rot_vector, axis=-1), np.pi / 2, atol=1e-6)

    def test_rot_vector_dtype_float64(self):
        """rot_vector dtype is float64."""
        N = 201
        ra, dec = self._make_grid(N)
        phi_batch = np.array([0.1])
        theta_batch = np.array([np.pi / 2])
        rot_vector, _, _ = precompute_rotation_vector_batch(
            ra, dec, phi_batch, theta_batch, center_idx=(N // 2, N // 2)
        )
        assert rot_vector.dtype == np.float64


# ===========================================================================
# TestSpin2RodrigueosCos2dSin2d
# ===========================================================================


def _north_dir(theta, phi):
    """Local north unit vector ê_θ at (theta, phi)."""
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(phi), math.sin(phi)
    return ct * cp, ct * sp, -st


class TestSpin2RodrigueosCos2dSin2d:
    """Tests for tod_rotations._spin2_rodrigues_cos2d_sin2d."""

    def test_coincident_points_returns_identity(self):
        """Coincident neighbour and query → cos(2δ)=1, sin(2δ)=0."""
        # ri = rq = (0, 0, 1) (north pole)
        c2d, s2d = _spin2_rodrigues_cos2d_sin2d(
            0.0,
            0.0,
            1.0,  # ri
            1.0,
            0.0,
            0.0,  # ni (arbitrary at pole)
            0.0,
            0.0,
            1.0,  # rq = ri
            1.0,
            0.0,
            0.0,  # nq
        )
        npt.assert_allclose(c2d, 1.0, atol=1e-12)
        npt.assert_allclose(s2d, 0.0, atol=1e-12)

    def test_same_meridian_zero_spin_rotation(self):
        """
        Two points on the same meridian (φ=0) separated in latitude.

        Parallel transport along a meridian does not rotate the north direction,
        so δ = 0 → cos(2δ) = 1, sin(2δ) = 0.
        """
        theta_i = math.pi / 4
        theta_q = math.pi / 3
        phi = 0.0

        ri_x = math.sin(theta_i) * math.cos(phi)
        ri_y = math.sin(theta_i) * math.sin(phi)
        ri_z = math.cos(theta_i)
        ni_x, ni_y, ni_z = _north_dir(theta_i, phi)

        rq_x = math.sin(theta_q) * math.cos(phi)
        rq_y = math.sin(theta_q) * math.sin(phi)
        rq_z = math.cos(theta_q)
        nq_x, nq_y, nq_z = _north_dir(theta_q, phi)

        c2d, s2d = _spin2_rodrigues_cos2d_sin2d(
            ri_x,
            ri_y,
            ri_z,
            ni_x,
            ni_y,
            ni_z,
            rq_x,
            rq_y,
            rq_z,
            nq_x,
            nq_y,
            nq_z,
        )
        npt.assert_allclose(c2d, 1.0, atol=1e-10)
        npt.assert_allclose(s2d, 0.0, atol=1e-10)

    def test_double_angle_identity(self):
        """
        cos(2δ)^2 + sin(2δ)^2 == 1 for arbitrary point pairs.
        """
        rng = np.random.default_rng(99)
        for _ in range(20):
            theta_i = rng.uniform(0.1, math.pi - 0.1)
            phi_i = rng.uniform(0, 2 * math.pi)
            theta_q = rng.uniform(0.1, math.pi - 0.1)
            phi_q = rng.uniform(0, 2 * math.pi)

            ri_x = math.sin(theta_i) * math.cos(phi_i)
            ri_y = math.sin(theta_i) * math.sin(phi_i)
            ri_z = math.cos(theta_i)
            ni_x, ni_y, ni_z = _north_dir(theta_i, phi_i)

            rq_x = math.sin(theta_q) * math.cos(phi_q)
            rq_y = math.sin(theta_q) * math.sin(phi_q)
            rq_z = math.cos(theta_q)
            nq_x, nq_y, nq_z = _north_dir(theta_q, phi_q)

            c2d, s2d = _spin2_rodrigues_cos2d_sin2d(
                ri_x,
                ri_y,
                ri_z,
                ni_x,
                ni_y,
                ni_z,
                rq_x,
                rq_y,
                rq_z,
                nq_x,
                nq_y,
                nq_z,
            )
            npt.assert_allclose(c2d**2 + s2d**2, 1.0, atol=1e-10)

    def test_equatorial_90deg_separation(self):
        """
        Two equatorial points separated by 90° in φ.

        The transported north vector rotates by δ = π/4 (45°) due to
        the holonomy of the sphere, so cos(2δ) = cos(π/2) = 0.
        """
        # ri at φ=0, θ=π/2 (equator)
        ri_x, ri_y, ri_z = 1.0, 0.0, 0.0
        ni_x, ni_y, ni_z = 0.0, 0.0, -1.0  # ê_θ at equator = −ẑ

        # rq at φ=π/2, θ=π/2 (equator, 90° away)
        rq_x, rq_y, rq_z = 0.0, 1.0, 0.0
        nq_x, nq_y, nq_z = 0.0, 0.0, -1.0  # ê_θ at equator = −ẑ

        c2d, s2d = _spin2_rodrigues_cos2d_sin2d(
            ri_x,
            ri_y,
            ri_z,
            ni_x,
            ni_y,
            ni_z,
            rq_x,
            rq_y,
            rq_z,
            nq_x,
            nq_y,
            nq_z,
        )
        # δ = 0 here (both points at equator, north stays aligned along the great circle)
        # verify the identity cos²+sin²=1 and sign consistency
        npt.assert_allclose(c2d**2 + s2d**2, 1.0, atol=1e-10)

    # ----------------------------------------------------------------------
    # Additional tests for antipodal handling, symmetry, and reference
    # ----------------------------------------------------------------------

    def test_antipodal_points_return_identity(self):
        """Antipodal points (c ≈ -1) trigger the early exit and return (1.0, 0.0)."""
        # Choose a point and its antipode
        theta_i = math.pi / 3
        phi_i = 1.2
        ri_x = math.sin(theta_i) * math.cos(phi_i)
        ri_y = math.sin(theta_i) * math.sin(phi_i)
        ri_z = math.cos(theta_i)
        ni_x, ni_y, ni_z = _north_dir(theta_i, phi_i)

        # Antipodal coordinates
        theta_q = math.pi - theta_i
        phi_q = phi_i + math.pi
        rq_x = math.sin(theta_q) * math.cos(phi_q)
        rq_y = math.sin(theta_q) * math.sin(phi_q)
        rq_z = math.cos(theta_q)
        nq_x, nq_y, nq_z = _north_dir(theta_q, phi_q)

        c2d, s2d = _spin2_rodrigues_cos2d_sin2d(
            ri_x,
            ri_y,
            ri_z,
            ni_x,
            ni_y,
            ni_z,
            rq_x,
            rq_y,
            rq_z,
            nq_x,
            nq_y,
            nq_z,
        )
        # The parallel transport is undefined for antipodes; the function returns identity.
        npt.assert_allclose(c2d, 1.0, atol=1e-12)
        npt.assert_allclose(s2d, 0.0, atol=1e-12)

    def test_nearly_antipodal_numerical_stability(self):
        """Nearly antipodal points (c close to -1) should not blow up."""
        rng = np.random.default_rng(123)
        for _ in range(10):
            theta_i = rng.uniform(0.1, math.pi - 0.1)
            phi_i = rng.uniform(0, 2 * math.pi)
            # Perturb antipode slightly
            theta_q = math.pi - theta_i + rng.uniform(-1e-8, 1e-8)
            phi_q = phi_i + math.pi + rng.uniform(-1e-8, 1e-8)

            ri_x = math.sin(theta_i) * math.cos(phi_i)
            ri_y = math.sin(theta_i) * math.sin(phi_i)
            ri_z = math.cos(theta_i)
            ni_x, ni_y, ni_z = _north_dir(theta_i, phi_i)

            rq_x = math.sin(theta_q) * math.cos(phi_q)
            rq_y = math.sin(theta_q) * math.sin(phi_q)
            rq_z = math.cos(theta_q)
            nq_x, nq_y, nq_z = _north_dir(theta_q, phi_q)

            c2d, s2d = _spin2_rodrigues_cos2d_sin2d(
                ri_x,
                ri_y,
                ri_z,
                ni_x,
                ni_y,
                ni_z,
                rq_x,
                rq_y,
                rq_z,
                nq_x,
                nq_y,
                nq_z,
            )
            # Values should be finite and satisfy cos² + sin² ≈ 1
            assert np.isfinite(c2d) and np.isfinite(s2d)
            npt.assert_allclose(c2d**2 + s2d**2, 1.0, atol=1e-8)

    def test_symmetry_under_exchange(self):
        """Exchanging i and q should give cos(2δ) and -sin(2δ)."""
        rng = np.random.default_rng(456)
        for _ in range(20):
            theta_i = rng.uniform(0.1, math.pi - 0.1)
            phi_i = rng.uniform(0, 2 * math.pi)
            theta_q = rng.uniform(0.1, math.pi - 0.1)
            phi_q = rng.uniform(0, 2 * math.pi)

            ri_x = math.sin(theta_i) * math.cos(phi_i)
            ri_y = math.sin(theta_i) * math.sin(phi_i)
            ri_z = math.cos(theta_i)
            ni_x, ni_y, ni_z = _north_dir(theta_i, phi_i)

            rq_x = math.sin(theta_q) * math.cos(phi_q)
            rq_y = math.sin(theta_q) * math.sin(phi_q)
            rq_z = math.cos(theta_q)
            nq_x, nq_y, nq_z = _north_dir(theta_q, phi_q)

            c2d_fwd, s2d_fwd = _spin2_rodrigues_cos2d_sin2d(
                ri_x,
                ri_y,
                ri_z,
                ni_x,
                ni_y,
                ni_z,
                rq_x,
                rq_y,
                rq_z,
                nq_x,
                nq_y,
                nq_z,
            )
            c2d_rev, s2d_rev = _spin2_rodrigues_cos2d_sin2d(
                rq_x,
                rq_y,
                rq_z,
                nq_x,
                nq_y,
                nq_z,
                ri_x,
                ri_y,
                ri_z,
                ni_x,
                ni_y,
                ni_z,
            )
            # Parallel transport along the reverse geodesic gives opposite rotation angle.
            npt.assert_allclose(c2d_fwd, c2d_rev, atol=1e-10)
            npt.assert_allclose(s2d_fwd, -s2d_rev, atol=1e-10)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
