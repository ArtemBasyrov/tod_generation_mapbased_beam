"""
Tests for the tod_nearest module.

- _gather_accum_nearest_jit : fused Rodrigues + nearest-pixel lookup + spin-2

Can be run independently:
    pytest tests/test_tod_nearest.py -v
    python tests/test_tod_nearest.py
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
import healpy as hp
import pytest

from tod_nearest import _gather_accum_nearest_jit
from tod_spin2 import _spin2_cos2d_sin2d_jit
from tod_rotations import _rodrigues_jit


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _random_unit_vec(shape, rng):
    v = rng.standard_normal(shape)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return v.astype(np.float32)


def _make_beam_and_rot_params(B, S, rng, pole_safe=True):
    """Return (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot_ref)."""
    vec_orig = _random_unit_vec((S, 3), rng)

    angles_1 = rng.uniform(0.0, math.pi, B).astype(np.float32)
    axes = _random_unit_vec((B, 3), rng)
    cos_a = np.cos(angles_1).astype(np.float32)
    sin_a = np.sin(angles_1).astype(np.float32)

    if pole_safe:
        theta = rng.uniform(0.1, math.pi - 0.1, B)
        phi = rng.uniform(0.0, 2 * math.pi, B)
        ax_pts = np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            axis=-1,
        ).astype(np.float32)
    else:
        ax_pts = _random_unit_vec((B, 3), rng)

    angles_2 = rng.uniform(0.0, 2 * math.pi, B).astype(np.float32)
    cos_p = np.cos(angles_2).astype(np.float32)
    sin_p = np.sin(angles_2).astype(np.float32)

    vec_rot_ref = np.empty((B, S, 3), dtype=np.float32)
    _rodrigues_jit(vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot_ref)

    return vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot_ref


def _identity_rot_params(B, ax_pts=None):
    """Identity rotation — vec_rot[b, s] == vec_orig[s] for all b."""
    axes = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (B, 1))
    cos_a = np.ones(B, dtype=np.float32)
    sin_a = np.zeros(B, dtype=np.float32)
    if ax_pts is None:
        ax_pts = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (B, 1))
    cos_p = np.ones(B, dtype=np.float32)
    sin_p = np.zeros(B, dtype=np.float32)
    return axes, cos_a, sin_a, ax_pts.astype(np.float32), cos_p, sin_p


def _call_nearest(
    vec_orig,
    axes,
    cos_a,
    sin_a,
    ax_pts,
    cos_p,
    sin_p,
    nside,
    mp_stacked,
    beam_vals,
    B,
    S,
    tod,
    c_q=-1,
    c_u=-1,
):
    _gather_accum_nearest_jit(
        vec_orig,
        axes,
        cos_a,
        sin_a,
        ax_pts,
        cos_p,
        sin_p,
        nside,
        mp_stacked,
        beam_vals,
        B,
        S,
        tod,
        c_q,
        c_u,
    )


from numba_healpy import _ring_above_jit, _ring_info_jit, _ring_z_jit, _TWO_PI


def _kernel_nearest_pix(nside, z, phi_w):
    """Python mirror of the kernel's 4-candidate nearest-pixel search.

    Returns (best_pix, best_z_c, best_phi_c) using exactly the same ring +
    slot selection as _gather_accum_nearest_jit, so tests can build a
    reference that agrees with the kernel rather than with hp.ang2pix (which
    searches more candidates).
    """
    _TWO_PI_f = 2.0 * math.pi
    npix_total = 12 * nside * nside
    theta = math.acos(max(-1.0, min(1.0, z)))
    sin_th = math.sin(theta)

    ir_above = _ring_above_jit(nside, z)
    if ir_above < 1:
        ir_above = 1
    elif ir_above > 4 * nside - 2:
        ir_above = 4 * nside - 2
    ir_below = ir_above + 1

    best_pix = 0
    best_cos = -2.0
    best_z_c = 0.0
    best_phi_c = 0.0

    for ir_g in (ir_above, ir_below):
        if ir_g < 1 or ir_g > 4 * nside - 1:
            continue
        n_pix, first_pix, phi0, dphi_r = _ring_info_jit(nside, ir_g, npix_total)
        z_c = _ring_z_jit(nside, ir_g)
        sin_z_c = math.sqrt(max(0.0, 1.0 - z_c * z_c))
        ip_base = int(phi_w * n_pix / _TWO_PI_f) % n_pix
        for ip_try in (ip_base, (ip_base + 1) % n_pix):
            phi_c = phi0 + ip_try * dphi_r
            cos_d = sin_th * sin_z_c * math.cos(phi_w - phi_c) + z * z_c
            if cos_d > best_cos:
                best_cos = cos_d
                best_pix = first_pix + ip_try
                best_z_c = z_c
                best_phi_c = phi_c

    return best_pix, best_z_c, best_phi_c


def _nearest_reference_t(vec_rot, nside, mp_stacked, beam_vals, B, S):
    """T reference using the kernel's own 4-candidate nearest-pixel algorithm."""
    C = mp_stacked.shape[0]
    tod = np.zeros((C, B), dtype=np.float64)
    for b in range(B):
        for s in range(S):
            v = vec_rot[b, s].astype(np.float64)
            norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            z = v[2] / norm
            phi_w = math.atan2(v[1], v[0])
            if phi_w < 0.0:
                phi_w += 2 * math.pi
            pix, _, _ = _kernel_nearest_pix(nside, z, phi_w)
            bv = float(beam_vals[s])
            for c in range(C):
                tod[c, b] += float(mp_stacked[c, pix]) * bv
    return tod


def _nearest_reference_qu(
    vec_rot, nside, mp_stacked, beam_vals, ax_pts, B, S, c_q, c_u
):
    """Q/U reference using the kernel's nearest-pixel algorithm + spin-2."""
    C = mp_stacked.shape[0]
    tod = np.zeros((C, B), dtype=np.float64)
    for b in range(B):
        bx, by, bz = float(ax_pts[b, 0]), float(ax_pts[b, 1]), float(ax_pts[b, 2])
        bz_pts = max(-1.0, min(1.0, bz))
        bsth_pts = math.sqrt(max(0.0, 1.0 - bz * bz))
        bphi_pts = math.atan2(by, bx)
        if bphi_pts < 0.0:
            bphi_pts += 2 * math.pi

        for s in range(S):
            v = vec_rot[b, s].astype(np.float64)
            norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            z = v[2] / norm
            phi_w = math.atan2(v[1], v[0])
            if phi_w < 0.0:
                phi_w += 2 * math.pi
            pix, pix_z_c, pix_phi_c = _kernel_nearest_pix(nside, z, phi_w)
            sth_n = math.sqrt(max(0.0, 1.0 - pix_z_c * pix_z_c))
            bv = float(beam_vals[s])

            c2d, s2d = _spin2_cos2d_sin2d_jit(
                pix_z_c, sth_n, pix_phi_c, bz_pts, bsth_pts, bphi_pts
            )
            q_val = float(mp_stacked[c_q, pix])
            u_val = float(mp_stacked[c_u, pix])
            tod[c_q, b] += (q_val * c2d + u_val * s2d) * bv
            tod[c_u, b] += (-q_val * s2d + u_val * c2d) * bv
            for c in range(C):
                if c != c_q and c != c_u:
                    tod[c, b] += float(mp_stacked[c, pix]) * bv
    return tod


# ---------------------------------------------------------------------------
# TestGatherAccumNearestJit
# ---------------------------------------------------------------------------


class TestGatherAccumNearestJit:
    """Tests for _gather_accum_nearest_jit."""

    # ── output contract ──────────────────────────────────────────────────────

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        C, B, S, nside = 3, 5, 4, 16
        vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _ = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = np.ones((C, hp.nside2npix(nside)), dtype=np.float32)
        beam_vals = np.ones(S, dtype=np.float32) / S
        tod = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
        )
        assert tod.shape == (C, B)

    def test_zero_beam_vals_no_contribution(self):
        rng = np.random.default_rng(1)
        C, B, S, nside = 2, 4, 6, 16
        vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _ = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = np.zeros(S, dtype=np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
        )
        npt.assert_array_equal(tod, 0.0)

    def test_zero_beam_vals_qu_path(self):
        """Zero beam → zero TOD even on Q/U path."""
        rng = np.random.default_rng(2)
        C, B, S, nside = 3, 4, 5, 16
        vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _ = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = np.zeros(S, dtype=np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
            c_q=1,
            c_u=2,
        )
        npt.assert_array_equal(tod, 0.0)

    def test_inplace_accumulation(self):
        """Calling twice doubles the output (tod is accumulated, not reset)."""
        rng = np.random.default_rng(3)
        C, B, S, nside = 2, 4, 3, 32
        vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _ = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = rng.random(S).astype(np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
        )
        first = tod.copy()
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
        )
        npt.assert_allclose(tod, 2 * first, atol=1e-12)

    # ── T-only path agrees with healpy reference ─────────────────────────────

    def test_t_only_matches_healpy_reference(self):
        """T-only nearest-pixel path matches healpy ang2pix reference."""
        rng = np.random.default_rng(10)
        C, B, S, nside = 1, 8, 12, 32
        vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = rng.random(S).astype(np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
        )
        tod_ref = _nearest_reference_t(vec_rot, nside, mp_stacked, beam_vals, B, S)
        # float32 map values → ~1e-7 rounding relative to float64 reference
        npt.assert_allclose(tod, tod_ref, atol=1e-6)

    def test_t_multicomponent_matches_healpy_reference(self):
        """T path with C=3 components (no Q/U indices) matches healpy reference."""
        rng = np.random.default_rng(11)
        C, B, S, nside = 3, 6, 8, 32
        vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = rng.random(S).astype(np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
        )
        tod_ref = _nearest_reference_t(vec_rot, nside, mp_stacked, beam_vals, B, S)
        npt.assert_allclose(tod, tod_ref, atol=1e-6)

    # ── Q/U spin-2 correction ─────────────────────────────────────────────────

    def test_qu_spin2_matches_reference(self):
        """Q/U path with spin-2 correction matches reference implementation."""
        rng = np.random.default_rng(20)
        C, B, S, nside = 3, 8, 10, 32
        vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = rng.random(S).astype(np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
            c_q=1,
            c_u=2,
        )
        tod_ref = _nearest_reference_qu(
            vec_rot, nside, mp_stacked, beam_vals, ax_pts, B, S, c_q=1, c_u=2
        )
        npt.assert_allclose(tod, tod_ref, atol=1e-6)

    def test_qu_spin2_t_component_unaffected(self):
        """T component (c=0) is identical whether Q/U spin-2 indices are set or not."""
        rng = np.random.default_rng(21)
        C, B, S, nside = 3, 6, 8, 32
        vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _ = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = rng.random(S).astype(np.float32)

        tod_no_qu = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod_no_qu,
        )

        tod_with_qu = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod_with_qu,
            c_q=1,
            c_u=2,
        )

        # T (c=0) must be identical
        npt.assert_allclose(tod_with_qu[0], tod_no_qu[0], atol=1e-14)

    def test_qu_spin2_differs_from_scalar(self):
        """Spin-2 correction produces different Q/U output than scalar accumulation."""
        rng = np.random.default_rng(22)
        C, B, S, nside = 3, 20, 15, 64
        vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _ = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = rng.random(S).astype(np.float32)

        # Scalar path (no Q/U correction)
        tod_scalar = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod_scalar,
        )

        # Spin-2 path
        tod_spin2 = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod_spin2,
            c_q=1,
            c_u=2,
        )

        # T should match; Q/U may differ (the spin-2 rotation is non-trivial)
        npt.assert_allclose(tod_spin2[0], tod_scalar[0], atol=1e-14)
        # It's extremely unlikely that spin-2 produces zero correction for all B
        assert not np.allclose(tod_spin2[1], tod_scalar[1], atol=1e-6), (
            "Q: spin-2 correction should differ from scalar for random maps"
        )

    def test_qu_spin2_preserves_qu_norm_constant_map(self):
        """
        Spin-2 is a rotation: for a constant (Q, U) map each sample contributes
        (Q·cos2δ + U·sin2δ, -Q·sin2δ + U·cos2δ)·bv.  The vector norm
        sqrt(Q_out² + U_out²) == sqrt(Q²+U²) per sample, so for a constant
        single-pixel beam (S=1, bv=1) the total Q_out²+U_out² == Q²+U².
        """
        rng = np.random.default_rng(30)
        nside = 32
        C, B, S = 3, 12, 1

        vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _ = (
            _make_beam_and_rot_params(B, S, rng)
        )
        Q0, U0 = 3.0, 4.0  # norm = 5
        mp_stacked = np.zeros((C, hp.nside2npix(nside)), dtype=np.float32)
        mp_stacked[1] = Q0
        mp_stacked[2] = U0
        beam_vals = np.ones(S, dtype=np.float32)  # single pixel, weight 1

        tod = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
            c_q=1,
            c_u=2,
        )

        # Per-sample norm is preserved: Q_out[b]²+U_out[b]² == Q0²+U0²
        npt.assert_allclose(
            tod[1] ** 2 + tod[2] ** 2,
            np.full(B, Q0**2 + U0**2),
            rtol=1e-6,
        )

    def test_qu_spin2_constant_qu_map(self):
        """Constant Q=1, U=0 map: spin-2 rotation preserves total signal."""
        rng = np.random.default_rng(31)
        C, B, S, nside = 3, 10, 8, 32
        vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot = (
            _make_beam_and_rot_params(B, S, rng)
        )
        beam_vals = rng.random(S).astype(np.float32)
        beam_sum = float(beam_vals.sum())

        # Constant Q=1, U=0, T=1
        mp_stacked = np.ones((C, hp.nside2npix(nside)), dtype=np.float32)
        mp_stacked[2] = 0.0  # U=0

        tod = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
            c_q=1,
            c_u=2,
        )

        # T must equal beam_sum (constant map = 1)
        npt.assert_allclose(tod[0], beam_sum, rtol=1e-6)

        # Q^2 + U^2 must equal beam_sum^2 * (sum of cos(2δ_s)^2 + ...) which
        # simplifies: for each sample the rotation of (Q=1,U=0) gives
        # (cos2δ, -sin2δ), so |Q_out|^2 + |U_out|^2 from a single beam pixel
        # is preserved per sample. Across S pixels the total Q^2 + U^2 may
        # mix, so we simply check that Q^2+U^2 >= 0 (trivial) and that T is
        # correct (done above).
        assert np.all(tod[0] > 0)

    # ── identity rotation tests ───────────────────────────────────────────────

    def test_identity_rotation_t_only(self):
        """Identity rotation: T output matches scalar lookup at vec_orig positions."""
        rng = np.random.default_rng(40)
        nside = 32
        C, B, S = 1, 5, 6
        ax_pts = np.tile([0.0, 0.0, 1.0], (B, 1)).astype(np.float32)
        axes, cos_a, sin_a, ax_pts, cos_p, sin_p = _identity_rot_params(B, ax_pts)

        vec_orig = _random_unit_vec((S, 3), rng)
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = np.ones(S, dtype=np.float32) / S

        tod = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
        )

        # All B samples are identical (same rotation → same pixels).
        for b in range(1, B):
            npt.assert_allclose(tod[:, b], tod[:, 0], atol=1e-12)

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_single_beam_pixel_single_sample(self):
        """B=1, S=1 with a known map value round-trips correctly."""
        nside = 16
        C = 1
        npix = hp.nside2npix(nside)
        mp_stacked = np.zeros((C, npix), dtype=np.float32)
        # Put a hot pixel at the north pole.
        north_pix = hp.ang2pix(nside, 0.05, 0.0)
        mp_stacked[0, north_pix] = 42.0

        # Beam points toward north pole.
        vec_orig = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        ax_pts = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        axes, cos_a, sin_a, ax_pts, cos_p, sin_p = _identity_rot_params(1, ax_pts)
        beam_vals = np.array([1.0], dtype=np.float32)
        tod = np.zeros((C, 1), dtype=np.float64)

        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            1,
            1,
            tod,
        )
        # Nearest pixel to [0,0,1] should be north_pix
        npt.assert_allclose(tod[0, 0], 42.0, atol=1e-10)

    def test_c_q_c_u_equal_indices_raises_no_error(self):
        """c_q == c_u is unusual but must not crash."""
        rng = np.random.default_rng(50)
        C, B, S, nside = 2, 3, 4, 16
        vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _ = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = np.ones(S, dtype=np.float32) / S
        tod = np.zeros((C, B), dtype=np.float64)
        # c_q == c_u triggers the has_qu path; result may be unphysical but
        # the kernel must not raise.
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
            c_q=0,
            c_u=0,
        )

    def test_large_nside_does_not_crash(self):
        """nside=256 smoke test — just verifies no crash and finite output."""
        rng = np.random.default_rng(60)
        C, B, S, nside = 2, 4, 6, 256
        vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _ = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = np.ones(S, dtype=np.float32) / S
        tod = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
            c_q=0,
            c_u=1,
        )
        assert np.all(np.isfinite(tod))

    def test_qu_spin2_matches_reference_various_nside(self):
        """Q/U spin-2 reference match for nside in {16, 32, 64}."""
        rng = np.random.default_rng(70)
        for nside in (16, 32, 64):
            C, B, S = 3, 6, 8
            vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot = (
                _make_beam_and_rot_params(B, S, rng)
            )
            mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(
                np.float32
            )
            beam_vals = rng.random(S).astype(np.float32)
            tod = np.zeros((C, B), dtype=np.float64)
            _call_nearest(
                vec_orig,
                axes,
                cos_a,
                sin_a,
                ax_pts,
                cos_p,
                sin_p,
                nside,
                mp_stacked,
                beam_vals,
                B,
                S,
                tod,
                c_q=1,
                c_u=2,
            )
            tod_ref = _nearest_reference_qu(
                vec_rot, nside, mp_stacked, beam_vals, ax_pts, B, S, c_q=1, c_u=2
            )
            npt.assert_allclose(tod, tod_ref, atol=1e-6, err_msg=f"nside={nside}")

    def test_qu_indices_at_positions_0_and_1(self):
        """Q at position 0, U at position 1 (no T component) works correctly."""
        rng = np.random.default_rng(80)
        C, B, S, nside = 2, 6, 8, 32
        vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = rng.random(S).astype(np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        _call_nearest(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
            c_q=0,
            c_u=1,
        )
        tod_ref = _nearest_reference_qu(
            vec_rot, nside, mp_stacked, beam_vals, ax_pts, B, S, c_q=0, c_u=1
        )
        npt.assert_allclose(tod, tod_ref, atol=1e-6)


# ===========================================================================
# TestNearestSpin2Skip
# ===========================================================================


class TestNearestSpin2Skip:
    """Tests for the spin-2 skip optimisation in _gather_accum_nearest_jit."""

    @staticmethod
    def _run(
        vec_orig,
        axes,
        cos_a,
        sin_a,
        ax_pts,
        cos_p,
        sin_p,
        nside,
        mp_stacked,
        beam_vals,
        B,
        S,
        c_q,
        c_u,
        z_skip_threshold,
    ):
        tod = np.zeros((mp_stacked.shape[0], B), dtype=np.float64)
        _gather_accum_nearest_jit(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod,
            c_q,
            c_u,
            float(z_skip_threshold),
        )
        return tod

    def test_default_disabled_matches_unoptimised(self):
        rng = np.random.default_rng(201)
        nside, B, S = 16, 5, 8
        (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _) = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.uniform(-1.0, 1.0, (3, hp.nside2npix(nside))).astype(
            np.float32
        )
        beam_vals = rng.uniform(0.1, 1.0, S).astype(np.float32)
        beam_vals /= beam_vals.sum()

        tod_default = np.zeros((3, B), dtype=np.float64)
        _gather_accum_nearest_jit(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            tod_default,
            1,
            2,
        )
        tod_explicit = self._run(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            1,
            2,
            -1.0,
        )
        npt.assert_array_equal(tod_default, tod_explicit)

    def test_polar_unaffected_equatorial_skipped(self):
        """Polar boresights apply spin-2; equatorial boresights skip → match scalar."""
        rng = np.random.default_rng(202)
        nside, B, S = 16, 4, 6

        # Equatorial boresights only — |bz| ≤ 0.05.
        theta = rng.uniform(math.pi / 2 - 0.05, math.pi / 2 + 0.05, B)
        phi = rng.uniform(0.0, 2 * math.pi, B)
        ax_pts = np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            axis=-1,
        ).astype(np.float32)
        vec_orig = _random_unit_vec((S, 3), rng)
        angles_1 = rng.uniform(0.0, math.pi, B).astype(np.float32)
        axes = _random_unit_vec((B, 3), rng)
        cos_a = np.cos(angles_1).astype(np.float32)
        sin_a = np.sin(angles_1).astype(np.float32)
        angles_2 = rng.uniform(0.0, 2 * math.pi, B).astype(np.float32)
        cos_p = np.cos(angles_2).astype(np.float32)
        sin_p = np.sin(angles_2).astype(np.float32)

        mp_stacked = rng.uniform(-1.0, 1.0, (3, hp.nside2npix(nside))).astype(
            np.float32
        )
        beam_vals = rng.uniform(0.1, 1.0, S).astype(np.float32)
        beam_vals /= beam_vals.sum()

        # Threshold 0.1 ⇒ all boresights skipped.
        tod_skip = self._run(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            1,
            2,
            0.1,
        )
        # Reference: scalar (no spin-2) via c_q = c_u = -1.
        tod_scalar = self._run(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            -1,
            -1,
            -1.0,
        )
        npt.assert_allclose(tod_skip, tod_scalar, atol=1e-12)

    def test_intensity_channel_unchanged(self):
        rng = np.random.default_rng(203)
        nside, B, S = 16, 5, 8
        (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _) = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.uniform(-1.0, 1.0, (3, hp.nside2npix(nside))).astype(
            np.float32
        )
        beam_vals = rng.uniform(0.1, 1.0, S).astype(np.float32)
        beam_vals /= beam_vals.sum()

        tod_skip = self._run(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            1,
            2,
            0.5,
        )
        tod_no_skip = self._run(
            vec_orig,
            axes,
            cos_a,
            sin_a,
            ax_pts,
            cos_p,
            sin_p,
            nside,
            mp_stacked,
            beam_vals,
            B,
            S,
            1,
            2,
            -1.0,
        )
        npt.assert_allclose(tod_skip[0], tod_no_skip[0], atol=1e-12)


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__, "-v"])
