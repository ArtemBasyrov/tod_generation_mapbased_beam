"""
Tests for the tod_bilinear module.

- _gather_accum_jit         : scalar bilinear accumulation
- _gather_accum_fused_jit   : fused interpolation + accumulation, with and
                               without Q/U spin-2 correction
- TestSpin2SkipOptimisation : equatorial-band skip behaviour of the fused
                              kernel as a whole — the spin-2 primitives in
                              tod_spin2 are tested in test_tod_spin2.py.

Can be run independently:
    pytest tests/test_tod_bilinear.py -v
    python tests/test_tod_bilinear.py
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

from tod_bilinear import (
    _gather_accum_jit,
    _gather_accum_fused_jit,
    get_interp_weights_numba,
)
from tod_rotations import _rodrigues_jit


# ---------------------------------------------------------------------------
# Helpers shared by the fused-kernel tests.
#
# The fused kernel takes beam-frame unit vectors (vec_orig) plus a full set of
# per-sample Rodrigues parameters and applies the rotation in registers.  These
# helpers build random rotation parameters and also compute the reference
# (B, S, 3) vec_rot array via the batch Rodrigues kernel so that the existing
# numpy/healpy reference pipelines keep working unchanged.
# ---------------------------------------------------------------------------


def _random_unit_vec(shape, rng):
    v = rng.standard_normal(shape)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return v.astype(np.float32)


def _make_beam_and_rot_params(B, S, rng, pole_safe=True):
    """Build (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot_ref).

    ``vec_rot_ref`` is what the batch Rodrigues kernel produces — the reference
    the fused kernel must agree with.
    """
    vec_orig = _random_unit_vec((S, 3), rng)

    angles_1 = rng.uniform(0.0, math.pi, B).astype(np.float32)
    axes = _random_unit_vec((B, 3), rng)
    cos_a = np.cos(angles_1).astype(np.float32)
    sin_a = np.sin(angles_1).astype(np.float32)

    if pole_safe:
        # Keep boresights away from the numerical north / south poles so that
        # spin-2 γ is well defined in the reference code (sin θ > 0).
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
    """Build rotation params that act as the identity on vec_orig.

    Useful for tests that want ``vec_rot[b, s] = vec_orig[s]`` (no rotation)
    regardless of ``b``, e.g. when checking the gather/accumulate stage against
    a map-value reference that's invariant under sample index.
    """
    axes = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (B, 1))
    cos_a = np.ones(B, dtype=np.float32)
    sin_a = np.zeros(B, dtype=np.float32)
    if ax_pts is None:
        ax_pts = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (B, 1))
    cos_p = np.ones(B, dtype=np.float32)
    sin_p = np.zeros(B, dtype=np.float32)
    return axes, cos_a, sin_a, ax_pts.astype(np.float32), cos_p, sin_p


def _recenter_rot_params_to_boresight(ax_pts):
    """Build rotation params that rotate vec_orig = [0,0,1] onto each ax_pts[b].

    Rodrigues-1 does the recentre; Rodrigues-2 is identity, so the pol-roll
    leaves the result at ax_pts[b].  Intended for tests that want
    ``vec_rot[b, 0] == ax_pts[b]`` with a single-pixel beam at the beam-frame
    north pole.
    """
    B = ax_pts.shape[0]
    axes = np.zeros((B, 3), dtype=np.float32)
    cos_a = np.ones(B, dtype=np.float32)
    sin_a = np.zeros(B, dtype=np.float32)
    for b in range(B):
        tx, ty, tz = float(ax_pts[b, 0]), float(ax_pts[b, 1]), float(ax_pts[b, 2])
        angle = math.acos(max(-1.0, min(1.0, tz)))
        if angle < 1e-8:
            axes[b] = [1.0, 0.0, 0.0]
        else:
            # axis = ẑ × target (unnormalised) = (-ty, tx, 0)
            ax = np.array([-ty, tx, 0.0])
            nrm = np.linalg.norm(ax)
            if nrm > 1e-10:
                axes[b] = (ax / nrm).astype(np.float32)
            else:
                # ax_pts[b] is antipodal to [0,0,1] — any perpendicular axis works
                axes[b] = [1.0, 0.0, 0.0]
        cos_a[b] = math.cos(angle)
        sin_a[b] = math.sin(angle)
    cos_p = np.ones(B, dtype=np.float32)
    sin_p = np.zeros(B, dtype=np.float32)
    return axes, cos_a, sin_a, cos_p, sin_p


# ===========================================================================
# TestGatherAccumJit
# ===========================================================================


class TestGatherAccumJit:
    """Tests for _gather_accum_jit Numba kernel."""

    def _make_simple_inputs(self, C=2, B=3, Sc=4):
        N = B * Sc
        pixels = np.zeros((4, N), dtype=np.int64)
        weights = np.zeros((4, N), dtype=np.float64)
        weights[0] = 1.0
        beam_vals = np.ones(Sc, dtype=np.float64)
        mp_stacked = np.zeros((C, 100), dtype=np.float64)
        tod = np.zeros((C, B), dtype=np.float64)
        return pixels, weights, beam_vals, mp_stacked, tod

    def test_constant_unit_map(self):
        C, B, Sc = 2, 3, 5
        N = B * Sc
        pixels = np.zeros((4, N), dtype=np.int64)
        weights = np.full((4, N), 0.25, dtype=np.float64)
        beam_vals = np.full(Sc, 1.0 / Sc, dtype=np.float64)
        mp_stacked = np.ones((C, 100), dtype=np.float64)
        tod = np.zeros((C, B), dtype=np.float64)
        _gather_accum_jit(pixels, weights, beam_vals, mp_stacked, B, Sc, tod)
        npt.assert_allclose(tod, np.ones((C, B)), atol=1e-10)

    def test_known_single_pixel(self):
        C, B, Sc = 1, 1, 1
        N = B * Sc
        pixels = np.full((4, N), 7, dtype=np.int64)
        weights = np.zeros((4, N), dtype=np.float64)
        weights[0] = 1.0
        beam_vals = np.array([1.0], dtype=np.float64)
        mp_stacked = np.zeros((C, 20), dtype=np.float64)
        mp_stacked[0, 7] = 42.0
        tod = np.zeros((C, B), dtype=np.float64)
        _gather_accum_jit(pixels, weights, beam_vals, mp_stacked, B, Sc, tod)
        npt.assert_allclose(tod[0, 0], 42.0, atol=1e-10)

    def test_inplace_accumulation(self):
        C, B, Sc = 2, 3, 4
        pixels, weights, beam_vals, mp_stacked, tod = self._make_simple_inputs(C, B, Sc)
        mp_stacked[:] = 1.0
        _gather_accum_jit(pixels, weights, beam_vals, mp_stacked, B, Sc, tod)
        first_call = tod.copy()
        _gather_accum_jit(pixels, weights, beam_vals, mp_stacked, B, Sc, tod)
        npt.assert_allclose(tod, 2 * first_call, atol=1e-10)

    def test_zero_beam_vals(self):
        C, B, Sc = 2, 3, 4
        pixels, weights, beam_vals, mp_stacked, tod = self._make_simple_inputs(C, B, Sc)
        mp_stacked[:] = 5.0
        beam_vals[:] = 0.0
        _gather_accum_jit(pixels, weights, beam_vals, mp_stacked, B, Sc, tod)
        npt.assert_allclose(tod, np.zeros((C, B)), atol=1e-10)

    def test_output_shape_invariant(self):
        C, B, Sc = 3, 5, 7
        N = B * Sc
        pixels = np.zeros((4, N), dtype=np.int64)
        weights = np.full((4, N), 0.25, dtype=np.float64)
        beam_vals = np.ones(Sc, dtype=np.float64)
        mp_stacked = np.ones((C, 50), dtype=np.float64)
        tod = np.zeros((C, B), dtype=np.float64)
        _gather_accum_jit(pixels, weights, beam_vals, mp_stacked, B, Sc, tod)
        assert tod.shape == (C, B)


# ===========================================================================
# TestGatherAccumFusedJit
# ===========================================================================


class TestGatherAccumFusedJit:
    """Tests for _gather_accum_fused_jit (fused-Rodrigues signature).

    The kernel now takes beam-frame ``vec_orig`` and per-sample Rodrigues
    parameters and applies the rotation in registers, so tests build the
    rotation parameters via :func:`_make_beam_and_rot_params` and compute the
    reference ``vec_rot`` via the batch Rodrigues kernel to feed into the
    split-call / toy-model numpy references.
    """

    @staticmethod
    def _call_fused(
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
        _gather_accum_fused_jit(
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

    @staticmethod
    def _split_call_reference(vec_rot, nside, mp_stacked, beam_vals, B, Sc):
        """Reference: atan2 vec2ang → get_interp_weights_numba → _gather_accum_jit."""
        vf = vec_rot.reshape(-1, 3).astype(np.float64)
        rxy = np.sqrt(vf[:, 0] ** 2 + vf[:, 1] ** 2)
        theta_flat = np.arctan2(rxy, vf[:, 2])
        phi_flat = np.arctan2(vf[:, 1], vf[:, 0])
        phi_flat = np.where(phi_flat < 0.0, phi_flat + 2 * np.pi, phi_flat)
        pixels, weights = get_interp_weights_numba(nside, theta_flat, phi_flat)
        tod_ref = np.zeros((mp_stacked.shape[0], B), dtype=np.float64)
        _gather_accum_jit(
            np.asarray(pixels, dtype=np.int64),
            weights,
            beam_vals,
            mp_stacked,
            B,
            Sc,
            tod_ref,
        )
        return tod_ref

    @staticmethod
    def _toy_model_qu_reference(vec_rot, nside, mp_stacked, beam_vals, ax_pts, B, Sc):
        """Reference Q/U accumulation matching example_QU_convolution.py."""
        C = mp_stacked.shape[0]
        tod = np.zeros((C, B), dtype=np.float64)
        for i in range(B):
            vec_loc = vec_rot[i].astype(np.float64)
            theta_g, phi_g = hp.vec2ang(vec_loc)
            pix, w = hp.get_interp_weights(nside, theta_g, phi_g)
            pix_uq, inv_inv = np.unique(pix, return_inverse=True)

            theta_pix, phi_pix = hp.pix2ang(nside, pix_uq)
            bx, by, bz = float(ax_pts[i, 0]), float(ax_pts[i, 1]), float(ax_pts[i, 2])
            theta_pts = math.atan2(math.sqrt(bx**2 + by**2), bz)
            phi_pts = math.atan2(by, bx)
            if phi_pts < 0.0:
                phi_pts += 2 * math.pi

            st_pix = np.sin(theta_pix)
            cs_pix = np.cos(theta_pix)
            st_pts = math.sin(theta_pts)
            cs_pts = math.cos(theta_pts)

            dphi = phi_pts - phi_pix
            cos_dphi = np.cos(dphi)
            sin_dphi = np.sin(dphi)

            cos_beta = st_pix * st_pts * cos_dphi + cs_pix * cs_pts
            cos_beta = np.clip(cos_beta, -1.0, 1.0)
            sin_beta = np.sqrt(np.maximum(0.0, 1.0 - cos_beta**2))

            sin_alpha = st_pts * sin_dphi / np.where(sin_beta > 1e-12, sin_beta, 1.0)
            cos_alpha = (st_pts * cs_pix * cos_dphi - cs_pts * st_pix) / np.where(
                sin_beta > 1e-12, sin_beta, 1.0
            )
            sin_gamma = st_pix * sin_dphi / np.where(sin_beta > 1e-12, sin_beta, 1.0)
            cos_gamma = -(st_pix * cs_pts * cos_dphi - cs_pix * st_pts) / np.where(
                sin_beta > 1e-12, sin_beta, 1.0
            )

            mask = sin_beta < 1e-12
            sin_alpha[mask] = 0.0
            cos_alpha[mask] = 1.0
            sin_gamma[mask] = 0.0
            cos_gamma[mask] = 1.0

            cos_d = cos_alpha * cos_gamma + sin_alpha * sin_gamma
            sin_d = sin_alpha * cos_gamma - cos_alpha * sin_gamma
            cos_2d = cos_d**2 - sin_d**2
            sin_2d = 2.0 * cos_d * sin_d

            Q_pix = mp_stacked[1][pix_uq]
            U_pix = mp_stacked[2][pix_uq]
            Q_rot = cos_2d * Q_pix + sin_2d * U_pix
            U_rot = -sin_2d * Q_pix + cos_2d * U_pix

            T_full = mp_stacked[0][pix_uq][inv_inv].reshape(w.shape)
            Q_full = Q_rot[inv_inv].reshape(w.shape)
            U_full = U_rot[inv_inv].reshape(w.shape)

            T_g = np.sum(w * T_full, axis=0)
            Q_g = np.sum(w * Q_full, axis=0)
            U_g = np.sum(w * U_full, axis=0)
            tod[0, i] = np.sum(T_g * beam_vals)
            tod[1, i] = np.sum(Q_g * beam_vals)
            tod[2, i] = np.sum(U_g * beam_vals)
        return tod

    # ── output contract ──────────────────────────────────────────────────────

    def test_output_shape(self):
        rng = np.random.default_rng(40)
        C, B, S, nside = 3, 5, 4, 16
        (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _) = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = np.ones((C, hp.nside2npix(nside)), dtype=np.float32)
        beam_vals = np.ones(S, dtype=np.float32) / S
        tod = np.zeros((C, B), dtype=np.float64)
        self._call_fused(
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
        rng = np.random.default_rng(41)
        C, B, S, nside = 2, 4, 6, 16
        (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _) = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = np.zeros(S, dtype=np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        self._call_fused(
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
        npt.assert_array_equal(tod, np.zeros((C, B)))

    def test_constant_map_normalised_beam(self):
        rng = np.random.default_rng(42)
        C, B, S, nside = 2, 6, 12, 32
        map_val = 7.5
        (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _) = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = np.full((C, hp.nside2npix(nside)), map_val, dtype=np.float32)
        beam_vals = np.full(S, 1.0 / S, dtype=np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        self._call_fused(
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
        npt.assert_allclose(tod, np.full((C, B), map_val), atol=1e-4)

    def test_inplace_accumulation(self):
        rng = np.random.default_rng(43)
        C, B, S, nside = 2, 4, 5, 16
        (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _) = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = np.ones((C, hp.nside2npix(nside)), dtype=np.float32)
        beam_vals = np.ones(S, dtype=np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        self._call_fused(
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
        self._call_fused(
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
        npt.assert_allclose(tod, 2.0 * first, atol=1e-14)

    def test_deterministic(self):
        rng = np.random.default_rng(44)
        C, B, S, nside = 2, 5, 8, 16
        (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _) = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.uniform(0.0, 1.0, (C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = np.ones(S, dtype=np.float32) / S
        tod1 = np.zeros((C, B), dtype=np.float64)
        tod2 = np.zeros((C, B), dtype=np.float64)
        self._call_fused(
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
            tod1,
        )
        self._call_fused(
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
            tod2,
        )
        npt.assert_array_equal(tod1, tod2)

    # ── agreement with reference pipeline (scalar path) ──────────────────────

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_agrees_with_split_call_reference(self, nside):
        """Scalar path (c_q = c_u = -1) matches the reference pipeline to 1e-5.

        The reference builds vec_rot via the batch Rodrigues kernel, whose
        intermediates round through float32 when they land in the output
        buffer.  The fused kernel keeps its Rodrigues intermediates in float64
        registers, so it is slightly more precise — a rounding gap that shows
        up at nside=64 with ~7e-6 max absolute difference, well below the
        1e-5 tolerance used throughout this suite.
        """
        rng = np.random.default_rng(45)
        C, B, S = 3, 8, 20
        (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot) = (
            _make_beam_and_rot_params(B, S, rng)
        )
        npix = hp.nside2npix(nside)
        mp_stacked = rng.uniform(0.5, 1.5, (C, npix)).astype(np.float32)
        beam_vals = rng.uniform(0.1, 1.0, S).astype(np.float32)
        beam_vals /= beam_vals.sum()

        tod_ref = self._split_call_reference(
            vec_rot, nside, mp_stacked, beam_vals, B, S
        )
        tod_fused = np.zeros((C, B), dtype=np.float64)
        self._call_fused(
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
            tod_fused,
        )
        npt.assert_allclose(tod_fused, tod_ref, atol=1e-5)

    @pytest.mark.parametrize("nside", [4, 16])
    def test_agrees_with_healpy_reference(self, nside):
        """Scalar path is consistent with hp.get_interp_weights → _gather_accum_jit
        to 1e-5."""
        rng = np.random.default_rng(46)
        C, B, S = 3, 8, 20
        (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot) = (
            _make_beam_and_rot_params(B, S, rng)
        )
        npix = hp.nside2npix(nside)
        mp_stacked = rng.uniform(0.5, 1.5, (C, npix)).astype(np.float32)
        beam_vals = rng.uniform(0.1, 1.0, S).astype(np.float32)
        beam_vals /= beam_vals.sum()

        theta_flat, phi_flat = hp.vec2ang(vec_rot.reshape(-1, 3).astype(np.float64))
        pix_hp, wgt_hp = hp.get_interp_weights(nside, theta_flat, phi_flat)
        tod_hp = np.zeros((C, B), dtype=np.float64)
        _gather_accum_jit(
            np.asarray(pix_hp, dtype=np.int64),
            wgt_hp,
            beam_vals,
            mp_stacked,
            B,
            S,
            tod_hp,
        )
        tod_fused = np.zeros((C, B), dtype=np.float64)
        self._call_fused(
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
            tod_fused,
        )
        npt.assert_allclose(tod_fused, tod_hp, atol=1e-5)

    @pytest.mark.parametrize("nside", [4, 16])
    def test_various_sphere_regions(self, nside):
        """Scalar path agrees with reference in NPC, equatorial belt, and SPC.

        Uses identity rotation (axes/cos_a/sin_a/cos_p/sin_p = identity) so
        that ``vec_rot[b, 0] == vec_orig[0]`` and the per-sample query direction
        is the same as the boresight target we picked — covering all three
        HEALPix zones.
        """
        rng = np.random.default_rng(47)
        C, B = 2, 10
        npix = hp.nside2npix(nside)
        mp_stacked = np.ones((C, npix), dtype=np.float32)
        beam_vals = np.ones(1, dtype=np.float32)

        for label, theta_range in [
            ("NPC", (0.05, math.acos(2.0 / 3.0) - 0.02)),
            ("equatorial", (math.acos(2.0 / 3.0) + 0.02, math.acos(-2.0 / 3.0) - 0.02)),
            ("SPC", (math.acos(-2.0 / 3.0) + 0.02, np.pi - 0.05)),
        ]:
            # One independent per-b query direction per region, realised by a
            # per-b recentering rotation of vec_orig = [0, 0, 1] onto ax_pts[b].
            theta = rng.uniform(*theta_range, B)
            phi = rng.uniform(0.0, 2 * np.pi, B)
            ax_pts = np.stack(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ],
                axis=-1,
            ).astype(np.float32)

            vec_orig = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
            axes, cos_a, sin_a, cos_p, sin_p = _recenter_rot_params_to_boresight(ax_pts)

            vec_rot = np.empty((B, 1, 3), dtype=np.float32)
            _rodrigues_jit(vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot)
            tod_ref = self._split_call_reference(
                vec_rot, nside, mp_stacked, beam_vals, B, 1
            )

            tod_fused = np.zeros((C, B), dtype=np.float64)
            self._call_fused(
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
                1,
                tod_fused,
            )
            npt.assert_allclose(
                tod_fused,
                tod_ref,
                atol=1e-10,
                err_msg=f"Disagreement in {label} region at nside={nside}",
            )

    # ── Q/U spin-2 correction ────────────────────────────────────────────────

    def test_qu_constant_map_no_rotation(self):
        """Constant (Q=1, U=0) map; beam pixel = boresight direction.

        Built via a single-pixel beam at the beam-frame north pole rotated onto
        each sample's boresight — so ``vec_rot[b, 0] == ax_pts[b]`` to within
        float32 precision, matching the original test's scenario.

        At nside=64 the pixel diameter is ≈ 0.92°, so |δ_j| ≲ 0.5° and the
        error is of order (0.5π/180)² / 2 ≈ 4 × 10⁻⁵, well below 1e-3.
        """
        nside = 64
        npix = hp.nside2npix(nside)
        B, S = 4, 1
        rng = np.random.default_rng(50)

        ax_pts = _random_unit_vec((B, 3), rng)
        # Avoid vectors too close to the south pole so the recenter rotation
        # stays numerically well conditioned.
        ax_pts[ax_pts[:, 2] < -0.95, 2] = -0.9
        ax_pts /= np.linalg.norm(ax_pts, axis=-1, keepdims=True)
        ax_pts = ax_pts.astype(np.float32)

        vec_orig = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        axes, cos_a, sin_a, cos_p, sin_p = _recenter_rot_params_to_boresight(ax_pts)

        mp_stacked = np.zeros((3, npix), dtype=np.float32)
        mp_stacked[1] = 1.0  # Q = 1
        beam_vals = np.ones(S, dtype=np.float32)

        tod = np.zeros((3, B), dtype=np.float64)
        self._call_fused(
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
        npt.assert_allclose(tod[0], 0.0, atol=1e-3)
        npt.assert_allclose(tod[1], 1.0, atol=1e-3)
        npt.assert_allclose(tod[2], 0.0, atol=1e-3)

    @pytest.mark.parametrize("nside", [16, 64])
    def test_qu_agrees_with_toy_model(self, nside):
        """Q/U path matches the toy-model numpy reference to 1e-4."""
        rng = np.random.default_rng(51 + nside)
        B, S = 6, 12
        npix = hp.nside2npix(nside)

        (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot) = (
            _make_beam_and_rot_params(B, S, rng)
        )

        mp_stacked = rng.uniform(-1.0, 1.0, (3, npix)).astype(np.float32)
        beam_vals = rng.uniform(0.1, 1.0, S).astype(np.float32)
        beam_vals /= beam_vals.sum()

        tod_ref = self._toy_model_qu_reference(
            vec_rot, nside, mp_stacked, beam_vals, ax_pts, B, S
        )
        tod_fused = np.zeros((3, B), dtype=np.float64)
        self._call_fused(
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
            tod_fused,
            c_q=1,
            c_u=2,
        )
        npt.assert_allclose(
            tod_fused,
            tod_ref,
            atol=1e-4,
            rtol=1e-4,
            err_msg=f"Q/U fused kernel disagrees with toy model at nside={nside}",
        )

    def test_qu_t_unchanged_by_correction(self):
        """T component is not affected by the Q/U spin-2 rotation."""
        rng = np.random.default_rng(52)
        nside, B, S = 16, 5, 8
        npix = hp.nside2npix(nside)

        (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _) = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.uniform(-1.0, 1.0, (3, npix)).astype(np.float32)
        beam_vals = rng.uniform(0.1, 1.0, S).astype(np.float32)
        beam_vals /= beam_vals.sum()

        tod_with = np.zeros((3, B), dtype=np.float64)
        self._call_fused(
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
            tod_with,
            c_q=1,
            c_u=2,
        )

        tod_without = np.zeros((3, B), dtype=np.float64)
        self._call_fused(
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
            tod_without,
        )
        npt.assert_allclose(tod_with[0], tod_without[0], atol=1e-12)


# ===========================================================================
# TestSpin2SkipOptimisation
# ===========================================================================


class TestSpin2SkipOptimisation:
    """Tests for the spin-2 equatorial-band skip optimisation.

    Convention:
      apply_spin2 = abs(bz) > z_skip_threshold
      z_skip_threshold = -1.0  → never skip (always apply correction)
      z_skip_threshold =  1.0  → always skip (never apply correction)
    """

    @staticmethod
    def _run_fused(
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
        _gather_accum_fused_jit(
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
        """z_skip_threshold = -1.0 (default) is bit-identical to omitting the kwarg."""
        rng = np.random.default_rng(101)
        nside, B, S = 16, 6, 10
        (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _) = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.uniform(-1.0, 1.0, (3, hp.nside2npix(nside))).astype(
            np.float32
        )
        beam_vals = rng.uniform(0.1, 1.0, S).astype(np.float32)
        beam_vals /= beam_vals.sum()

        tod_default = np.zeros((3, B), dtype=np.float64)
        _gather_accum_fused_jit(
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
        tod_explicit = self._run_fused(
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

    def test_skip_all_matches_no_correction(self):
        """z_skip_threshold ≥ 1 forces apply_spin2 = False everywhere → scalar Q/U."""
        rng = np.random.default_rng(102)
        nside, B, S = 16, 6, 10
        (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _) = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.uniform(-1.0, 1.0, (3, hp.nside2npix(nside))).astype(
            np.float32
        )
        beam_vals = rng.uniform(0.1, 1.0, S).astype(np.float32)
        beam_vals /= beam_vals.sum()

        # With c_q = c_u = -1 the kernel never applies spin-2: that's the
        # reference for "no rotation" Q/U accumulation.  We compare against it
        # by reading the same map twice (once as I-only for Q's mp index,
        # once for U) — easier: just compare to the kernel run with c_q/c_u
        # set but z_skip_threshold = 1.0, vs. the same with the spin-2 disabled
        # via a separate scalar reference.
        tod_skip = self._run_fused(
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
            1.0,
        )

        # Reference: scalar bilinear gather of Q and U (no rotation).
        # Re-use the kernel but pass c_q = c_u = -1 so it treats Q/U as plain
        # scalar channels via the no-Q/U fast path.
        tod_scalar = self._run_fused(
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

    def test_polar_boresight_unaffected_by_skip(self):
        """For boresights with |bz| > z_skip_threshold the skip is not active."""
        rng = np.random.default_rng(103)
        nside, B, S = 16, 5, 8
        # Force boresights near the north pole: θ ∈ [0.05, 0.2] → cos θ > 0.97
        theta = rng.uniform(0.05, 0.2, B)
        phi = rng.uniform(0.0, 2 * math.pi, B)
        ax_pts = np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            axis=-1,
        ).astype(np.float32)
        # Build the rest of the rotation params with these polar boresights.
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

        # z_skip_threshold = 0.5 — boresights have |cos θ| > 0.97 so all polar.
        tod_skip = self._run_fused(
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
        tod_no_skip = self._run_fused(
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
        npt.assert_allclose(tod_skip, tod_no_skip, atol=1e-12)

    def test_equatorial_boresight_skip_active(self):
        """For boresights with |bz| ≤ z_skip_threshold the skip is active.

        Compare the kernel run with z_skip_threshold ≥ |bz_max| against the
        scalar (no spin-2) reference: both should match.
        """
        rng = np.random.default_rng(104)
        nside, B, S = 16, 5, 8
        # Force boresights near the equator: θ ∈ [π/2 - 0.05, π/2 + 0.05].
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

        # |bz| ≤ sin(0.05) ≈ 0.05; threshold = 0.1 ⇒ all skipped.
        tod_skip = self._run_fused(
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
        tod_scalar = self._run_fused(
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
        """Skip optimisation must not affect the I (non-Q/U) channel."""
        rng = np.random.default_rng(105)
        nside, B, S = 16, 5, 8
        (vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, _) = (
            _make_beam_and_rot_params(B, S, rng)
        )
        mp_stacked = rng.uniform(-1.0, 1.0, (3, hp.nside2npix(nside))).astype(
            np.float32
        )
        beam_vals = rng.uniform(0.1, 1.0, S).astype(np.float32)
        beam_vals /= beam_vals.sum()

        # Run with skip aggressive (skips half the boresights randomly) and
        # without skip; I channel must be identical.
        tod_skip = self._run_fused(
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
        tod_no_skip = self._run_fused(
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
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
