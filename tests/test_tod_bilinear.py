"""
Tests for the tod_bilinear module.

- _gather_accum_jit         : scalar bilinear accumulation
- _spin2_cos2d_sin2d_jit    : spherical-trig spin-2 rotation angles
- _gather_accum_fused_jit   : fused interpolation + accumulation, with and
                               without Q/U spin-2 correction

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
    _spin2_cos2d_sin2d_jit,
    get_interp_weights_numba,
)


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
# TestSpin2Cos2dSin2d
# ===========================================================================


class TestSpin2Cos2dSin2d:
    """Unit tests for _spin2_cos2d_sin2d_jit."""

    @staticmethod
    def _angles(theta, phi):
        return math.cos(theta), math.sin(theta), phi

    def test_identity_same_point(self):
        """Same pixel and target → no rotation (cos=1, sin=0)."""
        z, s, p = self._angles(0.5, 1.2)
        c2, s2 = _spin2_cos2d_sin2d_jit(z, s, p, z, s, p)
        assert abs(c2 - 1.0) < 1e-12
        assert abs(s2) < 1e-12

    def test_unit_norm(self):
        """cos²(2δ) + sin²(2δ) ≈ 1 for random pairs."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            t1, p1 = rng.uniform(0.05, math.pi - 0.05), rng.uniform(0.0, 2 * math.pi)
            t2, p2 = rng.uniform(0.05, math.pi - 0.05), rng.uniform(0.0, 2 * math.pi)
            z1, s1 = math.cos(t1), math.sin(t1)
            z2, s2 = math.cos(t2), math.sin(t2)
            c2, si2 = _spin2_cos2d_sin2d_jit(z1, s1, p1, z2, s2, p2)
            npt.assert_allclose(c2**2 + si2**2, 1.0, atol=1e-12)

    def test_antisymmetry(self):
        """Swapping pixel and target negates δ: cos(2δ) is even, sin(2δ) is odd."""
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

    def test_near_pole_no_nan(self):
        """Near-antipodal/coincident points return (1.0, 0.0) without NaN."""
        # Nearly same point
        c2, s2 = _spin2_cos2d_sin2d_jit(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        assert math.isfinite(c2) and math.isfinite(s2)
        # Nearly antipodal
        c2, s2 = _spin2_cos2d_sin2d_jit(1.0, 0.0, 0.0, -1.0, 0.0, math.pi)
        assert math.isfinite(c2) and math.isfinite(s2)


# ===========================================================================
# TestGatherAccumFusedJit
# ===========================================================================


class TestGatherAccumFusedJit:
    """
    Tests for _gather_accum_fused_jit.

    Validates the fused vec2ang + HEALPix interpolation + beam accumulation
    kernel against the original split-call pipeline and against the toy-model
    Q/U rotation reference.
    """

    @staticmethod
    def _make_vec_rot(B, Sc, rng):
        v = rng.standard_normal((B, Sc, 3))
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        return v.astype(np.float32)

    @staticmethod
    def _dummy_rot_targets(B):
        return (
            np.zeros((B, 3), dtype=np.float32),
            np.zeros((B, 3), dtype=np.float32),
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
        """
        Reference Q/U accumulation matching example_QU_convolution.py.

        For each sample b: get interp weights for all Sc beam pixels,
        find unique HEALPix pixels, rotate (Q, U) at each unique pixel into
        the boresight frame via the spherical-trig delta formula, then
        accumulate with bilinear weights × beam weights.
        """
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
        C, B, Sc, nside = 3, 5, 4, 16
        vec_rot = self._make_vec_rot(B, Sc, rng)
        mp_stacked = np.ones((C, hp.nside2npix(nside)), dtype=np.float32)
        beam_vals = np.ones(Sc, dtype=np.float32) / Sc
        tod = np.zeros((C, B), dtype=np.float64)
        ax_pts, n_target = self._dummy_rot_targets(B)
        _gather_accum_fused_jit(
            vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod, ax_pts, n_target
        )
        assert tod.shape == (C, B)

    def test_zero_beam_vals_no_contribution(self):
        rng = np.random.default_rng(41)
        C, B, Sc, nside = 2, 4, 6, 16
        vec_rot = self._make_vec_rot(B, Sc, rng)
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = np.zeros(Sc, dtype=np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        ax_pts, n_target = self._dummy_rot_targets(B)
        _gather_accum_fused_jit(
            vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod, ax_pts, n_target
        )
        npt.assert_array_equal(tod, np.zeros((C, B)))

    def test_constant_map_normalised_beam(self):
        rng = np.random.default_rng(42)
        C, B, Sc, nside = 2, 6, 12, 32
        map_val = 7.5
        vec_rot = self._make_vec_rot(B, Sc, rng)
        mp_stacked = np.full((C, hp.nside2npix(nside)), map_val, dtype=np.float32)
        beam_vals = np.full(Sc, 1.0 / Sc, dtype=np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        ax_pts, n_target = self._dummy_rot_targets(B)
        _gather_accum_fused_jit(
            vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod, ax_pts, n_target
        )
        npt.assert_allclose(tod, np.full((C, B), map_val), atol=1e-4)

    def test_inplace_accumulation(self):
        rng = np.random.default_rng(43)
        C, B, Sc, nside = 2, 4, 5, 16
        vec_rot = self._make_vec_rot(B, Sc, rng)
        mp_stacked = np.ones((C, hp.nside2npix(nside)), dtype=np.float32)
        beam_vals = np.ones(Sc, dtype=np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        ax_pts, n_target = self._dummy_rot_targets(B)
        _gather_accum_fused_jit(
            vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod, ax_pts, n_target
        )
        first = tod.copy()
        _gather_accum_fused_jit(
            vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod, ax_pts, n_target
        )
        npt.assert_allclose(tod, 2.0 * first, atol=1e-14)

    def test_deterministic(self):
        rng = np.random.default_rng(44)
        C, B, Sc, nside = 2, 5, 8, 16
        vec_rot = self._make_vec_rot(B, Sc, rng)
        mp_stacked = rng.uniform(0.0, 1.0, (C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = np.ones(Sc, dtype=np.float32) / Sc
        tod1 = np.zeros((C, B), dtype=np.float64)
        tod2 = np.zeros((C, B), dtype=np.float64)
        ax_pts, n_target = self._dummy_rot_targets(B)
        _gather_accum_fused_jit(
            vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod1, ax_pts, n_target
        )
        _gather_accum_fused_jit(
            vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod2, ax_pts, n_target
        )
        npt.assert_array_equal(tod1, tod2)

    # ── agreement with reference pipeline (scalar path) ──────────────────────

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_agrees_with_split_call_reference(self, nside):
        """
        Scalar path (c_q = c_u = -1) matches the reference pipeline to 1e-6.
        """
        rng = np.random.default_rng(45)
        C, B, Sc = 3, 8, 20
        vec_rot = self._make_vec_rot(B, Sc, rng)
        npix = hp.nside2npix(nside)
        mp_stacked = rng.uniform(0.5, 1.5, (C, npix)).astype(np.float32)
        beam_vals = rng.uniform(0.1, 1.0, Sc).astype(np.float32)
        beam_vals /= beam_vals.sum()

        tod_ref = self._split_call_reference(
            vec_rot, nside, mp_stacked, beam_vals, B, Sc
        )
        tod_fused = np.zeros((C, B), dtype=np.float64)
        ax_pts, n_target = self._dummy_rot_targets(B)
        _gather_accum_fused_jit(
            vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod_fused, ax_pts, n_target
        )
        npt.assert_allclose(tod_fused, tod_ref, atol=1e-6)

    @pytest.mark.parametrize("nside", [4, 16])
    def test_agrees_with_healpy_reference(self, nside):
        """
        Scalar path is consistent with hp.get_interp_weights → _gather_accum_jit
        to 1e-5.
        """
        rng = np.random.default_rng(46)
        C, B, Sc = 3, 8, 20
        vec_rot = self._make_vec_rot(B, Sc, rng)
        npix = hp.nside2npix(nside)
        mp_stacked = rng.uniform(0.5, 1.5, (C, npix)).astype(np.float32)
        beam_vals = rng.uniform(0.1, 1.0, Sc).astype(np.float32)
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
            Sc,
            tod_hp,
        )

        tod_fused = np.zeros((C, B), dtype=np.float64)
        ax_pts, n_target = self._dummy_rot_targets(B)
        _gather_accum_fused_jit(
            vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod_fused, ax_pts, n_target
        )
        npt.assert_allclose(tod_fused, tod_hp, atol=1e-5)

    @pytest.mark.parametrize("nside", [4, 16])
    def test_various_sphere_regions(self, nside):
        """Scalar path agrees with reference in NPC, equatorial belt, and SPC."""
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
            theta = rng.uniform(*theta_range, B)
            phi = rng.uniform(0.0, 2 * np.pi, B)
            vec_bs = np.stack(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ],
                axis=-1,
            ).astype(np.float32)[:, np.newaxis, :]

            tod_ref = self._split_call_reference(
                vec_bs, nside, mp_stacked, beam_vals, B, 1
            )
            tod_fused = np.zeros((C, B), dtype=np.float64)
            ax_pts, n_target = self._dummy_rot_targets(B)
            _gather_accum_fused_jit(
                vec_bs, nside, mp_stacked, beam_vals, B, 1, tod_fused, ax_pts, n_target
            )
            npt.assert_allclose(
                tod_fused,
                tod_ref,
                atol=1e-10,
                err_msg=f"Disagreement in {label} region at nside={nside}",
            )

    # ── Q/U spin-2 correction ────────────────────────────────────────────────

    def _make_boresight_targets(self, B, rng):
        """Random boresight positions (B, 3) float32 unit vectors."""
        v = rng.standard_normal((B, 3))
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        return v.astype(np.float32)

    def test_qu_constant_map_no_rotation(self):
        """
        Constant (Q=1, U=0) map; beam pixel = boresight direction.

        The 4 bilinear neighbours are the ring pixels surrounding the boresight,
        each at angular separation ≲ half a HEALPix pixel.  The rotation angle δ
        for each neighbour is therefore small (≪ 1 rad), giving:
            Q_out = Σ w_j cos(2δ_j) ≈ 1 − O(δ²)
            U_out = Σ w_j sin(2δ_j) ≈ 0

        At nside=64 the pixel diameter is ≈ 0.92°, so |δ_j| ≲ 0.5° and
        the error is of order (0.5π/180)² / 2 ≈ 4 × 10⁻⁵, well below 1e-3.
        """
        nside = 64  # fine enough that neighbour δ is negligible
        npix = hp.nside2npix(nside)
        B, Sc = 4, 1
        rng = np.random.default_rng(50)

        ax_pts = self._make_boresight_targets(B, rng)
        vec_rot = ax_pts[:, np.newaxis, :].copy()

        mp_stacked = np.zeros((3, npix), dtype=np.float32)
        mp_stacked[1] = 1.0
        beam_vals = np.ones(Sc, dtype=np.float32)
        n_target = np.zeros((B, 3), dtype=np.float32)

        tod = np.zeros((3, B), dtype=np.float64)
        _gather_accum_fused_jit(
            vec_rot,
            nside,
            mp_stacked,
            beam_vals,
            B,
            Sc,
            tod,
            ax_pts,
            n_target,
            c_q=1,
            c_u=2,
        )
        npt.assert_allclose(tod[0], 0.0, atol=1e-3)
        npt.assert_allclose(tod[1], 1.0, atol=1e-3)
        npt.assert_allclose(tod[2], 0.0, atol=1e-3)

    @pytest.mark.parametrize("nside", [16, 64])
    def test_qu_agrees_with_toy_model(self, nside):
        """
        Q/U path matches the toy-model numpy reference (example_QU_convolution.py)
        to 1e-4 (float32 map + bilinear interp agreement).
        """
        rng = np.random.default_rng(51 + nside)
        B, Sc = 6, 12
        npix = hp.nside2npix(nside)

        vec_rot = self._make_vec_rot(B, Sc, rng)
        ax_pts = self._make_boresight_targets(B, rng)
        n_target = np.zeros((B, 3), dtype=np.float32)  # not used by new kernel

        mp_stacked = rng.uniform(-1.0, 1.0, (3, npix)).astype(np.float32)
        beam_vals = rng.uniform(0.1, 1.0, Sc).astype(np.float32)
        beam_vals /= beam_vals.sum()

        tod_ref = self._toy_model_qu_reference(
            vec_rot, nside, mp_stacked, beam_vals, ax_pts, B, Sc
        )
        tod_fused = np.zeros((3, B), dtype=np.float64)
        _gather_accum_fused_jit(
            vec_rot,
            nside,
            mp_stacked,
            beam_vals,
            B,
            Sc,
            tod_fused,
            ax_pts,
            n_target,
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
        nside, B, Sc = 16, 5, 8
        npix = hp.nside2npix(nside)

        vec_rot = self._make_vec_rot(B, Sc, rng)
        ax_pts = self._make_boresight_targets(B, rng)
        n_target = np.zeros((B, 3), dtype=np.float32)
        mp_stacked = rng.uniform(-1.0, 1.0, (3, npix)).astype(np.float32)
        beam_vals = rng.uniform(0.1, 1.0, Sc).astype(np.float32)
        beam_vals /= beam_vals.sum()

        # T component with Q/U correction active
        tod_with = np.zeros((3, B), dtype=np.float64)
        _gather_accum_fused_jit(
            vec_rot,
            nside,
            mp_stacked,
            beam_vals,
            B,
            Sc,
            tod_with,
            ax_pts,
            n_target,
            c_q=1,
            c_u=2,
        )

        # T component without Q/U correction (c_q=c_u=-1)
        tod_without = np.zeros((3, B), dtype=np.float64)
        _gather_accum_fused_jit(
            vec_rot,
            nside,
            mp_stacked,
            beam_vals,
            B,
            Sc,
            tod_without,
            ax_pts,
            n_target,
        )
        npt.assert_allclose(tod_with[0], tod_without[0], atol=1e-12)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
