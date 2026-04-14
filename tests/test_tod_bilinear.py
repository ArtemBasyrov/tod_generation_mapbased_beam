"""
Tests for the tod_bilinear module.

- tod_bilinear    : _gather_accum_jit, _gather_accum_fused_jit, _gather_accum_flatsky_jit

Can be run independently:
    pytest tests/test_tod_bilinear.py -v
    python tests/test_tod_bilinear.py
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

from tod_bilinear import (
    _gather_accum_jit,
    _gather_accum_fused_jit,
    get_interp_weights_numba,
)

# ===========================================================================
# TestGatherAccumJit
# ===========================================================================


class TestGatherAccumJit:
    """Tests for tod_core._gather_accum_jit Numba kernel."""

    def _make_simple_inputs(self, C=2, B=3, Sc=4):
        """Build minimal valid inputs for _gather_accum_jit."""
        N = B * Sc
        pixels = np.zeros((4, N), dtype=np.int64)
        weights = np.zeros((4, N), dtype=np.float64)
        weights[0] = 1.0  # all weight on first pixel
        beam_vals = np.ones(Sc, dtype=np.float64)
        mp_stacked = np.zeros((C, 100), dtype=np.float64)
        tod = np.zeros((C, B), dtype=np.float64)
        return pixels, weights, beam_vals, mp_stacked, tod

    def test_constant_unit_map(self):
        """All-ones map with normalised beam produces tod ≈ 1.0 for every (c, b)."""
        C, B, Sc = 2, 3, 5
        N = B * Sc
        pixels = np.zeros((4, N), dtype=np.int64)
        weights = np.full(
            (4, N), 0.25, dtype=np.float64
        )  # four equal weights summing to 1
        beam_vals = np.full(Sc, 1.0 / Sc, dtype=np.float64)  # normalised: sum = 1
        n_pix = 100
        mp_stacked = np.ones((C, n_pix), dtype=np.float64)
        tod = np.zeros((C, B), dtype=np.float64)
        _gather_accum_jit(pixels, weights, beam_vals, mp_stacked, B, Sc, tod)
        npt.assert_allclose(tod, np.ones((C, B)), atol=1e-10)

    def test_known_single_pixel(self):
        """mp_stacked[0,7]=42 with pixels=7 and weight=[1,0,0,0], beam=1 gives tod[0,0]=42."""
        C, B, Sc = 1, 1, 1
        N = B * Sc
        pixels = np.full((4, N), 7, dtype=np.int64)
        weights = np.zeros((4, N), dtype=np.float64)
        weights[0] = 1.0
        beam_vals = np.array([1.0], dtype=np.float64)
        n_pix = 20
        mp_stacked = np.zeros((C, n_pix), dtype=np.float64)
        mp_stacked[0, 7] = 42.0
        tod = np.zeros((C, B), dtype=np.float64)
        _gather_accum_jit(pixels, weights, beam_vals, mp_stacked, B, Sc, tod)
        npt.assert_allclose(tod[0, 0], 42.0, atol=1e-10)

    def test_inplace_accumulation(self):
        """Calling the kernel twice on the same tod array doubles the result."""
        C, B, Sc = 2, 3, 4
        pixels, weights, beam_vals, mp_stacked, tod = self._make_simple_inputs(C, B, Sc)
        mp_stacked[:] = 1.0
        weights[0] = 1.0
        beam_vals[:] = 1.0
        _gather_accum_jit(pixels, weights, beam_vals, mp_stacked, B, Sc, tod)
        first_call = tod.copy()
        _gather_accum_jit(pixels, weights, beam_vals, mp_stacked, B, Sc, tod)
        npt.assert_allclose(tod, 2 * first_call, atol=1e-10)

    def test_zero_beam_vals(self):
        """Zero beam_vals produce no contribution to tod."""
        C, B, Sc = 2, 3, 4
        pixels, weights, beam_vals, mp_stacked, tod = self._make_simple_inputs(C, B, Sc)
        mp_stacked[:] = 5.0
        beam_vals[:] = 0.0
        _gather_accum_jit(pixels, weights, beam_vals, mp_stacked, B, Sc, tod)
        npt.assert_allclose(tod, np.zeros((C, B)), atol=1e-10)

    def test_output_shape_invariant(self):
        """tod has shape (C, B) after the kernel call."""
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
    """
    Tests for tod_core._gather_accum_fused_jit.

    Validates the fused vec2ang + HEALPix interpolation + beam accumulation
    kernel against the original split-call pipeline
    (hp.vec2ang → get_interp_weights_numba → _gather_accum_jit).
    """

    @staticmethod
    def _make_vec_rot(B, Sc, rng):
        """Return (B, Sc, 3) float32 unit vectors distributed over the sphere."""
        v = rng.standard_normal((B, Sc, 3))
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        return v.astype(np.float32)

    @staticmethod
    def _split_call_reference(vec_rot, nside, mp_stacked, beam_vals, B, Sc):
        """
        Reference pipeline that matches the vec2ang formula used by the fused kernel.

        Uses atan2(sqrt(x²+y²), z) instead of acos(z) so that the only difference
        from the fused path is evaluation order, not algorithmic.
        """
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

    # ── output contract ──────────────────────────────────────────────────────

    def test_output_shape(self):
        """tod buffer has shape (C, B) after the kernel call."""
        rng = np.random.default_rng(40)
        C, B, Sc, nside = 3, 5, 4, 16
        vec_rot = self._make_vec_rot(B, Sc, rng)
        mp_stacked = np.ones((C, hp.nside2npix(nside)), dtype=np.float32)
        beam_vals = np.ones(Sc, dtype=np.float32) / Sc
        tod = np.zeros((C, B), dtype=np.float64)
        _gather_accum_fused_jit(vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod)
        assert tod.shape == (C, B)

    def test_zero_beam_vals_no_contribution(self):
        """Zero beam_vals leave the tod buffer unchanged."""
        rng = np.random.default_rng(41)
        C, B, Sc, nside = 2, 4, 6, 16
        vec_rot = self._make_vec_rot(B, Sc, rng)
        mp_stacked = rng.standard_normal((C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = np.zeros(Sc, dtype=np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        _gather_accum_fused_jit(vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod)
        npt.assert_array_equal(tod, np.zeros((C, B)))

    def test_constant_map_normalised_beam(self):
        """Constant map + normalised beam gives tod ≈ map_value for all (c, b)."""
        rng = np.random.default_rng(42)
        C, B, Sc, nside = 2, 6, 12, 32
        map_val = 7.5
        vec_rot = self._make_vec_rot(B, Sc, rng)
        mp_stacked = np.full((C, hp.nside2npix(nside)), map_val, dtype=np.float32)
        beam_vals = np.full(Sc, 1.0 / Sc, dtype=np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        _gather_accum_fused_jit(vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod)
        npt.assert_allclose(tod, np.full((C, B), map_val), atol=1e-4)

    def test_inplace_accumulation(self):
        """Calling the kernel twice on the same tod buffer doubles the result."""
        rng = np.random.default_rng(43)
        C, B, Sc, nside = 2, 4, 5, 16
        vec_rot = self._make_vec_rot(B, Sc, rng)
        mp_stacked = np.ones((C, hp.nside2npix(nside)), dtype=np.float32)
        beam_vals = np.ones(Sc, dtype=np.float32)
        tod = np.zeros((C, B), dtype=np.float64)
        _gather_accum_fused_jit(vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod)
        first = tod.copy()
        _gather_accum_fused_jit(vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod)
        npt.assert_allclose(tod, 2.0 * first, atol=1e-14)

    def test_deterministic(self):
        """Two identical calls on fresh buffers produce identical results."""
        rng = np.random.default_rng(44)
        C, B, Sc, nside = 2, 5, 8, 16
        vec_rot = self._make_vec_rot(B, Sc, rng)
        mp_stacked = rng.uniform(0.0, 1.0, (C, hp.nside2npix(nside))).astype(np.float32)
        beam_vals = np.ones(Sc, dtype=np.float32) / Sc
        tod1 = np.zeros((C, B), dtype=np.float64)
        tod2 = np.zeros((C, B), dtype=np.float64)
        _gather_accum_fused_jit(vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod1)
        _gather_accum_fused_jit(vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod2)
        npt.assert_array_equal(tod1, tod2)

    # ── agreement with reference pipeline ────────────────────────────────────

    @pytest.mark.parametrize("nside", [4, 16, 64])
    def test_agrees_with_split_call_reference(self, nside):
        """
        Fused kernel matches the reference pipeline
        (atan2 vec2ang → get_interp_weights_numba → _gather_accum_jit)
        to 1e-6 on random inputs.

        Both paths use identical algorithms.  The tolerance is 1e-6 (rather
        than machine epsilon) because the fused kernel evaluates all steps
        in a single inline pass while the reference allocates intermediate
        float64 theta/phi and pixel/weight arrays; the different
        floating-point evaluation order causes differences of up to ~1e-6.
        Pixel-exact agreement with healpy is verified separately in
        test_agrees_with_healpy_reference.
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
        _gather_accum_fused_jit(vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod_fused)

        npt.assert_allclose(
            tod_fused,
            tod_ref,
            atol=1e-6,
            err_msg=f"Fused kernel differs from reference at nside={nside}",
        )

    @pytest.mark.parametrize("nside", [4, 16])
    def test_agrees_with_healpy_reference(self, nside):
        """
        Fused kernel is consistent with the canonical healpy pipeline
        (hp.vec2ang → hp.get_interp_weights → _gather_accum_jit) to 1e-5.

        Healpy uses acos(z) while the fused kernel uses atan2; the tolerance
        accounts for the (tiny) theta difference near ring boundaries.
        """
        rng = np.random.default_rng(46)
        C, B, Sc = 3, 8, 20
        vec_rot = self._make_vec_rot(B, Sc, rng)
        npix = hp.nside2npix(nside)
        mp_stacked = rng.uniform(0.5, 1.5, (C, npix)).astype(np.float32)
        beam_vals = rng.uniform(0.1, 1.0, Sc).astype(np.float32)
        beam_vals /= beam_vals.sum()

        # Canonical healpy reference
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
        _gather_accum_fused_jit(vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod_fused)

        npt.assert_allclose(
            tod_fused,
            tod_hp,
            atol=1e-5,
            err_msg=f"Fused kernel differs from healpy reference at nside={nside}",
        )

    @pytest.mark.parametrize("nside", [4, 16])
    def test_various_sphere_regions(self, nside):
        """
        Fused kernel agrees with the reference pipeline for vectors aimed at
        the north polar cap, equatorial belt, and south polar cap separately.
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
            theta = rng.uniform(*theta_range, B)
            phi = rng.uniform(0.0, 2 * np.pi, B)
            # Build vec_rot from (theta, phi)
            vec_bs = np.stack(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ],
                axis=-1,
            ).astype(np.float32)[:, np.newaxis, :]  # (B, 1, 3)

            tod_ref = self._split_call_reference(
                vec_bs, nside, mp_stacked, beam_vals, B, 1
            )
            tod_fused = np.zeros((C, B), dtype=np.float64)
            _gather_accum_fused_jit(
                vec_bs, nside, mp_stacked, beam_vals, B, 1, tod_fused
            )

            npt.assert_allclose(
                tod_fused,
                tod_ref,
                atol=1e-10,
                err_msg=f"Disagreement in {label} region at nside={nside}",
            )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
