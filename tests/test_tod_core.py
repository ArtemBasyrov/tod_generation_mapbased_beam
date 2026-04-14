"""
Tests for the tod_core module.

- tod_core    : beam_tod_batch,

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
    precompute_rotation_vector_batch,
    beam_tod_batch,
)

# ---------------------------------------------------------------------------
# Shared RNG (deterministic across all tests)
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(0)

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
        rot_vecs, betas, _ = precompute_rotation_vector_batch(
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


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
