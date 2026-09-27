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
        """Output has exactly comp_indices keys, each (B,) in the working precision."""
        import tod_config

        nside = 32
        B = 5
        data = self._build_data(S=30)
        phi_b, theta_b, psis_b, rot_vecs = self._build_scan(B)
        mp = self._constant_maps(nside)

        tod = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b)

        assert set(tod.keys()) == set(data["comp_indices"])
        for comp in data["comp_indices"]:
            assert tod[comp].shape == (B,), f"Wrong shape for comp={comp}"
            assert tod[comp].dtype == tod_config.precision_dtype, (
                f"Wrong dtype for comp={comp}"
            )

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

    def test_per_component_weights_independent(self):
        """A (C, S) beam_vals weights each component by its own row, on both the
        fused and numpy-fallback paths."""
        nside = 32
        B = 6
        mp = self._constant_maps(nside)
        data_base = self._build_data(S=30)
        phi_b, theta_b, psis_b, rot_vecs = self._build_scan(B)
        comp_indices = data_base["comp_indices"]
        C = len(comp_indices)

        base = data_base["beam_vals"].astype(np.float32)
        bv_2d = np.stack([base * (i + 1) for i in range(C)]).astype(np.float32)
        mp_stacked = np.stack([mp[c] for c in comp_indices]).astype(np.float32)

        def run(bv, stacked):
            d = dict(data_base)
            d["beam_vals"] = bv
            d["mp_stacked"] = stacked
            return beam_tod_batch(nside, mp, d, rot_vecs, phi_b, theta_b, psis_b)

        tod = run(bv_2d, mp_stacked)
        tod_single = run(base, mp_stacked)
        for i, comp in enumerate(comp_indices):
            npt.assert_allclose(tod[comp], tod_single[comp] * (i + 1), rtol=1e-5)

        bv_scaled_q = bv_2d.copy()
        bv_scaled_q[1] *= 2.0
        tod_q = run(bv_scaled_q, mp_stacked)
        npt.assert_allclose(tod_q["I"], tod["I"], rtol=1e-6)
        npt.assert_allclose(tod_q["U"], tod["U"], rtol=1e-6)
        npt.assert_allclose(tod_q["Q"], tod["Q"] * 2.0, rtol=1e-5)

        tod_fallback = run(bv_2d, None)
        for comp in comp_indices:
            npt.assert_allclose(tod[comp], tod_fallback[comp], atol=1e-4)

    def test_separate_qu_sources_keep_transport(self):
        """Q and U read from separate beam files must still receive the spin-2
        transport once merged: the TOD equals a single shared Q/U source, and
        differs from gathering the two sources in separate calls."""
        from tod_pipeline_helpers import merge_beam_entries

        nside = 32
        B = 6
        rng = np.random.default_rng(11)
        npix = hp.nside2npix(nside)
        mp = {c: rng.uniform(-1.0, 1.0, npix).astype(np.float64) for c in (1, 2)}
        base = self._build_data(S=30)
        grid = np.zeros((3, 3))

        def entry(comps):
            return {
                "ra": grid,
                "dec": grid,
                "vec_orig": base["vec_orig"],
                "beam_vals": base["beam_vals"],
                "comp_indices": list(comps),
            }

        # High-latitude boresights, where the transport angle is large.
        N = 201
        phi_b = rng.uniform(0.0, 2 * np.pi, B)
        theta_b = rng.uniform(0.1, 0.3, B)
        rot_vecs, betas = precompute_rotation_vector_batch(
            np.zeros((N, N)),
            np.zeros((N, N)),
            phi_b,
            theta_b,
            center_idx=(N // 2, N // 2),
        )
        psis_b = -betas
        chi_b = rng.uniform(0.0, 2 * np.pi, B)

        def gather(beam_data):
            out = {1: np.zeros(B), 2: np.zeros(B)}
            for d in beam_data.values():
                d["mp_stacked"] = np.stack([mp[c] for c in d["comp_indices"]])
                for c, v in beam_tod_batch(
                    nside, mp, d, rot_vecs, phi_b, theta_b, psis_b, chi_b=chi_b
                ).items():
                    out[c] += v
            return out

        shared = gather({"qu": entry([1, 2])})
        split = {"q": entry([1]), "u": entry([2])}
        merged = gather(merge_beam_entries(split))
        unmerged = gather({k: entry(v["comp_indices"]) for k, v in split.items()})

        for c in (1, 2):
            npt.assert_allclose(merged[c], shared[c], rtol=1e-5, atol=1e-7)
        assert np.max(np.abs(unmerged[1] - shared[1])) > 1e-3


# ===========================================================================
# Unequal Q/U beams act in the detector basis
# ===========================================================================


class TestDetectorBasisBeams:
    """Per-component Q/U beams are defined in the detector basis, so with
    ``b_Q != b_U`` the TOD must equal weighting ``P e^{-2i chi}`` component-wise.
    The reference is built from two equal-beam runs (weights ``b_Q`` and ``b_U``),
    which do not depend on chi."""

    nside = 32
    B = 12
    N = 201

    @staticmethod
    def _rtol():
        import tod_config

        return 1e-10 if np.dtype(tod_config.precision_dtype) == np.float64 else 1e-6

    def _setup(self, S, seed):
        rng = np.random.default_rng(seed)
        npix = hp.nside2npix(self.nside)
        mp = {c: rng.uniform(-1.0, 1.0, npix) for c in (0, 1, 2)}
        th = rng.uniform(0.0, 0.03, S)
        ph = rng.uniform(0.0, 2 * np.pi, S)
        vec = np.stack(
            [np.cos(th), np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)], axis=-1
        )
        phi_b = rng.uniform(0.0, 2 * np.pi, self.B)
        # Polar and equatorial boresights, so the transport and the skip band both run.
        theta_b = np.concatenate(
            [rng.uniform(0.1, 0.4, self.B // 2), rng.uniform(1.5, 1.64, self.B // 2)]
        )
        psi_b = rng.uniform(0.0, 2 * np.pi, self.B)
        chi_b = rng.uniform(0.0, 2 * np.pi, self.B)
        g = np.zeros((self.N, self.N))
        rot_vecs, betas = precompute_rotation_vector_batch(
            g, g, phi_b, theta_b, center_idx=(self.N // 2, self.N // 2)
        )
        return rng, mp, vec, (rot_vecs, phi_b, theta_b, psi_b - betas), chi_b

    def _run(self, mp, vec, bv, scan, chi_b, mode, stacked, z_skip=-1.0):
        data = {"vec_orig": vec, "beam_vals": bv, "comp_indices": [0, 1, 2]}
        data["mp_stacked"] = np.stack([mp[c] for c in (0, 1, 2)]) if stacked else None
        rot_vecs, phi_b, theta_b, psis_b = scan
        out = beam_tod_batch(
            self.nside,
            mp,
            data,
            rot_vecs,
            phi_b,
            theta_b,
            psis_b,
            interp_mode=mode,
            z_skip_threshold=z_skip,
            chi_b=chi_b,
        )
        return out[0], out[1] + 1j * out[2]

    @pytest.mark.parametrize(
        "mode,stacked,z_skip",
        [
            ("bilinear", True, -1.0),
            ("bilinear", True, 0.3),
            ("nearest", True, -1.0),
            ("nearest", True, 0.3),
            ("bilinear", False, -1.0),
        ],
    )
    def test_matches_detector_basis_reference(self, mode, stacked, z_skip):
        S = 40
        rng, mp, vec, scan, chi_b = self._setup(S, seed=5)
        b_i = rng.uniform(0.5, 1.5, S)
        b_q = rng.uniform(0.5, 1.5, S)
        b_u = rng.uniform(0.5, 1.5, S)
        bv = np.stack([b_i, b_q, b_u])

        t_out, p_out = self._run(mp, vec, bv, scan, chi_b, mode, stacked, z_skip)
        _, p_q = self._run(mp, vec, b_q, scan, None, mode, stacked, z_skip)
        _, p_u = self._run(mp, vec, b_u, scan, None, mode, stacked, z_skip)
        t_ref, _ = self._run(mp, vec, b_i, scan, None, mode, stacked, z_skip)

        rot = np.exp(-2j * chi_b)
        det = p_out * rot
        atol = self._rtol() * np.abs(p_q).max()
        npt.assert_allclose(det.real, (p_q * rot).real, rtol=self._rtol(), atol=atol)
        npt.assert_allclose(det.imag, (p_u * rot).imag, rtol=self._rtol(), atol=atol)
        npt.assert_array_equal(t_out, t_ref)

    @pytest.mark.parametrize("mode", ["bilinear", "nearest"])
    def test_pencil_beam_zero_u_weight(self, mode):
        """One node with U weight 0: the detector-frame U output vanishes at
        every chi, while Q keeps the full weight."""
        _, mp, vec, scan, chi_b = self._setup(1, seed=9)
        bv = np.array([[1.0], [1.0], [0.0]])
        _, p_out = self._run(mp, vec, bv, scan, chi_b, mode, True)
        _, p_full = self._run(mp, vec, np.ones(1), scan, None, mode, True)
        det = p_out * np.exp(-2j * chi_b)
        atol = 10 * self._rtol() * np.abs(p_full).max()
        npt.assert_allclose(det.imag, 0.0, atol=atol)
        npt.assert_allclose(det.real, (p_full * np.exp(-2j * chi_b)).real, atol=atol)

    def test_unequal_beams_require_chi(self):
        _, mp, vec, scan, _ = self._setup(4, seed=1)
        bv = np.stack([np.ones(4), np.ones(4), 0.5 * np.ones(4)])
        with pytest.raises(ValueError, match="chi_b"):
            self._run(mp, vec, bv, scan, None, "bilinear", True)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
