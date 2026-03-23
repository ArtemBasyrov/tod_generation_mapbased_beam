"""
Tests for precompute_beam_cache module.

Covers: _roll_vectors_jit, _compute_angular_offsets,
        cache_filename, save_cache, load_cache, lookup_psi_bin.

Can be run independently:
    pytest tests/test_precompute_beam_cache.py -v
    python tests/test_precompute_beam_cache.py
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import pytest

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

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from precompute_beam_cache import (
    _roll_vectors_jit,
    _compute_angular_offsets,
    cache_filename,
    save_cache,
    load_cache,
    lookup_psi_bin,
)


# ===========================================================================
# TestRollVectorsJit
# ===========================================================================

class TestRollVectorsJit:
    """Tests for precompute_beam_cache._roll_vectors_jit."""

    _BEAM_CTR = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def _run(self, vec_orig, psi_grid):
        out = np.empty((len(psi_grid), len(vec_orig), 3), dtype=np.float32)
        _roll_vectors_jit(vec_orig, self._BEAM_CTR, psi_grid, out)
        return out

    def test_output_shape(self):
        """Output has shape (N_psi, S, 3)."""
        S, N_psi = 5, 8
        rng = np.random.default_rng(0)
        vec_orig = rng.standard_normal((S, 3)).astype(np.float32)
        vec_orig /= np.linalg.norm(vec_orig, axis=-1, keepdims=True)
        psi_grid = np.linspace(0, 2 * np.pi, N_psi, endpoint=False, dtype=np.float32)
        out = self._run(vec_orig, psi_grid)
        assert out.shape == (N_psi, S, 3)

    def test_identity_at_psi_zero(self):
        """psi = 0 produces identity rotation: vec_rolled == vec_orig."""
        rng = np.random.default_rng(1)
        S   = 6
        vec_orig = rng.standard_normal((S, 3)).astype(np.float32)
        vec_orig /= np.linalg.norm(vec_orig, axis=-1, keepdims=True)
        psi_grid = np.array([0.0], dtype=np.float32)
        out = self._run(vec_orig, psi_grid)
        npt.assert_allclose(out[0], vec_orig, atol=1e-5)

    def test_rotation_pi_half_around_beam_ctr(self):
        """psi = pi/2 around [1,0,0]: [0,1,0] -> [0,0,1]."""
        vec_orig = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        psi_grid = np.array([np.pi / 2], dtype=np.float32)
        out = self._run(vec_orig, psi_grid)
        npt.assert_allclose(out[0, 0], [0.0, 0.0, 1.0], atol=1e-5)

    def test_rotation_pi_around_beam_ctr(self):
        """psi = pi around [1,0,0]: [0,1,0] -> [0,-1,0]."""
        vec_orig = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        psi_grid = np.array([np.pi], dtype=np.float32)
        out = self._run(vec_orig, psi_grid)
        npt.assert_allclose(out[0, 0], [0.0, -1.0, 0.0], atol=1e-5)

    def test_unit_vector_norms_preserved(self):
        """Rodrigues rotation preserves unit vector norms."""
        rng = np.random.default_rng(2)
        S, N_psi = 10, 12
        vec_orig = rng.standard_normal((S, 3)).astype(np.float32)
        vec_orig /= np.linalg.norm(vec_orig, axis=-1, keepdims=True)
        psi_grid = np.linspace(0, 2 * np.pi, N_psi, endpoint=False, dtype=np.float32)
        out = self._run(vec_orig, psi_grid)
        norms = np.linalg.norm(out.astype(np.float64), axis=-1)
        npt.assert_allclose(norms, np.ones((N_psi, S)), atol=1e-4)

    def test_full_cycle_returns_to_start(self):
        """Rotations at psi=0 and psi=2pi give the same result."""
        rng = np.random.default_rng(3)
        S   = 4
        vec_orig = rng.standard_normal((S, 3)).astype(np.float32)
        vec_orig /= np.linalg.norm(vec_orig, axis=-1, keepdims=True)
        psi_grid = np.array([0.0, 2 * np.pi], dtype=np.float32)
        out = self._run(vec_orig, psi_grid)
        npt.assert_allclose(out[0], out[1], atol=1e-5)

    def test_beam_ctr_is_fixed_point(self):
        """beam_ctr = [1,0,0] is a fixed point of the rotation (it is the rotation axis)."""
        beam_ctr_row = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # (1, 3)
        psi_grid = np.linspace(0, 2 * np.pi, 8, endpoint=False, dtype=np.float32)
        out = self._run(beam_ctr_row, psi_grid)
        for k in range(len(psi_grid)):
            npt.assert_allclose(out[k, 0], [1.0, 0.0, 0.0], atol=1e-5,
                                err_msg=f"beam_ctr not fixed at psi={psi_grid[k]:.3f}")


# ===========================================================================
# TestComputeAngularOffsets
# ===========================================================================

class TestComputeAngularOffsets:
    """Tests for precompute_beam_cache._compute_angular_offsets."""

    _BEAM_CTR = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def test_extracts_z_component_as_dtheta(self):
        """dtheta = vec_rolled[:, :, 2]."""
        rng = np.random.default_rng(10)
        N_psi, S = 4, 5
        vec_rolled = rng.standard_normal((N_psi, S, 3)).astype(np.float32)
        dtheta, _ = _compute_angular_offsets(vec_rolled, self._BEAM_CTR)
        npt.assert_array_equal(dtheta, vec_rolled[:, :, 2])

    def test_extracts_y_component_as_dphi(self):
        """dphi = vec_rolled[:, :, 1]."""
        rng = np.random.default_rng(11)
        N_psi, S = 4, 5
        vec_rolled = rng.standard_normal((N_psi, S, 3)).astype(np.float32)
        _, dphi = _compute_angular_offsets(vec_rolled, self._BEAM_CTR)
        npt.assert_array_equal(dphi, vec_rolled[:, :, 1])

    def test_output_dtype_is_float32(self):
        """Both output arrays have dtype float32."""
        vec_rolled = np.ones((3, 4, 3), dtype=np.float32)
        dtheta, dphi = _compute_angular_offsets(vec_rolled, self._BEAM_CTR)
        assert dtheta.dtype == np.float32
        assert dphi.dtype == np.float32

    def test_output_shape(self):
        """dtheta and dphi have shape (N_psi, S)."""
        N_psi, S = 6, 7
        vec_rolled = np.zeros((N_psi, S, 3), dtype=np.float32)
        dtheta, dphi = _compute_angular_offsets(vec_rolled, self._BEAM_CTR)
        assert dtheta.shape == (N_psi, S)
        assert dphi.shape   == (N_psi, S)

    def test_independent_of_x_component(self):
        """Modifying only the x component of vec_rolled does not affect outputs."""
        rng = np.random.default_rng(12)
        vec_rolled  = rng.standard_normal((4, 5, 3)).astype(np.float32)
        dtheta1, dphi1 = _compute_angular_offsets(vec_rolled, self._BEAM_CTR)
        vec_rolled2 = vec_rolled.copy()
        vec_rolled2[:, :, 0] *= 2.5
        dtheta2, dphi2 = _compute_angular_offsets(vec_rolled2, self._BEAM_CTR)
        npt.assert_array_equal(dtheta1, dtheta2)
        npt.assert_array_equal(dphi1,   dphi2)


# ===========================================================================
# TestCacheFilename
# ===========================================================================

class TestCacheFilename:
    """Tests for precompute_beam_cache.cache_filename."""

    def test_basic_filename(self):
        """Generates expected path from a simple beam filename."""
        result = cache_filename("beam_I.fits", "/cache", 720)
        assert result == "/cache/beam_I_cache_npsi720.npz"

    def test_strips_directory_from_bf(self):
        """Strips leading directory from bf; only the stem is used."""
        result = cache_filename("/some/dir/beam_Q.fits", "/out", 360)
        assert result == "/out/beam_Q_cache_npsi360.npz"

    def test_n_psi_embedded_in_filename(self):
        """n_psi value is embedded in the filename."""
        for n_psi in [1, 180, 720, 4096]:
            result = cache_filename("b.fits", "/d", n_psi)
            assert f"npsi{n_psi}" in os.path.basename(result)

    def test_output_ends_with_npz(self):
        """Output always ends with .npz regardless of input extension."""
        for ext in [".fits", ".npy", ".npz", ".dat", ""]:
            result = cache_filename(f"beam{ext}", "/d", 720)
            assert result.endswith(".npz")

    def test_output_dir_is_dirname(self):
        """output_dir is the directory component of the returned path."""
        result = cache_filename("b.fits", "/my/cache", 100)
        assert os.path.dirname(result) == "/my/cache"


# ===========================================================================
# TestSaveCacheLoadCache
# ===========================================================================

class TestSaveCacheLoadCache:
    """Tests for precompute_beam_cache.save_cache and load_cache."""

    @staticmethod
    def _make_cache(rng=None):
        if rng is None:
            rng = np.random.default_rng(20)
        return {
            "psi_grid":   np.linspace(0, 2 * np.pi, 8, dtype=np.float32),
            "vec_rolled": rng.standard_normal((8, 5, 3)).astype(np.float32),
            "beam_vals":  rng.uniform(0, 1, 5).astype(np.float32),
            "beam_ctr":   np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }

    def test_round_trip_values(self):
        """save_cache → load_cache recovers the same arrays."""
        cache = self._make_cache()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            save_cache(cache, path)
            loaded = load_cache(path)
            assert set(loaded.keys()) == set(cache.keys())
            for key in cache:
                npt.assert_array_equal(loaded[key], cache[key],
                                       err_msg=f"Mismatch for key '{key}'")
        finally:
            os.unlink(path)

    def test_round_trip_dtypes_preserved(self):
        """Dtypes are preserved through save/load."""
        cache = self._make_cache()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            save_cache(cache, path)
            loaded = load_cache(path)
            for key in cache:
                assert loaded[key].dtype == cache[key].dtype, \
                    f"dtype mismatch for '{key}': {loaded[key].dtype} vs {cache[key].dtype}"
        finally:
            os.unlink(path)

    def test_load_returns_dict(self):
        """load_cache returns a plain dict, not an NpzFile object."""
        cache = self._make_cache()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            save_cache(cache, path)
            loaded = load_cache(path)
            assert isinstance(loaded, dict)
        finally:
            os.unlink(path)

    def test_with_optional_offset_arrays(self):
        """Round-trip works when cache contains optional dtheta/dphi arrays."""
        rng = np.random.default_rng(21)
        cache = self._make_cache(rng)
        cache["dtheta"] = rng.standard_normal((8, 5)).astype(np.float32)
        cache["dphi"]   = rng.standard_normal((8, 5)).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            save_cache(cache, path)
            loaded = load_cache(path)
            for key in cache:
                npt.assert_array_equal(loaded[key], cache[key])
        finally:
            os.unlink(path)


# ===========================================================================
# TestLookupPsiBin
# ===========================================================================

class TestLookupPsiBin:
    """Tests for precompute_beam_cache.lookup_psi_bin."""

    @staticmethod
    def _make_grid(n_psi):
        return np.linspace(0, 2 * np.pi, n_psi, endpoint=False, dtype=np.float32)

    def test_exact_bin_centers_round_trip(self):
        """psi_grid[k] maps to bin index k for all k."""
        n_psi    = 12
        psi_grid = self._make_grid(n_psi)
        indices  = lookup_psi_bin(psi_grid.astype(np.float64), psi_grid)
        npt.assert_array_equal(indices, np.arange(n_psi))

    def test_wrapping_full_circle(self):
        """psi = 2π wraps to bin 0."""
        n_psi    = 8
        psi_grid = self._make_grid(n_psi)
        idx = lookup_psi_bin(np.array([2 * np.pi]), psi_grid)
        assert idx[0] == 0

    def test_wrapping_negative_psi(self):
        """psi = -dpsi wraps to bin n_psi - 1."""
        n_psi    = 8
        psi_grid = self._make_grid(n_psi)
        dpsi     = 2 * np.pi / n_psi
        idx = lookup_psi_bin(np.array([-dpsi]), psi_grid)
        assert idx[0] == n_psi - 1

    def test_output_in_valid_range(self):
        """All output indices are in [0, n_psi)."""
        rng        = np.random.default_rng(30)
        n_psi      = 720
        psi_grid   = self._make_grid(n_psi)
        psi_values = rng.uniform(-10 * np.pi, 10 * np.pi, 100)
        indices    = lookup_psi_bin(psi_values, psi_grid)
        assert np.all(indices >= 0)
        assert np.all(indices < n_psi)

    def test_output_shape_matches_input(self):
        """Output shape matches the input psi_values shape."""
        psi_grid   = self._make_grid(4)
        psi_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        indices    = lookup_psi_bin(psi_values, psi_grid)
        assert indices.shape == psi_values.shape

    def test_output_dtype_int64(self):
        """Output array has dtype int64."""
        psi_grid   = self._make_grid(4)
        psi_values = np.array([0.0, 1.0])
        indices    = lookup_psi_bin(psi_values, psi_grid)
        assert indices.dtype == np.int64

    def test_large_positive_psi_wraps(self):
        """psi = 4π (two full rotations) maps to bin 0."""
        n_psi    = 8
        psi_grid = self._make_grid(n_psi)
        idx = lookup_psi_bin(np.array([4 * np.pi]), psi_grid)
        assert idx[0] == 0

    @pytest.mark.parametrize("n_psi", [4, 12, 360, 720])
    def test_monotone_within_interior(self, n_psi):
        """Indices are non-decreasing for psi values that stay away from the 2π wrap point."""
        psi_grid = self._make_grid(n_psi)
        dpsi     = 2 * np.pi / n_psi
        # Sample psi values covering [0, 2π - 2*dpsi], well clear of the wrap boundary.
        psi_values = np.linspace(0, 2 * np.pi - 2 * dpsi, n_psi)
        indices    = lookup_psi_bin(psi_values, psi_grid)
        assert np.all(np.diff(indices) >= 0)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
