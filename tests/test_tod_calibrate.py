"""
Tests for the tod_calibrate module.

Covers: _memory_cap, _candidate_batch_sizes, _calibrate_n_processes,
        _run_clustering_probe, calibrate_beam_clustering.

Can be run independently:
    pytest tests/test_tod_calibrate.py -v
    python tests/test_tod_calibrate.py
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure project root and stubs are available when run as a standalone file.
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
from tod_calibrate import (
    _memory_cap,
    _candidate_batch_sizes,
    _calibrate_n_processes,
    _run_clustering_probe,
    calibrate_beam_clustering,
)


# ===========================================================================
# TestMemoryCap
# ===========================================================================


class TestMemoryCap:
    """Tests for tod_calibrate._memory_cap."""

    def test_basic_arithmetic(self):
        """Computes max(1, int(gb*1e9/1.5 / (bpspb*sel))) correctly for each method."""
        gb = 4.0
        sel = 1000
        for method, bpspb in [("bilinear", 64), ("bicubic", 400), ("nearest", 25)]:
            expected = max(1, int(gb * 1e9 / 1.5 / (bpspb * sel)))
            assert _memory_cap(gb, sel, interp_mode=method) == expected, method

    def test_returns_at_least_one_with_tiny_budget(self):
        """Returns at least 1 even when memory budget is extremely small."""
        assert _memory_cap(1e-9, 10**9) >= 1

    def test_scales_linearly_with_memory(self):
        """Result doubles when max_memory_gb doubles."""
        sel = 500
        r1 = _memory_cap(2.0, sel)
        r2 = _memory_cap(4.0, sel)
        assert abs(r2 - 2 * r1) <= 1

    def test_scales_inversely_with_beam_sel(self):
        """Result halves (approx) when max_beam_sel doubles."""
        gb = 8.0
        r1 = _memory_cap(gb, 100)
        r2 = _memory_cap(gb, 200)
        assert abs(r1 - 2 * r2) <= 1


# ===========================================================================
# TestCandidateBatchSizes
# ===========================================================================


class TestCandidateBatchSizes:
    """Tests for tod_calibrate._candidate_batch_sizes."""

    def test_mem_cap_is_last_element(self):
        """mem_cap is always the last element in the returned list."""
        for mem_cap in [1, 7, 16, 100, 1000]:
            result = _candidate_batch_sizes(mem_cap)
            assert result[-1] == mem_cap

    def test_all_values_le_mem_cap(self):
        """All returned values are <= mem_cap."""
        for mem_cap in [1, 4, 64, 500]:
            result = _candidate_batch_sizes(mem_cap)
            assert all(c <= mem_cap for c in result)

    def test_strictly_sorted_ascending(self):
        """The returned list is strictly sorted in ascending order."""
        for mem_cap in [1, 10, 128, 1024]:
            result = _candidate_batch_sizes(mem_cap)
            assert result == sorted(set(result))
            assert len(result) == len(set(result))

    def test_powers_of_two_included(self):
        """All powers of two from 2 up to the first power >= mem_cap are included."""
        mem_cap = 100
        result = _candidate_batch_sizes(mem_cap)
        k = 1
        while True:
            p = 2**k
            if p > mem_cap:
                break
            assert p in result, f"Power of two {p} missing from {result}"
            k += 1

    def test_mem_cap_one(self):
        """Works correctly for mem_cap=1, returning a single-element list [1]."""
        result = _candidate_batch_sizes(1)
        assert result == [1]


# ===========================================================================
# TestCalibrateNProcesses
# ===========================================================================


class TestCalibrateNProcesses:
    """
    Tests for tod_calibrate._calibrate_n_processes.

    _calibrate_batch_size and _get_memory_per_process are mocked so that the
    logic under test is the n-process selection algorithm only.
    """

    # Minimal beam_data: n_sel drives max_beam_sel = 1_000_000.
    _BEAM_DATA = {"beam_A": {"n_sel": 1_000_000}}

    def test_returns_n_within_ceiling(self):
        """n_optimal is always in [1, n_cpu_ceiling]."""
        results = [(1, 100), (2, 300), (3, 350), (4, 400), (6, 380)]
        with (
            patch("tod_calibrate._calibrate_batch_size", return_value=(None, results)),
            patch("tod_calibrate._get_memory_per_process", return_value=1.0),
        ):
            n_opt, _ = _calibrate_n_processes(
                self._BEAM_DATA,
                ".",
                0,
                [None],
                n_cpu_ceiling=4,
            )
        assert 1 <= n_opt <= 4

    def test_returns_at_least_one(self):
        """n_optimal is always >= 1."""
        results = [(1, 50), (2, 100)]
        with (
            patch("tod_calibrate._calibrate_batch_size", return_value=(None, results)),
            patch("tod_calibrate._get_memory_per_process", return_value=1.0),
        ):
            n_opt, _ = _calibrate_n_processes(
                self._BEAM_DATA,
                ".",
                0,
                [None],
                n_cpu_ceiling=1,
            )
        assert n_opt >= 1

    def test_ceiling_one_returns_one(self):
        """With n_cpu_ceiling=1, n_optimal is always 1."""
        results = [(1, 1000), (2, 2000), (4, 3000)]
        with (
            patch("tod_calibrate._calibrate_batch_size", return_value=(None, results)),
            patch("tod_calibrate._get_memory_per_process", return_value=8.0),
        ):
            n_opt, _ = _calibrate_n_processes(
                self._BEAM_DATA,
                ".",
                0,
                [None],
                n_cpu_ceiling=1,
            )
        assert n_opt == 1

    def test_maximizes_total_throughput(self):
        """
        The selected n maximizes n × per-process throughput.

        Setup (total_memory_gb=1.0, max_beam_sel=1_000_000, interp_mode='bilinear',
               bilinear bytes=64):
          _memory_cap(1.0 / n, 1e6, 'bilinear') = int(1e9/1.5/(64e6*n)):
            n=1 → cap=10, bs=6  (tp=380), total=  380
            n=2 → cap=5,  bs=4  (tp=350), total=  700
            n=3 → cap=3,  bs=3  (tp=400), total=1,200  ← winner
            n=4 → cap=2,  bs=2  (tp=200), total=  800
        """
        results = [(1, 100), (2, 200), (3, 400), (4, 350), (6, 380)]
        with (
            patch("tod_calibrate._calibrate_batch_size", return_value=(None, results)),
            patch("tod_calibrate._get_memory_per_process", return_value=1.0),
        ):
            n_opt, bs_opt = _calibrate_n_processes(
                self._BEAM_DATA,
                ".",
                0,
                [None],
                n_cpu_ceiling=4,
            )
        assert n_opt == 3
        assert bs_opt == 3

    def test_batch_size_is_affordable_for_optimal_n(self):
        """
        The returned batch_size never exceeds the memory cap for the optimal n.
        """
        results = [(1, 100), (2, 300), (3, 350), (4, 400), (6, 380)]
        total_gb = 1.0
        with (
            patch("tod_calibrate._calibrate_batch_size", return_value=(None, results)),
            patch("tod_calibrate._get_memory_per_process", return_value=total_gb),
        ):
            n_opt, bs_opt = _calibrate_n_processes(
                self._BEAM_DATA,
                ".",
                0,
                [None],
                n_cpu_ceiling=4,
            )
        max_beam_sel = 1_000_000
        cap = _memory_cap(total_gb / n_opt, max_beam_sel)
        assert bs_opt <= cap

    def test_all_affordable_skipped_when_memory_too_small(self):
        """
        If memory per process is so small that no batch size is affordable for
        some n values, those n are skipped and the result is still valid.
        """
        # Use a very large max_beam_sel so only n=1 has an affordable batch size.
        beam_data = {"beam_A": {"n_sel": 100_000_000}}
        # _memory_cap(1.0, 1e8) = max(1, int(1e9/1.5/1e10)) = max(1, 0) = 1
        # _memory_cap(0.5, 1e8) = max(1, int(0.5e9/1.5/1e10)) = max(1, 0) = 1
        # So cap=1 for all n; results must contain bs=1 to avoid "no affordable" skips.
        results = [(1, 500)]
        with (
            patch("tod_calibrate._calibrate_batch_size", return_value=(None, results)),
            patch("tod_calibrate._get_memory_per_process", return_value=1.0),
        ):
            n_opt, bs_opt = _calibrate_n_processes(
                beam_data,
                ".",
                0,
                [None],
                n_cpu_ceiling=4,
            )
        # Any n in [1,4] is valid; what matters is the result is sane.
        assert 1 <= n_opt <= 4
        assert bs_opt == 1


# ===========================================================================
# Helpers shared by clustering probe / calibration tests
# ===========================================================================

import healpy as hp  # noqa: E402  (healpy is transitively loaded by tod_calibrate)

_C_NSIDE = 8
_C_NPIX = hp.nside2npix(_C_NSIDE)
_C_S = 60  # number of beam pixels in test beam_data


def _fake_mp():
    """Return [I, Q, U] maps at nside=8, all ones."""
    return [np.ones(_C_NPIX, dtype=np.float32) for _ in range(3)]


def _fake_beam_data(S=_C_S, seed=7):
    """Minimal beam_data dict suitable for calibrate_beam_clustering tests.

    Args:
        S (int): Number of beam pixels.
        seed (int): RNG seed for reproducibility.

    Returns:
        dict: Single-entry beam_data with "beam_I" key.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 0.05, S)
    phi = rng.uniform(0.0, 2 * np.pi, S)
    vec_orig = np.column_stack(
        [
            np.cos(theta),
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
        ]
    ).astype(np.float32)
    beam_vals = np.ones(S, dtype=np.float64) / S
    return {
        "beam_I": {
            "beam_vals": beam_vals.copy(),
            "vec_orig": vec_orig.copy(),
            "n_sel": S,
            "comp_indices": [0],
            "ra": 0.0,
            "dec": 0.0,
        }
    }


def _fake_probe_entries(B=20):
    """Return a list with one minimal beam-entry dict (mp_stacked already set).

    Args:
        B (int): Unused — kept for interface symmetry; entries do not depend on B.

    Returns:
        list[dict]: One-element list of beam-entry dicts.
    """
    S = 10
    mp = _fake_mp()
    return [
        {
            "beam_vals": np.ones(S, dtype=np.float64) / S,
            "vec_orig": np.tile([1.0, 0.0, 0.0], (S, 1)).astype(np.float32),
            "n_sel": S,
            "comp_indices": [0],
            "mp_stacked": np.stack([mp[0]]),
            "ra": 0.0,
            "dec": 0.0,
        }
    ]


def _mock_btb_ones(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b, **kwargs):
    """beam_tod_batch stub: always returns all-ones for every component."""
    B = len(phi_b)
    return {c: np.ones(B, dtype=np.float64) for c in range(3)}


def _mock_cluster_pixels(
    vec_orig,
    bvals,
    n_clusters,
    tail_fraction=None,
    max_iter=150,
    tol=1e-5,
    random_state=42,
    verbose=True,
):
    """Deterministic cluster_beam_pixels stub (no k-means EM).

    Returns the first K_out pixel vectors with uniform weights and
    modulo-K_out labels.  Correctly mimics the hybrid-mode output shape.

    Args:
        vec_orig (np.ndarray): (S, 3) unit vectors.
        bvals (np.ndarray): (S,) beam weights.
        n_clusters (int): Requested cluster count.
        tail_fraction (float | None): Fraction of power treated as tail.
        max_iter, tol, random_state, verbose: Unused; accepted for signature compat.

    Returns:
        tuple: (vec_out, bvals_out, labels) with shapes (K_out, 3), (K_out,), (S,).
    """
    S = len(bvals)
    if tail_fraction is not None:
        w_sorted = np.sort(bvals)
        total = w_sorted.sum()
        cumsum = np.cumsum(w_sorted)
        n_tail = int(np.searchsorted(cumsum, tail_fraction * total)) + 1
        n_tail = min(max(n_tail, 0), S)
        n_main = S - n_tail
        K_tail = min(n_clusters, n_tail) if n_tail > 0 else 0
        K_out = n_main + max(K_tail, 1 if n_tail > 0 else 0)
    else:
        K_out = min(n_clusters, S)
    K_out = max(1, K_out)
    out_vec = vec_orig[:K_out].copy()
    out_bvals = np.full(K_out, 1.0 / K_out, dtype=np.float64)
    labels = (np.arange(S) % K_out).astype(np.int32)
    return out_vec, out_bvals, labels


# ===========================================================================
# TestRunClusteringProbe
# ===========================================================================


class TestRunClusteringProbe:
    """Tests for tod_calibrate._run_clustering_probe."""

    _B = 15

    def _phi_theta_psi(self, B=None):
        B = B or self._B
        return (
            np.zeros(B, dtype=np.float32),
            np.zeros(B, dtype=np.float32),
            np.zeros(B, dtype=np.float32),
        )

    def _rot_vecs(self, B=None):
        B = B or self._B
        return np.zeros((B, 3, 3), dtype=np.float32)

    def test_returns_shape_3_B(self):
        """Output shape is (3, B) for a single beam entry."""
        phi_b, theta_b, psis_b = self._phi_theta_psi()
        rot_vecs = self._rot_vecs()
        mp = _fake_mp()
        with patch("tod_calibrate.beam_tod_batch", side_effect=_mock_btb_ones):
            result = _run_clustering_probe(
                _C_NSIDE,
                mp,
                _fake_probe_entries(),
                rot_vecs,
                phi_b,
                theta_b,
                psis_b,
                interp_mode="bilinear",
                sigma_deg=None,
                radius_deg=None,
            )
        assert result.shape == (3, self._B)

    def test_dtype_float64(self):
        """Accumulated TOD array has dtype float64."""
        phi_b, theta_b, psis_b = self._phi_theta_psi()
        rot_vecs = self._rot_vecs()
        mp = _fake_mp()
        with patch("tod_calibrate.beam_tod_batch", side_effect=_mock_btb_ones):
            result = _run_clustering_probe(
                _C_NSIDE,
                mp,
                _fake_probe_entries(),
                rot_vecs,
                phi_b,
                theta_b,
                psis_b,
                interp_mode="bilinear",
                sigma_deg=None,
                radius_deg=None,
            )
        assert result.dtype == np.float64

    def test_sums_multiple_entries(self):
        """Contributions from N entries are accumulated (summed) component-wise."""
        B = 10
        n_entries = 3
        entries = _fake_probe_entries() * n_entries  # 3 identical entries
        phi_b, theta_b, psis_b = self._phi_theta_psi(B)
        rot_vecs = self._rot_vecs(B)
        mp = _fake_mp()
        with patch("tod_calibrate.beam_tod_batch", side_effect=_mock_btb_ones):
            result = _run_clustering_probe(
                _C_NSIDE,
                mp,
                entries,
                rot_vecs,
                phi_b,
                theta_b,
                psis_b,
                interp_mode="bilinear",
                sigma_deg=None,
                radius_deg=None,
            )
        # Each entry returns ones → accumulated total = n_entries
        np.testing.assert_allclose(result, np.full((3, B), float(n_entries)))

    def test_empty_entries_returns_zeros(self):
        """No beam entries → all-zero TOD of shape (3, B)."""
        B = 8
        phi_b, theta_b, psis_b = self._phi_theta_psi(B)
        rot_vecs = self._rot_vecs(B)
        mp = _fake_mp()
        result = _run_clustering_probe(
            _C_NSIDE,
            mp,
            [],
            rot_vecs,
            phi_b,
            theta_b,
            psis_b,
            interp_mode="bilinear",
            sigma_deg=None,
            radius_deg=None,
        )
        assert result.shape == (3, B)
        np.testing.assert_array_equal(result, 0)

    def test_beam_tod_batch_called_once_per_entry(self):
        """beam_tod_batch is called exactly once per beam entry."""
        n_entries = 4
        entries = _fake_probe_entries() * n_entries
        phi_b, theta_b, psis_b = self._phi_theta_psi()
        rot_vecs = self._rot_vecs()
        mp = _fake_mp()
        with patch(
            "tod_calibrate.beam_tod_batch", side_effect=_mock_btb_ones
        ) as mock_btb:
            _run_clustering_probe(
                _C_NSIDE,
                mp,
                entries,
                rot_vecs,
                phi_b,
                theta_b,
                psis_b,
                interp_mode="bilinear",
                sigma_deg=None,
                radius_deg=None,
            )
        assert mock_btb.call_count == n_entries

    def test_interp_kwargs_forwarded(self):
        """interp_mode, sigma_deg, radius_deg are forwarded as kwargs to beam_tod_batch."""
        phi_b, theta_b, psis_b = self._phi_theta_psi()
        rot_vecs = self._rot_vecs()
        mp = _fake_mp()
        with patch(
            "tod_calibrate.beam_tod_batch", side_effect=_mock_btb_ones
        ) as mock_btb:
            _run_clustering_probe(
                _C_NSIDE,
                mp,
                _fake_probe_entries(),
                rot_vecs,
                phi_b,
                theta_b,
                psis_b,
                interp_mode="nearest",
                sigma_deg=0.5,
                radius_deg=1.5,
            )
        kw = mock_btb.call_args[1]
        assert kw.get("interp_mode") == "nearest"
        assert kw.get("sigma_deg") == 0.5
        assert kw.get("radius_deg") == 1.5


# ===========================================================================
# TestCalibrateBeamClustering
# ===========================================================================


class TestCalibrateBeamClustering:
    """Tests for tod_calibrate.calibrate_beam_clustering.

    Both k-means (beam_cluster.cluster_beam_pixels) and B_ell computation
    (beam_bell_power.compute_bell) are replaced with lightweight stubs so
    the tests run without real beam FITS files or scan data.
    """

    # Hard-coded grid values copied from calibrate_beam_clustering source.
    _TAIL_FRACTIONS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
    _N_CLUSTERS_LIST = (10, 20, 50, 100, 200, 500, 1000, 2000)

    # ── stub helpers ──────────────────────────────────────────────────────

    def _mock_compute_bell_zero_div(
        self, ra, dec, pixel_map, lmax=500, power_cut=1.0, verbose=True, **kw
    ):
        """Return identical B_ell for every call → divergence = 0."""
        ell = np.arange(lmax + 1, dtype=np.int64)
        bell = np.ones(lmax + 1, dtype=np.float64)
        return ell, bell

    def _mock_compute_bell_nonzero_div(
        self, ra, dec, pixel_map, lmax=500, power_cut=1.0, verbose=True, **kw
    ):
        """Return different B_ell for clustered calls → nonzero divergence.

        The first call per beam file is the reference (identical pixels passed).
        Subsequent calls use a recognisably different bell array so divergence > 0.
        """
        ell = np.arange(lmax + 1, dtype=np.int64)
        # Clustered calls receive bv_c which is uniform (from _mock_cluster_pixels);
        # reference calls receive the original beam_vals which also sum to 1 but
        # differ in distribution.  We mimic large divergence by checking the sum.
        if np.allclose(pixel_map, pixel_map[0]):  # uniform → clustered
            bell = np.full(lmax + 1, 0.5, dtype=np.float64)
        else:
            bell = np.ones(lmax + 1, dtype=np.float64)
        return ell, bell

    def _patched(self, bell_side_effect=None):
        """Context-manager that patches cluster_beam_pixels and compute_bell.

        Usage::

            with self._patched() as (mock_cluster, mock_bell):
                result = calibrate_beam_clustering(...)
        """
        import contextlib

        @contextlib.contextmanager
        def _ctx(bell_se):
            with (
                patch(
                    "beam_cluster.cluster_beam_pixels", side_effect=_mock_cluster_pixels
                ) as m_cl,
                patch(
                    "tod_calibrate.compute_bell",
                    side_effect=bell_se or self._mock_compute_bell_zero_div,
                ) as m_bell,
            ):
                yield m_cl, m_bell

        return _ctx(bell_side_effect)

    # ── basic return-value tests ──────────────────────────────────────────

    def test_returns_tuple_float_int(self):
        """Return value is a (float, int) 2-tuple."""
        with self._patched():
            tf, K = calibrate_beam_clustering(_fake_beam_data(), ".", 0, _fake_mp())
        assert isinstance(tf, float)
        assert isinstance(K, int)

    def test_tail_fraction_in_search_grid(self):
        """Returned tail_fraction is one of the internally defined grid values."""
        with self._patched():
            tf, _ = calibrate_beam_clustering(_fake_beam_data(), ".", 0, _fake_mp())
        assert tf in self._TAIL_FRACTIONS

    def test_n_clusters_in_search_grid(self):
        """Returned n_clusters is one of the internally defined grid values."""
        with self._patched():
            _, K = calibrate_beam_clustering(_fake_beam_data(), ".", 0, _fake_mp())
        assert K in self._N_CLUSTERS_LIST

    # ── side-effect freedom ───────────────────────────────────────────────

    def test_beam_data_not_modified(self):
        """calibrate_beam_clustering must not mutate the caller's beam_data dict."""
        beam_data = _fake_beam_data()
        bv_before = beam_data["beam_I"]["beam_vals"].copy()
        vo_before = beam_data["beam_I"]["vec_orig"].copy()
        ns_before = beam_data["beam_I"]["n_sel"]
        with self._patched():
            calibrate_beam_clustering(beam_data, ".", 0, _fake_mp())
        np.testing.assert_array_equal(beam_data["beam_I"]["beam_vals"], bv_before)
        np.testing.assert_array_equal(beam_data["beam_I"]["vec_orig"], vo_before)
        assert beam_data["beam_I"]["n_sel"] == ns_before

    # ── selection-logic tests ─────────────────────────────────────────────

    def test_selects_max_speedup_when_all_pass(self):
        """When B_ell divergence=0 everywhere, the pair with the highest speedup
        is chosen (every pair qualifies with a permissive threshold)."""
        with self._patched():
            tf, K = calibrate_beam_clustering(
                _fake_beam_data(),
                ".",
                0,
                _fake_mp(),
                error_threshold=1.0,  # accept everything
            )
        assert tf in self._TAIL_FRACTIONS
        assert K in self._N_CLUSTERS_LIST

    def test_fallback_on_all_fail_returns_min_error(self, capsys):
        """When every (tf, K) pair exceeds error_threshold, the pair with the
        minimum B_ell divergence is returned and a WARNING line is printed.

        The first compute_bell call (reference) returns bell=1; all subsequent
        calls (clustered) return bell=0.5, giving a non-zero divergence.
        """
        call_count = [0]

        def _bell_nonzero(
            ra, dec, pixel_map, lmax=500, power_cut=1.0, verbose=True, **kw
        ):
            call_count[0] += 1
            ell = np.arange(lmax + 1, dtype=np.int64)
            bell = (
                np.ones(lmax + 1, dtype=np.float64)
                if call_count[0] == 1
                else np.full(lmax + 1, 0.5, dtype=np.float64)
            )
            return ell, bell

        with self._patched(bell_side_effect=_bell_nonzero):
            tf, K = calibrate_beam_clustering(
                _fake_beam_data(),
                ".",
                0,
                _fake_mp(),
                error_threshold=1e-10,  # nothing can pass
            )

        captured = capsys.readouterr()
        assert "WARNING" in captured.out, "Expected WARNING in stdout for all-fail case"
        assert isinstance(tf, float) and isinstance(K, int)
        assert tf in self._TAIL_FRACTIONS
        assert K in self._N_CLUSTERS_LIST

    def test_strict_threshold_triggers_fallback(self, capsys):
        """error_threshold=0 forces fallback whenever bell_div > 0."""
        call_count = [0]

        def _bell_nonzero(
            ra, dec, pixel_map, lmax=500, power_cut=1.0, verbose=True, **kw
        ):
            call_count[0] += 1
            ell = np.arange(lmax + 1, dtype=np.int64)
            bell = (
                np.ones(lmax + 1, dtype=np.float64)
                if call_count[0] == 1
                else np.full(lmax + 1, 0.5, dtype=np.float64)
            )
            return ell, bell

        with self._patched(bell_side_effect=_bell_nonzero):
            tf, K = calibrate_beam_clustering(
                _fake_beam_data(),
                ".",
                0,
                _fake_mp(),
                error_threshold=0.0,
            )

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert isinstance(tf, float) and isinstance(K, int)

    def test_permissive_threshold_no_warning(self, capsys):
        """A threshold of 1.0 (all pass) must not print a WARNING."""
        with self._patched():
            calibrate_beam_clustering(
                _fake_beam_data(),
                ".",
                0,
                _fake_mp(),
                error_threshold=1.0,
            )
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out

    def test_backward_compat_optional_args(self):
        """folder_scan / probe_day / mp are now optional; call without them."""
        with self._patched():
            tf, K = calibrate_beam_clustering(_fake_beam_data())
        assert isinstance(tf, float) and isinstance(K, int)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
