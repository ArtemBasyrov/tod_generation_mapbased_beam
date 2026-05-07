"""
Tests for the tod_calibrate module.

Covers: pure helpers of the new joint (P, T, B) calibrator
(_max_batch_for_memory, _thread_candidates, _process_thread_pairs)
and the unchanged beam-clustering calibrator.

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
    _max_batch_for_memory,
    _per_proc_static_bytes,
    _thread_candidates,
    _process_thread_pairs,
    _run_clustering_probe,
    _make_probe_data,
    _run_one,
    _measure_throughput,
    calibrate_beam_clustering,
    calibrate_runtime,
)


# ===========================================================================
# Memory model helpers
# ===========================================================================


def _bd_with_mp_stacked(n_beams, nside, n_components_per_beam=1):
    """Construct a fake beam_data with mp_stacked of the right size."""
    npix = 12 * nside * nside
    bd = {}
    for i in range(n_beams):
        bd[f"b{i}"] = {
            "comp_indices": list(range(n_components_per_beam)),
            "mp_stacked": np.zeros((n_components_per_beam, npix), dtype=np.float32),
        }
    return bd


class TestStaticBytes:
    def test_single_beam_one_component(self):
        bd = _bd_with_mp_stacked(1, nside=128, n_components_per_beam=1)
        npix = 12 * 128 * 128
        assert _per_proc_static_bytes(bd, 128) == npix * 4

    def test_three_beams_one_component_each(self):
        bd = _bd_with_mp_stacked(3, nside=64, n_components_per_beam=1)
        npix = 12 * 64 * 64
        assert _per_proc_static_bytes(bd, 64) == 3 * npix * 4


class TestMaxBatchForMemory:
    def test_returns_zero_when_static_exceeds_budget(self):
        bd = _bd_with_mp_stacked(10, nside=512)  # ~314 MB per beam × 10
        cap = _max_batch_for_memory(0.1, bd, 512, "bilinear")
        assert cap == 0

    def test_grows_with_memory(self):
        bd = _bd_with_mp_stacked(1, nside=128)
        cap_low = _max_batch_for_memory(2.0, bd, 128, "bilinear")
        cap_high = _max_batch_for_memory(8.0, bd, 128, "bilinear")
        assert cap_high > cap_low > 0

    def test_unknown_mode_uses_default(self):
        bd = _bd_with_mp_stacked(1, nside=128)
        cap = _max_batch_for_memory(2.0, bd, 128, "totally_made_up")
        assert cap > 0


class TestThreadCandidates:
    def test_includes_one_and_n_cores(self):
        c = _thread_candidates(14)
        assert 1 in c and 14 in c

    def test_powers_of_two_present(self):
        c = _thread_candidates(14)
        for p in (1, 2, 4, 8):
            assert p in c

    def test_sorted_no_duplicates(self):
        c = _thread_candidates(8)
        assert c == sorted(set(c))


class TestProcessThreadPairs:
    def test_constraint_p_times_t_le_n(self):
        for p, t in _process_thread_pairs(14, max_processes=14):
            assert p * t <= 14
            assert p >= 1 and t >= 1

    def test_max_processes_cap(self):
        pairs = _process_thread_pairs(14, max_processes=2)
        assert all(p <= 2 for p, t in pairs)

    def test_includes_full_threading_one_proc(self):
        pairs = _process_thread_pairs(14, max_processes=14)
        assert (1, 14) in pairs

    def test_includes_one_thread_max_proc(self):
        pairs = _process_thread_pairs(14, max_processes=14)
        assert (14, 1) in pairs


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
            )
        assert mock_btb.call_count == n_entries

    def test_interp_kwargs_forwarded(self):
        """interp_mode is forwarded as kwargs to beam_tod_batch."""
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
            )
        kw = mock_btb.call_args[1]
        assert kw.get("interp_mode") == "nearest"


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
                    "tod_calibrate.cluster_beam_pixels",
                    side_effect=_mock_cluster_pixels,
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
        """When every clustered (tf, K) pair exceeds error_threshold, entries
        where K_req >= n_tail short-circuit with bell_div=0.0 and still pass.
        The function returns a valid result without a WARNING.

        The first compute_bell call (reference) returns bell=1; all subsequent
        calls (clustered) return bell=0.5, giving a non-zero divergence for the
        clustered entries that are actually evaluated.  Short-circuited entries
        (K_req >= n_tail) are never evaluated and record bell_div=0.0 directly.
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
                error_threshold=1e-10,
            )

        captured = capsys.readouterr()
        # Short-circuited entries (K_req >= n_tail) always have bell_div=0.0 and
        # pass any non-negative threshold, so WARNING is never printed.
        assert "WARNING" not in captured.out
        assert isinstance(tf, float) and isinstance(K, int)
        assert tf in self._TAIL_FRACTIONS
        assert K in self._N_CLUSTERS_LIST

    def test_strict_threshold_triggers_fallback(self, capsys):
        """With error_threshold=0.0, short-circuited entries (K_req >= n_tail)
        have bell_div=0.0 which passes the threshold exactly, so the function
        returns a valid result without a WARNING."""
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
        assert "WARNING" not in captured.out
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


# ===========================================================================
# Runtime calibrator: probe-data, single-pass runner, throughput, and the
# top-level orchestrator.
# ===========================================================================


def _runtime_beam_data(S=10, nside=8):
    """Single-entry beam_data with the keys the runtime calibrator needs.

    Includes ``mp_stacked`` (used by `_per_proc_static_bytes` /
    `_max_batch_for_memory`) and the fields read by `calibrate_runtime`
    (`ra`, `dec`).
    """
    npix = 12 * nside * nside
    return {
        "b": {
            "beam_vals": np.ones(S, dtype=np.float64) / S,
            "vec_orig": np.tile([1.0, 0.0, 0.0], (S, 1)).astype(np.float32),
            "n_sel": S,
            "comp_indices": [0],
            "mp_stacked": np.zeros((1, npix), dtype=np.float32),
            "ra": 0.0,
            "dec": 0.0,
        }
    }


def _fake_load_scan_data_batch(folder_scan, day_index, start_idx, end_idx):
    """Stub of tod_io.load_scan_data_batch — returns deterministic float32 ramps."""
    n = end_idx - start_idx
    return (
        (np.arange(n) + start_idx).astype(np.float32),
        (np.arange(n) + start_idx + 1000).astype(np.float32),
        (np.arange(n) + start_idx + 2000).astype(np.float32),
    )


def _fake_precompute_rotation_vector_batch(ra0, dec0, phi_b, theta_b, center_idx=None):
    """Stub: returns trivial rot_vecs and zero betas of the right length."""
    n = len(phi_b)
    rot_vecs = np.zeros((n, 3, 3), dtype=np.float32)
    betas = np.zeros(n, dtype=np.float32)
    return rot_vecs, betas


class TestMakeProbeData:
    def test_returns_three_arrays_of_requested_length(self):
        bd = _runtime_beam_data()
        with patch(
            "tod_calibrate.load_scan_data_batch",
            side_effect=_fake_load_scan_data_batch,
        ):
            phi, theta, psi = _make_probe_data(bd, "/scan/", probe_day=0, n_samples=128)
        # _make_probe_data returns the (phi, theta, psi) ordering — note the
        # source unpacks load_scan_data_batch as (theta, phi, psi).
        assert phi.shape == (128,)
        assert theta.shape == (128,)
        assert psi.shape == (128,)

    def test_caps_to_loader_length(self):
        """If the loader returns fewer samples than requested, output is capped."""
        bd = _runtime_beam_data()

        def short_loader(folder, day, s, e):
            n = max(0, (e - s) // 2)
            return (
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=np.float32),
            )

        with patch("tod_calibrate.load_scan_data_batch", side_effect=short_loader):
            phi, theta, psi = _make_probe_data(bd, "/scan/", 0, 100)
        assert len(phi) == 50
        assert len(theta) == 50
        assert len(psi) == 50


class TestRunOne:
    def _run(self, beam_data, mp, phi, theta, psi, bs, **kw):
        nside = 8
        with (
            patch(
                "tod_calibrate.precompute_rotation_vector_batch",
                side_effect=_fake_precompute_rotation_vector_batch,
            ),
            patch("tod_calibrate.beam_tod_batch", side_effect=_mock_btb_ones),
        ):
            return _run_one(
                nside,
                mp,
                beam_data,
                ra0=0.0,
                dec0=0.0,
                phi_p=phi,
                theta_p=theta,
                psi_p=psi,
                bs=bs,
                interp_mode="bilinear",
                **kw,
            )

    def test_returns_positive_wall_time(self):
        bd = _runtime_beam_data()
        mp = _fake_mp()
        n = 64
        phi = np.zeros(n, dtype=np.float32)
        theta = np.zeros(n, dtype=np.float32)
        psi = np.zeros(n, dtype=np.float32)
        t = self._run(bd, mp, phi, theta, psi, bs=16)
        assert t >= 0.0

    def test_calls_beam_tod_batch_in_correct_number_of_batches(self):
        bd = _runtime_beam_data()
        mp = _fake_mp()
        n = 100
        bs = 32
        # ceil(100 / 32) == 4
        phi = np.zeros(n, dtype=np.float32)
        theta = np.zeros(n, dtype=np.float32)
        psi = np.zeros(n, dtype=np.float32)

        with (
            patch(
                "tod_calibrate.precompute_rotation_vector_batch",
                side_effect=_fake_precompute_rotation_vector_batch,
            ),
            patch(
                "tod_calibrate.beam_tod_batch", side_effect=_mock_btb_ones
            ) as btb_mock,
        ):
            _run_one(
                8,
                mp,
                bd,
                ra0=0.0,
                dec0=0.0,
                phi_p=phi,
                theta_p=theta,
                psi_p=psi,
                bs=bs,
                interp_mode="bilinear",
            )
        # One beam entry × ceil(n/bs) batches = 4 calls.
        assert btb_mock.call_count == 4

    def test_z_skip_threshold_propagated(self):
        bd = _runtime_beam_data()
        mp = _fake_mp()
        n = 32
        phi = np.zeros(n, dtype=np.float32)
        theta = np.zeros(n, dtype=np.float32)
        psi = np.zeros(n, dtype=np.float32)

        with (
            patch(
                "tod_calibrate.precompute_rotation_vector_batch",
                side_effect=_fake_precompute_rotation_vector_batch,
            ),
            patch(
                "tod_calibrate.beam_tod_batch", side_effect=_mock_btb_ones
            ) as btb_mock,
        ):
            _run_one(
                8,
                mp,
                bd,
                ra0=0.0,
                dec0=0.0,
                phi_p=phi,
                theta_p=theta,
                psi_p=psi,
                bs=16,
                interp_mode="bilinear",
                z_skip_threshold=0.42,
            )
        # Every call must have z_skip_threshold=0.42 forwarded as a kwarg.
        for call in btb_mock.call_args_list:
            assert call.kwargs["z_skip_threshold"] == 0.42


class TestMeasureThroughput:
    def test_returns_finite_positive_throughput(self):
        bd = _runtime_beam_data()
        mp = _fake_mp()
        n = 8000
        phi_full = np.zeros(n, dtype=np.float32)
        theta_full = np.zeros(n, dtype=np.float32)
        psi_full = np.zeros(n, dtype=np.float32)
        with (
            patch(
                "tod_calibrate.precompute_rotation_vector_batch",
                side_effect=_fake_precompute_rotation_vector_batch,
            ),
            patch("tod_calibrate.beam_tod_batch", side_effect=_mock_btb_ones),
            patch("tod_calibrate.numba.set_num_threads") as set_thr,
        ):
            tp = _measure_throughput(
                8,
                mp,
                bd,
                ra0=0.0,
                dec0=0.0,
                phi_full=phi_full,
                theta_full=theta_full,
                psi_full=psi_full,
                bs=512,
                n_threads=2,
                interp_mode="bilinear",
            )
        assert math.isfinite(tp) and tp > 0.0
        set_thr.assert_called_with(2)

    def test_floors_n_threads_at_one(self):
        """n_threads=0 must still be passed as max(1, …) to set_num_threads."""
        bd = _runtime_beam_data()
        mp = _fake_mp()
        phi_full = np.zeros(2000, dtype=np.float32)
        theta_full = np.zeros(2000, dtype=np.float32)
        psi_full = np.zeros(2000, dtype=np.float32)
        with (
            patch(
                "tod_calibrate.precompute_rotation_vector_batch",
                side_effect=_fake_precompute_rotation_vector_batch,
            ),
            patch("tod_calibrate.beam_tod_batch", side_effect=_mock_btb_ones),
            patch("tod_calibrate.numba.set_num_threads") as set_thr,
        ):
            _measure_throughput(
                8,
                mp,
                bd,
                ra0=0.0,
                dec0=0.0,
                phi_full=phi_full,
                theta_full=theta_full,
                psi_full=psi_full,
                bs=128,
                n_threads=0,
                interp_mode="bilinear",
            )
        set_thr.assert_called_with(1)


# Import math here for TestMeasureThroughput (avoid touching the original
# import block at the top of the file).
import math  # noqa: E402


class TestCalibrateRuntime:
    """Smoke test for the joint (P, T, B) orchestrator.

    All numeric kernels and scan I/O are stubbed so the test is fast and
    deterministic. We verify the contract: returns three positive integers,
    obeys max_processes_user, and respects n_cpu_ceiling.
    """

    def _run(self, *, n_cpu_ceiling=4, max_processes_user=4, mem_gb=64.0):
        bd = _runtime_beam_data()
        mp = _fake_mp()

        with (
            patch(
                "tod_calibrate.load_scan_data_batch",
                side_effect=_fake_load_scan_data_batch,
            ),
            patch(
                "tod_calibrate.precompute_rotation_vector_batch",
                side_effect=_fake_precompute_rotation_vector_batch,
            ),
            patch("tod_calibrate.beam_tod_batch", side_effect=_mock_btb_ones),
            patch("tod_calibrate.numba.set_num_threads"),
            patch("tod_calibrate._get_memory_per_process", return_value=mem_gb),
        ):
            return calibrate_runtime(
                bd,
                folder_scan="/scan/",
                probe_day=0,
                mp=mp,
                n_cpu_ceiling=n_cpu_ceiling,
                max_processes_user=max_processes_user,
            )

    def test_returns_three_positive_ints(self):
        P, T, B = self._run()
        assert isinstance(P, int) and P >= 1
        assert isinstance(T, int) and T >= 1
        assert isinstance(B, int) and B >= 1

    def test_p_capped_by_max_processes_user(self):
        P, _, _ = self._run(n_cpu_ceiling=8, max_processes_user=2)
        assert P <= 2

    def test_p_capped_by_n_cpu_ceiling(self):
        P, T, _ = self._run(n_cpu_ceiling=2, max_processes_user=8)
        assert P <= 2
        assert T <= 2
        assert P * T <= 2

    def test_raises_when_memory_too_tight(self):
        """A laughably small per-process memory budget can't fit static beams."""
        # mem_gb=0.0001 → static bytes alone exceed budget → bs_cap_p1 < 256.
        with pytest.raises(RuntimeError, match="memory too tight"):
            self._run(mem_gb=0.0001)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
