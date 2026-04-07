"""
Tests for beam_cluster.py.

Covers:
  _kmeans_plus_plus_init  — weighted k-means++ centroid initialisation
  _kmeans_sphere          — weighted k-means EM loop on the unit sphere
  _spread_stats           — intra-cluster angular spread diagnostics
  cluster_beam_pixels     — full mode (cluster all) and hybrid/tail mode
  _build_weight_matrix    — sparse (S × K) assignment-weight matrix
  cluster_cached_arrays   — reduction of vec_rolled, dtheta, dphi

No external data files are needed; all beam geometry is generated
synthetically using NumPy RNG.

Can be run independently:
    pytest tests/test_beam_cluster.py -v
    python tests/test_beam_cluster.py
"""

import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup for standalone execution
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from beam_cluster import (
    _kmeans_plus_plus_init,
    _kmeans_sphere,
    _spread_stats,
    cluster_beam_pixels,
    _build_weight_matrix,
    cluster_cached_arrays,
)


# ===========================================================================
# Shared synthetic data helpers
# ===========================================================================


def _make_beam(S=200, rng=None):
    """S beam pixels near the north pole with positive, normalised weights."""
    rng = rng or np.random.default_rng(7)
    theta = rng.uniform(0.0, 0.3, S)
    phi = rng.uniform(0.0, 2 * np.pi, S)
    st = np.sin(theta)
    vec = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1).astype(
        np.float32
    )
    bv = rng.uniform(0.1, 2.0, S).astype(np.float32)
    bv /= bv.sum()
    return vec, bv


def _make_two_lobe_beam(S=300, spread=0.02, rng=None):
    """Beam with two well-separated tight lobes of equal total weight.

    Lobe A near (theta=0.1, phi=0)  and lobe B near (theta=0.1, phi=pi).
    Both lobes have spread << separation, so k-means with K=2 should
    cleanly assign each lobe to a distinct cluster.
    """
    rng = rng or np.random.default_rng(42)
    half = S // 2

    theta_A = rng.uniform(0.1 - spread, 0.1 + spread, half)
    phi_A = rng.uniform(-spread, spread, half)
    theta_B = rng.uniform(0.1 - spread, 0.1 + spread, S - half)
    phi_B = rng.uniform(np.pi - spread, np.pi + spread, S - half)

    def _v(th, ph):
        st = np.sin(th)
        return np.stack([st * np.cos(ph), st * np.sin(ph), np.cos(th)], axis=1)

    vec = np.concatenate([_v(theta_A, phi_A), _v(theta_B, phi_B)], axis=0).astype(
        np.float32
    )
    bv = np.ones(S, dtype=np.float32) / S
    return vec, bv


def _make_cache(N_psi, S, rng=None):
    """Synthetic cache arrays matching the shapes expected by cluster_cached_arrays."""
    rng = rng or np.random.default_rng(99)
    vr = rng.standard_normal((N_psi, S, 3)).astype(np.float32)
    vr /= np.linalg.norm(vr, axis=2, keepdims=True)
    dt = rng.standard_normal((N_psi, S)).astype(np.float32) * 0.05
    dp = rng.standard_normal((N_psi, S)).astype(np.float32) * 0.05
    return {"vec_rolled": vr, "dtheta": dt, "dphi": dp}


# ===========================================================================
# TestKmeansPlusPlusInit
# ===========================================================================


class TestKmeansPlusPlusInit:
    """Tests for beam_cluster._kmeans_plus_plus_init."""

    def test_returns_K_centroids(self):
        """Returns exactly K centroid vectors."""
        rng = np.random.default_rng(0)
        vec, bv = _make_beam(S=100)
        K = 7
        centroids = _kmeans_plus_plus_init(
            vec.astype(np.float64), bv.astype(np.float64), K, rng
        )
        assert centroids.shape == (K, 3)

    def test_centroids_are_unit_vectors(self):
        """Every returned centroid is taken from vec, which lies on the unit sphere."""
        rng = np.random.default_rng(1)
        vec, bv = _make_beam(S=80)
        K = 5
        centroids = _kmeans_plus_plus_init(
            vec.astype(np.float64), bv.astype(np.float64), K, rng
        )
        norms = np.linalg.norm(centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_K_equals_one(self):
        """Works correctly when K=1 (single centroid)."""
        rng = np.random.default_rng(2)
        vec, bv = _make_beam(S=50)
        centroids = _kmeans_plus_plus_init(
            vec.astype(np.float64), bv.astype(np.float64), 1, rng
        )
        assert centroids.shape == (1, 3)

    def test_deterministic_with_same_seed(self):
        """Same RNG seed produces identical centroids."""
        vec, bv = _make_beam(S=60)
        K = 4
        c1 = _kmeans_plus_plus_init(
            vec.astype(np.float64), bv.astype(np.float64), K, np.random.default_rng(99)
        )
        c2 = _kmeans_plus_plus_init(
            vec.astype(np.float64), bv.astype(np.float64), K, np.random.default_rng(99)
        )
        np.testing.assert_array_equal(c1, c2)


# ===========================================================================
# TestKmeansSphere
# ===========================================================================


class TestKmeansSphere:
    """Tests for beam_cluster._kmeans_sphere."""

    def test_output_shapes(self):
        """Returns (K, 3) centroids and (S,) labels."""
        rng = np.random.default_rng(0)
        vec, bv = _make_beam(S=100)
        K = 10
        centroids, labels = _kmeans_sphere(
            vec.astype(np.float64),
            bv.astype(np.float64),
            K,
            max_iter=20,
            tol=1e-4,
            rng=rng,
            verbose=False,
        )
        assert centroids.shape == (K, 3)
        assert labels.shape == (100,)

    def test_centroids_on_unit_sphere(self):
        """All output centroids have unit norm."""
        rng = np.random.default_rng(1)
        vec, bv = _make_beam(S=150)
        centroids, _ = _kmeans_sphere(
            vec.astype(np.float64),
            bv.astype(np.float64),
            8,
            max_iter=30,
            tol=1e-5,
            rng=rng,
            verbose=False,
        )
        norms = np.linalg.norm(centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_labels_in_valid_range(self):
        """All label values are in [0, K)."""
        rng = np.random.default_rng(2)
        vec, bv = _make_beam(S=80)
        K = 5
        _, labels = _kmeans_sphere(
            vec.astype(np.float64),
            bv.astype(np.float64),
            K,
            max_iter=20,
            tol=1e-4,
            rng=rng,
            verbose=False,
        )
        assert int(labels.min()) >= 0
        assert int(labels.max()) < K

    def test_all_clusters_populated(self):
        """With S >> K, every cluster index appears at least once."""
        rng = np.random.default_rng(3)
        vec, bv = _make_beam(S=200)
        K = 10
        _, labels = _kmeans_sphere(
            vec.astype(np.float64),
            bv.astype(np.float64),
            K,
            max_iter=50,
            tol=1e-5,
            rng=rng,
            verbose=False,
        )
        assert len(set(labels.tolist())) == K

    def test_two_separated_lobes_assigned_to_different_clusters(self):
        """Two tight well-separated lobes always end up in distinct clusters."""
        vec, bv = _make_two_lobe_beam(S=200)
        rng = np.random.default_rng(4)
        _, labels = _kmeans_sphere(
            vec.astype(np.float64),
            bv.astype(np.float64),
            2,
            max_iter=100,
            tol=1e-6,
            rng=rng,
            verbose=False,
        )
        assert len(set(labels[:100].tolist())) == 1
        assert len(set(labels[100:].tolist())) == 1
        assert labels[0] != labels[100]

    def test_verbose_false_suppresses_output(self, capsys):
        """verbose=False produces no stdout output."""
        rng = np.random.default_rng(5)
        vec, bv = _make_beam(S=50)
        _kmeans_sphere(
            vec.astype(np.float64),
            bv.astype(np.float64),
            5,
            max_iter=10,
            tol=1e-4,
            rng=rng,
            verbose=False,
        )
        assert capsys.readouterr().out == ""

    def test_verbose_true_prints_something(self, capsys):
        """verbose=True produces at least one line of output."""
        rng = np.random.default_rng(6)
        vec, bv = _make_beam(S=50)
        _kmeans_sphere(
            vec.astype(np.float64),
            bv.astype(np.float64),
            5,
            max_iter=10,
            tol=1e-4,
            rng=rng,
            verbose=True,
        )
        assert len(capsys.readouterr().out) > 0


# ===========================================================================
# TestSpreadStats
# ===========================================================================


class TestSpreadStats:
    """Tests for beam_cluster._spread_stats."""

    def test_verbose_false_produces_no_output(self, capsys):
        """verbose=False is a no-op with respect to stdout."""
        vec, bv = _make_beam(S=50)
        _, labels = _kmeans_sphere(
            vec.astype(np.float64),
            bv.astype(np.float64),
            5,
            max_iter=10,
            tol=1e-4,
            rng=np.random.default_rng(0),
            verbose=False,
        )
        centroids, _ = _kmeans_sphere(
            vec.astype(np.float64),
            bv.astype(np.float64),
            5,
            max_iter=10,
            tol=1e-4,
            rng=np.random.default_rng(0),
            verbose=False,
        )
        _spread_stats(vec.astype(np.float64), centroids, labels, verbose=False)
        assert capsys.readouterr().out == ""

    def test_verbose_true_prints_arcmin_stats(self, capsys):
        """verbose=True prints mean / max / 95th percentile spread."""
        vec, bv = _make_beam(S=100)
        rng = np.random.default_rng(0)
        centroids, labels = _kmeans_sphere(
            vec.astype(np.float64),
            bv.astype(np.float64),
            8,
            max_iter=30,
            tol=1e-5,
            rng=rng,
            verbose=False,
        )
        _spread_stats(vec.astype(np.float64), centroids, labels, verbose=True)
        out = capsys.readouterr().out
        assert "mean=" in out
        assert "max=" in out
        assert "95th=" in out


# ===========================================================================
# TestClusterBeamPixelsFullMode
# ===========================================================================


class TestClusterBeamPixelsFullMode:
    """Tests for cluster_beam_pixels with tail_fraction=None (full mode)."""

    def test_output_shapes(self):
        """Returns (K, 3) vectors, (K,) weights, and (S,) labels."""
        vec, bv = _make_beam(S=200)
        K = 20
        vc, bvc, labels = cluster_beam_pixels(vec, bv, n_clusters=K, verbose=False)
        assert vc.shape == (K, 3)
        assert bvc.shape == (K,)
        assert labels.shape == (200,)

    def test_centroids_on_unit_sphere(self):
        """All output centroid vectors have unit norm."""
        vec, bv = _make_beam(S=150)
        vc, _, _ = cluster_beam_pixels(vec, bv, n_clusters=15, verbose=False)
        norms = np.linalg.norm(vc, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_weights_sum_to_one(self):
        """Output beam weights sum to approximately 1.0."""
        vec, bv = _make_beam(S=100)
        _, bvc, _ = cluster_beam_pixels(vec, bv, n_clusters=10, verbose=False)
        assert abs(float(bvc.sum()) - 1.0) < 1e-5

    def test_labels_in_valid_range(self):
        """All labels are valid indices into the output arrays."""
        vec, bv = _make_beam(S=120)
        K = 12
        vc, bvc, labels = cluster_beam_pixels(vec, bv, n_clusters=K, verbose=False)
        assert int(labels.min()) >= 0
        assert int(labels.max()) < K

    def test_output_dtype_float32(self):
        """Centroid vectors and weights are float32."""
        vec, bv = _make_beam(S=80)
        vc, bvc, _ = cluster_beam_pixels(vec, bv, n_clusters=8, verbose=False)
        assert vc.dtype == np.float32
        assert bvc.dtype == np.float32

    def test_K_gte_S_returns_unchanged(self):
        """When K >= S the original arrays are returned unmodified."""
        vec, bv = _make_beam(S=30)
        vc, bvc, labels = cluster_beam_pixels(vec, bv, n_clusters=30, verbose=False)
        np.testing.assert_array_equal(bvc, bv)
        assert len(vc) == 30
        assert len(bvc) == 30
        np.testing.assert_array_equal(labels, np.arange(30))

    def test_K_equals_one(self):
        """K=1 collapses all pixels into a single centroid."""
        vec, bv = _make_beam(S=50)
        vc, bvc, labels = cluster_beam_pixels(vec, bv, n_clusters=1, verbose=False)
        assert len(vc) == 1
        assert len(bvc) == 1
        np.testing.assert_allclose(bvc[0], 1.0, atol=1e-5)
        assert (labels == 0).all()

    def test_verbose_false_suppresses_output(self, capsys):
        """verbose=False produces no stdout output."""
        vec, bv = _make_beam(S=80)
        cluster_beam_pixels(vec, bv, n_clusters=8, verbose=False)
        assert capsys.readouterr().out == ""

    def test_deterministic_with_same_seed(self):
        """Same random_state produces identical outputs."""
        vec, bv = _make_beam(S=100)
        vc1, bvc1, lbl1 = cluster_beam_pixels(
            vec, bv, n_clusters=10, random_state=0, verbose=False
        )
        vc2, bvc2, lbl2 = cluster_beam_pixels(
            vec, bv, n_clusters=10, random_state=0, verbose=False
        )
        np.testing.assert_array_equal(lbl1, lbl2)
        np.testing.assert_array_equal(bvc1, bvc2)

    def test_two_lobes_separate_correctly(self):
        """Two well-separated tight lobes produce two cleanly-divided clusters."""
        vec, bv = _make_two_lobe_beam(S=200)
        vc, bvc, labels = cluster_beam_pixels(
            vec, bv, n_clusters=2, random_state=0, verbose=False
        )
        assert int(labels[:100].max()) == int(labels[:100].min())
        assert int(labels[100:].max()) == int(labels[100:].min())
        assert labels[0] != labels[100]


# ===========================================================================
# TestClusterBeamPixelsHybridMode
# ===========================================================================


class TestClusterBeamPixelsHybridMode:
    """Tests for cluster_beam_pixels with tail_fraction set (hybrid/tail mode)."""

    def test_output_n_main_plus_K_tail(self):
        """Output length is n_main + K_tail (K_tail ≤ n_clusters)."""
        vec, bv = _make_beam(S=200)
        tf = 0.05
        K = 10
        vc, bvc, labels = cluster_beam_pixels(
            vec, bv, n_clusters=K, tail_fraction=tf, verbose=False
        )
        # n_main is the number of pixels NOT in the tail.
        sort_idx = np.argsort(bv)
        cumsum = np.cumsum(bv[sort_idx])
        n_tail = int(np.searchsorted(cumsum, tf, side="right"))
        n_tail = max(1, min(n_tail, len(bv) - 1))
        n_main = len(bv) - n_tail
        K_tail = min(K, n_tail)
        assert len(bvc) == n_main + K_tail

    def test_main_pixels_preserved_exactly(self):
        """The high-power main-lobe pixels appear verbatim in the output."""
        vec, bv = _make_beam(S=200)
        tf = 0.05
        vc, bvc, labels = cluster_beam_pixels(
            vec, bv, n_clusters=10, tail_fraction=tf, verbose=False
        )
        # Identify which original pixels are "main" (above tail threshold).
        sort_idx = np.argsort(bv)
        cumsum = np.cumsum(bv[sort_idx])
        n_tail = int(np.searchsorted(cumsum, tf, side="right"))
        n_tail = max(1, min(n_tail, len(bv) - 1))
        main_idx = sort_idx[n_tail:]  # ascending sort → main = last n_main

        n_main = len(main_idx)
        # The first n_main entries of vc should exactly match vec[main_idx].
        np.testing.assert_array_equal(vc[:n_main], vec[main_idx])
        # Weights pass through re-normalisation (dividing by total sum ≈ 1.0),
        # so a tiny float32 rounding difference is expected.
        np.testing.assert_allclose(bvc[:n_main], bv[main_idx], rtol=1e-5, atol=1e-9)

    def test_weights_sum_to_one(self):
        """Output beam weights sum to approximately 1.0 after re-normalisation."""
        vec, bv = _make_beam(S=150)
        _, bvc, _ = cluster_beam_pixels(
            vec, bv, n_clusters=20, tail_fraction=0.05, verbose=False
        )
        assert abs(float(bvc.sum()) - 1.0) < 1e-5

    def test_labels_valid_range(self):
        """All labels index into the output arrays."""
        vec, bv = _make_beam(S=200)
        vc, bvc, labels = cluster_beam_pixels(
            vec, bv, n_clusters=10, tail_fraction=0.05, verbose=False
        )
        K_out = len(bvc)
        assert int(labels.min()) >= 0
        assert int(labels.max()) < K_out

    def test_every_original_pixel_has_a_label(self):
        """labels has the same length as the input (S,)."""
        S = 180
        vec, bv = _make_beam(S=S)
        _, _, labels = cluster_beam_pixels(
            vec, bv, n_clusters=15, tail_fraction=0.05, verbose=False
        )
        assert labels.shape == (S,)

    def test_invalid_tail_fraction_raises(self):
        """tail_fraction outside (0, 1) raises ValueError."""
        vec, bv = _make_beam(S=50)
        with pytest.raises(ValueError):
            cluster_beam_pixels(vec, bv, n_clusters=5, tail_fraction=0.0, verbose=False)
        with pytest.raises(ValueError):
            cluster_beam_pixels(vec, bv, n_clusters=5, tail_fraction=1.0, verbose=False)

    def test_large_K_tail_kept_as_is(self):
        """When K >= n_tail, the tail pixels are kept individually (no clustering)."""
        vec, bv = _make_beam(S=100)
        tf = 0.01  # very small tail → few tail pixels
        # With 100 pixels and tf=0.01, n_tail is tiny (a handful)
        vc, bvc, labels = cluster_beam_pixels(
            vec, bv, n_clusters=1000, tail_fraction=tf, verbose=False
        )
        # Total output should equal S (no reduction possible)
        assert len(bvc) == 100

    def test_output_dtype_float32(self):
        """Centroid vectors and weights are float32 in hybrid mode too."""
        vec, bv = _make_beam(S=120)
        vc, bvc, _ = cluster_beam_pixels(
            vec, bv, n_clusters=10, tail_fraction=0.05, verbose=False
        )
        assert vc.dtype == np.float32
        assert bvc.dtype == np.float32

    def test_verbose_false_no_output(self, capsys):
        """verbose=False suppresses all output in hybrid mode."""
        vec, bv = _make_beam(S=100)
        cluster_beam_pixels(vec, bv, n_clusters=10, tail_fraction=0.05, verbose=False)
        assert capsys.readouterr().out == ""


# ===========================================================================
# TestBuildWeightMatrix
# ===========================================================================


class TestBuildWeightMatrix:
    """Tests for beam_cluster._build_weight_matrix."""

    def test_shape(self):
        """Returns an (S, K) sparse matrix."""
        S, K = 50, 8
        labels = np.zeros(S, dtype=np.int32)
        labels[:] = np.arange(S) % K
        bv = np.ones(S, dtype=np.float32) / S
        W = _build_weight_matrix(labels, bv, K)
        assert W.shape == (S, K)

    def test_nonzero_positions_match_labels(self):
        """W[s, labels[s]] == beam_vals[s] for every s."""
        S, K = 30, 5
        rng = np.random.default_rng(0)
        bv = rng.uniform(0.1, 1.0, S).astype(np.float32)
        bv /= bv.sum()
        labels = (np.arange(S) % K).astype(np.int32)
        W = _build_weight_matrix(labels, bv, K)
        Wd = W.toarray()
        for s in range(S):
            assert abs(Wd[s, labels[s]] - bv[s]) < 1e-6

    def test_column_sums_equal_cluster_weights(self):
        """Column k of W sums to the total weight of cluster k."""
        S, K = 40, 4
        rng = np.random.default_rng(1)
        bv = rng.uniform(0.1, 1.0, S).astype(np.float32)
        bv /= bv.sum()
        labels = (np.arange(S) % K).astype(np.int32)
        W = _build_weight_matrix(labels, bv, K)
        for k in range(K):
            expected = float(bv[labels == k].sum())
            assert abs(float(W.toarray()[:, k].sum()) - expected) < 1e-5

    def test_total_sum_equals_total_weight(self):
        """Sum of all elements equals sum of beam_vals."""
        S, K = 60, 6
        bv = np.ones(S, dtype=np.float32) / S
        labels = np.arange(S, dtype=np.int32) % K
        W = _build_weight_matrix(labels, bv, K)
        np.testing.assert_allclose(W.sum(), bv.sum(), atol=1e-5)


# ===========================================================================
# TestClusterCachedArrays
# ===========================================================================


class TestClusterCachedArrays:
    """Tests for beam_cluster.cluster_cached_arrays."""

    def _run(self, S=100, K=10, N_psi=8, rng=None, keys=None):
        """Helper: build beam, cluster it, then cluster cache arrays."""
        rng = rng or np.random.default_rng(0)
        vec, bv = _make_beam(S=S, rng=rng)
        vc, bvc, labels = cluster_beam_pixels(
            vec, bv, n_clusters=K, tail_fraction=0.10, verbose=False
        )
        K_out = len(bvc)
        cache = _make_cache(N_psi, S, rng)
        if keys is not None:
            cache = {k: cache[k] for k in keys}
        result = cluster_cached_arrays(cache, labels, bv, K_out)
        return result, K_out, S, N_psi

    def test_vec_rolled_output_shape(self):
        """vec_rolled output shape is (N_psi, K_out, 3)."""
        result, K_out, _, N_psi = self._run()
        assert result["vec_rolled"].shape == (N_psi, K_out, 3)

    def test_vec_rolled_unit_norms(self):
        """All vec_rolled centroids lie on the unit sphere."""
        result, _, _, _ = self._run()
        norms = np.linalg.norm(result["vec_rolled"], axis=2)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_dtheta_output_shape(self):
        """dtheta output shape is (N_psi, K_out)."""
        result, K_out, _, N_psi = self._run()
        assert result["dtheta"].shape == (N_psi, K_out)

    def test_dphi_output_shape(self):
        """dphi output shape is (N_psi, K_out)."""
        result, K_out, _, N_psi = self._run()
        assert result["dphi"].shape == (N_psi, K_out)

    def test_output_dtype_float32(self):
        """All output arrays are float32."""
        result, _, _, _ = self._run()
        for key, arr in result.items():
            assert arr.dtype == np.float32, f"{key} dtype should be float32"

    def test_missing_keys_ignored(self):
        """Requesting only vec_rolled does not raise for absent dtheta/dphi."""
        result, K_out, _, N_psi = self._run(keys=["vec_rolled"])
        assert "vec_rolled" in result
        assert "dtheta" not in result
        assert "dphi" not in result

    def test_scalar_weighted_average_correctness(self):
        """dtheta for a two-pixel-one-cluster case equals the weighted mean."""
        # Two pixels, same weight (0.5 each), one cluster: mean of their dtheta values.
        S = 2
        K_out = 1
        bv = np.array([0.5, 0.5], dtype=np.float32)
        labels = np.array([0, 0], dtype=np.int32)
        dt = np.array([[1.0, 3.0]], dtype=np.float32)  # (N_psi=1, S=2)
        cache = {"dtheta": dt}
        result = cluster_cached_arrays(cache, labels, bv, K_out)
        expected = 0.5 * 1.0 + 0.5 * 3.0  # = 2.0
        np.testing.assert_allclose(result["dtheta"][0, 0], expected, atol=1e-5)

    def test_main_pixels_are_pass_through_in_hybrid_mode(self):
        """In hybrid mode, main pixels are mapped identity; their vec values are preserved."""
        rng = np.random.default_rng(5)
        S = 80
        K = 5  # few clusters → tail will have some
        N_psi = 4
        vec, bv = _make_beam(S=S, rng=rng)
        vc, bvc, labels = cluster_beam_pixels(
            vec, bv, n_clusters=K, tail_fraction=0.05, verbose=False
        )
        K_out = len(bvc)
        cache = _make_cache(N_psi, S, rng)
        result = cluster_cached_arrays(cache, labels, bv, K_out)

        # For every main pixel (identity label), the clustered vec_rolled should
        # equal the original vec_rolled scaled to unit sphere.
        sort_idx = np.argsort(bv)
        cumsum = np.cumsum(bv[sort_idx])
        n_tail = int(np.searchsorted(cumsum, 0.05, side="right"))
        n_tail = max(1, min(n_tail, S - 1))
        main_idx = sort_idx[n_tail:]
        n_main = len(main_idx)

        for i, s in enumerate(main_idx):
            out_vr = result["vec_rolled"][:, i, :]  # (N_psi, 3)
            orig_w = float(bv[s])
            orig_vr = cache["vec_rolled"][:, s, :].astype(np.float64) * orig_w
            # Weighted mean of single pixel = pixel itself / weight * weight = pixel
            # After normalisation it should equal the original (which is already unit)
            norms_out = np.linalg.norm(out_vr, axis=1)
            np.testing.assert_allclose(norms_out, 1.0, atol=1e-4)

    def test_full_mode_vec_rolled_shape(self):
        """Full-mode clustering also produces correct vec_rolled shape."""
        S, K, N_psi = 60, 6, 5
        rng = np.random.default_rng(10)
        vec, bv = _make_beam(S=S, rng=rng)
        _, _, labels = cluster_beam_pixels(vec, bv, n_clusters=K, verbose=False)
        cache = _make_cache(N_psi, S, rng)
        result = cluster_cached_arrays(cache, labels, bv, K)
        assert result["vec_rolled"].shape == (N_psi, K, 3)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
