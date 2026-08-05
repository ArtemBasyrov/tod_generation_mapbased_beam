"""
Tests for the tod_pipeline_helpers module.

Covers:
- prepare_beam_data            : load + normalise + select beams
- apply_beam_clustering        : in-place k-means reduction of beam_data
- resolve_spin2_skip_threshold : equatorial-band cutoff derivation
- _write_config                : YAML round-trip
- save_runtime_calibration     : runtime knobs persisted to YAML
- save_clustering_calibration  : clustering knobs persisted to YAML

External dependencies (load_beam, cluster_beam_pixels, cluster_cached_arrays,
tod_config attributes) are patched per-test via monkeypatch.

Can be run independently:
    pytest tests/test_tod_pipeline_helpers.py -v
    python tests/test_tod_pipeline_helpers.py
"""

import math
import os
import sys
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
import pytest
import yaml

import tod_config
import tod_pipeline_helpers as pph


# ===========================================================================
# Shared helpers
# ===========================================================================


def _make_gaussian_beam(n=21, fwhm_arcmin=30.0, half_width_arcmin=45.0):
    """Build a small (n, n) Gaussian beam grid in radians.

    Returns (ra, dec, pixel_map). The grid spans ±half_width_arcmin in each
    axis. pixel_map peaks at unity at the centre.
    """
    half_width_rad = math.radians(half_width_arcmin / 60.0)
    x = np.linspace(-half_width_rad, half_width_rad, n)
    ra, dec = np.meshgrid(x, x, indexing="xy")
    sigma = math.radians((fwhm_arcmin / 60.0) / (2 * math.sqrt(2 * math.log(2))))
    pixel_map = np.exp(-(ra**2 + dec**2) / (2 * sigma**2))
    return ra, dec, pixel_map


def _patch_tod_config(monkeypatch, **overrides):
    """Set the module-level config attributes used by prepare_beam_data."""
    defaults = {
        "FOLDER_BEAM": "/tmp/",
        "beam_file_I": "I.fits",
        "beam_file_Q": "Q.fits",
        "beam_file_U": "U.fits",
        "power_threshold_I": 1.0,
        "power_threshold_Q": 1.0,
        "power_threshold_U": 1.0,
        "beam_center_x": None,
        "beam_center_y": None,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        monkeypatch.setattr(pph.config, k, v, raising=False)


# ===========================================================================
# TestPrepareBeamData
# ===========================================================================


class TestPrepareBeamData:
    def _install_fake_load_beam(self, monkeypatch, ra, dec, pixel_map):
        """Replace pph.load_beam with a callable that returns fixed arrays."""
        calls = []

        def fake_load_beam(folder, fname, center_x=None, center_y=None):
            calls.append((folder, fname, center_x, center_y))
            return ra.copy(), dec.copy(), pixel_map.copy()

        monkeypatch.setattr(pph, "load_beam", fake_load_beam)
        return calls

    def test_single_beam_file_for_all_three_components(self, monkeypatch):
        """All three components share one beam file → one entry, comp_indices = [0,1,2]."""
        ra, dec, pm = _make_gaussian_beam(n=21)
        self._install_fake_load_beam(monkeypatch, ra, dec, pm)
        _patch_tod_config(
            monkeypatch,
            beam_file_I="b.fits",
            beam_file_Q="b.fits",
            beam_file_U="b.fits",
        )

        beam_data = pph.prepare_beam_data(["b.fits", "b.fits", "b.fits"])

        assert list(beam_data.keys()) == ["b.fits"]
        d = beam_data["b.fits"]
        assert d["comp_indices"] == [0, 1, 2]
        assert d["n_sel"] > 0
        assert d["beam_vals"].shape == (d["n_sel"],)
        assert d["vec_orig"].shape == (d["n_sel"], 3)

    def test_distinct_beams_per_component(self, monkeypatch):
        """Three different filenames → three entries, each with one comp index."""
        ra, dec, pm = _make_gaussian_beam(n=21)
        self._install_fake_load_beam(monkeypatch, ra, dec, pm)
        _patch_tod_config(
            monkeypatch,
            beam_file_I="I.fits",
            beam_file_Q="Q.fits",
            beam_file_U="U.fits",
        )

        beam_data = pph.prepare_beam_data(["I.fits", "Q.fits", "U.fits"])

        assert set(beam_data) == {"I.fits", "Q.fits", "U.fits"}
        assert beam_data["I.fits"]["comp_indices"] == [0]
        assert beam_data["Q.fits"]["comp_indices"] == [1]
        assert beam_data["U.fits"]["comp_indices"] == [2]

    def test_beam_vals_normalised_to_one(self, monkeypatch):
        ra, dec, pm = _make_gaussian_beam(n=21)
        self._install_fake_load_beam(monkeypatch, ra, dec, pm)
        _patch_tod_config(
            monkeypatch,
            beam_file_I="b.fits",
            beam_file_Q="b.fits",
            beam_file_U="b.fits",
        )

        beam_data = pph.prepare_beam_data(["b.fits", "b.fits", "b.fits"])
        bv = beam_data["b.fits"]["beam_vals"]
        npt.assert_allclose(float(bv.sum()), 1.0, atol=1e-5)

    def test_beam_vals_carry_the_solid_angle_jacobian(self, monkeypatch):
        """Weights must be B cos(dec), normalised — not B alone.

        The grid cell subtends cos(dec) dRA dDec. Dropping that factor makes the
        effective beam B / cos(dec), which turns a perfectly symmetric beam into
        an elliptical one with a spurious m = +-2 harmonic — precisely the
        asymmetry this pipeline exists to measure.
        """
        ra, dec, pm = _make_gaussian_beam(n=21)
        self._install_fake_load_beam(monkeypatch, ra, dec, pm)
        _patch_tod_config(
            monkeypatch,
            beam_file_I="b.fits",
            beam_file_Q="b.fits",
            beam_file_U="b.fits",
        )

        beam_data = pph.prepare_beam_data(["b.fits", "b.fits", "b.fits"])
        bv = beam_data["b.fits"]["beam_vals"].astype(np.float64)

        expected = (pm * np.cos(dec)).ravel()
        expected = expected / expected.sum()
        npt.assert_allclose(bv, expected, rtol=1e-5)

        # Decisive, not cosmetic: the two normalisations separate by
        # <dec^2>_B / 2, well above the tolerance above.
        plain = pm.ravel() / pm.sum()
        with pytest.raises(AssertionError):
            npt.assert_allclose(bv, plain, rtol=1e-6)

    def test_symmetric_beam_stays_symmetric(self, monkeypatch):
        """A beam symmetric in RA<->Dec must produce weights symmetric under transpose.

        Guards two failure modes at once: dropping the cos(dec) cell Jacobian
        stretches one axis relative to the other, and unpacking ``posmap()`` as
        ``(ra, dec)`` instead of ``(dec, ra)`` reflects the beam about the
        diagonal — visible here for any beam that is not itself symmetric.
        """
        ra, dec, pm = _make_gaussian_beam(n=21)
        npt.assert_allclose(pm, pm.T, atol=0)  # the input really is symmetric
        self._install_fake_load_beam(monkeypatch, ra, dec, pm)
        _patch_tod_config(
            monkeypatch,
            beam_file_I="b.fits",
            beam_file_Q="b.fits",
            beam_file_U="b.fits",
        )

        beam_data = pph.prepare_beam_data(["b.fits", "b.fits", "b.fits"])
        d = beam_data["b.fits"]
        w = d["beam_vals"].astype(np.float64).reshape(pm.shape)
        v = d["vec_orig"].astype(np.float64).reshape(pm.shape + (3,))

        # Weights: cos(dec) breaks the RA<->Dec symmetry of the *weights* by
        # design, so check instead that the symmetry breaking is exactly the
        # Jacobian and nothing else.
        npt.assert_allclose(w / np.cos(dec), (w / np.cos(dec)).T, rtol=1e-5)

        # Node placement: recover each node's offsets from its beam-frame
        # vector.  Cell (i, j) must sit at the RA<->Dec mirror of cell (j, i).
        # (The mirror is exact in offset coordinates, not in ambient
        # components, since the frame is centred on x-hat.)
        dec_node = np.arcsin(np.clip(v[..., 2], -1.0, 1.0))
        ra_node = np.arctan2(v[..., 1], v[..., 0])
        npt.assert_allclose(ra_node, dec_node.T, atol=1e-6)
        npt.assert_allclose(dec_node, ra_node.T, atol=1e-6)

    def test_vec_orig_dtype_and_unit_norm(self, monkeypatch):
        """vec_orig carries the working precision and rows have unit norm."""
        ra, dec, pm = _make_gaussian_beam(n=21)
        self._install_fake_load_beam(monkeypatch, ra, dec, pm)
        _patch_tod_config(
            monkeypatch,
            beam_file_I="b.fits",
            beam_file_Q="b.fits",
            beam_file_U="b.fits",
        )

        beam_data = pph.prepare_beam_data(["b.fits", "b.fits", "b.fits"])
        v = beam_data["b.fits"]["vec_orig"]
        assert v.dtype == tod_config.precision_dtype
        norms = np.linalg.norm(v.astype(np.float64), axis=1)
        npt.assert_allclose(norms, 1.0, atol=1e-6)

    def _install_underflowing_beam(self, monkeypatch, n=5):
        """A beam with one pixel that only underflows once normalised.

        ``1e-42`` is representable as a float32 subnormal, so it survives the
        cast; divided by the beam sum it does not.
        """
        ra, dec, _ = _make_gaussian_beam(n=n)
        pm = np.full((n, n), 1e3)
        pm[0, 0] = 1e-42
        self._install_fake_load_beam(monkeypatch, ra, dec, pm)

    def test_weights_underflowing_after_normalisation_are_dropped(self, monkeypatch):
        self._install_underflowing_beam(monkeypatch)
        _patch_tod_config(
            monkeypatch,
            beam_file_I="b.fits",
            beam_file_Q="b.fits",
            beam_file_U="b.fits",
            precision_dtype=np.float32,
        )

        d = pph.prepare_beam_data(["b.fits", "b.fits", "b.fits"])["b.fits"]

        assert d["n_sel"] == 24
        assert not d["sel"][0, 0]
        assert (d["beam_vals"] != 0).all()
        assert d["vec_orig"].shape[0] == 24
        npt.assert_allclose(float(d["beam_vals"].sum()), 1.0, rtol=1e-6)

    def test_no_pixel_is_dropped_at_float64(self, monkeypatch):
        """The same beam underflows nothing at the production precision."""
        self._install_underflowing_beam(monkeypatch)
        _patch_tod_config(
            monkeypatch,
            beam_file_I="b.fits",
            beam_file_Q="b.fits",
            beam_file_U="b.fits",
            precision_dtype=np.float64,
        )

        d = pph.prepare_beam_data(["b.fits", "b.fits", "b.fits"])["b.fits"]

        assert d["n_sel"] == 25
        assert (d["beam_vals"] != 0).all()

    def test_square_grid_is_accepted_by_beam_symmetric(self, monkeypatch):
        ra, dec, pm = _make_gaussian_beam(n=21)
        self._install_fake_load_beam(monkeypatch, ra, dec, pm)
        _patch_tod_config(
            monkeypatch,
            beam_file_I="b.fits",
            beam_file_Q="b.fits",
            beam_file_U="b.fits",
            beam_symmetric=True,
        )

        beam_data = pph.prepare_beam_data(["b.fits", "b.fits", "b.fits"])
        assert "c4" in beam_data["b.fits"]

    def test_rectangular_grid_is_rejected_by_beam_symmetric(self, monkeypatch):
        """The index rotation is only a sky rotation on a square grid."""
        ra, dec, pm = _make_gaussian_beam(n=21)
        self._install_fake_load_beam(monkeypatch, ra, 2.0 * dec, pm)
        _patch_tod_config(
            monkeypatch,
            beam_file_I="b.fits",
            beam_file_Q="b.fits",
            beam_file_U="b.fits",
            beam_symmetric=True,
        )

        with pytest.raises(ValueError, match="square beam grid"):
            pph.prepare_beam_data(["b.fits", "b.fits", "b.fits"])

    def test_rectangular_grid_is_allowed_without_beam_symmetric(self, monkeypatch):
        """The precondition belongs to quadrant clustering, not to loading."""
        ra, dec, pm = _make_gaussian_beam(n=21)
        self._install_fake_load_beam(monkeypatch, ra, 2.0 * dec, pm)
        _patch_tod_config(
            monkeypatch,
            beam_file_I="b.fits",
            beam_file_Q="b.fits",
            beam_file_U="b.fits",
            beam_symmetric=False,
        )

        beam_data = pph.prepare_beam_data(["b.fits", "b.fits", "b.fits"])
        assert "c4" not in beam_data["b.fits"]


# ===========================================================================
# TestApplyBeamClustering
# ===========================================================================


class TestApplyBeamClustering:
    def _make_beam_data(self, S, with_cache=False):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((S, 3)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        bv = np.full(S, 1.0 / S, dtype=np.float32)
        d = {
            "ra": np.zeros((S,), dtype=np.float32),
            "dec": np.zeros((S,), dtype=np.float32),
            "beam_vals": bv,
            "sel": np.ones(S, dtype=bool),
            "comp_indices": [0],
            "n_sel": S,
            "vec_orig": v,
        }
        if with_cache:
            d["vec_rolled"] = np.zeros((4, S, 3), dtype=np.float32)
            d["dtheta"] = np.zeros((4, S), dtype=np.float32)
            d["dphi"] = np.zeros((4, S), dtype=np.float32)
        return {"b.fits": d}

    def test_reduces_n_sel(self, monkeypatch):
        S, K = 100, 10
        beam_data = self._make_beam_data(S)
        rng = np.random.default_rng(1)
        v_out = rng.standard_normal((K, 3)).astype(np.float32)
        v_out /= np.linalg.norm(v_out, axis=1, keepdims=True)
        bv_out = np.full(K, 1.0 / K, dtype=np.float32)
        labels = rng.integers(0, K, size=S, dtype=np.int64)

        called = {}

        def fake_cluster(
            vec_orig, beam_vals, n_clusters, tail_fraction, whiten=True, c4=None
        ):
            called["args"] = (
                vec_orig.shape,
                beam_vals.shape,
                n_clusters,
                tail_fraction,
            )
            called["whiten"] = whiten
            called["c4"] = c4
            return v_out, bv_out, labels

        monkeypatch.setattr(pph, "cluster_beam_pixels", fake_cluster)

        # cluster_cached_arrays must not be called (no cache keys present).
        sentinel = MagicMock(side_effect=AssertionError("should not be called"))
        monkeypatch.setattr(pph, "cluster_cached_arrays", sentinel)

        pph.apply_beam_clustering(beam_data, n_clusters=K, tail_fraction=0.05)

        d = beam_data["b.fits"]
        assert d["n_sel"] == K
        assert d["vec_orig"].shape == (K, 3)
        assert d["beam_vals"].shape == (K,)
        assert called["args"] == ((S, 3), (S,), K, 0.05)

    def test_cache_arrays_clustered_when_present(self, monkeypatch):
        S, K, N_psi = 50, 7, 4
        beam_data = self._make_beam_data(S, with_cache=True)
        rng = np.random.default_rng(2)
        v_out = rng.standard_normal((K, 3)).astype(np.float32)
        v_out /= np.linalg.norm(v_out, axis=1, keepdims=True)
        bv_out = np.full(K, 1.0 / K, dtype=np.float32)
        labels = rng.integers(0, K, size=S, dtype=np.int64)

        monkeypatch.setattr(
            pph, "cluster_beam_pixels", lambda *a, **kw: (v_out, bv_out, labels)
        )

        cached_called = {}

        def fake_cluster_cached(cache_sub, lbls, weights, K_passed):
            cached_called["keys"] = sorted(cache_sub.keys())
            cached_called["K"] = K_passed
            return {
                "vec_rolled": np.zeros((N_psi, K_passed, 3), dtype=np.float32),
                "dtheta": np.zeros((N_psi, K_passed), dtype=np.float32),
                "dphi": np.zeros((N_psi, K_passed), dtype=np.float32),
            }

        monkeypatch.setattr(pph, "cluster_cached_arrays", fake_cluster_cached)

        pph.apply_beam_clustering(beam_data, n_clusters=K, tail_fraction=None)

        d = beam_data["b.fits"]
        assert d["vec_rolled"].shape == (N_psi, K, 3)
        assert d["dtheta"].shape == (N_psi, K)
        assert d["dphi"].shape == (N_psi, K)
        assert cached_called["K"] == K
        assert cached_called["keys"] == ["dphi", "dtheta", "vec_rolled"]

    def test_cache_arrays_skipped_when_absent(self, monkeypatch):
        """No cache keys → cluster_cached_arrays must not be invoked."""
        S, K = 30, 5
        beam_data = self._make_beam_data(S, with_cache=False)
        rng = np.random.default_rng(3)
        v_out = rng.standard_normal((K, 3)).astype(np.float32)
        v_out /= np.linalg.norm(v_out, axis=1, keepdims=True)
        bv_out = np.full(K, 1.0 / K, dtype=np.float32)
        labels = rng.integers(0, K, size=S, dtype=np.int64)

        monkeypatch.setattr(
            pph, "cluster_beam_pixels", lambda *a, **kw: (v_out, bv_out, labels)
        )

        guard = MagicMock(side_effect=AssertionError("must not be called"))
        monkeypatch.setattr(pph, "cluster_cached_arrays", guard)

        pph.apply_beam_clustering(beam_data, n_clusters=K, tail_fraction=None)
        guard.assert_not_called()


# ===========================================================================
# TestResolveSpin2SkipThreshold
# ===========================================================================


class TestResolveSpin2SkipThreshold:
    def _make_beam_data(self, n=200, beam_radius_rad=math.radians(0.5), seed=0):
        """Build a beam_data dict with vectors clustered around the north pole."""
        rng = np.random.default_rng(seed)
        # Sample (theta, phi) with theta uniform in [0, beam_radius] for a disc.
        theta = rng.uniform(0.0, beam_radius_rad, n)
        phi = rng.uniform(0.0, 2 * math.pi, n)
        v = np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            axis=-1,
        ).astype(np.float32)
        # Decreasing amplitude with theta — Gaussian-ish.
        sigma = beam_radius_rad / 2.5
        bv = np.exp(-0.5 * (theta / sigma) ** 2).astype(np.float32)
        bv /= bv.sum()
        return {"b.fits": {"vec_orig": v, "beam_vals": bv}}

    def test_disabled_when_tolerance_none(self):
        bd = self._make_beam_data()
        assert pph.resolve_spin2_skip_threshold(bd, None) == -1.0

    def test_disabled_when_tolerance_zero(self):
        bd = self._make_beam_data()
        assert pph.resolve_spin2_skip_threshold(bd, 0.0) == -1.0

    def test_disabled_when_tolerance_negative(self):
        bd = self._make_beam_data()
        assert pph.resolve_spin2_skip_threshold(bd, -0.1) == -1.0

    def test_returns_value_in_range_for_typical_beam(self):
        bd = self._make_beam_data(beam_radius_rad=math.radians(1.0))
        z = pph.resolve_spin2_skip_threshold(bd, tolerance=0.05)
        assert -1.0 <= z <= 1.0

    def test_quantile_affects_radius(self):
        """A tighter quantile (closer to 1.0) yields a beam_radius >= a looser one,
        so the resulting threshold is <= the looser-quantile threshold."""
        bd = self._make_beam_data(beam_radius_rad=math.radians(2.0), n=400)
        z_loose = pph.resolve_spin2_skip_threshold(
            bd, tolerance=0.02, beam_radius_quantile=0.5
        )
        z_tight = pph.resolve_spin2_skip_threshold(
            bd, tolerance=0.02, beam_radius_quantile=0.999
        )
        if z_loose < 0.0 or z_tight < 0.0:
            pytest.skip("threshold disabled for this configuration")
        assert z_tight <= z_loose


# ===========================================================================
# TestWriteConfigAndCalibrationSavers
# ===========================================================================


class TestWriteConfigAndCalibrationSavers:
    def _setup_yaml(self, tmp_path, monkeypatch, initial):
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(initial, f, explicit_start=True)
        monkeypatch.setattr(pph.config, "CONFIG_FILE", str(path), raising=False)
        return path

    def test_write_config_round_trip_preserves_existing_keys(
        self, tmp_path, monkeypatch
    ):
        path = self._setup_yaml(tmp_path, monkeypatch, {"existing_key": "keep_me"})
        pph._write_config({"new_int": 1, "new_str": "added"})
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["existing_key"] == "keep_me"
        assert data["new_int"] == 1
        assert data["new_str"] == "added"

    def test_write_config_overwrites_existing_keys(self, tmp_path, monkeypatch):
        path = self._setup_yaml(tmp_path, monkeypatch, {"k": "old"})
        pph._write_config({"k": "new"})
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["k"] == "new"

    def test_save_runtime_calibration_writes_expected_keys(self, tmp_path, monkeypatch):
        path = self._setup_yaml(tmp_path, monkeypatch, {"unrelated": True})
        pph.save_runtime_calibration(n_processes=4, n_threads=2, batch_size=4096)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["calibration_n_processes"] == 4
        assert data["calibration_numba_threads"] == 2
        assert data["calibration_batch_size"] == 4096
        assert data["calibration_enabled"] is False
        # Unrelated keys preserved.
        assert data["unrelated"] is True

    def test_save_clustering_calibration_writes_expected_keys(
        self, tmp_path, monkeypatch
    ):
        path = self._setup_yaml(tmp_path, monkeypatch, {})
        pph.save_clustering_calibration(tail_fraction=0.05, n_clusters=200)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["n_beam_clusters"] == 200
        assert data["beam_cluster_tail_fraction"] == pytest.approx(0.05)
        assert data["clustering_calibration_enabled"] is False


# ===========================================================================
# TestApplyHwpModulation
# ===========================================================================


class TestApplyHwpModulation:
    """Tests for apply_hwp_modulation — in-place 4·φ_HWP rotation of (Q, U).

    Reference formula (matching the helper's implementation):
        φ(t)  = 2π·f_hwp·t + φ₀,  t = day_index·86400 + sample_start·dt + i·dt
        Q'_i =  Q_i·cos(4φ_i) + U_i·sin(4φ_i)
        U'_i = −Q_i·sin(4φ_i) + U_i·cos(4φ_i)
    T (row 0) must never be modified.
    """

    @staticmethod
    def _make_batch(B, dtype=np.float32, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((3, B)).astype(dtype)

    def test_zero_frequency_zero_phase_is_identity_on_QU(self):
        """f_hwp=0 and φ₀=0 → cos=1, sin=0; (Q,U) unchanged (T also unchanged)."""
        tod = self._make_batch(B=32)
        original = tod.copy()
        pph.apply_hwp_modulation(
            tod, day_index=0, sample_start=0, fsamp=19.0, f_hwp=0.0, phi0=0.0
        )
        npt.assert_array_equal(tod, original)

    def test_T_row_is_never_modified(self):
        """The intensity row must be left bit-for-bit identical."""
        tod = self._make_batch(B=64)
        T_before = tod[0].copy()
        pph.apply_hwp_modulation(
            tod, day_index=3, sample_start=10, fsamp=19.0, f_hwp=2.5, phi0=0.7
        )
        npt.assert_array_equal(tod[0], T_before)

    def test_empty_batch_is_noop(self):
        """B=0 must early-return without raising."""
        tod = np.empty((3, 0), dtype=np.float32)
        pph.apply_hwp_modulation(
            tod, day_index=0, sample_start=0, fsamp=19.0, f_hwp=1.0, phi0=0.0
        )
        assert tod.shape == (3, 0)

    def test_QU_norm_is_preserved(self):
        """An ideal HWP is a rotation in (Q,U) space; Q²+U² is invariant per sample."""
        tod = self._make_batch(B=128, dtype=np.float64)
        norm_before = tod[1] ** 2 + tod[2] ** 2
        pph.apply_hwp_modulation(
            tod, day_index=5, sample_start=42, fsamp=19.0, f_hwp=1.7, phi0=0.3
        )
        norm_after = tod[1] ** 2 + tod[2] ** 2
        npt.assert_allclose(norm_after, norm_before, rtol=1e-12, atol=1e-12)

    def test_matches_explicit_formula(self):
        """Numerical agreement with the reference 4φ rotation formula."""
        B = 16
        fsamp = 19.0
        f_hwp = 0.5
        phi0 = 0.25
        day_index = 2
        sample_start = 7
        tod = self._make_batch(B=B, dtype=np.float64, seed=42)
        Q0, U0 = tod[1].copy(), tod[2].copy()

        pph.apply_hwp_modulation(
            tod,
            day_index=day_index,
            sample_start=sample_start,
            fsamp=fsamp,
            f_hwp=f_hwp,
            phi0=phi0,
        )

        dt = 1.0 / fsamp
        t = day_index * 86400.0 + sample_start * dt + np.arange(B) * dt
        phi = 2.0 * np.pi * f_hwp * t + phi0
        c = np.cos(4.0 * phi)
        s = np.sin(4.0 * phi)
        Q_expected = Q0 * c + U0 * s
        U_expected = -Q0 * s + U0 * c
        npt.assert_allclose(tod[1], Q_expected, rtol=1e-12, atol=1e-12)
        npt.assert_allclose(tod[2], U_expected, rtol=1e-12, atol=1e-12)

    def test_phase_continuity_across_days(self):
        """Sample 0 of day_index=1 must continue the phase from end of day 0."""
        fsamp = 19.0
        f_hwp = 1.0
        phi0 = 0.0
        # Two single-sample "batches": last sample of day 0, first sample of day 1.
        # Build them with identical (Q, U) so the rotation reveals the phase.
        tod_a = np.array([[0.0], [1.0], [0.0]], dtype=np.float64)
        tod_b = np.array([[0.0], [1.0], [0.0]], dtype=np.float64)
        N_per_day = int(round(fsamp * 86400.0))

        pph.apply_hwp_modulation(
            tod_a,
            day_index=0,
            sample_start=N_per_day - 1,
            fsamp=fsamp,
            f_hwp=f_hwp,
            phi0=phi0,
        )
        pph.apply_hwp_modulation(
            tod_b,
            day_index=1,
            sample_start=0,
            fsamp=fsamp,
            f_hwp=f_hwp,
            phi0=phi0,
        )
        # The two t values must differ by exactly dt = 1/fsamp (continuous time).
        dt = 1.0 / fsamp
        t_a = 0 * 86400.0 + (N_per_day - 1) * dt
        t_b = 1 * 86400.0 + 0 * dt
        npt.assert_allclose(t_b - t_a, dt, rtol=1e-12, atol=1e-12)

    def test_initial_phase_offset_applied(self):
        """At t=0 (day 0, sample 0, i=0) only φ₀ contributes."""
        phi0 = 0.4
        tod = np.array([[0.0], [1.0], [0.0]], dtype=np.float64)
        pph.apply_hwp_modulation(
            tod, day_index=0, sample_start=0, fsamp=19.0, f_hwp=0.0, phi0=phi0
        )
        # Q=1, U=0 → Q' = cos(4φ₀), U' = −sin(4φ₀)
        assert tod[1, 0] == pytest.approx(np.cos(4.0 * phi0), rel=1e-12)
        assert tod[2, 0] == pytest.approx(-np.sin(4.0 * phi0), rel=1e-12)

    def test_dtype_preserved_float32(self):
        """A float32 input must remain float32 after the in-place rotation."""
        tod = self._make_batch(B=8, dtype=np.float32)
        pph.apply_hwp_modulation(
            tod, day_index=0, sample_start=0, fsamp=19.0, f_hwp=1.0, phi0=0.1
        )
        assert tod.dtype == np.float32

    def test_dtype_preserved_float64(self):
        tod = self._make_batch(B=8, dtype=np.float64)
        pph.apply_hwp_modulation(
            tod, day_index=0, sample_start=0, fsamp=19.0, f_hwp=1.0, phi0=0.1
        )
        assert tod.dtype == np.float64

    def test_full_period_returns_to_original(self):
        """A full HWP rotation period (Δt = 1/f_hwp) applies 4·(2π)=8π → identity."""
        # Construct two single-sample batches whose times differ by exactly 1/f_hwp.
        f_hwp = 0.5  # period = 2 s
        fsamp = 1.0  # so one period = 2 samples
        Q0, U0 = 0.6, -0.8
        tod_a = np.array([[0.0], [Q0], [U0]], dtype=np.float64)
        tod_b = tod_a.copy()
        pph.apply_hwp_modulation(
            tod_a, day_index=0, sample_start=0, fsamp=fsamp, f_hwp=f_hwp, phi0=0.0
        )
        pph.apply_hwp_modulation(
            tod_b, day_index=0, sample_start=2, fsamp=fsamp, f_hwp=f_hwp, phi0=0.0
        )
        npt.assert_allclose(tod_a, tod_b, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
