"""
Tests for the focal-plane / multi-detector layer (tod_focalplane.py).

Covers:
- quaternion algebra matching furax.math.quaternion exactly (T2),
- the gamma-only pointing invariant (psi shifts by gamma, sky position fixed —
  the unit-level core of validation test T3),
- detector loading, subset selection, defaults, and output naming.
"""

import numpy as np
import pytest

import tod_config as config
import tod_focalplane as fp
from tod_focalplane import Detector


# ── Quaternion convention cross-check against furax (T2) ──────────────────────

# furax pulls in jax; skip the cross-check cleanly if it is not importable in
# the test environment. The numpy ports stand alone for the remaining tests.
furax_q = pytest.importorskip("furax.math.quaternion")


def _rng_angles(n, seed):
    rng = np.random.default_rng(seed)
    # theta in (0, pi) avoids the pole degeneracy of the ISO decomposition.
    theta = rng.uniform(0.1, np.pi - 0.1, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    psi = rng.uniform(-np.pi, np.pi, n)
    return theta, phi, psi


def test_from_iso_angles_matches_furax():
    theta, phi, psi = _rng_angles(50, seed=1)
    ours = fp.from_iso_angles(theta, phi, psi)
    theirs = np.asarray(furax_q.from_iso_angles(theta, phi, psi))
    np.testing.assert_allclose(ours, theirs, atol=1e-6, rtol=0)


def test_to_iso_angles_matches_furax():
    theta, phi, psi = _rng_angles(50, seed=2)
    q = fp.from_iso_angles(theta, phi, psi)
    th, ph, ps = fp.to_iso_angles(q)
    tth, tph, tps = (np.asarray(a) for a in furax_q.to_iso_angles(q))
    np.testing.assert_allclose(th, tth, atol=1e-6, rtol=0)
    np.testing.assert_allclose(ph, tph, atol=1e-6, rtol=0)
    np.testing.assert_allclose(ps, tps, atol=1e-6, rtol=0)


def test_qmul_matches_furax():
    rng = np.random.default_rng(3)
    q1 = rng.normal(size=(20, 4))
    q2 = rng.normal(size=(20, 4))
    ours = fp.qmul(q1, q2)
    theirs = np.asarray(furax_q.qmul(q1, q2))
    np.testing.assert_allclose(ours, theirs, atol=1e-6, rtol=0)


def test_from_xieta_matches_furax():
    rng = np.random.default_rng(4)
    for _ in range(20):
        xi = rng.uniform(-0.05, 0.05)
        eta = rng.uniform(-0.05, 0.05)
        gamma = rng.uniform(-np.pi, np.pi)
        ours = fp.from_xieta(xi, eta, gamma)
        theirs = np.asarray(furax_q.from_xieta_angles(xi, eta, gamma))
        np.testing.assert_allclose(ours, theirs, atol=1e-6, rtol=0)


def test_composition_matches_furax_pointing_order():
    """detector_pointing_batch reproduces qmul(q_boresight, q_det) -> iso."""
    theta, phi, psi = _rng_angles(30, seed=5)
    q_det = fp.from_xieta(0.01, -0.02, np.deg2rad(45.0))

    th_d, ph_d, ps_d = fp.detector_pointing_batch(theta, phi, psi, q_det)

    q_b = np.asarray(furax_q.from_iso_angles(theta, phi, psi))
    q_full = np.asarray(furax_q.qmul(q_b, q_det))
    tth, tph, tps = (np.asarray(a) for a in furax_q.to_iso_angles(q_full))
    np.testing.assert_allclose(th_d, tth, atol=1e-6, rtol=0)
    np.testing.assert_allclose(ph_d, tph, atol=1e-6, rtol=0)
    np.testing.assert_allclose(ps_d, tps, atol=1e-6, rtol=0)


# ── Pointing invariants (no furax needed) ─────────────────────────────────────


def test_identity_offset_roundtrips():
    """A zero (xi, eta, gamma) offset returns the boresight pointing unchanged."""
    theta, phi, psi = _rng_angles(40, seed=6)
    q_id = fp.from_xieta(0.0, 0.0, 0.0)
    th_d, ph_d, ps_d = fp.detector_pointing_batch(theta, phi, psi, q_id)
    np.testing.assert_allclose(th_d, theta, atol=1e-12, rtol=0)
    # phi/psi wrap; compare via angle difference.
    np.testing.assert_allclose(np.cos(ph_d), np.cos(phi), atol=1e-12)
    np.testing.assert_allclose(np.sin(ph_d), np.sin(phi), atol=1e-12)
    np.testing.assert_allclose(np.cos(ps_d), np.cos(psi), atol=1e-12)
    np.testing.assert_allclose(np.sin(ps_d), np.sin(psi), atol=1e-12)


def test_gamma_only_shifts_psi_keeps_position():
    """T3 (unit level): a pure gamma offset rotates psi by gamma and leaves the
    sky position (theta, phi) unchanged — the property the symmetric-beam T3
    validation run relies on (T row identical, Q/U rotate by 2*gamma)."""
    theta, phi, psi = _rng_angles(40, seed=7)
    gamma = np.deg2rad(90.0)
    q_det = fp.from_xieta(0.0, 0.0, gamma)
    th_d, ph_d, ps_d = fp.detector_pointing_batch(theta, phi, psi, q_det)

    np.testing.assert_allclose(th_d, theta, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.cos(ph_d), np.cos(phi), atol=1e-12)
    np.testing.assert_allclose(np.sin(ph_d), np.sin(phi), atol=1e-12)
    # psi advanced by exactly gamma (compare on the circle to dodge wrapping).
    np.testing.assert_allclose(np.cos(ps_d), np.cos(psi + gamma), atol=1e-12)
    np.testing.assert_allclose(np.sin(ps_d), np.sin(psi + gamma), atol=1e-12)


# ── Detector loading / subset / naming ────────────────────────────────────────


def test_load_detectors_default_is_boresight(monkeypatch):
    monkeypatch.setattr(config, "detectors", None)
    monkeypatch.setattr(config, "detector_subset", None)
    dets = fp.load_detectors()
    assert len(dets) == 1
    assert dets[0].name == "boresight"
    assert dets[0].is_boresight
    assert dets[0].quat is None


def test_load_detectors_builds_quaternions(monkeypatch):
    monkeypatch.setattr(
        config,
        "detectors",
        [
            {"name": "A", "xi_deg": 0.0, "eta_deg": 0.0, "gamma_deg": 0.0},
            {"name": "B", "xi_deg": 0.0, "eta_deg": 0.0, "gamma_deg": 90.0},
        ],
    )
    monkeypatch.setattr(config, "detector_subset", None)
    dets = fp.load_detectors()
    assert [d.name for d in dets] == ["A", "B"]
    assert all(d.quat is not None and d.quat.shape == (4,) for d in dets)
    assert not any(d.is_boresight for d in dets)
    np.testing.assert_allclose(dets[1].gamma, np.deg2rad(90.0))


def test_detector_subset_by_name_and_index(monkeypatch):
    monkeypatch.setattr(
        config,
        "detectors",
        [
            {"name": "A", "xi_deg": 0.0, "eta_deg": 0.0, "gamma_deg": 0.0},
            {"name": "B", "xi_deg": 1.0, "eta_deg": 0.0, "gamma_deg": 90.0},
            {"name": "C", "xi_deg": 0.0, "eta_deg": 1.0, "gamma_deg": 45.0},
        ],
    )
    monkeypatch.setattr(config, "detector_subset", ["C", 0])
    dets = fp.load_detectors()
    # Configured order is preserved regardless of subset listing order.
    assert [d.name for d in dets] == ["A", "C"]


def test_detector_subset_unknown_name_raises(monkeypatch):
    monkeypatch.setattr(
        config,
        "detectors",
        [{"name": "A", "xi_deg": 0.0, "eta_deg": 0.0, "gamma_deg": 0.0}],
    )
    monkeypatch.setattr(config, "detector_subset", ["nope"])
    with pytest.raises(ValueError, match="not a configured detector name"):
        fp.load_detectors()


def test_detector_subset_index_out_of_range_raises(monkeypatch):
    monkeypatch.setattr(
        config,
        "detectors",
        [{"name": "A", "xi_deg": 0.0, "eta_deg": 0.0, "gamma_deg": 0.0}],
    )
    monkeypatch.setattr(config, "detector_subset", [5])
    with pytest.raises(ValueError, match="out of range"):
        fp.load_detectors()


def test_output_path_naming():
    boresight = Detector(name="boresight", quat=None, gamma=0.0)
    det = Detector(name="det_000B", quat=np.zeros(4), gamma=0.0)
    assert fp.tod_output_path("/out/", 7, boresight) == "/out/tod_day_7.npy"
    assert fp.tod_output_path("/out/", 7, det) == "/out/tod_day_7_det_000B.npy"


# ── Per-detector beam sets (Phase 2) ──────────────────────────────────────────


@pytest.fixture
def _global_beams(monkeypatch):
    """Pin the global beam files/thresholds so beam_key resolution is stable."""
    monkeypatch.setattr(config, "beam_file_I", "I.fits")
    monkeypatch.setattr(config, "beam_file_Q", "Q.fits")
    monkeypatch.setattr(config, "beam_file_U", "U.fits")
    monkeypatch.setattr(config, "power_threshold_I", 1.0)
    monkeypatch.setattr(config, "power_threshold_Q", 1.0)
    monkeypatch.setattr(config, "power_threshold_U", 1.0)


def test_default_focal_plane_uses_default_beam_key(monkeypatch, _global_beams):
    monkeypatch.setattr(config, "detectors", None)
    monkeypatch.setattr(config, "detector_subset", None)
    dets = fp.load_detectors()
    assert dets[0].beam_key == "default"
    specs = fp.build_beam_specs(dets)
    assert list(specs) == ["default"]
    assert specs["default"] == fp._global_beam_spec()


def test_detectors_without_overrides_share_default_set(monkeypatch, _global_beams):
    monkeypatch.setattr(
        config,
        "detectors",
        [
            {"name": "A", "xi_deg": 0.0, "eta_deg": 0.0, "gamma_deg": 0.0},
            {"name": "B", "xi_deg": 0.0, "eta_deg": 0.0, "gamma_deg": 90.0},
        ],
    )
    monkeypatch.setattr(config, "detector_subset", None)
    dets = fp.load_detectors()
    assert [d.beam_key for d in dets] == ["default", "default"]
    # One shared beam set despite two detectors.
    assert list(fp.build_beam_specs(dets)) == ["default"]


def test_per_detector_beam_override_creates_distinct_set(monkeypatch, _global_beams):
    monkeypatch.setattr(
        config,
        "detectors",
        [
            {"name": "A", "xi_deg": 0.0, "eta_deg": 0.0, "gamma_deg": 0.0},
            {
                "name": "B",
                "xi_deg": 0.0,
                "eta_deg": 0.0,
                "gamma_deg": 90.0,
                "beam_file_I": "I_asym.fits",
                "beam_file_Q": "Q_asym.fits",
                "beam_file_U": "U_asym.fits",
            },
        ],
    )
    monkeypatch.setattr(config, "detector_subset", None)
    dets = fp.load_detectors()
    assert dets[0].beam_key == "default"
    assert dets[1].beam_key != "default"
    specs = fp.build_beam_specs(dets)
    assert set(specs) == {"default", dets[1].beam_key}
    assert specs[dets[1].beam_key].beam_files == (
        "I_asym.fits",
        "Q_asym.fits",
        "U_asym.fits",
    )
    # The non-overridden detector keeps the global files.
    assert specs["default"].beam_files == ("I.fits", "Q.fits", "U.fits")


def test_ab_pair_sharing_override_files_share_one_set(monkeypatch, _global_beams):
    """An A/B pair pointing at the same (overridden) beam files shares a set."""
    shared = {
        "beam_file_I": "I_asym.fits",
        "beam_file_Q": "Q_asym.fits",
        "beam_file_U": "U_asym.fits",
    }
    monkeypatch.setattr(
        config,
        "detectors",
        [
            {"name": "A", "xi_deg": 0.0, "eta_deg": 0.0, "gamma_deg": 0.0, **shared},
            {"name": "B", "xi_deg": 0.0, "eta_deg": 0.0, "gamma_deg": 90.0, **shared},
        ],
    )
    monkeypatch.setattr(config, "detector_subset", None)
    dets = fp.load_detectors()
    assert dets[0].beam_key == dets[1].beam_key != "default"
    assert len(fp.build_beam_specs(dets)) == 1


def test_partial_override_falls_back_to_global(monkeypatch, _global_beams):
    """Overriding only beam_file_Q keeps I/U at the global files."""
    monkeypatch.setattr(
        config,
        "detectors",
        [
            {
                "name": "A",
                "xi_deg": 0.0,
                "eta_deg": 0.0,
                "gamma_deg": 0.0,
                "beam_file_Q": "Q_special.fits",
            }
        ],
    )
    monkeypatch.setattr(config, "detector_subset", None)
    dets = fp.load_detectors()
    spec = fp.build_beam_specs(dets)[dets[0].beam_key]
    assert spec.beam_files == ("I.fits", "Q_special.fits", "U.fits")


def test_clustering_override_creates_distinct_set(monkeypatch, _global_beams):
    """Same beam files but a per-detector clustering override → its own set."""
    monkeypatch.setattr(
        config,
        "detectors",
        [
            {"name": "A", "xi_deg": 0.0, "eta_deg": 0.0, "gamma_deg": 0.0},
            {
                "name": "B",
                "xi_deg": 0.0,
                "eta_deg": 0.0,
                "gamma_deg": 90.0,
                "n_beam_clusters": 100,
                "beam_cluster_tail_fraction": 0.03,
            },
        ],
    )
    monkeypatch.setattr(config, "detector_subset", None)
    dets = fp.load_detectors()
    assert dets[0].beam_key == "default"
    assert dets[1].beam_key != "default"
    specs = fp.build_beam_specs(dets)
    # A inherits global clustering (None); B carries the explicit override.
    assert specs["default"].n_clusters is None
    assert specs["default"].tail_fraction is None
    assert specs[dets[1].beam_key].n_clusters == 100
    assert specs[dets[1].beam_key].tail_fraction == 0.03
    # Files are still the global ones — only the clustering differs.
    assert specs[dets[1].beam_key].beam_files == ("I.fits", "Q.fits", "U.fits")


def test_clustering_only_partial_override(monkeypatch, _global_beams):
    """Overriding only n_beam_clusters leaves tail_fraction inheriting global."""
    monkeypatch.setattr(
        config,
        "detectors",
        [
            {
                "name": "A",
                "xi_deg": 0.0,
                "eta_deg": 0.0,
                "gamma_deg": 0.0,
                "n_beam_clusters": 50,
            }
        ],
    )
    monkeypatch.setattr(config, "detector_subset", None)
    dets = fp.load_detectors()
    spec = fp.build_beam_specs(dets)[dets[0].beam_key]
    assert spec.n_clusters == 50
    assert spec.tail_fraction is None


def test_build_beam_specs_respects_subset(monkeypatch, _global_beams):
    """Only beam sets of selected detectors are loaded."""
    monkeypatch.setattr(
        config,
        "detectors",
        [
            {"name": "A", "xi_deg": 0.0, "eta_deg": 0.0, "gamma_deg": 0.0},
            {
                "name": "B",
                "xi_deg": 0.0,
                "eta_deg": 0.0,
                "gamma_deg": 90.0,
                "beam_file_I": "I_asym.fits",
                "beam_file_Q": "Q_asym.fits",
                "beam_file_U": "U_asym.fits",
            },
        ],
    )
    monkeypatch.setattr(config, "detector_subset", ["A"])
    dets = fp.load_detectors()
    assert [d.name for d in dets] == ["A"]
    # B's distinct beam set is not loaded for this shard.
    assert list(fp.build_beam_specs(dets)) == ["default"]
