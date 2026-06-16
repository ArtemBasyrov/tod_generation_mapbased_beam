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
