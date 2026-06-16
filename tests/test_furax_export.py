"""
Tests for the multi-detector furax export (tod_to_furax.py).

The end-to-end roundtrip needs toast (writes a real TOAST HDF5 observation);
those tests skip cleanly when toast is unavailable. The convention-level
checks (per-detector signal combination, scalar-last quaternion ordering) are
validated by reading the HDF5 back with h5py.
"""

import numpy as np
import pytest

import tod_focalplane as fp
from tod_focalplane import Detector

# The converter imports toast at module level.
toast = pytest.importorskip("toast")
h5py = pytest.importorskip("h5py")

import tod_to_furax as tf  # noqa: E402


# ── Pure-function convention checks ───────────────────────────────────────────


def test_detector_quat_scalar_last_boresight_is_identity():
    boresight = Detector("boresight", None, 0.0)
    np.testing.assert_array_equal(
        tf._detector_quat_scalar_last(boresight), [0.0, 0.0, 0.0, 1.0]
    )


def test_detector_quat_scalar_last_reorders():
    q_sf = fp.from_xieta(0.01, -0.02, np.deg2rad(30.0))  # (w, x, y, z)
    det = Detector("d", q_sf, np.deg2rad(30.0))
    sl = tf._detector_quat_scalar_last(det)  # (x, y, z, w)
    np.testing.assert_allclose(sl, [q_sf[1], q_sf[2], q_sf[3], q_sf[0]], atol=0)


def test_combine_iqu_to_signal_matches_formula():
    rng = np.random.default_rng(0)
    iqu = rng.normal(size=(3, 7))
    psi_d = rng.uniform(-np.pi, np.pi, 7)
    out = tf._combine_iqu_to_signal(iqu, psi_d)
    expect = iqu[0] + iqu[1] * np.cos(2 * psi_d) + iqu[2] * np.sin(2 * psi_d)
    np.testing.assert_allclose(out, expect, atol=1e-12)


def test_combine_detector_signal_boresight_vs_gamma():
    rng = np.random.default_rng(3)
    n = 8
    theta = rng.uniform(0.2, np.pi - 0.2, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    psi = rng.uniform(-np.pi, np.pi, n)
    iqu = rng.normal(size=(3, n))

    boresight = Detector("boresight", None, 0.0)
    out_b = tf.combine_detector_signal(iqu, theta, phi, psi, boresight)
    exp_b = iqu[0] + iqu[1] * np.cos(2 * psi) + iqu[2] * np.sin(2 * psi)
    np.testing.assert_allclose(out_b, exp_b, atol=1e-12)

    B = Detector("B", fp.from_xieta(0.0, 0.0, np.deg2rad(90.0)), np.deg2rad(90.0))
    psi_d = fp.detector_pointing_batch(theta, phi, psi, B.quat)[2]
    out_g = tf.combine_detector_signal(iqu, theta, phi, psi, B)
    exp_g = iqu[0] + iqu[1] * np.cos(2 * psi_d) + iqu[2] * np.sin(2 * psi_d)
    np.testing.assert_allclose(out_g, exp_g, atol=1e-12)


# ── End-to-end convert_day roundtrip (2 detectors) ────────────────────────────


def _write_inputs(tmp_path, n=6, seed=1):
    rng = np.random.default_rng(seed)
    scan = tmp_path / "scan"
    tod = tmp_path / "tod"
    scan.mkdir()
    tod.mkdir()
    theta = rng.uniform(0.2, np.pi - 0.2, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    psi = rng.uniform(-np.pi, np.pi, n)
    np.save(scan / "theta_0.npy", theta)
    np.save(scan / "phi_0.npy", phi)
    np.save(scan / "psi_0.npy", psi)
    return scan, tod, theta, phi, psi, rng


def test_convert_day_writes_multidet_signal(tmp_path):
    scan, tod, theta, phi, psi, rng = _write_inputs(tmp_path)
    n = theta.shape[0]
    A = Detector("A", fp.from_xieta(0.0, 0.0, 0.0), 0.0)
    B = Detector("B", fp.from_xieta(0.0, 0.0, np.deg2rad(90.0)), np.deg2rad(90.0))
    detectors = [A, B]

    iqu_A = rng.normal(size=(3, n))
    iqu_B = rng.normal(size=(3, n))
    np.save(tod / "tod_day_0_A.npy", iqu_A)
    np.save(tod / "tod_day_0_B.npy", iqu_B)

    out = tmp_path / "h5"
    paths = tf.convert_day(
        day_index=0,
        folder_scan=str(scan) + "/",
        folder_tod=str(tod) + "/",
        folder_out=str(out),
        fsamp=19.0,
        t0_unix=0.0,
        hwp_enabled=False,
        f_hwp=0.0,
        phi0_hwp=0.0,
        detectors=detectors,
        n_splits=1,
        signal_dtype=np.float64,
    )
    assert len(paths) == 1 and paths[0].exists()

    # Expected per-detector signal: A uses boresight psi; B uses psi_d.
    psi_dB = fp.detector_pointing_batch(theta, phi, psi, B.quat)[2]
    exp_A = iqu_A[0] + iqu_A[1] * np.cos(2 * psi) + iqu_A[2] * np.sin(2 * psi)
    exp_B = iqu_B[0] + iqu_B[1] * np.cos(2 * psi_dB) + iqu_B[2] * np.sin(2 * psi_dB)

    with h5py.File(paths[0], "r") as f:
        signal = f["detdata/signal"][:]
        fpl = f["instrument/focalplane"][:]
        boresight = f["shared/boresight_radec"][:]

    assert signal.shape == (2, n)
    np.testing.assert_allclose(signal[0], exp_A, atol=1e-10)
    np.testing.assert_allclose(signal[1], exp_B, atol=1e-10)

    # Focalplane rows: order, names, gamma, scalar-last quats.
    names = [name.decode() if isinstance(name, bytes) else name for name in fpl["name"]]
    assert names == ["A", "B"]
    np.testing.assert_allclose(fpl["gamma"], [0.0, np.deg2rad(90.0)], atol=1e-12)
    np.testing.assert_allclose(fpl["quat"][0], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        fpl["quat"][1], tf._detector_quat_scalar_last(B), atol=1e-12
    )
    assert boresight.shape == (n, 4)


def test_convert_day_boresight_legacy_name(tmp_path):
    """A single implicit boresight detector reads the legacy tod_day_N.npy."""
    scan, tod, theta, phi, psi, rng = _write_inputs(tmp_path, seed=2)
    n = theta.shape[0]
    boresight = Detector("boresight", None, 0.0)
    iqu = rng.normal(size=(3, n))
    np.save(tod / "tod_day_0.npy", iqu)  # legacy name (no detector suffix)

    out = tmp_path / "h5"
    paths = tf.convert_day(
        day_index=0,
        folder_scan=str(scan) + "/",
        folder_tod=str(tod) + "/",
        folder_out=str(out),
        fsamp=19.0,
        t0_unix=0.0,
        hwp_enabled=False,
        f_hwp=0.0,
        phi0_hwp=0.0,
        detectors=[boresight],
        n_splits=1,
        signal_dtype=np.float64,
    )
    with h5py.File(paths[0], "r") as f:
        signal = f["detdata/signal"][:]
    exp = iqu[0] + iqu[1] * np.cos(2 * psi) + iqu[2] * np.sin(2 * psi)
    assert signal.shape == (1, n)
    np.testing.assert_allclose(signal[0], exp, atol=1e-10)


def test_write_day_observation_matches_convert_day(tmp_path):
    """The in-memory writer (no .npy) produces the same detdata as convert_day."""
    scan, tod, theta, phi, psi, rng = _write_inputs(tmp_path, seed=4)
    n = theta.shape[0]
    A = Detector("A", fp.from_xieta(0.0, 0.0, 0.0), 0.0)
    B = Detector("B", fp.from_xieta(0.01, -0.02, np.deg2rad(60.0)), np.deg2rad(60.0))
    detectors = [A, B]

    iqu = {d.name: rng.normal(size=(3, n)) for d in detectors}
    for d in detectors:
        np.save(tod / f"tod_day_0_{d.name}.npy", iqu[d.name])

    # Path 1: standalone npy → HDF5.
    out_npy = tmp_path / "h5_npy"
    p_npy = tf.convert_day(
        day_index=0,
        folder_scan=str(scan) + "/",
        folder_tod=str(tod) + "/",
        folder_out=str(out_npy),
        fsamp=19.0,
        t0_unix=0.0,
        hwp_enabled=False,
        f_hwp=0.0,
        phi0_hwp=0.0,
        detectors=detectors,
        n_splits=1,
        signal_dtype=np.float64,
    )

    # Path 2: in-memory assembled signal → HDF5 (the generator's path).
    signal = np.empty((2, n))
    for di, d in enumerate(detectors):
        signal[di] = tf.combine_detector_signal(iqu[d.name], theta, phi, psi, d)
    out_mem = tmp_path / "h5_mem"
    p_mem = tf.write_day_observation(
        day_index=0,
        signal=signal,
        theta=theta,
        phi=phi,
        psi=psi,
        detectors=detectors,
        folder_out=str(out_mem),
        fsamp=19.0,
        t0_unix=0.0,
        hwp_enabled=False,
        f_hwp=0.0,
        phi0_hwp=0.0,
        n_splits=1,
        signal_dtype=np.float64,
    )

    with h5py.File(p_npy[0], "r") as f:
        sig_npy = f["detdata/signal"][:]
    with h5py.File(p_mem[0], "r") as f:
        sig_mem = f["detdata/signal"][:]
    np.testing.assert_array_equal(sig_npy, sig_mem)


# ── Generator finalizer helper ────────────────────────────────────────────────


def test_expected_obs_paths_naming():
    import sample_based_tod_generation_gridint as gen

    assert gen._expected_obs_paths("/h5", 4, 1) == ["/h5/obs_day_4.h5"]
    assert gen._expected_obs_paths("/h5", 4, 3) == [
        "/h5/obs_day_4_p0.h5",
        "/h5/obs_day_4_p1.h5",
        "/h5/obs_day_4_p2.h5",
    ]
