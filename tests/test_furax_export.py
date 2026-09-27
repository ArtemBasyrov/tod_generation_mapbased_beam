"""
Tests for the furax HDF5 export (tod_to_furax.py) and its generator hook.

The roundtrip tests write a real TOAST HDF5 observation and read it back with
h5py; they skip when toast is unavailable.

Can be run independently:
    pytest tests/test_furax_export.py -v
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import numpy.testing as npt
import pytest

from tod_pipeline_helpers import _combine_iqu_to_signal, _hwp_angle

toast = pytest.importorskip("toast")
h5py = pytest.importorskip("h5py")

import tod_to_furax as tf  # noqa: E402

_FSAMP = 19.0


def _write_inputs(tmp_path, n=6, seed=1, day=0):
    rng = np.random.default_rng(seed)
    scan = tmp_path / "scan"
    tod = tmp_path / "tod"
    scan.mkdir(exist_ok=True)
    tod.mkdir(exist_ok=True)
    theta = rng.uniform(0.2, np.pi - 0.2, n)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    psi = rng.uniform(-np.pi, np.pi, n)
    np.save(scan / f"theta_{day}.npy", theta)
    np.save(scan / f"phi_{day}.npy", phi)
    np.save(scan / f"psi_{day}.npy", psi)
    iqu = rng.normal(size=(3, n))
    np.save(tod / f"tod_day_{day}.npy", iqu)
    return scan, tod, theta, phi, psi, iqu


def _convert(tmp_path, scan, tod, day=0, hwp=False, f_hwp=0.0, phi0=0.0, **kw):
    return tf.convert_day(
        day,
        str(scan),
        str(tod),
        str(tmp_path / "h5"),
        _FSAMP,
        0.0,
        hwp,
        f_hwp,
        phi0,
        **kw,
    )


def _read(path):
    with h5py.File(path, "r") as f:
        return {
            "signal": f["detdata/signal"][:],
            "boresight": f["shared/boresight_radec"][:],
            "hwp": f["shared/hwp_angle"][:],
            "times": f["shared/times"][:],
        }


def test_combine_iqu_to_signal_matches_formula():
    rng = np.random.default_rng(0)
    iqu = rng.normal(size=(3, 7))
    psi = rng.uniform(-np.pi, np.pi, 7)
    expect = iqu[0] + iqu[1] * np.cos(2 * psi) + iqu[2] * np.sin(2 * psi)
    npt.assert_allclose(_combine_iqu_to_signal(iqu, psi), expect, atol=1e-12)


def test_observation_paths_naming(tmp_path):
    assert tf.observation_paths(tmp_path, 4) == [tmp_path / "obs_day_4.h5"]
    assert [p.name for p in tf.observation_paths(tmp_path, 4, 3)] == [
        "obs_day_4_p0.h5",
        "obs_day_4_p1.h5",
        "obs_day_4_p2.h5",
    ]


def test_convert_day_roundtrip(tmp_path):
    scan, tod, theta, phi, psi, iqu = _write_inputs(tmp_path)
    paths = _convert(tmp_path, scan, tod, day=0)
    assert paths == [tmp_path / "h5" / "obs_day_0.h5"] and paths[0].exists()

    out = _read(paths[0])
    assert out["signal"].shape == (1, theta.size)
    npt.assert_allclose(out["signal"][0], _combine_iqu_to_signal(iqu, psi), atol=1e-12)
    npt.assert_allclose(
        out["boresight"], tf._angles_to_boresight_radec_quats(theta, phi, psi)
    )
    npt.assert_array_equal(out["hwp"], 0.0)


def test_hwp_angle_is_wrapped_generator_phase(tmp_path):
    """The stored HWP angle is the generator's phase, wrapped to [0, 2 pi)."""
    day, f_hwp, phi0 = 3, 1.3, 0.4
    scan, tod, *_ = _write_inputs(tmp_path, day=day)
    out = _read(
        _convert(tmp_path, scan, tod, day=day, hwp=True, f_hwp=f_hwp, phi0=phi0)[0]
    )
    raw = _hwp_angle(day, 0, out["hwp"].size, _FSAMP, f_hwp, phi0)
    assert raw.min() > 2 * np.pi  # the test exercises the wrap
    assert np.all((out["hwp"] >= 0.0) & (out["hwp"] < 2 * np.pi))
    npt.assert_allclose(np.cos(4 * out["hwp"]), np.cos(4 * raw), atol=1e-9)
    npt.assert_allclose(np.sin(4 * out["hwp"]), np.sin(4 * raw), atol=1e-9)


def test_split_per_day_covers_all_samples(tmp_path):
    scan, tod, theta, phi, psi, iqu = _write_inputs(tmp_path, n=7)
    paths = _convert(tmp_path, scan, tod, n_splits=3)
    assert [p.name for p in paths] == [
        "obs_day_0_p0.h5",
        "obs_day_0_p1.h5",
        "obs_day_0_p2.h5",
    ]
    signal = np.concatenate([_read(p)["signal"][0] for p in paths])
    npt.assert_allclose(signal, _combine_iqu_to_signal(iqu, psi), atol=1e-12)


def test_rerun_overwrites_day(tmp_path):
    """A second export of the same day replaces the SQLite index row."""
    scan, tod, *_ = _write_inputs(tmp_path)
    _convert(tmp_path, scan, tod)
    iqu2 = np.ones((3, 6))
    np.save(tod / "tod_day_0.npy", iqu2)
    psi = np.load(scan / "psi_0.npy")
    out = _read(_convert(tmp_path, scan, tod)[0])
    npt.assert_allclose(out["signal"][0], _combine_iqu_to_signal(iqu2, psi))


def test_non_finite_signal_is_rejected(tmp_path):
    scan, tod, theta, phi, psi, iqu = _write_inputs(tmp_path)
    signal = _combine_iqu_to_signal(iqu, psi)
    signal[2] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        tf.write_day_observation(
            0,
            signal,
            theta,
            phi,
            psi,
            str(tmp_path / "h5"),
            _FSAMP,
            0.0,
            False,
            0.0,
            0.0,
        )
    assert not (tmp_path / "h5" / "obs_day_0.h5").exists()


def test_float32_signal_dtype(tmp_path):
    scan, tod, *_ = _write_inputs(tmp_path)
    path = _convert(tmp_path, scan, tod, signal_dtype=np.float32)[0]
    with h5py.File(path, "r") as f:
        assert f["detdata/signal"].dtype == np.float32


# ── Generator hook ────────────────────────────────────────────────────────────


@pytest.fixture
def gen(tmp_path, monkeypatch):
    import sample_based_tod_generation_gridint as gen

    monkeypatch.setattr(gen, "folder_scan", str(tmp_path / "scan") + "/")
    # conftest stubs tod_io (it needs pixell); read the scan files directly.
    monkeypatch.setattr(
        gen,
        "open_scan_day",
        lambda folder, day: tuple(
            np.load(os.path.join(folder, f"{k}_{day}.npy"))
            for k in ("theta", "phi", "psi")
        ),
    )
    monkeypatch.setattr(gen, "folder_tod_output", str(tmp_path / "out"))
    (tmp_path / "out").mkdir()
    return gen


def test_day_output_export_returns_signal(gen, tmp_path, monkeypatch):
    _, _, theta, phi, psi, iqu = _write_inputs(tmp_path)
    monkeypatch.setattr(gen.config, "furax_export", True)
    signal = gen._day_output(0, iqu)
    npt.assert_allclose(signal, _combine_iqu_to_signal(iqu, psi))
    assert not os.listdir(tmp_path / "out")


def test_day_output_npy_when_export_off(gen, tmp_path, monkeypatch):
    _, _, _, _, _, iqu = _write_inputs(tmp_path)
    monkeypatch.setattr(gen.config, "furax_export", False)
    assert gen._day_output(0, iqu) is None
    npt.assert_array_equal(np.load(tmp_path / "out" / "tod_day_0.npy"), iqu)


def test_write_observation_matches_convert_day(gen, tmp_path, monkeypatch):
    """The generator's in-memory path writes the same observation as the
    standalone re-export of the equivalent .npy."""
    scan, tod, theta, phi, psi, iqu = _write_inputs(tmp_path)
    monkeypatch.setattr(gen.config, "hwp_enabled", False)
    monkeypatch.setattr(gen.config, "precision_dtype", np.float64)
    gen._write_observation(tf, 0, _combine_iqu_to_signal(iqu, psi), _FSAMP, 0.0)
    mem = _read(tmp_path / "out" / "obs_day_0.h5")
    npy = _read(_convert(tmp_path, scan, tod)[0])
    for key in mem:
        npt.assert_array_equal(mem[key], npy[key])
