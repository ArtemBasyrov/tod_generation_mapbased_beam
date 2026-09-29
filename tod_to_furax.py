"""Export pipeline TODs as TOAST HDF5 observations for furax mapmaking.

One observation is written per day, loadable with
``furax.interfaces.toast.ToastObservation.from_file``. It holds a single
detector at boresight, whose signal is

    d[t] = I[t] + Q[t] cos(2 psi[t]) + U[t] sin(2 psi[t])

in the TOAST temperature convention (furax applies the polariser's 1/2 on
read). The generator calls :func:`write_day_observation` with the TOD held in
memory; this module's command line re-exports existing ``tod_day_<N>.npy``
files through :func:`convert_day`:

    micromamba run -n beam_main python tod_to_furax.py [--output DIR] ...

Requires toast, astropy and h5py.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import toast
    import toast.qarray as qa
    from astropy import units as u
    from astropy.table import QTable
    from toast.instrument import Focalplane, SpaceSite, Telescope
    from toast.observation import Session, default_values as defaults
    from toast.ops.save_hdf5 import SaveHDF5
except ImportError as exc:
    raise ImportError(
        "The furax export needs toast: install it in the active environment "
        "(`pip install toast`), or set `furax_export: false`."
    ) from exc

import tod_config as config
from tod_io import load_scan_information
from tod_pipeline_helpers import _combine_iqu_to_signal, _hwp_angle


_DET_NAME = "boresight"  # one detector at boresight, gamma = 0


def _angles_to_boresight_radec_quats(theta, phi, psi):
    """TOAST scalar-last boresight quaternions from ISO angles (theta, phi, psi).

    theta is colatitude and phi longitude in the celestial frame used as
    RA/Dec; psi is the polarisation roll about the line of sight.
    """
    return np.asarray(
        qa.from_iso_angles(
            np.ascontiguousarray(theta, dtype=np.float64),
            np.ascontiguousarray(phi, dtype=np.float64),
            np.ascontiguousarray(psi, dtype=np.float64),
        ),
        dtype=np.float64,
    )


def _check_finite(day_index, **buffers):
    """Reject an observation carrying any NaN or inf, reporting where it is.

    A single non-finite sample enters furax's CG through the global inner
    product, pins every step length to zero and returns the zero map after the
    full iteration cap. Catching it here turns that silent GPU failure into an
    immediate error.

    Args:
        day_index (int): Day the buffers belong to, for the message.
        **buffers (numpy.ndarray | None): Arrays to check; ``None`` is skipped.

    Raises:
        ValueError: If any buffer holds a non-finite value.
    """
    bad = []
    for name, buf in buffers.items():
        if buf is None:
            continue
        arr = np.asarray(buf)
        per_sample = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr[:, None]
        idx = np.flatnonzero(~np.isfinite(per_sample).all(axis=1))
        if idx.size:
            bad.append(f"{name}: {idx.size} sample(s), first at {idx[:5].tolist()}")
    if bad:
        raise ValueError(
            f"Non-finite values in day {day_index} before HDF5 export -- "
            "refusing to write. " + "; ".join(bad) + ". Check the scan-strategy "
            "files (theta/phi/psi) and the generated TOD for this day."
        )


def _make_focalplane(sample_rate_hz):
    """Single-detector focal plane: one detector at boresight, gamma = 0."""
    det_table = QTable(
        [
            np.array([_DET_NAME], dtype="U64"),
            np.array([[0.0, 0.0, 0.0, 1.0]]),  # identity, scalar-last
            np.array([0.0]) * u.rad,  # gamma
            np.array([0.0]) * u.rad,  # psi_pol
            np.array([0.0]) * u.rad,  # psi_uv
            np.array([0.0]) * u.rad,  # alpha
            np.array([1.0]),  # pol_efficiency
            np.array([1.0]) * u.dimensionless_unscaled,  # fwhm placeholder
        ],
        names=[
            "name",
            "quat",
            "gamma",
            "psi_pol",
            "psi_uv",
            "alpha",
            "pol_efficiency",
            "fwhm",
        ],
    )
    return Focalplane(detector_data=det_table, sample_rate=sample_rate_hz * u.Hz)


def _build_observation(
    day_index,
    name,
    times,
    signal,
    boresight_quats,
    hwp_angles,
    theta,
    phi,
    focalplane,
    signal_dtype,
):
    """Build a single-detector ``toast.Observation`` holding one day's buffers."""
    n_samples = signal.shape[0]
    telescope = Telescope(
        name="tod_gen", focalplane=focalplane, site=SpaceSite(name="L2")
    )
    session = Session(
        name=f"session_day_{day_index}",
        start=datetime.fromtimestamp(float(times[0]), tz=timezone.utc),
        end=datetime.fromtimestamp(float(times[-1]), tz=timezone.utc),
    )
    obs = toast.Observation(
        comm=toast.Comm(),
        telescope=telescope,
        n_samples=n_samples,
        name=name,
        session=session,
    )
    # LoadHDF5 needs at least one observation-level metadata entry.
    obs["source"] = "tod_from_beam_generation"
    obs["day_index"] = int(day_index)

    obs.shared.create_column(defaults.times, shape=(n_samples,), dtype=np.float64)
    obs.shared[defaults.times].set(times, offset=(0,), fromrank=0)

    obs.shared.create_column(
        defaults.boresight_radec, shape=(n_samples, 4), dtype=np.float64
    )
    obs.shared[defaults.boresight_radec].set(boresight_quats, offset=(0, 0), fromrank=0)

    # furax always reads hwp_angle, and reproduces the stored signal only at the
    # generator's own HWP phase (zeros when the HWP is off).
    if hwp_angles is None:
        hwp_angles = np.zeros(n_samples, dtype=np.float64)
    obs.shared.create_column(defaults.hwp_angle, shape=(n_samples,), dtype=np.float64)
    obs.shared[defaults.hwp_angle].set(hwp_angles, offset=(0,), fromrank=0)

    # Celestial stand-ins so furax's get_azimuth/get_elevation do not raise;
    # these are not ground-frame az/el.
    obs.shared.create_column(defaults.azimuth, shape=(n_samples,), dtype=np.float64)
    obs.shared[defaults.azimuth].set(
        np.asarray(phi, dtype=np.float64), offset=(0,), fromrank=0
    )
    obs.shared.create_column(defaults.elevation, shape=(n_samples,), dtype=np.float64)
    obs.shared[defaults.elevation].set(
        0.5 * np.pi - np.asarray(theta, dtype=np.float64), offset=(0,), fromrank=0
    )

    # detdata dtype must match furax's double_precision (float32 / float64).
    obs.detdata.create(defaults.det_data, dtype=signal_dtype, units=u.K)
    obs.detdata[defaults.det_data][_DET_NAME, :] = signal.astype(
        signal_dtype, copy=False
    )
    obs.detdata.create(defaults.det_flags, dtype=np.uint8)
    obs.detdata[defaults.det_flags][_DET_NAME, :] = 0

    obs.shared.create_column(defaults.shared_flags, shape=(n_samples,), dtype=np.uint8)
    obs.shared[defaults.shared_flags].set(
        np.zeros(n_samples, dtype=np.uint8), offset=(0,), fromrank=0
    )
    return obs


def observation_paths(folder_out, day_index, n_splits=1):
    """HDF5 paths :func:`write_day_observation` writes for one day.

    Args:
        folder_out (str): Output folder.
        day_index (int): Observation day.
        n_splits (int): Number of observations the day is split into.

    Returns:
        list[pathlib.Path]: ``obs_day_<N>.h5``, or ``obs_day_<N>_p<k>.h5`` for
            each part when ``n_splits > 1``.
    """
    if n_splits == 1:
        names = [f"obs_day_{day_index}"]
    else:
        names = [f"obs_day_{day_index}_p{k}" for k in range(n_splits)]
    return [Path(folder_out) / f"{n}.h5" for n in names]


def write_day_observation(
    day_index,
    signal,
    theta,
    phi,
    psi,
    folder_out,
    fsamp,
    t0_unix,
    hwp_enabled,
    f_hwp,
    phi0_hwp,
    n_splits=1,
    signal_dtype=np.float64,
):
    """Write one day's detector signal as TOAST HDF5 observation(s).

    Args:
        day_index (int): Observation day.
        signal (numpy.ndarray): ``(n,)`` detector timestream, from
            :func:`tod_pipeline_helpers._combine_iqu_to_signal`.
        theta, phi, psi (numpy.ndarray): ``(n,)`` boresight pointing [rad].
        folder_out (str): Output folder; created if absent.
        fsamp (float): Sample rate [Hz].
        t0_unix (float): Unix time of sample 0 of day 0.
        hwp_enabled (bool): Whether the generator applied HWP modulation.
        f_hwp (float): HWP rotation frequency [Hz].
        phi0_hwp (float): HWP phase at t = 0 [rad].
        n_splits (int): Split the day into this many observations, which lowers
            furax's per-step GPU working set.
        signal_dtype (numpy.dtype): Stored signal dtype; must match furax's
            ``double_precision``.

    Returns:
        list[pathlib.Path]: The written ``.h5`` paths.

    Raises:
        ValueError: On a length mismatch, ``n_splits < 1`` or a non-finite
            buffer.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}.")
    n_samples = signal.shape[0]
    if not (theta.shape == phi.shape == psi.shape == (n_samples,)):
        raise ValueError(
            f"Pointing/signal length mismatch for day {day_index}: "
            f"signal={signal.shape}, theta={theta.shape}, phi={phi.shape}, "
            f"psi={psi.shape}."
        )

    # Same time origin as the generator's HWP phase: t = day*86400 + i/fsamp.
    t_rel = day_index * 86400.0 + np.arange(n_samples) * (1.0 / fsamp)
    times = t0_unix + t_rel

    hwp_angles = None
    if hwp_enabled:
        # Bit-identical to the angle the generator modulated with. It is already
        # wrapped to [0, 2pi), which furax needs: it casts this column to its run
        # dtype, and an unwrapped ~1e8 rad phase collapses to a few float32 values.
        hwp_angles = _hwp_angle(day_index, 0, n_samples, fsamp, f_hwp, phi0_hwp)

    boresight_quats = _angles_to_boresight_radec_quats(theta, phi, psi)
    _check_finite(
        day_index,
        signal=signal,
        boresight_quats=boresight_quats,
        times=times,
        hwp_angles=hwp_angles,
    )
    focalplane = _make_focalplane(fsamp)

    out_dir = Path(folder_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.sqlite"
    shared_keys = [
        defaults.times,
        defaults.boresight_radec,
        defaults.shared_flags,
        defaults.azimuth,
        defaults.elevation,
        defaults.hwp_angle,
    ]

    written = []
    parts = np.array_split(np.arange(n_samples), n_splits)
    for path, idx in zip(observation_paths(out_dir, day_index, n_splits), parts):
        if idx.size == 0:
            continue
        s = slice(int(idx[0]), int(idx[-1]) + 1)
        obs = _build_observation(
            day_index,
            path.stem,
            times[s],
            signal[s],
            boresight_quats[s],
            None if hwp_angles is None else hwp_angles[s],
            theta[s],
            phi[s],
            focalplane,
            signal_dtype,
        )
        data = toast.Data()
        data.obs.append(obs)

        # SaveHDF5 keeps a SQLite index with a UNIQUE observation name; drop a
        # stale row so a re-run can overwrite the day.
        if index_path.exists():
            with sqlite3.connect(str(index_path)) as conn:
                conn.execute("DELETE FROM observations WHERE name = ?", (path.stem,))
                conn.commit()

        SaveHDF5(
            volume=str(out_dir),
            detdata=[defaults.det_data, defaults.det_flags],
            shared=shared_keys,
            intervals=[],
            force_serial=True,
        ).apply(data)
        written.append(path)
    return written


def convert_day(
    day_index,
    folder_scan,
    folder_tod,
    folder_out,
    fsamp,
    t0_unix,
    hwp_enabled,
    f_hwp,
    phi0_hwp,
    n_splits=1,
    signal_dtype=np.float64,
):
    """Re-export one day's ``tod_day_<N>.npy`` as TOAST HDF5.

    Args:
        day_index (int): Observation day.
        folder_scan (str): Folder holding ``theta/phi/psi_<N>.npy``.
        folder_tod (str): Folder holding ``tod_day_<N>.npy``, shape ``(3, n)``.
        folder_out, fsamp, t0_unix, hwp_enabled, f_hwp, phi0_hwp, n_splits,
            signal_dtype: As in :func:`write_day_observation`.

    Returns:
        list[pathlib.Path]: The written ``.h5`` paths.

    Raises:
        ValueError: If the TOD is not ``(3, n)``.
    """
    tod_path = Path(folder_tod) / f"tod_day_{day_index}.npy"
    iqu = np.load(tod_path)
    if iqu.ndim != 2 or iqu.shape[0] != 3:
        raise ValueError(
            f"Unexpected TOD shape {iqu.shape} in {tod_path}; expected (3, n)."
        )
    theta, phi, psi = (
        np.load(Path(folder_scan) / f"{k}_{day_index}.npy").astype(np.float64)
        for k in ("theta", "phi", "psi")
    )
    return write_day_observation(
        day_index,
        _combine_iqu_to_signal(iqu, psi),
        theta,
        phi,
        psi,
        folder_out,
        fsamp,
        t0_unix,
        hwp_enabled,
        f_hwp,
        phi0_hwp,
        n_splits=n_splits,
        signal_dtype=signal_dtype,
    )


def main():
    """Command-line re-export of existing ``tod_day_<N>.npy`` files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help="Output folder (default: <FOLDER_TOD_OUTPUT>/furax_h5/).",
    )
    parser.add_argument("--start-day", type=int, default=None)
    parser.add_argument("--end-day", type=int, default=None)
    parser.add_argument(
        "--t0",
        default=config.furax_export_t0,
        help="ISO-8601 UTC time of sample 0 of day 0 (default: furax_export_t0).",
    )
    parser.add_argument(
        "--split-per-day",
        type=int,
        default=1,
        help="Observations per day (default: 1). More, smaller observations "
        "lower furax's per-step GPU working set.",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float64",
        help="Stored signal dtype (default: float64). Must match furax's "
        "double_precision, or JAX raises a cotangent dtype error in H.T.",
    )
    args = parser.parse_args()

    Nb, fsamp = load_scan_information(config.FOLDER_SCAN)
    start = max(args.start_day or config.start_day or 0, 0)
    end = min(args.end_day or config.end_day or Nb, Nb)
    out_folder = args.output or os.path.join(config.FOLDER_TOD_OUTPUT, "furax_h5")
    t0_unix = datetime.fromisoformat(args.t0).timestamp()
    signal_dtype = np.dtype(args.precision)

    print(f"Converting days [{start}, {end}) -> {out_folder}")
    print(f"  fsamp = {fsamp:.6f} Hz, hwp_enabled = {config.hwp_enabled}")
    print(f"  signal dtype = {signal_dtype.name}")
    for day in range(start, end):
        for path in convert_day(
            day,
            config.FOLDER_SCAN,
            config.FOLDER_TOD_OUTPUT,
            out_folder,
            fsamp,
            t0_unix,
            config.hwp_enabled,
            config.hwp_rotation_frequency_hz,
            config.hwp_initial_phase_rad,
            n_splits=args.split_per_day,
            signal_dtype=signal_dtype,
        ):
            print(f"  day {day}: wrote {path}")


if __name__ == "__main__":
    main()
