"""Export pipeline TODs to TOAST HDF5 observations for furax mapmaking.

Reads the per-day, per-detector TODs (``tod_day_{N}[_{name}].npy``, shape
``(3, n_samples)`` = [I, Q, U]) and the boresight scan
(``theta/phi/psi_{N}.npy``) and writes one observation per day with a
``(n_det, n_samples)`` detdata block, loadable via
``furax.interfaces.toast.ToastObservation.from_file``.

The focal plane is taken from the ``detectors:`` config
(:func:`tod_focalplane.load_detectors`): ``boresight_radec`` holds the shared
boresight pointing and each detector carries its offset quaternion, so furax
reconstructs per-detector pointing as the generator composed it. The stored
detector signal is ``I + Q cos(2 psi_d) + U sin(2 psi_d)`` evaluated at the
detector's polarization angle ``psi_d`` (the roll the generator used to define
that detector's Q/U frame).

Run in a toast-aware environment, e.g.
    micromamba run -n beam_main python tod_to_furax.py
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
    from toast.instrument import Focalplane, SpaceSite, Telescope
    from toast.observation import Session, default_values as defaults
    from toast.ops.save_hdf5 import SaveHDF5
    from astropy.table import QTable
except ImportError as exc:
    raise SystemExit(
        "toast is required for this conversion script. Install in the active env, "
        "e.g.  `micromamba run -n beam_main pip install toast`."
    ) from exc

import tod_config as config
from tod_io import load_scan_information
from tod_focalplane import (
    combine_detector_signal,
    load_detectors,
    tod_output_path,
    _combine_iqu_to_signal,
)


def _angles_to_boresight_radec_quats(
    theta: np.ndarray, phi: np.ndarray, psi: np.ndarray
) -> np.ndarray:
    """Boresight quaternions (TOAST scalar-last) from ISO angles (theta, phi, psi).

    theta is colatitude, phi longitude (celestial/equatorial, used as RA/Dec),
    psi the polarization roll about the line of sight.
    """
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    phi = np.ascontiguousarray(phi, dtype=np.float64)
    psi = np.ascontiguousarray(psi, dtype=np.float64)
    return np.asarray(qa.from_iso_angles(theta, phi, psi), dtype=np.float64)


def _detector_quat_scalar_last(detector) -> np.ndarray:
    """TOAST scalar-last ``(x, y, z, w)`` offset quaternion for a detector.

    The implicit boresight detector (``quat is None``) maps to the identity.
    Configured detectors carry a scalar-first ``(w, x, y, z)`` furax-convention
    quaternion; reorder it to TOAST's scalar-last storage for the focalplane
    table.
    """
    if detector.quat is None:
        return np.array([0.0, 0.0, 0.0, 1.0])
    w, x, y, z = (float(v) for v in detector.quat)
    return np.array([x, y, z, w])


def _make_focalplane(sample_rate_hz: float, detectors) -> Focalplane:
    """Build a focal plane QTable with one row per detector.

    Args:
        sample_rate_hz (float): Sampling rate.
        detectors (list[Detector]): Focal-plane detectors (from
            :func:`tod_focalplane.load_detectors`).
    """
    n_det = len(detectors)
    names = np.array([d.name for d in detectors], dtype="U64")
    quats = np.array([_detector_quat_scalar_last(d) for d in detectors])
    gammas = np.array([float(d.gamma) for d in detectors])
    zeros = np.zeros(n_det)
    ones = np.ones(n_det)
    det_table = QTable(
        [
            names,
            quats,
            gammas * u.rad,  # gamma (pol angle offset)
            zeros * u.rad,  # psi_pol
            zeros * u.rad,  # psi_uv
            zeros * u.rad,  # alpha
            ones,  # pol_efficiency
            ones * u.dimensionless_unscaled,  # fwhm placeholder
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
    return Focalplane(
        detector_data=det_table,
        sample_rate=sample_rate_hz * u.Hz,
    )


def _build_observation(
    day_index: int,
    times: np.ndarray,
    signal: np.ndarray,
    det_names: list[str],
    boresight_quats: np.ndarray,
    hwp_angles: np.ndarray | None,
    theta: np.ndarray,
    phi: np.ndarray,
    focalplane: Focalplane,
    telescope_name: str = "tod_gen",
    name: str | None = None,
    signal_dtype: np.dtype = np.float32,
) -> toast.Observation:
    """Build a multi-detector toast.Observation populated with our buffers.

    ``signal`` has shape ``(n_det, n_samples)`` with rows ordered to match
    ``det_names`` (which in turn match the focalplane table row order).
    """
    telescope = Telescope(
        name=telescope_name, focalplane=focalplane, site=SpaceSite(name="L2")
    )
    n_samples = signal.shape[1]

    t_start = datetime.fromtimestamp(float(times[0]), tz=timezone.utc)
    t_end = datetime.fromtimestamp(float(times[-1]), tz=timezone.utc)
    session = Session(name=f"session_day_{day_index}", start=t_start, end=t_end)

    obs = toast.Observation(
        comm=toast.Comm(),
        telescope=telescope,
        n_samples=n_samples,
        name=name if name is not None else f"obs_day_{day_index}",
        session=session,
    )
    # LoadHDF5 requires at least one observation-level metadata entry.
    obs["source"] = "tod_from_beam_generation"
    obs["day_index"] = int(day_index)

    # Shared columns (per-sample, common to all detectors).
    obs.shared.create_column(defaults.times, shape=(n_samples,), dtype=np.float64)
    obs.shared[defaults.times].set(times, offset=(0,), fromrank=0)

    obs.shared.create_column(
        defaults.boresight_radec, shape=(n_samples, 4), dtype=np.float64
    )
    obs.shared[defaults.boresight_radec].set(boresight_quats, offset=(0, 0), fromrank=0)

    # hwp_angle must be the actual pipeline HWP phase (zeros when HWP is off):
    # furax's forward model reproduces the stored signal only at this phase.
    if hwp_angles is None:
        hwp_angles = np.zeros(n_samples, dtype=np.float64)
    obs.shared.create_column(defaults.hwp_angle, shape=(n_samples,), dtype=np.float64)
    obs.shared[defaults.hwp_angle].set(
        hwp_angles.astype(np.float64, copy=False), offset=(0,), fromrank=0
    )

    # Celestial-frame stand-ins so furax's get_azimuth/get_elevation getters do
    # not raise; these are NOT true ground-frame az/el.
    obs.shared.create_column(defaults.azimuth, shape=(n_samples,), dtype=np.float64)
    obs.shared[defaults.azimuth].set(
        phi.astype(np.float64, copy=False), offset=(0,), fromrank=0
    )
    obs.shared.create_column(defaults.elevation, shape=(n_samples,), dtype=np.float64)
    obs.shared[defaults.elevation].set(
        (0.5 * np.pi - theta.astype(np.float64, copy=False)),
        offset=(0,),
        fromrank=0,
    )

    # detdata dtype must match furax's double_precision (float32 / float64).
    obs.detdata.create(defaults.det_data, dtype=signal_dtype, units=u.K)
    obs.detdata.create(defaults.det_flags, dtype=np.uint8)
    for di, det_name in enumerate(det_names):
        obs.detdata[defaults.det_data][det_name, :] = signal[di].astype(
            signal_dtype, copy=False
        )
        obs.detdata[defaults.det_flags][det_name, :] = 0

    obs.shared.create_column(defaults.shared_flags, shape=(n_samples,), dtype=np.uint8)
    obs.shared[defaults.shared_flags].set(
        np.zeros(n_samples, dtype=np.uint8), offset=(0,), fromrank=0
    )
    return obs


def write_day_observation(
    day_index: int,
    signal: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    detectors,
    folder_out: str,
    fsamp: float,
    t0_unix: float,
    hwp_enabled: bool,
    f_hwp: float,
    phi0_hwp: float,
    n_splits: int = 1,
    signal_dtype: np.dtype = np.float32,
) -> list[Path]:
    """Write one day's assembled ``(n_det, n_samples)`` signal as TOAST HDF5.

    The boresight pointing ``(theta, phi, psi)`` is supplied directly so the
    caller (either :func:`convert_day` reading npy, or the generator holding the
    TOD in memory) need not round-trip the signal through disk. One observation
    is written per day, optionally split into ``n_splits`` parts.

    Args:
        signal (numpy.ndarray): ``(n_det, n_samples)`` per-detector scalar
            timestreams, row order matching ``detectors``.
        detectors (list[Detector]): focal plane (defines the HDF5 focalplane).

    Returns:
        list[pathlib.Path]: the written ``.h5`` paths.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}.")
    n_samples = signal.shape[1]
    if not (theta.shape == phi.shape == psi.shape == (n_samples,)):
        raise ValueError(
            f"Pointing/signal length mismatch for day {day_index}: "
            f"signal n={n_samples}, theta={theta.shape}, phi={phi.shape}, "
            f"psi={psi.shape}."
        )
    det_names = [d.name for d in detectors]

    # Continuous timestamps across days, matching the pipeline's HWP time
    # origin (t = day_index*86400 + i/fsamp).
    dt = 1.0 / fsamp
    t_rel = day_index * 86400.0 + np.arange(n_samples) * dt
    times = t0_unix + t_rel

    hwp_angles = None
    if hwp_enabled:
        hwp_angles = (2.0 * np.pi * f_hwp * t_rel + phi0_hwp).astype(np.float64)

    boresight_quats_full = _angles_to_boresight_radec_quats(theta, phi, psi)

    focalplane = _make_focalplane(sample_rate_hz=fsamp, detectors=detectors)

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

    split_indices = np.array_split(np.arange(n_samples), n_splits)
    written: list[Path] = []
    for part_index, idx in enumerate(split_indices):
        if idx.size == 0:
            continue
        s = slice(int(idx[0]), int(idx[-1]) + 1)
        if n_splits == 1:
            obs_name = f"obs_day_{day_index}"
        else:
            obs_name = f"obs_day_{day_index}_p{part_index}"

        obs = _build_observation(
            day_index=day_index,
            times=times[s],
            signal=signal[:, s],
            det_names=det_names,
            boresight_quats=boresight_quats_full[s],
            hwp_angles=None if hwp_angles is None else hwp_angles[s],
            theta=theta[s],
            phi=phi[s],
            focalplane=focalplane,
            name=obs_name,
            signal_dtype=signal_dtype,
        )

        data = toast.Data()
        data.obs.append(obs)

        # Drop any stale row for this observation from SaveHDF5's SQLite index
        # so re-runs don't hit a UNIQUE constraint error.
        if index_path.exists():
            with sqlite3.connect(str(index_path)) as conn:
                conn.execute("DELETE FROM observations WHERE name = ?", (obs_name,))
                conn.commit()

        saver = SaveHDF5(
            volume=str(out_dir),
            detdata=[defaults.det_data, defaults.det_flags],
            shared=shared_keys,
            intervals=[],
            force_serial=True,
        )
        saver.apply(data)
        written.append(out_dir / f"{obs_name}.h5")

    return written


def convert_day(
    day_index: int,
    folder_scan: str,
    folder_tod: str,
    folder_out: str,
    fsamp: float,
    t0_unix: float,
    hwp_enabled: bool,
    f_hwp: float,
    phi0_hwp: float,
    detectors=None,
    n_splits: int = 1,
    signal_dtype: np.dtype = np.float32,
) -> list[Path]:
    """Convert one day's per-detector TOD ``.npy`` files into TOAST HDF5.

    Standalone entry: reads each detector's ``tod_day_{N}[_{name}].npy``,
    combines to a scalar timestream, and writes the observation. The integrated
    generator path skips the ``.npy`` round-trip and calls
    :func:`write_day_observation` directly with the in-memory TODs.

    Args:
        detectors (list[Detector] | None): Focal plane to export. ``None``
            loads it from config via :func:`tod_focalplane.load_detectors`.
    """
    if detectors is None:
        detectors = load_detectors()

    theta = np.load(Path(folder_scan) / f"theta_{day_index}.npy").astype(np.float64)
    phi = np.load(Path(folder_scan) / f"phi_{day_index}.npy").astype(np.float64)
    psi = np.load(Path(folder_scan) / f"psi_{day_index}.npy").astype(np.float64)
    n_samples = psi.shape[0]
    if not (theta.shape == phi.shape == psi.shape == (n_samples,)):
        raise ValueError(
            f"Boresight pointing length mismatch for day {day_index}: "
            f"theta={theta.shape}, phi={phi.shape}, psi={psi.shape}."
        )

    signal = np.empty((len(detectors), n_samples), dtype=np.float64)
    for di, det in enumerate(detectors):
        tod_path = Path(tod_output_path(folder_tod, day_index, det))
        iqu = np.load(tod_path)
        if iqu.ndim != 2 or iqu.shape != (3, n_samples):
            raise ValueError(
                f"Unexpected TOD shape {iqu.shape} in {tod_path}; "
                f"expected (3, {n_samples})."
            )
        signal[di] = combine_detector_signal(iqu, theta, phi, psi, det)

    return write_day_observation(
        day_index=day_index,
        signal=signal,
        theta=theta,
        phi=phi,
        psi=psi,
        detectors=detectors,
        folder_out=folder_out,
        fsamp=fsamp,
        t0_unix=t0_unix,
        hwp_enabled=hwp_enabled,
        f_hwp=f_hwp,
        phi0_hwp=phi0_hwp,
        n_splits=n_splits,
        signal_dtype=signal_dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help="Output folder for HDF5 observations "
        "(default: <FOLDER_TOD_OUTPUT>/furax_h5/)",
    )
    parser.add_argument("--start-day", type=int, default=None)
    parser.add_argument("--end-day", type=int, default=None)
    parser.add_argument(
        "--t0",
        default=datetime(2030, 1, 1, tzinfo=timezone.utc).isoformat(),
        help="ISO-8601 UTC timestamp of sample 0 of day 0 "
        "(default: 2030-01-01T00:00:00+00:00)",
    )
    parser.add_argument(
        "--split-per-day",
        type=int,
        default=1,
        help="Emit this many smaller observations per day (default: 1). "
        "Larger values reduce the per-step working set inside furax's "
        "`accumulate_rhs` scan and lower GPU memory pressure; in exchange "
        "the scan runs over more (smaller) steps.",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float32",
        help="dtype for the stored detector signal (default: float32). "
        "Must match furax's `double_precision`: float32 when "
        "`double_precision=False`, float64 when `double_precision=True`. "
        "A mismatch raises a JAX cotangent dtype error in `H.T`.",
    )
    args = parser.parse_args()

    Nb, fsamp = load_scan_information(config.FOLDER_SCAN)
    start = max(args.start_day or config.start_day or 0, 0)
    end = min(args.end_day or config.end_day or Nb, Nb)

    out_folder = args.output or os.path.join(config.FOLDER_TOD_OUTPUT, "furax_h5")
    t0_unix = datetime.fromisoformat(args.t0).timestamp()

    detectors = load_detectors()

    print(f"Converting days [{start}, {end}) -> {out_folder}")
    print(f"  fsamp = {fsamp:.6f} Hz, n_days_total = {Nb}")
    print(f"  hwp_enabled = {config.hwp_enabled}")
    print(f"  detectors = {[d.name for d in detectors]}")

    signal_dtype = np.float32 if args.precision == "float32" else np.float64
    print(f"  signal dtype = {np.dtype(signal_dtype).name}")

    for day in range(start, end):
        out_paths = convert_day(
            day_index=day,
            folder_scan=config.FOLDER_SCAN,
            folder_tod=config.FOLDER_TOD_OUTPUT,
            folder_out=out_folder,
            fsamp=fsamp,
            t0_unix=t0_unix,
            hwp_enabled=config.hwp_enabled,
            f_hwp=config.hwp_rotation_frequency_hz,
            phi0_hwp=config.hwp_initial_phase_rad,
            detectors=detectors,
            n_splits=args.split_per_day,
            signal_dtype=signal_dtype,
        )
        for out_path in out_paths:
            print(f"  day {day}: wrote {out_path}")


if __name__ == "__main__":
    main()
