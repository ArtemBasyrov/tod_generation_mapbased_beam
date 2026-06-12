"""Run-scoped execution context for the TOD generator.

A single frozen :class:`RunContext` bundles the parameters that stay constant
for an entire run — folder paths, interpolation mode, nside, batch size,
precision, the spin-2 skip threshold and the HWP settings. It is built once in
``main()`` and passed to the worker pool, where it is pickled to each worker a
single time at start-up. It carries only small scalars; the large sky-map and
beam arrays travel separately via POSIX shared memory.

Anything that varies per task — the observation-day index — stays an explicit
function argument rather than living on the context.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import tod_config as config


@dataclass(frozen=True)
class RunContext:
    """Immutable bundle of run-scoped parameters for TOD generation.

    Attributes
    ----------
    folder_scan : str
        Directory holding the per-day boresight scan files. Must end with a
        path separator.
    folder_tod_output : str
        Directory the per-day TOD ``.npy`` files are written to.
    interp_mode : str
        Sky-map interpolation strategy (``'bilinear'`` or ``'nearest'``).
    nside : int
        HEALPix nside of the input sky map.
    batch_size : int
        Number of boresight samples processed per kernel batch.
    z_skip_threshold : float
        Spin-2 Q/U rotation-skip cutoff on ``|cos θ|`` (``-1.0`` disables it).
    fsamp : float
        Sample rate [samples/s], used for the HWP phase clock.
    precision_dtype : np.dtype
        Working precision for the float32-side of the pipeline (sky map,
        pointings, beam values, saved TOD).
    beam_center_idx : tuple[int, int] | None
        Beam-centre pixel (row, col) override, or ``None`` for the array centre.
    hwp_enabled : bool
        Whether to apply ideal continuously-rotating HWP modulation to Q/U.
    hwp_freq_hz : float
        HWP physical rotation frequency [Hz].
    hwp_phi0_rad : float
        Initial HWP phase at t=0 [rad].
    """

    folder_scan: str
    folder_tod_output: str
    interp_mode: str
    nside: int
    batch_size: int
    z_skip_threshold: float
    fsamp: float
    precision_dtype: np.dtype
    beam_center_idx: Optional[Tuple[int, int]]
    hwp_enabled: bool
    hwp_freq_hz: float
    hwp_phi0_rad: float


def build_run_context(nside, batch_size, z_skip_threshold, fsamp):
    """Assemble a :class:`RunContext` from the loaded ``tod_config`` and the
    run-derived quantities computed in ``main()``.

    Args:
        nside (int): HEALPix nside of the input sky map.
        batch_size (int): Calibrated (or cached) batch size.
        z_skip_threshold (float): Spin-2 skip cutoff from
            :func:`resolve_spin2_skip_threshold`.
        fsamp (float): Sample rate from :func:`load_scan_information`.

    Returns:
        RunContext: the immutable run context.
    """
    _cx, _cy = config.beam_center_x, config.beam_center_y
    beam_center_idx = (_cx, _cy) if (_cx is not None and _cy is not None) else None
    return RunContext(
        folder_scan=config.FOLDER_SCAN,
        folder_tod_output=config.FOLDER_TOD_OUTPUT,
        interp_mode=config.beam_interp_method,
        nside=int(nside),
        batch_size=int(batch_size),
        z_skip_threshold=float(z_skip_threshold),
        fsamp=float(fsamp),
        precision_dtype=config.precision_dtype,
        beam_center_idx=beam_center_idx,
        hwp_enabled=bool(config.hwp_enabled),
        hwp_freq_hz=float(config.hwp_rotation_frequency_hz),
        hwp_phi0_rad=float(config.hwp_initial_phase_rad),
    )
