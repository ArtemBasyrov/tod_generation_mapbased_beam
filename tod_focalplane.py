"""Focal-plane detectors and on-the-fly per-detector pointing.

The boresight scan files hold a single shared pointing stream ``(θ, φ, ψ)``.
A focal plane adds one or more detectors, each described by a static offset
from the boresight in the TOAST ``xi-eta-gamma`` convention. Rather than
storing an expanded scan per detector, the per-detector pointing is composed
on the fly from the boresight pointing and the detector's offset quaternion::

    q_b              = from_iso_angles(θ_b, φ_b, ψ_b)
    q_d              = qmul(q_b, q_det)          # furax PointingOperator order
    θ_d, φ_d, ψ_d    = to_iso_angles(q_d)

The quaternion algebra mirrors ``furax.math.quaternion`` exactly (scalar-first
``(w, x, y, z)`` storage, Hamilton product, ISO/xieta angle conventions) so the
generator and the furax mapmaker compose the *identical* per-detector
quaternions — pointing-model consistency holds by construction rather than by
matching two independent implementations.

All quaternion math here is float64; callers cast the resulting pointing to the
run precision afterwards.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import tod_config as config

# Stokes component index → config suffix used by the per-component beam knobs.
_COMP_SUFFIX = ("I", "Q", "U")


@dataclass(frozen=True)
class Detector:
    """A single focal-plane detector.

    Attributes:
        name (str): Detector label, used in output filenames.
        quat (numpy.ndarray | None): Static offset quaternion ``(w, x, y, z)``,
            float64, scalar-first. ``None`` marks the implicit boresight
            detector: its pointing is the boresight stream unchanged and no
            quaternion composition is performed (guaranteeing bit-identical,
            legacy-named output for single-detector runs).
        gamma (float): Polarization-angle offset [rad]. Folded into ``quat``
            for the pointing transform; retained here for the furax focal-plane
            table export.
        beam_key (str): Identifier of the beam set this detector convolves with.
            Detectors whose resolved beam files coincide share one key (and one
            set of beam arrays in shared memory). ``"default"`` is the global
            beam set used when no per-detector beam overrides are configured.
    """

    name: str
    quat: Optional[np.ndarray]
    gamma: float
    beam_key: str = "default"

    @property
    def is_boresight(self) -> bool:
        """True for the implicit boresight detector (no offset transform)."""
        return self.quat is None


@dataclass(frozen=True)
class BeamSpec:
    """Resolved per-detector beam files, thresholds and clustering overrides.

    A detector's beam set is fully described by the three per-component beam
    files and their power-fraction thresholds (with ``None`` for components not
    in ``map_fields``), plus optional beam-clustering overrides. Two detectors
    with equal :class:`BeamSpec` share a :attr:`Detector.beam_key`.

    ``n_clusters`` / ``tail_fraction`` are ``None`` when the detector does not
    override them; the generator then falls back to the global
    ``config.n_beam_clusters`` / ``config.beam_cluster_tail_fraction`` (which may
    themselves be set by clustering calibration) at clustering time. Storing the
    *override* rather than the resolved value keeps the ``"default"`` key stable
    across calibration.

    Attributes:
        beam_files (tuple): ``(beam_file_I, beam_file_Q, beam_file_U)``; entries
            may be ``None`` for inactive components.
        thresholds (tuple): ``(thr_I, thr_Q, thr_U)`` power-fraction thresholds;
            entries may be ``None`` for inactive components.
        n_clusters (int | None): Per-detector beam-cluster count override, or
            ``None`` to inherit the global value.
        tail_fraction (float | None): Per-detector clustering tail-fraction
            override, or ``None`` to inherit the global value.
    """

    beam_files: Tuple[Optional[str], Optional[str], Optional[str]]
    thresholds: Tuple[Optional[float], Optional[float], Optional[float]]
    n_clusters: Optional[int] = None
    tail_fraction: Optional[float] = None


def _global_beam_spec() -> BeamSpec:
    """The global (config-level) beam set used by detectors without overrides.

    ``n_clusters`` / ``tail_fraction`` are left ``None`` (meaning "inherit the
    global clustering config") so a detector that overrides nothing compares
    equal to this spec and gets the ``"default"`` key.
    """
    return BeamSpec(
        beam_files=(config.beam_file_I, config.beam_file_Q, config.beam_file_U),
        thresholds=(
            config.power_threshold_I,
            config.power_threshold_Q,
            config.power_threshold_U,
        ),
    )


def _resolve_beam_spec(entry: dict) -> BeamSpec:
    """Resolve a detector config entry's beam set, falling back to globals.

    Args:
        entry (dict): A cleaned ``config.detectors`` entry, optionally carrying
            ``beam_file_{I,Q,U}`` / ``power_fraction_threshold_{I,Q,U}`` and
            ``n_beam_clusters`` / ``beam_cluster_tail_fraction`` keys.

    Returns:
        BeamSpec: Per-component files/thresholds with global file/threshold
            defaults applied; clustering overrides kept as-is (``None`` →
            inherit the global clustering config at run time).
    """
    g = _global_beam_spec()
    files = tuple(
        entry.get(f"beam_file_{s}", None) or g.beam_files[i]
        for i, s in enumerate(_COMP_SUFFIX)
    )
    thresholds = tuple(
        entry.get(f"power_fraction_threshold_{s}", None)
        if entry.get(f"power_fraction_threshold_{s}", None) is not None
        else g.thresholds[i]
        for i, s in enumerate(_COMP_SUFFIX)
    )
    return BeamSpec(
        beam_files=files,
        thresholds=thresholds,
        n_clusters=entry.get("n_beam_clusters"),
        tail_fraction=entry.get("beam_cluster_tail_fraction"),
    )


def _beam_key(spec: BeamSpec) -> str:
    """Stable identifier for a beam set; ``"default"`` for the global spec."""
    if spec == _global_beam_spec():
        return "default"
    parts = [("" if f is None else str(f)) for f in spec.beam_files]
    parts += [("" if t is None else repr(t)) for t in spec.thresholds]
    parts += [repr(spec.n_clusters), repr(spec.tail_fraction)]
    return "beam:" + "|".join(parts)


def build_beam_specs(detectors: "List[Detector]") -> "Dict[str, BeamSpec]":
    """Map each detector's ``beam_key`` to its resolved :class:`BeamSpec`.

    Only beam sets actually referenced by ``detectors`` (after any
    ``detector_subset`` filtering) are included, so a sharded run loads only the
    beams it needs.

    Args:
        detectors (list[Detector]): The detectors returned by
            :func:`load_detectors` for this run.

    Returns:
        dict[str, BeamSpec]: ``beam_key`` → resolved beam files/thresholds.
            ``{"default": <global spec>}`` for a run without per-detector beams.
    """
    raw = config.detectors
    if not raw:
        return {"default": _global_beam_spec()}
    by_name = {e["name"]: e for e in raw}
    specs: Dict[str, BeamSpec] = {}
    for d in detectors:
        if d.beam_key not in specs:
            specs[d.beam_key] = _resolve_beam_spec(by_name[d.name])
    return specs


# ── Quaternion algebra (scalar-first (w, x, y, z), matches furax) ─────────────


def qmul(q1, q2):
    """Hamilton product of two quaternions (scalar-first), batched.

    Mirrors :func:`furax.math.quaternion.qmul`. Either argument may be a single
    quaternion ``(4,)`` or a batch ``(B, 4)``; standard broadcasting applies.

    Args:
        q1 (numpy.ndarray): Left quaternion(s), shape ``(..., 4)``.
        q2 (numpy.ndarray): Right quaternion(s), shape ``(..., 4)``.

    Returns:
        numpy.ndarray: Product quaternion(s), shape ``(..., 4)``, float64.
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def from_iso_angles(theta, phi, psi):
    """Quaternion(s) from ISO polar angles ``(θ, φ, ψ)``.

    Mirrors :func:`furax.math.quaternion.from_iso_angles`. Inputs broadcast
    against each other.

    Args:
        theta (array_like): Colatitude [rad].
        phi (array_like): Longitude [rad].
        psi (array_like): Polarization roll [rad].

    Returns:
        numpy.ndarray: Quaternion(s) ``(..., 4)``, scalar-first, float64.
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    cos_th = np.cos(theta * 0.5)
    sin_th = np.sin(theta * 0.5)
    cos_pp = np.cos((psi + phi) * 0.5)
    sin_pp = np.sin((psi + phi) * 0.5)
    cos_pm = np.cos((psi - phi) * 0.5)
    sin_pm = np.sin((psi - phi) * 0.5)
    return np.stack(
        [cos_th * cos_pp, sin_th * sin_pm, sin_th * cos_pm, cos_th * sin_pp],
        axis=-1,
    )


def to_iso_angles(q):
    """ISO polar angles ``(θ, φ, ψ)`` from quaternion(s).

    Mirrors :func:`furax.math.quaternion.to_iso_angles`.

    Args:
        q (numpy.ndarray): Quaternion(s) ``(..., 4)``, scalar-first.

    Returns:
        tuple: ``(theta, phi, psi)`` arrays [rad], each shape ``q.shape[:-1]``.
    """
    q = np.asarray(q, dtype=np.float64)
    a, b, c, d = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    theta = 2.0 * np.arctan2(np.sqrt(b**2 + c**2), np.sqrt(a**2 + d**2))
    phi = np.arctan2(c * d - a * b, a * c + b * d)
    psi = np.arctan2(c * d + a * b, a * c - b * d)
    return theta, phi, psi


def from_xieta(xi, eta, gamma):
    """Detector offset quaternion from TOAST ``xi-eta-gamma`` angles.

    Mirrors :func:`furax.math.quaternion.from_xieta_angles`.

    Args:
        xi (float): Cross-scan focal-plane offset [rad].
        eta (float): In-scan focal-plane offset [rad].
        gamma (float): Polarization-angle offset [rad].

    Returns:
        numpy.ndarray: Offset quaternion ``(4,)``, scalar-first, float64.
    """
    xi = float(xi)
    eta = float(eta)
    gamma = float(gamma)
    theta = np.arcsin(np.sqrt(xi**2 + eta**2))
    phi = np.arctan2(-xi, -eta)
    psi = gamma - phi
    return from_iso_angles(theta, phi, psi)


# ── Detector loading ─────────────────────────────────────────────────────────


def load_detectors():
    """Build the focal-plane detector list from :mod:`tod_config`.

    When no ``detectors:`` section is configured, returns a single implicit
    boresight detector (``quat=None``) so the pipeline reproduces the legacy
    single-detector behaviour bit-for-bit. Otherwise each YAML entry becomes a
    :class:`Detector` with its offset quaternion built from
    ``(xi_deg, eta_deg, gamma_deg)``. ``detector_subset`` (names or indices)
    filters the list, preserving the configured order.

    Returns:
        list[Detector]: The detectors to process this run.
    """
    raw = config.detectors
    if not raw:
        return [Detector(name="boresight", quat=None, gamma=0.0)]

    detectors = []
    for entry in raw:
        xi = np.deg2rad(entry["xi_deg"])
        eta = np.deg2rad(entry["eta_deg"])
        gamma = np.deg2rad(entry["gamma_deg"])
        detectors.append(
            Detector(
                name=entry["name"],
                quat=from_xieta(xi, eta, gamma),
                gamma=gamma,
                beam_key=_beam_key(_resolve_beam_spec(entry)),
            )
        )

    subset = config.detector_subset
    if subset:
        detectors = _select_subset(detectors, subset)
    return detectors


def _select_subset(detectors, subset):
    """Filter ``detectors`` by a ``detector_subset`` of names or indices.

    Args:
        detectors (list[Detector]): Full configured detector list.
        subset (list): Entries are detector names (str) or 0-based indices
            (int). Mixed types are allowed.

    Returns:
        list[Detector]: The selected detectors, in configured order.

    Raises:
        ValueError: If an entry names an unknown detector or an out-of-range
            index, or if the selection is empty.
    """
    by_name = {d.name: d for d in detectors}
    selected = {}
    for item in subset:
        if isinstance(item, str):
            if item not in by_name:
                raise ValueError(
                    f"detector_subset entry {item!r} is not a configured "
                    f"detector name; known names: {sorted(by_name)}"
                )
            det = by_name[item]
        else:
            idx = int(item)
            if not (0 <= idx < len(detectors)):
                raise ValueError(
                    f"detector_subset index {idx} out of range [0, {len(detectors)})"
                )
            det = detectors[idx]
        selected[det.name] = det
    if not selected:
        raise ValueError("detector_subset selected no detectors")
    # Preserve configured order, drop duplicates.
    return [d for d in detectors if d.name in selected]


# ── Per-detector pointing ────────────────────────────────────────────────────


def detector_pointing_batch(theta_b, phi_b, psi_b, q_det):
    """Compose per-detector pointing for a batch of boresight samples.

    Implements ``q_d = qmul(from_iso_angles(θ_b, φ_b, ψ_b), q_det)`` followed by
    ``to_iso_angles(q_d)``, matching furax's ``PointingOperator`` composition
    order so the detector's polarization offset ``γ`` is folded into ``ψ_d``
    automatically. All arithmetic is float64.

    Args:
        theta_b (array_like): Boresight colatitude [rad], shape ``(B,)``.
        phi_b (array_like): Boresight longitude [rad], shape ``(B,)``.
        psi_b (array_like): Boresight roll [rad], shape ``(B,)``.
        q_det (numpy.ndarray): Detector offset quaternion ``(4,)``.

    Returns:
        tuple: ``(theta_d, phi_d, psi_d)`` float64 arrays, each shape ``(B,)``.
    """
    q_b = from_iso_angles(theta_b, phi_b, psi_b)  # (B, 4)
    q_d = qmul(q_b, np.asarray(q_det, dtype=np.float64))  # broadcast (4,)
    return to_iso_angles(q_d)


def tod_output_path(folder, day_index, detector):
    """Per-detector, per-day output filename.

    The implicit boresight detector keeps the legacy ``tod_day_{N}.npy`` name;
    explicitly configured detectors get a ``tod_day_{N}_{name}.npy`` suffix.

    Args:
        folder (str): Output directory.
        day_index (int): Zero-based observation-day index.
        detector (Detector): The detector this TOD belongs to.

    Returns:
        str: Absolute or folder-relative output path.
    """
    import os

    if detector.is_boresight:
        fname = f"tod_day_{day_index}.npy"
    else:
        fname = f"tod_day_{day_index}_{detector.name}.npy"
    return os.path.join(folder, fname)


def _combine_iqu_to_signal(iqu, psi_d):
    """``I + Q cos(2 psi_d) + U sin(2 psi_d)`` for one detector, float64."""
    I, Q, U = np.asarray(iqu).astype(np.float64)
    c2 = np.cos(2.0 * np.asarray(psi_d).astype(np.float64))
    s2 = np.sin(2.0 * np.asarray(psi_d).astype(np.float64))
    return I + Q * c2 + U * s2


def combine_detector_signal(iqu, theta, phi, psi, detector):
    """Combine one detector's ``(3, n)`` [I, Q, U] TOD into its scalar timestream.

    The combination uses the detector's polarization angle ``psi_d`` — the roll
    of the per-detector pointing (``psi`` itself for the boresight detector). Q/U
    are already in that detector frame (and, under HWP, pre-rotated), so the
    stored signal is ``I + Q cos(2 psi_d) + U sin(2 psi_d)``.

    Args:
        iqu (numpy.ndarray): ``(3, n)`` [I, Q, U] TOD for this detector.
        theta, phi, psi (numpy.ndarray): boresight pointing ``(n,)`` [rad].
        detector (Detector): the detector (``quat is None`` → boresight).

    Returns:
        numpy.ndarray: scalar signal ``(n,)``, float64.
    """
    if detector.quat is None:
        psi_d = psi
    else:
        psi_d = detector_pointing_batch(theta, phi, psi, detector.quat)[2]
    return _combine_iqu_to_signal(iqu, psi_d)
