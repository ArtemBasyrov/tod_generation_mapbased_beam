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
from typing import List, Optional

import numpy as np

import tod_config as config


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
    """

    name: str
    quat: Optional[np.ndarray]
    gamma: float

    @property
    def is_boresight(self) -> bool:
        """True for the implicit boresight detector (no offset transform)."""
        return self.quat is None


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
            Detector(name=entry["name"], quat=from_xieta(xi, eta, gamma), gamma=gamma)
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
