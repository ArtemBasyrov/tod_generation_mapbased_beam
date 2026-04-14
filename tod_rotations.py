"""
Rotation routines for sample-based TOD generation.

Numba JIT kernels
-----------------
_rodrigues_jit              — fused double Rodrigues rotation (recenter + pol. roll).
                              Writes directly into a pre-allocated (B, S, 3) buffer.

_spin2_rodrigues_cos2d_sin2d — cos(2δ) and sin(2δ) for spin-2 frame rotation via
                               inlined Rodrigues parallel transport.

Public numpy functions
----------------------
precompute_rotation_vector_batch — Rodrigues rotation vectors and pol. angle offsets
                                   for a batch of boresight pointings.
_rotation_params                 — per-sample scalars needed by _rodrigues_jit.
_recenter_and_rotate             — fused recenter + pol-roll via _rodrigues_jit.
"""

import math
import numpy as np
import numba


# ── Numba JIT kernels ─────────────────────────────────────────────────────────


@numba.jit(nopython=True, cache=True)
def _rodrigues_jit(vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, out):
    """
    Fused double-Rodrigues rotation (recenter + polarisation roll).

    All input/output arrays are float32.  Computes each output element in a
    single (b, s) pass with no temporaries beyond a handful of scalars.

    Parameters
    ----------
    vec_orig : (S, 3)
    axes     : (B, 3)  normalised rotation axes (zero where angle ≈ 0)
    cos_a    : (B,)
    sin_a    : (B,)
    ax_pts   : (B, 3)  pointing unit vectors
    cos_p    : (B,)
    sin_p    : (B,)
    out      : (B, S, 3)  written in place
    """
    B = axes.shape[0]
    S = vec_orig.shape[0]
    for b in range(B):
        kx = axes[b, 0]
        ky = axes[b, 1]
        kz = axes[b, 2]
        ca = cos_a[b]
        sa = sin_a[b]
        oma = 1.0 - ca
        px = ax_pts[b, 0]
        py = ax_pts[b, 1]
        pz = ax_pts[b, 2]
        cp_ = cos_p[b]
        sp_ = sin_p[b]
        omp = 1.0 - cp_
        for s in range(S):
            vx = vec_orig[s, 0]
            vy = vec_orig[s, 1]
            vz = vec_orig[s, 2]
            # Rodrigues 1 – recenter beam
            dkv = kx * vx + ky * vy + kz * vz
            rx = vx * ca + (ky * vz - kz * vy) * sa + kx * dkv * oma
            ry = vy * ca + (kz * vx - kx * vz) * sa + ky * dkv * oma
            rz = vz * ca + (kx * vy - ky * vx) * sa + kz * dkv * oma
            # Rodrigues 2 – polarisation roll
            dpr = px * rx + py * ry + pz * rz
            out[b, s, 0] = rx * cp_ + (py * rz - pz * ry) * sp_ + px * dpr * omp
            out[b, s, 1] = ry * cp_ + (pz * rx - px * rz) * sp_ + py * dpr * omp
            out[b, s, 2] = rz * cp_ + (px * ry - py * rx) * sp_ + pz * dpr * omp


@numba.jit(nopython=True, cache=True)
def _spin2_rodrigues_cos2d_sin2d(
    ri_x,
    ri_y,
    ri_z,
    ni_x,
    ni_y,
    ni_z,
    rq_x,
    rq_y,
    rq_z,
    nq_x,
    nq_y,
    nq_z,
):
    """Compute cos(2δ) and sin(2δ) for spin-2 frame rotation.

    Uses inlined Rodrigues parallel transport with the double-angle trick
    to avoid both sqrt and atan2.  δ is the angle by which the local-north
    frame at neighbour (ri) rotates when parallel-transported along the
    geodesic to query point (rq).

    Unnormalised Rodrigues formula (valid for non-antipodal pairs):
        v = r̂_i × r̂_q
        c = r̂_i · r̂_q
        t = n_i·c + (v × n_i) + v·(v·n_i)/(1+c)

    Then cos(δ) = t · ê_θ(q), sin(δ) = (ê_θ(q) × t) · r̂_q, and
    cos(2δ), sin(2δ) follow from double-angle identities.

    Parameters
    ----------
    ri_x, ri_y, ri_z : float   Neighbour position unit vector
    ni_x, ni_y, ni_z : float   Neighbour north direction ê_θ(i)
    rq_x, rq_y, rq_z : float   Query position unit vector
    nq_x, nq_y, nq_z : float   Query north direction ê_θ(q)

    Returns
    -------
    cos_2d, sin_2d : float
    """
    # v = r̂_i × r̂_q (unnormalised)
    vvx = ri_y * rq_z - ri_z * rq_y
    vvy = ri_z * rq_x - ri_x * rq_z
    vvz = ri_x * rq_y - ri_y * rq_x

    # c = r̂_i · r̂_q
    cc = ri_x * rq_x + ri_y * rq_y + ri_z * rq_z

    denom = 1.0 + cc
    if denom < 1.0e-15:
        return 1.0, 0.0  # antipodal / coincident — identity

    # Transported north: t = n_i·c + (v × n_i) + v·(v·n_i)/(1+c)
    vdn = vvx * ni_x + vvy * ni_y + vvz * ni_z
    fac = vdn / denom

    vxn_x = vvy * ni_z - vvz * ni_y
    vxn_y = vvz * ni_x - vvx * ni_z
    vxn_z = vvx * ni_y - vvy * ni_x

    tx = ni_x * cc + vxn_x + vvx * fac
    ty = ni_y * cc + vxn_y + vvy * fac
    tz = ni_z * cc + vxn_z + vvz * fac

    # cos δ = t · ê_θ(q)
    cos_d = tx * nq_x + ty * nq_y + tz * nq_z

    # sin δ = (ê_θ(q) × t) · r̂_q
    sin_d = (
        (nq_y * tz - nq_z * ty) * rq_x
        + (nq_z * tx - nq_x * tz) * rq_y
        + (nq_x * ty - nq_y * tx) * rq_z
    )

    # Double-angle identities — no atan2 needed
    return cos_d * cos_d - sin_d * sin_d, 2.0 * cos_d * sin_d


# ── Public numpy functions ────────────────────────────────────────────────────


def precompute_rotation_vector_batch(ra, dec, phi_batch, theta_batch, center_idx=None):
    """Compute Rodrigues rotation vectors and polarisation angle offsets for a batch.

    For each boresight pointing ``(phi_b, theta_b)`` in the batch, computes
    the Rodrigues rotation vector that brings the beam centre
    ``(ra[center_idx], dec[center_idx])`` to that pointing direction, together
    with the parallactic angle offset ``beta`` needed to align the polarisation
    frame.  The combined rotation angle applied at sample time is
    ``psi_b - beta``.

    Args:
        ra (numpy.ndarray): RA offsets of beam pixels from beam centre [rad],
            shape ``(H, W)``.
        dec (numpy.ndarray): Dec offsets of beam pixels from beam centre [rad],
            shape ``(H, W)``.
        phi_batch (numpy.ndarray): Boresight longitude for each sample [rad],
            shape ``(B,)``.
        theta_batch (numpy.ndarray): Boresight colatitude for each sample [rad],
            shape ``(B,)``.
        center_idx (tuple[int, int] or None): 2-D index of the beam-centre
            pixel in the ``ra``/``dec`` arrays. When ``None`` (default), the
            centre is derived from the array shape as
            ``(ra.shape[0] // 2, ra.shape[1] // 2)``, which matches the
            convention used by :func:`~tod_io.load_beam` and
            :mod:`precompute_beam_cache`.

    Returns:
        tuple:
            - **rot_vector** (*numpy.ndarray*) – Rodrigues rotation vectors
              (axis × angle), shape ``(B, 3)``.
            - **beta** (*numpy.ndarray*) – Polarisation angle offset [rad],
              shape ``(B,)``.  Always in ``[0, 2π)``.
            - **n_target** (*numpy.ndarray*) – Boresight-frame local north
              unit vector ``(cosφ cosθ, sinφ cosθ, -sinθ)``, shape ``(B, 3)``.
              Reused by the spin-2 Q/U correction in the gather-accumulate
              kernel as the rotation target.
    """
    if center_idx is None:
        center_idx = (ra.shape[0] // 2, ra.shape[1] // 2)

    def sph2vec(phi, theta):
        return np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            axis=-1,
        )

    phi_orig = ra
    theta_orig = np.pi / 2 - dec

    vec_center = sph2vec(phi_orig[center_idx], theta_orig[center_idx])[np.newaxis, :]
    vec_target = sph2vec(phi_batch, theta_batch)

    N_center = np.array(
        [
            np.cos(phi_orig[center_idx]) * np.cos(theta_orig[center_idx]),
            np.sin(phi_orig[center_idx]) * np.cos(theta_orig[center_idx]),
            -np.sin(theta_orig[center_idx]),
        ]
    )
    N_target = np.array(
        [
            np.cos(phi_batch) * np.cos(theta_batch),
            np.sin(phi_batch) * np.cos(theta_batch),
            -np.sin(theta_batch),
        ]
    ).T
    E_target = np.array(
        [-np.sin(phi_batch), np.cos(phi_batch), np.zeros_like(phi_batch)]
    ).T

    axis = np.cross(vec_center, vec_target)
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = np.where(axis_norm > 1e-10, axis / axis_norm, 0)

    angle = np.arccos(np.clip(np.sum(vec_center * vec_target, axis=-1), -1, 1))
    rot_vector = axis * angle[..., np.newaxis]

    ca = np.cos(angle)
    v = N_center[np.newaxis, :]
    dot_kv = np.sum(axis * v, axis=-1, keepdims=True)
    w = (
        v * ca[..., np.newaxis]
        + np.cross(axis, v) * np.sin(angle)[..., np.newaxis]
        + axis * dot_kv * (1 - ca)[..., np.newaxis]
    )

    beta = np.arctan2(np.sum(w * E_target, axis=-1), np.sum(w * N_target, axis=-1))
    beta = np.where(beta < 0, beta + 2 * np.pi, beta)

    return rot_vector, beta, N_target


def _rotation_params(rot_vecs, phi_b, theta_b, psis_b):
    """
    Pre-compute the per-sample scalars needed by _rodrigues_jit from the
    Rodrigues vectors and pointing angles.  All outputs are float32.

    Returns axes (B,3), cos_a (B,), sin_a (B,), ax_pts (B,3), cos_p (B,), sin_p (B,)
    """
    angles = np.linalg.norm(rot_vecs, axis=-1).astype(np.float32)  # (B,)
    safe = angles > np.float32(1e-10)
    axes = (
        rot_vecs / np.where(safe[:, None], angles[:, None], np.float32(1.0))
    ).astype(np.float32)
    axes = np.where(safe[:, None], axes, np.float32(0.0))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    phi_f = np.asarray(phi_b, dtype=np.float32)
    theta_f = np.asarray(theta_b, dtype=np.float32)
    psis_f = np.asarray(psis_b, dtype=np.float32)
    st = np.sin(theta_f)
    ct = np.cos(theta_f)
    sp = np.sin(phi_f)
    cp = np.cos(phi_f)
    ax_pts = np.stack([st * cp, st * sp, ct], axis=-1)
    cos_p = np.cos(psis_f)
    sin_p = np.sin(psis_f)

    return axes, cos_a, sin_a, ax_pts, cos_p, sin_p


def _recenter_and_rotate(vec_orig, rot_vecs, phi_pix, theta_pix, psis):
    """Apply a fused recenter + polarisation-roll rotation to beam pixel vectors.

    Executes the double Rodrigues rotation via the
    :func:`_rodrigues_jit` Numba kernel:

    1. **Recenter** (Rodrigues 1): rotate ``vec_orig`` from the beam-centre
       direction to the boresight direction defined by ``rot_vecs``.
    2. **Pol. roll** (Rodrigues 2): rotate by ``psis`` around the boresight
       unit vector.

    Args:
        vec_orig (numpy.ndarray): Beam-pixel unit vectors in the beam frame,
            shape ``(S, 3)``, dtype ``float32``.
        rot_vecs (numpy.ndarray): Rodrigues rotation vectors (axis × angle)
            from :func:`precompute_rotation_vector_batch`, shape ``(B, 3)``.
        phi_pix (numpy.ndarray): Boresight longitude [rad], shape ``(B,)``.
        theta_pix (numpy.ndarray): Boresight colatitude [rad], shape ``(B,)``.
        psis (numpy.ndarray): Combined rotation angle ``psi_b - beta`` [rad],
            shape ``(B,)``.

    Returns:
        numpy.ndarray: Rotated beam-pixel unit vectors in sky-frame coordinates,
            shape ``(B, S, 3)``, dtype ``float32``.
    """
    B = rot_vecs.shape[0]
    S = vec_orig.shape[0]
    axes, cos_a, sin_a, ax_pts, cos_p, sin_p = _rotation_params(
        rot_vecs, phi_pix, theta_pix, psis
    )
    out = np.empty((B, S, 3), dtype=np.float32)
    _rodrigues_jit(
        np.asarray(vec_orig, dtype=np.float32),
        axes,
        cos_a,
        sin_a,
        ax_pts,
        cos_p,
        sin_p,
        out,
    )
    return out
