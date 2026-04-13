"""
Core numerical routines for sample-based TOD generation.

All functions are stateless and take only arrays as arguments.

Numba JIT kernels
-----------------
_rodrigues_jit                — fused double Rodrigues rotation (recenter + pol. roll).
                                Writes directly into a pre-allocated (B, S, 3) buffer,
                                eliminating the 4-5 large intermediate arrays that the
                                numpy version creates.

_rodrigues1_from_rolled_jit   — single Rodrigues rotation (recenter only) applied to
                                per-sample pre-rolled (B, Sc, 3) beam vectors loaded
                                from the beam cache.  Replaces the double rotation when
                                vec_rolled / psi_grid are present in beam_data.

_gather_accum_jit       — fused HEALPix bilinear gather + beam-weighted accumulation.
                          Replaces the (C, 4, B*Sc) mp_gathered intermediate and the
                          separate einsum + matmul calls.  Kept for backward compatibility.

_gather_accum_fused_jit   — fully fused vec2ang + HEALPix bilinear interpolation +
                            beam accumulation; prange over B, sequential over S.
                            Replaces the hp.vec2ang + hp.get_interp_weights +
                            _gather_accum_jit triplet in beam_tod_batch, eliminating
                            all four intermediate arrays (theta_flat, phi_flat, pixels,
                            weights) from the hot tile loop.

_gather_accum_flatsky_jit — fully fused flat-sky HEALPix interpolation + accumulation.
                            Skips vec2ang and both Rodrigues rotations; computes sky
                            positions directly from precomputed (dtheta, dphi) offsets
                            and pointing angles (theta_b, phi_b).  Used when the beam
                            cache provides dtheta/dphi alongside vec_rolled/psi_grid.

HEALPix RING helpers (in numba_healpy.py)
-----------------------------------------
_ring_above_jit, _ring_info_jit, _ring_z_jit,
_get_interp_weights_jit, get_interp_weights_numba
"""

import math
import numpy as np
import healpy as hp
import numba

from numba_healpy import (
    _TWO_PI,
    _TWO_THIRDS,
    _ring_above_jit,
    _ring_info_jit,
    _ring_z_jit,
    get_interp_weights_numba,
)
from tod_nearest import (
    _gather_accum_nearest_jit,
    _gather_accum_nearest_flatsky_jit,
)
from tod_bilinear import (
    _gather_accum_jit,
    _gather_accum_fused_jit,
    _gather_accum_flatsky_jit,
)

# Target working-set size for the (B × Sc × 3 × float32) vec_rot intermediate.
# Sized to stay within a typical L2 cache (2 MB).
_S_TILE_TARGET_BYTES = 2 * 1024 * 1024

# Maximum number of S-tiles per beam entry.  Each tile makes one call into the
# HEALPix interpolation logic.  Capping at _MAX_TILES ensures Sc is always at
# least S/_MAX_TILES, keeping per-tile overhead bounded while still preventing
# out-of-memory.
_MAX_TILES = 8

_INV_TWO_PI = 1.0 / _TWO_PI


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
def _rodrigues1_from_rolled_jit(vec_rolled_b, axes, cos_a, sin_a, out):
    """
    Apply only Rodrigues 1 (recentering) to pre-rolled beam pixel vectors.

    Used when vec_rolled is loaded from the beam cache — the psi-roll
    (Rodrigues 2) is already baked in, so only the recentering rotation
    to the current pointing direction is needed.

    Parameters
    ----------
    vec_rolled_b : (B, Sc, 3)  float32  — per-sample pre-rolled vectors
    axes         : (B, 3)      float32  — Rodrigues rotation axes
    cos_a        : (B,)        float32
    sin_a        : (B,)        float32
    out          : (B, Sc, 3)  float32  — written in place
    """
    B = axes.shape[0]
    Sc = vec_rolled_b.shape[1]
    for b in range(B):
        kx = axes[b, 0]
        ky = axes[b, 1]
        kz = axes[b, 2]
        ca = cos_a[b]
        sa = sin_a[b]
        oma = 1.0 - ca
        for s in range(Sc):
            vx = vec_rolled_b[b, s, 0]
            vy = vec_rolled_b[b, s, 1]
            vz = vec_rolled_b[b, s, 2]
            dkv = kx * vx + ky * vy + kz * vz
            out[b, s, 0] = vx * ca + (ky * vz - kz * vy) * sa + kx * dkv * oma
            out[b, s, 1] = vy * ca + (kz * vx - kx * vz) * sa + ky * dkv * oma
            out[b, s, 2] = vz * ca + (kx * vy - ky * vx) * sa + kz * dkv * oma


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

    return rot_vector, beta


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


def beam_tod_batch(
    nside,
    mp,
    data,
    rot_vecs,
    phi_b,
    theta_b,
    psis_b,
    interp_mode="bilinear",
    spin2_corr=0,
):
    """Accumulate the TOD contribution of one beam entry for a batch of samples.

    Tiles over the ``S`` selected beam pixels so that the
    ``(B × Sc × 3 × float32)`` intermediate vector buffer stays within the L2
    cache target. Uses Numba JIT kernels for both the rotation and the
    gather + accumulation steps.

    Three execution paths are selected automatically based on the contents of
    ``data``:

    * **Flat-sky** *(fastest)* — requires ``vec_rolled``, ``psi_grid``,
      ``dtheta``, ``dphi``, and ``mp_stacked`` in ``data``. Skips both
      Rodrigues rotations and the ``vec2ang`` call; computes sky positions from
      precomputed angular offsets and feeds directly into the HEALPix
      interpolation kernel. Valid for narrow beams (≲ 5°).

    * **Single-Rodrigues** — requires ``vec_rolled``, ``psi_grid``, and
      ``mp_stacked`` in ``data``. The psi-roll is baked into the precomputed
      vectors; only the recentering rotation (Rodrigues 1) is applied at
      runtime. Roughly half the rotation cost of the full path.

    * **Full double-Rodrigues** *(fallback)* — used when no cache arrays are
      present. Applies both Rodrigues rotations per ``(B, S)`` element at
      runtime.

    Args:
        nside (int): HEALPix ``nside`` of the sky map.
        mp (list[numpy.ndarray]): Sky map components ``[I, Q, U]``. Each
            element is a 1-D ``float32`` array of length ``12 * nside**2``.
            Used only on the full double-Rodrigues fallback path.
        data (dict): Beam data entry as returned by :func:`prepare_beam_data`.
            Required keys: ``'vec_orig'``, ``'beam_vals'``, ``'comp_indices'``.
            Optional keys for cached paths: ``'mp_stacked'``, ``'vec_rolled'``,
            ``'psi_grid'``, ``'dtheta'``, ``'dphi'``.
        rot_vecs (numpy.ndarray): Rodrigues rotation vectors from
            :func:`precompute_rotation_vector_batch`, shape ``(B, 3)``.
        phi_b (numpy.ndarray): Boresight longitude [rad], shape ``(B,)``.
        theta_b (numpy.ndarray): Boresight colatitude [rad], shape ``(B,)``.
        psis_b (numpy.ndarray): Combined rotation angle ``psi_b - beta`` [rad],
            shape ``(B,)``.
        interp_mode (str): Sky-map interpolation strategy. One of:

            * ``'bilinear'`` *(default)* — 4-pixel bilinear HEALPix
              interpolation via the fused Numba kernel.
            * ``'nearest'`` — single nearest-pixel lookup; fastest, no pixel
              mixing.
            (``'gaussian'`` and ``'bicubic'`` are available on their respective branches.)
        spin2_corr (int): Spin-2 Q/U frame correction for bilinear
            interpolation. 0 = none (default), 1 = approx, 2 = exact.
            See :func:`numba_healpy._spin2_delta_approx_jit` and
            :func:`numba_healpy._spin2_delta_exact_jit`.

    Returns:
        dict[int, numpy.ndarray]: Mapping from Stokes component index to a
            ``(B,)`` ``float32`` array containing the beam-weighted sky-map
            accumulation for that component over the batch.
    """
    B = phi_b.shape[0]
    vec_orig = data["vec_orig"]  # (S, 3)
    beam_vals = data["beam_vals"]  # (S,)
    S = vec_orig.shape[0]
    comp_indices = data["comp_indices"]
    C = len(comp_indices)
    mp_stacked = data.get("mp_stacked")  # (C, N) float32, or None

    # Q and U channel positions within the C-dim of mp_stacked
    c_q = -1
    c_u = -1
    for _ci, _comp in enumerate(comp_indices):
        if _comp == 1:
            c_q = _ci
        elif _comp == 2:
            c_u = _ci
    vec_rolled = data.get("vec_rolled")  # (N_psi, S, 3) float32, or None
    psi_grid = data.get("psi_grid")  # (N_psi,) float32, or None
    dtheta = data.get("dtheta")  # (N_psi, S) float32, or None
    dphi = data.get("dphi")  # (N_psi, S) float32, or None

    use_cache = vec_rolled is not None and psi_grid is not None
    # Flat-sky path: skips both Rodrigues rotations and vec2ang entirely.
    # Requires mp_stacked since it feeds directly into _gather_accum_flatsky_jit.
    use_flatsky = (
        use_cache and dtheta is not None and dphi is not None and mp_stacked is not None
    )
    # Near-pole fallback (issue: sin θ ≈ 0 makes dphi/sin θ ill-defined on the
    # flat-sky path).  When any sample in the batch is within 1e-10 of a pole,
    # demote all the way to the full double-Rodrigues path, which handles every
    # latitude correctly without any precomputed-cache approximation.
    if use_flatsky and np.any(
        np.abs(np.sin(np.asarray(theta_b, dtype=np.float64))) < 1e-10
    ):
        use_flatsky = False
        use_cache = False
    use_nearest = interp_mode == "nearest"
    if interp_mode not in ("nearest", "bilinear"):
        raise ValueError(
            f"interp_mode {interp_mode!r} not available on main branch; "
            "switch to the 'gaussian' or 'bicubic' branch"
        )

    # Lower bound from L2 target; upper bound from _MAX_TILES cap.
    # The max() ensures we never produce more than _MAX_TILES tiles even when
    # the memory-based Sc is tiny (e.g. Sc=79 at B=2212 → 64 tiles → 64 C calls).
    Sc = max(1, _S_TILE_TARGET_BYTES // (B * 3 * 4))  # memory target
    Sc = max(Sc, -(-S // _MAX_TILES))  # tile-count cap (ceiling div)
    Sc = min(Sc, S)

    if use_cache:
        # Map each sample's psi angle to the nearest precomputed bin index.
        # Used by both the flat-sky and single-Rodrigues cached paths.
        n_psi = len(psi_grid)
        dpsi = _TWO_PI / n_psi
        k_b = np.mod(
            np.round(np.mod(psis_b, _TWO_PI) / dpsi).astype(np.int64),
            n_psi,
        )  # (B,)

    # Rotation scalars are not needed on the flat-sky path — both rotations
    # are bypassed. Compute them only for the other two paths.
    if not use_flatsky:
        axes, cos_a, sin_a, ax_pts, cos_p, sin_p = _rotation_params(
            rot_vecs, phi_b, theta_b, psis_b
        )

    tod = {comp: np.zeros(B, dtype=np.float32) for comp in comp_indices}

    for s0 in range(0, S, Sc):
        s1 = min(s0 + Sc, S)
        bv_chunk = beam_vals[s0:s1]  # (Sc,)

        if use_flatsky:
            tod_arr = np.zeros((C, B), dtype=np.float64)
            if use_nearest:
                # Flat-sky nearest-pixel path.
                _gather_accum_nearest_flatsky_jit(
                    np.ascontiguousarray(dtheta[:, s0:s1]),
                    np.ascontiguousarray(dphi[:, s0:s1]),
                    k_b,
                    np.asarray(theta_b, dtype=np.float32),
                    np.asarray(phi_b, dtype=np.float32),
                    nside,
                    mp_stacked,
                    bv_chunk,
                    B,
                    s1 - s0,
                    tod_arr,
                )
            else:
                # Flat-sky bilinear path (default).
                _gather_accum_flatsky_jit(
                    np.ascontiguousarray(dtheta[:, s0:s1]),
                    np.ascontiguousarray(dphi[:, s0:s1]),
                    k_b,
                    np.asarray(theta_b, dtype=np.float32),
                    np.asarray(phi_b, dtype=np.float32),
                    nside,
                    mp_stacked,
                    bv_chunk,
                    B,
                    s1 - s0,
                    tod_arr,
                )
            for i, comp in enumerate(comp_indices):
                tod[comp] += tod_arr[i].astype(np.float32)

        elif use_cache:
            # Single-Rodrigues path: psi-roll baked in, only recentering needed.
            vec_chunk = vec_rolled[k_b[:, None], np.arange(s0, s1)[None, :], :]
            vec_chunk = np.ascontiguousarray(vec_chunk.astype(np.float32))  # (B, Sc, 3)
            vec_rot = np.empty((B, s1 - s0, 3), dtype=np.float32)
            _rodrigues1_from_rolled_jit(vec_chunk, axes, cos_a, sin_a, vec_rot)

            if mp_stacked is not None:
                tod_arr = np.zeros((C, B), dtype=np.float64)
                if use_nearest:
                    _gather_accum_nearest_jit(
                        vec_rot,
                        nside,
                        mp_stacked,
                        bv_chunk,
                        B,
                        s1 - s0,
                        tod_arr,
                        ax_pts,
                        c_q,
                        c_u,
                    )
                else:
                    _gather_accum_fused_jit(
                        vec_rot,
                        nside,
                        mp_stacked,
                        bv_chunk,
                        B,
                        s1 - s0,
                        tod_arr,
                        spin2_corr,
                        c_q,
                        c_u,
                    )
                for i, comp in enumerate(comp_indices):
                    tod[comp] += tod_arr[i].astype(np.float32)
            else:
                theta_flat, phi_flat = hp.vec2ang(
                    vec_rot.reshape(-1, 3).astype(np.float64)
                )
                pixels, weights = get_interp_weights_numba(nside, theta_flat, phi_flat)
                mp_gathered = np.stack([mp[c][pixels] for c in comp_indices])
                mp_flat = np.einsum("ckn,kn->cn", mp_gathered, weights)
                tod_chunk = mp_flat.reshape(C, B, s1 - s0) @ bv_chunk
                for i, comp in enumerate(comp_indices):
                    tod[comp] += tod_chunk[i].astype(np.float32)

        else:
            # Original path: double Rodrigues (recenter + psi roll).
            vec_chunk = np.asarray(vec_orig[s0:s1], dtype=np.float32)  # (Sc, 3)
            vec_rot = np.empty((B, s1 - s0, 3), dtype=np.float32)
            _rodrigues_jit(vec_chunk, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot)

            if mp_stacked is not None:
                tod_arr = np.zeros((C, B), dtype=np.float64)
                if use_nearest:
                    _gather_accum_nearest_jit(
                        vec_rot,
                        nside,
                        mp_stacked,
                        bv_chunk,
                        B,
                        s1 - s0,
                        tod_arr,
                        ax_pts,
                        c_q,
                        c_u,
                    )
                else:
                    _gather_accum_fused_jit(
                        vec_rot,
                        nside,
                        mp_stacked,
                        bv_chunk,
                        B,
                        s1 - s0,
                        tod_arr,
                        spin2_corr,
                        c_q,
                        c_u,
                    )
                for i, comp in enumerate(comp_indices):
                    tod[comp] += tod_arr[i].astype(np.float32)
            else:
                theta_flat, phi_flat = hp.vec2ang(
                    vec_rot.reshape(-1, 3).astype(np.float64)
                )
                pixels, weights = get_interp_weights_numba(nside, theta_flat, phi_flat)
                mp_gathered = np.stack([mp[c][pixels] for c in comp_indices])
                mp_flat = np.einsum("ckn,kn->cn", mp_gathered, weights)
                tod_chunk = mp_flat.reshape(C, B, s1 - s0) @ bv_chunk
                for i, comp in enumerate(comp_indices):
                    tod[comp] += tod_chunk[i].astype(np.float32)

    return tod
