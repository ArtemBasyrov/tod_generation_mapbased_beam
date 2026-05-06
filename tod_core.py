"""
Core numerical routines for sample-based TOD generation.

All functions are stateless and take only arrays as arguments.

Rotation kernels (in tod_rotations.py)
---------------------------------------
_rodrigues_jit                — fused double Rodrigues rotation (recenter + pol. roll),
                                materialises a (B, S, 3) buffer.  Used by the
                                numpy-fallback path and by tests.
_rodrigues_apply_one_jit      — scalar per-(b, s) fused Rodrigues; inlined into
                                the production gather kernels so the (B, S, 3)
                                intermediate is never materialised.
_rotation_params              — per-sample scalars needed by the Rodrigues kernels.
_recenter_and_rotate          — fused recenter + pol-roll wrapper.
precompute_rotation_vector_batch — Rodrigues vectors and pol. angle offsets for a batch.

Gather/accumulate kernels
-------------------------
_gather_accum_jit          — scalar bilinear accumulation from pre-computed pixels/weights
                             (in tod_bilinear.py; used by tests).
_gather_accum_fused_jit    — fully fused Rodrigues + bilinear gather + per-b
                             direct-mapped spin-2 cache + accumulation
                             (in tod_bilinear.py).
_gather_accum_nearest_jit  — fused Rodrigues + nearest-pixel gather + accumulation
                             (in tod_nearest.py).

HEALPix RING helpers (in numba_healpy.py)
-----------------------------------------
_ring_above_jit, _ring_info_jit, _ring_z_jit,
_get_interp_weights_jit, get_interp_weights_numba
"""

import numpy as np
import healpy as hp

from numba_healpy import get_interp_weights_numba
from tod_rotations import (
    _rotation_params,
    _recenter_and_rotate,
    precompute_rotation_vector_batch,
)
from tod_nearest import (
    _gather_accum_nearest_jit,
)
from tod_bilinear import (
    _gather_accum_jit,
    _gather_accum_fused_jit,
)


def beam_tod_batch(
    nside,
    mp,
    data,
    rot_vecs,
    phi_b,
    theta_b,
    psis_b,
    interp_mode="bilinear",
    z_skip_threshold=-1.0,
):
    """Accumulate the TOD contribution of one beam entry for a batch of samples.

    Uses Numba JIT kernels that fuse the Rodrigues rotation into the gather +
    accumulation step: no ``(B, S, 3)`` intermediate is materialised on the
    production ``mp_stacked`` path, and there is no S-tile loop — one kernel
    call per beam entry per batch.

    Args:
        nside (int): HEALPix ``nside`` of the sky map.
        mp (list[numpy.ndarray]): Sky map components ``[I, Q, U]``. Each
            element is a 1-D ``float32`` array of length ``12 * nside**2``.
            Used only on the numpy-fallback path (when ``mp_stacked`` is not
            provided).
        data (dict): Beam data entry as returned by :func:`prepare_beam_data`.
            Required keys: ``'vec_orig'``, ``'beam_vals'``, ``'comp_indices'``.
            Production path additionally requires ``'mp_stacked'``.
        rot_vecs (numpy.ndarray): Rodrigues rotation vectors from
            :func:`precompute_rotation_vector_batch`, shape ``(B, 3)``.
        phi_b (numpy.ndarray): Boresight longitude [rad], shape ``(B,)``.
        theta_b (numpy.ndarray): Boresight colatitude [rad], shape ``(B,)``.
        psis_b (numpy.ndarray): Combined rotation angle ``psi_b - beta`` [rad],
            shape ``(B,)``.
        interp_mode (str): Sky-map interpolation strategy. One of:

            * ``'bilinear'`` *(default)* — 4-pixel bilinear HEALPix
              interpolation with spin-2 Q/U frame correction.
            * ``'nearest'`` — single nearest-pixel lookup; fastest, no pixel
              mixing.
            (``'gaussian'`` and ``'bicubic'`` are available on their respective branches.)
        z_skip_threshold (float): Per-``b`` spin-2 skip cutoff on
            ``|cos θ_pts|``.  Boresight samples with
            ``|bz| > z_skip_threshold`` apply the full Q/U frame correction;
            samples in the equatorial band (``|bz| <= z_skip_threshold``)
            bypass it.  ``-1.0`` (default) disables the optimisation —
            spin-2 is always applied, bit-identical to the un-optimised path.
            Pass the value returned by
            :func:`tod_bilinear.compute_spin2_skip_z_threshold` to enable.

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

    use_nearest = interp_mode == "nearest"
    if interp_mode not in ("nearest", "bilinear"):
        raise ValueError(
            f"interp_mode {interp_mode!r} not available on main branch; "
            "switch to the 'gaussian' or 'bicubic' branch"
        )

    axes, cos_a, sin_a, ax_pts, cos_p, sin_p = _rotation_params(
        rot_vecs, phi_b, theta_b, psis_b
    )
    vec_orig_f32 = np.ascontiguousarray(vec_orig, dtype=np.float32)
    beam_vals_f32 = np.ascontiguousarray(beam_vals, dtype=np.float32)

    if mp_stacked is not None:
        tod_arr = np.zeros((C, B), dtype=np.float64)
        if use_nearest:
            _gather_accum_nearest_jit(
                vec_orig_f32,
                axes,
                cos_a,
                sin_a,
                ax_pts,
                cos_p,
                sin_p,
                nside,
                mp_stacked,
                beam_vals_f32,
                B,
                S,
                tod_arr,
                c_q,
                c_u,
                float(z_skip_threshold),
            )
        else:
            _gather_accum_fused_jit(
                vec_orig_f32,
                axes,
                cos_a,
                sin_a,
                ax_pts,
                cos_p,
                sin_p,
                nside,
                mp_stacked,
                beam_vals_f32,
                B,
                S,
                tod_arr,
                c_q,
                c_u,
                float(z_skip_threshold),
            )
        return {
            comp: tod_arr[i].astype(np.float32) for i, comp in enumerate(comp_indices)
        }

    # ── Fallback: healpy-based gather when mp_stacked is not provided.
    # Not on the production hot path — materialises the (B, S, 3) rotated
    # vector buffer via the batch Rodrigues kernel.
    vec_rot = _recenter_and_rotate(vec_orig_f32, rot_vecs, phi_b, theta_b, psis_b)
    theta_flat, phi_flat = hp.vec2ang(vec_rot.reshape(-1, 3).astype(np.float64))
    pixels, weights = get_interp_weights_numba(nside, theta_flat, phi_flat)
    mp_gathered = np.stack([mp[c][pixels] for c in comp_indices])
    mp_flat = np.einsum("ckn,kn->cn", mp_gathered, weights)
    tod_chunk = mp_flat.reshape(C, B, S) @ beam_vals_f32
    return {
        comp: tod_chunk[i].astype(np.float32) for i, comp in enumerate(comp_indices)
    }
