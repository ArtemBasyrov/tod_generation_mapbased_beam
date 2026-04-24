"""
Nearest pixel interpolation methods for TOD generation.

This module contains the functions for nearest pixel interpolation
used in the TOD generation pipeline.
"""

import math
import numpy as np
import numba
from numba_healpy import (
    _TWO_PI,
    _ring_above_jit,
    _ring_info_jit,
    _ring_z_jit,
)
from tod_rotations import _rodrigues_apply_one_jit


@numba.jit(nopython=True, parallel=True, cache=True)
def _gather_accum_nearest_jit(
    vec_orig,
    axes,
    cos_a,
    sin_a,
    ax_pts,
    cos_p,
    sin_p,
    nside,
    mp_stacked,
    beam_vals,
    B,
    S,
    tod,
    c_q=-1,
    c_u=-1,
):
    """
    Fully fused Rodrigues + HEALPix nearest-pixel lookup + beam accumulation.

    For each ``(b, s)`` pair the beam-frame vector is rotated into the sky
    frame in registers via :func:`_rodrigues_apply_one_jit`, the nearest
    RING-scheme pixel is found via the inlined ang2pix algorithm, and its
    sky-map value is accumulated with the beam weight.  Parallelised over B.
    No intermediate ``(B, S, 3)`` buffer is materialised.

    Parameters
    ----------
    vec_orig   : (S, 3)       float32   beam-frame unit vectors (un-rotated)
    axes       : (B, 3)       float32   Rodrigues-1 rotation axes
    cos_a      : (B,)         float32   cos of Rodrigues-1 angle
    sin_a      : (B,)         float32   sin of Rodrigues-1 angle
    ax_pts     : (B, 3)       float32   boresight unit vectors (Rodrigues-2
                                        axis).  Passed for API consistency
                                        with the bilinear path; not used for
                                        Q/U frame correction on the nearest
                                        path (see bilinear note).
    cos_p      : (B,)         float32   cos of Rodrigues-2 angle (ψ_b − β)
    sin_p      : (B,)         float32   sin of Rodrigues-2 angle
    nside      : int
    mp_stacked : (C, N_hp)    float32   stacked sky-map components
    beam_vals  : (S,)         float32   beam weights
    B, S       : int
    tod        : (C, B)       float64   accumulated in place
    c_q        : int          index of Q within C-dim of mp_stacked (−1 = absent)
    c_u        : int          index of U within C-dim of mp_stacked (−1 = absent)
    """
    C = mp_stacked.shape[0]
    npix_total = 12 * nside * nside

    for b in numba.prange(B):
        kx = float(axes[b, 0])
        ky = float(axes[b, 1])
        kz = float(axes[b, 2])
        ca = float(cos_a[b])
        sa = float(sin_a[b])
        bx = float(ax_pts[b, 0])
        by = float(ax_pts[b, 1])
        bz = float(ax_pts[b, 2])
        cp_ = float(cos_p[b])
        sp_ = float(sin_p[b])

        for s in range(S):
            vx, vy, vz = _rodrigues_apply_one_jit(
                float(vec_orig[s, 0]),
                float(vec_orig[s, 1]),
                float(vec_orig[s, 2]),
                kx,
                ky,
                kz,
                ca,
                sa,
                bx,
                by,
                bz,
                cp_,
                sp_,
            )

            theta = math.atan2(math.sqrt(vx * vx + vy * vy), vz)
            phi = math.atan2(vy, vx)
            if phi < 0.0:
                phi += _TWO_PI

            z = vz / math.sqrt(vx * vx + vy * vy + vz * vz)  # cos(theta)
            phi_w = phi

            ir_above = _ring_above_jit(nside, z)
            if ir_above < 1:
                ir_above = 1
            elif ir_above > 4 * nside - 2:
                ir_above = 4 * nside - 2
            ir_below = ir_above + 1

            best_pix = 0
            best_cos = -2.0
            sin_th = math.sin(theta)

            for ir_g in (ir_above, ir_below):
                if ir_g < 1 or ir_g > 4 * nside - 1:
                    continue
                n_pix, first_pix, phi0, dphi_r = _ring_info_jit(nside, ir_g, npix_total)
                z_c = _ring_z_jit(nside, ir_g)
                sin_z_c = math.sqrt(max(0.0, 1.0 - z_c * z_c))
                ip_base = int(phi_w * n_pix / _TWO_PI) % n_pix
                for ip_try in (ip_base, (ip_base + 1) % n_pix):
                    phi_c = phi0 + ip_try * dphi_r
                    cos_d = sin_th * sin_z_c * math.cos(phi_w - phi_c) + z * z_c
                    if cos_d > best_cos:
                        best_cos = cos_d
                        best_pix = first_pix + ip_try

            bv = float(beam_vals[s])
            if c_q < 0 or c_u < 0:
                for c in range(C):
                    tod[c, b] += mp_stacked[c, best_pix] * bv
            else:
                for c in range(C):
                    if c != c_q and c != c_u:
                        tod[c, b] += mp_stacked[c, best_pix] * bv
                    elif c == c_q:
                        tod[c_q, b] += float(mp_stacked[c_q, best_pix]) * bv
                    elif c == c_u:
                        tod[c_u, b] += float(mp_stacked[c_u, best_pix]) * bv
