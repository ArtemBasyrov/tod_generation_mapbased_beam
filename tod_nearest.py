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


@numba.jit(nopython=True, parallel=True, cache=True)
def _gather_accum_nearest_jit(
    vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod, ax_pts, c_q=-1, c_u=-1
):
    """
    Fully fused vec2ang + HEALPix nearest-pixel lookup + beam accumulation.

    For each rotated beam vector the single nearest RING-scheme pixel is found
    via the inline ang2pix algorithm (_ang2pix_ring_jit inlined) and its
    sky-map value is accumulated with the beam weight.  Parallelised over B.

    Parameters
    ----------
    vec_rot    : (B, Sc, 3)   float32   rotated beam unit vectors
    nside      : int
    mp_stacked : (C, N_hp)    float32   stacked sky-map components
    beam_vals  : (Sc,)        float32   beam weights for this tile
    B, Sc      : int
    tod        : (C, B)       float64   accumulated in place
    ax_pts     : (B, 3)       float32   boresight unit vectors (API consistency;
                              not used for Q/U frame correction — see bilinear note)
    c_q        : int          index of Q within C-dim of mp_stacked (−1 = absent)
    c_u        : int          index of U within C-dim of mp_stacked (−1 = absent)
    """
    C = mp_stacked.shape[0]
    npix_total = 12 * nside * nside

    for b in numba.prange(B):
        for s in range(Sc):
            # vec2ang (inline)
            vx = float(vec_rot[b, s, 0])
            vy = float(vec_rot[b, s, 1])
            vz = float(vec_rot[b, s, 2])
            theta = math.atan2(math.sqrt(vx * vx + vy * vy), vz)
            phi = math.atan2(vy, vx)
            if phi < 0.0:
                phi += _TWO_PI

            # nearest-pixel lookup (inlined _ang2pix_ring_jit)
            z = vz / math.sqrt(vx * vx + vy * vy + vz * vz)  # cos(theta)
            phi_w = phi  # already in [0, 2π)

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
