"""
Bicubic (Keys/Catmull-Rom) interpolation for TOD generation.

Provides a gnomonic-projection-based bicubic interpolation using the
Keys/Catmull-Rom kernel as an alternative to the default bilinear HEALPix
interpolation.

_keys_1d_jit             — Keys cubic kernel, Horner form
_gather_accum_bicubic_jit — fully fused vec2ang + bicubic + beam accumulation
_bicubic_interp_accum    — Python wrapper
"""

import math

import healpy as hp
import numba
import numpy as np

from numba_healpy import (
    _TWO_PI,
    _ang2pix_ring_jit,
    _ring_above_jit,
    _ring_info_jit,
    _ring_z_jit,
)

# ── Keys cubic kernel (Catmull-Rom, a = −0.5) ────────────────────────────────


@numba.jit(nopython=True, cache=True)
def _keys_1d_jit(t):
    """Keys cubic convolution kernel, Horner form.

    Support (−2, 2):
        |t| < 1 :  (3/2)|t|³ − (5/2)|t|² + 1
        |t| < 2 :  (−1/2)|t|³ + (5/2)|t|² − 4|t| + 2
        else    :  0
    """
    t = abs(t)
    if t < 1.0:
        return (1.5 * t - 2.5) * t * t + 1.0
    if t < 2.0:
        return ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0
    return 0.0


@numba.jit(nopython=True, parallel=True, cache=True)
def _gather_accum_bicubic_jit(
    vec_rot, B, Sc, nside, h_pix, mp_stacked, beam_vals, tod_arr
):
    """
    Fully fused vec2ang + Keys-bicubic interpolation + beam accumulation.

    For each detector sample b (parallelised), iterates sequentially over Sc
    beam pixels s and accumulates directly into tod_arr[c, b].  This avoids
    the (C, B*Sc) intermediate tod_tmp buffer and the second reduction pass,
    keeping tod_arr[c, b] hot in L1/L2 cache throughout the inner Sc loop.

    Three improvements over the previous two-pass (gather-stencil-then-iterate)
    design:

    1. Inlined stencil gather — ring geometry (z_ring, phi_n, pixel index) is
       computed inline inside a single ring-by-ring loop, eliminating ~210
       scratch-buffer memory operations per (b, s) element and the
       _gather_ring_stencil_jit function-call overhead.

    2. Per-ring yi early exit — before iterating over a ring's phi pixels, the
       approximate north-axis gnomonic coordinate
           yi_approx = −(sn·cos_th − cn·sin_th) / h_pix
       is evaluated at dphi = 0.  Rings where |yi_approx| > 2 + 0.2 are skipped
       entirely.  The approximation error is < 0.003 pixel units for nside ≥ 512
       (max dphi ≈ 2 h_pix, |Δcos_dphi| < 2e-6), so the 0.2-unit margin is
       conservative.  Saving is ~15–28% of candidate evaluations, highest in the
       polar cap where ring spacing ≈ 0.80 h_pix puts rings ±3 at |yi| ≈ 2.4.

    3. 4-pixel phi stencil — only 4 phi pixels per ring are visited instead of 5.
       For fractional phi offset f ∈ (−0.5, 0.5] from the nearest pixel center:
         f >= 0  →  dip ∈ {-1,  0, +1, +2}  (right-biased window)
         f <  0  →  dip ∈ {-2, -1,  0, +1}  (left-biased window)
       In the right-biased case dip = -2 gives |xi| = 2 + f ≥ 2 (always zero);
       in the left-biased case dip = +2 gives |xi| = 2 + |f| > 2 (always zero).
       Choosing the window by sign(f) avoids computing dphi/xi/kxi for that dead
       candidate, saving ~14% of per-ring iterations (7 per (b,s) pair).

    Parameters
    ----------
    vec_rot    : (B, Sc, 3)   float32   rotated beam unit vectors
    B, Sc      : int
    nside      : int
    h_pix      : float64      angular pixel scale [rad] = nside2resol(nside)
    mp_stacked : (C, N_hp)    float32   stacked sky-map components
    beam_vals  : (Sc,)        float32   beam weights for this tile
    tod_arr    : (C, B)       float64   accumulated in place
    """
    C = mp_stacked.shape[0]
    inv_h = 1.0 / h_pix
    npix_total = 12 * nside * nside
    _YI_MARGIN = 0.2  # pixel units; see docstring for derivation
    _TAYLOR_THR = (
        0.05  # rad; cos Taylor error < 1.6e-8 → value error < 1e-5 (nside≥512)
    )

    for b in numba.prange(B):
        for s in range(Sc):
            # ── vec2ang (inline) ──────────────────────────────────────────────
            vx = float(vec_rot[b, s, 0])
            vy = float(vec_rot[b, s, 1])
            vz = float(vec_rot[b, s, 2])
            r_xy = math.sqrt(vx * vx + vy * vy)
            th = math.atan2(r_xy, vz)
            ph = math.atan2(vy, vx)
            if ph < 0.0:
                ph += _TWO_PI

            bv = float(beam_vals[s])
            sin_th = r_xy
            cos_th = vz

            # ── ring bounds ────────────────────────────────────────────────────
            ir_center = _ring_above_jit(nside, vz)
            if ir_center < 1:
                ir_center = 1
            elif ir_center > 4 * nside - 1:
                ir_center = 4 * nside - 1
            ir_lo = max(1, ir_center - 2)
            ir_hi = min(4 * nside - 1, ir_center + 2)

            # ── fused ring walk + Keys weight + map accumulation ──────────────
            # · sin(θ_ring) = sqrt(1−z²): 1 sqrt per ring replaces acos→sin→cos.
            # · Per-ring products (st_sn, ct_cn, sn_ct, cn_st) amortise 4 muls
            #   over the 4-pixel phi stencil.
            # · dphi is always small (≤ 2 phi-pixel steps); Taylor branch
            #   (|dphi| < 0.025 rad) replaces sin/cos with 3 muls each.
            # · Early exit on kxi = 0 skips north projection + second kyi call.
            # · Early exit on w = 0 skips C map reads (cache misses, 600 MB map).
            # · Weight and map value accumulated together — pixel index read once.
            w_sum = (
                0.0  # float64: Keys kernel returns float64, so accumulator must match
            )
            acc = np.zeros(
                C, dtype=np.float64
            )  # Numba stack-allocates this (C = 3 for T/Q/U)

            for ir in range(ir_lo, ir_hi + 1):
                n_p, fp, phi0, dphi_ring = _ring_info_jit(nside, ir, npix_total)
                cn = _ring_z_jit(nside, ir)
                sn = math.sqrt(max(0.0, 1.0 - cn * cn))
                sn_ct = sn * cos_th  # for yi numerator and yi_approx
                cn_st = cn * sin_th  # for yi numerator and yi_approx

                # ── per-ring yi early exit ─────────────────────────────────────
                # yi_approx = -(sn·cos_th − cn·sin_th) / h_pix  at dphi = 0.
                # Error vs actual yi < 0.003 pixels at nside ≥ 512; margin = 0.2.
                if abs(-(sn_ct - cn_st) * inv_h) > 2.0 + _YI_MARGIN:
                    continue

                st_sn = sin_th * sn  # for cos_c
                ct_cn = cos_th * cn  # for cos_c

                # Normal ring: 4-pixel phi stencil.
                # t_phi  = fractional position in phi-pixel units from phi0.
                # ip_f   = nearest pixel as float (before modulo).
                # f_frac = sub-pixel offset ∈ (−0.5, 0.5].
                # dip_lo = -1 when f_frac >= 0 (right-biased; dip=-2 dead)
                #        = -2 when f_frac <  0 (left-biased;  dip=+2 dead)
                t_phi = (ph - phi0) / dphi_ring
                ip_f = math.floor(t_phi + 0.5)
                ip_center = int(ip_f) % n_p
                f_frac = t_phi - ip_f
                dip_lo = -1 if f_frac >= 0.0 else -2
                for dip in range(dip_lo, dip_lo + 4):
                    ip_in = (ip_center + dip) % n_p
                    phi_n = phi0 + ip_in * dphi_ring
                    p = fp + ip_in
                    dphi_c = phi_n - ph
                    if dphi_c > math.pi:
                        dphi_c -= _TWO_PI
                    elif dphi_c < -math.pi:
                        dphi_c += _TWO_PI
                    if abs(dphi_c) < _TAYLOR_THR:
                        dp2 = dphi_c * dphi_c
                        sin_dphi = dphi_c * (1.0 - dp2 * (1.0 / 6.0))
                        cos_dphi = 1.0 - dp2 * 0.5
                    else:
                        sin_dphi = math.sin(dphi_c)
                        cos_dphi = math.cos(dphi_c)
                    cos_c = st_sn * cos_dphi + ct_cn
                    if cos_c < 1e-10:
                        continue
                    inv_c = inv_h / cos_c
                    xi = sn * sin_dphi * inv_c
                    kxi = _keys_1d_jit(xi)
                    if kxi == 0.0:
                        continue
                    yi = -(sn_ct * cos_dphi - cn_st) * inv_c
                    w = kxi * _keys_1d_jit(yi)
                    if w == 0.0:
                        continue
                    w_sum += w
                    for c in range(C):
                        acc[c] += w * mp_stacked[c, p]

            # ── write to tod_arr (no intermediate buffer) ─────────────────────
            if abs(w_sum) < 1e-10:
                # Degenerate weight sum (pole or very sparse disc) — nearest fallback
                nearest = _ang2pix_ring_jit(nside, th, ph)
                for c in range(C):
                    tod_arr[c, b] += mp_stacked[c, nearest] * bv
            else:
                inv_w = bv / w_sum
                for c in range(C):
                    tod_arr[c, b] += acc[c] * inv_w


def _bicubic_interp_accum(vec_rot, B, Sc, nside, mp_stacked, beam_vals, tod_arr):
    """
    Wrapper: resolves pixel scale and calls the JIT kernel.

    No thread-local scratch buffers needed: the kernel inlines the ring-walk
    stencil gather and accumulates directly into tod_arr[c, b].

    Parameters
    ----------
    vec_rot    : (B, Sc, 3)  float32
    B, Sc      : int
    nside      : int
    mp_stacked : (C, N_hp)   float32
    beam_vals  : (Sc,)       float32
    tod_arr    : (C, B)      float64   accumulated in place
    """
    h_pix = hp.nside2resol(nside)
    _gather_accum_bicubic_jit(
        vec_rot, B, Sc, nside, h_pix, mp_stacked, beam_vals, tod_arr
    )
