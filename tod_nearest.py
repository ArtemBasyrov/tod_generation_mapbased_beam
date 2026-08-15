"""
Nearest-pixel interpolation kernel for TOD generation.

_gather_accum_nearest_jit — fully fused Rodrigues + HEALPix nearest-pixel
                            lookup + spin-2 Q/U transport + accumulation.

The lookup selects the pixel whose *area contains* the direction, which is
what ``healpy.ang2pix`` returns and what the map value represents: a HEALPix
map stores the sky averaged over each pixel's area, so a direction inside
pixel p is estimated by ``map[p]``.  This is not the same as the pixel whose
*centre* is nearest — HEALPix pixels are equal-area but not round, so their
boundaries are not the bisectors between centres, and the two rules disagree
in a shell around every pixel edge (9.5% of directions at nside 1024).
Selecting by nearest centre would attribute the sample to a pixel whose area
excludes it.
"""

import math
import numpy as np
import numba
from numba_healpy import (
    _TWO_PI,
    _TWO_THIRDS,
    _ring_info_jit,
    _ring_z_jit,
)
from tod_rotations import _rodrigues_apply_one_jit
from tod_spin2 import (
    _SPIN2_CACHE_SIZE,
    _SPIN2_CACHE_MASK,
    _spin2_lookup_cached,
)


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
    z_skip_threshold=-1.0,
):
    """
    Fully fused Rodrigues + HEALPix nearest-pixel lookup + beam accumulation.

    For each ``(b, s)`` pair the beam-frame vector is rotated into the sky
    frame in registers via :func:`_rodrigues_apply_one_jit`, the containing
    RING-scheme pixel is found via the inlined closed-form ang2pix, and its
    sky-map value is accumulated with the beam weight.  Parallelised over B.
    No intermediate ``(B, S, 3)`` buffer is materialised.

    Parameters
    ----------
    The six rotation parameters are float64 at any ``precision`` setting; see
    :func:`~tod_rotations._rotation_params` for why the pointing is never
    rounded to the pipeline precision.

    vec_orig   : (S, 3)       precision  beam-frame unit vectors (un-rotated)
    axes       : (B, 3)       float64    Rodrigues-1 rotation axes
    cos_a      : (B,)         float64    cos of Rodrigues-1 angle
    sin_a      : (B,)         float64    sin of Rodrigues-1 angle
    ax_pts     : (B, 3)       float64    boresight unit vectors (Rodrigues-2
                                         axis, also used as the spin-2
                                         query direction for Q/U frame
                                         correction).
    cos_p      : (B,)         float64    cos of Rodrigues-2 angle (ψ_b − β)
    sin_p      : (B,)         float64    sin of Rodrigues-2 angle
    nside      : int
    mp_stacked : (C, N_hp)    precision  stacked sky-map components
    beam_vals  : (S,)         precision  beam weights
    B, S       : int
    tod        : (C, B)       float64   accumulated in place
    c_q        : int          index of Q within C-dim of mp_stacked (−1 = absent)
    c_u        : int          index of U within C-dim of mp_stacked (−1 = absent)
    """
    C = mp_stacked.shape[0]
    npix_total = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)  # pixels in the north polar cap
    has_qu = c_q >= 0 and c_u >= 0

    # Dummy buffers for the branches that never touch the spin-2 cache —
    # hoisted so those b's don't pay three zero-size allocs each.
    _empty_pix = np.empty(0, dtype=np.int64)
    _empty_c2d = np.empty(0, dtype=np.float64)
    _empty_s2d = np.empty(0, dtype=np.float64)

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

        # Boresight coordinates for spin-2, hoisted out of the s-loop.
        bz_pts = max(-1.0, min(1.0, bz))
        bsth_pts = math.sqrt(max(0.0, 1.0 - bz * bz))
        bphi_pts = math.atan2(by, bx)
        if bphi_pts < 0.0:
            bphi_pts += _TWO_PI

        # Spin-2 skip: equatorial boresights have negligible Q/U frame
        # rotation across the beam footprint.  z_skip_threshold = -1.0
        # disables the optimisation (always apply correction).
        bz_abs = bz if bz >= 0.0 else -bz
        apply_spin2 = bz_abs > z_skip_threshold

        # Per-b spin-2 cache: the footprint revisits far fewer distinct pixels
        # than it has beam nodes, so each transport is computed once and reused.
        # Boresight-scoped, hence allocated and cleared per b.
        use_cache = has_qu and apply_spin2
        if use_cache:
            cache_pix = np.empty(_SPIN2_CACHE_SIZE, dtype=np.int64)
            for _i in range(_SPIN2_CACHE_SIZE):
                cache_pix[_i] = -1
            cache_c2d = np.empty(_SPIN2_CACHE_SIZE, dtype=np.float64)
            cache_s2d = np.empty(_SPIN2_CACHE_SIZE, dtype=np.float64)
        else:
            cache_pix = _empty_pix
            cache_c2d = _empty_c2d
            cache_s2d = _empty_s2d

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

            phi_w = math.atan2(vy, vx)
            if phi_w < 0.0:
                phi_w += _TWO_PI

            z = vz
            za = z if z >= 0.0 else -z
            tt = phi_w * (2.0 / math.pi)  # in [0, 4)

            # HEALPix ang2pix_z_phi.  The ring index and the index within that
            # ring are intermediates of the pixel number, so the pixel centre
            # needed by the spin-2 transport costs no second lookup.
            if za <= _TWO_THIRDS:  # equatorial belt
                temp1 = nside * (0.5 + tt)
                temp2 = nside * z * 0.75
                jp = int(temp1 - temp2)  # ascending edge-line index
                jm = int(temp1 + temp2)  # descending edge-line index
                ir_eq = nside + 1 + jp - jm
                kshift = 1 - (ir_eq & 1)
                ip = (jp + jm - nside + kshift + 1) // 2
                n_pix = 4 * nside
                if ip >= n_pix:
                    ip -= n_pix
                best_pix = ncap + (ir_eq - 1) * n_pix + ip
                ring = nside + ir_eq - 1
            else:  # polar caps
                tp = tt - int(tt)
                tmp = nside * math.sqrt(3.0 * (1.0 - za))
                jp = int(tp * tmp)
                jm = int((1.0 - tp) * tmp)
                ir_cap = jp + jm + 1
                ip = int(tt * ir_cap)
                n_pix = 4 * ir_cap
                if ip >= n_pix:
                    ip -= n_pix
                if z > 0.0:
                    best_pix = 2 * ir_cap * (ir_cap - 1) + ip
                    ring = ir_cap
                else:
                    best_pix = npix_total - 2 * ir_cap * (ir_cap + 1) + ip
                    ring = 4 * nside - ir_cap

            bv = float(beam_vals[s])
            if not has_qu:
                for c in range(C):
                    tod[c, b] += mp_stacked[c, best_pix] * bv
            elif apply_spin2:
                _, _, phi0, dphi_r = _ring_info_jit(nside, ring, npix_total)
                c2d, s2d = _spin2_lookup_cached(
                    best_pix,
                    _ring_z_jit(nside, ring),
                    phi0 + ip * dphi_r,
                    bz_pts,
                    bsth_pts,
                    bphi_pts,
                    cache_pix,
                    cache_c2d,
                    cache_s2d,
                    _SPIN2_CACHE_MASK,
                )
                q_val = float(mp_stacked[c_q, best_pix])
                u_val = float(mp_stacked[c_u, best_pix])
                tod[c_q, b] += (q_val * c2d + u_val * s2d) * bv
                tod[c_u, b] += (-q_val * s2d + u_val * c2d) * bv
                for c in range(C):
                    if c != c_q and c != c_u:
                        tod[c, b] += mp_stacked[c, best_pix] * bv
            else:
                # Equatorial boresight: skip spin-2 rotation; scalar Q/U.
                for c in range(C):
                    tod[c, b] += mp_stacked[c, best_pix] * bv
