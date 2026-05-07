"""
Bilinear interpolation kernels for TOD generation.

_gather_accum_jit         — scalar bilinear accumulation from pre-computed
                            pixels and weights.  Reference implementation;
                            tests use it to verify the fused kernel below.
_gather_accum_fused_jit   — fully fused Rodrigues + bilinear gather + spin-2
                            cache + accumulation.  Production hot path.
"""

import math
import numpy as np
import numba
from numba_healpy import (
    _TWO_PI,
    _ring_interp_single_jit,
    _ring_interp_with_angles_jit,
    get_interp_weights_numba,  # re-exported for tests
)
from tod_rotations import _rodrigues_apply_one_jit
from tod_spin2 import (
    _SPIN2_CACHE_SIZE,
    _SPIN2_CACHE_MASK,
    _spin2_lookup_cached,
)


__all__ = (
    "_gather_accum_jit",
    "_gather_accum_fused_jit",
    "get_interp_weights_numba",
)


@numba.jit(nopython=True, cache=True)
def _gather_accum_jit(pixels, weights, beam_vals, mp_stacked, B, Sc, tod):
    """
    Fused HEALPix gather, bilinear interpolation, and beam accumulation.

    Replaces:  np.stack gather → einsum → reshape → matmul
    with a single scalar loop over (b, s) that reads map values once and
    accumulates directly, avoiding the (C, 4, B*Sc) mp_gathered intermediate.

    Parameters
    ----------
    pixels     : (4, B*Sc)  int64   HEALPix pixel indices
    weights    : (4, B*Sc)  float64 bilinear weights
    beam_vals  : (Sc,)      float32 beam weights
    mp_stacked : (C, N)     float32 stacked sky-map components
    B, Sc      : int
    tod        : (C, B)     float64 — accumulated in place
    """
    C = mp_stacked.shape[0]
    for b in range(B):
        for s in range(Sc):
            n = b * Sc + s
            bv = float(beam_vals[s])
            p0 = pixels[0, n]
            p1 = pixels[1, n]
            p2 = pixels[2, n]
            p3 = pixels[3, n]
            w0 = weights[0, n]
            w1 = weights[1, n]
            w2 = weights[2, n]
            w3 = weights[3, n]
            for c in range(C):
                tod[c, b] += (
                    mp_stacked[c, p0] * w0
                    + mp_stacked[c, p1] * w1
                    + mp_stacked[c, p2] * w2
                    + mp_stacked[c, p3] * w3
                ) * bv


# ── Fully fused kernel (spin-2 amortised via direct-mapped cache) ─────────────


@numba.jit(nopython=True, parallel=True, cache=True)
def _gather_accum_fused_jit(
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
    """Fully fused Rodrigues + HEALPix bilinear gather + spin-2 + accumulation.

    The rotated beam-pixel vector is computed scalar-wise inside the inner
    loop via :func:`_rodrigues_apply_one_jit`, so no ``(B, S, 3)`` intermediate
    is materialised.  That eliminates the DRAM write-then-read round-trip of
    the old "pre-rotate to buffer, then gather" pipeline and lets the S-tile
    loop go away entirely: the kernel is called once per beam entry per
    batch, with all ``S`` beam pixels processed inside a single parallel
    region.

    Parallelised over the boresight dimension ``B``; each ``b`` owns
    ``tod[:, b]`` exclusively so there are no write races.

    When Q/U are present the kernel amortises the spin-2 frame rotation
    across repeated HEALPix neighbour pixels using a per-``b`` direct-mapped
    cache (``_spin2_lookup_cached``): each pixel's ``cos(2δ), sin(2δ)`` is
    computed at most once per boresight and reused on subsequent
    occurrences.  Because the kernel is now called once per beam entry, the
    cache amortises across all ``4 * S`` bilinear stencils — not just one
    tile's worth, as in the pre-fusion pipeline.

    Cache layout (Q/U path)
    -----------------------
    Three parallel arrays of size ``_SPIN2_CACHE_SIZE``:
    ``cache_pix`` (int64, HEALPix index currently in slot, or -1),
    ``cache_c2d`` / ``cache_s2d`` (float64, cos 2δ / sin 2δ for that pixel).
    At ``_SPIN2_CACHE_SIZE = 1024`` the three arrays total ~24 KiB, fitting
    in L1.  Reset to -1 at the start of each ``b`` because spin-2 values
    depend on the boresight direction.

    When Q/U are absent the kernel collapses to a single fused Rodrigues +
    vec2ang + bilinear-gather + accumulate loop with no scratch.

    Parameters
    ----------
    vec_orig   : (S, 3)       float32   beam-frame unit vectors (un-rotated)
    axes       : (B, 3)       float32   Rodrigues-1 rotation axes
    cos_a      : (B,)         float32   cos of Rodrigues-1 angle
    sin_a      : (B,)         float32   sin of Rodrigues-1 angle
    ax_pts     : (B, 3)       float32   boresight unit vectors (Rodrigues-2
                                        axis, also used as the spin-2
                                        query-direction)
    cos_p      : (B,)         float32   cos of Rodrigues-2 angle (ψ_b − β)
    sin_p      : (B,)         float32   sin of Rodrigues-2 angle
    nside      : int
    mp_stacked : (C, N_hp)    float32   stacked sky-map components
    beam_vals  : (S,)         float32   beam weights
    B, S       : int
    tod        : (C, B)       float64   accumulated in place
    c_q        : int          index of Q in C-dim of mp_stacked (−1 = absent)
    c_u        : int          index of U in C-dim of mp_stacked (−1 = absent)
    """
    C = mp_stacked.shape[0]
    has_qu = c_q >= 0 and c_u >= 0
    npix_total = 12 * nside * nside

    # Precompute which channels are neither Q nor U (used in the has_qu path).
    n_other = 0
    _other_ch = np.empty(C, dtype=np.int64)
    for _c in range(C):
        if _c != c_q and _c != c_u:
            _other_ch[n_other] = _c
            n_other += 1

    if has_qu:
        for b in numba.prange(B):
            # Per-b rotation scalars — hoisted out of the inner s-loop.
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

            # Spin-2 skip test: equatorial boresights (small |bz|) have
            # negligible Q/U frame rotation across the beam footprint, so the
            # spin-2 lookup can be bypassed.  z_skip_threshold = -1.0
            # disables the optimisation (always apply correction).
            bz_abs = bz if bz >= 0.0 else -bz
            apply_spin2 = bz_abs > z_skip_threshold

            # Cache only allocated when actually used.  ~24 KiB per active b.
            if apply_spin2:
                cache_pix = np.full(_SPIN2_CACHE_SIZE, -1, dtype=np.int64)
                cache_c2d = np.empty(_SPIN2_CACHE_SIZE, dtype=np.float64)
                cache_s2d = np.empty(_SPIN2_CACHE_SIZE, dtype=np.float64)
            else:
                cache_pix = np.empty(0, dtype=np.int64)
                cache_c2d = np.empty(0, dtype=np.float64)
                cache_s2d = np.empty(0, dtype=np.float64)

            # Boresight (z, sin θ, φ) for spin-2 — computed once per b.
            z_pts = max(-1.0, min(1.0, bz))
            sth_pts = math.sqrt(max(0.0, 1.0 - bz * bz))
            phi_pts = math.atan2(by, bx)
            if phi_pts < 0.0:
                phi_pts += _TWO_PI

            if apply_spin2:
                for s in range(S):
                    # Rodrigues in registers — no (B, S, 3) intermediate.
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
                    z = max(-1.0, min(1.0, vz))
                    phi_w = math.atan2(vy, vx)
                    if phi_w < 0.0:
                        phi_w += _TWO_PI
                    bv = float(beam_vals[s])

                    (
                        p0,
                        p1,
                        p2,
                        p3,
                        w0,
                        w1,
                        w2,
                        w3,
                        z_n0,
                        z_n1,
                        z_n2,
                        z_n3,
                        phi_n0,
                        phi_n1,
                        phi_n2,
                        phi_n3,
                    ) = _ring_interp_with_angles_jit(nside, z, phi_w, npix_total)

                    c2d0, s2d0 = _spin2_lookup_cached(
                        p0,
                        z_n0,
                        phi_n0,
                        z_pts,
                        sth_pts,
                        phi_pts,
                        cache_pix,
                        cache_c2d,
                        cache_s2d,
                        _SPIN2_CACHE_MASK,
                    )
                    c2d1, s2d1 = _spin2_lookup_cached(
                        p1,
                        z_n1,
                        phi_n1,
                        z_pts,
                        sth_pts,
                        phi_pts,
                        cache_pix,
                        cache_c2d,
                        cache_s2d,
                        _SPIN2_CACHE_MASK,
                    )
                    c2d2, s2d2 = _spin2_lookup_cached(
                        p2,
                        z_n2,
                        phi_n2,
                        z_pts,
                        sth_pts,
                        phi_pts,
                        cache_pix,
                        cache_c2d,
                        cache_s2d,
                        _SPIN2_CACHE_MASK,
                    )
                    c2d3, s2d3 = _spin2_lookup_cached(
                        p3,
                        z_n3,
                        phi_n3,
                        z_pts,
                        sth_pts,
                        phi_pts,
                        cache_pix,
                        cache_c2d,
                        cache_s2d,
                        _SPIN2_CACHE_MASK,
                    )

                    q0 = float(mp_stacked[c_q, p0])
                    u0 = float(mp_stacked[c_u, p0])
                    q1 = float(mp_stacked[c_q, p1])
                    u1 = float(mp_stacked[c_u, p1])
                    q2 = float(mp_stacked[c_q, p2])
                    u2 = float(mp_stacked[c_u, p2])
                    q3 = float(mp_stacked[c_q, p3])
                    u3 = float(mp_stacked[c_u, p3])

                    tod[c_q, b] += (
                        w0 * (q0 * c2d0 + u0 * s2d0)
                        + w1 * (q1 * c2d1 + u1 * s2d1)
                        + w2 * (q2 * c2d2 + u2 * s2d2)
                        + w3 * (q3 * c2d3 + u3 * s2d3)
                    ) * bv
                    tod[c_u, b] += (
                        w0 * (-q0 * s2d0 + u0 * c2d0)
                        + w1 * (-q1 * s2d1 + u1 * c2d1)
                        + w2 * (-q2 * s2d2 + u2 * c2d2)
                        + w3 * (-q3 * s2d3 + u3 * c2d3)
                    ) * bv

                    for _oi in range(n_other):
                        c = _other_ch[_oi]
                        tod[c, b] += (
                            w0 * float(mp_stacked[c, p0])
                            + w1 * float(mp_stacked[c, p1])
                            + w2 * float(mp_stacked[c, p2])
                            + w3 * float(mp_stacked[c, p3])
                        ) * bv
            else:
                # Equatorial boresight: skip spin-2 rotation.  Q/U accumulate
                # as scalars, identical to I in the bilinear gather.
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
                    z = max(-1.0, min(1.0, vz))
                    phi_w = math.atan2(vy, vx)
                    if phi_w < 0.0:
                        phi_w += _TWO_PI
                    bv = float(beam_vals[s])

                    p0, p1, p2, p3, w0, w1, w2, w3 = _ring_interp_single_jit(
                        nside, z, phi_w, npix_total
                    )

                    tod[c_q, b] += (
                        w0 * float(mp_stacked[c_q, p0])
                        + w1 * float(mp_stacked[c_q, p1])
                        + w2 * float(mp_stacked[c_q, p2])
                        + w3 * float(mp_stacked[c_q, p3])
                    ) * bv
                    tod[c_u, b] += (
                        w0 * float(mp_stacked[c_u, p0])
                        + w1 * float(mp_stacked[c_u, p1])
                        + w2 * float(mp_stacked[c_u, p2])
                        + w3 * float(mp_stacked[c_u, p3])
                    ) * bv

                    for _oi in range(n_other):
                        c = _other_ch[_oi]
                        tod[c, b] += (
                            w0 * float(mp_stacked[c, p0])
                            + w1 * float(mp_stacked[c, p1])
                            + w2 * float(mp_stacked[c, p2])
                            + w3 * float(mp_stacked[c, p3])
                        ) * bv
    else:
        # ── No Q/U: no spin-2 to amortise; skip the cache and go direct.
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
                z = max(-1.0, min(1.0, vz))
                phi_w = math.atan2(vy, vx)
                if phi_w < 0.0:
                    phi_w += _TWO_PI
                p0, p1, p2, p3, w0, w1, w2, w3 = _ring_interp_single_jit(
                    nside, z, phi_w, npix_total
                )
                bv = float(beam_vals[s])
                for c in range(C):
                    tod[c, b] += (
                        w0 * float(mp_stacked[c, p0])
                        + w1 * float(mp_stacked[c, p1])
                        + w2 * float(mp_stacked[c, p2])
                        + w3 * float(mp_stacked[c, p3])
                    ) * bv
