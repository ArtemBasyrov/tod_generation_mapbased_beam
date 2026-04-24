"""
Bilinear interpolation methods for TOD generation.

This module contains the functions for bilinear interpolation
used in the TOD generation pipeline.
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

# Knuth multiplicative-hash constant for the direct-mapped spin-2 cache.
# Chosen to break the spatial clustering of consecutive HEALPix pixel indices
# that would collide on a plain low-bit mask.
_SPIN2_CACHE_HASH = 2654435769

# Direct-mapped spin-2 cache size (power of 2).  Sized to hold the ~100-300
# unique HEALPix pixels covered by a 30′ beam disc at nside=1024 with load
# ≪ 0.1, so collision misses are negligible.  The three backing arrays total
# ~24 KiB, fitting comfortably in L1.
_SPIN2_CACHE_SIZE = 1024
_SPIN2_CACHE_MASK = _SPIN2_CACHE_SIZE - 1


__all__ = (
    "_gather_accum_jit",
    "_gather_accum_fused_jit",
    "_spin2_cos2d_sin2d_jit",
    "_spin2_lookup_cached",
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


# ── Spin-2 frame rotation ─────────────────────────────────────────────────────


@numba.jit(nopython=True, cache=True)
def _spin2_cos2d_sin2d_jit(z_pix, sth_pix, phi_pix, z_pts, sth_pts, phi_pts):
    """Compute cos(2δ) and sin(2δ) for the spin-2 frame rotation.

    δ = alpha - gamma, where alpha is the bearing from the HEALPix pixel toward
    the boresight and gamma is the bearing from the boresight back toward the
    pixel, both measured from their respective local meridians.

    Uses the spherical triangle bearing formulas; avoids atan2 and beta by working
    directly with (cos α, sin α) and (cos γ, sin γ) and applying the
    double-angle identities to find tan(δ) = N / D, then reconstructing cos(2δ) and sin(2δ) via the
    double-angle formulas in terms of tan(δ) to avoid any loss of precision when
    δ is small.

    Parameters
    ----------
    z_pix, sth_pix, phi_pix : float   cos θ, sin θ, φ of the HEALPix pixel
    z_pts, sth_pts, phi_pts : float   cos θ, sin θ, φ of the boresight

    Returns
    -------
    cos_2d, sin_2d : float
    """
    # same point check (also covers the sin_beta ≈ 0 case)
    if phi_pts == phi_pix and sth_pts == sth_pix and z_pts == z_pix:
        return 1.0, 0.0

    # step 1
    dphi = phi_pts - phi_pix
    ch = math.cos(dphi * 0.5)
    sh = math.sin(dphi * 0.5)
    sin_dphi = 2.0 * ch * sh
    cos_dphi = 1 - 2.0 * sh * sh
    sh2 = sh * sh  # sin^2 (Delta phi / 2)

    # step 2
    dz = z_pts - z_pix
    ds = sth_pts - sth_pix
    st2 = 0.25 * (dz * dz + ds * ds)  # sin^2 (Delta theta / 2)
    S = sth_pix * sth_pts
    h = st2 + S * sh2  # haversine of the angular separation

    # step 3
    sin2_dtheta = 4.0 * st2 * (1 - st2)  # sin^2 (Delta theta)

    # step 4
    z_sum = z_pts + z_pix
    N = 2.0 * sin_dphi * z_sum * h  # numerator for tan(delta)

    # step 5
    C = z_pts * z_pix
    term1 = sin2_dtheta * cos_dphi
    term2 = 4.0 * S * C * sh2 * sh2
    term3 = S * sin_dphi * sin_dphi
    D = term1 - term2 + term3  # denominator for tan(delta)

    # step 6
    u = 1.0 / (N * N + D * D)
    cos_2d = (D * D - N * N) * u
    sin_2d = 2.0 * N * D * u

    return cos_2d, sin_2d


# ── Spin-2 cache probe helper ─────────────────────────────────────────────────


@numba.jit(nopython=True, cache=True, inline="always")
def _spin2_lookup_cached(
    p,
    z_n,
    phi_n,
    z_pts,
    sth_pts,
    phi_pts,
    cache_pix,
    cache_c2d,
    cache_s2d,
    cmask,
):
    """Probe the direct-mapped spin-2 cache for pixel ``p``; compute+store on miss.

    Cache slot is selected by Knuth's multiplicative hash so consecutive RING
    pixel indices on the same ring don't collide.  A slot whose stored pixel
    index equals ``p`` is a hit (returns the cached ``(cos 2δ, sin 2δ)``); any
    other value — including the initial sentinel ``-1`` — is a miss, in which
    case the spin-2 rotation is computed via :func:`_spin2_cos2d_sin2d_jit`
    and written into the slot, evicting any previous occupant.

    The cache is boresight-scoped: its contents are valid only for the
    ``(z_pts, sth_pts, phi_pts)`` passed in.  Callers must reset the cache
    (``cache_pix[:] = -1``) whenever they move to a new boresight.

    Parameters
    ----------
    p                         : int64    HEALPix RING pixel index
    z_n, phi_n                : float64  cos θ and φ of pixel ``p``'s centre
    z_pts, sth_pts, phi_pts   : float64  cos θ, sin θ, φ of the boresight
    cache_pix                 : (N,) int64    slot → pixel index (or -1)
    cache_c2d                 : (N,) float64  slot → cached cos(2δ)
    cache_s2d                 : (N,) float64  slot → cached sin(2δ)
    cmask                     : int      N − 1, where N is a power of 2

    Returns
    -------
    c2d, s2d : float64  cos(2δ), sin(2δ) for the pixel → boresight transport.
    """
    slot = (p * _SPIN2_CACHE_HASH) & cmask
    if cache_pix[slot] == p:
        return cache_c2d[slot], cache_s2d[slot]
    sth_n = math.sqrt(max(0.0, 1.0 - z_n * z_n))
    c2d, s2d = _spin2_cos2d_sin2d_jit(z_n, sth_n, phi_n, z_pts, sth_pts, phi_pts)
    cache_pix[slot] = p
    cache_c2d[slot] = c2d
    cache_s2d[slot] = s2d
    return c2d, s2d


# ── Fully fused kernel (spin-2 amortised via direct-mapped cache) ─────────────


@numba.jit(nopython=True, parallel=True, cache=True)
def _gather_accum_fused_jit(
    vec_rot,
    nside,
    mp_stacked,
    beam_vals,
    B,
    Sc,
    tod,
    ax_pts,
    n_target,
    c_q=-1,
    c_u=-1,
):
    """Fused HEALPix bilinear gather + spin-2 amortisation + accumulation.

    Parallelised over the boresight dimension B; each ``b`` owns ``tod[:, b]``
    exclusively so there are no write races.

    When Q/U are present the kernel amortises the spin-2 frame rotation across
    repeated HEALPix neighbour pixels using a per-``b`` direct-mapped cache:
    each pixel's ``cos(2δ), sin(2δ)`` is computed at most once per boresight
    and reused on subsequent occurrences.  Since beam pixels are spatially
    clustered around the boresight, the same HEALPix neighbour appears many
    times across the 4*Sc bilinear stencils — the cache turns those into
    near-free L1 hits.

    Why a direct-mapped cache instead of sort-then-dedup
    ----------------------------------------------------
    A full sort-based dedup (``argsort`` over 4*Sc pixel indices, build
    ``inv_idx``, then compute spin-2 on ``n_uniq`` uniques) was measured to
    be 2-3× SLOWER than the inline-spin-2 approach: at typical Sc the
    ``argsort`` cost (O(n log n) on ~32k elements ≈ 1-3 ms per b) exceeds the
    spin-2 calls it saves (~30 ns each × ~25 k saved calls ≈ 750 µs).

    The direct-mapped cache replaces the global O(n log n) dedup with a
    per-lookup O(1) probe that fits in L1.  No sort, no permutation arrays,
    no separate passes — the ring lookup, spin-2 cache, and accumulation all
    run in one tight loop over s.

    Cache layout (Q/U path)
    -----------------------
    The cache is a triple of parallel arrays of size ``_SPIN2_CACHE_SIZE``:
    ``cache_pix`` (int64, HEALPix index currently in slot, or -1),
    ``cache_c2d`` / ``cache_s2d`` (float64, cos 2δ / sin 2δ for that pixel).
    Slot selection and miss-fallback live in :func:`_spin2_lookup_cached`.

    At ``_SPIN2_CACHE_SIZE = 1024`` the three arrays total ~24 KiB, fitting
    in L1.  The cache is reset to -1 at the start of each ``b`` because
    spin-2 values depend on the boresight direction.

    When Q/U are absent the kernel collapses to a single fused vec2ang +
    bilinear-gather + accumulate loop with no scratch.

    Parameters
    ----------
    vec_rot    : (B, Sc, 3)   float32   rotated beam unit vectors
    nside      : int
    mp_stacked : (C, N_hp)    float32   stacked sky-map components
    beam_vals  : (Sc,)        float32   beam weights for this tile
    B, Sc      : int
    tod        : (C, B)       float64   accumulated in place
    ax_pts     : (B, 3)       float32   boresight unit vectors
    n_target   : (B, 3)       float32   boresight local-north (unused; kept
                                        for API compatibility with callers)
    c_q        : int          index of Q in C-dim of mp_stacked (−1 = absent)
    c_u        : int          index of U in C-dim of mp_stacked (−1 = absent)
    """
    C = mp_stacked.shape[0]
    has_qu = c_q >= 0 and c_u >= 0

    if has_qu:
        for b in numba.prange(B):
            # Per-b spin-2 cache.  Numba's NRT allocator handles this in
            # ~100 ns plus the ~3 µs memset; going thread-local would force
            # ``cache=False`` (dynamic-globals NumbaWarning), which costs
            # 5-15 s of recompile per fresh worker process — far worse.
            cache_pix = np.full(_SPIN2_CACHE_SIZE, -1, dtype=np.int64)
            cache_c2d = np.empty(_SPIN2_CACHE_SIZE, dtype=np.float64)
            cache_s2d = np.empty(_SPIN2_CACHE_SIZE, dtype=np.float64)

            # Boresight (z, sin θ, φ) — computed once per b.
            bx = float(ax_pts[b, 0])
            by = float(ax_pts[b, 1])
            bz = float(ax_pts[b, 2])
            z_pts = max(-1.0, min(1.0, bz))
            sth_pts = math.sqrt(max(0.0, 1.0 - bz * bz))
            phi_pts = math.atan2(by, bx)
            if phi_pts < 0.0:
                phi_pts += _TWO_PI

            for s in range(Sc):
                vx = float(vec_rot[b, s, 0])
                vy = float(vec_rot[b, s, 1])
                vz = float(vec_rot[b, s, 2])
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
                ) = _ring_interp_with_angles_jit(nside, z, phi_w)

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

                for c in range(C):
                    if c == c_q or c == c_u:
                        continue
                    tod[c, b] += (
                        w0 * float(mp_stacked[c, p0])
                        + w1 * float(mp_stacked[c, p1])
                        + w2 * float(mp_stacked[c, p2])
                        + w3 * float(mp_stacked[c, p3])
                    ) * bv
    else:
        # ── No Q/U: no spin-2 to amortise; skip the cache and go direct.
        for b in numba.prange(B):
            for s in range(Sc):
                vx = float(vec_rot[b, s, 0])
                vy = float(vec_rot[b, s, 1])
                vz = float(vec_rot[b, s, 2])
                z = max(-1.0, min(1.0, vz))
                phi_w = math.atan2(vy, vx)
                if phi_w < 0.0:
                    phi_w += _TWO_PI
                p0, p1, p2, p3, w0, w1, w2, w3 = _ring_interp_single_jit(
                    nside, z, phi_w
                )
                bv = float(beam_vals[s])
                for c in range(C):
                    tod[c, b] += (
                        w0 * float(mp_stacked[c, p0])
                        + w1 * float(mp_stacked[c, p1])
                        + w2 * float(mp_stacked[c, p2])
                        + w3 * float(mp_stacked[c, p3])
                    ) * bv
