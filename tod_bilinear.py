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
    _ring_above_jit,
    _ring_info_jit,
    _ring_z_jit,
    _ring_interp_single_jit,
    get_interp_weights_numba,  # re-exported for tests
)

__all__ = (
    "_gather_accum_jit",
    "_gather_accum_fused_jit",
    "_spin2_cos2d_sin2d_jit",
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
    ``cache_pix[N_CACHE]``  : int64    HEALPix index currently in slot, or -1
    ``cache_c2d[N_CACHE]``  : float64  cos(2δ) for that pixel + this boresight
    ``cache_s2d[N_CACHE]``  : float64  sin(2δ) for that pixel + this boresight

    The slot is selected by a Knuth-style multiplicative hash
    ``(p * 2654435769) & CMASK`` to break the spatial clustering of
    consecutive HEALPix pixel indices that would collide on a plain low-bit
    mask.  ``N_CACHE = 4096`` keeps the three arrays at ~96 KiB total — fits
    comfortably in L2, mostly in L1.  The cache is reset to -1 at the start
    of each ``b`` (one ~3 µs memset) because spin-2 values depend on the
    boresight direction.

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
    npix_total = 12 * nside * nside
    has_qu = c_q >= 0 and c_u >= 0

    # Direct-mapped spin-2 cache — power of 2.  Sized for the realistic
    # workload: at 1-arcmin beam pixels and nside=1024 the typical
    # 30-arcmin beam disc covers ~100-300 HEALPix pixels, so a 4096-slot
    # cache holds the entire working set with load ≪ 0.1 and essentially
    # no collision misses.  The three arrays total ~96 KiB, fitting in L2.
    # Reset cost (memset to -1) ~3 µs per b — small vs the spin-2 work
    # it amortises.
    N_CACHE = 1024
    CMASK = N_CACHE - 1

    if has_qu:
        for b in numba.prange(B):
            # Per-b cache.  Numba's NRT allocator handles this in ~100 ns
            # plus the ~3 µs memset; going thread-local would force
            # ``cache=False`` (dynamic-globals NumbaWarning), which costs
            # 5-15 s of recompile per fresh worker process — far worse.
            cache_pix = np.full(N_CACHE, -1, dtype=np.int64)
            cache_c2d = np.empty(N_CACHE, dtype=np.float64)
            cache_s2d = np.empty(N_CACHE, dtype=np.float64)

            # Boresight (z, sin θ, φ) once per b
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

                # ── Inlined HEALPix bilinear neighbour lookup ──────────────
                # Same math as _ring_interp_single_jit (acos-exact), expanded
                # so we keep the per-neighbour (z_n, φ_n) needed by the
                # spin-2 miss path WITHOUT having to recompute via
                # _pix2zphi_ring_jit.  This makes a cache miss cost the same
                # as the old inline-spin-2 kernel, and a cache hit ~3× faster.
                ir_above = _ring_above_jit(nside, z)
                ir_below = ir_above + 1

                if ir_above == 0:
                    # ── North-pole boundary ────────────────────────────────
                    na, fpa, phi0a, dphia = _ring_info_jit(nside, 1, npix_total)
                    tw = ((phi_w - phi0a) / dphia) % float(na)
                    ip_a = int(tw)
                    frac = tw - ip_a
                    ip_a2 = (ip_a + 1) % na
                    p0 = fpa + (ip_a + 2) % na
                    p1 = fpa + (ip_a2 + 2) % na
                    p2 = fpa + ip_a
                    p3 = fpa + ip_a2
                    za = _ring_z_jit(nside, 1)
                    ta = math.acos(za)
                    theta = math.acos(z)
                    w_theta = theta / ta
                    nf = (1.0 - w_theta) * 0.25
                    w0 = nf
                    w1 = nf
                    w2 = (1.0 - frac) * w_theta + nf
                    w3 = frac * w_theta + nf
                    z_n0 = za
                    z_n1 = za
                    z_n2 = za
                    z_n3 = za
                    phi_n0 = phi0a + ((ip_a + 2) % na) * dphia
                    phi_n1 = phi0a + ((ip_a2 + 2) % na) * dphia
                    phi_n2 = phi0a + ip_a * dphia
                    phi_n3 = phi0a + ip_a2 * dphia

                elif ir_below == 4 * nside:
                    # ── South-pole boundary ────────────────────────────────
                    ir_last = 4 * nside - 1
                    na, fpa, phi0a, dphia = _ring_info_jit(nside, ir_last, npix_total)
                    tw = ((phi_w - phi0a) / dphia) % float(na)
                    ip_a = int(tw)
                    frac = tw - ip_a
                    ip_a2 = (ip_a + 1) % na
                    p0 = fpa + ip_a
                    p1 = fpa + ip_a2
                    p2 = fpa + (ip_a + 2) % na
                    p3 = fpa + (ip_a2 + 2) % na
                    za = _ring_z_jit(nside, ir_last)
                    ta = math.acos(za)
                    theta = math.acos(z)
                    w_theta_south = (theta - ta) / (math.pi - ta)
                    sf = w_theta_south * 0.25
                    w0 = (1.0 - frac) * (1.0 - w_theta_south) + sf
                    w1 = frac * (1.0 - w_theta_south) + sf
                    w2 = sf
                    w3 = sf
                    z_n0 = za
                    z_n1 = za
                    z_n2 = za
                    z_n3 = za
                    phi_n0 = phi0a + ip_a * dphia
                    phi_n1 = phi0a + ip_a2 * dphia
                    phi_n2 = phi0a + ((ip_a + 2) % na) * dphia
                    phi_n3 = phi0a + ((ip_a2 + 2) % na) * dphia

                else:
                    # ── Normal case ────────────────────────────────────────
                    za = _ring_z_jit(nside, ir_above)
                    zb = _ring_z_jit(nside, ir_below)
                    ta = math.acos(za)
                    tb = math.acos(zb)
                    theta = math.acos(z)
                    w_below = (theta - ta) / (tb - ta)
                    w_above = 1.0 - w_below

                    na, fpa, phi0a, dphia = _ring_info_jit(nside, ir_above, npix_total)
                    tw = ((phi_w - phi0a) / dphia) % float(na)
                    iphia = int(tw)
                    fphia = tw - iphia
                    iphia1 = (iphia + 1) % na
                    p0 = fpa + iphia
                    p1 = fpa + iphia1
                    w0 = w_above * (1.0 - fphia)
                    w1 = w_above * fphia

                    nb_, fpb, phi0b, dphib = _ring_info_jit(nside, ir_below, npix_total)
                    tw = ((phi_w - phi0b) / dphib) % float(nb_)
                    iphib = int(tw)
                    fphib = tw - iphib
                    iphib1 = (iphib + 1) % nb_
                    p2 = fpb + iphib
                    p3 = fpb + iphib1
                    w2 = w_below * (1.0 - fphib)
                    w3 = w_below * fphib

                    z_n0 = za
                    z_n1 = za
                    z_n2 = zb
                    z_n3 = zb
                    phi_n0 = phi0a + iphia * dphia
                    phi_n1 = phi0a + iphia1 * dphia
                    phi_n2 = phi0b + iphib * dphib
                    phi_n3 = phi0b + iphib1 * dphib

                # ── Spin-2 lookup for each of the 4 corners ────────────────
                # Cache key: HEALPix pixel index, hashed by Knuth's golden
                # multiplier so consecutive ring neighbours don't collide.
                slot0 = (p0 * 2654435769) & CMASK
                if cache_pix[slot0] == p0:
                    c2d0 = cache_c2d[slot0]
                    s2d0 = cache_s2d[slot0]
                else:
                    sth_n = math.sqrt(max(0.0, 1.0 - z_n0 * z_n0))
                    c2d0, s2d0 = _spin2_cos2d_sin2d_jit(
                        z_n0, sth_n, phi_n0, z_pts, sth_pts, phi_pts
                    )
                    cache_pix[slot0] = p0
                    cache_c2d[slot0] = c2d0
                    cache_s2d[slot0] = s2d0

                slot1 = (p1 * 2654435769) & CMASK
                if cache_pix[slot1] == p1:
                    c2d1 = cache_c2d[slot1]
                    s2d1 = cache_s2d[slot1]
                else:
                    sth_n = math.sqrt(max(0.0, 1.0 - z_n1 * z_n1))
                    c2d1, s2d1 = _spin2_cos2d_sin2d_jit(
                        z_n1, sth_n, phi_n1, z_pts, sth_pts, phi_pts
                    )
                    cache_pix[slot1] = p1
                    cache_c2d[slot1] = c2d1
                    cache_s2d[slot1] = s2d1

                slot2 = (p2 * 2654435769) & CMASK
                if cache_pix[slot2] == p2:
                    c2d2 = cache_c2d[slot2]
                    s2d2 = cache_s2d[slot2]
                else:
                    sth_n = math.sqrt(max(0.0, 1.0 - z_n2 * z_n2))
                    c2d2, s2d2 = _spin2_cos2d_sin2d_jit(
                        z_n2, sth_n, phi_n2, z_pts, sth_pts, phi_pts
                    )
                    cache_pix[slot2] = p2
                    cache_c2d[slot2] = c2d2
                    cache_s2d[slot2] = s2d2

                slot3 = (p3 * 2654435769) & CMASK
                if cache_pix[slot3] == p3:
                    c2d3 = cache_c2d[slot3]
                    s2d3 = cache_s2d[slot3]
                else:
                    sth_n = math.sqrt(max(0.0, 1.0 - z_n3 * z_n3))
                    c2d3, s2d3 = _spin2_cos2d_sin2d_jit(
                        z_n3, sth_n, phi_n3, z_pts, sth_pts, phi_pts
                    )
                    cache_pix[slot3] = p3
                    cache_c2d[slot3] = c2d3
                    cache_s2d[slot3] = s2d3

                # ── Accumulate ────────────────────────────────────────────
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
        # ── No Q/U: single-pass fused vec2ang + gather + accumulate ──────
        # No dedup needed — there is no spin-2 cost to amortise.  Skipping
        # the scratch allocation and the pass split avoids ~6 array allocs
        # per boresight and keeps everything in registers.
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
