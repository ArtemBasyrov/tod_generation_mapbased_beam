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
    _TWO_THIRDS,
    _ring_above_jit,
    _ring_info_jit,
    _ring_z_jit,
    _get_interp_weights_jit,
    get_interp_weights_numba,
    _pix2z_cosphi_sinphi_jit,
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


@numba.jit(nopython=True, cache=True)
def _spin2_cos2d_sin2d_trig_free_jit(
    z_pix, sth_pix, cphi_pix, sphi_pix, z_pts, sth_pts, cphi_pts, sphi_pts
):
    """Trig-free variant of _spin2_cos2d_sin2d_jit.

    Takes precomputed (cos φ, sin φ) for both pixel and boresight instead of raw
    φ values, eliminating the two math.cos/math.sin calls from the inner body.
    The loop over unique pixels in _gather_accum_dedup_jit then contains no trig
    and LLVM can auto-vectorize it with AVX2/AVX-512.

    Parameters
    ----------
    z_pix, sth_pix, cphi_pix, sphi_pix : float  pixel  (cos θ, sin θ, cos φ, sin φ)
    z_pts, sth_pts, cphi_pts, sphi_pts  : float  boresight (cos θ, sin θ, cos φ, sin φ)
    """
    if (
        cphi_pts == cphi_pix
        and sphi_pts == sphi_pix
        and sth_pts == sth_pix
        and z_pts == z_pix
    ):
        return 1.0, 0.0

    # cos(Δφ) and sin(Δφ) via angle-subtraction — no trig calls
    cos_dphi = cphi_pts * cphi_pix + sphi_pts * sphi_pix
    sin_dphi = sphi_pts * cphi_pix - cphi_pts * sphi_pix
    sh2 = (1.0 - cos_dphi) * 0.5  # sin²(Δφ/2)

    dz = z_pts - z_pix
    ds = sth_pts - sth_pix
    st2 = 0.25 * (dz * dz + ds * ds)
    S = sth_pix * sth_pts
    h = st2 + S * sh2

    sin2_dtheta = 4.0 * st2 * (1.0 - st2)

    z_sum = z_pts + z_pix
    N = 2.0 * sin_dphi * z_sum * h

    C = z_pts * z_pix
    term1 = sin2_dtheta * cos_dphi
    term2 = 4.0 * S * C * sh2 * sh2
    term3 = S * sin_dphi * sin_dphi
    D = term1 - term2 + term3

    denom = N * N + D * D
    if denom == 0.0:
        return 1.0, 0.0
    u = 1.0 / denom
    return (D * D - N * N) * u, 2.0 * N * D * u


# ── Fully fused kernel ────────────────────────────────────────────────────────


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
    """
    Fully fused vec2ang + HEALPix bilinear interpolation + beam accumulation.

    Parallelised over the B (sample) dimension; each ``b`` owns ``tod[:, b]``
    exclusively so there are no write races.  Inlines the HEALPix RING
    get_interpol algorithm so Numba can apply cross-step optimisations.

    Spin-2 Q/U handling (active when c_q >= 0 and c_u >= 0)
    --------------------------------------------------------
    For each HEALPix neighbour pixel, (Q, U) is rotated from its own local
    frame into the boresight frame at sample b via the spherical-triangle
    bearing formula (δ = α − γ).  The rotation is performed inline within
    the single beam-pixel loop, immediately before accumulation.

    Parameters
    ----------
    vec_rot    : (B, Sc, 3)   float32   rotated beam unit vectors
    nside      : int
    mp_stacked : (C, N_hp)    float32   stacked sky-map components
    beam_vals  : (Sc,)        float32   beam weights for this tile
    B, Sc      : int
    tod        : (C, B)       float64   accumulated in place
    ax_pts     : (B, 3)       float32   boresight unit vectors
    n_target   : (B, 3)       float32   boresight local-north unit vectors
                                        (unused; kept for API compatibility)
    c_q        : int          index of Q in C-dim of mp_stacked (−1 = absent)
    c_u        : int          index of U in C-dim of mp_stacked (−1 = absent)
    """
    C = mp_stacked.shape[0]
    npix_total = 12 * nside * nside
    has_qu = c_q >= 0 and c_u >= 0

    for b in numba.prange(B):
        # Boresight (z, sin θ, φ) — used only when Q/U correction is active.
        if has_qu:
            bx = float(ax_pts[b, 0])
            by = float(ax_pts[b, 1])
            bz = float(ax_pts[b, 2])
            z_pts = bz
            z_pts = max(-1.0, min(1.0, z_pts))
            sth_pts = math.sqrt(max(0.0, 1.0 - bz * bz))
            phi_pts = math.atan2(by, bx)
            if phi_pts < 0.0:
                phi_pts += _TWO_PI

        for s in range(Sc):
            vx = float(vec_rot[b, s, 0])
            vy = float(vec_rot[b, s, 1])
            vz = float(vec_rot[b, s, 2])
            theta = math.atan2(math.sqrt(vx * vx + vy * vy), vz)
            phi_w = math.atan2(vy, vx)
            if phi_w < 0.0:
                phi_w += _TWO_PI

            z = vz  # unit vector → cos θ = z component
            z = max(-1.0, min(1.0, vz))
            bv = float(beam_vals[s])

            ir_above = _ring_above_jit(nside, z)
            ir_below = ir_above + 1

            if ir_above == 0:
                # ── North-pole boundary ──────────────────────────────────────
                na, fpa, phi0a, dphia = _ring_info_jit(nside, 1, npix_total)
                tw = ((phi_w - phi0a) / dphia) % float(na)
                ip_a = int(tw)
                frac = tw - ip_a
                ip_a2 = (ip_a + 1) % na
                ip0 = fpa + (ip_a + 2) % na
                ip1 = fpa + (ip_a2 + 2) % na
                ip2 = fpa + ip_a
                ip3 = fpa + ip_a2
                za = _ring_z_jit(nside, 1)
                ta = math.acos(za)
                w_theta = theta / ta
                nf = (1.0 - w_theta) * 0.25
                w0 = nf
                w1 = nf
                w2 = (1.0 - frac) * w_theta + nf
                w3 = frac * w_theta + nf
                zn = za  # all 4 neighbours share ring 1
                sin_zn = math.sqrt(max(0.0, 1.0 - za * za))
                phi_n0 = phi0a + ip0 * dphia
                phi_n1 = phi0a + ip1 * dphia
                phi_n2 = phi0a + ip_a * dphia
                phi_n3 = phi0a + ip_a2 * dphia

            elif ir_below == 4 * nside:
                # ── South-pole boundary ──────────────────────────────────────
                ir_last = 4 * nside - 1
                na, fpa, phi0a, dphia = _ring_info_jit(nside, ir_last, npix_total)
                tw = ((phi_w - phi0a) / dphia) % float(na)
                ip_a = int(tw)
                frac = tw - ip_a
                ip_a2 = (ip_a + 1) % na
                ip0 = fpa + ip_a
                ip1 = fpa + ip_a2
                ip2 = (ip_a + 2) % na + fpa
                ip3 = (ip_a2 + 2) % na + fpa
                za = _ring_z_jit(nside, ir_last)
                ta = math.acos(za)
                w_theta_south = (theta - ta) / (math.pi - ta)
                sf = w_theta_south * 0.25
                w0 = (1.0 - frac) * (1.0 - w_theta_south) + sf
                w1 = frac * (1.0 - w_theta_south) + sf
                w2 = sf
                w3 = sf
                zn = za  # all 4 neighbours share the last ring
                sin_zn = math.sqrt(max(0.0, 1.0 - za * za))
                phi_n0 = phi0a + (ip0 - fpa) * dphia
                phi_n1 = phi0a + (ip1 - fpa) * dphia
                phi_n2 = phi0a + ((ip2 - fpa) % na) * dphia
                phi_n3 = phi0a + ((ip3 - fpa) % na) * dphia

            else:
                # ── Normal case ──────────────────────────────────────────────
                za = _ring_z_jit(nside, ir_above)
                zb = _ring_z_jit(nside, ir_below)
                ta = math.acos(za)
                tb = math.acos(zb)
                w_below = (theta - ta) / (tb - ta)
                w_above = 1.0 - w_below

                na, fpa, phi0a, dphia = _ring_info_jit(nside, ir_above, npix_total)
                tw = ((phi_w - phi0a) / dphia) % float(na)
                iphia = int(tw)
                fphia = tw - iphia
                ip0 = fpa + iphia
                ip1 = fpa + (iphia + 1) % na
                w0 = w_above * (1.0 - fphia)
                w1 = w_above * fphia

                nb, fpb, phi0b, dphib = _ring_info_jit(nside, ir_below, npix_total)
                tw = ((phi_w - phi0b) / dphib) % float(nb)
                iphib = int(tw)
                fphib = tw - iphib
                ip2 = fpb + iphib
                ip3 = fpb + (iphib + 1) % nb
                w2 = w_below * (1.0 - fphib)
                w3 = w_below * fphib

                # z and phi for each neighbour
                sin_za = math.sqrt(max(0.0, 1.0 - za * za))
                sin_zb = math.sqrt(max(0.0, 1.0 - zb * zb))
                phi_n0 = phi0a + iphia * dphia
                phi_n1 = phi0a + ((iphia + 1) % na) * dphia
                phi_n2 = phi0b + iphib * dphib
                phi_n3 = phi0b + ((iphib + 1) % nb) * dphib

            # ── Accumulate ───────────────────────────────────────────────────

            if has_qu:
                # For the normal case, neighbours 0/1 are on ring above and
                # 2/3 are on ring below.  For polar boundaries all four share
                # the same ring (zn, sin_zn).  Assign the correct (z, sin_z)
                # per neighbour.
                if ir_above == 0 or ir_below == 4 * nside:
                    za0 = zn
                    sa0 = sin_zn
                    za1 = zn
                    sa1 = sin_zn
                    za2 = zn
                    sa2 = sin_zn
                    za3 = zn
                    sa3 = sin_zn
                else:
                    za0 = za
                    sa0 = sin_za
                    za1 = za
                    sa1 = sin_za
                    za2 = zb
                    sa2 = sin_zb
                    za3 = zb
                    sa3 = sin_zb

                c2d0, s2d0 = _spin2_cos2d_sin2d_jit(
                    za0, sa0, phi_n0, z_pts, sth_pts, phi_pts
                )
                c2d1, s2d1 = _spin2_cos2d_sin2d_jit(
                    za1, sa1, phi_n1, z_pts, sth_pts, phi_pts
                )
                c2d2, s2d2 = _spin2_cos2d_sin2d_jit(
                    za2, sa2, phi_n2, z_pts, sth_pts, phi_pts
                )
                c2d3, s2d3 = _spin2_cos2d_sin2d_jit(
                    za3, sa3, phi_n3, z_pts, sth_pts, phi_pts
                )

                q0 = float(mp_stacked[c_q, ip0])
                u0 = float(mp_stacked[c_u, ip0])
                q1 = float(mp_stacked[c_q, ip1])
                u1 = float(mp_stacked[c_u, ip1])
                q2 = float(mp_stacked[c_q, ip2])
                u2 = float(mp_stacked[c_u, ip2])
                q3 = float(mp_stacked[c_q, ip3])
                u3 = float(mp_stacked[c_u, ip3])

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
                    if c != c_q and c != c_u:
                        tod[c, b] += (
                            w0 * float(mp_stacked[c, ip0])
                            + w1 * float(mp_stacked[c, ip1])
                            + w2 * float(mp_stacked[c, ip2])
                            + w3 * float(mp_stacked[c, ip3])
                        ) * bv
            else:
                for c in range(C):
                    tod[c, b] += (
                        w0 * float(mp_stacked[c, ip0])
                        + w1 * float(mp_stacked[c, ip1])
                        + w2 * float(mp_stacked[c, ip2])
                        + w3 * float(mp_stacked[c, ip3])
                    ) * bv


# ── Dedup kernel ──────────────────────────────────────────────────────────────


@numba.jit(nopython=True, parallel=True, cache=True)
def _gather_accum_dedup_jit(
    vec_rot,
    nside,
    mp_stacked,
    beam_vals,
    B,
    Sc,
    tod,
    ax_pts,
    c_q=-1,
    c_u=-1,
):
    """HEALPix bilinear gather + spin-2 dedup + beam accumulation.

    Drop-in replacement for _gather_accum_fused_jit that eliminates redundant
    spin-2 computations by deduplicating HEALPix neighbour pixels per boresight
    before evaluating _spin2_cos2d_sin2d_trig_free_jit.

    Algorithm (per boresight b, inside prange)
    ------------------------------------------
    Pass 1  Inline HEALPix ring logic for all Sc beam pixels.  Stores the four
            neighbour pixel indices in pix_flat[4*Sc] (layout: corner k occupies
            slice [k*Sc:(k+1)*Sc]) and the four bilinear weights in w{0..3}[Sc].

    Pass 2  (only when c_q >= 0 and c_u >= 0)
      2a.   np.sort(pix_flat) → sorted unique list.  At 1-arcmin beam pixels and
            nside=1024 (~3.4 arcmin pixels), ~12 beam pixels share each HEALPix
            pixel, so n_uniq ≈ Sc/12 — spin-2 cost drops proportionally.
      2b.   Boresight (cos φ, sin φ) extracted directly from the unit-vector
            components bx/sth, by/sth — avoids atan2 + cos + sin entirely.
      2c.   Loop over pix_unique: _pix2z_cosphi_sinphi_jit + trig-free spin-2.
            No trig in this loop; LLVM auto-vectorises with AVX2/AVX-512.
      2d.   inv_idx[4*Sc] via np.searchsorted — O(4·Sc·log n_uniq) lookups,
            done as a single pre-pass so pix_unique stays hot in cache.

    Pass 3  Accumulate Q/U (spin-2 weighted) and all other components (standard
            bilinear) using the pre-computed inv_idx.

    Parameters
    ----------
    vec_rot    : (B, Sc, 3)   float32   rotated beam unit vectors
    nside      : int
    mp_stacked : (C, N_hp)    float32   stacked sky-map components
    beam_vals  : (Sc,)        float32   beam weights for this tile
    B, Sc      : int
    tod        : (C, B)       float64   accumulated in place
    ax_pts     : (B, 3)       float32   boresight unit vectors
    c_q        : int          index of Q in C-dim of mp_stacked (-1 = absent)
    c_u        : int          index of U in C-dim of mp_stacked (-1 = absent)
    """
    C = mp_stacked.shape[0]
    npix_total = 12 * nside * nside
    has_qu = c_q >= 0 and c_u >= 0
    n_flat = 4 * Sc

    for b in numba.prange(B):
        # ── thread-local scratch (heap-allocated per prange iteration) ─────────
        pix_flat = np.empty(n_flat, dtype=np.int64)
        w0_arr = np.empty(Sc, dtype=np.float64)
        w1_arr = np.empty(Sc, dtype=np.float64)
        w2_arr = np.empty(Sc, dtype=np.float64)
        w3_arr = np.empty(Sc, dtype=np.float64)

        # ── Pass 1: inline HEALPix ring lookup ────────────────────────────────
        for s in range(Sc):
            vx = float(vec_rot[b, s, 0])
            vy = float(vec_rot[b, s, 1])
            vz = float(vec_rot[b, s, 2])
            theta = math.atan2(math.sqrt(vx * vx + vy * vy), vz)
            phi_w = math.atan2(vy, vx)
            if phi_w < 0.0:
                phi_w += _TWO_PI
            z = max(-1.0, min(1.0, vz))

            ir_above = _ring_above_jit(nside, z)
            ir_below = ir_above + 1

            if ir_above == 0:
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
                w_theta = theta / ta
                nf = (1.0 - w_theta) * 0.25
                w0 = nf
                w1 = nf
                w2 = (1.0 - frac) * w_theta + nf
                w3 = frac * w_theta + nf

            elif ir_below == 4 * nside:
                ir_last = 4 * nside - 1
                na, fpa, phi0a, dphia = _ring_info_jit(nside, ir_last, npix_total)
                tw = ((phi_w - phi0a) / dphia) % float(na)
                ip_a = int(tw)
                frac = tw - ip_a
                ip_a2 = (ip_a + 1) % na
                p0 = fpa + ip_a
                p1 = fpa + ip_a2
                p2 = (ip_a + 2) % na + fpa
                p3 = (ip_a2 + 2) % na + fpa
                za = _ring_z_jit(nside, ir_last)
                ta = math.acos(za)
                w_theta_s = (theta - ta) / (math.pi - ta)
                sf = w_theta_s * 0.25
                w0 = (1.0 - frac) * (1.0 - w_theta_s) + sf
                w1 = frac * (1.0 - w_theta_s) + sf
                w2 = sf
                w3 = sf

            else:
                za = _ring_z_jit(nside, ir_above)
                zb = _ring_z_jit(nside, ir_below)
                ta = math.acos(za)
                tb = math.acos(zb)
                w_below = (theta - ta) / (tb - ta)
                w_above = 1.0 - w_below

                na, fpa, phi0a, dphia = _ring_info_jit(nside, ir_above, npix_total)
                tw = ((phi_w - phi0a) / dphia) % float(na)
                iphia = int(tw)
                fphia = tw - iphia
                p0 = fpa + iphia
                p1 = fpa + (iphia + 1) % na
                w0 = w_above * (1.0 - fphia)
                w1 = w_above * fphia

                nb, fpb, phi0b, dphib = _ring_info_jit(nside, ir_below, npix_total)
                tw = ((phi_w - phi0b) / dphib) % float(nb)
                iphib = int(tw)
                fphib = tw - iphib
                p2 = fpb + iphib
                p3 = fpb + (iphib + 1) % nb
                w2 = w_below * (1.0 - fphib)
                w3 = w_below * fphib

            pix_flat[s] = p0
            pix_flat[Sc + s] = p1
            pix_flat[2 * Sc + s] = p2
            pix_flat[3 * Sc + s] = p3
            w0_arr[s] = w0
            w1_arr[s] = w1
            w2_arr[s] = w2
            w3_arr[s] = w3

        # ── Pass 2: dedup + spin-2 (Q/U only) ─────────────────────────────────
        if has_qu:
            # 2a. sort pix_flat; build pix_unique
            sorted_pix = np.sort(pix_flat)
            n_uniq = 1
            for i in range(1, n_flat):
                if sorted_pix[i] != sorted_pix[i - 1]:
                    n_uniq += 1
            pix_unique = np.empty(n_uniq, dtype=np.int64)
            pix_unique[0] = sorted_pix[0]
            j = 0
            for i in range(1, n_flat):
                if sorted_pix[i] != sorted_pix[i - 1]:
                    j += 1
                    pix_unique[j] = sorted_pix[i]

            # 2b. boresight (cos φ, sin φ) — extracted from unit vector,
            #     avoids atan2 + cos + sin
            bx = float(ax_pts[b, 0])
            by = float(ax_pts[b, 1])
            bz = float(ax_pts[b, 2])
            z_pts = max(-1.0, min(1.0, bz))
            sth_pts = math.sqrt(max(0.0, 1.0 - bz * bz))
            if sth_pts > 1e-10:
                cphi_pts = bx / sth_pts
                sphi_pts = by / sth_pts
            else:
                cphi_pts = 1.0
                sphi_pts = 0.0

            # 2c. trig-free spin-2 for each unique pixel — no trig in this loop
            c2d_u = np.empty(n_uniq, dtype=np.float64)
            s2d_u = np.empty(n_uniq, dtype=np.float64)
            for p in range(n_uniq):
                z_p, cphi_p, sphi_p = _pix2z_cosphi_sinphi_jit(nside, pix_unique[p])
                sth_p = math.sqrt(max(0.0, 1.0 - z_p * z_p))
                c2d_u[p], s2d_u[p] = _spin2_cos2d_sin2d_trig_free_jit(
                    z_p,
                    sth_p,
                    cphi_p,
                    sphi_p,
                    z_pts,
                    sth_pts,
                    cphi_pts,
                    sphi_pts,
                )

            # 2d. inv_idx: pre-compute all searchsorted results in one pass
            #     so pix_unique stays in cache throughout
            inv_idx = np.empty(n_flat, dtype=np.int64)
            for i in range(n_flat):
                inv_idx[i] = np.searchsorted(pix_unique, pix_flat[i])

        # ── Pass 3: accumulate ─────────────────────────────────────────────────
        for s in range(Sc):
            bv = float(beam_vals[s])
            p0 = pix_flat[s]
            p1 = pix_flat[Sc + s]
            p2 = pix_flat[2 * Sc + s]
            p3 = pix_flat[3 * Sc + s]
            w0 = w0_arr[s]
            w1 = w1_arr[s]
            w2 = w2_arr[s]
            w3 = w3_arr[s]

            if has_qu:
                c2d0 = c2d_u[inv_idx[s]]
                s2d0 = s2d_u[inv_idx[s]]
                c2d1 = c2d_u[inv_idx[Sc + s]]
                s2d1 = s2d_u[inv_idx[Sc + s]]
                c2d2 = c2d_u[inv_idx[2 * Sc + s]]
                s2d2 = s2d_u[inv_idx[2 * Sc + s]]
                c2d3 = c2d_u[inv_idx[3 * Sc + s]]
                s2d3 = s2d_u[inv_idx[3 * Sc + s]]

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
