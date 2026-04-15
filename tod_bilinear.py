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

    Uses the spherical triangle bearing formulas; avoids atan2 by working
    directly with (cos α, sin α) and (cos γ, sin γ) and applying the
    double-angle identities.

    Parameters
    ----------
    z_pix, sth_pix, phi_pix : float   cos θ, sin θ, φ of the HEALPix pixel
    z_pts, sth_pts, phi_pts : float   cos θ, sin θ, φ of the boresight

    Returns
    -------
    cos_2d, sin_2d : float
    """
    dphi = phi_pts - phi_pix
    cd = math.cos(dphi)
    sd = math.sin(dphi)

    cos_beta = sth_pix * sth_pts * cd + z_pix * z_pts
    cos_beta = max(-1.0, min(1.0, cos_beta))
    sin_beta = math.sqrt(max(0.0, 1.0 - cos_beta * cos_beta))

    if sin_beta < 1e-12:
        return 1.0, 0.0

    #isb = 1.0 / sin_beta

    # Bearing at pixel toward boresight (alpha)
    sa = sth_pts * sd / sin_beta #* isb
    ca = (sth_pts * z_pix * cd - z_pts * sth_pix) / sin_beta #* isb
    #ca = max(-1.0, min(1.0, ca))
    a = math.atan2(sa, ca)

    # Bearing at boresight toward pixel (gamma)
    sg = sth_pix * sd / sin_beta #* isb
    cg = -(sth_pix * z_pts * cd - z_pix * sth_pts) / sin_beta #* isb
    #cg = max(-1.0, min(1.0, cg))
    g = math.atan2(sg, cg)

    # delta = alpha - gamma  →  cos/sin via subtraction formula
    #cos_d = ca * cg + sa * sg
    #sin_d = sa * cg - ca * sg

    d = a - g
    cos_2d = math.cos(2.0 * d)
    sin_2d = math.sin(2.0 * d)

    #cos_2d = cos_d * cos_d - sin_d * sin_d
    #sin_2d = 2.0 * cos_d * sin_d
    #cos_2d = max(-1.0, min(1.0, cos_2d))
    #sin_2d = max(-1.0, min(1.0, sin_2d))

    return cos_2d, sin_2d


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
