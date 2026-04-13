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
def _spin2_rodrigues_cos2d_sin2d(
    ri_x,
    ri_y,
    ri_z,
    ni_x,
    ni_y,
    ni_z,
    rq_x,
    rq_y,
    rq_z,
    nq_x,
    nq_y,
    nq_z,
):
    """Compute cos(2δ) and sin(2δ) for spin-2 frame rotation.

    Uses inlined Rodrigues parallel transport with the double-angle trick
    to avoid both sqrt and atan2.  δ is the angle by which the local-north
    frame at neighbour (ri) rotates when parallel-transported along the
    geodesic to query point (rq).

    Unnormalised Rodrigues formula (valid for non-antipodal pairs):
        v = r̂_i × r̂_q
        c = r̂_i · r̂_q
        t = n_i·c + (v × n_i) + v·(v·n_i)/(1+c)

    Then cos(δ) = t · ê_θ(q), sin(δ) = (ê_θ(q) × t) · r̂_q, and
    cos(2δ), sin(2δ) follow from double-angle identities.

    Parameters
    ----------
    ri_x, ri_y, ri_z : float   Neighbour position unit vector
    ni_x, ni_y, ni_z : float   Neighbour north direction ê_θ(i)
    rq_x, rq_y, rq_z : float   Query position unit vector
    nq_x, nq_y, nq_z : float   Query north direction ê_θ(q)

    Returns
    -------
    cos_2d, sin_2d : float
    """
    # v = r̂_i × r̂_q (unnormalised)
    vvx = ri_y * rq_z - ri_z * rq_y
    vvy = ri_z * rq_x - ri_x * rq_z
    vvz = ri_x * rq_y - ri_y * rq_x

    # c = r̂_i · r̂_q
    cc = ri_x * rq_x + ri_y * rq_y + ri_z * rq_z

    denom = 1.0 + cc
    if denom < 1.0e-15:
        return 1.0, 0.0  # antipodal / coincident — identity

    # Transported north: t = n_i·c + (v × n_i) + v·(v·n_i)/(1+c)
    vdn = vvx * ni_x + vvy * ni_y + vvz * ni_z
    fac = vdn / denom

    vxn_x = vvy * ni_z - vvz * ni_y
    vxn_y = vvz * ni_x - vvx * ni_z
    vxn_z = vvx * ni_y - vvy * ni_x

    tx = ni_x * cc + vxn_x + vvx * fac
    ty = ni_y * cc + vxn_y + vvy * fac
    tz = ni_z * cc + vxn_z + vvz * fac

    # cos δ = t · ê_θ(q)
    cos_d = tx * nq_x + ty * nq_y + tz * nq_z

    # sin δ = (ê_θ(q) × t) · r̂_q
    sin_d = (
        (nq_y * tz - nq_z * ty) * rq_x
        + (nq_z * tx - nq_x * tz) * rq_y
        + (nq_x * ty - nq_y * tx) * rq_z
    )

    # Double-angle identities — no atan2 needed
    return cos_d * cos_d - sin_d * sin_d, 2.0 * cos_d * sin_d


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
    spin2_corr=0,
    c_q=-1,
    c_u=-1,
):
    """
    Fully fused vec2ang + HEALPix bilinear interpolation + beam accumulation.

    Eliminates the (N,) theta/phi and (4, N) pixels/weights intermediate arrays
    that the split-call version allocates per S-tile.  Parallelised over the B
    (sample) dimension; each ``b`` owns ``tod[:, b]`` exclusively so there are
    no write races.

    Inlines the HEALPix RING get_interpol algorithm so that Numba can see the
    complete computation and apply cross-step optimisations.

    One optional spin-2 correction is applied when c_q >= 0 and c_u >= 0 and
    spin2_corr > 0:

    Neighbour-frame alignment (spin2_corr 1 or 2): rotates each of the 4
    bilinear neighbours into the query-point local frame before interpolating,
    so the interpolated Q/U is in the sky-local frame at (θ_s, φ_s).

    Note: a parallactic-angle boresight-frame rotation (e^{-2iγ}) is NOT applied
    here.  For a symmetric Gaussian beam the spin-2 and scalar convolutions are
    equivalent at the precision of the bilinear interpolation (the azimuthal
    symmetry of the beam causes the e^{-2iγ} correction to integrate to zero).
    Applying it would cancel nearly all off-axis beam pixel contributions, leaving
    only the centre-pixel weight (~0.4% of beam power), suppressing the signal.

    Parameters
    ----------
    vec_rot    : (B, Sc, 3)   float32   rotated beam unit vectors
    nside      : int
    mp_stacked : (C, N_hp)    float32   stacked sky-map components
    beam_vals  : (Sc,)        float32   beam weights for this tile
    B, Sc      : int
    tod        : (C, B)       float64   accumulated in place
    spin2_corr : int          spin-2 correction mode (0 = none, 1 = approx, 2 = exact)
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
            

            # Bilinear interpolation
            # Inlined _get_interp_weights_jit (RING scheme, two-ring bilinear)
            z = vz / math.sqrt(vx * vx + vy * vy + vz * vz)  # cos(theta)
            phi_w = phi  # already in [0, 2π)

            ir_above = _ring_above_jit(nside, z)
            ir_below = ir_above + 1

            if ir_above == 0:
                # North-pole boundary: use ring 1 for all four pixels
                na, fpa, phi0a, dphia = _ring_info_jit(nside, 1, npix_total)
                tw = ((phi_w - phi0a) / dphia) % float(na)
                ip_a = int(tw)
                frac = tw - ip_a
                ip_a2 = (ip_a + 1) % na
                # fpa == 0 for ring 1
                ip0 = (ip_a + 2) % na
                ip1 = (ip_a2 + 2) % na
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
                zb = za
                sin_z_ca = math.sqrt(max(0.0, 1.0 - za * za))
                sin_z_cb = sin_z_ca
                phi_c0 = phi0a + ip0 * dphia
                phi_c1 = phi0a + ip1 * dphia
                phi_c2 = phi0a + ip_a * dphia
                phi_c3 = phi0a + ip_a2 * dphia
            elif ir_below == 4 * nside:
                # South-pole boundary: use last ring for all four pixels
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
                zb = za
                sin_z_ca = math.sqrt(max(0.0, 1.0 - za * za))
                sin_z_cb = sin_z_ca
                phi_c0 = phi0a + (ip0 - fpa) * dphia
                phi_c1 = phi0a + (ip1 - fpa) * dphia
                phi_c2 = phi0a + ((ip2 - fpa) % na) * dphia
                phi_c3 = phi0a + ((ip3 - fpa) % na) * dphia
            else:
                # Normal case: interpolate between ring above (pixels 0,1)
                # and ring below (pixels 2,3)
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

                sin_z_ca = math.sqrt(max(0.0, 1.0 - za * za))
                sin_z_cb = math.sqrt(max(0.0, 1.0 - zb * zb))
                phi_c0 = phi0a + iphia * dphia
                phi_c1 = phi0a + ((iphia + 1) % na) * dphia
                phi_c2 = phi0b + iphib * dphib
                phi_c3 = phi0b + ((iphib + 1) % nb) * dphib

            bv = float(beam_vals[s])

            if spin2_corr > 0 and c_q >= 0 and c_u >= 0:
                # ── Per-neighbour cos(2δ), sin(2δ) for spin-2 frame rotation ──
                # δ_j = parallel-transport angle from neighbour j's local
                # frame to the query point's local frame.
                #
                # Q_j^(q) =  Q_j cos(2δ_j) + U_j sin(2δ_j)
                # U_j^(q) = -Q_j sin(2δ_j) + U_j cos(2δ_j)

                if spin2_corr == 1:
                    # Approximate: δ_j ≈ cos(θ_q) · (φ_j − φ_q)
                    dphi = phi_c0 - phi_w
                    if dphi > math.pi:
                        dphi -= _TWO_PI
                    elif dphi < -math.pi:
                        dphi += _TWO_PI
                    a = 2.0 * z * dphi
                    c2d_0 = math.cos(a)
                    s2d_0 = math.sin(a)

                    dphi = phi_c1 - phi_w
                    if dphi > math.pi:
                        dphi -= _TWO_PI
                    elif dphi < -math.pi:
                        dphi += _TWO_PI
                    a = 2.0 * z * dphi
                    c2d_1 = math.cos(a)
                    s2d_1 = math.sin(a)

                    dphi = phi_c2 - phi_w
                    if dphi > math.pi:
                        dphi -= _TWO_PI
                    elif dphi < -math.pi:
                        dphi += _TWO_PI
                    a = 2.0 * z * dphi
                    c2d_2 = math.cos(a)
                    s2d_2 = math.sin(a)

                    dphi = phi_c3 - phi_w
                    if dphi > math.pi:
                        dphi -= _TWO_PI
                    elif dphi < -math.pi:
                        dphi += _TWO_PI
                    a = 2.0 * z * dphi
                    c2d_3 = math.cos(a)
                    s2d_3 = math.sin(a)

                else:
                    # Exact: inlined Rodrigues parallel transport + double-angle
                    # Query-point north direction (0 trig — derived from vec)
                    st_q = math.sqrt(vx * vx + vy * vy)
                    if st_q > 1.0e-15:
                        inv_st = 1.0 / st_q
                        nq_x = z * vx * inv_st
                        nq_y = z * vy * inv_st
                        nq_z = -st_q
                    else:
                        # At pole, north undefined; arbitrary tangent direction
                        nq_x = 1.0
                        nq_y = 0.0
                        nq_z = 0.0

                    # cos/sin of each neighbour phi — computed independently so
                    # that pixels 0,1 (ring above, za/sin_z_ca) and pixels 2,3
                    # (ring below, zb/sin_z_cb) use the correct ring geometry.
                    cp0 = math.cos(phi_c0)
                    sp0 = math.sin(phi_c0)
                    cp1 = math.cos(phi_c1)
                    sp1 = math.sin(phi_c1)
                    cp2 = math.cos(phi_c2)
                    sp2 = math.sin(phi_c2)
                    cp3 = math.cos(phi_c3)
                    sp3 = math.sin(phi_c3)

                    # ip0, ip1 come from ring above (za, sin_z_ca)
                    c2d_0, s2d_0 = _spin2_rodrigues_cos2d_sin2d(
                        sin_z_ca * cp0,
                        sin_z_ca * sp0,
                        za,
                        za * cp0,
                        za * sp0,
                        -sin_z_ca,
                        vx,
                        vy,
                        vz,
                        nq_x,
                        nq_y,
                        nq_z,
                    )
                    c2d_1, s2d_1 = _spin2_rodrigues_cos2d_sin2d(
                        sin_z_ca * cp1,
                        sin_z_ca * sp1,
                        za,
                        za * cp1,
                        za * sp1,
                        -sin_z_ca,
                        vx,
                        vy,
                        vz,
                        nq_x,
                        nq_y,
                        nq_z,
                    )
                    # ip2, ip3 come from ring below (zb, sin_z_cb)
                    c2d_2, s2d_2 = _spin2_rodrigues_cos2d_sin2d(
                        sin_z_cb * cp2,
                        sin_z_cb * sp2,
                        zb,
                        zb * cp2,
                        zb * sp2,
                        -sin_z_cb,
                        vx,
                        vy,
                        vz,
                        nq_x,
                        nq_y,
                        nq_z,
                    )
                    c2d_3, s2d_3 = _spin2_rodrigues_cos2d_sin2d(
                        sin_z_cb * cp3,
                        sin_z_cb * sp3,
                        zb,
                        zb * cp3,
                        zb * sp3,
                        -sin_z_cb,
                        vx,
                        vy,
                        vz,
                        nq_x,
                        nq_y,
                        nq_z,
                    )

                # ── Read Q/U from all 4 neighbours ──
                q0 = float(mp_stacked[c_q, ip0])
                u0 = float(mp_stacked[c_u, ip0])
                q1 = float(mp_stacked[c_q, ip1])
                u1 = float(mp_stacked[c_u, ip1])
                q2 = float(mp_stacked[c_q, ip2])
                u2 = float(mp_stacked[c_u, ip2])
                q3 = float(mp_stacked[c_q, ip3])
                u3 = float(mp_stacked[c_u, ip3])

                # ── Rotated Q accumulation ──
                tod[c_q, b] += (
                    (q0 * c2d_0 + u0 * s2d_0) * w0
                    + (q1 * c2d_1 + u1 * s2d_1) * w1
                    + (q2 * c2d_2 + u2 * s2d_2) * w2
                    + (q3 * c2d_3 + u3 * s2d_3) * w3
                ) * bv

                # ── Rotated U accumulation ──
                tod[c_u, b] += (
                    (-q0 * s2d_0 + u0 * c2d_0) * w0
                    + (-q1 * s2d_1 + u1 * c2d_1) * w1
                    + (-q2 * s2d_2 + u2 * c2d_2) * w2
                    + (-q3 * s2d_3 + u3 * c2d_3) * w3
                ) * bv

                # ── T and other scalar components ──
                for c in range(C):
                    if c != c_q and c != c_u:
                        tod[c, b] += (
                            mp_stacked[c, ip0] * w0
                            + mp_stacked[c, ip1] * w1
                            + mp_stacked[c, ip2] * w2
                            + mp_stacked[c, ip3] * w3
                        ) * bv
            else:
                for c in range(C):
                    tod[c, b] += (
                        mp_stacked[c, ip0] * w0
                        + mp_stacked[c, ip1] * w1
                        + mp_stacked[c, ip2] * w2
                        + mp_stacked[c, ip3] * w3
                    ) * bv


@numba.jit(nopython=True, parallel=True, cache=True)
def _gather_accum_flatsky_jit(
    dtheta_tile,
    dphi_tile,
    k_b,
    theta_b,
    phi_b,
    nside,
    mp_stacked,
    beam_vals,
    B,
    Sc,
    tod,
):
    """
    Flat-sky bilinear HEALPix interpolation + beam accumulation.

    Applies the flat-sky approximation to the bilinear interpolation, skipping
    the vec2ang and both Rodrigues rotations.  Computes sky positions directly
    from precomputed (dtheta, dphi) offsets and pointing angles (theta_b, phi_b).
    Parallelised over B.

    Parameters
    ----------
    dtheta_tile : (N_psi, Sc)  float32   flat-sky colatitude offsets
    dphi_tile   : (N_psi, Sc)  float32   flat-sky phi offsets
    k_b         : (B,)         int64     psi-bin indices for each sample
    theta_b     : (B,)         float32   sample colatitude [rad]
    phi_b       : (B,)         float32   sample longitude [rad]
    nside       : int
    mp_stacked  : (C, N_hp)    float32   stacked sky-map components
    beam_vals   : (Sc,)        float32   beam weights for this tile
    B, Sc       : int
    tod         : (C, B)       float64   accumulated in place
    """
    C = mp_stacked.shape[0]
    npix_total = 12 * nside * nside
    N_psi = dtheta_tile.shape[0]

    for b in numba.prange(B):
        # Find the psi-bin index
        k = k_b[b] % N_psi

        # Apply flat-sky offsets
        theta_off = theta_b[b] + dtheta_tile[k, :]
        phi_off = phi_b[b] + dphi_tile[k, :]

        # Wrap phi to [0, 2π)
        phi_off = np.remainder(phi_off, _TWO_PI)

        # Bilinear interpolation for each sample
        for s in range(Sc):
            # vec2ang (inline)
            theta = float(theta_off[s])
            phi = float(phi_off[s])

            # Bilinear interpolation
            # Inlined _get_interp_weights_jit (RING scheme, two-ring bilinear)
            z = math.cos(theta)
            phi_w = phi

            ir_above = _ring_above_jit(nside, z)
            ir_below = ir_above + 1

            if ir_above == 0:
                # North-pole boundary
                na, fpa, phi0a, dphia = _ring_info_jit(nside, 1, npix_total)
                tw = ((phi_w - phi0a) / dphia) % float(na)
                ip_a = int(tw)
                frac = tw - ip_a
                ip_a2 = (ip_a + 1) % na
                ip0 = (ip_a + 2) % na  # fpa == 0 for ring 1
                ip1 = (ip_a2 + 2) % na
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
            elif ir_below == 4 * nside:
                # South-pole boundary
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
            else:
                # Normal case: ring above → pixels 0,1; ring below → pixels 2,3
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

            bv = float(beam_vals[s])
            for c in range(C):
                tod[c, b] += (
                    mp_stacked[c, ip0] * w0
                    + mp_stacked[c, ip1] * w1
                    + mp_stacked[c, ip2] * w2
                    + mp_stacked[c, ip3] * w3
                ) * bv
