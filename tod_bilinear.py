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
from tod_rotations import _spin2_rodrigues_cos2d_sin2d


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
    Each HEALPix pixel's (Q, U) is rotated by parallel transport from its own
    local frame into the **boresight frame** at sample ``b`` (north direction
    ``n_target[b]`` at position ``ax_pts[b]``) *before* the bilinear
    interpolation, so the four-neighbour weighted sum stays spin-2-consistent.

    This replaces the older per-neighbour rotation into the query-point local
    frame at each beam-pixel sky position (θ_s, φ_s).  The rotation target now
    changes once per sample, not once per beam pixel, and it is identical for
    every beam pixel that belongs to the same detector sample.

    The rotation is precomputed: within one ``b`` the work is (a) identifying
    the unique HEALPix pixel indices touched by the 4·Sc bilinear neighbours of
    the Sc beam pixels in the current tile, (b) rotating (Q, U) **once per
    unique pixel**, and (c) reusing those pre-rotated values in the bilinear
    accumulator.  With dense beam sampling and nside-scale HEALPix resolution,
    each unique pixel is typically reused many times, so this is strictly
    fewer rotations than the old per-(b, s, j) scheme.

    Parameters
    ----------
    vec_rot    : (B, Sc, 3)   float32   rotated beam unit vectors
    nside      : int
    mp_stacked : (C, N_hp)    float32   stacked sky-map components
    beam_vals  : (Sc,)        float32   beam weights for this tile
    B, Sc      : int
    tod        : (C, B)       float64   accumulated in place
    ax_pts     : (B, 3)       float32   boresight unit vectors (rotation target
                                        position)
    n_target   : (B, 3)       float32   boresight-frame local-north unit
                                        vectors; see
                                        ``precompute_rotation_vector_batch``
    c_q        : int          index of Q within C-dim of mp_stacked (−1 = absent)
    c_u        : int          index of U within C-dim of mp_stacked (−1 = absent)
    """
    C = mp_stacked.shape[0]
    npix_total = 12 * nside * nside
    n4 = 4 * Sc

    for b in numba.prange(B):
        # Per-b scratch (Numba allocates per prange iteration).
        pixels_local = np.empty((Sc, 4), dtype=np.int64)
        weights_local = np.empty((Sc, 4), dtype=np.float64)
        ring_z_local = np.empty((Sc, 4), dtype=np.float64)
        sin_z_local = np.empty((Sc, 4), dtype=np.float64)
        phi_local = np.empty((Sc, 4), dtype=np.float64)

        # ── Pass 1: inline get_interp_weights per beam pixel, store pixel
        #           indices, weights and (z, sin θ, φ) of each neighbour.
        for s in range(Sc):
            vx = float(vec_rot[b, s, 0])
            vy = float(vec_rot[b, s, 1])
            vz = float(vec_rot[b, s, 2])
            theta = math.atan2(math.sqrt(vx * vx + vy * vy), vz)
            phi = math.atan2(vy, vx)
            if phi < 0.0:
                phi += _TWO_PI

            z = vz / math.sqrt(vx * vx + vy * vy + vz * vz)  # cos θ
            phi_w = phi  # already in [0, 2π)

            ir_above = _ring_above_jit(nside, z)
            ir_below = ir_above + 1

            if ir_above == 0:
                # North-pole boundary: all four pixels on ring 1
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
                # South-pole boundary: all four pixels on the last ring
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
                # Normal case: interpolate between ring above (pixels 0, 1)
                # and ring below (pixels 2, 3)
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

            pixels_local[s, 0] = ip0
            pixels_local[s, 1] = ip1
            pixels_local[s, 2] = ip2
            pixels_local[s, 3] = ip3
            weights_local[s, 0] = w0
            weights_local[s, 1] = w1
            weights_local[s, 2] = w2
            weights_local[s, 3] = w3
            # Pixels 0, 1 live on the ring above (za, sin_z_ca); pixels 2, 3
            # on the ring below (zb, sin_z_cb).  At the polar boundaries these
            # coincide (all four on a single ring).
            ring_z_local[s, 0] = za
            ring_z_local[s, 1] = za
            ring_z_local[s, 2] = zb
            ring_z_local[s, 3] = zb
            sin_z_local[s, 0] = sin_z_ca
            sin_z_local[s, 1] = sin_z_ca
            sin_z_local[s, 2] = sin_z_cb
            sin_z_local[s, 3] = sin_z_cb
            phi_local[s, 0] = phi_c0
            phi_local[s, 1] = phi_c1
            phi_local[s, 2] = phi_c2
            phi_local[s, 3] = phi_c3

        if c_q >= 0 and c_u >= 0:
            # ── Pass 2: sort-based deduplication of the 4·Sc neighbour
            #           pixel indices, with per-unique-pixel pre-rotation of
            #           (Q, U) into the boresight frame at sample b.
            flat_pix = pixels_local.ravel()  # (4*Sc,)
            sort_idx = np.argsort(flat_pix)

            rot_Q = np.empty(n4, dtype=np.float64)
            rot_U = np.empty(n4, dtype=np.float64)
            inv_map = np.empty((Sc, 4), dtype=np.int64)

            rb_x = float(ax_pts[b, 0])
            rb_y = float(ax_pts[b, 1])
            rb_z = float(ax_pts[b, 2])
            nb_x = float(n_target[b, 0])
            nb_y = float(n_target[b, 1])
            nb_z = float(n_target[b, 2])

            n_unique = 0
            prev_pix = -1
            for k in range(n4):
                orig = sort_idx[k]
                s_i = orig // 4
                j_i = orig % 4
                p_i = pixels_local[s_i, j_i]

                if p_i != prev_pix:
                    # New unique pixel: compute rotation factors once and
                    # cache the rotated (Q, U) at index n_unique.
                    z_u = ring_z_local[s_i, j_i]
                    sin_z_u = sin_z_local[s_i, j_i]
                    phi_u = phi_local[s_i, j_i]

                    cp = math.cos(phi_u)
                    sp = math.sin(phi_u)

                    rp_x = sin_z_u * cp
                    rp_y = sin_z_u * sp
                    rp_z = z_u

                    np_x = z_u * cp
                    np_y = z_u * sp
                    np_z = -sin_z_u

                    c2d, s2d = _spin2_rodrigues_cos2d_sin2d(
                        rp_x,
                        rp_y,
                        rp_z,
                        np_x,
                        np_y,
                        np_z,
                        rb_x,
                        rb_y,
                        rb_z,
                        nb_x,
                        nb_y,
                        nb_z,
                    )

                    q_v = float(mp_stacked[c_q, p_i])
                    u_v = float(mp_stacked[c_u, p_i])
                    # Q_b = Q cos(2δ) + U sin(2δ),   U_b = -Q sin(2δ) + U cos(2δ)
                    rot_Q[n_unique] = q_v * c2d + u_v * s2d
                    rot_U[n_unique] = -q_v * s2d + u_v * c2d

                    n_unique += 1
                    prev_pix = p_i

                inv_map[s_i, j_i] = n_unique - 1

            # ── Pass 3: bilinear accumulation ──
            for s in range(Sc):
                bv = float(beam_vals[s])
                w0 = weights_local[s, 0]
                w1 = weights_local[s, 1]
                w2 = weights_local[s, 2]
                w3 = weights_local[s, 3]

                u0 = inv_map[s, 0]
                u1 = inv_map[s, 1]
                u2 = inv_map[s, 2]
                u3 = inv_map[s, 3]

                tod[c_q, b] += (
                    w0 * rot_Q[u0] + w1 * rot_Q[u1] + w2 * rot_Q[u2] + w3 * rot_Q[u3]
                ) * bv
                tod[c_u, b] += (
                    w0 * rot_U[u0] + w1 * rot_U[u1] + w2 * rot_U[u2] + w3 * rot_U[u3]
                ) * bv

                # Scalar components (T): read directly from mp_stacked — no
                # rotation applies.
                p0 = pixels_local[s, 0]
                p1 = pixels_local[s, 1]
                p2 = pixels_local[s, 2]
                p3 = pixels_local[s, 3]
                for c in range(C):
                    if c != c_q and c != c_u:
                        tod[c, b] += (
                            w0 * mp_stacked[c, p0]
                            + w1 * mp_stacked[c, p1]
                            + w2 * mp_stacked[c, p2]
                            + w3 * mp_stacked[c, p3]
                        ) * bv
        else:
            # No Q/U correction — straight bilinear accumulation.
            for s in range(Sc):
                bv = float(beam_vals[s])
                w0 = weights_local[s, 0]
                w1 = weights_local[s, 1]
                w2 = weights_local[s, 2]
                w3 = weights_local[s, 3]
                p0 = pixels_local[s, 0]
                p1 = pixels_local[s, 1]
                p2 = pixels_local[s, 2]
                p3 = pixels_local[s, 3]
                for c in range(C):
                    tod[c, b] += (
                        w0 * mp_stacked[c, p0]
                        + w1 * mp_stacked[c, p1]
                        + w2 * mp_stacked[c, p2]
                        + w3 * mp_stacked[c, p3]
                    ) * bv
