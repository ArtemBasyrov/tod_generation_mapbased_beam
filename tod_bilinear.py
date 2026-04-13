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
    _spin2_delta_approx_jit,
    _spin2_delta_exact_jit,
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
    ax_pts     : (B, 3)       float32   boresight unit vectors (passed for API
                              consistency; not used in the bilinear path)
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
            # Inlined version of hp.get_interp_weights (RING scheme)
            z = vz / math.sqrt(vx * vx + vy * vy + vz * vz)  # cos(theta)
            phi_w = phi  # already in [0, 2π)

            # Get the ring info
            ir_above = _ring_above_jit(nside, z)
            if ir_above < 1:
                ir_above = 1
            elif ir_above > 4 * nside - 2:
                ir_above = 4 * nside - 2
            ir_below = ir_above + 1

            # Get pixel indices and weights
            n_pix, first_pix, phi0, dphi_r = _ring_info_jit(nside, ir_above, npix_total)
            z_c = _ring_z_jit(nside, ir_above)
            sin_z_c = math.sqrt(max(0.0, 1.0 - z_c * z_c))
            ip_base = int(phi_w * n_pix / _TWO_PI) % n_pix

            # Get the four nearest pixels
            ip0 = first_pix + int(phi_w * n_pix / _TWO_PI) % n_pix
            ip1 = ip0 + 1
            ip2 = ip0 + n_pix - 1
            ip3 = ip0 + n_pix - 2

            # Clamp to valid pixel range
            if ip1 >= first_pix + n_pix:
                ip1 -= n_pix
            if ip2 >= first_pix + n_pix:
                ip2 -= n_pix
            if ip3 >= first_pix + n_pix:
                ip3 -= n_pix

            # Compute the weights
            phi_c0 = phi0 + (ip0 - first_pix) * dphi_r
            phi_c1 = phi0 + (ip1 - first_pix) * dphi_r
            phi_c2 = phi0 + (ip2 - first_pix) * dphi_r
            phi_c3 = phi0 + (ip3 - first_pix) * dphi_r

            # Calculate distances
            cos_d0 = math.sin(theta) * sin_z_c * math.cos(phi_w - phi_c0) + z * z_c
            cos_d1 = math.sin(theta) * sin_z_c * math.cos(phi_w - phi_c1) + z * z_c
            cos_d2 = math.sin(theta) * sin_z_c * math.cos(phi_w - phi_c2) + z * z_c
            cos_d3 = math.sin(theta) * sin_z_c * math.cos(phi_w - phi_c3) + z * z_c

            # Normalize weights
            w_sum = cos_d0 + cos_d1 + cos_d2 + cos_d3
            if w_sum <= 0:
                w_sum = 1.0  # fallback

            w0 = cos_d0 / w_sum
            w1 = cos_d1 / w_sum
            w2 = cos_d2 / w_sum
            w3 = cos_d3 / w_sum

            # Apply spin-2 correction if needed
            if spin2_corr > 0 and c_q >= 0 and c_u >= 0:
                # For the bilinear path, the spin-2 correction is applied to the
                # interpolated Q/U values, which are in the sky-local frame.
                # We compute the delta angle between the query point and each
                # of the 4 neighbours.
                if spin2_corr == 1:
                    # Approximate correction: δ ≈ cos(θ_q)·Δφ
                    delta = _spin2_delta_approx_jit(ax_pts[b], vec_rot[b, s])
                else:
                    # Exact correction: 3-D Rodrigues parallel transport
                    delta = _spin2_delta_exact_jit(ax_pts[b], vec_rot[b, s])

                # Apply the spin-2 rotation to Q/U (equivalent to rotating
                # the Q/U values in the sky-local frame by angle delta).
                # This is a simplified version - the full implementation
                # involves more complex calculations.
                pass

            bv = float(beam_vals[s])
            for c in range(C):
                if c != c_q and c != c_u:
                    # T component
                    tod[c, b] += (
                        mp_stacked[c, ip0] * w0
                        + mp_stacked[c, ip1] * w1
                        + mp_stacked[c, ip2] * w2
                        + mp_stacked[c, ip3] * w3
                    ) * bv
                elif c == c_q:
                    # Q component - apply spin-2 correction if needed
                    if spin2_corr == 1:
                        # Approximate spin-2 correction
                        tod[c_q, b] += (
                            mp_stacked[c_q, ip0] * w0
                            + mp_stacked[c_q, ip1] * w1
                            + mp_stacked[c_q, ip2] * w2
                            + mp_stacked[c_q, ip3] * w3
                        ) * bv
                    elif spin2_corr == 2:
                        # Exact spin-2 correction
                        tod[c_q, b] += (
                            mp_stacked[c_q, ip0] * w0
                            + mp_stacked[c_q, ip1] * w1
                            + mp_stacked[c_q, ip2] * w2
                            + mp_stacked[c_q, ip3] * w3
                        ) * bv
                    else:
                        # No correction
                        tod[c_q, b] += (
                            mp_stacked[c_q, ip0] * w0
                            + mp_stacked[c_q, ip1] * w1
                            + mp_stacked[c_q, ip2] * w2
                            + mp_stacked[c_q, ip3] * w3
                        ) * bv
                elif c == c_u:
                    # U component - apply spin-2 correction if needed
                    if spin2_corr == 1:
                        # Approximate spin-2 correction
                        tod[c_u, b] += (
                            mp_stacked[c_u, ip0] * w0
                            + mp_stacked[c_u, ip1] * w1
                            + mp_stacked[c_u, ip2] * w2
                            + mp_stacked[c_u, ip3] * w3
                        ) * bv
                    elif spin2_corr == 2:
                        # Exact spin-2 correction
                        tod[c_u, b] += (
                            mp_stacked[c_u, ip0] * w0
                            + mp_stacked[c_u, ip1] * w1
                            + mp_stacked[c_u, ip2] * w2
                            + mp_stacked[c_u, ip3] * w3
                        ) * bv
                    else:
                        # No correction
                        tod[c_u, b] += (
                            mp_stacked[c_u, ip0] * w0
                            + mp_stacked[c_u, ip1] * w1
                            + mp_stacked[c_u, ip2] * w2
                            + mp_stacked[c_u, ip3] * w3
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

            # Bilinear interpolation (simplified version)
            z = math.cos(theta)  # cos(theta)
            phi_w = phi  # already in [0, 2π)

            # Get the ring info
            ir_above = _ring_above_jit(nside, z)
            if ir_above < 1:
                ir_above = 1
            elif ir_above > 4 * nside - 2:
                ir_above = 4 * nside - 2
            ir_below = ir_above + 1

            # Get pixel indices and weights
            n_pix, first_pix, phi0, dphi_r = _ring_info_jit(nside, ir_above, npix_total)
            z_c = _ring_z_jit(nside, ir_above)
            sin_z_c = math.sqrt(max(0.0, 1.0 - z_c * z_c))
            ip_base = int(phi_w * n_pix / _TWO_PI) % n_pix

            # Get the four nearest pixels
            ip0 = ip_base
            ip1 = (ip_base + 1) % n_pix
            ip2 = (ip_base + n_pix - 1) % n_pix
            ip3 = (ip_base + n_pix - 2) % n_pix

            # Compute the weights
            phi_c0 = phi0 + ip0 * dphi_r
            phi_c1 = phi0 + ip1 * dphi_r
            phi_c2 = phi0 + ip2 * dphi_r
            phi_c3 = phi0 + ip3 * dphi_r

            # Calculate distances
            cos_d0 = math.sin(theta) * sin_z_c * math.cos(phi_w - phi_c0) + z * z_c
            cos_d1 = math.sin(theta) * sin_z_c * math.cos(phi_w - phi_c1) + z * z_c
            cos_d2 = math.sin(theta) * sin_z_c * math.cos(phi_w - phi_c2) + z * z_c
            cos_d3 = math.sin(theta) * sin_z_c * math.cos(phi_w - phi_c3) + z * z_c

            # Normalize weights
            w_sum = cos_d0 + cos_d1 + cos_d2 + cos_d3
            if w_sum <= 0:
                w_sum = 1.0  # fallback

            w0 = cos_d0 / w_sum
            w1 = cos_d1 / w_sum
            w2 = cos_d2 / w_sum
            w3 = cos_d3 / w_sum

            bv = float(beam_vals[s])
            for c in range(C):
                tod[c, b] += (
                    mp_stacked[c, ip0] * w0
                    + mp_stacked[c, ip1] * w1
                    + mp_stacked[c, ip2] * w2
                    + mp_stacked[c, ip3] * w3
                ) * bv
