"""
Core numerical routines for sample-based TOD generation.

All functions are stateless and take only arrays as arguments.

Numba JIT kernels
-----------------
_rodrigues_jit                — fused double Rodrigues rotation (recenter + pol. roll).
                                Writes directly into a pre-allocated (B, S, 3) buffer,
                                eliminating the 4-5 large intermediate arrays that the
                                numpy version creates.

_rodrigues1_from_rolled_jit   — single Rodrigues rotation (recenter only) applied to
                                per-sample pre-rolled (B, Sc, 3) beam vectors loaded
                                from the beam cache.  Replaces the double rotation when
                                vec_rolled / psi_grid are present in beam_data.

_gather_accum_jit       — fused HEALPix bilinear gather + beam-weighted accumulation.
                          Replaces the (C, 4, B*Sc) mp_gathered intermediate and the
                          separate einsum + matmul calls.  Kept for backward compatibility.

_gather_accum_fused_jit   — fully fused vec2ang + HEALPix bilinear interpolation +
                            beam accumulation; prange over B, sequential over S.
                            Replaces the hp.vec2ang + hp.get_interp_weights +
                            _gather_accum_jit triplet in beam_tod_batch, eliminating
                            all four intermediate arrays (theta_flat, phi_flat, pixels,
                            weights) from the hot tile loop.

_gather_accum_flatsky_jit — fully fused flat-sky HEALPix interpolation + accumulation.
                            Skips vec2ang and both Rodrigues rotations; computes sky
                            positions directly from precomputed (dtheta, dphi) offsets
                            and pointing angles (theta_b, phi_b).  Used when the beam
                            cache provides dtheta/dphi alongside vec_rolled/psi_grid.

HEALPix RING helpers (in numba_healpy.py)
-----------------------------------------
_ring_above_jit, _ring_info_jit, _ring_z_jit,
_get_interp_weights_jit, get_interp_weights_numba
"""

import math
import numpy as np
import healpy as hp
import numba

from numba_healpy import (
    _TWO_PI,
    _TWO_THIRDS,
    _ring_above_jit,
    _ring_info_jit,
    _ring_z_jit,
    get_interp_weights_numba,
)

# Target working-set size for the (B × Sc × 3 × float32) vec_rot intermediate.
# Sized to stay within a typical L2 cache (2 MB).
_S_TILE_TARGET_BYTES = 2 * 1024 * 1024

# Maximum number of S-tiles per beam entry.  Each tile makes one call into the
# HEALPix interpolation logic.  Capping at _MAX_TILES ensures Sc is always at
# least S/_MAX_TILES, keeping per-tile overhead bounded while still preventing
# out-of-memory.
_MAX_TILES = 8

_INV_TWO_PI = 1.0 / _TWO_PI


# ── Numba JIT kernels ─────────────────────────────────────────────────────────


@numba.jit(nopython=True, cache=True)
def _rodrigues_jit(vec_orig, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, out):
    """
    Fused double-Rodrigues rotation (recenter + polarisation roll).

    All input/output arrays are float32.  Computes each output element in a
    single (b, s) pass with no temporaries beyond a handful of scalars.

    Parameters
    ----------
    vec_orig : (S, 3)
    axes     : (B, 3)  normalised rotation axes (zero where angle ≈ 0)
    cos_a    : (B,)
    sin_a    : (B,)
    ax_pts   : (B, 3)  pointing unit vectors
    cos_p    : (B,)
    sin_p    : (B,)
    out      : (B, S, 3)  written in place
    """
    B = axes.shape[0]
    S = vec_orig.shape[0]
    for b in range(B):
        kx = axes[b, 0]
        ky = axes[b, 1]
        kz = axes[b, 2]
        ca = cos_a[b]
        sa = sin_a[b]
        oma = 1.0 - ca
        px = ax_pts[b, 0]
        py = ax_pts[b, 1]
        pz = ax_pts[b, 2]
        cp_ = cos_p[b]
        sp_ = sin_p[b]
        omp = 1.0 - cp_
        for s in range(S):
            vx = vec_orig[s, 0]
            vy = vec_orig[s, 1]
            vz = vec_orig[s, 2]
            # Rodrigues 1 – recenter beam
            dkv = kx * vx + ky * vy + kz * vz
            rx = vx * ca + (ky * vz - kz * vy) * sa + kx * dkv * oma
            ry = vy * ca + (kz * vx - kx * vz) * sa + ky * dkv * oma
            rz = vz * ca + (kx * vy - ky * vx) * sa + kz * dkv * oma
            # Rodrigues 2 – polarisation roll
            dpr = px * rx + py * ry + pz * rz
            out[b, s, 0] = rx * cp_ + (py * rz - pz * ry) * sp_ + px * dpr * omp
            out[b, s, 1] = ry * cp_ + (pz * rx - px * rz) * sp_ + py * dpr * omp
            out[b, s, 2] = rz * cp_ + (px * ry - py * rx) * sp_ + pz * dpr * omp


@numba.jit(nopython=True, cache=True)
def _rodrigues1_from_rolled_jit(vec_rolled_b, axes, cos_a, sin_a, out):
    """
    Apply only Rodrigues 1 (recentering) to pre-rolled beam pixel vectors.

    Used when vec_rolled is loaded from the beam cache — the psi-roll
    (Rodrigues 2) is already baked in, so only the recentering rotation
    to the current pointing direction is needed.

    Parameters
    ----------
    vec_rolled_b : (B, Sc, 3)  float32  — per-sample pre-rolled vectors
    axes         : (B, 3)      float32  — Rodrigues rotation axes
    cos_a        : (B,)        float32
    sin_a        : (B,)        float32
    out          : (B, Sc, 3)  float32  — written in place
    """
    B = axes.shape[0]
    Sc = vec_rolled_b.shape[1]
    for b in range(B):
        kx = axes[b, 0]
        ky = axes[b, 1]
        kz = axes[b, 2]
        ca = cos_a[b]
        sa = sin_a[b]
        oma = 1.0 - ca
        for s in range(Sc):
            vx = vec_rolled_b[b, s, 0]
            vy = vec_rolled_b[b, s, 1]
            vz = vec_rolled_b[b, s, 2]
            dkv = kx * vx + ky * vy + kz * vz
            out[b, s, 0] = vx * ca + (ky * vz - kz * vy) * sa + kx * dkv * oma
            out[b, s, 1] = vy * ca + (kz * vx - kx * vz) * sa + ky * dkv * oma
            out[b, s, 2] = vz * ca + (kx * vy - ky * vx) * sa + kz * dkv * oma


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
def _gather_accum_fused_jit(vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod):
    """
    Fully fused vec2ang + HEALPix bilinear interpolation + beam accumulation.

    Eliminates the (N,) theta/phi and (4, N) pixels/weights intermediate arrays
    that the split-call version allocates per S-tile.  Parallelised over the B
    (sample) dimension; each ``b`` owns ``tod[:, b]`` exclusively so there are
    no write races.

    Inlines the HEALPix RING get_interpol algorithm so that Numba can see the
    complete computation and apply cross-step optimisations.

    Parameters
    ----------
    vec_rot    : (B, Sc, 3)   float32   rotated beam unit vectors
    nside      : int
    mp_stacked : (C, N_hp)    float32   stacked sky-map components
    beam_vals  : (Sc,)        float32   beam weights for this tile
    B, Sc      : int
    tod        : (C, B)       float64   accumulated in place
    """
    C = mp_stacked.shape[0]
    npix_total = 12 * nside * nside

    for b in numba.prange(B):  # ← parallel; no race on tod[:, b]
        for s in range(Sc):
            # ── step 1: vec2ang (inline) ──────────────────────────────────────
            # atan2(r_xy, z) is better conditioned near the poles than acos(z).
            vx = float(vec_rot[b, s, 0])
            vy = float(vec_rot[b, s, 1])
            vz = float(vec_rot[b, s, 2])
            theta = math.atan2(math.sqrt(vx * vx + vy * vy), vz)
            phi = math.atan2(vy, vx)
            if phi < 0.0:
                phi += _TWO_PI

            # ── step 2: HEALPix RING bilinear interpolation (inline) ──────────
            z = math.cos(theta)
            az = abs(z)
            if az > _TWO_THIRDS:  # polar cap
                tp = nside * math.sqrt(3.0 * (1.0 - az))
                ir_above = int(tp)
                if z < 0.0:
                    ir_above = 4 * nside - ir_above - 1
            else:  # equatorial belt
                ir_above = int(nside * (2.0 - 1.5 * z))
            ir_below = ir_above + 1

            if ir_above == 0:
                # ── North-pole boundary ───────────────────────────────────────
                # Ring 1 layout
                na1 = 4
                fpa1 = 0
                sa1 = 1
                dphia1 = _TWO_PI / na1
                phi0a1 = sa1 * dphia1 * 0.5
                tw = ((phi - phi0a1) / dphia1) % float(na1)
                ip = int(tw)
                frac = tw - ip
                ip2 = (ip + 1) % na1
                p2 = fpa1 + ip
                p3 = fpa1 + ip2
                p0 = (ip + 2) % na1
                p1 = (ip2 + 2) % na1
                # theta weight (theta1 = 0 at north pole)
                tmp_r1 = 1.0
                za1 = 1.0 - tmp_r1 * tmp_r1 / (3.0 * nside * nside)
                ta1 = math.acos(za1)
                wt = theta / ta1
                nf = (1.0 - wt) * 0.25
                w0 = nf
                w1 = nf
                w2 = (1.0 - frac) * wt + nf
                w3 = frac * wt + nf

            elif ir_below == 4 * nside:
                # ── South-pole boundary ───────────────────────────────────────
                ir_last = 4 * nside - 1
                i2last = 4 * nside - ir_last  # = 1
                na_l = 4 * i2last  # = 4
                fpa_l = npix_total - 2 * i2last * (i2last + 1)
                dphia_l = _TWO_PI / na_l
                phi0a_l = dphia_l * 0.5  # shift=1 always for last ring
                tw = ((phi - phi0a_l) / dphia_l) % float(na_l)
                ip = int(tw)
                frac = tw - ip
                ip2 = (ip + 1) % na_l
                p0 = fpa_l + ip
                p1 = fpa_l + ip2
                p2 = (ip + 2) % na_l + fpa_l
                p3 = (ip2 + 2) % na_l + fpa_l
                # theta weight toward south pole
                tmp_rl = float(i2last)
                za_l = -(1.0 - tmp_rl * tmp_rl / (3.0 * nside * nside))
                ta_l = math.acos(za_l)
                wts = (theta - ta_l) / (math.pi - ta_l)
                sf = wts * 0.25
                w0 = (1.0 - frac) * (1.0 - wts) + sf
                w1 = frac * (1.0 - wts) + sf
                w2 = sf
                w3 = sf

            else:
                # ── Normal case ───────────────────────────────────────────────
                # z of ring above
                if ir_above < nside:
                    tmp_a = float(ir_above)
                    za = 1.0 - tmp_a * tmp_a / (3.0 * nside * nside)
                elif ir_above <= 3 * nside:
                    za = _TWO_THIRDS * (2.0 - float(ir_above) / nside)
                else:
                    tmp_a = float(4 * nside - ir_above)
                    za = -(1.0 - tmp_a * tmp_a / (3.0 * nside * nside))
                # z of ring below
                if ir_below < nside:
                    tmp_b = float(ir_below)
                    zb = 1.0 - tmp_b * tmp_b / (3.0 * nside * nside)
                elif ir_below <= 3 * nside:
                    zb = _TWO_THIRDS * (2.0 - float(ir_below) / nside)
                else:
                    tmp_b = float(4 * nside - ir_below)
                    zb = -(1.0 - tmp_b * tmp_b / (3.0 * nside * nside))
                ta = math.acos(za)
                tb = math.acos(zb)
                w_below = (theta - ta) / (tb - ta)
                w_above = 1.0 - w_below

                # Ring above: pixel layout + phi interpolation
                if ir_above < nside:
                    na = 4 * ir_above
                    fpa = 2 * ir_above * (ir_above - 1)
                    sa = 1
                elif ir_above <= 3 * nside:
                    na = 4 * nside
                    fpa = 2 * nside * (nside - 1) + (ir_above - nside) * 4 * nside
                    sa = 1 if (ir_above - nside) % 2 == 0 else 0
                else:
                    i2a = 4 * nside - ir_above
                    na = 4 * i2a
                    fpa = npix_total - 2 * i2a * (i2a + 1)
                    sa = 1
                dphia = _TWO_PI / na
                phi0a = sa * dphia * 0.5
                tw = ((phi - phi0a) / dphia) % float(na)
                iphia = int(tw)
                fphia = tw - iphia
                p0 = fpa + iphia
                p1 = fpa + (iphia + 1) % na
                w0 = w_above * (1.0 - fphia)
                w1 = w_above * fphia

                # Ring below: pixel layout + phi interpolation
                if ir_below < nside:
                    nb = 4 * ir_below
                    fpb = 2 * ir_below * (ir_below - 1)
                    sb = 1
                elif ir_below <= 3 * nside:
                    nb = 4 * nside
                    fpb = 2 * nside * (nside - 1) + (ir_below - nside) * 4 * nside
                    sb = 1 if (ir_below - nside) % 2 == 0 else 0
                else:
                    i2b = 4 * nside - ir_below
                    nb = 4 * i2b
                    fpb = npix_total - 2 * i2b * (i2b + 1)
                    sb = 1
                dphib = _TWO_PI / nb
                phi0b = sb * dphib * 0.5
                tw = ((phi - phi0b) / dphib) % float(nb)
                iphib = int(tw)
                fphib = tw - iphib
                p2 = fpb + iphib
                p3 = fpb + (iphib + 1) % nb
                w2 = w_below * (1.0 - fphib)
                w3 = w_below * fphib

            # ── step 3: gather + beam-weighted accumulation ───────────────────
            bv = float(beam_vals[s])
            for c in range(C):
                tod[c, b] += (
                    mp_stacked[c, p0] * w0
                    + mp_stacked[c, p1] * w1
                    + mp_stacked[c, p2] * w2
                    + mp_stacked[c, p3] * w3
                ) * bv


# ── Flat-sky fused kernel ─────────────────────────────────────────────────────


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
    Fully fused flat-sky HEALPix interpolation + beam accumulation.

    Skips both Rodrigues rotations and vec2ang entirely.  Uses precomputed
    angular offsets (dtheta, dphi) from the beam cache to compute sky positions
    directly from the pointing angles (theta_b, phi_b):

        theta_s(b, s) = theta_b[b] + dtheta_tile[k_b[b], s]
        phi_s  (b, s) = phi_b  [b] + dphi_tile  [k_b[b], s] / sin(theta_b[b])

    Then runs the standard HEALPix RING bilinear interpolation and accumulates
    beam-weighted sky-map values into tod.  Parallelised over B (samples).

    Parameters
    ----------
    dtheta_tile : (N_psi, Sc)  float32  — colatitude offsets for this S-tile
    dphi_tile   : (N_psi, Sc)  float32  — raw phi offsets (divide by sin(theta_b))
    k_b         : (B,)         int64    — psi bin index per sample
    theta_b     : (B,)         float32  — pointing colatitude [rad]
    phi_b       : (B,)         float32  — pointing longitude  [rad]
    nside       : int
    mp_stacked  : (C, N_hp)    float32  — stacked sky-map components
    beam_vals   : (Sc,)        float32  — beam weights for this tile
    B, Sc       : int
    tod         : (C, B)       float64  — accumulated in place
    """
    C = mp_stacked.shape[0]
    npix_total = 12 * nside * nside

    for b in numba.prange(B):
        th_point = float(theta_b[b])
        ph_point = float(phi_b[b])
        kb = k_b[b]
        sin_th = math.sin(th_point)
        inv_sin = 1.0 / sin_th if sin_th > 1e-10 else 0.0

        for s in range(Sc):
            theta = th_point + float(dtheta_tile[kb, s])
            phi = ph_point + float(dphi_tile[kb, s]) * inv_sin

            # Clamp theta to [0, π] — flat-sky approximation can push edge
            # beam pixels slightly outside the valid range near the poles.
            if theta < 0.0:
                theta = -theta
                phi += math.pi
            elif theta > math.pi:
                theta = _TWO_PI - theta
                phi += math.pi

            # Wrap phi to [0, 2π)
            if phi < 0.0:
                phi += _TWO_PI
            elif phi >= _TWO_PI:
                phi -= _TWO_PI

            # ── HEALPix RING bilinear interpolation (inline) ──────────────────
            z = math.cos(theta)
            az = abs(z)
            if az > _TWO_THIRDS:
                tp = nside * math.sqrt(3.0 * (1.0 - az))
                ir_above = int(tp)
                if z < 0.0:
                    ir_above = 4 * nside - ir_above - 1
            else:
                ir_above = int(nside * (2.0 - 1.5 * z))
            ir_below = ir_above + 1

            if ir_above == 0:
                na1 = 4
                fpa1 = 0
                sa1 = 1
                dphia1 = _TWO_PI / na1
                phi0a1 = sa1 * dphia1 * 0.5
                tw = ((phi - phi0a1) / dphia1) % float(na1)
                ip = int(tw)
                frac = tw - ip
                ip2 = (ip + 1) % na1
                p2 = fpa1 + ip
                p3 = fpa1 + ip2
                p0 = (ip + 2) % na1
                p1 = (ip2 + 2) % na1
                tmp_r1 = 1.0
                za1 = 1.0 - tmp_r1 * tmp_r1 / (3.0 * nside * nside)
                ta1 = math.acos(za1)
                wt = theta / ta1
                nf = (1.0 - wt) * 0.25
                w0 = nf
                w1 = nf
                w2 = (1.0 - frac) * wt + nf
                w3 = frac * wt + nf

            elif ir_below == 4 * nside:
                ir_last = 4 * nside - 1
                i2last = 4 * nside - ir_last
                na_l = 4 * i2last
                fpa_l = npix_total - 2 * i2last * (i2last + 1)
                dphia_l = _TWO_PI / na_l
                phi0a_l = dphia_l * 0.5
                tw = ((phi - phi0a_l) / dphia_l) % float(na_l)
                ip = int(tw)
                frac = tw - ip
                ip2 = (ip + 1) % na_l
                p0 = fpa_l + ip
                p1 = fpa_l + ip2
                p2 = (ip + 2) % na_l + fpa_l
                p3 = (ip2 + 2) % na_l + fpa_l
                tmp_rl = float(i2last)
                za_l = -(1.0 - tmp_rl * tmp_rl / (3.0 * nside * nside))
                ta_l = math.acos(za_l)
                wts = (theta - ta_l) / (math.pi - ta_l)
                sf = wts * 0.25
                w0 = (1.0 - frac) * (1.0 - wts) + sf
                w1 = frac * (1.0 - wts) + sf
                w2 = sf
                w3 = sf

            else:
                if ir_above < nside:
                    tmp_a = float(ir_above)
                    za = 1.0 - tmp_a * tmp_a / (3.0 * nside * nside)
                elif ir_above <= 3 * nside:
                    za = _TWO_THIRDS * (2.0 - float(ir_above) / nside)
                else:
                    tmp_a = float(4 * nside - ir_above)
                    za = -(1.0 - tmp_a * tmp_a / (3.0 * nside * nside))
                if ir_below < nside:
                    tmp_b = float(ir_below)
                    zb = 1.0 - tmp_b * tmp_b / (3.0 * nside * nside)
                elif ir_below <= 3 * nside:
                    zb = _TWO_THIRDS * (2.0 - float(ir_below) / nside)
                else:
                    tmp_b = float(4 * nside - ir_below)
                    zb = -(1.0 - tmp_b * tmp_b / (3.0 * nside * nside))
                ta = math.acos(za)
                tb = math.acos(zb)
                w_below = (theta - ta) / (tb - ta)
                w_above = 1.0 - w_below

                if ir_above < nside:
                    na = 4 * ir_above
                    fpa = 2 * ir_above * (ir_above - 1)
                    sa = 1
                elif ir_above <= 3 * nside:
                    na = 4 * nside
                    fpa = 2 * nside * (nside - 1) + (ir_above - nside) * 4 * nside
                    sa = 1 if (ir_above - nside) % 2 == 0 else 0
                else:
                    i2a = 4 * nside - ir_above
                    na = 4 * i2a
                    fpa = npix_total - 2 * i2a * (i2a + 1)
                    sa = 1
                dphia = _TWO_PI / na
                phi0a = sa * dphia * 0.5
                tw = ((phi - phi0a) / dphia) % float(na)
                iphia = int(tw)
                fphia = tw - iphia
                p0 = fpa + iphia
                p1 = fpa + (iphia + 1) % na
                w0 = w_above * (1.0 - fphia)
                w1 = w_above * fphia

                if ir_below < nside:
                    nb = 4 * ir_below
                    fpb = 2 * ir_below * (ir_below - 1)
                    sb = 1
                elif ir_below <= 3 * nside:
                    nb = 4 * nside
                    fpb = 2 * nside * (nside - 1) + (ir_below - nside) * 4 * nside
                    sb = 1 if (ir_below - nside) % 2 == 0 else 0
                else:
                    i2b = 4 * nside - ir_below
                    nb = 4 * i2b
                    fpb = npix_total - 2 * i2b * (i2b + 1)
                    sb = 1
                dphib = _TWO_PI / nb
                phi0b = sb * dphib * 0.5
                tw = ((phi - phi0b) / dphib) % float(nb)
                iphib = int(tw)
                fphib = tw - iphib
                p2 = fpb + iphib
                p3 = fpb + (iphib + 1) % nb
                w2 = w_below * (1.0 - fphib)
                w3 = w_below * fphib

            # ── gather + beam-weighted accumulation ───────────────────────────
            bv = float(beam_vals[s])
            for c in range(C):
                tod[c, b] += (
                    mp_stacked[c, p0] * w0
                    + mp_stacked[c, p1] * w1
                    + mp_stacked[c, p2] * w2
                    + mp_stacked[c, p3] * w3
                ) * bv


# ── Nearest-pixel kernels ─────────────────────────────────────────────────────


@numba.jit(nopython=True, parallel=True, cache=True)
def _gather_accum_nearest_jit(vec_rot, nside, mp_stacked, beam_vals, B, Sc, tod):
    """
    Fully fused vec2ang + HEALPix nearest-pixel lookup + beam accumulation.

    For each rotated beam vector the single nearest RING-scheme pixel is found
    via the inline ang2pix algorithm (_ang2pix_ring_jit inlined) and its
    sky-map value is accumulated with the beam weight.  Parallelised over B.

    Parameters
    ----------
    vec_rot    : (B, Sc, 3)   float32   rotated beam unit vectors
    nside      : int
    mp_stacked : (C, N_hp)    float32   stacked sky-map components
    beam_vals  : (Sc,)        float32   beam weights for this tile
    B, Sc      : int
    tod        : (C, B)       float64   accumulated in place
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

            # nearest-pixel lookup (inlined _ang2pix_ring_jit)
            z = vz / math.sqrt(vx * vx + vy * vy + vz * vz)  # cos(theta)
            phi_w = phi  # already in [0, 2π)

            ir_above = _ring_above_jit(nside, z)
            if ir_above < 1:
                ir_above = 1
            elif ir_above > 4 * nside - 2:
                ir_above = 4 * nside - 2
            ir_below = ir_above + 1

            best_pix = 0
            best_cos = -2.0
            sin_th = math.sin(theta)

            for ir_g in (ir_above, ir_below):
                if ir_g < 1 or ir_g > 4 * nside - 1:
                    continue
                n_pix, first_pix, phi0, dphi_r = _ring_info_jit(nside, ir_g, npix_total)
                z_c = _ring_z_jit(nside, ir_g)
                sin_z_c = math.sqrt(max(0.0, 1.0 - z_c * z_c))
                ip_base = int(phi_w * n_pix / _TWO_PI) % n_pix
                for ip_try in (ip_base, (ip_base + 1) % n_pix):
                    phi_c = phi0 + ip_try * dphi_r
                    cos_d = sin_th * sin_z_c * math.cos(phi_w - phi_c) + z * z_c
                    if cos_d > best_cos:
                        best_cos = cos_d
                        best_pix = first_pix + ip_try

            bv = float(beam_vals[s])
            for c in range(C):
                tod[c, b] += mp_stacked[c, best_pix] * bv


@numba.jit(nopython=True, parallel=True, cache=True)
def _gather_accum_nearest_flatsky_jit(
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
    Flat-sky nearest-pixel HEALPix interpolation + beam accumulation.

    Same flat-sky approximation as _gather_accum_flatsky_jit but replaces the
    4-pixel bilinear step with a single nearest-pixel lookup.  Parallelised over B.

    Parameters
    ----------
    dtheta_tile : (N_psi, Sc)  float32  — colatitude offsets for this S-tile
    dphi_tile   : (N_psi, Sc)  float32  — raw phi offsets (divide by sin(theta_b))
    k_b         : (B,)         int64    — psi bin index per sample
    theta_b     : (B,)         float32  — pointing colatitude [rad]
    phi_b       : (B,)         float32  — pointing longitude  [rad]
    nside       : int
    mp_stacked  : (C, N_hp)    float32  — stacked sky-map components
    beam_vals   : (Sc,)        float32  — beam weights for this tile
    B, Sc       : int
    tod         : (C, B)       float64  — accumulated in place
    """
    C = mp_stacked.shape[0]
    npix_total = 12 * nside * nside

    for b in numba.prange(B):
        th_point = float(theta_b[b])
        ph_point = float(phi_b[b])
        kb = k_b[b]
        sin_th = math.sin(th_point)
        inv_sin = 1.0 / sin_th if sin_th > 1e-10 else 0.0

        for s in range(Sc):
            theta = th_point + float(dtheta_tile[kb, s])
            phi = ph_point + float(dphi_tile[kb, s]) * inv_sin

            if theta < 0.0:
                theta = -theta
                phi += math.pi
            elif theta > math.pi:
                theta = _TWO_PI - theta
                phi += math.pi

            if phi < 0.0:
                phi += _TWO_PI
            elif phi >= _TWO_PI:
                phi -= _TWO_PI

            # nearest-pixel lookup
            z = math.cos(theta)
            phi_w = phi

            ir_above = _ring_above_jit(nside, z)
            if ir_above < 1:
                ir_above = 1
            elif ir_above > 4 * nside - 2:
                ir_above = 4 * nside - 2
            ir_below = ir_above + 1

            best_pix = 0
            best_cos = -2.0
            sin_th_s = math.sin(theta)

            for ir_g in (ir_above, ir_below):
                if ir_g < 1 or ir_g > 4 * nside - 1:
                    continue
                n_pix, first_pix, phi0, dphi_r = _ring_info_jit(nside, ir_g, npix_total)
                z_c = _ring_z_jit(nside, ir_g)
                sin_z_c = math.sqrt(max(0.0, 1.0 - z_c * z_c))
                ip_base = int(phi_w * n_pix / _TWO_PI) % n_pix
                for ip_try in (ip_base, (ip_base + 1) % n_pix):
                    phi_c = phi0 + ip_try * dphi_r
                    cos_d = sin_th_s * sin_z_c * math.cos(phi_w - phi_c) + z * z_c
                    if cos_d > best_cos:
                        best_cos = cos_d
                        best_pix = first_pix + ip_try

            bv = float(beam_vals[s])
            for c in range(C):
                tod[c, b] += mp_stacked[c, best_pix] * bv


# ── Public numpy functions ────────────────────────────────────────────────────


def precompute_rotation_vector_batch(ra, dec, phi_batch, theta_batch, center_idx=None):
    """Compute Rodrigues rotation vectors and polarisation angle offsets for a batch.

    For each boresight pointing ``(phi_b, theta_b)`` in the batch, computes
    the Rodrigues rotation vector that brings the beam centre
    ``(ra[center_idx], dec[center_idx])`` to that pointing direction, together
    with the parallactic angle offset ``beta`` needed to align the polarisation
    frame.  The combined rotation angle applied at sample time is
    ``psi_b - beta``.

    Args:
        ra (numpy.ndarray): RA offsets of beam pixels from beam centre [rad],
            shape ``(H, W)``.
        dec (numpy.ndarray): Dec offsets of beam pixels from beam centre [rad],
            shape ``(H, W)``.
        phi_batch (numpy.ndarray): Boresight longitude for each sample [rad],
            shape ``(B,)``.
        theta_batch (numpy.ndarray): Boresight colatitude for each sample [rad],
            shape ``(B,)``.
        center_idx (tuple[int, int] or None): 2-D index of the beam-centre
            pixel in the ``ra``/``dec`` arrays. When ``None`` (default), the
            centre is derived from the array shape as
            ``(ra.shape[0] // 2, ra.shape[1] // 2)``, which matches the
            convention used by :func:`~tod_io.load_beam` and
            :mod:`precompute_beam_cache`.

    Returns:
        tuple:
            - **rot_vector** (*numpy.ndarray*) – Rodrigues rotation vectors
              (axis × angle), shape ``(B, 3)``.
            - **beta** (*numpy.ndarray*) – Polarisation angle offset [rad],
              shape ``(B,)``.  Always in ``[0, 2π)``.
    """
    if center_idx is None:
        center_idx = (ra.shape[0] // 2, ra.shape[1] // 2)

    def sph2vec(phi, theta):
        return np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            axis=-1,
        )

    phi_orig = ra
    theta_orig = np.pi / 2 - dec

    vec_center = sph2vec(phi_orig[center_idx], theta_orig[center_idx])[np.newaxis, :]
    vec_target = sph2vec(phi_batch, theta_batch)

    N_center = np.array(
        [
            np.cos(phi_orig[center_idx]) * np.cos(theta_orig[center_idx]),
            np.sin(phi_orig[center_idx]) * np.cos(theta_orig[center_idx]),
            -np.sin(theta_orig[center_idx]),
        ]
    )
    N_target = np.array(
        [
            np.cos(phi_batch) * np.cos(theta_batch),
            np.sin(phi_batch) * np.cos(theta_batch),
            -np.sin(theta_batch),
        ]
    ).T
    E_target = np.array(
        [-np.sin(phi_batch), np.cos(phi_batch), np.zeros_like(phi_batch)]
    ).T

    axis = np.cross(vec_center, vec_target)
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = np.where(axis_norm > 1e-10, axis / axis_norm, 0)

    angle = np.arccos(np.clip(np.sum(vec_center * vec_target, axis=-1), -1, 1))
    rot_vector = axis * angle[..., np.newaxis]

    ca = np.cos(angle)
    v = N_center[np.newaxis, :]
    dot_kv = np.sum(axis * v, axis=-1, keepdims=True)
    w = (
        v * ca[..., np.newaxis]
        + np.cross(axis, v) * np.sin(angle)[..., np.newaxis]
        + axis * dot_kv * (1 - ca)[..., np.newaxis]
    )

    beta = np.arctan2(np.sum(w * E_target, axis=-1), np.sum(w * N_target, axis=-1))
    beta = np.where(beta < 0, beta + 2 * np.pi, beta)

    return rot_vector, beta


def _rotation_params(rot_vecs, phi_b, theta_b, psis_b):
    """
    Pre-compute the per-sample scalars needed by _rodrigues_jit from the
    Rodrigues vectors and pointing angles.  All outputs are float32.

    Returns axes (B,3), cos_a (B,), sin_a (B,), ax_pts (B,3), cos_p (B,), sin_p (B,)
    """
    angles = np.linalg.norm(rot_vecs, axis=-1).astype(np.float32)  # (B,)
    safe = angles > np.float32(1e-10)
    axes = (
        rot_vecs / np.where(safe[:, None], angles[:, None], np.float32(1.0))
    ).astype(np.float32)
    axes = np.where(safe[:, None], axes, np.float32(0.0))
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    phi_f = np.asarray(phi_b, dtype=np.float32)
    theta_f = np.asarray(theta_b, dtype=np.float32)
    psis_f = np.asarray(psis_b, dtype=np.float32)
    st = np.sin(theta_f)
    ct = np.cos(theta_f)
    sp = np.sin(phi_f)
    cp = np.cos(phi_f)
    ax_pts = np.stack([st * cp, st * sp, ct], axis=-1)
    cos_p = np.cos(psis_f)
    sin_p = np.sin(psis_f)

    return axes, cos_a, sin_a, ax_pts, cos_p, sin_p


def _recenter_and_rotate(vec_orig, rot_vecs, phi_pix, theta_pix, psis):
    """Apply a fused recenter + polarisation-roll rotation to beam pixel vectors.

    Executes the double Rodrigues rotation via the
    :func:`_rodrigues_jit` Numba kernel:

    1. **Recenter** (Rodrigues 1): rotate ``vec_orig`` from the beam-centre
       direction to the boresight direction defined by ``rot_vecs``.
    2. **Pol. roll** (Rodrigues 2): rotate by ``psis`` around the boresight
       unit vector.

    Args:
        vec_orig (numpy.ndarray): Beam-pixel unit vectors in the beam frame,
            shape ``(S, 3)``, dtype ``float32``.
        rot_vecs (numpy.ndarray): Rodrigues rotation vectors (axis × angle)
            from :func:`precompute_rotation_vector_batch`, shape ``(B, 3)``.
        phi_pix (numpy.ndarray): Boresight longitude [rad], shape ``(B,)``.
        theta_pix (numpy.ndarray): Boresight colatitude [rad], shape ``(B,)``.
        psis (numpy.ndarray): Combined rotation angle ``psi_b - beta`` [rad],
            shape ``(B,)``.

    Returns:
        numpy.ndarray: Rotated beam-pixel unit vectors in sky-frame coordinates,
            shape ``(B, S, 3)``, dtype ``float32``.
    """
    B = rot_vecs.shape[0]
    S = vec_orig.shape[0]
    axes, cos_a, sin_a, ax_pts, cos_p, sin_p = _rotation_params(
        rot_vecs, phi_pix, theta_pix, psis
    )
    out = np.empty((B, S, 3), dtype=np.float32)
    _rodrigues_jit(
        np.asarray(vec_orig, dtype=np.float32),
        axes,
        cos_a,
        sin_a,
        ax_pts,
        cos_p,
        sin_p,
        out,
    )
    return out


def beam_tod_batch(
    nside,
    mp,
    data,
    rot_vecs,
    phi_b,
    theta_b,
    psis_b,
    interp_mode="bilinear",
):
    """Accumulate the TOD contribution of one beam entry for a batch of samples.

    Tiles over the ``S`` selected beam pixels so that the
    ``(B × Sc × 3 × float32)`` intermediate vector buffer stays within the L2
    cache target. Uses Numba JIT kernels for both the rotation and the
    gather + accumulation steps.

    Three execution paths are selected automatically based on the contents of
    ``data``:

    * **Flat-sky** *(fastest)* — requires ``vec_rolled``, ``psi_grid``,
      ``dtheta``, ``dphi``, and ``mp_stacked`` in ``data``. Skips both
      Rodrigues rotations and the ``vec2ang`` call; computes sky positions from
      precomputed angular offsets and feeds directly into the HEALPix
      interpolation kernel. Valid for narrow beams (≲ 5°).

    * **Single-Rodrigues** — requires ``vec_rolled``, ``psi_grid``, and
      ``mp_stacked`` in ``data``. The psi-roll is baked into the precomputed
      vectors; only the recentering rotation (Rodrigues 1) is applied at
      runtime. Roughly half the rotation cost of the full path.

    * **Full double-Rodrigues** *(fallback)* — used when no cache arrays are
      present. Applies both Rodrigues rotations per ``(B, S)`` element at
      runtime.

    Args:
        nside (int): HEALPix ``nside`` of the sky map.
        mp (list[numpy.ndarray]): Sky map components ``[I, Q, U]``. Each
            element is a 1-D ``float32`` array of length ``12 * nside**2``.
            Used only on the full double-Rodrigues fallback path.
        data (dict): Beam data entry as returned by :func:`prepare_beam_data`.
            Required keys: ``'vec_orig'``, ``'beam_vals'``, ``'comp_indices'``.
            Optional keys for cached paths: ``'mp_stacked'``, ``'vec_rolled'``,
            ``'psi_grid'``, ``'dtheta'``, ``'dphi'``.
        rot_vecs (numpy.ndarray): Rodrigues rotation vectors from
            :func:`precompute_rotation_vector_batch`, shape ``(B, 3)``.
        phi_b (numpy.ndarray): Boresight longitude [rad], shape ``(B,)``.
        theta_b (numpy.ndarray): Boresight colatitude [rad], shape ``(B,)``.
        psis_b (numpy.ndarray): Combined rotation angle ``psi_b - beta`` [rad],
            shape ``(B,)``.
        interp_mode (str): Sky-map interpolation strategy. One of:

            * ``'bilinear'`` *(default)* — 4-pixel bilinear HEALPix
              interpolation via the fused Numba kernel.
            * ``'nearest'`` — single nearest-pixel lookup; fastest, no pixel
              mixing.
            (``'gaussian'`` and ``'bicubic'`` are available on their respective branches.)

    Returns:
        dict[int, numpy.ndarray]: Mapping from Stokes component index to a
            ``(B,)`` ``float32`` array containing the beam-weighted sky-map
            accumulation for that component over the batch.
    """
    B = phi_b.shape[0]
    vec_orig = data["vec_orig"]  # (S, 3)
    beam_vals = data["beam_vals"]  # (S,)
    S = vec_orig.shape[0]
    comp_indices = data["comp_indices"]
    C = len(comp_indices)
    mp_stacked = data.get("mp_stacked")  # (C, N) float32, or None
    vec_rolled = data.get("vec_rolled")  # (N_psi, S, 3) float32, or None
    psi_grid = data.get("psi_grid")  # (N_psi,) float32, or None
    dtheta = data.get("dtheta")  # (N_psi, S) float32, or None
    dphi = data.get("dphi")  # (N_psi, S) float32, or None

    use_cache = vec_rolled is not None and psi_grid is not None
    # Flat-sky path: skips both Rodrigues rotations and vec2ang entirely.
    # Requires mp_stacked since it feeds directly into _gather_accum_flatsky_jit.
    use_flatsky = (
        use_cache and dtheta is not None and dphi is not None and mp_stacked is not None
    )
    # Near-pole fallback (issue: sin θ ≈ 0 makes dphi/sin θ ill-defined on the
    # flat-sky path).  When any sample in the batch is within 1e-10 of a pole,
    # demote all the way to the full double-Rodrigues path, which handles every
    # latitude correctly without any precomputed-cache approximation.
    if use_flatsky and np.any(
        np.abs(np.sin(np.asarray(theta_b, dtype=np.float64))) < 1e-10
    ):
        use_flatsky = False
        use_cache = False
    use_nearest = interp_mode == "nearest"
    if interp_mode not in ("nearest", "bilinear"):
        raise ValueError(
            f"interp_mode {interp_mode!r} not available on main branch; "
            "switch to the 'gaussian' or 'bicubic' branch"
        )

    # Lower bound from L2 target; upper bound from _MAX_TILES cap.
    # The max() ensures we never produce more than _MAX_TILES tiles even when
    # the memory-based Sc is tiny (e.g. Sc=79 at B=2212 → 64 tiles → 64 C calls).
    Sc = max(1, _S_TILE_TARGET_BYTES // (B * 3 * 4))  # memory target
    Sc = max(Sc, -(-S // _MAX_TILES))  # tile-count cap (ceiling div)
    Sc = min(Sc, S)

    if use_cache:
        # Map each sample's psi angle to the nearest precomputed bin index.
        # Used by both the flat-sky and single-Rodrigues cached paths.
        n_psi = len(psi_grid)
        dpsi = _TWO_PI / n_psi
        k_b = np.mod(
            np.round(np.mod(psis_b, _TWO_PI) / dpsi).astype(np.int64),
            n_psi,
        )  # (B,)

    # Rotation scalars are not needed on the flat-sky path — both rotations
    # are bypassed. Compute them only for the other two paths.
    if not use_flatsky:
        axes, cos_a, sin_a, ax_pts, cos_p, sin_p = _rotation_params(
            rot_vecs, phi_b, theta_b, psis_b
        )

    tod = {comp: np.zeros(B, dtype=np.float32) for comp in comp_indices}

    for s0 in range(0, S, Sc):
        s1 = min(s0 + Sc, S)
        bv_chunk = beam_vals[s0:s1]  # (Sc,)

        if use_flatsky:
            tod_arr = np.zeros((C, B), dtype=np.float64)
            if use_nearest:
                # Flat-sky nearest-pixel path.
                _gather_accum_nearest_flatsky_jit(
                    np.ascontiguousarray(dtheta[:, s0:s1]),
                    np.ascontiguousarray(dphi[:, s0:s1]),
                    k_b,
                    np.asarray(theta_b, dtype=np.float32),
                    np.asarray(phi_b, dtype=np.float32),
                    nside,
                    mp_stacked,
                    bv_chunk,
                    B,
                    s1 - s0,
                    tod_arr,
                )
            else:
                # Flat-sky bilinear path (default).
                _gather_accum_flatsky_jit(
                    np.ascontiguousarray(dtheta[:, s0:s1]),
                    np.ascontiguousarray(dphi[:, s0:s1]),
                    k_b,
                    np.asarray(theta_b, dtype=np.float32),
                    np.asarray(phi_b, dtype=np.float32),
                    nside,
                    mp_stacked,
                    bv_chunk,
                    B,
                    s1 - s0,
                    tod_arr,
                )
            for i, comp in enumerate(comp_indices):
                tod[comp] += tod_arr[i].astype(np.float32)

        elif use_cache:
            # Single-Rodrigues path: psi-roll baked in, only recentering needed.
            vec_chunk = vec_rolled[k_b[:, None], np.arange(s0, s1)[None, :], :]
            vec_chunk = np.ascontiguousarray(vec_chunk.astype(np.float32))  # (B, Sc, 3)
            vec_rot = np.empty((B, s1 - s0, 3), dtype=np.float32)
            _rodrigues1_from_rolled_jit(vec_chunk, axes, cos_a, sin_a, vec_rot)

            if mp_stacked is not None:
                tod_arr = np.zeros((C, B), dtype=np.float64)
                if use_nearest:
                    _gather_accum_nearest_jit(
                        vec_rot, nside, mp_stacked, bv_chunk, B, s1 - s0, tod_arr
                    )
                else:
                    _gather_accum_fused_jit(
                        vec_rot, nside, mp_stacked, bv_chunk, B, s1 - s0, tod_arr
                    )
                for i, comp in enumerate(comp_indices):
                    tod[comp] += tod_arr[i].astype(np.float32)
            else:
                theta_flat, phi_flat = hp.vec2ang(
                    vec_rot.reshape(-1, 3).astype(np.float64)
                )
                pixels, weights = get_interp_weights_numba(nside, theta_flat, phi_flat)
                mp_gathered = np.stack([mp[c][pixels] for c in comp_indices])
                mp_flat = np.einsum("ckn,kn->cn", mp_gathered, weights)
                tod_chunk = mp_flat.reshape(C, B, s1 - s0) @ bv_chunk
                for i, comp in enumerate(comp_indices):
                    tod[comp] += tod_chunk[i].astype(np.float32)

        else:
            # Original path: double Rodrigues (recenter + psi roll).
            vec_chunk = np.asarray(vec_orig[s0:s1], dtype=np.float32)  # (Sc, 3)
            vec_rot = np.empty((B, s1 - s0, 3), dtype=np.float32)
            _rodrigues_jit(vec_chunk, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot)

            if mp_stacked is not None:
                tod_arr = np.zeros((C, B), dtype=np.float64)
                if use_nearest:
                    _gather_accum_nearest_jit(
                        vec_rot, nside, mp_stacked, bv_chunk, B, s1 - s0, tod_arr
                    )
                else:
                    _gather_accum_fused_jit(
                        vec_rot, nside, mp_stacked, bv_chunk, B, s1 - s0, tod_arr
                    )
                for i, comp in enumerate(comp_indices):
                    tod[comp] += tod_arr[i].astype(np.float32)
            else:
                theta_flat, phi_flat = hp.vec2ang(
                    vec_rot.reshape(-1, 3).astype(np.float64)
                )
                pixels, weights = get_interp_weights_numba(nside, theta_flat, phi_flat)
                mp_gathered = np.stack([mp[c][pixels] for c in comp_indices])
                mp_flat = np.einsum("ckn,kn->cn", mp_gathered, weights)
                tod_chunk = mp_flat.reshape(C, B, s1 - s0) @ bv_chunk
                for i, comp in enumerate(comp_indices):
                    tod[comp] += tod_chunk[i].astype(np.float32)

    return tod
