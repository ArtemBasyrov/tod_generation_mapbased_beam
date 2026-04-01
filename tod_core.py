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
    _get_interp_weights_jit,
    get_interp_weights_numba,
    _query_disc_jit,
    _query_disc_into_jit,
    _pix2ang_ring_jit,
    _pix2zphi_ring_jit,
    _ang2pix_ring_jit,
)

# Target working-set size for the (B × Sc × 3 × float32) vec_rot intermediate.
# Sized to stay within a typical L2 cache (2 MB).
_S_TILE_TARGET_BYTES = 2 * 1024 * 1024

# Maximum number of S-tiles per beam entry.  Each tile makes one call into the
# HEALPix interpolation logic.  Capping at _MAX_TILES ensures Sc is always at
# least S/_MAX_TILES, keeping per-tile overhead bounded while still preventing
# out-of-memory.
_MAX_TILES = 8

_INV_TWO_PI  = 1.0 / _TWO_PI


# ── Keys cubic kernel (Catmull-Rom, a = −0.5) ────────────────────────────────

@numba.jit(nopython=True, cache=True)
def _keys_1d_jit(t):
    """Keys cubic convolution kernel, Horner form.

    Support (−2, 2):
        |t| < 1 :  (3/2)|t|³ − (5/2)|t|² + 1
        |t| < 2 :  (−1/2)|t|³ + (5/2)|t|² − 4|t| + 2
        else    :  0
    """
    t = abs(t)
    if t < 1.0:
        return (1.5 * t - 2.5) * t * t + 1.0
    if t < 2.0:
        return ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0
    return 0.0


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
        kx  = axes[b, 0];  ky  = axes[b, 1];  kz  = axes[b, 2]
        ca  = cos_a[b];    sa  = sin_a[b];    oma = 1.0 - ca
        px  = ax_pts[b, 0]; py = ax_pts[b, 1]; pz = ax_pts[b, 2]
        cp_ = cos_p[b];    sp_ = sin_p[b];    omp = 1.0 - cp_
        for s in range(S):
            vx = vec_orig[s, 0]; vy = vec_orig[s, 1]; vz = vec_orig[s, 2]
            # Rodrigues 1 – recenter beam
            dkv = kx*vx + ky*vy + kz*vz
            rx = vx*ca + (ky*vz - kz*vy)*sa + kx*dkv*oma
            ry = vy*ca + (kz*vx - kx*vz)*sa + ky*dkv*oma
            rz = vz*ca + (kx*vy - ky*vx)*sa + kz*dkv*oma
            # Rodrigues 2 – polarisation roll
            dpr = px*rx + py*ry + pz*rz
            out[b, s, 0] = rx*cp_ + (py*rz - pz*ry)*sp_ + px*dpr*omp
            out[b, s, 1] = ry*cp_ + (pz*rx - px*rz)*sp_ + py*dpr*omp
            out[b, s, 2] = rz*cp_ + (px*ry - py*rx)*sp_ + pz*dpr*omp


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
    B  = axes.shape[0]
    Sc = vec_rolled_b.shape[1]
    for b in range(B):
        kx  = axes[b, 0]; ky = axes[b, 1]; kz = axes[b, 2]
        ca  = cos_a[b];   sa = sin_a[b];   oma = 1.0 - ca
        for s in range(Sc):
            vx = vec_rolled_b[b, s, 0]
            vy = vec_rolled_b[b, s, 1]
            vz = vec_rolled_b[b, s, 2]
            dkv = kx*vx + ky*vy + kz*vz
            out[b, s, 0] = vx*ca + (ky*vz - kz*vy)*sa + kx*dkv*oma
            out[b, s, 1] = vy*ca + (kz*vx - kx*vz)*sa + ky*dkv*oma
            out[b, s, 2] = vz*ca + (kx*vy - ky*vx)*sa + kz*dkv*oma


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
            n  = b * Sc + s
            bv = float(beam_vals[s])
            p0 = pixels[0, n]; p1 = pixels[1, n]
            p2 = pixels[2, n]; p3 = pixels[3, n]
            w0 = weights[0, n]; w1 = weights[1, n]
            w2 = weights[2, n]; w3 = weights[3, n]
            for c in range(C):
                tod[c, b] += (mp_stacked[c, p0]*w0 + mp_stacked[c, p1]*w1 +
                              mp_stacked[c, p2]*w2 + mp_stacked[c, p3]*w3) * bv


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
    C          = mp_stacked.shape[0]
    npix_total = 12 * nside * nside

    for b in numba.prange(B):               # ← parallel; no race on tod[:, b]
        for s in range(Sc):

            # ── step 1: vec2ang (inline) ──────────────────────────────────────
            # atan2(r_xy, z) is better conditioned near the poles than acos(z).
            vx = float(vec_rot[b, s, 0])
            vy = float(vec_rot[b, s, 1])
            vz = float(vec_rot[b, s, 2])
            theta = math.atan2(math.sqrt(vx*vx + vy*vy), vz)
            phi   = math.atan2(vy, vx)
            if phi < 0.0:
                phi += _TWO_PI

            # ── step 2: HEALPix RING bilinear interpolation (inline) ──────────
            z        = math.cos(theta)
            az       = abs(z)
            if az > _TWO_THIRDS:                 # polar cap
                tp       = nside * math.sqrt(3.0 * (1.0 - az))
                ir_above = int(tp)
                if z < 0.0:
                    ir_above = 4 * nside - ir_above - 1
            else:                                # equatorial belt
                ir_above = int(nside * (2.0 - 1.5 * z))
            ir_below = ir_above + 1

            if ir_above == 0:
                # ── North-pole boundary ───────────────────────────────────────
                # Ring 1 layout
                na1  = 4
                fpa1 = 0
                sa1  = 1
                dphia1 = _TWO_PI / na1
                phi0a1 = sa1 * dphia1 * 0.5
                tw   = ((phi - phi0a1) / dphia1) % float(na1)
                ip   = int(tw)
                frac = tw - ip
                ip2  = (ip + 1) % na1
                p2   = fpa1 + ip
                p3   = fpa1 + ip2
                p0   = (ip  + 2) % na1
                p1   = (ip2 + 2) % na1
                # theta weight (theta1 = 0 at north pole)
                tmp_r1 = 1.0
                za1    = 1.0 - tmp_r1 * tmp_r1 / (3.0 * nside * nside)
                ta1    = math.acos(za1)
                wt     = theta / ta1
                nf     = (1.0 - wt) * 0.25
                w0     = nf
                w1     = nf
                w2     = (1.0 - frac) * wt + nf
                w3     = frac          * wt + nf

            elif ir_below == 4 * nside:
                # ── South-pole boundary ───────────────────────────────────────
                ir_last = 4 * nside - 1
                i2last  = 4 * nside - ir_last          # = 1
                na_l    = 4 * i2last                   # = 4
                fpa_l   = npix_total - 2 * i2last * (i2last + 1)
                dphia_l = _TWO_PI / na_l
                phi0a_l = dphia_l * 0.5                # shift=1 always for last ring
                tw      = ((phi - phi0a_l) / dphia_l) % float(na_l)
                ip      = int(tw)
                frac    = tw - ip
                ip2     = (ip + 1) % na_l
                p0      = fpa_l + ip
                p1      = fpa_l + ip2
                p2      = (ip  + 2) % na_l + fpa_l
                p3      = (ip2 + 2) % na_l + fpa_l
                # theta weight toward south pole
                tmp_rl  = float(i2last)
                za_l    = -(1.0 - tmp_rl * tmp_rl / (3.0 * nside * nside))
                ta_l    = math.acos(za_l)
                wts     = (theta - ta_l) / (math.pi - ta_l)
                sf      = wts * 0.25
                w0      = (1.0 - frac) * (1.0 - wts) + sf
                w1      = frac          * (1.0 - wts) + sf
                w2      = sf
                w3      = sf

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
                ta      = math.acos(za)
                tb      = math.acos(zb)
                w_below = (theta - ta) / (tb - ta)
                w_above = 1.0 - w_below

                # Ring above: pixel layout + phi interpolation
                if ir_above < nside:
                    na  = 4 * ir_above
                    fpa = 2 * ir_above * (ir_above - 1)
                    sa  = 1
                elif ir_above <= 3 * nside:
                    na  = 4 * nside
                    fpa = 2 * nside * (nside - 1) + (ir_above - nside) * 4 * nside
                    sa  = 1 if (ir_above - nside) % 2 == 0 else 0
                else:
                    i2a = 4 * nside - ir_above
                    na  = 4 * i2a
                    fpa = npix_total - 2 * i2a * (i2a + 1)
                    sa  = 1
                dphia = _TWO_PI / na
                phi0a = sa * dphia * 0.5
                tw    = ((phi - phi0a) / dphia) % float(na)
                iphia = int(tw)
                fphia = tw - iphia
                p0    = fpa + iphia
                p1    = fpa + (iphia + 1) % na
                w0    = w_above * (1.0 - fphia)
                w1    = w_above * fphia

                # Ring below: pixel layout + phi interpolation
                if ir_below < nside:
                    nb  = 4 * ir_below
                    fpb = 2 * ir_below * (ir_below - 1)
                    sb  = 1
                elif ir_below <= 3 * nside:
                    nb  = 4 * nside
                    fpb = 2 * nside * (nside - 1) + (ir_below - nside) * 4 * nside
                    sb  = 1 if (ir_below - nside) % 2 == 0 else 0
                else:
                    i2b = 4 * nside - ir_below
                    nb  = 4 * i2b
                    fpb = npix_total - 2 * i2b * (i2b + 1)
                    sb  = 1
                dphib = _TWO_PI / nb
                phi0b = sb * dphib * 0.5
                tw    = ((phi - phi0b) / dphib) % float(nb)
                iphib = int(tw)
                fphib = tw - iphib
                p2    = fpb + iphib
                p3    = fpb + (iphib + 1) % nb
                w2    = w_below * (1.0 - fphib)
                w3    = w_below * fphib

            # ── step 3: gather + beam-weighted accumulation ───────────────────
            bv = float(beam_vals[s])
            for c in range(C):
                tod[c, b] += (  mp_stacked[c, p0] * w0
                              + mp_stacked[c, p1] * w1
                              + mp_stacked[c, p2] * w2
                              + mp_stacked[c, p3] * w3) * bv


# ── Flat-sky fused kernel ─────────────────────────────────────────────────────

@numba.jit(nopython=True, parallel=True, cache=True)
def _gather_accum_flatsky_jit(dtheta_tile, dphi_tile, k_b, theta_b, phi_b,
                               nside, mp_stacked, beam_vals, B, Sc, tod):
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
    C          = mp_stacked.shape[0]
    npix_total = 12 * nside * nside

    for b in numba.prange(B):
        th_point = float(theta_b[b])
        ph_point = float(phi_b[b])
        kb       = k_b[b]
        sin_th   = math.sin(th_point)
        inv_sin  = 1.0 / sin_th if sin_th > 1e-10 else 0.0

        for s in range(Sc):
            theta = th_point + float(dtheta_tile[kb, s])
            phi   = ph_point + float(dphi_tile[kb, s]) * inv_sin

            # Clamp theta to [0, π] — flat-sky approximation can push edge
            # beam pixels slightly outside the valid range near the poles.
            if theta < 0.0:
                theta = -theta
                phi  += math.pi
            elif theta > math.pi:
                theta = _TWO_PI - theta
                phi  += math.pi

            # Wrap phi to [0, 2π)
            if phi < 0.0:
                phi += _TWO_PI
            elif phi >= _TWO_PI:
                phi -= _TWO_PI

            # ── HEALPix RING bilinear interpolation (inline) ──────────────────
            z        = math.cos(theta)
            az       = abs(z)
            if az > _TWO_THIRDS:
                tp       = nside * math.sqrt(3.0 * (1.0 - az))
                ir_above = int(tp)
                if z < 0.0:
                    ir_above = 4 * nside - ir_above - 1
            else:
                ir_above = int(nside * (2.0 - 1.5 * z))
            ir_below = ir_above + 1

            if ir_above == 0:
                na1    = 4
                fpa1   = 0
                sa1    = 1
                dphia1 = _TWO_PI / na1
                phi0a1 = sa1 * dphia1 * 0.5
                tw     = ((phi - phi0a1) / dphia1) % float(na1)
                ip     = int(tw)
                frac   = tw - ip
                ip2    = (ip + 1) % na1
                p2     = fpa1 + ip
                p3     = fpa1 + ip2
                p0     = (ip  + 2) % na1
                p1     = (ip2 + 2) % na1
                tmp_r1 = 1.0
                za1    = 1.0 - tmp_r1 * tmp_r1 / (3.0 * nside * nside)
                ta1    = math.acos(za1)
                wt     = theta / ta1
                nf     = (1.0 - wt) * 0.25
                w0     = nf
                w1     = nf
                w2     = (1.0 - frac) * wt + nf
                w3     = frac          * wt + nf

            elif ir_below == 4 * nside:
                ir_last = 4 * nside - 1
                i2last  = 4 * nside - ir_last
                na_l    = 4 * i2last
                fpa_l   = npix_total - 2 * i2last * (i2last + 1)
                dphia_l = _TWO_PI / na_l
                phi0a_l = dphia_l * 0.5
                tw      = ((phi - phi0a_l) / dphia_l) % float(na_l)
                ip      = int(tw)
                frac    = tw - ip
                ip2     = (ip + 1) % na_l
                p0      = fpa_l + ip
                p1      = fpa_l + ip2
                p2      = (ip  + 2) % na_l + fpa_l
                p3      = (ip2 + 2) % na_l + fpa_l
                tmp_rl  = float(i2last)
                za_l    = -(1.0 - tmp_rl * tmp_rl / (3.0 * nside * nside))
                ta_l    = math.acos(za_l)
                wts     = (theta - ta_l) / (math.pi - ta_l)
                sf      = wts * 0.25
                w0      = (1.0 - frac) * (1.0 - wts) + sf
                w1      = frac          * (1.0 - wts) + sf
                w2      = sf
                w3      = sf

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
                ta      = math.acos(za)
                tb      = math.acos(zb)
                w_below = (theta - ta) / (tb - ta)
                w_above = 1.0 - w_below

                if ir_above < nside:
                    na  = 4 * ir_above
                    fpa = 2 * ir_above * (ir_above - 1)
                    sa  = 1
                elif ir_above <= 3 * nside:
                    na  = 4 * nside
                    fpa = 2 * nside * (nside - 1) + (ir_above - nside) * 4 * nside
                    sa  = 1 if (ir_above - nside) % 2 == 0 else 0
                else:
                    i2a = 4 * nside - ir_above
                    na  = 4 * i2a
                    fpa = npix_total - 2 * i2a * (i2a + 1)
                    sa  = 1
                dphia = _TWO_PI / na
                phi0a = sa * dphia * 0.5
                tw    = ((phi - phi0a) / dphia) % float(na)
                iphia = int(tw)
                fphia = tw - iphia
                p0    = fpa + iphia
                p1    = fpa + (iphia + 1) % na
                w0    = w_above * (1.0 - fphia)
                w1    = w_above * fphia

                if ir_below < nside:
                    nb  = 4 * ir_below
                    fpb = 2 * ir_below * (ir_below - 1)
                    sb  = 1
                elif ir_below <= 3 * nside:
                    nb  = 4 * nside
                    fpb = 2 * nside * (nside - 1) + (ir_below - nside) * 4 * nside
                    sb  = 1 if (ir_below - nside) % 2 == 0 else 0
                else:
                    i2b = 4 * nside - ir_below
                    nb  = 4 * i2b
                    fpb = npix_total - 2 * i2b * (i2b + 1)
                    sb  = 1
                dphib = _TWO_PI / nb
                phi0b = sb * dphib * 0.5
                tw    = ((phi - phi0b) / dphib) % float(nb)
                iphib = int(tw)
                fphib = tw - iphib
                p2    = fpb + iphib
                p3    = fpb + (iphib + 1) % nb
                w2    = w_below * (1.0 - fphib)
                w3    = w_below * fphib

            # ── gather + beam-weighted accumulation ───────────────────────────
            bv = float(beam_vals[s])
            for c in range(C):
                tod[c, b] += (  mp_stacked[c, p0] * w0
                              + mp_stacked[c, p1] * w1
                              + mp_stacked[c, p2] * w2
                              + mp_stacked[c, p3] * w3) * bv


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
    C          = mp_stacked.shape[0]
    npix_total = 12 * nside * nside

    for b in numba.prange(B):
        for s in range(Sc):

            # vec2ang (inline)
            vx = float(vec_rot[b, s, 0])
            vy = float(vec_rot[b, s, 1])
            vz = float(vec_rot[b, s, 2])
            theta = math.atan2(math.sqrt(vx*vx + vy*vy), vz)
            phi   = math.atan2(vy, vx)
            if phi < 0.0:
                phi += _TWO_PI

            # nearest-pixel lookup (inlined _ang2pix_ring_jit)
            z     = vz / math.sqrt(vx*vx + vy*vy + vz*vz)  # cos(theta)
            phi_w = phi  # already in [0, 2π)

            ir_above = _ring_above_jit(nside, z)
            if ir_above < 1:
                ir_above = 1
            elif ir_above > 4 * nside - 2:
                ir_above = 4 * nside - 2
            ir_below = ir_above + 1

            best_pix = 0
            best_cos = -2.0
            sin_th   = math.sin(theta)

            for ir_g in (ir_above, ir_below):
                if ir_g < 1 or ir_g > 4 * nside - 1:
                    continue
                n_pix, first_pix, phi0, dphi_r = _ring_info_jit(nside, ir_g, npix_total)
                z_c     = _ring_z_jit(nside, ir_g)
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
def _gather_accum_nearest_flatsky_jit(dtheta_tile, dphi_tile, k_b, theta_b, phi_b,
                                       nside, mp_stacked, beam_vals, B, Sc, tod):
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
    C          = mp_stacked.shape[0]
    npix_total = 12 * nside * nside

    for b in numba.prange(B):
        th_point = float(theta_b[b])
        ph_point = float(phi_b[b])
        kb       = k_b[b]
        sin_th   = math.sin(th_point)
        inv_sin  = 1.0 / sin_th if sin_th > 1e-10 else 0.0

        for s in range(Sc):
            theta = th_point + float(dtheta_tile[kb, s])
            phi   = ph_point + float(dphi_tile[kb, s]) * inv_sin

            if theta < 0.0:
                theta = -theta
                phi  += math.pi
            elif theta > math.pi:
                theta = _TWO_PI - theta
                phi  += math.pi

            if phi < 0.0:
                phi += _TWO_PI
            elif phi >= _TWO_PI:
                phi -= _TWO_PI

            # nearest-pixel lookup
            z     = math.cos(theta)
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
                z_c     = _ring_z_jit(nside, ir_g)
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


# ── Gaussian kernel interpolation helpers ────────────────────────────────────

def _angular_distance(th1, ph1, th2, ph2):
    """Great-circle angular distance between (th1,ph1) and arrays (th2,ph2) [rad]."""
    cos_d = (np.sin(th1) * np.sin(th2) * np.cos(ph1 - ph2)
             + np.cos(th1) * np.cos(th2))
    return np.arccos(np.clip(cos_d, -1.0, 1.0))


@numba.jit(nopython=True, parallel=True, cache=True)
def _gaussian_interp_accum_jit(theta_flat, phi_flat, B, Sc,
                                nside, mp_stacked, beam_vals, tod_arr,
                                sigma_rad, radius_rad,
                                scratch_pix, scratch_w, tod_tmp):
    """
    Numba JIT isotropic Gaussian kernel interpolation + beam-weighted
    accumulation.  Parallelised over N = B*Sc (prange); each thread uses
    pre-allocated scratch buffers indexed by thread ID.

    Optimisations:
    - prange over N = B*Sc, maximising parallel work.
    - Thread-local scratch_pix / scratch_w avoid any allocation in the hot loop.
    - _pix2ang_ring_jit replaces ang_lut table lookups: pure arithmetic
      (~10 ns) vs DRAM load into a 200–800 MB table (~200 ns at nside≥1024).
    - _query_disc_into_jit writes directly into the scratch buffer.
    - tod_tmp pre-allocated by caller; reduces into tod_arr in a second prange.

    Parameters
    ----------
    theta_flat  : (B*Sc,)            float64  co-latitude [rad]
    phi_flat    : (B*Sc,)            float64  longitude   [rad]
    B, Sc       : int
    nside       : int
    mp_stacked  : (C, N_hp)          float32
    beam_vals   : (Sc,)              float32
    tod_arr     : (C, B)             float64  accumulated in place
    sigma_rad   : float  Gaussian width [rad]
    radius_rad  : float  search radius  [rad]
    scratch_pix : (n_threads, max_M) int64    thread-local pixel index buffer
    scratch_w   : (n_threads, max_M) float64  thread-local weight buffer
    tod_tmp     : (C, B*Sc)          float64  caller-allocated work array
    """
    two_sigma2 = 2.0 * sigma_rad * sigma_rad
    C = mp_stacked.shape[0]
    N = B * Sc

    for idx in numba.prange(N):
        tid = numba.get_thread_id()
        b   = idx // Sc
        s   = idx  % Sc
        th  = theta_flat[idx]
        ph  = phi_flat[idx]
        bv  = float(beam_vals[s])

        pix_buf = scratch_pix[tid]
        w_buf   = scratch_w[tid]

        M = _query_disc_into_jit(nside, th, ph, radius_rad, True, pix_buf)

        if M == 0:
            nearest = _ang2pix_ring_jit(nside, th, ph)
            for c in range(C):
                tod_tmp[c, idx] = mp_stacked[c, nearest] * bv
            continue

        # Flat-sky dist²: Δθ² + sin²(θ)·Δφ².  Valid for radii < ~5°.
        # _pix2ang_ring_jit is pure arithmetic — no ang_lut memory access.
        sin2_th = math.sin(th) ** 2
        w_sum   = 0.0
        for k in range(M):
            theta_n, phi_n = _pix2ang_ring_jit(nside, pix_buf[k])
            dth    = theta_n - th
            dph    = phi_n   - ph
            if dph >  math.pi: dph -= _TWO_PI
            elif dph < -math.pi: dph += _TWO_PI
            w      = math.exp(-(dth * dth + sin2_th * dph * dph) / two_sigma2)
            w_buf[k] = w
            w_sum   += w

        if w_sum < 1e-300:
            best_k = 0
            for k in range(1, M):
                if w_buf[k] > w_buf[best_k]:
                    best_k = k
            for c in range(C):
                tod_tmp[c, idx] = mp_stacked[c, pix_buf[best_k]] * bv
        else:
            inv_w = 1.0 / w_sum
            for c in range(C):
                acc = 0.0
                for k in range(M):
                    acc += w_buf[k] * mp_stacked[c, pix_buf[k]]
                tod_tmp[c, idx] = acc * inv_w * bv

    for b in numba.prange(B):
        for s in range(Sc):
            for c in range(C):
                tod_arr[c, b] += tod_tmp[c, b * Sc + s]


def _gaussian_interp_accum(theta_flat, phi_flat, B, Sc,
                            nside, mp_stacked, beam_vals, tod_arr,
                            sigma_deg, radius_deg):
    """
    Wrapper: converts degrees → radians, allocates thread-local scratch
    buffers and tod_tmp work array, then calls the JIT kernel.
    """
    sigma_rad  = math.radians(sigma_deg)
    radius_rad = math.radians(radius_deg)

    # Solid-angle upper bound on disc pixel count (with 3× safety margin).
    # Formula: Ω_cap = 2π(1−cos r); pixels ≈ Ω_cap × 3·nside²/π.
    # This gives ~132 at all production nsides (actual M ≈ 46), keeping
    # scratch buffers at ~2 KB/thread (L1-resident) vs the old ring-spanning
    # formula that produced 40 000–82 000 (640–1280 KB/thread, L3 thrashing).
    search_rad  = radius_rad + math.sqrt(math.pi / (3.0 * nside * nside))
    max_M       = max(64, int(12.0 * nside * nside * (1.0 - math.cos(search_rad))) + 32)

    n_threads   = numba.get_num_threads()
    scratch_pix = np.empty((n_threads, max_M), dtype=np.int64)
    scratch_w   = np.empty((n_threads, max_M), dtype=np.float64)
    tod_tmp     = np.zeros((mp_stacked.shape[0], B * Sc), dtype=np.float64)

    _gaussian_interp_accum_jit(theta_flat, phi_flat, B, Sc,
                                nside, mp_stacked, beam_vals, tod_arr,
                                sigma_rad, radius_rad,
                                scratch_pix, scratch_w, tod_tmp)


@numba.jit(nopython=True, parallel=True, cache=True)
def _gaussian_accum_flatsky_jit(dtheta_tile, dphi_tile, k_b, theta_b, phi_b,
                                 B, Sc, nside, mp_stacked, beam_vals, tod_arr,
                                 sigma_rad, radius_rad,
                                 scratch_pix, scratch_w, tod_tmp):
    """
    Fused flat-sky Gaussian interpolation.

    Combines the per-tile theta/phi construction (previously a serial Python
    loop in beam_tod_batch) with the Gaussian weight loop, both inside a single
    prange(N) kernel.  Uses the flat-sky distance approximation to replace
    sin(θ_n) + cos(Δφ) + acos with pure arithmetic, leaving only exp per pixel.
    Pixel angles computed via _pix2ang_ring_jit (arithmetic) rather than
    ang_lut table lookups (200–800 MB DRAM loads at production nside).

    Parameters
    ----------
    dtheta_tile : (N_psi, Sc) float32  co-latitude offsets for this S-tile
    dphi_tile   : (N_psi, Sc) float32  raw phi offsets (to be divided by sin(θ_b))
    k_b         : (B,)        int64    psi-bin index per sample
    theta_b     : (B,)        float64  boresight co-latitude [rad]
    phi_b       : (B,)        float64  boresight longitude   [rad]
    B, Sc       : int
    nside       : int
    mp_stacked  : (C, N_hp)   float32
    beam_vals   : (Sc,)       float32
    tod_arr     : (C, B)      float64  accumulated in place
    sigma_rad   : float  Gaussian width [rad]
    radius_rad  : float  search radius  [rad]
    scratch_pix : (n_threads, max_M) int64
    scratch_w   : (n_threads, max_M) float64
    tod_tmp     : (C, B*Sc)   float64  caller-allocated work array
    """
    two_sigma2 = 2.0 * sigma_rad * sigma_rad
    C = mp_stacked.shape[0]
    N = B * Sc

    for idx in numba.prange(N):
        tid = numba.get_thread_id()
        b   = idx // Sc
        s   = idx  % Sc
        kb  = k_b[b]

        th_b    = theta_b[b]
        sin_th_b = math.sin(th_b)
        inv_sin  = 1.0 / sin_th_b if sin_th_b > 1e-10 else 0.0

        th = th_b + float(dtheta_tile[kb, s])
        ph = phi_b[b] + float(dphi_tile[kb, s]) * inv_sin
        if th < 0.0:
            th = -th
            ph += math.pi
        elif th > math.pi:
            th = _TWO_PI - th
            ph += math.pi
        if ph < 0.0:
            ph += _TWO_PI
        elif ph >= _TWO_PI:
            ph -= _TWO_PI

        bv = float(beam_vals[s])

        pix_buf = scratch_pix[tid]
        w_buf   = scratch_w[tid]

        M = _query_disc_into_jit(nside, th, ph, radius_rad, True, pix_buf)

        if M == 0:
            nearest = _ang2pix_ring_jit(nside, th, ph)
            for c in range(C):
                tod_tmp[c, idx] = mp_stacked[c, nearest] * bv
            continue

        # Flat-sky dist²: only exp per pixel, no sin/cos/acos.
        sin2_th = sin_th_b * sin_th_b
        w_sum = 0.0
        for k in range(M):
            theta_n, phi_n = _pix2ang_ring_jit(nside, pix_buf[k])
            dth   = theta_n - th
            dph   = phi_n   - ph
            if dph >  math.pi: dph -= _TWO_PI
            elif dph < -math.pi: dph += _TWO_PI
            dist2 = dth * dth + sin2_th * dph * dph
            w     = math.exp(-dist2 / two_sigma2)
            w_buf[k] = w
            w_sum   += w

        if w_sum < 1e-300:
            best_k = 0
            for k in range(1, M):
                if w_buf[k] > w_buf[best_k]:
                    best_k = k
            for c in range(C):
                tod_tmp[c, idx] = mp_stacked[c, pix_buf[best_k]] * bv
        else:
            inv_w = 1.0 / w_sum
            for c in range(C):
                acc = 0.0
                for k in range(M):
                    acc += w_buf[k] * mp_stacked[c, pix_buf[k]]
                tod_tmp[c, idx] = acc * inv_w * bv

    for b in numba.prange(B):
        for s in range(Sc):
            for c in range(C):
                tod_arr[c, b] += tod_tmp[c, b * Sc + s]


def _gaussian_accum_flatsky(dtheta_tile, dphi_tile, k_b, theta_b, phi_b,
                             B, Sc, nside, mp_stacked, beam_vals, tod_arr,
                             sigma_deg, radius_deg):
    """Wrapper: allocates scratch buffers and calls _gaussian_accum_flatsky_jit."""
    sigma_rad  = math.radians(sigma_deg)
    radius_rad = math.radians(radius_deg)

    search_rad  = radius_rad + math.sqrt(math.pi / (3.0 * nside * nside))
    max_M       = max(64, int(12.0 * nside * nside * (1.0 - math.cos(search_rad))) + 32)

    n_threads   = numba.get_num_threads()
    scratch_pix = np.empty((n_threads, max_M), dtype=np.int64)
    scratch_w   = np.empty((n_threads, max_M), dtype=np.float64)
    tod_tmp     = np.zeros((mp_stacked.shape[0], B * Sc), dtype=np.float64)

    _gaussian_accum_flatsky_jit(dtheta_tile, dphi_tile, k_b, theta_b, phi_b,
                                 B, Sc, nside, mp_stacked, beam_vals, tod_arr,
                                 sigma_rad, radius_rad,
                                 scratch_pix, scratch_w, tod_tmp)


# ── Bicubic (Keys/Catmull-Rom) interpolation kernel ──────────────────────────

@numba.jit(nopython=True, parallel=True, cache=True)
def _gather_accum_bicubic_jit(vec_rot, B, Sc, nside, h_pix, mp_stacked, beam_vals,
                               tod_arr):
    """
    Fully fused vec2ang + Keys-bicubic interpolation + beam accumulation.

    For each detector sample b (parallelised), iterates sequentially over Sc
    beam pixels s and accumulates directly into tod_arr[c, b].  This avoids
    the (C, B*Sc) intermediate tod_tmp buffer and the second reduction pass,
    keeping tod_arr[c, b] hot in L1/L2 cache throughout the inner Sc loop.

    Three improvements over the previous two-pass (gather-stencil-then-iterate)
    design:

    1. Inlined stencil gather — ring geometry (z_ring, phi_n, pixel index) is
       computed inline inside a single ring-by-ring loop, eliminating ~210
       scratch-buffer memory operations per (b, s) element and the
       _gather_ring_stencil_jit function-call overhead.

    2. Per-ring yi early exit — before iterating over a ring's phi pixels, the
       approximate north-axis gnomonic coordinate
           yi_approx = −(sn·cos_th − cn·sin_th) / h_pix
       is evaluated at dphi = 0.  Rings where |yi_approx| > 2 + 0.2 are skipped
       entirely.  The approximation error is < 0.003 pixel units for nside ≥ 512
       (max dphi ≈ 2 h_pix, |Δcos_dphi| < 2e-6), so the 0.2-unit margin is
       conservative.  Saving is ~15–28% of candidate evaluations, highest in the
       polar cap where ring spacing ≈ 0.80 h_pix puts rings ±3 at |yi| ≈ 2.4.

    3. 4-pixel phi stencil — only 4 phi pixels per ring are visited instead of 5.
       For fractional phi offset f ∈ (−0.5, 0.5] from the nearest pixel center:
         f >= 0  →  dip ∈ {-1,  0, +1, +2}  (right-biased window)
         f <  0  →  dip ∈ {-2, -1,  0, +1}  (left-biased window)
       In the right-biased case dip = -2 gives |xi| = 2 + f ≥ 2 (always zero);
       in the left-biased case dip = +2 gives |xi| = 2 + |f| > 2 (always zero).
       Choosing the window by sign(f) avoids computing dphi/xi/kxi for that dead
       candidate, saving ~14% of per-ring iterations (7 per (b,s) pair).

    Parameters
    ----------
    vec_rot    : (B, Sc, 3)   float32   rotated beam unit vectors
    B, Sc      : int
    nside      : int
    h_pix      : float64      angular pixel scale [rad] = nside2resol(nside)
    mp_stacked : (C, N_hp)    float32   stacked sky-map components
    beam_vals  : (Sc,)        float32   beam weights for this tile
    tod_arr    : (C, B)       float64   accumulated in place
    """
    C          = mp_stacked.shape[0]
    inv_h      = 1.0 / h_pix
    npix_total = 12 * nside * nside
    _YI_MARGIN  = 0.2    # pixel units; see docstring for derivation
    _TAYLOR_THR = 0.05  # rad; cos Taylor error < 1.6e-8 → value error < 1e-5 (nside≥512)

    for b in numba.prange(B):
        for s in range(Sc):
            # ── vec2ang (inline) ──────────────────────────────────────────────
            vx   = float(vec_rot[b, s, 0])
            vy   = float(vec_rot[b, s, 1])
            vz   = float(vec_rot[b, s, 2])
            r_xy = math.sqrt(vx * vx + vy * vy)
            th   = math.atan2(r_xy, vz)
            ph   = math.atan2(vy, vx)
            if ph < 0.0:
                ph += _TWO_PI

            bv     = float(beam_vals[s])
            sin_th = r_xy
            cos_th = vz

            # ── ring bounds ────────────────────────────────────────────────────
            ir_center = _ring_above_jit(nside, vz)
            if ir_center < 1:
                ir_center = 1
            elif ir_center > 4 * nside - 1:
                ir_center = 4 * nside - 1
            ir_lo = max(1,             ir_center - 2)
            ir_hi = min(4 * nside - 1, ir_center + 2)

            # ── fused ring walk + Keys weight + map accumulation ──────────────
            # · sin(θ_ring) = sqrt(1−z²): 1 sqrt per ring replaces acos→sin→cos.
            # · Per-ring products (st_sn, ct_cn, sn_ct, cn_st) amortise 4 muls
            #   over the 4-pixel phi stencil.
            # · dphi is always small (≤ 2 phi-pixel steps); Taylor branch
            #   (|dphi| < 0.025 rad) replaces sin/cos with 3 muls each.
            # · Early exit on kxi = 0 skips north projection + second kyi call.
            # · Early exit on w = 0 skips C map reads (cache misses, 600 MB map).
            # · Weight and map value accumulated together — pixel index read once.
            w_sum = 0.0  # float64: Keys kernel returns float64, so accumulator must match
            acc   = np.zeros(C, dtype=np.float64)   # Numba stack-allocates this (C = 3 for T/Q/U)

            for ir in range(ir_lo, ir_hi + 1):
                n_p, fp, phi0, dphi_ring = _ring_info_jit(nside, ir, npix_total)
                cn    = _ring_z_jit(nside, ir)
                sn    = math.sqrt(max(0.0, 1.0 - cn * cn))
                sn_ct = sn * cos_th    # for yi numerator and yi_approx
                cn_st = cn * sin_th    # for yi numerator and yi_approx

                # ── per-ring yi early exit ─────────────────────────────────────
                # yi_approx = -(sn·cos_th − cn·sin_th) / h_pix  at dphi = 0.
                # Error vs actual yi < 0.003 pixels at nside ≥ 512; margin = 0.2.
                if abs(-(sn_ct - cn_st) * inv_h) > 2.0 + _YI_MARGIN:
                    continue

                st_sn = sin_th * sn    # for cos_c
                ct_cn = cos_th * cn    # for cos_c

                # Normal ring: 4-pixel phi stencil.
                # t_phi  = fractional position in phi-pixel units from phi0.
                # ip_f   = nearest pixel as float (before modulo).
                # f_frac = sub-pixel offset ∈ (−0.5, 0.5].
                # dip_lo = -1 when f_frac >= 0 (right-biased; dip=-2 dead)
                #        = -2 when f_frac <  0 (left-biased;  dip=+2 dead)
                t_phi     = (ph - phi0) / dphi_ring
                ip_f      = math.floor(t_phi + 0.5)
                ip_center = int(ip_f) % n_p
                f_frac    = t_phi - ip_f
                dip_lo    = -1 if f_frac >= 0.0 else -2
                for dip in range(dip_lo, dip_lo + 4):
                    ip_in  = (ip_center + dip) % n_p
                    phi_n  = phi0 + ip_in * dphi_ring
                    p      = fp + ip_in
                    dphi_c = phi_n - ph
                    if   dphi_c >  math.pi: dphi_c -= _TWO_PI
                    elif dphi_c < -math.pi: dphi_c += _TWO_PI
                    if abs(dphi_c) < _TAYLOR_THR:
                        dp2      = dphi_c * dphi_c
                        sin_dphi = dphi_c * (1.0 - dp2 * (1.0 / 6.0))
                        cos_dphi = 1.0    - dp2 * 0.5
                    else:
                        sin_dphi = math.sin(dphi_c)
                        cos_dphi = math.cos(dphi_c)
                    cos_c = st_sn * cos_dphi + ct_cn
                    if cos_c < 1e-10:
                        continue
                    inv_c = inv_h / cos_c
                    xi    = sn * sin_dphi * inv_c
                    kxi   = _keys_1d_jit(xi)
                    if kxi == 0.0:
                        continue
                    yi = -(sn_ct * cos_dphi - cn_st) * inv_c
                    w  = kxi * _keys_1d_jit(yi)
                    if w == 0.0:
                        continue
                    w_sum += w
                    acc[0] += w * mp_stacked[0, p]
                    acc[1] += w * mp_stacked[1, p]
                    acc[2] += w * mp_stacked[2, p]

            # ── write to tod_arr (no intermediate buffer) ─────────────────────
            if abs(w_sum) < 1e-10:
                # Degenerate weight sum (pole or very sparse disc) — nearest fallback
                nearest = _ang2pix_ring_jit(nside, th, ph)
                tod_arr[0, b] += mp_stacked[0, nearest] * bv
                tod_arr[1, b] += mp_stacked[1, nearest] * bv
                tod_arr[2, b] += mp_stacked[2, nearest] * bv
            else:
                inv_w = bv / w_sum
                tod_arr[0, b] += acc[0] * inv_w
                tod_arr[1, b] += acc[1] * inv_w
                tod_arr[2, b] += acc[2] * inv_w


def _bicubic_interp_accum(vec_rot, B, Sc, nside, mp_stacked, beam_vals, tod_arr):
    """
    Wrapper: resolves pixel scale and calls the JIT kernel.

    No thread-local scratch buffers needed: the kernel inlines the ring-walk
    stencil gather and accumulates directly into tod_arr[c, b].

    Parameters
    ----------
    vec_rot    : (B, Sc, 3)  float32
    B, Sc      : int
    nside      : int
    mp_stacked : (C, N_hp)   float32
    beam_vals  : (Sc,)       float32
    tod_arr    : (C, B)      float64   accumulated in place
    """
    h_pix = hp.nside2resol(nside)
    _gather_accum_bicubic_jit(vec_rot, B, Sc, nside, h_pix, mp_stacked, beam_vals,
                               tod_arr)


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
        return np.stack([np.sin(theta)*np.cos(phi),
                         np.sin(theta)*np.sin(phi),
                         np.cos(theta)], axis=-1)

    phi_orig   = ra
    theta_orig = np.pi/2 - dec

    vec_center = sph2vec(phi_orig[center_idx], theta_orig[center_idx])[np.newaxis, :]
    vec_target = sph2vec(phi_batch, theta_batch)

    N_center = np.array([ np.cos(phi_orig[center_idx]) * np.cos(theta_orig[center_idx]),
                          np.sin(phi_orig[center_idx]) * np.cos(theta_orig[center_idx]),
                         -np.sin(theta_orig[center_idx])])
    N_target = np.array([ np.cos(phi_batch) * np.cos(theta_batch),
                          np.sin(phi_batch) * np.cos(theta_batch),
                         -np.sin(theta_batch)]).T
    E_target = np.array([-np.sin(phi_batch),
                          np.cos(phi_batch),
                          np.zeros_like(phi_batch)]).T

    axis      = np.cross(vec_center, vec_target)
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis      = np.where(axis_norm > 1e-10, axis / axis_norm, 0)

    angle      = np.arccos(np.clip(np.sum(vec_center * vec_target, axis=-1), -1, 1))
    rot_vector = axis * angle[..., np.newaxis]

    ca     = np.cos(angle)
    v      = N_center[np.newaxis, :]
    dot_kv = np.sum(axis * v, axis=-1, keepdims=True)
    w      = (v * ca[..., np.newaxis]
              + np.cross(axis, v) * np.sin(angle)[..., np.newaxis]
              + axis * dot_kv * (1 - ca)[..., np.newaxis])

    beta = np.arctan2(np.sum(w * E_target, axis=-1), np.sum(w * N_target, axis=-1))
    beta = np.where(beta < 0, beta + 2*np.pi, beta)

    return rot_vector, beta


def _rotation_params(rot_vecs, phi_b, theta_b, psis_b):
    """
    Pre-compute the per-sample scalars needed by _rodrigues_jit from the
    Rodrigues vectors and pointing angles.  All outputs are float32.

    Returns axes (B,3), cos_a (B,), sin_a (B,), ax_pts (B,3), cos_p (B,), sin_p (B,)
    """
    angles = np.linalg.norm(rot_vecs, axis=-1).astype(np.float32)      # (B,)
    safe   = angles > np.float32(1e-10)
    axes   = (rot_vecs / np.where(safe[:, None],
                                   angles[:, None],
                                   np.float32(1.))).astype(np.float32)
    axes   = np.where(safe[:, None], axes, np.float32(0.))
    cos_a  = np.cos(angles)
    sin_a  = np.sin(angles)

    phi_f   = np.asarray(phi_b,   dtype=np.float32)
    theta_f = np.asarray(theta_b, dtype=np.float32)
    psis_f  = np.asarray(psis_b,  dtype=np.float32)
    st = np.sin(theta_f); ct = np.cos(theta_f)
    sp = np.sin(phi_f);   cp = np.cos(phi_f)
    ax_pts = np.stack([st*cp, st*sp, ct], axis=-1)
    cos_p  = np.cos(psis_f)
    sin_p  = np.sin(psis_f)

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
        rot_vecs, phi_pix, theta_pix, psis)
    out = np.empty((B, S, 3), dtype=np.float32)
    _rodrigues_jit(np.asarray(vec_orig, dtype=np.float32),
                   axes, cos_a, sin_a, ax_pts, cos_p, sin_p, out)
    return out


def beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                   interp_mode='bilinear', sigma_deg=None, radius_deg=None):
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
            * ``'bicubic'`` — Keys/Catmull-Rom bicubic via gnomonic projection
              over ~30–50 disc neighbours; O(h⁴) accuracy, same rotational
              stability as bilinear, ~8–12× more pixel lookups.
            * ``'gaussian'`` — isotropic Gaussian kernel over all pixels within
              ``radius_deg``; slowest, avoids grid-aligned artefacts.

        sigma_deg (float | None): Gaussian kernel width [degrees].
            Defaults to one HEALPix pixel resolution.
            Ignored when ``interp_mode != 'gaussian'``.
        radius_deg (float | None): Neighbour search radius [degrees].
            Defaults to ``3 × sigma_deg``.
            Ignored when ``interp_mode != 'gaussian'``.

    Returns:
        dict[int, numpy.ndarray]: Mapping from Stokes component index to a
            ``(B,)`` ``float32`` array containing the beam-weighted sky-map
            accumulation for that component over the batch.
    """
    B            = phi_b.shape[0]
    vec_orig     = data['vec_orig']    # (S, 3)
    beam_vals    = data['beam_vals']   # (S,)
    S            = vec_orig.shape[0]
    comp_indices = data['comp_indices']
    C            = len(comp_indices)
    mp_stacked   = data.get('mp_stacked')   # (C, N) float32, or None
    vec_rolled   = data.get('vec_rolled')   # (N_psi, S, 3) float32, or None
    psi_grid     = data.get('psi_grid')     # (N_psi,) float32, or None
    dtheta       = data.get('dtheta')       # (N_psi, S) float32, or None
    dphi         = data.get('dphi')         # (N_psi, S) float32, or None

    use_cache    = vec_rolled is not None and psi_grid is not None
    # Flat-sky path: skips both Rodrigues rotations and vec2ang entirely.
    # Requires mp_stacked since it feeds directly into _gather_accum_flatsky_jit.
    use_flatsky  = use_cache and dtheta is not None and dphi is not None \
                   and mp_stacked is not None
    use_nearest  = interp_mode == 'nearest'
    use_gaussian = interp_mode == 'gaussian'
    use_bicubic  = interp_mode == 'bicubic'

    # Resolve Gaussian defaults once (avoids repeating hp.nside2resol per tile).
    if use_gaussian:
        pix_res_deg   = np.degrees(hp.nside2resol(nside))
        _sigma_deg    = sigma_deg  if sigma_deg  is not None else pix_res_deg
        _radius_deg   = radius_deg if radius_deg is not None else 3.0 * _sigma_deg

    # Lower bound from L2 target; upper bound from _MAX_TILES cap.
    # The max() ensures we never produce more than _MAX_TILES tiles even when
    # the memory-based Sc is tiny (e.g. Sc=79 at B=2212 → 64 tiles → 64 C calls).
    Sc = max(1, _S_TILE_TARGET_BYTES // (B * 3 * 4))   # memory target
    Sc = max(Sc, -(-S // _MAX_TILES))                  # tile-count cap (ceiling div)
    Sc = min(Sc, S)

    if use_cache:
        # Map each sample's psi angle to the nearest precomputed bin index.
        # Used by both the flat-sky and single-Rodrigues cached paths.
        n_psi = len(psi_grid)
        dpsi  = _TWO_PI / n_psi
        k_b   = np.mod(
            np.round(np.mod(psis_b, _TWO_PI) / dpsi).astype(np.int64),
            n_psi,
        )   # (B,)

    # Rotation scalars are not needed on the flat-sky path — both rotations
    # are bypassed. Compute them only for the other two paths.
    if not use_flatsky:
        axes, cos_a, sin_a, ax_pts, cos_p, sin_p = _rotation_params(
            rot_vecs, phi_b, theta_b, psis_b)

    tod = {comp: np.zeros(B, dtype=np.float32) for comp in comp_indices}

    for s0 in range(0, S, Sc):
        s1       = min(s0 + Sc, S)
        bv_chunk = beam_vals[s0:s1]   # (Sc,)

        if use_flatsky:
            tod_arr = np.zeros((C, B), dtype=np.float64)
            if use_gaussian:
                # Flat-sky + Gaussian: fused JIT kernel takes dtheta/dphi
                # directly — no Python preprocessing loop, flat-sky dist².
                _gaussian_accum_flatsky(
                    np.ascontiguousarray(dtheta[:, s0:s1]),
                    np.ascontiguousarray(dphi[:, s0:s1]),
                    k_b,
                    np.asarray(theta_b, dtype=np.float64),
                    np.asarray(phi_b,   dtype=np.float64),
                    B, s1 - s0, nside, mp_stacked, bv_chunk, tod_arr,
                    _sigma_deg, _radius_deg,
                )
            elif use_nearest:
                # Flat-sky nearest-pixel path.
                _gather_accum_nearest_flatsky_jit(
                    np.ascontiguousarray(dtheta[:, s0:s1]),
                    np.ascontiguousarray(dphi[:, s0:s1]),
                    k_b,
                    np.asarray(theta_b, dtype=np.float32),
                    np.asarray(phi_b,   dtype=np.float32),
                    nside, mp_stacked, bv_chunk, B, s1 - s0, tod_arr,
                )
            else:
                # Flat-sky bilinear path (default).
                _gather_accum_flatsky_jit(
                    np.ascontiguousarray(dtheta[:, s0:s1]),
                    np.ascontiguousarray(dphi[:, s0:s1]),
                    k_b,
                    np.asarray(theta_b, dtype=np.float32),
                    np.asarray(phi_b,   dtype=np.float32),
                    nside, mp_stacked, bv_chunk, B, s1 - s0, tod_arr,
                )
            for i, comp in enumerate(comp_indices):
                tod[comp] += tod_arr[i].astype(np.float32)

        elif use_cache:
            # Single-Rodrigues path: psi-roll baked in, only recentering needed.
            vec_chunk = vec_rolled[k_b[:, None], np.arange(s0, s1)[None, :], :]
            vec_chunk = np.ascontiguousarray(vec_chunk.astype(np.float32))  # (B, Sc, 3)
            vec_rot   = np.empty((B, s1 - s0, 3), dtype=np.float32)
            _rodrigues1_from_rolled_jit(vec_chunk, axes, cos_a, sin_a, vec_rot)

            if mp_stacked is not None:
                tod_arr = np.zeros((C, B), dtype=np.float64)
                if use_gaussian:
                    theta_flat, phi_flat = hp.vec2ang(
                        vec_rot.reshape(-1, 3).astype(np.float64))
                    _gaussian_interp_accum(theta_flat, phi_flat, B, s1 - s0,
                                           nside, mp_stacked, bv_chunk, tod_arr,
                                           _sigma_deg, _radius_deg)
                elif use_nearest:
                    _gather_accum_nearest_jit(vec_rot, nside, mp_stacked, bv_chunk,
                                              B, s1 - s0, tod_arr)
                elif use_bicubic:
                    _bicubic_interp_accum(vec_rot, B, s1 - s0,
                                          nside, mp_stacked, bv_chunk, tod_arr)
                else:
                    _gather_accum_fused_jit(vec_rot, nside, mp_stacked, bv_chunk,
                                            B, s1 - s0, tod_arr)
                for i, comp in enumerate(comp_indices):
                    tod[comp] += tod_arr[i].astype(np.float32)
            else:
                theta_flat, phi_flat = hp.vec2ang(vec_rot.reshape(-1, 3).astype(np.float64))
                pixels, weights      = get_interp_weights_numba(nside, theta_flat, phi_flat)
                mp_gathered = np.stack([mp[c][pixels] for c in comp_indices])
                mp_flat     = np.einsum('ckn,kn->cn', mp_gathered, weights)
                tod_chunk   = mp_flat.reshape(C, B, s1 - s0) @ bv_chunk
                for i, comp in enumerate(comp_indices):
                    tod[comp] += tod_chunk[i]

        else:
            # Original path: double Rodrigues (recenter + psi roll).
            vec_chunk = np.asarray(vec_orig[s0:s1], dtype=np.float32)  # (Sc, 3)
            vec_rot   = np.empty((B, s1 - s0, 3), dtype=np.float32)
            _rodrigues_jit(vec_chunk, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot)

            if mp_stacked is not None:
                tod_arr = np.zeros((C, B), dtype=np.float64)
                if use_gaussian:
                    theta_flat, phi_flat = hp.vec2ang(
                        vec_rot.reshape(-1, 3).astype(np.float64))
                    _gaussian_interp_accum(theta_flat, phi_flat, B, s1 - s0,
                                           nside, mp_stacked, bv_chunk, tod_arr,
                                           _sigma_deg, _radius_deg)
                elif use_nearest:
                    _gather_accum_nearest_jit(vec_rot, nside, mp_stacked, bv_chunk,
                                              B, s1 - s0, tod_arr)
                elif use_bicubic:
                    _bicubic_interp_accum(vec_rot, B, s1 - s0,
                                          nside, mp_stacked, bv_chunk, tod_arr)
                else:
                    _gather_accum_fused_jit(vec_rot, nside, mp_stacked, bv_chunk,
                                            B, s1 - s0, tod_arr)
                for i, comp in enumerate(comp_indices):
                    tod[comp] += tod_arr[i].astype(np.float32)
            else:
                theta_flat, phi_flat = hp.vec2ang(vec_rot.reshape(-1, 3).astype(np.float64))
                pixels, weights      = get_interp_weights_numba(nside, theta_flat, phi_flat)
                mp_gathered = np.stack([mp[c][pixels] for c in comp_indices])
                mp_flat     = np.einsum('ckn,kn->cn', mp_gathered, weights)
                tod_chunk   = mp_flat.reshape(C, B, s1 - s0) @ bv_chunk
                for i, comp in enumerate(comp_indices):
                    tod[comp] += tod_chunk[i]

    return tod
