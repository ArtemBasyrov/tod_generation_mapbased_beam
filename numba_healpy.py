"""
Numba JIT replacements for HEALPix RING-scheme helper routines.

These functions mirror the HEALPix C++ internals for get_interpol and are
designed to be called from within parallel Numba kernels.

_ring_above_jit         — scalar ring_above helper (nopython, no parallel).
_ring_info_jit          — scalar ring layout helper: (n_pix, first_pix, phi0, dphi).
_ring_z_jit             — scalar ring centre z = cos(theta) helper.
_get_interp_weights_jit — parallel (prange over N) replacement for
                          hp.get_interp_weights; mirrors the HEALPix C++
                          get_interpol algorithm exactly.
get_interp_weights_numba— public wrapper; drop-in replacement for hp.get_interp_weights.
"""
import math
import numpy as np
import numba

# Module-level float64 constants captured by Numba as compile-time literals.
_TWO_PI      = 2.0 * math.pi
_INV_TWO_PI  = 1.0 / _TWO_PI
_TWO_THIRDS  = 2.0 / 3.0          # HEALPix polar-cap / equatorial boundary


# ── HEALPix RING-scheme helpers (nopython, no parallel) ───────────────────────
# These three functions mirror the HEALPix C++ internals for get_interpol.
# They must NOT carry parallel=True because they are called from within a
# prange body inside _get_interp_weights_jit.

@numba.jit(nopython=True, cache=True)
def _ring_above_jit(nside, z):
    """
    Index of the last ring whose z-centre is > z (HEALPix RING, 1-based).

    Mirrors ring_above() in healpix_base.cc.  Returns 0 when z is above ring 1
    (near north pole) and 4*nside-1 when z is below ring 4*nside-1 (near south
    pole); the caller is responsible for clamping as needed.
    """
    az = abs(z)
    if az > _TWO_THIRDS:                          # polar cap
        tp = nside * math.sqrt(3.0 * (1.0 - az))
        ir = int(tp)                              # floor for tp >= 0
        if z < 0.0:
            ir = 4 * nside - ir - 1              # south-cap mirror
    else:                                         # equatorial belt
        ir = int(nside * (2.0 - 1.5 * z))
    return ir


@numba.jit(nopython=True, cache=True)
def _ring_info_jit(nside, ir, npix_total):
    """
    Ring layout for ring ir (1-based, RING scheme).

    Returns
    -------
    n_pix     : int    number of pixels in the ring
    first_pix : int    index of the first pixel in the ring
    phi0      : float  longitude of the first pixel [rad]
    dphi      : float  pixel angular spacing [rad]
    """
    if ir < nside:                                # north polar cap
        n_pix     = 4 * ir
        first_pix = 2 * ir * (ir - 1)
        s         = 1                             # always shifted
    elif ir <= 3 * nside:                         # equatorial belt
        n_pix     = 4 * nside
        first_pix = 2 * nside * (nside - 1) + (ir - nside) * 4 * nside
        # shifted when (ir - nside) is EVEN — matches HEALPix C++ get_ring_info_small
        s         = 1 if (ir - nside) % 2 == 0 else 0
    else:                                         # south polar cap
        i2        = 4 * nside - ir
        n_pix     = 4 * i2
        first_pix = npix_total - 2 * i2 * (i2 + 1)
        s         = 1                             # always shifted
    dphi = _TWO_PI / n_pix
    phi0 = s * dphi * 0.5
    return n_pix, first_pix, phi0, dphi


@numba.jit(nopython=True, cache=True)
def _ring_z_jit(nside, ir):
    """cos(theta) at the centre of ring ir (1-based, RING scheme)."""
    if ir < nside:
        tmp = float(ir)
        return 1.0 - tmp * tmp / (3.0 * nside * nside)
    elif ir <= 3 * nside:
        return (2.0 / 3.0) * (2.0 - float(ir) / nside)
    else:
        tmp = float(4 * nside - ir)
        return -(1.0 - tmp * tmp / (3.0 * nside * nside))


# ── Standalone parallel interp-weights kernel ─────────────────────────────────

@numba.jit(nopython=True, parallel=True, cache=True)
def _get_interp_weights_jit(nside, theta_arr, phi_arr, pix_out, wgt_out):
    """
    Parallel Numba replacement for hp.get_interp_weights (RING scheme).

    Mirrors the HEALPix get_interpol algorithm including the north/south pole
    boundary special cases.  Each of the N iterations is fully independent;
    parallelised with prange.

    Phi interpolation uses float-modulo then floor (equivalent to the
    jax_healpy reference) so that points with phi < phi0 wrap correctly.

    Parameters
    ----------
    nside     : int
    theta_arr : (N,) float64   colatitude [rad]
    phi_arr   : (N,) float64   longitude  [rad]
    pix_out   : (4, N) int64   written in place
    wgt_out   : (4, N) float64 written in place
    """
    npix_total = 12 * nside * nside
    N = theta_arr.shape[0]
    for i in numba.prange(N):
        theta = theta_arr[i]
        phi   = phi_arr[i]

        z        = math.cos(theta)
        ir_above = _ring_above_jit(nside, z)
        ir_below = ir_above + 1

        if ir_above == 0:
            # ── North-pole boundary ───────────────────────────────────────────
            # Point is north of ring 1.  Use ring 1 for all pixel selection;
            # the "above" pair are the two opposite pixels in ring 1 (shift +2).
            na, fpa, phi0a, dphia = _ring_info_jit(nside, 1, npix_total)
            tw   = ((phi - phi0a) / dphia) % float(na)
            ip   = int(tw)
            frac = tw - ip
            ip2  = (ip + 1) % na
            # "below" pixels: the two straddling ring-1 neighbours
            p2 = fpa + ip
            p3 = fpa + ip2
            # "above" pixels: opposite pixels (shifted by na/2 = 2 for ring 1)
            p0 = (ip  + 2) % na          # fpa = 0 for ring 1
            p1 = (ip2 + 2) % na
            # theta weight: theta1 = 0 at north pole → w = theta / theta2
            za      = _ring_z_jit(nside, 1)
            ta      = math.acos(za)
            w_theta = theta / ta
            nf      = (1.0 - w_theta) * 0.25   # north_factor (equal spread)
            pix_out[0, i] = p0
            pix_out[1, i] = p1
            pix_out[2, i] = p2
            pix_out[3, i] = p3
            wgt_out[0, i] = nf
            wgt_out[1, i] = nf
            wgt_out[2, i] = (1.0 - frac) * w_theta + nf
            wgt_out[3, i] = frac          * w_theta + nf

        elif ir_below == 4 * nside:
            # ── South-pole boundary ───────────────────────────────────────────
            # Point is south of the last ring (4*nside-1).  Use that ring for
            # all pixel selection; the "below" pair are the two opposite pixels.
            ir_last = 4 * nside - 1
            na, fpa, phi0a, dphia = _ring_info_jit(nside, ir_last, npix_total)
            tw   = ((phi - phi0a) / dphia) % float(na)
            ip   = int(tw)
            frac = tw - ip
            ip2  = (ip + 1) % na
            # "above" pixels: normal ring ir_last neighbours
            p0 = fpa + ip
            p1 = fpa + ip2
            # "below" pixels: opposite pixels in the same 4-pixel last ring
            p2 = (ip  + 2) % na + fpa
            p3 = (ip2 + 2) % na + fpa
            # theta weight toward south pole
            za            = _ring_z_jit(nside, ir_last)
            ta            = math.acos(za)
            w_theta_south = (theta - ta) / (math.pi - ta)
            sf            = w_theta_south * 0.25           # south_factor
            pix_out[0, i] = p0
            pix_out[1, i] = p1
            pix_out[2, i] = p2
            pix_out[3, i] = p3
            wgt_out[0, i] = (1.0 - frac) * (1.0 - w_theta_south) + sf
            wgt_out[1, i] = frac          * (1.0 - w_theta_south) + sf
            wgt_out[2, i] = sf
            wgt_out[3, i] = sf

        else:
            # ── Normal case ───────────────────────────────────────────────────
            za = _ring_z_jit(nside, ir_above)
            zb = _ring_z_jit(nside, ir_below)
            ta = math.acos(za)
            tb = math.acos(zb)
            w_below = (theta - ta) / (tb - ta)
            w_above = 1.0 - w_below

            # Ring above → pixels 0, 1
            na, fpa, phi0a, dphia = _ring_info_jit(nside, ir_above, npix_total)
            tw    = ((phi - phi0a) / dphia) % float(na)
            iphia = int(tw)
            fphia = tw - iphia
            pix_out[0, i] = fpa + iphia
            pix_out[1, i] = fpa + (iphia + 1) % na
            wgt_out[0, i] = w_above * (1.0 - fphia)
            wgt_out[1, i] = w_above * fphia

            # Ring below → pixels 2, 3
            nb, fpb, phi0b, dphib = _ring_info_jit(nside, ir_below, npix_total)
            tw    = ((phi - phi0b) / dphib) % float(nb)
            iphib = int(tw)
            fphib = tw - iphib
            pix_out[2, i] = fpb + iphib
            pix_out[3, i] = fpb + (iphib + 1) % nb
            wgt_out[2, i] = w_below * (1.0 - fphib)
            wgt_out[3, i] = w_below * fphib


def get_interp_weights_numba(nside, theta, phi):
    """
    Drop-in Numba replacement for ``hp.get_interp_weights(nside, theta, phi)``.

    Returns ``(pixels, weights)`` with shapes ``(4, N)`` and dtypes ``int64`` /
    ``float64``, identical to the healpy convention.  Input arrays are
    automatically cast to float64 and ravelled.
    """
    theta   = np.asarray(theta, dtype=np.float64).ravel()
    phi     = np.asarray(phi,   dtype=np.float64).ravel()
    N       = theta.shape[0]
    pix_out = np.empty((4, N), dtype=np.int64)
    wgt_out = np.empty((4, N), dtype=np.float64)
    _get_interp_weights_jit(nside, theta, phi, pix_out, wgt_out)
    return pix_out, wgt_out
