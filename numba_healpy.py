"""
Numba JIT replacements for HEALPix RING-scheme helper routines.

These functions mirror the HEALPix C++ internals and are designed to be
called from within parallel Numba kernels.

_ring_above_jit         — scalar ring_above helper (nopython, no parallel).
_ring_info_jit          — scalar ring layout helper: (n_pix, first_pix, phi0, dphi).
_ring_z_jit             — scalar ring centre z = cos(theta) helper.
_get_interp_weights_jit — parallel (prange over N) replacement for
                          hp.get_interp_weights; mirrors the HEALPix C++
                          get_interpol algorithm exactly.
get_interp_weights_numba— public wrapper; drop-in replacement for hp.get_interp_weights.

_ring_interp_single_jit     — bilinear neighbour lookup for one unit-vector query;
                               acos-free for the normal case (sin weight formula).
_ring_interp_with_angles_jit — same as _ring_interp_single_jit but also returns
                               (z_n, phi_n) of each of the 4 neighbours for
                               callers that need the neighbour sky positions
                               (e.g. the spin-2 Q/U frame-rotation kernel).
_pix2ang_ring_jit           — scalar (theta, phi) from RING pixel index (nopython).
_pix2zphi_ring_jit          — scalar (z, phi), avoids acos.
_pix2z_cosphi_sinphi_jit    — scalar (z, cos φ, sin φ), no trig in caller needed.
_pix2ang_ring_batch         — parallel batch kernel over an array of pixel indices.
pix2ang_numba               — public wrapper; drop-in for hp.pix2ang(nest=False).

_query_disc_jit         — nopython query_disc: returns int64 array of RING pixel
                          indices within a disc, callable from inside JIT kernels.
query_disc_numba        — public wrapper; drop-in for hp.query_disc(nest=False).

_gather_ring_stencil_jit — fast Keys/Catmull-Rom stencil gather via ring walk.
                           Replaces _query_disc_into_jit in the bicubic hot loop,
                           eliminating the ~9 acos calls per (b,s) element.

"""

import math
import numpy as np
import numba

# Module-level float64 constants captured by Numba as compile-time literals.
_TWO_PI = 2.0 * math.pi
_INV_TWO_PI = 1.0 / _TWO_PI
_TWO_THIRDS = 2.0 / 3.0  # HEALPix polar-cap / equatorial boundary


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
    if az > _TWO_THIRDS:  # polar cap
        tp = nside * math.sqrt(3.0 * (1.0 - az))
        ir = int(tp)  # floor for tp >= 0
        if z < 0.0:
            ir = 4 * nside - ir - 1  # south-cap mirror
    else:  # equatorial belt
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
    if ir < nside:  # north polar cap
        n_pix = 4 * ir
        first_pix = 2 * ir * (ir - 1)
        s = 1  # always shifted
    elif ir <= 3 * nside:  # equatorial belt
        n_pix = 4 * nside
        first_pix = 2 * nside * (nside - 1) + (ir - nside) * 4 * nside
        # shifted when (ir - nside) is EVEN — matches HEALPix C++ get_ring_info_small
        s = 1 if (ir - nside) % 2 == 0 else 0
    else:  # south polar cap
        i2 = 4 * nside - ir
        n_pix = 4 * i2
        first_pix = npix_total - 2 * i2 * (i2 + 1)
        s = 1  # always shifted
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
        phi = phi_arr[i]

        z = math.cos(theta)
        ir_above = _ring_above_jit(nside, z)
        ir_below = ir_above + 1

        if ir_above == 0:
            # ── North-pole boundary ───────────────────────────────────────────
            # Point is north of ring 1.  Use ring 1 for all pixel selection;
            # the "above" pair are the two opposite pixels in ring 1 (shift +2).
            na, fpa, phi0a, dphia = _ring_info_jit(nside, 1, npix_total)
            tw = ((phi - phi0a) / dphia) % float(na)
            ip = int(tw)
            frac = tw - ip
            ip2 = (ip + 1) % na
            # "below" pixels: the two straddling ring-1 neighbours
            p2 = fpa + ip
            p3 = fpa + ip2
            # "above" pixels: opposite pixels (shifted by na/2 = 2 for ring 1)
            p0 = (ip + 2) % na  # fpa = 0 for ring 1
            p1 = (ip2 + 2) % na
            # theta weight: theta1 = 0 at north pole → w = theta / theta2
            za = _ring_z_jit(nside, 1)
            ta = math.acos(za)
            w_theta = theta / ta
            nf = (1.0 - w_theta) * 0.25  # north_factor (equal spread)
            pix_out[0, i] = p0
            pix_out[1, i] = p1
            pix_out[2, i] = p2
            pix_out[3, i] = p3
            wgt_out[0, i] = nf
            wgt_out[1, i] = nf
            wgt_out[2, i] = (1.0 - frac) * w_theta + nf
            wgt_out[3, i] = frac * w_theta + nf

        elif ir_below == 4 * nside:
            # ── South-pole boundary ───────────────────────────────────────────
            # Point is south of the last ring (4*nside-1).  Use that ring for
            # all pixel selection; the "below" pair are the two opposite pixels.
            ir_last = 4 * nside - 1
            na, fpa, phi0a, dphia = _ring_info_jit(nside, ir_last, npix_total)
            tw = ((phi - phi0a) / dphia) % float(na)
            ip = int(tw)
            frac = tw - ip
            ip2 = (ip + 1) % na
            # "above" pixels: normal ring ir_last neighbours
            p0 = fpa + ip
            p1 = fpa + ip2
            # "below" pixels: opposite pixels in the same 4-pixel last ring
            p2 = (ip + 2) % na + fpa
            p3 = (ip2 + 2) % na + fpa
            # theta weight toward south pole
            za = _ring_z_jit(nside, ir_last)
            ta = math.acos(za)
            w_theta_south = (theta - ta) / (math.pi - ta)
            sf = w_theta_south * 0.25  # south_factor
            pix_out[0, i] = p0
            pix_out[1, i] = p1
            pix_out[2, i] = p2
            pix_out[3, i] = p3
            wgt_out[0, i] = (1.0 - frac) * (1.0 - w_theta_south) + sf
            wgt_out[1, i] = frac * (1.0 - w_theta_south) + sf
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
            tw = ((phi - phi0a) / dphia) % float(na)
            iphia = int(tw)
            fphia = tw - iphia
            pix_out[0, i] = fpa + iphia
            pix_out[1, i] = fpa + (iphia + 1) % na
            wgt_out[0, i] = w_above * (1.0 - fphia)
            wgt_out[1, i] = w_above * fphia

            # Ring below → pixels 2, 3
            nb, fpb, phi0b, dphib = _ring_info_jit(nside, ir_below, npix_total)
            tw = ((phi - phi0b) / dphib) % float(nb)
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
    theta = np.asarray(theta, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()
    N = theta.shape[0]
    pix_out = np.empty((4, N), dtype=np.int64)
    wgt_out = np.empty((4, N), dtype=np.float64)
    _get_interp_weights_jit(nside, theta, phi, pix_out, wgt_out)
    return pix_out, wgt_out


# ── Single-pixel bilinear interpolation ──────────────────────────────────────


@numba.jit(nopython=True, cache=True)
def _ring_interp_single_jit(nside, z, phi_w, npix_total):
    """Bilinear HEALPix RING neighbour lookup for one unit-vector query.

    Mirrors the HEALPix C++ ``get_interpol`` algorithm bit-for-bit, including
    the polar boundary cases.  The θ-linear weight is computed via ``acos``
    so the kernel is numerically identical to ``hp.get_interp_weights`` at
    every nside.

    Parameters
    ----------
    nside      : int
    z          : float   cos θ of the query point, clamped to [−1, 1]
    phi_w      : float   longitude of the query point in [0, 2π)
    npix_total : int     12 * nside * nside (pre-computed by caller)

    Returns
    -------
    p0, p1, p2, p3 : int64    RING pixel indices of the four neighbours
    w0, w1, w2, w3 : float64  bilinear interpolation weights (sum to 1)
    """
    ir_above = _ring_above_jit(nside, z)
    ir_below = ir_above + 1

    if ir_above == 0:
        # ── North-pole boundary ───────────────────────────────────────────────
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
        # Query colatitude θ from clamped z; safe since z in [-1, 1].
        theta = math.acos(z)
        w_theta = theta / ta
        nf = (1.0 - w_theta) * 0.25
        w0 = nf
        w1 = nf
        w2 = (1.0 - frac) * w_theta + nf
        w3 = frac * w_theta + nf

    elif ir_below == 4 * nside:
        # ── South-pole boundary ───────────────────────────────────────────────
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
        theta = math.acos(z)
        w_theta_south = (theta - ta) / (math.pi - ta)
        sf = w_theta_south * 0.25
        w0 = (1.0 - frac) * (1.0 - w_theta_south) + sf
        w1 = frac * (1.0 - w_theta_south) + sf
        w2 = sf
        w3 = sf

    else:
        # ── Normal case — exact θ via acos, matches hp.get_interp_weights ────
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

    return p0, p1, p2, p3, w0, w1, w2, w3


@numba.jit(nopython=True, cache=True)
def _ring_interp_with_angles_jit(nside, z, phi_w, npix_total):
    """Bilinear HEALPix RING neighbour lookup, returning neighbour angles too.

    Identical math to :func:`_ring_interp_single_jit` but additionally returns
    ``(z_n, phi_n)`` for each of the four neighbours.  Intended for callers
    (e.g. the spin-2 Q/U kernel) that need the neighbour sky positions and
    would otherwise have to re-derive them via :func:`_pix2zphi_ring_jit`.

    Parameters
    ----------
    nside      : int
    z          : float   cos θ of the query point, clamped to [−1, 1]
    phi_w      : float   longitude of the query point in [0, 2π)
    npix_total : int     12 * nside * nside (pre-computed by caller)

    Returns
    -------
    p0, p1, p2, p3         : int64    RING pixel indices of the four neighbours
    w0, w1, w2, w3         : float64  bilinear interpolation weights (sum to 1)
    z_n0, z_n1, z_n2, z_n3 : float64  cos θ of each neighbour's ring
    phi_n0, phi_n1, phi_n2, phi_n3 : float64  φ of each neighbour [rad]
    """
    ir_above = _ring_above_jit(nside, z)
    ir_below = ir_above + 1

    if ir_above == 0:
        # ── North-pole boundary ───────────────────────────────────────────────
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
        # ── South-pole boundary ───────────────────────────────────────────────
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
        # ── Normal case — exact θ via acos, matches hp.get_interp_weights ────
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

        nb, fpb, phi0b, dphib = _ring_info_jit(nside, ir_below, npix_total)
        tw = ((phi_w - phi0b) / dphib) % float(nb)
        iphib = int(tw)
        fphib = tw - iphib
        iphib1 = (iphib + 1) % nb
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

    return (
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
    )


# ── HEALPix pix2ang (RING scheme, scalar) ────────────────────────────────────


@numba.jit(nopython=True, cache=True)
def _pix2ang_ring_jit(nside, pix):
    """
    (theta, phi) [rad] for a single RING-scheme pixel index.

    Mirrors hp.pix2ang(nside, pix, nest=False) for a scalar pixel.
    Covers all three zones (north polar cap / equatorial belt / south polar cap)
    and delegates phi to _ring_info_jit so that the shift convention stays
    consistent with the rest of this module.

    Parameters
    ----------
    nside : int
    pix   : int    pixel index in RING scheme

    Returns
    -------
    theta : float  colatitude [rad]
    phi   : float  longitude  [rad]
    """
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)  # pixels in north polar cap

    if pix < ncap:  # ── north polar cap ──
        # Ring iring (1-based) starts at pixel 2*iring*(iring-1).
        iring = int(0.5 * (1.0 + math.sqrt(1.0 + 2.0 * pix)))
        ip_in = pix - 2 * iring * (iring - 1)

    elif pix < npix - ncap:  # ── equatorial belt ──
        ip = pix - ncap
        iring = ip // (4 * nside) + nside
        ip_in = ip % (4 * nside)

    else:  # ── south polar cap ──
        # ip_s counts from the south-pole end (pix=npix-1 → ip_s=0).
        ip_s = npix - pix - 1
        iring_s = int(0.5 * (1.0 + math.sqrt(1.0 + 2.0 * ip_s)))
        iring = 4 * nside - iring_s
        first_s = npix - 2 * iring_s * (iring_s + 1)
        ip_in = pix - first_s

    _n, _fp, phi0, dphi = _ring_info_jit(nside, iring, npix)
    phi = phi0 + ip_in * dphi
    z = _ring_z_jit(nside, iring)
    return math.acos(z), phi


@numba.jit(nopython=True, cache=True)
def _pix2zphi_ring_jit(nside, pix):
    """(z, phi) for a single RING-scheme pixel, where z = cos(theta).

    Identical logic to _pix2ang_ring_jit but returns the ring's z = cos(theta)
    directly instead of acos(z).  Callers that need sin(theta) compute it as
    sqrt(max(0, 1 - z²)) — one sqrt instead of one acos + one sin + one cos.
    """
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)

    if pix < ncap:
        iring = int(0.5 * (1.0 + math.sqrt(1.0 + 2.0 * pix)))
        ip_in = pix - 2 * iring * (iring - 1)
    elif pix < npix - ncap:
        ip = pix - ncap
        iring = ip // (4 * nside) + nside
        ip_in = ip % (4 * nside)
    else:
        ip_s = npix - pix - 1
        iring_s = int(0.5 * (1.0 + math.sqrt(1.0 + 2.0 * ip_s)))
        iring = 4 * nside - iring_s
        first_s = npix - 2 * iring_s * (iring_s + 1)
        ip_in = pix - first_s

    _n, _fp, phi0, dphi_r = _ring_info_jit(nside, iring, npix)
    phi = phi0 + ip_in * dphi_r
    z = _ring_z_jit(nside, iring)
    return z, phi


@numba.jit(nopython=True, cache=True)
def _pix2z_cosphi_sinphi_jit(nside, pix):
    """(z, cos φ, sin φ) for a single RING-scheme pixel.

    Extends _pix2zphi_ring_jit by returning cos and sin of the pixel longitude
    directly, so the caller never needs to call math.cos/math.sin on the result.
    Provided as a trig-saving alternative to _pix2zphi_ring_jit when the caller
    needs cos φ / sin φ rather than φ itself.
    """
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)

    if pix < ncap:
        iring = int(0.5 * (1.0 + math.sqrt(1.0 + 2.0 * pix)))
        ip_in = pix - 2 * iring * (iring - 1)
    elif pix < npix - ncap:
        ip = pix - ncap
        iring = ip // (4 * nside) + nside
        ip_in = ip % (4 * nside)
    else:
        ip_s = npix - pix - 1
        iring_s = int(0.5 * (1.0 + math.sqrt(1.0 + 2.0 * ip_s)))
        iring = 4 * nside - iring_s
        first_s = npix - 2 * iring_s * (iring_s + 1)
        ip_in = pix - first_s

    _n, _fp, phi0, dphi_r = _ring_info_jit(nside, iring, npix)
    phi = phi0 + ip_in * dphi_r
    z = _ring_z_jit(nside, iring)
    return z, math.cos(phi), math.sin(phi)


@numba.jit(nopython=True, parallel=True, cache=True)
def _pix2ang_ring_batch(nside, pix_arr, theta_out, phi_out):
    """Parallel batch kernel: fills theta_out / phi_out for every pix in pix_arr."""
    for i in numba.prange(pix_arr.shape[0]):
        theta_out[i], phi_out[i] = _pix2ang_ring_jit(nside, pix_arr[i])


def pix2ang_numba(nside, pix, nest=False):
    """
    Drop-in Numba replacement for ``hp.pix2ang(nside, pix, nest=False)``.

    Returns ``(theta, phi)`` float64 arrays of shape ``(N,)``.
    Only RING scheme (nest=False) is supported.
    """
    if nest:
        raise ValueError("pix2ang_numba only supports nest=False (RING scheme)")
    pix_arr = np.asarray(pix, dtype=np.int64).ravel()
    N = pix_arr.shape[0]
    theta_out = np.empty(N, dtype=np.float64)
    phi_out = np.empty(N, dtype=np.float64)
    _pix2ang_ring_batch(nside, pix_arr, theta_out, phi_out)
    return theta_out, phi_out


# ── HEALPix ang2pix (RING scheme, scalar) ────────────────────────────────────


@numba.jit(nopython=True, cache=True)
def _ang2pix_ring_jit(nside, theta, phi):
    """
    Nearest RING-scheme pixel index for (theta, phi) [rad].

    Returns the pixel whose centre is geometrically closest to (theta, phi).
    Matches hp.ang2pix(nside, theta, phi, nest=False) for scalar inputs.

    Algorithm
    ---------
    1. Use _ring_above_jit to identify two candidate rings (ir_above and
       ir_above+1) that bracket the query latitude.
    2. For each candidate ring the nearest pixel in phi is found via
       ip = int(phi * n_pix / (2π)) % n_pix — the Voronoi boundary between
       pixel k and k+1 falls at k*dphi regardless of phi0 (shift).
    3. Check both ip and ip+1 for each ring (4 candidates total) and return
       the one with the maximum cos(angular distance).
    """
    npix_total = 12 * nside * nside
    z = math.cos(theta)
    phi_w = phi % _TWO_PI  # wrap to [0, 2π)

    # ── Two candidate global rings bracketing z ───────────────────────────────
    ir_above = _ring_above_jit(nside, z)
    # Clamp so ir_above and ir_below are both valid ring indices.
    if ir_above < 1:
        ir_above = 1
    elif ir_above > 4 * nside - 2:
        ir_above = 4 * nside - 2
    ir_below = ir_above + 1

    # ── For each candidate ring find the best-phi pixel ───────────────────────
    best_pix = -1
    best_cos = -2.0  # maximise cos(angular_dist) ≡ minimise distance
    sin_th = math.sin(theta)
    cos_th = z

    for ir_g in (ir_above, ir_below):
        if ir_g < 1 or ir_g > 4 * nside - 1:
            continue
        n_pix, first_pix, phi0, dphi = _ring_info_jit(nside, ir_g, npix_total)
        z_c = _ring_z_jit(nside, ir_g)
        sin_z_c = math.sqrt(max(0.0, 1.0 - z_c * z_c))
        # Nearest pixel in phi: Voronoi boundary at multiples of dphi.
        ip_base = int(phi_w * n_pix / _TWO_PI) % n_pix
        for ip_try in (ip_base, (ip_base + 1) % n_pix):
            phi_c = phi0 + ip_try * dphi
            cos_d = sin_th * sin_z_c * math.cos(phi_w - phi_c) + cos_th * z_c
            if cos_d > best_cos:
                best_cos = cos_d
                best_pix = first_pix + ip_try

    return best_pix


# ── HEALPix query_disc (RING scheme) ─────────────────────────────────────────


@numba.jit(nopython=True, cache=True)
def _query_disc_jit(nside, theta_q, phi_q, radius_rad, inclusive):
    """
    Pixel indices within angular radius *radius_rad* of (theta_q, phi_q).

    Mirrors hp.query_disc(nside, vec, radius, inclusive=..., nest=False).
    Returns a 1-D int64 array of RING pixel indices (order not guaranteed).

    This function is nopython-safe and can be called from within prange bodies
    or other JIT-compiled kernels.

    Algorithm
    ---------
    1. Optionally widen *radius_rad* by the maximum pixel angular radius when
       inclusive=True (approximated as sqrt(π / (3·nside²))).
    2. Find the ring band [ir_min, ir_max] whose z-centres are within the
       widened disc by calling _ring_above_jit on the disc's latitude limits.
    3. For each ring, compute the phi half-width of the disc cross-section
       (from the spherical-cap intersection formula) and map it to a pixel-
       index range; handle wrap-around via modulo.

    Parameters
    ----------
    nside      : int
    theta_q    : float  disc-centre colatitude [rad]
    phi_q      : float  disc-centre longitude  [rad]
    radius_rad : float  disc radius [rad]
    inclusive  : bool   if True enlarge radius by ~max pixel radius

    Returns
    -------
    result : (M,) int64  RING pixel indices inside the disc
    """
    npix_total = 12 * nside * nside
    z_q = math.cos(theta_q)
    sin_th_q = math.sqrt(max(0.0, 1.0 - z_q * z_q))

    # Approximate max pixel angular radius: sqrt(π / (3·nside²))
    if inclusive:
        search_rad = radius_rad + math.sqrt(math.pi / (3.0 * nside * nside))
    else:
        search_rad = radius_rad

    if search_rad >= math.pi:
        return np.arange(npix_total, dtype=np.int64)

    cos_search = math.cos(search_rad)

    # Ring-index band whose z-centres intersect the widened disc.
    theta_lo = max(0.0, theta_q - search_rad)
    theta_hi = min(math.pi, theta_q + search_rad)
    ir_min = max(1, _ring_above_jit(nside, math.cos(theta_lo)) + 1)
    ir_max = min(4 * nside - 1, _ring_above_jit(nside, math.cos(theta_hi)))

    if ir_min > ir_max:
        return np.empty(0, dtype=np.int64)

    # Conservative upper bound: every ring in the band has at most 4*nside pixels.
    max_pix = 4 * nside * (ir_max - ir_min + 1) + 8
    if max_pix > npix_total:
        max_pix = npix_total

    result = np.empty(max_pix, dtype=np.int64)
    count = 0

    for ir in range(ir_min, ir_max + 1):
        z_r = _ring_z_jit(nside, ir)
        sin_th_r = math.sqrt(max(0.0, 1.0 - z_r * z_r))
        n_p, fp, phi0, dphi = _ring_info_jit(nside, ir, npix_total)

        denom = sin_th_q * sin_th_r
        if denom < 1e-12:
            # Near pole: the whole ring is inside the disc.
            for ip in range(n_p):
                result[count] = fp + ip
                count += 1
            continue

        # cos(dphi_half) from the spherical-cap intersection formula:
        #   cos(d) = sin(θ_q)·sin(θ_r)·cos(Δφ) + cos(θ_q)·cos(θ_r) = cos(search_rad)
        x = (cos_search - z_q * z_r) / denom
        if x > 1.0:
            continue  # ring too far from disc centre
        if x <= -1.0:
            for ip in range(n_p):  # entire ring inside disc
                result[count] = fp + ip
                count += 1
            continue

        dphi_half = math.acos(x)

        # Pixel-index range within ring (may be negative or > n_p, handled by %).
        # Exact ceil/floor — no fudge factor needed since the disc test already
        # has a small tolerance and boundary-exact pixels are not science-critical.
        ip_lo = int(math.ceil((phi_q - dphi_half - phi0) / dphi))
        ip_hi = int(math.floor((phi_q + dphi_half - phi0) / dphi))

        if ip_hi - ip_lo + 1 >= n_p:
            for ip in range(n_p):
                result[count] = fp + ip
                count += 1
        else:
            for ip_idx in range(ip_lo, ip_hi + 1):
                result[count] = fp + ip_idx % n_p
                count += 1

    return result[:count]


@numba.jit(nopython=True, cache=True)
def _query_disc_into_jit(nside, theta_q, phi_q, radius_rad, inclusive, out_buf):
    """
    Like _query_disc_jit but writes pixel indices into a pre-allocated buffer
    instead of allocating a new array.  Returns the count M of pixels found.

    Parameters
    ----------
    nside      : int
    theta_q    : float  disc-centre colatitude [rad]
    phi_q      : float  disc-centre longitude  [rad]
    radius_rad : float  disc radius [rad]
    inclusive  : bool   if True enlarge radius by ~max pixel radius
    out_buf    : (max_M,) int64  caller-allocated scratch buffer

    Returns
    -------
    M : int  number of pixels written into out_buf[:M]
    """
    npix_total = 12 * nside * nside
    z_q = math.cos(theta_q)
    sin_th_q = math.sqrt(max(0.0, 1.0 - z_q * z_q))

    if inclusive:
        search_rad = radius_rad + math.sqrt(math.pi / (3.0 * nside * nside))
    else:
        search_rad = radius_rad

    if search_rad >= math.pi:
        for i in range(npix_total):
            out_buf[i] = i
        return npix_total

    cos_search = math.cos(search_rad)

    theta_lo = max(0.0, theta_q - search_rad)
    theta_hi = min(math.pi, theta_q + search_rad)
    ir_min = max(1, _ring_above_jit(nside, math.cos(theta_lo)) + 1)
    ir_max = min(4 * nside - 1, _ring_above_jit(nside, math.cos(theta_hi)))

    if ir_min > ir_max:
        return 0

    count = 0

    for ir in range(ir_min, ir_max + 1):
        z_r = _ring_z_jit(nside, ir)
        sin_th_r = math.sqrt(max(0.0, 1.0 - z_r * z_r))
        n_p, fp, phi0, dphi = _ring_info_jit(nside, ir, npix_total)

        denom = sin_th_q * sin_th_r
        if denom < 1e-12:
            for ip in range(n_p):
                out_buf[count] = fp + ip
                count += 1
            continue

        x = (cos_search - z_q * z_r) / denom
        if x > 1.0:
            continue
        if x <= -1.0:
            for ip in range(n_p):
                out_buf[count] = fp + ip
                count += 1
            continue

        dphi_half = math.acos(x)

        ip_lo = int(math.ceil((phi_q - dphi_half - phi0) / dphi - 1e-10))
        ip_hi = int(math.floor((phi_q + dphi_half - phi0) / dphi + 1e-10))

        if ip_hi - ip_lo + 1 >= n_p:
            for ip in range(n_p):
                out_buf[count] = fp + ip
                count += 1
        else:
            for ip_idx in range(ip_lo, ip_hi + 1):
                out_buf[count] = fp + ip_idx % n_p
                count += 1

    return count


@numba.jit(nopython=True, cache=True)
def _gather_ring_stencil_jit(nside, vz, ph, out_buf, z_buf, phi_buf):
    """
    Gather RING pixel indices for the Keys/Catmull-Rom bicubic stencil,
    and simultaneously populate z_buf / phi_buf with (cos θ, φ) for each
    gathered pixel — eliminating the need to call _pix2zphi_ring_jit in the
    hot loop.

    While building the stencil this function already has ring geometry in hand
    (from _ring_info_jit and _ring_z_jit).  Returning (z, phi) alongside the
    pixel index removes ~40 redundant function-call chains per (b, s) element:
    each chain would otherwise re-run _ring_info_jit + _ring_z_jit + branch
    logic to recover the same values from the pixel index.

    Replaces _query_disc_into_jit in the bicubic hot loop, eliminating the
    ~9 acos + ~4 cos calls that dominate disc-search cost (~240 ns → ~5 ns).

    Geometry
    --------
    HEALPix ring spacing is ~0.65 h_pix in both the equatorial belt and the
    polar cap, so ±4 rings cover the Keys north–south support |yi| < 2.

    The phi pixel step satisfies:
      · equatorial belt: step ≥ 1.14 h_pix  (at the equatorial–polar boundary)
      · polar cap:       step ≈ √π h_pix ≈ 1.77 h_pix
    In both zones ±2 phi pixels per ring covers the east–west support |xi| < 2.

    Stencil: 7 rings × 5 phi pixels = 35 candidates maximum.
    Rings with n_p ≤ 4 (only ir = 1 at any nside) include all their pixels
    directly, to avoid duplicate indices from modulo wrapping.

    Parameters
    ----------
    nside   : int
    vz      : float64   cos(θ) of the query point  (= vec_rot[b,s,2])
    ph      : float64   longitude [rad], in [0, 2π)
    out_buf : (≥ 45,) int64    caller-allocated pixel index buffer
    z_buf   : (≥ 45,) float64  caller-allocated cos(θ) buffer
    phi_buf : (≥ 45,) float64  caller-allocated φ [rad] buffer

    Returns
    -------
    M : int   number of entries written into out_buf[:M] / z_buf[:M] / phi_buf[:M]
    """
    npix_total = 12 * nside * nside

    ir_center = _ring_above_jit(nside, vz)
    # _ring_above_jit returns 0 at/above the very north pole; clamp to [1, 4n-1].
    if ir_center < 1:
        ir_center = 1
    elif ir_center > 4 * nside - 1:
        ir_center = 4 * nside - 1

    # Ring ±4 is always outside the Keys support: even at the minimum equatorial
    # ring spacing of ~0.65 h_pix, ring ±4 sits at ±2.6 h_pix > 2 h_pix (support
    # boundary).  In the polar cap (spacing ~0.80 h_pix) ring ±3 is already at
    # ±2.4 h_pix > 2, so only rings ±1 and ±2 ever contribute there.  Gathering
    # ring ±4 forces the inner loop to compute coordinates for 10 always-zero-
    # weight candidates.  Using ±3 eliminates those 10 wasted iterations entirely.
    ir_lo = max(1, ir_center - 3)
    ir_hi = min(4 * nside - 1, ir_center + 3)

    count = 0
    for ir in range(ir_lo, ir_hi + 1):
        n_p, fp, phi0, dphi = _ring_info_jit(nside, ir, npix_total)
        z_ring = _ring_z_jit(nside, ir)  # cos(θ) for this ring — computed once

        if n_p <= 4:
            # ir = 1 at any nside: only 4 pixels in the ring.
            # ±2 wrapping would repeat pixel indices, so include all directly.
            for ip in range(n_p):
                out_buf[count] = fp + ip
                z_buf[count] = z_ring
                phi_buf[count] = phi0 + ip * dphi
                count += 1
        else:
            # Nearest pixel in phi, then ±2 neighbours with wrap-around.
            ip_center = int(math.floor((ph - phi0) / dphi + 0.5)) % n_p
            for dip in range(-2, 3):
                ip_in = (ip_center + dip) % n_p
                out_buf[count] = fp + ip_in
                z_buf[count] = z_ring
                phi_buf[count] = phi0 + ip_in * dphi
                count += 1

    return count


def query_disc_numba(nside, vec, radius_rad, inclusive=True, nest=False):
    """
    Drop-in Numba replacement for ``hp.query_disc(nside, vec, radius, ...)``.

    Parameters
    ----------
    nside      : int
    vec        : array-like, shape (3,)   unit vector pointing to disc centre
    radius_rad : float                    disc radius [rad]
    inclusive  : bool   if True, pixels that partially overlap are included
                        (default True, same as healpy's default)
    nest       : bool   only nest=False (RING) is supported

    Returns
    -------
    pix : (M,) int64   RING pixel indices inside the disc (order not guaranteed)
    """
    if nest:
        raise ValueError("query_disc_numba only supports nest=False (RING scheme)")
    v = np.asarray(vec, dtype=np.float64).ravel()
    z = float(np.clip(v[2], -1.0, 1.0))
    theta_q = math.acos(z)
    phi_q = math.atan2(float(v[1]), float(v[0])) % _TWO_PI
    return _query_disc_jit(nside, theta_q, phi_q, float(radius_rad), bool(inclusive))
