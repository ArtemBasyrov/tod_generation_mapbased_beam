"""
Numba JIT replacements for HEALPix RING-scheme helper routines.

These functions mirror the HEALPix C++ internals and are designed to be
called from within parallel Numba kernels.

_ring_above_jit         — scalar ring_above helper (nopython, no parallel).
_ring_info_jit          — scalar ring layout helper: (n_pix, first_pix, phi0, dphi).
_ring_z_jit             — scalar ring centre z = cos(theta) helper.
_wrap_ring_phase        — exact branch-based replacement for the fractional
                          ring position's modulo.
_build_ring_theta       — per-ring acos table hoisted out of the sample loop.
_get_interp_weights_jit — parallel (prange over N) replacement for
                          hp.get_interp_weights; mirrors the HEALPix C++
                          get_interpol algorithm exactly.
get_interp_weights_numba— public wrapper; drop-in replacement for hp.get_interp_weights.

_ring_interp_single_jit     — bilinear neighbour lookup for one unit-vector
                               query; numerically identical to
                               ``hp.get_interp_weights`` (uses acos for the
                               θ-linear weight).
_ring_interp_with_angles_jit — same as _ring_interp_single_jit but also returns
                               (z_n, phi_n) of each of the 4 neighbours for
                               callers that need the neighbour sky positions
                               (e.g. the spin-2 Q/U frame-rotation kernel).

Scope: only what the production kernels on this branch call.  Nearest-pixel
lookup lives inlined in ``tod_nearest`` rather than here, and the pix2ang,
query_disc and bicubic-stencil helpers this module used to carry were only
ever reachable from the retired ``gaussian`` and ``bicubic`` branches — they
remain on those branches, which keep their own copy of this file.
"""

import math
import numpy as np
import numba

# Module-level float64 constants captured by Numba as compile-time literals.
_TWO_PI = 2.0 * math.pi
_TWO_THIRDS = 2.0 / 3.0  # HEALPix polar-cap / equatorial boundary


# These three helpers must NOT carry parallel=True: they are called from
# within a prange body inside _get_interp_weights_jit.


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


@numba.jit(nopython=True, cache=True, inline="always")
def _wrap_ring_phase(tw, n_pix):
    """Reduce a fractional ring position into ``[0, n_pix)``, as ``tw % n_pix``.

    ``tw = (phi - phi0) / dphi`` lies in ``[-0.5, n_pix]``: ``phi0`` is either 0
    or half a pixel, and ``phi`` reaches exactly 2π when the caller maps a
    negative ``atan2`` result into ``[0, 2π)``.  A single correction on either
    side therefore reproduces the modulo exactly, with no ``fmod`` call.

    **Both branches are load-bearing.** Dropping the upper one is silently wrong
    rather than merely imprecise: on the half of equatorial-belt rings that are
    unshifted (``phi0 = 0``), ``2π / dphi`` is an exact power-of-two quotient, so
    ``tw == n_pix`` exactly, and leaving it unreduced selects the *first pixel of
    the next ring*.  See ``TestWrapRingPhase`` in ``tests/test_numba_healpy.py``.
    """
    if tw < 0.0:
        return tw + n_pix
    elif tw >= n_pix:
        return tw - n_pix
    return tw


@numba.jit(nopython=True, cache=True)
def _build_ring_theta(nside):
    """Colatitudes of every RING ring centre, indexed by 1-based ring number.

    ``acos`` of a ring centre depends only on the ring, so the bilinear kernels
    hoist it here instead of recomputing it per sample.  Entry ``ir`` holds
    exactly ``math.acos(_ring_z_jit(nside, ir))`` — the same call on the same
    input — so a table lookup is bit-identical to the inline computation, not an
    approximation.  Slot 0 is unused (rings are 1-based).

    The table must be rebuilt whenever ``nside`` changes; the kernels build it
    per call, which makes that automatic.

    Returns:
        numpy.ndarray: ``(4 * nside,)`` float64 ring colatitudes [rad].
    """
    n = 4 * nside
    out = np.empty(n, dtype=np.float64)
    out[0] = 0.0
    for ir in range(1, n):
        out[ir] = math.acos(_ring_z_jit(nside, ir))
    return out


# ── Standalone parallel interp-weights kernel ─────────────────────────────────


@numba.jit(nopython=True, parallel=True, cache=True)
def _get_interp_weights_jit(nside, theta_arr, phi_arr, pix_out, wgt_out):
    """
    Parallel Numba replacement for hp.get_interp_weights (RING scheme).

    Mirrors the HEALPix get_interpol algorithm including the north/south pole
    boundary special cases.  Each of the N iterations is fully independent;
    parallelised with prange.

    Phi interpolation reduces the fractional ring position with
    :func:`_wrap_ring_phase` then floors it, so points with phi < phi0 wrap
    correctly.  Ring colatitudes come from a :func:`_build_ring_theta` table
    built once here rather than an ``acos`` per sample.

    Parameters
    ----------
    nside     : int
    theta_arr : (N,) float64   colatitude [rad]
    phi_arr   : (N,) float64   longitude  [rad]
    pix_out   : (4, N) int64   written in place
    wgt_out   : (4, N) float64 written in place
    """
    npix_total = 12 * nside * nside
    ring_theta = _build_ring_theta(nside)
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
            tw = _wrap_ring_phase((phi - phi0a) / dphia, na)
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
            ta = ring_theta[1]
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
            tw = _wrap_ring_phase((phi - phi0a) / dphia, na)
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
            ta = ring_theta[ir_last]
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
            ta = ring_theta[ir_above]
            tb = ring_theta[ir_below]
            w_below = (theta - ta) / (tb - ta)
            w_above = 1.0 - w_below

            # Ring above → pixels 0, 1
            na, fpa, phi0a, dphia = _ring_info_jit(nside, ir_above, npix_total)
            tw = _wrap_ring_phase((phi - phi0a) / dphia, na)
            iphia = int(tw)
            fphia = tw - iphia
            pix_out[0, i] = fpa + iphia
            pix_out[1, i] = fpa + (iphia + 1) % na
            wgt_out[0, i] = w_above * (1.0 - fphia)
            wgt_out[1, i] = w_above * fphia

            # Ring below → pixels 2, 3
            nb, fpb, phi0b, dphib = _ring_info_jit(nside, ir_below, npix_total)
            tw = _wrap_ring_phase((phi - phi0b) / dphib, nb)
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
def _ring_interp_single_jit(nside, z, phi_w, npix_total, ring_theta=None):
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
    ring_theta : (4*nside,) float64 or None
        Optional :func:`_build_ring_theta` table.  When given, ring
        colatitudes are read from it instead of recomputed with ``acos``;
        the table stores exactly those ``acos`` values, so results are
        bit-identical either way.  Hot callers pass it; everyone else omits it.

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
        tw = _wrap_ring_phase((phi_w - phi0a) / dphia, na)
        ip_a = int(tw)
        frac = tw - ip_a
        ip_a2 = (ip_a + 1) % na
        p0 = fpa + (ip_a + 2) % na
        p1 = fpa + (ip_a2 + 2) % na
        p2 = fpa + ip_a
        p3 = fpa + ip_a2
        za = _ring_z_jit(nside, 1)
        ta = math.acos(za) if ring_theta is None else ring_theta[1]
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
        tw = _wrap_ring_phase((phi_w - phi0a) / dphia, na)
        ip_a = int(tw)
        frac = tw - ip_a
        ip_a2 = (ip_a + 1) % na
        p0 = fpa + ip_a
        p1 = fpa + ip_a2
        p2 = (ip_a + 2) % na + fpa
        p3 = (ip_a2 + 2) % na + fpa
        za = _ring_z_jit(nside, ir_last)
        ta = math.acos(za) if ring_theta is None else ring_theta[ir_last]
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
        if ring_theta is None:
            ta = math.acos(za)
            tb = math.acos(zb)
        else:
            ta = ring_theta[ir_above]
            tb = ring_theta[ir_below]
        theta = math.acos(z)
        w_below = (theta - ta) / (tb - ta)
        w_above = 1.0 - w_below

        na, fpa, phi0a, dphia = _ring_info_jit(nside, ir_above, npix_total)
        tw = _wrap_ring_phase((phi_w - phi0a) / dphia, na)
        iphia = int(tw)
        fphia = tw - iphia
        p0 = fpa + iphia
        p1 = fpa + (iphia + 1) % na
        w0 = w_above * (1.0 - fphia)
        w1 = w_above * fphia

        nb, fpb, phi0b, dphib = _ring_info_jit(nside, ir_below, npix_total)
        tw = _wrap_ring_phase((phi_w - phi0b) / dphib, nb)
        iphib = int(tw)
        fphib = tw - iphib
        p2 = fpb + iphib
        p3 = fpb + (iphib + 1) % nb
        w2 = w_below * (1.0 - fphib)
        w3 = w_below * fphib

    return p0, p1, p2, p3, w0, w1, w2, w3


@numba.jit(nopython=True, cache=True)
def _ring_interp_with_angles_jit(nside, z, phi_w, npix_total, ring_theta=None):
    """Bilinear HEALPix RING neighbour lookup, returning neighbour angles too.

    Identical math to :func:`_ring_interp_single_jit` but additionally returns
    ``(z_n, phi_n)`` for each of the four neighbours.  Intended for callers
    (e.g. the spin-2 Q/U kernel) that need the neighbour sky positions and
    would otherwise have to re-derive them from the pixel index.

    Parameters
    ----------
    nside      : int
    z          : float   cos θ of the query point, clamped to [−1, 1]
    phi_w      : float   longitude of the query point in [0, 2π)
    npix_total : int     12 * nside * nside (pre-computed by caller)
    ring_theta : (4*nside,) float64 or None
        Optional :func:`_build_ring_theta` table.  When given, ring
        colatitudes are read from it instead of recomputed with ``acos``;
        the table stores exactly those ``acos`` values, so results are
        bit-identical either way.  Hot callers pass it; everyone else omits it.

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
        tw = _wrap_ring_phase((phi_w - phi0a) / dphia, na)
        ip_a = int(tw)
        frac = tw - ip_a
        ip_a2 = (ip_a + 1) % na
        p0 = fpa + (ip_a + 2) % na
        p1 = fpa + (ip_a2 + 2) % na
        p2 = fpa + ip_a
        p3 = fpa + ip_a2
        za = _ring_z_jit(nside, 1)
        ta = math.acos(za) if ring_theta is None else ring_theta[1]
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
        tw = _wrap_ring_phase((phi_w - phi0a) / dphia, na)
        ip_a = int(tw)
        frac = tw - ip_a
        ip_a2 = (ip_a + 1) % na
        p0 = fpa + ip_a
        p1 = fpa + ip_a2
        p2 = fpa + (ip_a + 2) % na
        p3 = fpa + (ip_a2 + 2) % na
        za = _ring_z_jit(nside, ir_last)
        ta = math.acos(za) if ring_theta is None else ring_theta[ir_last]
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
        if ring_theta is None:
            ta = math.acos(za)
            tb = math.acos(zb)
        else:
            ta = ring_theta[ir_above]
            tb = ring_theta[ir_below]
        theta = math.acos(z)
        w_below = (theta - ta) / (tb - ta)
        w_above = 1.0 - w_below

        na, fpa, phi0a, dphia = _ring_info_jit(nside, ir_above, npix_total)
        tw = _wrap_ring_phase((phi_w - phi0a) / dphia, na)
        iphia = int(tw)
        fphia = tw - iphia
        iphia1 = (iphia + 1) % na
        p0 = fpa + iphia
        p1 = fpa + iphia1
        w0 = w_above * (1.0 - fphia)
        w1 = w_above * fphia

        nb, fpb, phi0b, dphib = _ring_info_jit(nside, ir_below, npix_total)
        tw = _wrap_ring_phase((phi_w - phi0b) / dphib, nb)
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
