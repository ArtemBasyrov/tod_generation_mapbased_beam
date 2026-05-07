"""
Spin-2 frame-rotation primitives for Q/U convolution kernels.

Used by every interpolation path that touches Q/U; not specific to any one
HEALPix interpolation method.

_spin2_cos2d_sin2d_jit       — scalar cos(2δ), sin(2δ) for the spin-2 frame
                               rotation between a HEALPix neighbour and the
                               boresight, using spherical-trig + double-angle
                               identities (no atan2).
_spin2_lookup_cached         — per-boresight direct-mapped cache wrapper that
                               amortises repeated calls to
                               _spin2_cos2d_sin2d_jit on the same neighbour
                               pixel within a beam footprint.
compute_spin2_skip_z_threshold — derives the |cos θ_pts| cutoff below which
                                 the spin-2 correction can be skipped within a
                                 user-supplied tolerance.
"""

import math
import numpy as np
import numba


# Knuth multiplicative-hash constant for the direct-mapped spin-2 cache.
# Chosen to break the spatial clustering of consecutive HEALPix pixel indices
# that would collide on a plain low-bit mask.
_SPIN2_CACHE_HASH = 2654435769

# Direct-mapped spin-2 cache size (power of 2).  Sized to hold the ~100-300
# unique HEALPix pixels covered by a 30′ beam disc at nside=1024 with load
# ≪ 0.1, so collision misses are negligible.  The three backing arrays total
# ~24 KiB, fitting comfortably in L1.
_SPIN2_CACHE_SIZE = 1024
_SPIN2_CACHE_MASK = _SPIN2_CACHE_SIZE - 1


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


@numba.jit(nopython=True, cache=True, inline="always")
def _spin2_lookup_cached(
    p,
    z_n,
    phi_n,
    z_pts,
    sth_pts,
    phi_pts,
    cache_pix,
    cache_c2d,
    cache_s2d,
    cmask,
):
    """Probe the direct-mapped spin-2 cache for pixel ``p``; compute+store on miss.

    Cache slot is selected by Knuth's multiplicative hash so consecutive RING
    pixel indices on the same ring don't collide.  A slot whose stored pixel
    index equals ``p`` is a hit (returns the cached ``(cos 2δ, sin 2δ)``); any
    other value — including the initial sentinel ``-1`` — is a miss, in which
    case the spin-2 rotation is computed via :func:`_spin2_cos2d_sin2d_jit`
    and written into the slot, evicting any previous occupant.

    The cache is boresight-scoped: its contents are valid only for the
    ``(z_pts, sth_pts, phi_pts)`` passed in.  Callers must reset the cache
    (``cache_pix[:] = -1``) whenever they move to a new boresight.

    Parameters
    ----------
    p                         : int64    HEALPix RING pixel index
    z_n, phi_n                : float64  cos θ and φ of pixel ``p``'s centre
    z_pts, sth_pts, phi_pts   : float64  cos θ, sin θ, φ of the boresight
    cache_pix                 : (N,) int64    slot → pixel index (or -1)
    cache_c2d                 : (N,) float64  slot → cached cos(2δ)
    cache_s2d                 : (N,) float64  slot → cached sin(2δ)
    cmask                     : int      N − 1, where N is a power of 2

    Returns
    -------
    c2d, s2d : float64  cos(2δ), sin(2δ) for the pixel → boresight transport.
    """
    slot = (p * _SPIN2_CACHE_HASH) & cmask
    if cache_pix[slot] == p:
        return cache_c2d[slot], cache_s2d[slot]
    sth_n = math.sqrt(max(0.0, 1.0 - z_n * z_n))
    c2d, s2d = _spin2_cos2d_sin2d_jit(z_n, sth_n, phi_n, z_pts, sth_pts, phi_pts)
    cache_pix[slot] = p
    cache_c2d[slot] = c2d
    cache_s2d[slot] = s2d
    return c2d, s2d


def compute_spin2_skip_z_threshold(beam_radius_rad, tol, n_az=128, n_theta=2048):
    """Largest |cos θ_pts| for which the spin-2 Q/U correction can be skipped.

    The spin-2 correction angle |2δ| between a sky pixel and the boresight
    grows as the boresight approaches the poles.  Near the equator |2δ| is
    negligible and skipping the rotation introduces an error
    ≈ |2δ| · max(|Q|, |U|).  This helper sweeps boresight colatitudes
    θ_pts ∈ (0, π/2] and finds the largest |z| = |cos θ_pts| at which the
    worst-case |2δ| over all beam-pixel positions within
    ``beam_radius_rad`` of the boresight is still below ``tol``.

    The returned value is intended for the kernel's per-``b`` skip test:

        apply_spin2 = abs(bz) > z_skip_threshold

    With this convention the spin-2 lookup runs near the poles (large |bz|)
    and is bypassed in the equatorial band (small |bz|).  Returning ``-1.0``
    means the optimisation is disabled — the test never succeeds and the
    correction is always applied (bit-identical to the un-optimised path).

    Parameters
    ----------
    beam_radius_rad : float   maximum angular distance from the boresight to
                              any beam pixel that can contribute (radians).
    tol             : float   tolerance on |2δ| in radians; interpreted as a
                              fractional Q/U accuracy bound (e.g. 0.01 → 1%).
                              Set ``tol <= 0`` to disable the optimisation.
    n_az            : int     azimuth samples on the beam-edge ring used to
                              find the worst-case |2δ| at each boresight.
    n_theta         : int     boresight colatitude samples between equator
                              and pole; finer → tighter (closer-to-pole)
                              threshold within the same tolerance.

    Returns
    -------
    z_threshold : float in [-1.0, 1.0]
        ``-1.0`` when the optimisation is disabled; otherwise the largest
        ``|cos θ_pts|`` for which |2δ| stays under ``tol``.
    """
    if tol is None or tol <= 0.0:
        return -1.0
    if beam_radius_rad <= 0.0:
        return 1.0  # no beam → no correction needed anywhere

    # Sweep θ_pts from equator (max |z| = 0) toward pole.  Stop when the
    # worst-case |2δ| crosses ``tol``; the previous (closer-to-equator) θ_pts
    # is the safest bound.
    z_threshold = -1.0
    # Skip the equator point itself (cos θ_pts = 0 → z_threshold = 0 means
    # a useless test); start one step in.
    for i in range(1, n_theta + 1):
        theta_pts = math.pi * 0.5 * (1.0 - i / n_theta)  # → 0 at i = n_theta
        if theta_pts <= 0.0:
            break
        max_2d = _max_two_delta_at_boresight(theta_pts, beam_radius_rad, n_az)
        if max_2d > tol:
            break
        z_threshold = abs(math.cos(theta_pts))

    return z_threshold


def _max_two_delta_at_boresight(theta_pts, R, n_az):
    """Worst-case |2δ| at boresight colatitude ``theta_pts`` over a beam edge ring.

    Samples sky positions at angular distance ``R`` from the boresight on a
    ring of ``n_az`` azimuths and returns the maximum |2δ|, where
    2δ is the spin-2 frame-rotation angle returned by
    :func:`_spin2_cos2d_sin2d_jit`.  Pure-equator boresight (sin θ_pts = 0)
    is not handled — the caller passes θ_pts > 0 only.
    """
    z_pts = math.cos(theta_pts)
    sth_pts = math.sin(theta_pts)
    phi_pts = 0.0  # arbitrary; rotational symmetry in φ

    cos_R = math.cos(R)
    sin_R = math.sin(R)

    max_two_d = 0.0
    for i in range(n_az):
        alpha = 2.0 * math.pi * i / n_az
        # Spherical destination: from boresight (θ_pts, 0), step R along bearing α.
        z_pix = cos_R * z_pts + sin_R * sth_pts * math.cos(alpha)
        if z_pix > 1.0:
            z_pix = 1.0
        elif z_pix < -1.0:
            z_pix = -1.0
        sth_pix = math.sqrt(max(0.0, 1.0 - z_pix * z_pix))
        if sth_pix < 1e-12:
            continue  # destination at pole — δ undefined, skip
        sin_dphi = sin_R * math.sin(alpha) / sth_pix
        # cos(Δφ) from spherical-trig: cos(R) = z_pix·z_pts + sth_pix·sth_pts·cos(Δφ)
        cos_dphi = (cos_R - z_pix * z_pts) / (sth_pix * sth_pts)
        # Numerical safety
        if cos_dphi > 1.0:
            cos_dphi = 1.0
        elif cos_dphi < -1.0:
            cos_dphi = -1.0
        phi_pix = math.atan2(sin_dphi, cos_dphi)

        c2d, s2d = _spin2_cos2d_sin2d_jit(
            z_pix, sth_pix, phi_pix, z_pts, sth_pts, phi_pts
        )
        two_d = abs(math.atan2(s2d, c2d))
        if two_d > max_two_d:
            max_two_d = two_d

    return max_two_d
