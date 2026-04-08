"""
Gaussian kernel interpolation for TOD generation.

Provides isotropic Gaussian-weighted sky-map accumulation as an
alternative to the default bilinear HEALPix interpolation.

_angular_distance           — great-circle distance helper
_gaussian_interp_accum_jit  — Numba JIT kernel (parallel over N=B*Sc)
_gaussian_interp_accum      — Python wrapper (allocates scratch buffers)
_gaussian_accum_flatsky_jit — fused flat-sky JIT kernel
_gaussian_accum_flatsky     — Python wrapper for flat-sky variant
"""

import math

import numba
import numpy as np

from numba_healpy import (
    _TWO_PI,
    _ang2pix_ring_jit,
    _pix2ang_ring_jit,
    _query_disc_into_jit,
)


def _angular_distance(th1, ph1, th2, ph2):
    """Great-circle angular distance between (th1,ph1) and arrays (th2,ph2) [rad]."""
    cos_d = np.sin(th1) * np.sin(th2) * np.cos(ph1 - ph2) + np.cos(th1) * np.cos(th2)
    return np.arccos(np.clip(cos_d, -1.0, 1.0))


@numba.jit(nopython=True, parallel=True, cache=True)
def _gaussian_interp_accum_jit(
    theta_flat,
    phi_flat,
    B,
    Sc,
    nside,
    mp_stacked,
    beam_vals,
    tod_arr,
    sigma_rad,
    radius_rad,
    scratch_pix,
    scratch_w,
    tod_tmp,
):
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
        b = idx // Sc
        s = idx % Sc
        th = theta_flat[idx]
        ph = phi_flat[idx]
        bv = float(beam_vals[s])

        pix_buf = scratch_pix[tid]
        w_buf = scratch_w[tid]

        M = _query_disc_into_jit(nside, th, ph, radius_rad, True, pix_buf)

        if M == 0:
            nearest = _ang2pix_ring_jit(nside, th, ph)
            for c in range(C):
                tod_tmp[c, idx] = mp_stacked[c, nearest] * bv
            continue

        # Flat-sky dist²: Δθ² + sin²(θ)·Δφ².  Valid for radii < ~5°.
        # _pix2ang_ring_jit is pure arithmetic — no ang_lut memory access.
        sin2_th = math.sin(th) ** 2
        w_sum = 0.0
        for k in range(M):
            theta_n, phi_n = _pix2ang_ring_jit(nside, pix_buf[k])
            dth = theta_n - th
            dph = phi_n - ph
            if dph > math.pi:
                dph -= _TWO_PI
            elif dph < -math.pi:
                dph += _TWO_PI
            w = math.exp(-(dth * dth + sin2_th * dph * dph) / two_sigma2)
            w_buf[k] = w
            w_sum += w

        # With radius = 3σ the nearest disc pixel has weight ≥ exp(-9/2) ≈ 0.011,
        # so w_sum < 1e-30 is unreachable under normal configuration. Guard kept
        # as a safety net against extreme sigma/radius misconfigurations.
        if w_sum < 1e-30:
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


def _gaussian_interp_accum(
    theta_flat,
    phi_flat,
    B,
    Sc,
    nside,
    mp_stacked,
    beam_vals,
    tod_arr,
    sigma_deg,
    radius_deg,
):
    """
    Wrapper: converts degrees → radians, allocates thread-local scratch
    buffers and tod_tmp work array, then calls the JIT kernel.
    """
    sigma_rad = math.radians(sigma_deg)
    radius_rad = math.radians(radius_deg)

    # Solid-angle upper bound on disc pixel count (with 3× safety margin).
    # Formula: Ω_cap = 2π(1−cos r); pixels ≈ Ω_cap × 3·nside²/π.
    # This gives ~132 at all production nsides (actual M ≈ 46), keeping
    # scratch buffers at ~2 KB/thread (L1-resident) vs the old ring-spanning
    # formula that produced 40 000–82 000 (640–1280 KB/thread, L3 thrashing).
    search_rad = radius_rad + math.sqrt(math.pi / (3.0 * nside * nside))
    max_M = max(64, int(12.0 * nside * nside * (1.0 - math.cos(search_rad))) + 32)

    n_threads = numba.get_num_threads()
    scratch_pix = np.empty((n_threads, max_M), dtype=np.int64)
    scratch_w = np.empty((n_threads, max_M), dtype=np.float64)
    tod_tmp = np.zeros((mp_stacked.shape[0], B * Sc), dtype=np.float64)

    _gaussian_interp_accum_jit(
        theta_flat,
        phi_flat,
        B,
        Sc,
        nside,
        mp_stacked,
        beam_vals,
        tod_arr,
        sigma_rad,
        radius_rad,
        scratch_pix,
        scratch_w,
        tod_tmp,
    )


@numba.jit(nopython=True, parallel=True, cache=True)
def _gaussian_accum_flatsky_jit(
    dtheta_tile,
    dphi_tile,
    k_b,
    theta_b,
    phi_b,
    B,
    Sc,
    nside,
    mp_stacked,
    beam_vals,
    tod_arr,
    sigma_rad,
    radius_rad,
    scratch_pix,
    scratch_w,
    tod_tmp,
):
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
        b = idx // Sc
        s = idx % Sc
        kb = k_b[b]

        th_b = theta_b[b]
        sin_th_b = math.sin(th_b)
        inv_sin = 1.0 / sin_th_b if sin_th_b > 1e-10 else 0.0

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
        w_buf = scratch_w[tid]

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
            dth = theta_n - th
            dph = phi_n - ph
            if dph > math.pi:
                dph -= _TWO_PI
            elif dph < -math.pi:
                dph += _TWO_PI
            dist2 = dth * dth + sin2_th * dph * dph
            w = math.exp(-dist2 / two_sigma2)
            w_buf[k] = w
            w_sum += w

        # With radius = 3σ the nearest disc pixel has weight ≥ exp(-9/2) ≈ 0.011,
        # so w_sum < 1e-30 is unreachable under normal configuration. Guard kept
        # as a safety net against extreme sigma/radius misconfigurations.
        if w_sum < 1e-30:
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


def _gaussian_accum_flatsky(
    dtheta_tile,
    dphi_tile,
    k_b,
    theta_b,
    phi_b,
    B,
    Sc,
    nside,
    mp_stacked,
    beam_vals,
    tod_arr,
    sigma_deg,
    radius_deg,
):
    """Wrapper: allocates scratch buffers and calls _gaussian_accum_flatsky_jit."""
    sigma_rad = math.radians(sigma_deg)
    radius_rad = math.radians(radius_deg)

    search_rad = radius_rad + math.sqrt(math.pi / (3.0 * nside * nside))
    max_M = max(64, int(12.0 * nside * nside * (1.0 - math.cos(search_rad))) + 32)

    n_threads = numba.get_num_threads()
    scratch_pix = np.empty((n_threads, max_M), dtype=np.int64)
    scratch_w = np.empty((n_threads, max_M), dtype=np.float64)
    tod_tmp = np.zeros((mp_stacked.shape[0], B * Sc), dtype=np.float64)

    _gaussian_accum_flatsky_jit(
        dtheta_tile,
        dphi_tile,
        k_b,
        theta_b,
        phi_b,
        B,
        Sc,
        nside,
        mp_stacked,
        beam_vals,
        tod_arr,
        sigma_rad,
        radius_rad,
        scratch_pix,
        scratch_w,
        tod_tmp,
    )
