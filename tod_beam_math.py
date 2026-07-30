"""
Beam harmonic / power utilities.

compute_bell                       — effective beam transfer function B_ell
                                     from a pixelised beam.
_compute_dB_threshold_from_power   — dB threshold retaining a target fraction
                                     of total beam power; used for hard pixel
                                     selection upstream of the gather kernels.
"""

import numpy as np


def compute_bell(
    ra, dec, pixel_map, lmax=1024, power_cut=1.00, normalise=True, verbose=True
):
    """Compute the effective beam transfer function B_ell from a pixelised beam.

    Evaluates

        B_ell = sum_i [ w_i * P_ell(cos θ_i) ]

    where ``w_i`` are the normalised beam weights ``B_i cos(dec_i)`` — the
    ``cos(dec)`` factor is the solid-angle Jacobian of the grid cell, without
    which the sum evaluates ``int B / cos(dec) dOmega`` — ``θ_i`` is the angular
    distance of pixel *i* from the beam centre, and ``P_ell`` are Legendre
    polynomials computed via the three-term recurrence (O(N) memory).

    The beam centre is taken to be the point with zero offset, i.e.
    ``ra_offset = 0, dec_offset = 0``, and the angular distance to each
    pixel is

        cos θ_i = cos(dec_i) * cos(ra_i)

    which is the standard haversine approximation for small-angle offsets.

    Args:
        ra (array-like): RA offsets from beam centre [rad].  Can be any
            shape; flattened internally.
        dec (array-like): Dec offsets from beam centre [rad].  Same shape
            as ``ra``.
        pixel_map (array-like): Beam power values (linear, **not** dB).
            Same shape as ``ra`` and ``dec``.
        lmax (int): Maximum multipole ℓ (inclusive).
        power_cut (float): Fraction of total power for hard pixel selection.
            ``1.0`` selects all pixels (fast path, no sorting needed).
        normalise (bool): If ``True`` (default) normalise so that B_0 = 1.
        verbose (bool): Print selection and pixel-count diagnostics.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - **ell**  – integer array ``[0, 1, …, lmax]``
            - **bell** – float64 array of B_ell values, length ``lmax + 1``
    """
    pixel_map = np.asarray(pixel_map, dtype=np.float64)
    ra = np.asarray(ra, dtype=np.float64).ravel()
    dec = np.asarray(dec, dtype=np.float64).ravel()
    # Weights are a quadrature rule for int B dOmega, so they carry the
    # cos(dec) cell Jacobian; omitting it tilts B_ell by the beam's own
    # second moment along Dec.
    flat = pixel_map.ravel() * np.cos(dec)

    # ── 1. Pixel selection ────────────────────────────────────────────────────
    if power_cut >= 1.0:
        # Fast path: include all pixels, skip the O(N log N) threshold sort.
        sel = np.ones(flat.shape, dtype=bool)
        if verbose:
            print(f"  power_cut=1.0: selecting all {len(flat)} pixels")
    else:
        dB_cut = _compute_dB_threshold_from_power(flat, power_cut)
        log_map = 10.0 * np.log10(np.abs(flat) + 1e-30)
        sel = log_map >= dB_cut
        if verbose:
            print(
                f"  power_cut={power_cut}: "
                f"{np.sum(sel)}/{len(flat)} pixels selected "
                f"(dB_cut={dB_cut:.2f})"
            )

    if not np.any(sel):
        raise ValueError(
            "No pixels survive the power-cut selection. "
            "Check that pixel_map is in linear (not dB) units."
        )

    beam_vals = flat[sel]
    ra_sel = ra[sel]
    dec_sel = dec[sel]

    # ── 2. Normalise ──────────────────────────────────────────────────────────
    norm = beam_vals.sum()
    if norm <= 0:
        raise ValueError("Sum of beam values is non-positive after selection.")
    beam_vals = beam_vals / norm

    # ── 3. Angular distance cos θ from beam centre ────────────────────────────
    cos_theta = np.cos(dec_sel) * np.cos(ra_sel)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # ── 4. Legendre recurrence ────────────────────────────────────────────────
    N = len(cos_theta)
    bell = np.empty(lmax + 1, dtype=np.float64)

    P_prev2 = np.ones(N, dtype=np.float64)  # P_0 = 1
    P_prev1 = cos_theta.copy()  # P_1 = x

    bell[0] = np.dot(beam_vals, P_prev2)
    if lmax >= 1:
        bell[1] = np.dot(beam_vals, P_prev1)

    for ell_idx in range(1, lmax):
        l = float(ell_idx)
        P_curr = ((2.0 * l + 1.0) * cos_theta * P_prev1 - l * P_prev2) / (l + 1.0)
        bell[ell_idx + 1] = np.dot(beam_vals, P_curr)
        P_prev2 = P_prev1
        P_prev1 = P_curr

    # ── 5. Normalise B_0 = 1 ─────────────────────────────────────────────────
    if normalise and bell[0] != 0.0:
        bell /= bell[0]

    ell = np.arange(lmax + 1, dtype=np.int64)
    return ell, np.abs(bell)


def _compute_dB_threshold_from_power(beam_vals, power_cut):
    """Compute the dB threshold that retains a given fraction of total beam power.

    Returns the dB level of the faintest pixel that must be kept, so callers
    select inclusively with ``10 log10(|val| + 1e-30) >= threshold``. Pixels are
    ranked by ``|val|``: a lobe contributes its magnitude to the power budget
    regardless of sign, so beams carrying negative sidelobes are handled.

    Args:
        beam_vals (numpy.ndarray): Beam pixel amplitude values (linear, not dB).
            May be positive or negative. Any shape; flattened internally.
        power_cut (float): Fraction of total power to retain, e.g. ``0.99`` to
            keep 99 % of the beam power. ``>= 1.0`` retains every pixel.

    Returns:
        float: dB threshold, or ``-inf`` when ``power_cut >= 1.0``. Pixels at or
            above this level collectively contribute ``>= power_cut × total``.
    """
    # Retaining all the power means retaining every pixel, the faintest
    # included; -inf makes the callers' inclusive test unconditionally true.
    if power_cut >= 1.0:
        return -np.inf

    prof = np.abs(np.asarray(beam_vals, dtype=np.float64).ravel())
    # Same epsilon as the callers' selection expression, so the comparison is
    # exact at the boundary pixel instead of off by a rounding step.
    prof_dB = 10.0 * np.log10(prof + 1e-30)

    order = np.argsort(prof_dB)
    sorted_dB = prof_dB[order]
    # tail_power[i] = power carried by ranks i..N-1, i.e. every pixel at or
    # above sorted_dB[i]. Non-increasing in i.
    tail_power = np.cumsum(prof[order][::-1])[::-1]

    target_power = prof.sum() * power_cut
    # Faintest rank whose tail still covers the target. Negating makes the
    # array ascending so searchsorted can locate it in one pass.
    idx = int(np.searchsorted(-tail_power, -target_power, side="right")) - 1
    idx = min(max(idx, 0), len(sorted_dB) - 1)

    return sorted_dB[idx]
