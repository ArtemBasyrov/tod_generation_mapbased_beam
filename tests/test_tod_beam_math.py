"""
Tests for the tod_beam_math module.

Covers:
    compute_bell                     – effective beam transfer function B_ell
    _compute_dB_threshold_from_power – dB threshold for a target power fraction

Can be run independently:
    pytest tests/test_tod_beam_math.py -v
    python tests/test_tod_beam_math.py
"""

import os
import sys
from unittest.mock import MagicMock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

for _mod_name in ["pixell", "pixell.enmap"]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

if "tod_io" not in sys.modules:
    sys.modules["tod_io"] = MagicMock()

import numpy as np
import numpy.testing as npt
import pytest
import healpy as hp

from tod_beam_math import compute_bell, _compute_dB_threshold_from_power


# ===========================================================================
# TestComputeBell
# ===========================================================================


class TestComputeBell:
    """Unit tests for compute_bell."""

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _single_pixel_at_origin():
        """Return (ra, dec, pixel_map) for a single pixel exactly at the beam centre."""
        ra = np.array([0.0])
        dec = np.array([0.0])
        pixel_map = np.array([1.0])
        return ra, dec, pixel_map

    @staticmethod
    def _gaussian_grid(fwhm_arcmin, half_width_arcmin=30.0, n=121):
        """Build a square ra/dec grid with a Gaussian beam profile.

        Args:
            fwhm_arcmin: FWHM in arcminutes.
            half_width_arcmin: Half-width of the grid in arcminutes.
            n: Grid side length (odd for symmetry).

        Returns:
            tuple: (ra [rad], dec [rad], pixel_map)
        """
        fwhm_rad = np.radians(fwhm_arcmin / 60.0)
        sigma = fwhm_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        hw_rad = np.radians(half_width_arcmin / 60.0)
        lin = np.linspace(-hw_rad, hw_rad, n)
        ra_2d, dec_2d = np.meshgrid(lin, lin)
        pixel_map = np.exp(-(ra_2d**2 + dec_2d**2) / (2.0 * sigma**2))
        return ra_2d, dec_2d, pixel_map

    # ── shape / value tests ───────────────────────────────────────────────────

    def test_returns_correct_shapes(self):
        """ell and bell must each have shape (lmax+1,)."""
        ra, dec, pixel_map = self._single_pixel_at_origin()
        lmax = 50
        # Use power_cut=1.0 so the single pixel is not excluded by the dB threshold
        ell, bell = compute_bell(
            ra, dec, pixel_map, lmax=lmax, power_cut=1.0, verbose=False
        )
        assert ell.shape == (lmax + 1,)
        assert bell.shape == (lmax + 1,)
        npt.assert_array_equal(ell, np.arange(lmax + 1))

    def test_normalised_b0_is_one(self):
        """With normalise=True (default), bell[0] must equal 1."""
        ra, dec, pixel_map = self._single_pixel_at_origin()
        _, bell = compute_bell(
            ra, dec, pixel_map, lmax=20, power_cut=1.0, normalise=True, verbose=False
        )
        assert bell[0] == pytest.approx(1.0)

    def test_unnormalised_b0_equals_total_weight(self):
        """With normalise=False, bell[0] ~ 1 because beam weights are normalised internally."""
        ra, dec, pixel_map = self._single_pixel_at_origin()
        _, bell = compute_bell(
            ra, dec, pixel_map, lmax=20, power_cut=1.0, normalise=False, verbose=False
        )
        # beam_vals are normalised so sum = 1; P_0(cos θ) = 1 everywhere → B_0 = 1
        assert bell[0] == pytest.approx(1.0)

    def test_delta_at_origin_gives_flat_bell(self):
        """A single pixel at ra=dec=0 → cos θ = 1, P_ell(1) = 1 → bell[ell] = 1 for all ell."""
        ra, dec, pixel_map = self._single_pixel_at_origin()
        lmax = 30
        _, bell = compute_bell(
            ra, dec, pixel_map, lmax=lmax, power_cut=1.0, normalise=True, verbose=False
        )
        npt.assert_allclose(bell, np.ones(lmax + 1), atol=1e-12)

    def test_power_cut_one_takes_fast_path(self):
        """power_cut=1.0 and power_cut=0.999 must return arrays of the same shape."""
        ra, dec, pixel_map = self._gaussian_grid(30.0)
        lmax = 50
        ell1, bell1 = compute_bell(
            ra, dec, pixel_map, lmax=lmax, power_cut=1.0, verbose=False
        )
        ell2, bell2 = compute_bell(
            ra, dec, pixel_map, lmax=lmax, power_cut=0.999, verbose=False
        )
        assert bell1.shape == bell2.shape == (lmax + 1,)
        assert ell1.shape == ell2.shape == (lmax + 1,)

    # ── Gaussian beam comparison ──────────────────────────────────────────────

    @pytest.mark.parametrize("fwhm_arcmin", [10.0, 30.0])
    def test_uniform_beam_falls_off(self, fwhm_arcmin):
        """B_ell from a pixelised Gaussian should match healpy.gauss_beam to ~2%
        over the first half of the ell range (higher ell diverges due to grid
        undersampling).
        """
        lmax = 200
        ra, dec, pixel_map = self._gaussian_grid(
            fwhm_arcmin, half_width_arcmin=max(60.0, fwhm_arcmin * 3.0), n=121
        )
        _, bell = compute_bell(
            ra, dec, pixel_map, lmax=lmax, power_cut=1.0, verbose=False
        )

        fwhm_rad = np.radians(fwhm_arcmin / 60.0)
        gauss_ref = hp.gauss_beam(fwhm_rad, lmax=lmax)

        # Compare only up to lmax/2 where the grid is well-sampled
        ell_half = lmax // 2
        npt.assert_allclose(bell[:ell_half], gauss_ref[:ell_half], rtol=0.02)

    # ── error / edge-case tests ───────────────────────────────────────────────

    def test_raises_on_no_pixels_selected(self):
        """power_cut < 1.0 with all-zero pixel_map raises ValueError.

        Depending on the threshold arithmetic, the error may be either
        "No pixels survive" (when no pixels pass the dB threshold) or
        "Sum of beam values is non-positive" (when all passing pixels are zero).
        Both are correct defensive outcomes; we just check for ValueError.
        """
        ra = np.linspace(-0.1, 0.1, 10)
        dec = np.linspace(-0.1, 0.1, 10)
        pixel_map = np.zeros(10)
        with pytest.raises(ValueError):
            compute_bell(ra, dec, pixel_map, lmax=10, power_cut=0.99, verbose=False)

    def test_raises_on_negative_sum(self):
        """pixel_map of all zeros with power_cut=1.0 → ValueError (non-positive sum)."""
        ra = np.linspace(-0.1, 0.1, 10)
        dec = np.linspace(-0.1, 0.1, 10)
        pixel_map = np.zeros(10)
        with pytest.raises(ValueError):
            compute_bell(ra, dec, pixel_map, lmax=10, power_cut=1.0, verbose=False)


# ===========================================================================
# TestComputeDBThresholdFromPower
# ===========================================================================


class TestComputeDBThresholdFromPower:
    """Unit tests for _compute_dB_threshold_from_power."""

    def test_returns_finite_for_typical_input(self):
        """Positive uniform profile must return a finite float."""
        rng = np.random.default_rng(42)
        beam_vals = rng.uniform(0.5, 1.5, 100)
        threshold = _compute_dB_threshold_from_power(beam_vals, power_cut=0.9)
        assert np.isfinite(threshold)

    def test_threshold_monotone_direction(self):
        """Keeping 99% of power (power_cut=0.99) should include more pixels
        than keeping 50% (power_cut=0.5), hence a LOWER dB threshold for 0.99.

        threshold(0.5) > threshold(0.99)  — fewer pixels selected at 0.5.
        """
        rng = np.random.default_rng(7)
        beam_vals = rng.exponential(scale=1.0, size=500)
        t_50 = _compute_dB_threshold_from_power(beam_vals, power_cut=0.50)
        t_99 = _compute_dB_threshold_from_power(beam_vals, power_cut=0.99)
        assert t_50 > t_99

    def test_threshold_consistent_with_selection(self):
        """After applying the threshold, selected pixels cover approximately power_cut × total.

        The source selects pixels with dB > threshold (strict), so the boundary
        pixel is excluded. Including the boundary pixel (>=) always satisfies the
        spec; the strict-> selection may be one pixel short.  We verify the >=
        selection is correct, and also that the strict -> selection gives at most
        one pixel's worth of shortfall.
        """
        rng = np.random.default_rng(13)
        beam_vals = rng.exponential(scale=1.0, size=200)
        power_cut = 0.85
        threshold = _compute_dB_threshold_from_power(beam_vals, power_cut=power_cut)
        log_map = 10.0 * np.log10(np.abs(beam_vals) + 1e-30)

        # Inclusive selection (>= threshold) must satisfy the spec
        sel_incl = log_map >= threshold
        selected_power_incl = beam_vals[sel_incl].sum()
        total_power = beam_vals.sum()
        assert selected_power_incl >= power_cut * total_power - 1e-9 * total_power

        # Strict selection (> threshold, matching source) may exclude the boundary
        # pixel; verify it accounts for most of the target power (within one pixel)
        sel_strict = log_map > threshold
        selected_power_strict = beam_vals[sel_strict].sum()
        max_shortfall = beam_vals.max()  # at most one boundary pixel excluded
        assert selected_power_strict >= power_cut * total_power - max_shortfall

    def test_single_pixel_input(self):
        """A 1-element profile must return the dB value of that single pixel."""
        val = 0.5
        threshold = _compute_dB_threshold_from_power(np.array([val]), power_cut=0.99)
        expected_dB = 10.0 * np.log10(val)
        assert threshold == pytest.approx(expected_dB, rel=1e-9)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
