"""
Zero-map null test (sanity check for additive contamination).

Every error mechanism of the pipeline is multiplicative in the sky signal:
the interpolation error field is eps = interp(m) - f with interp linear, the
coherent channel is (T~(l) - 1) * C_l, and the aliasing amplitude scales with
the sky's gradient/curvature power (see interp_characterization/
ERROR_MODEL_SUMMARY.md). An all-zero input map must therefore produce
bit-exact zero TOD — 0.0 in every sample of every Stokes component, not just
small values. Any nonzero output signals an additive defect (uninitialised
buffer, sentinel/UNSEEN leakage, out-of-bounds gather, wired-in offset),
never interpolation error.

Covered here, on the production ``mp_stacked`` path:

- both main-branch interpolation methods (``bilinear``, ``nearest``);
- the full spin-2 Q/U path ([0, 1, 2]) and the T-only scalar path ([0]);
- pointings from the pole to the equator (the spin-2 frame correction is
  largest near the poles), with the correction both always-on and skipped
  via the equatorial-band threshold;
- the harmonic reference leg: ``hp.smoothing`` of a zero map is exactly
  zero, so the residual map_test - map_true of the null test is zero too.
"""

import os
import sys
import math
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
import healpy as hp
import pytest

import tod_pipeline_helpers as pph
from tod_core import beam_tod_batch, precompute_rotation_vector_batch


def _make_gaussian_beam(n=21, half_width_arcmin=15.0, fwhm_arcmin=30.0):
    half_width_rad = math.radians(half_width_arcmin / 60.0)
    x = np.linspace(-half_width_rad, half_width_rad, n)
    ra, dec = np.meshgrid(x, x, indexing="xy")
    sigma = math.radians((fwhm_arcmin / 60.0) / (2 * math.sqrt(2 * math.log(2))))
    pixel_map = np.exp(-(ra**2 + dec**2) / (2 * sigma**2))
    return ra, dec, pixel_map


def _patch_config(monkeypatch, map_fields, beam_files):
    monkeypatch.setattr(pph.config, "FOLDER_BEAM", "/tmp/", raising=False)
    monkeypatch.setattr(pph.config, "beam_file_I", beam_files[0], raising=False)
    monkeypatch.setattr(pph.config, "beam_file_Q", beam_files[1], raising=False)
    monkeypatch.setattr(pph.config, "beam_file_U", beam_files[2], raising=False)
    monkeypatch.setattr(pph.config, "power_threshold_I", 1.0, raising=False)
    monkeypatch.setattr(pph.config, "power_threshold_Q", 1.0, raising=False)
    monkeypatch.setattr(pph.config, "power_threshold_U", 1.0, raising=False)
    monkeypatch.setattr(pph.config, "beam_center_x", None, raising=False)
    monkeypatch.setattr(pph.config, "beam_center_y", None, raising=False)
    monkeypatch.setattr(pph.config, "map_fields", tuple(map_fields), raising=False)


def _install_fake_load_beam(monkeypatch, ra, dec, pixel_map):
    def fake_load_beam(folder, fname, center_x=None, center_y=None):
        return ra.copy(), dec.copy(), pixel_map.copy()

    monkeypatch.setattr(pph, "load_beam", fake_load_beam)


def _build_scan(ra, dec, N, B=24, seed=5):
    """Pointings spanning pole to equator, where spin-2 behaviour differs most."""
    rng = np.random.default_rng(seed)
    theta_batch = np.linspace(0.005, np.pi / 2, B).astype(np.float32)
    phi_batch = rng.uniform(0, 2 * np.pi, B).astype(np.float32)
    rot_vecs, betas = precompute_rotation_vector_batch(
        ra, dec, phi_batch, theta_batch, center_idx=(N // 2, N // 2)
    )
    psis_b = (-betas).astype(np.float32)
    return phi_batch, theta_batch, psis_b, rot_vecs


@pytest.mark.parametrize("interp_mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("active", [(0, 1, 2), (0,)])
@pytest.mark.parametrize("z_skip_threshold", [-1.0, 0.5])
def test_zero_map_gives_bitexact_zero_tod(
    interp_mode, active, z_skip_threshold, monkeypatch
):
    """An all-zero sky map must produce exactly 0.0 in every TOD sample."""
    nside = 64
    N = 21
    npix = hp.nside2npix(nside)

    ra, dec, pm = _make_gaussian_beam(n=N)
    _install_fake_load_beam(monkeypatch, ra, dec, pm)
    beam_files = [
        "I.fits" if 0 in active else None,
        "Q.fits" if 1 in active else None,
        "U.fits" if 2 in active else None,
    ]
    _patch_config(monkeypatch, active, beam_files)

    beam_data = pph.prepare_beam_data(beam_files)

    MP = {c: np.zeros(npix, dtype=np.float32) for c in active}
    for entry in beam_data.values():
        entry["mp_stacked"] = np.ascontiguousarray(
            np.stack([MP[c] for c in entry["comp_indices"]])
        )

    phi_b, theta_b, psis_b, rot_vecs = _build_scan(ra, dec, N)

    for entry in beam_data.values():
        contrib = beam_tod_batch(
            nside,
            MP,
            entry,
            rot_vecs,
            phi_b,
            theta_b,
            psis_b,
            interp_mode=interp_mode,
            z_skip_threshold=z_skip_threshold,
        )
        for comp, vals in contrib.items():
            nonzero = np.count_nonzero(vals)
            assert nonzero == 0, (
                f"{interp_mode}, fields={active}, comp {comp}: "
                f"{nonzero}/{vals.size} nonzero TOD samples from a zero map "
                f"(max |tod| = {np.max(np.abs(vals)):.3e}) — additive "
                "contamination in the gather/accumulate path"
            )


def test_zero_map_harmonic_reference_is_zero():
    """The reference leg: harmonic smoothing of a zero map is exactly zero."""
    nside = 64
    zero_iqu = np.zeros((3, hp.nside2npix(nside)))
    smoothed = hp.smoothing(zero_iqu, fwhm=math.radians(0.5), pol=True)
    assert np.count_nonzero(smoothed) == 0
