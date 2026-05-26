"""
Integration tests for the map_fields config option.

Exercises every non-empty subset of {0, 1, 2} (T, Q, U) end-to-end through
prepare_beam_data → beam_tod_batch with the production mp_stacked path:

- [0]            T only            — kernel falls into the no-Q/U scalar path
- [1]            Q only            — c_q=0, c_u=-1 → scalar path (no spin-2)
- [2]            U only            — c_q=-1, c_u=0 → scalar path (no spin-2)
- [0, 1]         T + Q             — c_q=1, c_u=-1 → scalar path
- [1, 2]         Q + U             — c_q=0, c_u=1 → full spin-2 path
- [0, 2]         T + U             — c_q=-1, c_u=1 → scalar path
- [0, 1, 2]      T + Q + U         — c_q=1, c_u=2 → full spin-2 path

For each subset we verify:
1. prepare_beam_data drops beam files for inactive components.
2. beam_tod_batch returns one entry per active comp_indices value.
3. The returned arrays have the right shape and contain finite numbers.
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
    """Patch tod_config attributes consumed by prepare_beam_data + tod_core."""
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


def _build_scan(B=8, N=21):
    rng = np.random.default_rng(7)
    ra = np.zeros((N, N))
    dec = np.zeros((N, N))
    phi_batch = rng.uniform(0, 0.04, B).astype(np.float32)
    theta_batch = rng.uniform(np.pi / 2 - 0.04, np.pi / 2, B).astype(np.float32)
    rot_vecs, betas = precompute_rotation_vector_batch(
        ra, dec, phi_batch, theta_batch, center_idx=(N // 2, N // 2)
    )
    psis_b = (-betas).astype(np.float32)
    return phi_batch, theta_batch, psis_b, rot_vecs


_ALL_SUBSETS = [
    (0,),
    (1,),
    (2,),
    (0, 1),
    (1, 2),
    (0, 2),
    (0, 1, 2),
]


@pytest.mark.parametrize("active", _ALL_SUBSETS)
def test_prepare_beam_data_filters_to_active(active, monkeypatch):
    """prepare_beam_data drops beam entries whose comp_indices are entirely inactive."""
    ra, dec, pm = _make_gaussian_beam(n=21)
    _install_fake_load_beam(monkeypatch, ra, dec, pm)
    _patch_config(monkeypatch, active, ["I.fits", "Q.fits", "U.fits"])

    beam_data = pph.prepare_beam_data(["I.fits", "Q.fits", "U.fits"])

    expected_files = {["I.fits", "Q.fits", "U.fits"][c] for c in active}
    assert set(beam_data) == expected_files
    for bf, entry in beam_data.items():
        assert all(c in active for c in entry["comp_indices"])


@pytest.mark.parametrize("active", _ALL_SUBSETS)
def test_beam_tod_batch_runs_for_subset(active, monkeypatch):
    """End-to-end run through the production mp_stacked path for every subset."""
    nside = 32
    B = 8
    npix = hp.nside2npix(nside)

    ra, dec, pm = _make_gaussian_beam(n=21)
    _install_fake_load_beam(monkeypatch, ra, dec, pm)
    _patch_config(monkeypatch, active, ["I.fits", "Q.fits", "U.fits"])

    beam_data = pph.prepare_beam_data(["I.fits", "Q.fits", "U.fits"])

    # Synthetic sky map with one component per active field.
    rng = np.random.default_rng(11)
    MP = {c: rng.standard_normal(npix).astype(np.float32) for c in active}
    for entry in beam_data.values():
        entry["mp_stacked"] = np.ascontiguousarray(
            np.stack([MP[c] for c in entry["comp_indices"]])
        )

    phi_b, theta_b, psis_b, rot_vecs = _build_scan(B=B)

    tod_full = np.zeros((3, B), dtype=np.float32)
    seen_components = set()
    for entry in beam_data.values():
        contrib = beam_tod_batch(nside, MP, entry, rot_vecs, phi_b, theta_b, psis_b)
        for comp, vals in contrib.items():
            assert comp in active, f"got comp {comp} but active={active}"
            assert vals.shape == (B,)
            assert np.all(np.isfinite(vals))
            tod_full[comp] += vals
            seen_components.add(comp)

    assert seen_components == set(active)
    # Inactive rows of the (3, B) output must remain exactly zero.
    for c in (0, 1, 2):
        if c not in active:
            assert np.all(tod_full[c] == 0.0)


def test_t_only_skips_spin2_and_zeros_q_u(monkeypatch):
    """With map_fields=[0] the Q and U rows of the output are exact zeros."""
    nside = 32
    B = 6
    npix = hp.nside2npix(nside)

    ra, dec, pm = _make_gaussian_beam(n=21)
    _install_fake_load_beam(monkeypatch, ra, dec, pm)
    _patch_config(monkeypatch, (0,), ["I.fits", None, None])

    beam_data = pph.prepare_beam_data(["I.fits", None, None])
    assert set(beam_data) == {"I.fits"}
    assert beam_data["I.fits"]["comp_indices"] == [0]

    rng = np.random.default_rng(3)
    MP = {0: rng.standard_normal(npix).astype(np.float32)}
    beam_data["I.fits"]["mp_stacked"] = np.ascontiguousarray(MP[0][np.newaxis, :])

    phi_b, theta_b, psis_b, rot_vecs = _build_scan(B=B)
    contrib = beam_tod_batch(
        nside, MP, beam_data["I.fits"], rot_vecs, phi_b, theta_b, psis_b
    )
    assert set(contrib) == {0}
    assert contrib[0].shape == (B,)
    assert np.all(np.isfinite(contrib[0]))


def test_prepare_beam_data_rejects_none_for_active_index(monkeypatch):
    """Active component with a None beam filename raises a clear error."""
    ra, dec, pm = _make_gaussian_beam(n=21)
    _install_fake_load_beam(monkeypatch, ra, dec, pm)
    _patch_config(monkeypatch, (0, 1), ["I.fits", None, None])

    with pytest.raises(ValueError, match="beam_filenames"):
        pph.prepare_beam_data(["I.fits", None, None])
