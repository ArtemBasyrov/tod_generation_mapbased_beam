"""
Tests for tod_runcontext: the frozen RunContext value object and the
build_run_context() factory that maps tod_config into it.
"""

import dataclasses

import numpy as np
import pytest

import tod_config as config
from tod_runcontext import RunContext, build_run_context


def test_runcontext_is_frozen():
    ctx = RunContext(
        folder_scan="/scan/",
        folder_tod_output="/out/",
        interp_mode="bilinear",
        nside=1024,
        batch_size=8192,
        z_skip_threshold=-1.0,
        fsamp=19.0,
        precision_dtype=np.float32,
        beam_center_idx=None,
        hwp_enabled=False,
        hwp_freq_hz=0.0,
        hwp_phi0_rad=0.0,
        detectors=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.nside = 2048


def test_build_run_context_maps_config(monkeypatch):
    """build_run_context() copies the run-scoped config values verbatim."""
    monkeypatch.setattr(config, "FOLDER_SCAN", "/scan/")
    monkeypatch.setattr(config, "FOLDER_TOD_OUTPUT", "/out/")
    monkeypatch.setattr(config, "beam_interp_method", "nearest")
    monkeypatch.setattr(config, "precision_dtype", np.float64)
    monkeypatch.setattr(config, "beam_center_x", None)
    monkeypatch.setattr(config, "beam_center_y", None)
    monkeypatch.setattr(config, "hwp_enabled", True)
    monkeypatch.setattr(config, "hwp_rotation_frequency_hz", 1.5)
    monkeypatch.setattr(config, "hwp_initial_phase_rad", 0.25)

    ctx = build_run_context(
        nside=512, batch_size=4096, z_skip_threshold=0.7, fsamp=19.0
    )

    assert ctx.folder_scan == "/scan/"
    assert ctx.folder_tod_output == "/out/"
    assert ctx.interp_mode == "nearest"
    assert ctx.nside == 512
    assert ctx.batch_size == 4096
    assert ctx.z_skip_threshold == 0.7
    assert ctx.fsamp == 19.0
    assert ctx.precision_dtype is np.float64
    assert ctx.beam_center_idx is None
    assert ctx.hwp_enabled is True
    assert ctx.hwp_freq_hz == 1.5
    assert ctx.hwp_phi0_rad == 0.25
    # With no detectors: section, the single implicit boresight detector is used.
    monkeypatch.setattr(config, "detectors", None)
    monkeypatch.setattr(config, "detector_subset", None)
    ctx = build_run_context(
        nside=512, batch_size=4096, z_skip_threshold=0.7, fsamp=19.0
    )
    assert len(ctx.detectors) == 1
    assert ctx.detectors[0].is_boresight


def test_build_run_context_beam_center_idx(monkeypatch):
    """beam_center_idx is a (row, col) tuple only when both overrides are set."""
    monkeypatch.setattr(config, "beam_center_x", 7)
    monkeypatch.setattr(config, "beam_center_y", 9)
    ctx = build_run_context(nside=8, batch_size=1, z_skip_threshold=-1.0, fsamp=1.0)
    assert ctx.beam_center_idx == (7, 9)

    # Only one set → fall back to None (array centre is used downstream).
    monkeypatch.setattr(config, "beam_center_x", 7)
    monkeypatch.setattr(config, "beam_center_y", None)
    ctx = build_run_context(nside=8, batch_size=1, z_skip_threshold=-1.0, fsamp=1.0)
    assert ctx.beam_center_idx is None
