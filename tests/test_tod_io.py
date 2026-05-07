"""
Tests for the tod_io module.

Covers:
- load_beam              : pixell.enmap-based beam loader (read_map is mocked)
- load_scan_information  : discovers nb_of_days and fsamp from psi_*.npy files
- open_scan_day          : returns three numpy memmaps for one day
- load_scan_data_batch   : returns a contiguous float32 slice for one day

The project's conftest stubs out `tod_io` itself (and `pixell`) at sys.modules
level so other test modules can import `tod_calibrate` etc. without pixell.
This test file removes the `tod_io` stub before importing the real module so
that the actual implementation runs.

Can be run independently:
    pytest tests/test_tod_io.py -v
    python tests/test_tod_io.py
"""

import os
import sys
from unittest.mock import MagicMock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Drop the conftest stub so we get the real tod_io.
sys.modules.pop("tod_io", None)

# Make sure pixell is mocked (conftest does this; preserve for standalone runs).
for _mod_name in ["pixell", "pixell.enmap"]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

import numpy as np
import numpy.testing as npt
import pytest

import tod_io


# ===========================================================================
# TestLoadBeam
# ===========================================================================


class _FakeBeamMap:
    """Mimics the small subset of pixell.enmap.ndmap used by load_beam.

    load_beam calls beam_map.posmap() and beam_map[0] (the first slice along
    the leading axis). Everything else is irrelevant to the function under
    test.
    """

    def __init__(self, ra, dec, amp):
        self._ra = ra
        self._dec = dec
        self._amp = amp

    def posmap(self):
        return self._ra, self._dec

    def __getitem__(self, idx):
        if idx == 0:
            return self._amp
        raise IndexError(idx)


class TestLoadBeam:
    def _make_grid(self, n=5):
        """Build a small (n, n) RA/Dec grid in radians plus a deterministic amp map."""
        x = np.linspace(-0.01, 0.01, n)  # 1°-ish strip
        ra, dec = np.meshgrid(x, x, indexing="xy")
        amp = (np.arange(n * n).reshape(n, n) + 1.0).astype(np.float64)
        return ra, dec, amp

    def test_returns_offsets_and_pixel_map(self, monkeypatch):
        """Default centering: ra/dec at the centre pixel must be zero."""
        ra, dec, amp = self._make_grid(n=5)
        # Wrap into a 1-channel "ndmap": load_beam reads beam_map[0].
        amp_3d = amp[None, :, :]
        fake_map = _FakeBeamMap(ra.copy(), dec.copy(), amp_3d[0])

        monkeypatch.setattr(tod_io.enmap, "read_map", lambda path: fake_map)

        ra_out, dec_out, pix = tod_io.load_beam("/tmp/", "fake.fits")

        # Default centre is (n//2, n//2) = (2, 2) for n=5.
        assert ra_out.shape == (5, 5)
        assert dec_out.shape == (5, 5)
        assert pix.shape == (5, 5)
        npt.assert_allclose(ra_out[2, 2], 0.0, atol=0.0)
        npt.assert_allclose(dec_out[2, 2], 0.0, atol=0.0)
        # Amplitudes are passed through unchanged.
        npt.assert_array_equal(pix, amp)

    def test_custom_center(self, monkeypatch):
        """When center_x/center_y are provided, that pixel becomes the origin."""
        ra, dec, amp = self._make_grid(n=5)
        fake_map = _FakeBeamMap(ra.copy(), dec.copy(), amp)
        monkeypatch.setattr(tod_io.enmap, "read_map", lambda path: fake_map)

        ra_out, dec_out, _ = tod_io.load_beam(
            "/tmp/", "fake.fits", center_x=1, center_y=1
        )
        npt.assert_allclose(ra_out[1, 1], 0.0, atol=0.0)
        npt.assert_allclose(dec_out[1, 1], 0.0, atol=0.0)

    def test_path_concatenation(self, monkeypatch):
        """folder + filename is passed straight to enmap.read_map."""
        ra, dec, amp = self._make_grid(n=3)
        fake_map = _FakeBeamMap(ra, dec, amp)
        seen_paths = []

        def fake_read(path):
            seen_paths.append(path)
            return fake_map

        monkeypatch.setattr(tod_io.enmap, "read_map", fake_read)
        tod_io.load_beam("/some/dir/", "beam.fits")
        assert seen_paths == ["/some/dir/beam.fits"]


# ===========================================================================
# TestLoadScanInformation
# ===========================================================================


def _write_psi_files(folder, indices_to_lengths):
    """Helper: write psi_<i>.npy files of the given lengths."""
    for idx, length in indices_to_lengths.items():
        np.save(folder / f"psi_{idx}.npy", np.zeros(length, dtype=np.float32))


class TestLoadScanInformation:
    def test_counts_days_from_indices(self, tmp_path):
        """Highest index + 1 = nb_of_days, even with gaps."""
        _write_psi_files(tmp_path, {0: 86_400, 1: 86_400, 3: 86_400})
        nb, fsamp = tod_io.load_scan_information(str(tmp_path))
        assert nb == 4
        assert fsamp == 1.0

    def test_fsamp_from_first_file_length(self, tmp_path):
        """fsamp = length(first file) / 86400."""
        # Single file of 100 Hz × 1 day.
        _write_psi_files(tmp_path, {0: 8_640_000})
        _, fsamp = tod_io.load_scan_information(str(tmp_path))
        assert fsamp == pytest.approx(100.0)

    def test_raises_when_no_psi_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            tod_io.load_scan_information(str(tmp_path))

    def test_raises_on_unparseable_filenames(self, tmp_path):
        np.save(tmp_path / "psi_bogus.npy", np.zeros(10, dtype=np.float32))
        with pytest.raises(ValueError):
            tod_io.load_scan_information(str(tmp_path))


# ===========================================================================
# TestOpenScanDay
# ===========================================================================


def _write_scan_day(folder, day, n=1000, fill_theta=None, fill_phi=None, fill_psi=None):
    """Write theta_<day>.npy / phi_<day>.npy / psi_<day>.npy of length n."""
    if fill_theta is None:
        fill_theta = np.arange(n, dtype=np.float32)
    if fill_phi is None:
        fill_phi = np.arange(n, dtype=np.float32) + 100.0
    if fill_psi is None:
        fill_psi = np.arange(n, dtype=np.float32) + 200.0
    np.save(folder / f"theta_{day}.npy", fill_theta.astype(np.float32))
    np.save(folder / f"phi_{day}.npy", fill_phi.astype(np.float32))
    np.save(folder / f"psi_{day}.npy", fill_psi.astype(np.float32))


class TestOpenScanDay:
    def test_returns_three_memmaps_with_expected_shapes(self, tmp_path):
        _write_scan_day(tmp_path, day=0, n=500)
        folder = str(tmp_path) + os.sep
        theta, phi, psi = tod_io.open_scan_day(folder, 0)
        assert theta.shape == (500,)
        assert phi.shape == (500,)
        assert psi.shape == (500,)
        # Memmaps preserve the on-disk dtype.
        assert theta.dtype == np.float32
        assert phi.dtype == np.float32
        assert psi.dtype == np.float32

    def test_uses_correct_day_index(self, tmp_path):
        """Day 2 must be opened, not day 0."""
        _write_scan_day(
            tmp_path,
            day=0,
            n=10,
            fill_theta=np.ones(10, dtype=np.float32),
            fill_phi=np.ones(10, dtype=np.float32),
            fill_psi=np.ones(10, dtype=np.float32),
        )
        _write_scan_day(
            tmp_path,
            day=2,
            n=10,
            fill_theta=np.full(10, 2.0, dtype=np.float32),
            fill_phi=np.full(10, 2.0, dtype=np.float32),
            fill_psi=np.full(10, 2.0, dtype=np.float32),
        )
        folder = str(tmp_path) + os.sep
        theta, phi, psi = tod_io.open_scan_day(folder, 2)
        npt.assert_array_equal(np.asarray(theta), np.full(10, 2.0, dtype=np.float32))
        npt.assert_array_equal(np.asarray(phi), np.full(10, 2.0, dtype=np.float32))
        npt.assert_array_equal(np.asarray(psi), np.full(10, 2.0, dtype=np.float32))


# ===========================================================================
# TestLoadScanDataBatch
# ===========================================================================


class TestLoadScanDataBatch:
    def test_returns_correct_slice(self, tmp_path):
        n = 1000
        _write_scan_day(
            tmp_path,
            day=0,
            n=n,
            fill_theta=np.arange(n, dtype=np.float32),
            fill_phi=np.arange(n, dtype=np.float32) + 1000,
            fill_psi=np.arange(n, dtype=np.float32) + 2000,
        )
        folder = str(tmp_path) + os.sep

        theta, phi, psi = tod_io.load_scan_data_batch(folder, 0, 100, 200)

        assert theta.shape == (100,)
        assert phi.shape == (100,)
        assert psi.shape == (100,)
        assert theta.dtype == np.float32
        assert phi.dtype == np.float32
        assert psi.dtype == np.float32
        npt.assert_array_equal(theta, np.arange(100, 200, dtype=np.float32))
        npt.assert_array_equal(phi, np.arange(100, 200, dtype=np.float32) + 1000)
        npt.assert_array_equal(psi, np.arange(100, 200, dtype=np.float32) + 2000)

    def test_full_range(self, tmp_path):
        n = 50
        _write_scan_day(tmp_path, day=0, n=n)
        folder = str(tmp_path) + os.sep
        theta, phi, psi = tod_io.load_scan_data_batch(folder, 0, 0, n)
        assert theta.shape == phi.shape == psi.shape == (n,)
        npt.assert_array_equal(theta, np.arange(n, dtype=np.float32))

    def test_empty_slice(self, tmp_path):
        n = 50
        _write_scan_day(tmp_path, day=0, n=n)
        folder = str(tmp_path) + os.sep
        theta, phi, psi = tod_io.load_scan_data_batch(folder, 0, 10, 10)
        assert theta.shape == (0,)
        assert phi.shape == (0,)
        assert psi.shape == (0,)

    def test_routes_to_correct_day(self, tmp_path):
        """day_index determines which set of files is read."""
        _write_scan_day(
            tmp_path,
            day=0,
            n=10,
            fill_theta=np.zeros(10, dtype=np.float32),
            fill_phi=np.zeros(10, dtype=np.float32),
            fill_psi=np.zeros(10, dtype=np.float32),
        )
        _write_scan_day(
            tmp_path,
            day=1,
            n=10,
            fill_theta=np.full(10, 9.0, dtype=np.float32),
            fill_phi=np.full(10, 9.0, dtype=np.float32),
            fill_psi=np.full(10, 9.0, dtype=np.float32),
        )
        folder = str(tmp_path) + os.sep
        theta, phi, psi = tod_io.load_scan_data_batch(folder, 1, 0, 10)
        npt.assert_array_equal(theta, np.full(10, 9.0, dtype=np.float32))
        npt.assert_array_equal(phi, np.full(10, 9.0, dtype=np.float32))
        npt.assert_array_equal(psi, np.full(10, 9.0, dtype=np.float32))


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
