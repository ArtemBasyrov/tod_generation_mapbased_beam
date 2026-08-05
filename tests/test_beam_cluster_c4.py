"""Tests for the C4 (quadrant-replication) clustering path.

The headline case is a null test.  A beam that is a function of angular
distance alone has no ``m = ±2`` harmonic and leaks no temperature into
polarisation, so any ellipticity measured after clustering was manufactured by
the pipeline.  Plain k-means manufactures a large one; the C4 path must not.

Can be run independently::

    pytest tests/test_beam_cluster_c4.py -v
"""

import os
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from beam_cluster import (  # noqa: E402
    c4_asymmetry,
    c4_orbits,
    cluster_beam_pixels,
)

N_GRID = 61  # odd, so the centre pixel exists
A_ARCMIN = 1.0
R2A = 180.0 * 60.0 / np.pi


def _grid_offsets(n=N_GRID):
    """Integer (di, dj) offsets from the centre pixel of an n x n grid."""
    ii, jj = np.divmod(np.arange(n * n), n)
    return ii - n // 2, jj - n // 2


def _beam(sigma_x_arcmin, sigma_y_arcmin, n=N_GRID):
    """Elliptical Gaussian on a CAR grid, in the pipeline's conventions.

    Returns (vec, weights, di, dj) with weights carrying the cos(dec) cell
    Jacobian and normalised to sum to one, matching ``prepare_beam_data``.
    """
    ax = (np.arange(n) - n // 2) * A_ARCMIN / R2A
    ra, dec = np.meshgrid(ax, ax, indexing="xy")
    rr = (ra * R2A / sigma_x_arcmin) ** 2 + (dec * R2A / sigma_y_arcmin) ** 2
    w = (np.exp(-0.5 * rr) * np.cos(dec)).ravel()
    w = w / w.sum()
    th = np.pi / 2 - dec.ravel()
    vec = np.stack(
        [np.sin(th) * np.cos(ra.ravel()), np.sin(th) * np.sin(ra.ravel()), np.cos(th)],
        axis=-1,
    )
    di, dj = _grid_offsets(n)
    return vec, w, di, dj


def _added_ellipticity(vec, w, vec_out, w_out):
    """|traceless part of the second-moment deficit|, in the beam frame.

    Clustering's whole leading error is ``Delta M = -Sigma_bar``; the part of
    that deficit which is not proportional to the beam's own second moment is
    ellipticity the clustering added.  For a round beam ``M`` is isotropic, so
    that reduces to the traceless part of the deficit itself.
    """

    def moment(v, wt):
        c = (wt[:, None] * v).sum(axis=0)
        c /= np.linalg.norm(c)
        e1 = np.cross(c, [0.0, 0.0, 1.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(c, e1)
        t1, t2 = v @ e1, v @ e2
        return (
            np.array(
                [
                    [(wt * t1 * t1).sum(), (wt * t1 * t2).sum()],
                    [(wt * t1 * t2).sum(), (wt * t2 * t2).sum()],
                ]
            )
            / wt.sum()
        )

    d = moment(vec_out, w_out) - moment(vec, w)
    ev = np.linalg.eigvalsh(d - np.trace(d) / 2.0 * np.eye(2))
    return float(ev[1] - ev[0])


class TestC4Orbits:
    def test_generic_orbits_have_four_members(self):
        di, dj = _grid_offsets()
        orbit, rot = c4_orbits(di, dj)
        counts = np.bincount(orbit)
        # exactly one fixed point (the centre); everything else in fours
        assert (counts == 1).sum() == 1
        assert set(np.unique(counts)) <= {1, 4}

    def test_rotation_indices_are_a_permutation(self):
        di, dj = _grid_offsets()
        orbit, rot = c4_orbits(di, dj)
        for o in np.unique(orbit):
            m = orbit == o
            if m.sum() == 4:
                assert sorted(rot[m]) == [0, 1, 2, 3]

    def test_orbit_members_are_true_90_degree_rotations(self):
        di, dj = _grid_offsets()
        orbit, _ = c4_orbits(di, dj)
        pos = {(int(a), int(b)) for a, b in zip(di, dj)}
        for o in np.unique(orbit)[:50]:
            m = np.flatnonzero(orbit == o)
            if len(m) != 4:
                continue
            a, b = int(di[m[0]]), int(dj[m[0]])
            assert {(a, b), (-b, a), (-a, -b), (b, -a)} <= pos

    def test_asymmetry_separates_round_from_elliptical(self):
        di, dj = _grid_offsets()
        orbit, rot = c4_orbits(di, dj)
        _, w_round, _, _ = _beam(8.0, 8.0)
        _, w_ellip, _, _ = _beam(8.0, 4.0)
        assert c4_asymmetry(w_round, orbit, rot) < 1e-4
        assert c4_asymmetry(w_ellip, orbit, rot) > 1e-2


class TestC4Clustering:
    """The C4 path must preserve every identity plain clustering does."""

    def _run(self, sx, sy, K=48, f=0.3):
        vec, w, di, dj = _beam(sx, sy)
        c4 = c4_orbits(di, dj)
        v_c4, w_c4, lab_c4 = cluster_beam_pixels(
            vec, w, n_clusters=K, tail_fraction=f, verbose=False, c4=c4
        )
        v_km, w_km, lab_km = cluster_beam_pixels(
            vec, w, n_clusters=K, tail_fraction=f, verbose=False
        )
        return vec, w, (v_c4, w_c4, lab_c4), (v_km, w_km, lab_km)

    def test_mass_is_preserved(self):
        _, w, (_, w_c4, _), _ = self._run(8.0, 8.0)
        assert w_c4.sum() == pytest.approx(w.sum(), rel=1e-12)

    def test_labels_cover_every_input_pixel(self):
        vec, w, (v_c4, w_c4, lab), _ = self._run(8.0, 8.0)
        assert lab.shape == (len(w),)
        assert lab.min() >= 0
        assert lab.max() == len(w_c4) - 1
        assert len(np.unique(lab)) == len(w_c4)

    def test_cluster_masses_match_their_labels(self):
        vec, w, (v_c4, w_c4, lab), _ = self._run(8.0, 8.0)
        summed = np.bincount(lab, weights=w, minlength=len(w_c4))
        np.testing.assert_allclose(summed, w_c4, rtol=1e-12, atol=0)

    def test_centroids_are_unit_vectors(self):
        _, _, (v_c4, _, _), _ = self._run(8.0, 8.0)
        np.testing.assert_allclose(np.linalg.norm(v_c4, axis=1), 1.0, atol=1e-12)

    def test_reduces_the_node_count(self):
        vec, w, (_, w_c4, _), _ = self._run(8.0, 8.0)
        assert len(w_c4) < len(w)

    def test_node_count_is_competitive_with_plain_kmeans(self):
        """C4 must not buy its accuracy by keeping more nodes."""
        _, _, (_, w_c4, _), (_, w_km, _) = self._run(8.0, 8.0)
        assert len(w_c4) <= 1.15 * len(w_km)

    def test_dtype_is_inherited(self):
        vec, w, di, dj = _beam(8.0, 8.0)
        v, bv, _ = cluster_beam_pixels(
            vec,
            w.astype(np.float32),
            n_clusters=48,
            tail_fraction=0.3,
            verbose=False,
            c4=c4_orbits(di, dj),
        )
        assert bv.dtype == np.float32 and v.dtype == np.float32

    def test_is_deterministic(self):
        vec, w, di, dj = _beam(8.0, 8.0)
        c4 = c4_orbits(di, dj)
        a = cluster_beam_pixels(
            vec, w, n_clusters=48, tail_fraction=0.3, verbose=False, c4=c4
        )
        b = cluster_beam_pixels(
            vec, w, n_clusters=48, tail_fraction=0.3, verbose=False, c4=c4
        )
        np.testing.assert_array_equal(a[2], b[2])
        np.testing.assert_allclose(a[1], b[1], rtol=0, atol=0)

    def test_partition_is_c4_symmetric(self):
        """Every cell's siblings must be its own 90-degree rotations."""
        vec, w, di, dj = _beam(8.0, 8.0)
        orbit, rot = c4_orbits(di, dj)
        _, _, lab = cluster_beam_pixels(
            vec,
            w,
            n_clusters=48,
            tail_fraction=0.3,
            verbose=False,
            c4=(orbit, rot),
        )
        # Nodes sharing an orbit must land in four distinct cells, and the map
        # orbit -> set(labels) must be consistent for every orbit in a cell.
        for o in np.unique(orbit):
            m = orbit == o
            if m.sum() == 4:
                assert len(np.unique(lab[m])) == 4


class TestRaggedNodeSet:
    """An orbit that does not close has no siblings, so it must stay apart.

    ``prepare_beam_data`` drops zero-weight pixels, and it drops them from a
    region that is not 90-degree symmetric, so the node set reaching the
    clusterer can be ragged even on an odd grid.  Merging such a node into a
    cell gives that cell members its three images do not have, which is the
    equivariance the cancellation rests on.
    """

    def _ragged(self, sx=8.0, sy=8.0):
        """A beam with one asymmetric wedge of nodes removed."""
        vec, w, di, dj = _beam(sx, sy)
        keep = ~((di > 15) & (dj > 20))
        return vec[keep], w[keep] / w[keep].sum(), di[keep], dj[keep]

    def test_broken_orbits_are_marked(self):
        _, _, di, dj = self._ragged()
        orbit, rot = c4_orbits(di, dj)
        lone = rot < 0
        assert lone.any()
        counts = np.bincount(orbit)
        assert (counts[orbit[lone]] == 1).all()
        assert (counts[orbit[~lone]] != 3).all()

    def test_unpaired_nodes_get_a_cell_of_their_own(self):
        vec, w, di, dj = self._ragged()
        orbit, rot = c4_orbits(di, dj)
        _, w_out, lab = cluster_beam_pixels(
            vec, w, n_clusters=48, tail_fraction=0.3, verbose=False, c4=(orbit, rot)
        )
        sizes = np.bincount(lab, minlength=len(w_out))
        assert (sizes[lab[rot < 0]] == 1).all()

    def test_closed_orbits_still_land_in_four_cells(self):
        vec, w, di, dj = self._ragged()
        orbit, rot = c4_orbits(di, dj)
        _, _, lab = cluster_beam_pixels(
            vec, w, n_clusters=48, tail_fraction=0.3, verbose=False, c4=(orbit, rot)
        )
        for o in np.unique(orbit[rot > 0]):
            m = orbit == o
            assert len(np.unique(lab[m])) == 4

    def test_mass_is_preserved_on_a_ragged_set(self):
        vec, w, di, dj = self._ragged()
        _, w_out, _ = cluster_beam_pixels(
            vec,
            w,
            n_clusters=48,
            tail_fraction=0.3,
            verbose=False,
            c4=c4_orbits(di, dj),
        )
        assert w_out.sum() == pytest.approx(w.sum(), rel=1e-12)


class TestNullTestSymmetricBeam:
    """A round beam has no ellipticity, so clustering must not add one."""

    def test_c4_beats_plain_kmeans_on_a_round_beam(self):
        vec, w, (v_c4, w_c4, _), (v_km, w_km, _) = TestC4Clustering()._run(8.0, 8.0)
        e_c4 = _added_ellipticity(vec, w, v_c4, w_c4)
        e_km = _added_ellipticity(vec, w, v_km, w_km)
        assert e_c4 < e_km / 100.0, f"C4 {e_c4:.3e} vs k-means {e_km:.3e}"

    def test_added_ellipticity_is_negligible_against_the_beam_width(self):
        """The effective ellipticity handed to a round beam must be tiny.

        Quoted as ``eps = |Sigma_shape| / tr M``, the same measure used for a
        beam's own ellipticity, so it is directly comparable: a real SAT 90 GHz
        beam sits at 1.9e-2. C4 measures ~1e-7 here, five orders below that.
        """
        vec, w, (v_c4, w_c4, _), _ = TestC4Clustering()._run(8.0, 8.0)
        tr_m = 2.0 * (8.0 / R2A) ** 2
        eps = _added_ellipticity(vec, w, v_c4, w_c4) / tr_m
        assert eps < 1e-6, f"effective ellipticity {eps:.3e}"

    def test_null_test_holds_across_cluster_counts(self):
        for K in (16, 48, 120):
            vec, w, (v_c4, w_c4, _), (v_km, w_km, _) = TestC4Clustering()._run(
                8.0, 8.0, K=K
            )
            e_c4 = _added_ellipticity(vec, w, v_c4, w_c4)
            e_km = _added_ellipticity(vec, w, v_km, w_km)
            assert e_c4 < e_km, f"K={K}: C4 {e_c4:.3e} vs k-means {e_km:.3e}"
