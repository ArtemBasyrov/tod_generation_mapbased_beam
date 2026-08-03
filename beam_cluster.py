"""
beam_cluster.py — Weighted k-means clustering of beam pixels on the unit sphere.

Reduces the beam-pixel count before TOD generation, giving a direct speedup
in the innermost Numba gather loops with controllable approximation error.

Why this works for TOD (and not for B_ell)
-------------------------------------------
In TOD generation the accumulated signal is

    T(b) = Σ_s  w_s · sky(R_b · v_s)

where `sky` is a *smooth* CMB map.  Pixels within the same angular cluster
share nearly identical rotated positions R_b·v_s, so replacing them with one
centroid pixel introduces an error that scales as

    δT ~ cluster_radius · |∇ sky|

which is tiny for cluster radii well below the beam FWHM.

In contrast, B_ell computation involves Legendre polynomials P_ℓ(cos θ_s)
that oscillate on scales ~π/ℓ.  Merging pixels there destroys the
destructive interference that makes simple truncation work; hence k-means
must NOT be applied to B_ell computation.

Hybrid (tail-only) mode  ← recommended
----------------------------------------
Only the *tail* pixels — those whose cumulative beam power falls below a
configurable threshold (e.g. 3 %) — are clustered.  The main-lobe pixels
that carry the bulk of the power are kept individually.

Layout of the output arrays (n_main + K_tail elements):

    index 0 … n_main-1        → main-lobe pixels, unchanged (identity mapping)
    index n_main … n_main+K_tail-1 → tail clusters

The `labels` array encodes this mapping for every original pixel, allowing
`cluster_cached_arrays` to apply the same grouping to vec_rolled / dtheta /
dphi without any code changes.

Full mode  (tail_fraction=None)
---------------------------------
All S pixels are clustered into K centroids.  Useful for extreme memory
constraints; accuracy is lower than tail-only mode at the same K.

Cache-array clustering
-----------------------
A sparse (S × K_out) weight matrix W enables all three cache arrays to be
reduced in a single vectorised matrix multiply.  This is mathematically
exact for `vec_rolled` (rotation is linear) and a valid weighted-mean
approximation for the flat-sky offsets `dtheta`/`dphi`.

Algorithm
---------
Weighted spherical k-means with k-means++ initialisation:
1. Initialise K centroids by weighted k-means++ sampling in 3-D.
2. Assign each pixel to its nearest centroid (argmax of cosine similarity).
3. Recompute weighted centroids; project back to the unit sphere.
4. Repeat until max centroid displacement < tol or max_iter is reached.
"""

import time
import numpy as np
from scipy.sparse import csr_matrix


# ── k-means++ initialisation ──────────────────────────────────────────────────


def _kmeans_plus_plus_init(
    vec: np.ndarray, weights: np.ndarray, K: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Weighted k-means++ initialisation on the unit sphere.

    Parameters
    ----------
    vec     : (S, 3) float64  unit vectors
    weights : (S,)   float64  sampling probabilities (need not sum to 1)
    K       : int             number of centroids to initialise
    rng     : np.random.Generator

    Returns
    -------
    centroids : (K, 3) float64  initial centroid unit vectors
    """
    S = len(weights)
    prob = weights / weights.sum()
    centroids = np.empty((K, 3), dtype=np.float64)

    centroids[0] = vec[rng.choice(S, p=prob)]
    nearest_cos = vec @ centroids[0]  # (S,) — running max similarity

    for k in range(1, K):
        # k-means++ draws proportional to the squared chordal distance to the
        # nearest centroid, |v - c|^2 = 2(1 - cos ρ); the constant cancels in
        # the normalisation below.  Squaring this again would sample on ρ^4,
        # which over-weights the far tail and costs ~35 % in final scatter.
        chord_sq = np.maximum(1.0 - nearest_cos, 0.0)  # (S,)

        p = weights * chord_sq
        total = p.sum()
        p = prob if total == 0.0 else p / total
        centroids[k] = vec[rng.choice(S, p=p)]
        nearest_cos = np.maximum(nearest_cos, vec @ centroids[k])  # incremental update

    return centroids


# ── Core k-means EM loop ──────────────────────────────────────────────────────


def _to_unit(vec: np.ndarray) -> np.ndarray:
    """
    Promote beam-pixel vectors to float64 and restore unit norm.

    Assignment ranks candidates by the raw dot product v·c, so a centroid
    seeded straight from a stored pixel carries that pixel's norm error into
    the comparison.  Vectors stored in float32 miss unit norm by ~2e-8, while
    1 - cos(ρ) is only ~1e-8 across a half-arcmin cluster: the norm error would
    then outrank the angular separation the clustering exists to minimise, and
    the run would lose a factor of four in intra-cluster scatter.

    Parameters
    ----------
    vec : (S, 3) array   beam-pixel vectors in any precision

    Returns
    -------
    (S, 3) float64 unit vectors
    """
    v = np.asarray(vec, dtype=np.float64)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms > 0, norms, 1.0)


# Below this deviation from the identity the whitening is not worth applying:
# it would displace nodes by less than 1e-9 of a beam radius, which is two
# orders below the smallest term the error budget resolves, while the map's own
# round-trip is enough to flip a node across a Voronoi boundary and perturb the
# partition.  Skipping it makes a symmetric beam reproduce the unwhitened run
# bit for bit.  Real beams sit at 1e-2, eleven orders above this.
_WHITEN_MIN_ANISOTROPY = 1e-9


def _tangent_frame(vec: np.ndarray, bvals: np.ndarray) -> tuple | None:
    """Beam-centre direction plus an orthonormal basis of its tangent plane."""
    v = np.asarray(vec, dtype=np.float64)
    w = np.asarray(bvals, dtype=np.float64)
    c = (w[:, None] * v).sum(axis=0)
    n = float(np.linalg.norm(c))
    if n <= 0.0:
        return None
    c = c / n
    e1 = np.cross(np.array([0.0, 0.0, 1.0]), c)
    if np.linalg.norm(e1) < 1e-12:  # centre on a pole: any perpendicular serves
        e1 = np.cross(np.array([0.0, 1.0, 0.0]), c)
    e1 = e1 / np.linalg.norm(e1)
    return c, e1, np.cross(c, e1)


def _log_map(vec: np.ndarray, frame: tuple) -> tuple:
    """
    Geodesic-polar (exponential-map) coordinates about the beam centre.

    Distances from the centre are true arc lengths and angles about it are
    true bearings, so a beam that is a function of angular distance alone has
    an exactly isotropic second moment here.  The ``(RA, Dec)`` offset chart
    does not: ``cos rho = cos(dec) cos(RA)`` gives ``rho^2 = x^2 + y^2 -
    x^2y^2/3``, which turns a perfectly round beam into one with a traceless
    second moment of relative size ``sigma^2/2``.  Whitening on that would
    correct an asymmetry the instrument does not have.
    """
    c, e1, e2 = frame
    # Unit norm is load-bearing here, not hygiene: |perp| is compared against
    # sin(rho) recovered from the dot product, and a norm error d inflates it
    # by 2d/sin^2(rho) — a factor of 10^4 at half-arcmin separations.
    v = _to_unit(vec)
    cos_r = np.clip(v @ c, -1.0, 1.0)
    rho = np.arccos(cos_r)
    perp = v - cos_r[:, None] * c[None, :]
    n = np.linalg.norm(perp, axis=1)
    scale = np.where(n > 0, rho / np.where(n > 0, n, 1.0), 0.0)
    return scale * (perp @ e1), scale * (perp @ e2)


def _exp_map(t1: np.ndarray, t2: np.ndarray, frame: tuple) -> np.ndarray:
    """Inverse of :func:`_log_map`: tangent coordinates back onto the sphere."""
    c, e1, e2 = frame
    r = np.hypot(t1, t2)
    s = np.where(r > 0, np.sin(r) / np.where(r > 0, r, 1.0), 1.0)
    return (
        np.cos(r)[:, None] * c[None, :]
        + (s * t1)[:, None] * e1[None, :]
        + (s * t2)[:, None] * e2[None, :]
    )


def _whitener(vec: np.ndarray, bvals: np.ndarray, frame: tuple) -> np.ndarray | None:
    """
    Map that makes the beam's own second moment isotropic.

    The clustering error is exactly a beam second-moment deficit,
    ``Delta M = -Sigma_bar``.  A deficit *proportional* to the beam's own
    second moment M is a uniform narrowing: it moves b_ell but leaves the
    ellipticity untouched, so it adds no asymmetry the beam did not already
    have.  Any other deficit changes the beam's shape, and for a symmetric
    beam manufactures a quadrupole out of nothing.

    In ``u = W t`` with ``W = sqrt(mean eig) M^-1/2`` the beam is circular, so
    the isotropic distortion k-means minimises is the right functional there.
    Mapping back, ``Sigma_t = W^-1 Sigma_u W^-T = sigma^2 M`` — proportional by
    construction.  The ``sqrt(mean eig)`` prefactor keeps the angular scale of
    the footprint, which matters only because the assignment runs on the sphere.

    M is formed in the geodesic-polar coordinates of :func:`_log_map`, so a
    beam that is a function of angular distance alone gives exactly the
    identity and the clustering is left untouched.

    Args:
        vec: (S, 3) beam-pixel unit vectors.
        bvals: (S,) beam weights.
        frame: tangent frame from :func:`_tangent_frame`.

    Returns:
        (2, 2) float64 whitening matrix, or ``None`` when no whitening should
        be applied — either the beam's second moment is degenerate and the
        frame is undefined, or the beam is already round to within
        ``_WHITEN_MIN_ANISOTROPY`` and the caller should use the nodes as they
        stand.
    """
    t1, t2 = _log_map(vec, frame)
    w = np.asarray(bvals, dtype=np.float64)
    tot = w.sum()
    if tot <= 0:
        return None
    M = (
        np.array(
            [
                [(w * t1 * t1).sum(), (w * t1 * t2).sum()],
                [(w * t1 * t2).sum(), (w * t2 * t2).sum()],
            ]
        )
        / tot
    )
    ev, evec = np.linalg.eigh(M)
    if ev.min() <= 0 or not np.all(np.isfinite(ev)):
        return None
    W = np.sqrt(ev.mean()) * (evec @ np.diag(ev**-0.5) @ evec.T)
    if np.abs(W - np.eye(2)).max() < _WHITEN_MIN_ANISOTROPY:
        return None  # beam already round in this frame — leave it alone
    return W


def _apply_whitener(vec: np.ndarray, W: np.ndarray, frame: tuple) -> np.ndarray:
    """Unit vectors whose tangent coordinates are ``W`` applied to those of
    ``vec``.  ``W = I`` reproduces the input exactly."""
    t1, t2 = _log_map(vec, frame)
    p = np.stack([t1, t2], axis=1) @ W.T
    return _exp_map(p[:, 0], p[:, 1], frame)


def _centroids_from_labels(
    vec: np.ndarray, bvals: np.ndarray, labels: np.ndarray, K: int
) -> tuple:
    """
    Weighted-mean centroids of a partition, re-projected to the unit sphere.

    Taking the centroid from the labels rather than from the k-means state
    keeps the exact identity the error model rests on: the centroid *is* the
    weighted mean of its own members, so the clustered measure carries the
    grid's dipole unchanged and its whole leading error is the within-cluster
    covariance.  With a whitened assignment it is also the only correct
    source, since the k-means centroids live in the whitened frame.

    Returns:
        tuple: ``(centroids (K, 3) float64, weights (K,) float64)``.  Empty
        clusters are parked on the beam centroid and carry zero weight.
    """
    w = np.asarray(bvals, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    weights = np.bincount(labels, weights=w, minlength=K)
    mom = np.column_stack(
        [np.bincount(labels, weights=w * v[:, i], minlength=K) for i in range(3)]
    )
    norms = np.linalg.norm(mom, axis=1, keepdims=True)
    dead = norms[:, 0] <= 0.0
    centroids = mom / np.where(norms > 0, norms, 1.0)
    if dead.any():
        centre = (w[:, None] * v).sum(axis=0)
        n = np.linalg.norm(centre)
        centroids[dead] = centre / n if n > 0 else np.array([1.0, 0.0, 0.0])
    return centroids, weights


def _kmeans_sphere(
    vec: np.ndarray,
    bvals: np.ndarray,
    K: int,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
    label: str = "",
    verbose: bool = True,
) -> tuple:
    """
    Weighted spherical k-means.

    Parameters
    ----------
    vec    : (S, 3) float64  unit vectors, norm restored by `_to_unit`
    bvals  : (S,)   float64  beam weights (already float64, need not sum to 1)
    K      : int             number of clusters (must be < S)
    max_iter, tol, rng, label, verbose : as in cluster_beam_pixels

    Returns
    -------
    centroids : (K, 3) float64  unit-sphere centroids
    labels    : (S,)   int32   cluster index per pixel  (0 … K-1)
    """
    S = len(bvals)
    centroids = _kmeans_plus_plus_init(vec, bvals, K, rng)
    labels = np.empty(S, dtype=np.int32)
    cos_disp = np.full(K, -1.0)  # sentinel: triggers arccos only when needed

    cos_tol = np.cos(tol)
    for iteration in range(max_iter):
        sim = vec @ centroids.T  # (S, K)
        labels = sim.argmax(axis=1).astype(np.int32)

        new_weights = np.bincount(labels, weights=bvals, minlength=K)
        new_centroids = np.column_stack(
            [
                np.bincount(labels, weights=bvals * vec[:, i], minlength=K)
                for i in range(3)
            ]
        )

        # Reinitialise empty clusters from the pixels farthest from their own
        # centroid — one pixel each, or the empty clusters land on top of one
        # another, only one of them wins the next assignment, and the rest stay
        # empty for every remaining iteration.
        empty = np.where(new_weights == 0)[0]
        if empty.size:
            cos_to_assigned = sim[np.arange(S), labels]
            far = np.argpartition(cos_to_assigned, empty.size - 1)[: empty.size]
            new_centroids[empty] = vec[far]
            new_weights[empty] = bvals[far]

        norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        new_centroids /= norms

        cos_disp = np.clip((centroids * new_centroids).sum(axis=1), -1.0, 1.0)
        centroids = new_centroids

        if cos_disp.min() > cos_tol:
            if verbose:
                max_disp = float(np.arccos(cos_disp).max())
                print(
                    f"    [cluster{label}] Converged at iter {iteration + 1}  "
                    f"(max_disp={np.degrees(max_disp) * 60:.3f}')"
                )
            break
    else:
        if verbose:
            max_disp = float(np.arccos(cos_disp).max())
            print(
                f"    [cluster{label}] max_iter={max_iter} reached  "
                f"(max_disp={np.degrees(max_disp) * 60:.3f}')"
            )

    # Final assignment
    sim = vec @ centroids.T
    labels = sim.argmax(axis=1).astype(np.int32)

    return centroids, labels


def _spread_stats(
    vec: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    label: str = "",
    verbose: bool = True,
) -> None:
    """Print intra-cluster angular spread statistics [arcmin]."""
    if not verbose:
        return
    sim_assigned = (vec * centroids[labels]).sum(axis=1)
    cos_clamped = np.clip(sim_assigned, -1.0, 1.0)
    spread_arcmin = np.degrees(np.arccos(cos_clamped)) * 60.0
    print(
        f"    [cluster{label}] Intra-cluster spread — "
        f"mean={spread_arcmin.mean():.2f}'  "
        f"max={spread_arcmin.max():.2f}'  "
        f"95th={np.percentile(spread_arcmin, 95):.2f}'"
    )


# ── Public API ────────────────────────────────────────────────────────────────


def cluster_beam_pixels(
    vec_orig: np.ndarray,
    beam_vals: np.ndarray,
    n_clusters: int,
    tail_fraction: float | None = None,
    max_iter: int = 150,
    tol: float = 1e-5,
    random_state: int = 42,
    verbose: bool = True,
    whiten: bool = True,
) -> tuple:
    """
    Cluster beam pixels into representative centroids using weighted spherical
    k-means with k-means++ initialisation.

    Two modes are available:

    **Full mode** (``tail_fraction=None``):
        All S selected beam pixels are clustered into exactly K = n_clusters
        centroids.  Output has K pixels.

    **Hybrid / tail-only mode** (``tail_fraction`` is a float in (0, 1)):
        The pixels are split by cumulative beam power:

        * **Main** pixels (top ``1 - tail_fraction`` of power) — kept as-is,
          one-to-one.
        * **Tail** pixels (bottom ``tail_fraction`` of power) — clustered into
          at most ``n_clusters`` centroids.

        Output has ``n_main + K_tail`` pixels, where ``K_tail ≤ n_clusters``.
        The main lobe is reproduced exactly; only the low-power fringe is
        approximated.

    The return value's ``labels`` array maps every original pixel index to a
    position in the output arrays, so ``cluster_cached_arrays`` can apply the
    exact same grouping to ``vec_rolled`` / ``dtheta`` / ``dphi`` without any
    further changes.

    The k-means itself runs in float64 — the assignment resolves angular
    separations far below the working precision of the pipeline — but the
    arrays handed back carry the caller's own dtype, taken from ``beam_vals``,
    so a run configured for one precision is never silently served another.

    Parameters
    ----------
    vec_orig      : (S, 3)          unit vectors of selected beam pixels
    beam_vals     : (S,)            normalised beam weights (sum = 1); its
                                    dtype is the dtype of the returned arrays
    n_clusters    : int             max clusters for the tail (or all pixels in
                                    full mode); capped at the number of pixels
                                    being clustered
    tail_fraction : float | None    fraction of total beam power in the tail;
                                    ``None`` → full-mode (cluster all pixels)
    max_iter      : int             maximum EM iterations
    tol           : float           convergence threshold [rad]
    random_state  : int
    verbose       : bool            print progress messages; default ``True``
    whiten        : bool            group pixels in the frame where the beam's
                                    second moment is isotropic (see
                                    ``_whitener``), so the clustering narrows
                                    the beam without reshaping it.  Default
                                    ``True``\, ``False`` reproduces a plain
                                    spherical k-means, which manufactures a
                                    grid-aligned ellipticity even from a
                                    perfectly symmetric beam

    Returns
    -------
    vec_out    : (K_out, 3) beam_vals.dtype  centroid unit vectors
    bvals_out  : (K_out,)   beam_vals.dtype  weights per output pixel (sum ≈ 1)
    labels     : (S,)       int32     index into the output arrays for every
                                      original pixel (0 … K_out-1)
    """
    S = len(beam_vals)
    dtype = beam_vals.dtype

    if tail_fraction is None:
        # ── Full mode: cluster all pixels ─────────────────────────────────
        K = min(n_clusters, S)
        if K >= S:
            return (
                vec_orig.astype(dtype),
                beam_vals.copy(),
                np.arange(S, dtype=np.int32),
            )

        t0 = time.time()
        vec = _to_unit(vec_orig)
        bvals = beam_vals.astype(np.float64)
        rng = np.random.default_rng(random_state)

        frame = _tangent_frame(vec, bvals) if whiten else None
        W = _whitener(vec, bvals, frame) if frame is not None else None
        vec_assign = _apply_whitener(vec, W, frame) if W is not None else vec

        _, labels = _kmeans_sphere(
            vec_assign, bvals, K, max_iter, tol, rng, verbose=verbose
        )
        centroids, cluster_weights = _centroids_from_labels(vec, bvals, labels, K)
        cluster_weights /= cluster_weights.sum()

        if verbose:
            print(
                f"    [cluster] Full mode: {S} → {K} pixels  "
                f"({S / K:.1f}× reduction)  in {time.time() - t0:.2f}s"
            )
        _spread_stats(vec, centroids, labels, verbose=verbose)

        return (
            centroids.astype(dtype),
            cluster_weights.astype(dtype),
            labels,
        )

    # ── Hybrid mode: keep main lobe, cluster tail ──────────────────────────
    if not (0.0 < tail_fraction < 1.0):
        raise ValueError(f"tail_fraction must be in (0, 1), got {tail_fraction!r}")

    t0 = time.time()

    # Split pixels into main (high-power) and tail (low-power).
    # Ascending sort → cumulative sum from the weakest pixel up.
    sort_idx = np.argsort(beam_vals)  # ascending by weight
    cumsum = np.cumsum(beam_vals[sort_idx])  # cumulative power from weakest

    # Number of tail pixels: smallest subset whose total weight ≥ tail_fraction.
    # searchsorted('right') returns the first index k where cumsum[k] > tail_fraction,
    # i.e. k is already the count of pixels needed (unambiguous when cumsum[k] == tail_fraction).
    n_tail = int(np.searchsorted(cumsum, tail_fraction, side="right"))
    n_tail = max(1, min(n_tail, S - 1))  # keep at least 1 main pixel

    tail_idx = sort_idx[:n_tail]  # (n_tail,)  low-power pixels
    main_idx = sort_idx[n_tail:]  # (n_main,)  high-power pixels, ascending
    n_main = len(main_idx)

    tail_power = float(beam_vals[tail_idx].sum())
    if verbose:
        print(
            f"    [cluster] Hybrid mode: {S} pixels  →  "
            f"{n_main} main (top {100 * (1 - tail_power):.1f}% power, kept exact)  +  "
            f"tail {n_tail} px ({100 * tail_power:.1f}% power, to be clustered)"
        )

    K_tail = min(n_clusters, n_tail)

    if K_tail >= n_tail:
        # Tail is already small enough — no clustering needed
        vec_tail = vec_orig[tail_idx]
        bv_tail = beam_vals[tail_idx]
        tail_labels = np.arange(n_tail, dtype=np.int32)
        if verbose:
            print(
                f"    [cluster] Tail has only {n_tail} pixels — kept as-is (K_tail={K_tail})"
            )
    else:
        vec_t = _to_unit(vec_orig[tail_idx])
        bv_t = beam_vals[tail_idx].astype(np.float64)
        rng = np.random.default_rng(random_state)

        # The whitening frame is set by the whole beam, not by the tail: it is
        # the beam's own second moment that the clustering must not reshape.
        vec_all, bv_all = _to_unit(vec_orig), beam_vals.astype(np.float64)
        frame = _tangent_frame(vec_all, bv_all) if whiten else None
        W = _whitener(vec_all, bv_all, frame) if frame is not None else None
        vec_assign = _apply_whitener(vec_t, W, frame) if W is not None else vec_t

        _, tail_labels = _kmeans_sphere(
            vec_assign,
            bv_t,
            K_tail,
            max_iter,
            tol,
            rng,
            label=" tail",
            verbose=verbose,
        )
        # Centroids come from the labels in the ORIGINAL frame; the k-means
        # ones are whitened.  (No re-normalisation here — combined with main.)
        centroids_t, cw_t = _centroids_from_labels(vec_t, bv_t, tail_labels, K_tail)

        # Left in float64: both are combined with the main lobe and
        # renormalised below, and only then cast to the caller's dtype.
        vec_tail = centroids_t  # (K_tail, 3)
        bv_tail = cw_t  # (K_tail,)

        if verbose:
            print(
                f"    [cluster] Tail: {n_tail} → {K_tail} clusters  "
                f"({n_tail / K_tail:.1f}× reduction)  in {time.time() - t0:.2f}s"
            )
        _spread_stats(vec_t, centroids_t, tail_labels, label=" tail", verbose=verbose)

    # ── Build combined output arrays ───────────────────────────────────────
    # Layout: [main_0, main_1, …, main_{n_main-1}, tail_cluster_0, …]
    vec_out = np.concatenate([vec_orig[main_idx], vec_tail], axis=0).astype(dtype)
    bv_out = np.concatenate(
        [beam_vals[main_idx].astype(np.float64), bv_tail], axis=0
    )  # (K_out,) — summed in float64, the main lobe runs to tens of thousands
    bv_out = (bv_out / bv_out.sum()).astype(dtype)  # re-normalise

    K_out = n_main + K_tail

    # ── Build labels: original pixel → position in output array ───────────
    labels = np.empty(S, dtype=np.int32)
    labels[main_idx] = np.arange(n_main, dtype=np.int32)  # identity
    labels[tail_idx] = n_main + tail_labels  # cluster offset

    if verbose:
        print(
            f"    [cluster] Combined: {S} → {K_out} pixels  "
            f"({S / K_out:.1f}× total reduction)"
        )

    return vec_out, bv_out, labels


# ── Sparse assignment matrix ──────────────────────────────────────────────────


def _build_weight_matrix(
    labels: np.ndarray, beam_vals: np.ndarray, K: int
) -> csr_matrix:
    """
    Build a sparse (S × K) weight matrix W where W[s, k] = beam_vals[s]
    when labels[s] == k, and 0 otherwise.

    For main pixels (identity mapping) each column contains exactly one entry.
    For tail clusters each column aggregates all member pixels.

    Cluster-weighted averages:  (W.T @ X) / cluster_weights[:, None]

    Held in float64: the matrix carries the accumulation of every member of a
    cluster, and the caller casts the finished means back to its own dtype.
    """
    S = len(labels)
    return csr_matrix(
        (beam_vals.astype(np.float64), (np.arange(S, dtype=np.int32), labels)),
        shape=(S, K),
    )


# ── Cache-array clustering ────────────────────────────────────────────────────


def cluster_cached_arrays(
    cache_dict: dict, labels: np.ndarray, beam_vals: np.ndarray, K: int
) -> dict:
    """
    Apply the same pixel-clustering to pre-computed beam-cache arrays.

    Works for both full mode and hybrid mode because the `labels` array
    already encodes the full mapping (main pixels → identity columns, tail
    pixels → shared cluster columns).

    Cache arrays:
        vec_rolled : (N_psi, S, 3)  → (N_psi, K, 3)
        dtheta     : (N_psi, S)     → (N_psi, K)
        dphi       : (N_psi, S)     → (N_psi, K)

    For `vec_rolled`, the weighted mean of member unit vectors is re-projected
    onto the unit sphere (exact for rotations, second-order error otherwise).
    For `dtheta` / `dphi`, the weighted mean is a valid flat-sky approximation
    within each cluster's angular radius.

    Parameters
    ----------
    cache_dict : dict   keys 'vec_rolled', 'dtheta', 'dphi' (any subset)
    labels     : (S,) int32   output index per original pixel
    beam_vals  : (S,)         original (pre-clustering) normalised beam weights;
                              its dtype is the dtype of the returned arrays
    K          : int          total number of output pixels (n_main + K_tail)

    Returns
    -------
    dict with the same keys, arrays replaced by their K-element versions
    """
    result = {}
    dtype = beam_vals.dtype

    W = _build_weight_matrix(labels, beam_vals, K)  # (S, K) csr float64

    # Per-output-pixel weight sum — used to convert weighted sums to means.
    # For main pixels this equals beam_vals[s]; for clusters it is the sum.
    cluster_w = np.asarray(
        W.T @ np.ones(len(labels), dtype=np.float64), dtype=np.float64
    )  # (K,)
    cluster_w = np.where(cluster_w > 0, cluster_w, 1.0)  # guard /0

    # ── vec_rolled : (N_psi, S, 3) ───────────────────────────────────────────
    if "vec_rolled" in cache_dict:
        vr = cache_dict["vec_rolled"]  # (N_psi, S, 3)
        N_psi, S_c, _ = vr.shape

        # Transpose to (S, N_psi*3), sparse-multiply, reshape back
        vr_2d = np.ascontiguousarray(
            vr.transpose(1, 0, 2).reshape(S_c, N_psi * 3)
        )  # (S, N_psi*3)
        res_2d = W.T @ vr_2d  # (K, N_psi*3)
        res = res_2d.reshape(K, N_psi, 3).transpose(1, 0, 2)  # (N_psi, K, 3)

        res /= cluster_w[None, :, None]  # weighted mean

        # Re-project to unit sphere
        norms = np.linalg.norm(res, axis=2, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        res /= norms

        result["vec_rolled"] = np.ascontiguousarray(res.astype(dtype))
        print(f"    [cluster] vec_rolled: ({N_psi}, {S_c}, 3) → ({N_psi}, {K}, 3)")

    # ── Scalar arrays: dtheta and dphi  (N_psi, S) ───────────────────────────
    for key in ("dtheta", "dphi"):
        if key not in cache_dict:
            continue
        arr = cache_dict[key]  # (N_psi, S)
        N_psi, S_c = arr.shape

        arr_2d = np.ascontiguousarray(arr.T)  # (S, N_psi)
        res_2d = W.T @ arr_2d  # (K, N_psi)
        res = (res_2d / cluster_w[:, None]).T  # (N_psi, K)

        result[key] = np.ascontiguousarray(res.astype(dtype))
        print(f"    [cluster] {key}: ({N_psi}, {S_c}) → ({N_psi}, {K})")

    return result
