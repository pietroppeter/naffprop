import numpy as np
import pytest
import scipy.sparse
from sklearn.cluster import AffinityPropagation as SkAffinityPropagation
from sklearn.cluster import affinity_propagation as sk_affinity_propagation
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from naffprop import (AffinityPropagation, apcluster, apcluster_k, apcluster_l, neg_dist_mat,
                      preference_range)
from reference import reference_apcluster

rng = np.random.default_rng(7)


def blobs(n=150, centers=3, seed=0):
    x, _ = make_blobs(n_samples=n, centers=centers, cluster_std=0.6, random_state=seed)
    return x


@pytest.mark.parametrize("n,r", [(2, 1), (10, 2), (60, 1), (120, 2)])
def test_neg_dist_mat(n, r):
    x = rng.normal(size=(n, 3))
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    np.testing.assert_allclose(neg_dist_mat(x, r), -(d**r), atol=1e-12)


@pytest.mark.parametrize("data", ["normal", "blobs"])
@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_matches_reference(data, q):
    x = rng.normal(size=(80, 2)) if data == "normal" else blobs(100)
    s = neg_dist_mat(x, 2)
    res = apcluster(s, q=q, nonoise=True)
    off = s[~np.eye(len(s), dtype=bool)]
    assert res.p[0] == pytest.approx(np.quantile(off, q))
    ex, labels, its = reference_apcluster(s, res.p[0])
    np.testing.assert_array_equal(res.exemplars, ex)
    np.testing.assert_array_equal(res.labels, labels)
    assert res.iterations == its
    assert res.converged


def test_matches_reference_damping_and_per_point_preference():
    x = blobs(90, centers=4, seed=3)
    s = neg_dist_mat(x, 1)
    p = rng.uniform(-8, -1, size=len(s))
    res = apcluster(s, p=p, lam=0.6, convits=20, nonoise=True)
    ex, labels, its = reference_apcluster(s, p, convits=20, lam=0.6)
    np.testing.assert_array_equal(res.exemplars, ex)
    np.testing.assert_array_equal(res.labels, labels)
    assert res.iterations == its


def test_netsim():
    s = neg_dist_mat(blobs(60), 2)
    res = apcluster(s, nonoise=True)
    idx = res.exemplars[res.labels]
    expref = res.p[res.exemplars].sum()
    notex = np.setdiff1d(np.arange(len(s)), res.exemplars)
    dpsim = s[notex, idx[notex]].sum()
    assert res.expref == pytest.approx(expref)
    assert res.dpsim == pytest.approx(dpsim)
    assert res.netsim == pytest.approx(dpsim + expref)
    assert [len(c) for c in res.clusters] == list(np.bincount(res.labels))


def test_matches_sklearn():
    # Same similarities, preference and parameters: same exemplars and labels
    # (both add the same kind of tiny noise, so a well separated dataset).
    x = blobs(150)
    s = neg_dist_mat(x, 2)
    p = np.median(s)
    for damping, max_iter, conv in [(0.5, 200, 15), (0.9, 1000, 100)]:
        centers, labels = sk_affinity_propagation(
            s, preference=p, damping=damping, max_iter=max_iter,
            convergence_iter=conv, random_state=0)
        res = apcluster(s, p=p, lam=damping, maxits=max_iter, convits=conv, seed=0)
        np.testing.assert_array_equal(res.exemplars, centers)
        np.testing.assert_array_equal(res.labels, labels)


def test_quantile_controls_clusters():
    s = neg_dist_mat(rng.normal(size=(100, 2)), 2)
    counts = [len(apcluster(s, q=q, seed=1)) for q in (0.01, 0.5, 0.99)]
    assert counts[0] < counts[1] < counts[2]


def test_noise_is_seeded():
    s = neg_dist_mat(rng.integers(0, 4, size=(60, 2)).astype(float), 2)  # many ties
    a, b = apcluster(s, seed=5), apcluster(s, seed=5)
    np.testing.assert_array_equal(a.labels, b.labels)


def brute_force_range(s):
    n = len(s)
    s = s.copy()
    np.fill_diagonal(s, 0)
    dpsim1 = s.sum(axis=0).max()
    dpsim2 = max(np.maximum(s[:, i], s[:, j]).sum() for i in range(n) for j in range(i + 1, n))
    return dpsim1 - dpsim2, s[~np.eye(n, dtype=bool)].max()


def test_preference_range():
    s = neg_dist_mat(rng.normal(size=(40, 2)), 2)
    lo, hi = preference_range(s, exact=True)
    assert (lo, hi) == pytest.approx(brute_force_range(s))
    lo_approx, hi_approx = preference_range(s)
    assert hi_approx == hi and lo_approx <= lo
    assert len(apcluster(s, p=hi + 1e-9, seed=0)) == len(s)
    assert len(apcluster(s, p=lo - 1, seed=0)) <= 2


@pytest.mark.parametrize("k", [2, 3, 5, 8])
def test_apcluster_k(k):
    s = neg_dist_mat(blobs(120, centers=k, seed=k), 2)
    assert len(apcluster_k(s, k, prc=0, seed=0)) == k


def test_minus_inf_similarities():
    # Two groups that must never be linked.
    x = blobs(40, centers=2)
    s = neg_dist_mat(x, 2)
    group = np.arange(40) < 20
    s[np.ix_(group, ~group)] = -np.inf
    s[np.ix_(~group, group)] = -np.inf
    res = apcluster(s, seed=0)
    assert res.converged
    for c in res.clusters:
        assert group[c].all() or (~group[c]).all()
    est = AffinityPropagation(affinity="precomputed", random_state=0).fit(s)
    np.testing.assert_array_equal(est.labels_, res.labels)
    with pytest.raises(ValueError):
        AffinityPropagation(affinity="precomputed").fit(np.where(group[:, None], np.nan, s))


def test_input_not_modified_and_strided():
    s = neg_dist_mat(rng.normal(size=(30, 2)), 2)
    s0 = s.copy()
    res = apcluster(s, nonoise=True)
    np.testing.assert_array_equal(s, s0)
    big = np.zeros((60, 60))
    big[::2, ::2] = s
    res2 = apcluster(big[::2, ::2], nonoise=True)
    np.testing.assert_array_equal(res.labels, res2.labels)


def test_copy_false_works_in_place():
    s = neg_dist_mat(rng.normal(size=(30, 2)), 2)
    expected = apcluster(s, seed=3)
    res = apcluster(s, seed=3, copy=False)
    np.testing.assert_array_equal(res.labels, expected.labels)
    np.testing.assert_array_equal(np.diag(s), res.p)  # s now holds the preferences


@pytest.mark.parametrize("nonoise", [True, False])
@pytest.mark.parametrize("q", [0.1, 0.5])
def test_float32_same_clusters(nonoise, q):
    s = neg_dist_mat(blobs(200, centers=5, seed=2), 2)
    res64 = apcluster(s, q=q, nonoise=nonoise, seed=0)
    res32 = apcluster(s, q=q, nonoise=nonoise, seed=0, dtype=np.float32)
    np.testing.assert_array_equal(res32.exemplars, res64.exemplars)
    np.testing.assert_array_equal(res32.labels, res64.labels)
    assert res32.netsim == pytest.approx(res64.netsim, rel=1e-6)
    # float32 similarities are kept in float32 (no float64 copy).
    s32 = neg_dist_mat(blobs(200, centers=5, seed=2), 2, dtype=np.float32)
    assert s32.dtype == np.float32
    np.testing.assert_allclose(s32, s, rtol=1e-6)
    np.testing.assert_array_equal(apcluster(s32, q=q, nonoise=nonoise, seed=0).labels, res64.labels)
    assert preference_range(s32) == pytest.approx(preference_range(s), rel=1e-6)


def test_estimator_float32():
    x = blobs(150)
    ap64 = AffinityPropagation(random_state=0).fit(x)
    ap32 = AffinityPropagation(random_state=0).fit(x.astype(np.float32))
    assert ap32.affinity_matrix_.dtype == np.float32
    np.testing.assert_array_equal(ap32.labels_, ap64.labels_)


@pytest.mark.parametrize("kwargs", [
    dict(s=np.zeros((3, 2))),
    dict(p=[1.0, 2.0]),
    dict(q=1.5),
    dict(lam=0.3),
    dict(maxits=0),
    dict(dtype=np.float16),
])
def test_invalid_arguments(kwargs):
    kwargs = {"s": -np.ones((3, 3)), **kwargs}
    with pytest.raises(ValueError):
        apcluster(**kwargs)


def test_estimator_like_sklearn():
    x = blobs(150)
    params = dict(damping=0.5, max_iter=200, convergence_iter=15, random_state=0)
    ours = AffinityPropagation(preference=-50, **params).fit(x)
    theirs = SkAffinityPropagation(preference=-50, **params).fit(x)
    np.testing.assert_array_equal(ours.labels_, theirs.labels_)
    np.testing.assert_array_equal(ours.cluster_centers_indices_, theirs.cluster_centers_indices_)
    np.testing.assert_allclose(ours.cluster_centers_, theirs.cluster_centers_)
    xn = rng.normal(size=(20, 2)) * 5
    np.testing.assert_array_equal(ours.predict(xn), theirs.predict(xn))


def test_estimator_n_clusters_and_precomputed():
    x = blobs(120, centers=4, seed=1)
    est = AffinityPropagation(n_clusters=4, random_state=0).fit(x)
    assert len(est.cluster_centers_indices_) == 4
    s = neg_dist_mat(x, 2)
    pre = AffinityPropagation(affinity="precomputed", random_state=0).fit(s)
    eu = AffinityPropagation(random_state=0).fit(x)
    np.testing.assert_array_equal(pre.labels_, eu.labels_)
    with pytest.raises(ValueError):
        pre.predict(s)


def test_sklearn_check_estimator():
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(AffinityPropagation(random_state=0))


# Sparse similarities (R's apcluster on a sparse matrix)

def knn_sparse(s, k):
    """The k most similar other points of each point, as a sparse matrix, and
    the same similarities as a dense matrix with -Inf for the pairs left out."""
    n = len(s)
    off = s + np.diag(np.full(n, -np.inf))
    nbrs = np.argsort(-off, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    sp = scipy.sparse.coo_array((s[rows, nbrs.ravel()], (rows, nbrs.ravel())), shape=(n, n))
    dense = np.full((n, n), -np.inf)
    dense[rows, nbrs.ravel()] = s[rows, nbrs.ravel()]
    return sp, dense


def test_sparse_all_pairs_same_as_dense():
    s = neg_dist_mat(blobs(80, centers=4, seed=4), 2)
    dense = apcluster(s, nonoise=True)
    sparse = apcluster(scipy.sparse.csr_array(s), nonoise=True)
    np.testing.assert_array_equal(sparse.exemplars, dense.exemplars)
    np.testing.assert_array_equal(sparse.labels, dense.labels)
    assert sparse.iterations == dense.iterations
    assert sparse.netsim == pytest.approx(dense.netsim)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_sparse_same_as_dense_with_minus_inf(dtype):
    s = neg_dist_mat(blobs(150, centers=5, seed=5), 2)
    sp, dense = knn_sparse(s, 15)
    p = np.median(sp.data)
    res_sparse = apcluster(sp, nonoise=True, dtype=dtype)
    res_dense = apcluster(dense, p=p, nonoise=True, dtype=dtype)
    assert res_sparse.p[0] == p  # the median of the stored similarities
    np.testing.assert_array_equal(res_sparse.exemplars, res_dense.exemplars)
    np.testing.assert_array_equal(res_sparse.labels, res_dense.labels)
    assert res_sparse.iterations == res_dense.iterations


def test_sparse_with_noise_and_estimator():
    x = blobs(200, centers=4, seed=6)
    sp, _ = knn_sparse(neg_dist_mat(x, 2), 30)
    dense = AffinityPropagation(random_state=0).fit(x)
    # The default preference would be the median of the stored (nearest) pairs:
    # use the dense one, the median of all the pairs.
    res = AffinityPropagation(affinity="precomputed", preference=dense.preference_[0],
                              random_state=0).fit(scipy.sparse.csr_array(sp))
    assert adjusted_rand_score(res.labels_, dense.labels_) > 0.9
    with pytest.raises(TypeError):
        preference_range(sp)
    with pytest.raises(ValueError):
        AffinityPropagation(affinity="precomputed", n_clusters=3).fit(sp)


# Leveraged affinity propagation (R's apclusterL)

def test_neg_dist_mat_sel():
    x = rng.normal(size=(20, 3))
    sel = [1, 4, 9]
    np.testing.assert_allclose(neg_dist_mat(x, 2, sel=sel), neg_dist_mat(x, 2)[:, sel])
    with pytest.raises(ValueError):
        neg_dist_mat(x, 2, sel=[4, 1])


def test_leveraged_all_points_same_as_dense():
    x = blobs(100, centers=4, seed=7)
    dense = apcluster(neg_dist_mat(x, 2), nonoise=True)
    lev = apcluster_l(x, frac=1, sweeps=1, nonoise=True, seed=0)
    np.testing.assert_array_equal(lev.sel, np.arange(100))
    np.testing.assert_array_equal(lev.exemplars, dense.exemplars)
    np.testing.assert_array_equal(lev.labels, dense.labels)
    assert lev.iterations == dense.iterations


def test_leveraged_same_as_sparse_on_its_pairs():
    # Leveraged AP is AP on the pairs (i, sel) and the diagonal.
    x = blobs(120, centers=4, seed=8)
    sel = np.sort(np.random.default_rng(0).choice(120, 40, replace=False))
    sim = neg_dist_mat(x, 2, sel=sel)
    p = np.median(sim[sim < 0])
    rows = np.repeat(np.arange(120), 40)
    sp = scipy.sparse.coo_array((sim.ravel(), (rows, np.tile(sel, 120))), shape=(120, 120))
    res_sparse = apcluster(sp, p=p, nonoise=True)
    res_lev = apcluster_l(x, frac=None, sweeps=1, p=p, nonoise=True, sel=sel)
    assert set(res_sparse.exemplars) <= set(sel)
    np.testing.assert_array_equal(res_lev.exemplars, res_sparse.exemplars)
    np.testing.assert_array_equal(res_lev.labels, res_sparse.labels)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_leveraged_finds_the_blobs(dtype):
    x, y = make_blobs(n_samples=600, centers=5, cluster_std=0.5, random_state=9)
    res = apcluster_l(x, frac=0.1, sweeps=5, q=0.01, seed=1, dtype=dtype)
    assert len(res.sel) == 60
    assert set(res.exemplars) <= set(res.sel)
    assert adjusted_rand_score(res.labels, y) > 0.95
    est = AffinityPropagation(leveraged=0.1, sweeps=5, preference_quantile=0.01,
                              random_state=1).fit(x.astype(dtype))
    assert adjusted_rand_score(est.labels_, y) > 0.95
    assert est.predict(x[:5]).shape == (5,)


# ScaleAP's pruning: the same messages, most of them not stored


@pytest.mark.parametrize("data", ["normal", "blobs"])
@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_scaleap_matches_reference(data, q):
    x = rng.normal(size=(80, 2)) if data == "normal" else blobs(100)
    s = neg_dist_mat(x, 2)
    res = apcluster(s, q=q, nonoise=True, scaleap=True)
    ex, labels, its = reference_apcluster(s, res.p[0])
    np.testing.assert_array_equal(res.exemplars, ex)
    np.testing.assert_array_equal(res.labels, labels)
    assert res.iterations == its
    assert res.netsim == pytest.approx(apcluster(s, q=q, nonoise=True).netsim)


def test_scaleap_damping_and_per_point_preference():
    x = blobs(90, centers=4, seed=3)
    s = neg_dist_mat(x, 1)
    p = rng.uniform(-8, -1, size=len(s))
    res = apcluster(s, p=p, lam=0.6, convits=20, nonoise=True, scaleap=True)
    ex, labels, its = reference_apcluster(s, p, convits=20, lam=0.6)
    np.testing.assert_array_equal(res.exemplars, ex)
    np.testing.assert_array_equal(res.labels, labels)
    assert res.iterations == its


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_scaleap_same_as_dense(dtype):
    x = blobs(400, centers=8, seed=4)
    s = neg_dist_mat(x, 2)
    s[rng.random(s.shape) < 0.2] = -np.inf  # pairs never linked
    dense = apcluster(s, seed=1, dtype=dtype)
    pruned = apcluster(s, seed=1, dtype=dtype, scaleap=True)
    np.testing.assert_array_equal(pruned.exemplars, dense.exemplars)
    np.testing.assert_array_equal(pruned.labels, dense.labels)
    assert pruned.iterations == dense.iterations
    assert apcluster_k(s, 5, prc=0, seed=1, scaleap=True).labels.max() == 4


def test_scaleap_estimator():
    from sklearn.utils.estimator_checks import check_estimator

    x = blobs(150)
    np.testing.assert_array_equal(AffinityPropagation(scaleap=True, random_state=0).fit(x).labels_,
                                  AffinityPropagation(random_state=0).fit(x).labels_)
    check_estimator(AffinityPropagation(scaleap=True, random_state=0))
    with pytest.raises(ValueError):
        AffinityPropagation(scaleap=True, leveraged=0.5).fit(x)
    with pytest.raises(ValueError):
        apcluster(scipy.sparse.csr_array(neg_dist_mat(x, 2)), scaleap=True)


def test_nimble_version_matches_pyproject():
    # The Nim package (naffprop.nimble) and the Python package are released together.
    import pathlib
    import re

    root = pathlib.Path(__file__).parent.parent
    nimble = re.search(r'^version\s*=\s*"(.+)"', (root / "naffprop.nimble").read_text(), re.M)
    pyproject = re.search(r'^version\s*=\s*"(.+)"', (root / "pyproject.toml").read_text(), re.M)
    assert nimble.group(1) == pyproject.group(1)
