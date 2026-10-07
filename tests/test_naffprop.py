import numpy as np
import pytest
from sklearn.cluster import AffinityPropagation as SkAffinityPropagation
from sklearn.cluster import affinity_propagation as sk_affinity_propagation
from sklearn.datasets import make_blobs

from naffprop import AffinityPropagation, apcluster, apcluster_k, neg_dist_mat, preference_range
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


@pytest.mark.parametrize("kwargs", [
    dict(s=np.zeros((3, 2))),
    dict(p=[1.0, 2.0]),
    dict(q=1.5),
    dict(lam=0.3),
    dict(maxits=0),
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
