"""ScaleAP's equations next to Frey and Dueck's, in numpy (small n).

1. The paper's AP (Shiokawa, AAAI 2021, "Preliminaries"): the self-responsibility
   is s(k, k) - max_{j != k} s(k, j), without the availabilities, and the
   availabilities are computed from the previous iteration's responsibilities.
   This compares its clusters with those of Frey and Dueck's updates (the numpy
   port of their MATLAB code that naffprop is tested against).
2. ScaleAP's pruning applied to Frey and Dueck's updates, as naffprop's
   scaleap.nim does: responsibilities of pairs never the best candidate are
   (1 - lam^t) s(i, k) - b(i), availabilities of pairs never with a positive
   responsibility are one number per column. It finds the same exemplars in the
   same number of iterations, and only a few pairs keep their own messages.

    uv run python experiments/paper_equations.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))
from reference import reference_apcluster  # noqa: E402


def frey_dueck_exemplars(s, p, maxits=1000, convits=100, lam=0.9):
    """The exemplars of the MATLAB code before refinement, and the iterations."""
    s = np.array(s, float)
    n = len(s)
    rows = np.arange(n)
    s[rows, rows] = p
    a = np.zeros((n, n))
    r = np.zeros((n, n))
    e = np.zeros((n, convits), bool)
    for i in range(maxits):
        as_ = a + s
        y = as_.argmax(1)
        m1 = as_[rows, y]
        as_[rows, y] = -np.inf
        m2 = as_.max(1)
        rnew = s - m1[:, None]
        rnew[rows, y] = s[rows, y] - m2
        r = (1 - lam) * rnew + lam * r
        rp = np.maximum(r, 0)
        rp[rows, rows] = r[rows, rows]
        anew = rp.sum(0)[None, :] - rp
        da = anew[rows, rows].copy()
        anew = np.minimum(anew, 0)
        anew[rows, rows] = da
        a = (1 - lam) * anew + lam * a
        ex = np.diag(a) + np.diag(r) > 0
        e[:, i % convits] = ex
        if i >= convits - 1:
            se = e.sum(1)
            if ((se == convits) | (se == 0)).all() and ex.sum() > 0:
                break
    return np.flatnonzero(ex), i + 1


def paper_ap(s, p, maxits=1000, convits=100, lam=0.9):
    """The paper's equations: exemplar of i is argmax_j r(i, j) + a(i, j), stop
    when no exemplar changed for convits iterations."""
    s = np.array(s, float)
    n = len(s)
    rows = np.arange(n)
    s[rows, rows] = p
    off = s.copy()
    off[rows, rows] = -np.inf
    rself = s[rows, rows] - off.max(1)
    a = np.zeros((n, n))
    r = np.zeros((n, n))
    r[rows, rows] = rself
    prev, stable = None, 0
    for t in range(maxits):
        as_ = a + s
        y = as_.argmax(1)
        m1 = as_[rows, y]
        as_[rows, y] = -np.inf
        m2 = as_.max(1)
        rho = s - m1[:, None]
        rho[rows, y] = s[rows, y] - m2
        rho[rows, rows] = rself
        rprev = r
        r = (1 - lam) * rho + lam * r
        rp = np.maximum(rprev, 0)
        rp[rows, rows] = 0
        cs = rp.sum(0)
        alpha = np.minimum(0, rprev[rows, rows][None, :] + cs[None, :] - rp)
        alpha[rows, rows] = cs - rp[rows, rows]
        a = (1 - lam) * alpha + lam * a
        e = (r + a).argmax(1)
        stable = stable + 1 if prev is not None and (e == prev).all() else 0
        prev = e
        if stable >= convits:
            break
    return e, t + 1


def pruned_ap(s, p, maxits=1000, convits=100, lam=0.9):
    """Frey and Dueck's updates, storing messages only for the pairs that were
    a best candidate or had a positive responsibility (as scaleap.nim)."""
    s = np.array(s, float)
    n = len(s)
    rows = np.arange(n)
    s[rows, rows] = p
    b, acol, rd, ad = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    pairs = [dict() for _ in range(n)]  # k -> [r, a]
    cs_prev = np.zeros(n)
    hist = np.zeros((n, convits), bool)
    most = 0
    for t in range(1, maxits + 1):
        cs = np.zeros(n)
        for i in range(n):
            if t > 1:  # the stored availabilities of the previous iteration
                for k, v in pairs[i].items():
                    v[1] = (1 - lam) * min(0.0, cs_prev[k] - max(v[0], 0.0)) + lam * v[1]
            row = s[i] + acol
            row[i] = s[i, i] + ad[i]
            for k, v in pairs[i].items():
                row[k] = s[i, k] + v[1]
            y = row.argmax()
            m1 = row[y]
            row[y] = -np.inf
            m2 = row.max()
            if y != i and y not in pairs[i]:
                pairs[i][y] = [(1 - lam ** (t - 1)) * s[i, y] - b[i], acol[y]]
            for k, v in pairs[i].items():
                v[0] = (1 - lam) * (s[i, k] - (m2 if k == y else m1)) + lam * v[0]
            rd[i] = (1 - lam) * (s[i, i] - (m2 if y == i else m1)) + lam * rd[i]
            b[i] = lam * b[i] + (1 - lam) * m1
            implicit = (1 - lam ** t) * s[i] - b[i]
            for k in np.flatnonzero(implicit > 0):
                if k != i and k not in pairs[i]:
                    pairs[i][k] = [implicit[k], acol[k]]
            for k, v in pairs[i].items():
                if v[0] > 0:
                    cs[k] += v[0]
            cs[i] += rd[i]
        acol = (1 - lam) * np.minimum(0, cs) + lam * acol
        ad = (1 - lam) * (cs - rd) + lam * ad
        cs_prev = cs
        most = max(most, sum(len(x) for x in pairs))
        ex = ad + rd > 0
        hist[:, (t - 1) % convits] = ex
        if t >= convits:
            se = hist.sum(1)
            if ((se == convits) | (se == 0)).all() and ex.sum() > 0:
                break
    return np.flatnonzero(ex), t, most


def netsim(s, p, exemplar_of):
    """The objective of AP: similarities of the points to their exemplars plus
    the preferences of the exemplars (higher is better)."""
    i = np.arange(len(s))
    return np.where(exemplar_of == i, p, s[i, exemplar_of]).sum()


def main():
    print("| n | centers | damping | clusters (Frey-Dueck / paper) | net similarity (Frey-Dueck / paper) "
          "| ARI paper vs Frey-Dueck "
          "| pruned: same exemplars | pruned: same iterations | pairs stored (% of n^2) |")
    print("|--:|--:|--:|--:|--:|--:|:-:|:-:|--:|")
    for seed, n, c in [(0, 150, 3), (1, 300, 5), (2, 400, 8)]:
        x, _ = make_blobs(n_samples=n, centers=c, cluster_std=1.0, random_state=seed)
        s = -((x[:, None] - x[None]) ** 2).sum(-1)
        s = s + 1e-9 * np.random.default_rng(seed).standard_normal(s.shape)  # no ties
        p = np.median(s[~np.eye(n, dtype=bool)])
        for lam in (0.9, 0.5):
            refex, labels, _ = reference_apcluster(s, p, lam=lam)
            ex, its = frey_dueck_exemplars(s, p, lam=lam)
            e, _ = paper_ap(s, p, lam=lam)
            pex, pits, most = pruned_ap(s, p, lam=lam)
            k_paper = len(np.unique(e))
            ari = adjusted_rand_score(labels, np.unique(e, return_inverse=True)[1])
            ns = f"{netsim(s, p, refex[labels]):.1f} / {netsim(s, p, e):.1f}"
            print(f"| {n} | {c} | {lam} | {len(np.unique(labels))} / {k_paper} | {ns} | {ari:.2f} "
                  f"| {'yes' if np.array_equal(ex, pex) else 'no'} | {'yes' if its == pits else 'no'} "
                  f"| {100 * most / n**2:.1f} |", flush=True)


if __name__ == "__main__":
    main()
