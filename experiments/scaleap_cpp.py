"""ScaleAP's C++ reference code (Shiokawa, AAAI 2021) against naffprop, on the
same similarities and preference, without noise.

It clones LazyShion/ScaleAP (MIT) at a fixed commit into experiments/.scaleap
(not part of naffprop) and builds it with its Makefile, so it needs git, make
and g++. For each data set it reports the clusters, the net similarity (the
objective of affinity propagation: similarities of the points to their
exemplars plus the preferences of the exemplars, higher is better), the
agreement with naffprop's clustering (adjusted Rand index) and the time. Then
it checks whether the C++ code converges: how many points change exemplar
between consecutive iterations.

    uv run python experiments/scaleap_cpp.py

The C++ time grows about as n^3.7 here (its availability update loops over all
points for each candidate pair), so n stays small.
"""

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

import naffprop

REPO = "https://github.com/LazyShion/ScaleAP"
COMMIT = "aedda8dc2875d522f6b5e8e6438763c96d72d9f4"  # 2021-12-23, the latest
DIR = Path(__file__).with_name(".scaleap")


def build():
    exe = DIR / "ScaleAP"
    if not exe.exists():
        if not DIR.exists():
            subprocess.run(["git", "clone", "-q", REPO, str(DIR)], check=True)
        subprocess.run(["git", "-C", str(DIR), "checkout", "-q", COMMIT], check=True)
        subprocess.run(["make", "-C", str(DIR), "ScaleAP"], check=True, capture_output=True)
    return exe


def scaleap_cpp(exe, s, p, lam, maxits=1000, stop=100):
    """The exemplar of each point, from the C++ code (its input: the number of
    points as uint32, then the similarities as float64, by row)."""
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(np.uint32(len(s)).tobytes())
        f.write(np.ascontiguousarray(s, dtype=np.float64).tobytes())
        f.flush()
        t = time.perf_counter()
        out = subprocess.run([str(exe), "-i", f.name, "-p", repr(p), "-l", str(lam), "-t", str(maxits),
                              "-s", str(stop), "-R"], capture_output=True, text=True, check=True).stdout
        t = time.perf_counter() - t
    return np.array([int(line.split()[1]) for line in out.splitlines()[1:] if line.strip()]), t


def netsim(s, p, exemplar_of):
    i = np.arange(len(s))
    return np.where(exemplar_of == i, p, s[i, exemplar_of]).sum()


def data(n, centers, seed):
    x, _ = make_blobs(n_samples=n, centers=centers, cluster_std=1.0, random_state=seed)
    s = naffprop.neg_dist_mat(x, 2)
    return s, float(np.median(s[~np.eye(n, dtype=bool)]))


def main():
    exe = build()
    print("| n | damping | clusters naffprop / C++ | net similarity naffprop / C++ | ARI "
          "| naffprop (s) | scaleap=True (s) | C++ (s) |")
    print("|--:|--:|--:|--:|--:|--:|--:|--:|")
    for n, centers, seed in [(300, 5, 0), (600, 10, 1)]:
        s, p = data(n, centers, seed)
        for lam in (0.9, 0.5):
            t = time.perf_counter()
            res = naffprop.apcluster(s, p=p, lam=lam, nonoise=True)
            t_dense = time.perf_counter() - t
            t = time.perf_counter()
            naffprop.apcluster(s, p=p, lam=lam, nonoise=True, scaleap=True)
            t_scaleap = time.perf_counter() - t
            e, t_cpp = scaleap_cpp(exe, s, p, lam)
            ari = adjusted_rand_score(res.labels, np.unique(e, return_inverse=True)[1])
            print(f"| {n} | {lam} | {len(res.exemplars)} / {len(np.unique(e))} "
                  f"| {res.netsim:.1f} / {netsim(s, p, e):.1f} | {ari:.2f} "
                  f"| {t_dense:.2f} | {t_scaleap:.2f} | {t_cpp:.1f} |", flush=True)

    print("\nDoes the C++ code converge? n = 300, damping 0.5: points whose exemplar differs "
          "from the run one iteration shorter.\n")
    print("| iterations | clusters | points changed |")
    print("|--:|--:|--:|")
    s, p = data(300, 5, 0)
    prev = None
    for t in (398, 399, 400, 401, 402):
        e, _ = scaleap_cpp(exe, s, p, 0.5, maxits=t, stop=10**6)
        if prev is not None:
            print(f"| {t} | {len(np.unique(e))} | {(e != prev).sum()} |", flush=True)
        prev = e


if __name__ == "__main__":
    main()
